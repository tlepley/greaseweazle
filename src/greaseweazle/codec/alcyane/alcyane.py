# greaseweazle/codec/alcyane/alcyane.py

from typing import List, Optional
import bisect
from bitarray import bitarray

from greaseweazle import error
from greaseweazle.codec import codec
from greaseweazle.codec.ibm.ibm import encode, fm_encode
from greaseweazle.track import MasterTrack, PLL
from greaseweazle.flux import HasFlux

default_revs = 1

header_sync_bits = bitarray('0' * (16*8) + '1' * 8, endian='big')
data_sync_bits = bitarray('0' * (4*8) + '1' * 8, endian='big')
data_sync_window_bits = 256
next_rev_extension_time = 100 * 4e-6


def alcyane_csum(dat: bytes) -> int:
    x = 0
    for b in dat:
        x ^= b
    return x


def dump_bits_window(bits, poi_offset, before_bits=0, after_bits=512,
                     label="bits") -> None:
    window_start = max(0, poi_offset - max(0, before_bits))
    window_end = min(len(bits), poi_offset + max(0, after_bits))

    print("[Debug]", label,
          "offset", poi_offset,
          "before_bits", before_bits,
          "after_bits", after_bits,
          "window", f"[{window_start}:{window_end}]")

    # Print bits before the point of interest.
    pre = bits[window_start:poi_offset].to01()
    pre_abs = window_start
    for i in range(0, len(pre), 64):
        chunk = pre[i:i+64]
        print("[Debug]", label, "PRE ", "%06d" % pre_abs, chunk)
        pre_abs += len(chunk)

    # Print the reference line that starts exactly at the point of interest.
    ref = bits[poi_offset:min(window_end, poi_offset + 64)].to01()
    print("[Debug]", label, "REF ", "%06d" % poi_offset, ref)

    # Print bits after the reference line.
    post_start = poi_offset + len(ref)
    post = bits[post_start:window_end].to01()
    post_abs = post_start
    for i in range(0, len(post), 64):
        chunk = post[i:i+64]
        print("[Debug]", label, "POST", "%06d" % post_abs, chunk)
        post_abs += len(chunk)


def flux_intervals_to_data_bits(flux_list, sample_freq,
                                saw_4us_in=False) -> tuple[bitarray, bool]:
    bits = bitarray(endian='big')
    saw_4us = saw_4us_in

    for ticks in flux_list:
        if ticks <= 0:
            saw_4us = False
            continue

        t = ticks / sample_freq

        if 2.5e-6 <= t <= 5.5e-6:
            if saw_4us:
                bits.append(1)
                saw_4us = False
            else:
                saw_4us = True
        elif 6.0e-6 <= t <= 10.5e-6:
            saw_4us = False
            bits.append(0)
        else:
            saw_4us = False

    return bits, saw_4us


def flux_intervals_to_data_bits_limited(flux_list, sample_freq, max_bits,
                                        max_time_s=None,
                                        saw_4us_in=False) -> tuple[bitarray, bool]:
    bits = bitarray(endian='big')
    saw_4us = saw_4us_in
    elapsed_s = 0.0

    for ticks in flux_list:
        if len(bits) >= max_bits:
            break
        if max_time_s is not None and elapsed_s >= max_time_s:
            break
        if ticks <= 0:
            saw_4us = False
            continue

        t = ticks / sample_freq
        elapsed_s += t

        if 2.5e-6 <= t <= 5.5e-6:
            if saw_4us:
                bits.append(1)
                saw_4us = False
            else:
                saw_4us = True
        elif 6.0e-6 <= t <= 10.5e-6:
            saw_4us = False
            bits.append(0)
        else:
            saw_4us = False

    return bits, saw_4us


def concat_flux_with_time_window(cur_flux, next_flux, sample_freq,
                                 next_time_s, stitch_split=False):
    out = list(cur_flux)
    taken = []
    remain_ticks = next_time_s * sample_freq
    if remain_ticks <= 0:
        return out, taken

    # Re-stitch only if index split actually cut one interval into
    # tail(cur_rev) + head(next_rev).
    i = 0
    if stitch_split and out and next_flux and out[-1] > 0 and next_flux[0] > 0:
        out[-1] += next_flux[0]
        remain_ticks -= next_flux[0]
        i = 1

    for ticks in next_flux[i:]:
        if ticks <= 0:
            out.append(ticks)
            taken.append(ticks)
            continue
        if remain_ticks <= 0:
            break
        if ticks <= remain_ticks:
            out.append(ticks)
            taken.append(ticks)
            remain_ticks -= ticks
        else:
            # Do not emit a truncated interval: that would create an
            # artificial transition at the extension boundary.
            break
    return out, taken


def split_flux_by_index(flux_list, index_list):
    if not index_list:
        return [list(flux_list)], []

    revs = []
    rev_split_end = []
    i = 0
    carry = None

    for rev_ticks in index_list:
        remaining = rev_ticks
        rev = []
        split_end = False

        while remaining > 0:
            if carry is not None:
                ticks = carry
                carry = None
            else:
                if i >= len(flux_list):
                    break
                ticks = flux_list[i]
                i += 1

            if ticks <= remaining:
                rev.append(ticks)
                remaining -= ticks
            else:
                rev.append(remaining)
                carry = ticks - remaining
                split_end = True
                remaining = 0

        if rev:
            revs.append(rev)
            rev_split_end.append(split_end)

    return revs, rev_split_end


def find_syncs(bits, start, end, pat):
    return [start + x for x in bits[start:end].search(pat)]


def bytes_from_bits_no_pad(bits, start, nbytes):
    raw_bits = bits[start:min(len(bits), start + nbytes * 8)]
    valid_bits = (len(raw_bits) // 8) * 8
    return raw_bits[:valid_bits].tobytes()


class Alcyane(codec.Codec):

    time_per_rev = 0.2
    verify_revs: float = default_revs

    def __init__(self, cyl: int, head: int, config):
        self.clock = 4e-6
        self.cyl, self.head = cyl, head
        self.config = config
        # Alcyane only supports one FM floppy type for this codec instance.
        # Force IMD export mode to FM (250 kbps / 300 RPM).
        self.imd_mode = 2
        self.sector: List[Optional[bytes]] = [None] * self.nsec
        self.sector_present: List[bool] = [False] * self.nsec
        self.sector_good: List[bool] = [False] * self.nsec
        self.sector_data_error: List[bool] = [False] * self.nsec
        self.sector_cyls: List[int] = [self.cyl] * self.nsec
        self.sector_heads: List[int] = [self.head] * self.nsec
        self.sector_ids: List[int] = list(range(self.nsec))
        # Logical sector ids in physical (rotational) order as observed
        # during decode_flux(). Falls back to logical order if unavailable.
        self.physical_sector_order: List[int] = list(range(self.nsec))
        self.sector_seen: List[List[tuple[int, bytes]]] = [
            [] for _ in range(self.nsec)
        ]

    @property
    def nsec(self) -> int:
        return self.config.secs

    @property
    def img_bps(self) -> int:
        return self.config.img_bps

    def dbg(self) -> bool:
        return self.config.debug

    def dbg_bits(self) -> bool:
        return self.config.debug_bits

    def summary_string(self) -> str:
        nsec, nbad = self.nsec, self.nr_missing()
        return "Alcyane (%d/%d sectors)" % (nsec - nbad, nsec)

    def bad_sector(self) -> bytes:
        return bytes(self.img_bps)

    def has_sec(self, sec_id: int) -> bool:
        return self.sector_good[sec_id]

    def nr_missing(self) -> int:
        return len([ok for ok in self.sector_good if not ok])

    def add(self, sec_id: int, data: bytes) -> None:
        self.sector[sec_id] = data
        self.sector_present[sec_id] = True

    def note_sector_id(self, sec_id: int, cyl: int) -> None:
        self.sector_present[sec_id] = True
        self.sector_cyls[sec_id] = cyl
        self.sector_heads[sec_id] = self.head
        self.sector_ids[sec_id] = sec_id

    def add_or_compare(self, sec_id: int, payload: bytes, rev: int) -> None:
        seen = self.sector_seen[sec_id]
        had_good = bool(seen)
        was_present = self.sector_present[sec_id]
        had_data_error = self.sector_data_error[sec_id]

        if seen:
            first_rev, first_payload = seen[0]
            if payload != first_payload:
                if self.dbg():
                    for i, (a, b) in enumerate(zip(first_payload, payload)):
                        if a != b:
                            print("DBG first diff S%d offset=%d "
                                  "rev%d=%02x rev%d=%02x" %
                                  (sec_id, i, first_rev, a, rev, b))
                            break
                raise error.Fatal(
                    "sector data differs T%d.%d S%d rev %d differs from rev %d"
                    % (self.cyl, self.head, sec_id, rev, first_rev)
                )
            elif self.dbg():
                print("[Debug] sector data match "
                      "CHS=%d:%d:%d rev %d matches rev %d" %
                      (self.cyl, self.head, sec_id, rev, first_rev))

        seen.append((rev, payload))

        # First good copy wins: overwrite any previously stored bad/partial data.
        if not had_good:
            self.sector[sec_id] = payload
            self.sector_present[sec_id] = True
            if rev > 0 and (was_present or had_data_error):
                print("[INFO][tr %d.%d][rev %d] recovered sector %d with good checksum"
                      % (self.cyl, self.head, rev, sec_id))
        self.sector_good[sec_id] = True
        self.sector_data_error[sec_id] = False

    def get_img_track(self) -> bytearray:
        tdat = bytearray()
        for sec in self.sector:
            tdat += sec if sec is not None else self.bad_sector()
        return tdat

    def set_img_track(self, tdat: bytes) -> int:
        totsize = self.nsec * self.img_bps
        if len(tdat) < totsize:
            tdat += bytes(totsize - len(tdat))

        for sec in range(self.nsec):
            self.sector[sec] = tdat[
                sec*self.img_bps:(sec+1)*self.img_bps
            ]
            self.sector_present[sec] = True
            self.sector_good[sec] = True
            self.sector_data_error[sec] = False
            self.sector_cyls[sec] = self.cyl
            self.sector_heads[sec] = self.head
            self.sector_ids[sec] = sec

        return totsize

    def decode_flux(self, track: HasFlux, pll: Optional[PLL]=None) -> None:
        flux = track.flux()
        flux.cue_at_index()
        found_order_by_rev: List[List[int]] = []
        merged_order: List[int] = []

        rev_fluxes, rev_split_end = split_flux_by_index(flux.list, flux.index_list)
        rev_bits = []
        rev_start_saw_4us = []
        saw_4us = False
        for rev_flux in rev_fluxes:
            rev_start_saw_4us.append(saw_4us)
            bits, saw_4us = flux_intervals_to_data_bits(
                rev_flux, flux.sample_freq, saw_4us
            )
            rev_bits.append(bits)

        rev_slot_starts: List[Optional[List[int]]] = []
        if flux.sector_list is not None:
            for rev, bits in enumerate(rev_bits):
                slots = flux.sector_list[rev] if rev < len(flux.sector_list) else None
                if slots is None or len(slots) != self.nsec:
                    rev_slot_starts.append(None)
                    continue
                total_ticks = sum(slots)
                if total_ticks <= 0:
                    rev_slot_starts.append(None)
                    continue
                starts = [0]
                acc = 0.0
                for t in slots[:-1]:
                    acc += t
                    starts.append(round(acc * len(bits) / total_ticks))
                rev_slot_starts.append(starts)
        else:
            rev_slot_starts = [None] * len(rev_bits)

        for rev, bits in enumerate(rev_bits):
            found_order: List[int] = []
            found_order_seen = set()
            slot_starts = rev_slot_starts[rev] if rev < len(rev_slot_starts) else None

            if rev + 1 < len(rev_fluxes):
                flux_ext, ext_taken = concat_flux_with_time_window(
                    rev_fluxes[rev], rev_fluxes[rev + 1],
                    flux.sample_freq, next_rev_extension_time,
                    stitch_split=rev_split_end[rev]
                )
                if self.dbg():
                    def us(vals):
                        return [round((x / flux.sample_freq) * 1e6, 3)
                                for x in vals]
                    print("[Debug] flux_join rev", rev)
                    print("[Debug]   cur_tail_us", us(rev_fluxes[rev][-8:]))
                    print("[Debug]   next_head_us", us(rev_fluxes[rev + 1][:8]))
                    print("[Debug]   ext_taken_us", us(ext_taken[:16]))
                    print("[Debug]   ext_taken_count", len(ext_taken))
                    print("[Debug]   ext_taken_total_us",
                          round(sum(x for x in ext_taken if x > 0)
                                / flux.sample_freq * 1e6, 3))
                    join_ticks = rev_fluxes[rev][-8:] + ext_taken[:16]
                    print("[Debug]   join_tail_plus_ext_us", us(join_ticks))
                bits_ext, _ = flux_intervals_to_data_bits(
                    flux_ext, flux.sample_freq, rev_start_saw_4us[rev]
                )
            else:
                bits_ext = bits

            if self.dbg():
                print("DBG T%d.%d REV %d bits=%d" %
                      (self.cyl, self.head, rev, len(bits)))

            search_pos = 0
            sectors_decoded = 0

            while search_pos < len(bits) and sectors_decoded < self.nsec:
                header_syncs = find_syncs(
                    bits, search_pos, len(bits), header_sync_bits
                )
                if not header_syncs:
                    break

                header_sync_start = header_syncs[0]
                header_bit = header_sync_start + 16*8
                hdr = bytes_from_bits_no_pad(bits, header_bit, 3)

                if len(hdr) < 3 or hdr[0] != 0xff:
                    search_pos = header_sync_start + 1
                    continue

                cyl = hdr[1]
                sec_id = hdr[2]
                next_slot_floor = None
                if slot_starts:
                    slot_idx = max(0, bisect.bisect_right(slot_starts,
                                                          header_sync_start) - 1)
                    if slot_idx + 1 < len(slot_starts):
                        next_slot_floor = slot_starts[slot_idx + 1]
                    else:
                        next_slot_floor = len(bits)

                if self.dbg():
                    print("FOUND header phyCH=%d:%d, logiCS=%d:%d "
                          "(rev=%d, sync_offset=%d)" %
                          (self.cyl, self.head, cyl, sec_id,
                           rev, header_bit))

                if self.dbg():
                    print("[Debug] header data: ", hdr.hex())

                if self.dbg_bits() and rev == 0:
                    dump_bits_window(bits_ext, header_sync_start,
                                     before_bits=(16+10)*8,
                                     after_bits=(16+10)*8,
                                     label="around_header")

                if cyl != self.cyl:
                    print("[ERROR][tr %d.%d][rev %d] header cyl %d does "
                          "not match track cyl %d" %
                          (self.cyl, self.head, rev, cyl, self.cyl))
                    search_pos = header_sync_start + 1
                    if next_slot_floor is not None:
                        search_pos = max(search_pos, next_slot_floor)
                    continue

                if sec_id >= self.nsec:
                    print("[ERROR][tr %d.%d][rev %d] header sec_id %d "
                          "out of range (max %d)" %
                          (self.cyl, self.head, rev, sec_id, self.nsec-1))
                    search_pos = header_sync_start + 1
                    if next_slot_floor is not None:
                        search_pos = max(search_pos, next_slot_floor)
                    continue

                # Track observed physical order for this revolution from valid
                # headers, even if data later fails CRC.
                if sec_id not in found_order_seen:
                    found_order_seen.add(sec_id)
                    found_order.append(sec_id)
                self.note_sector_id(sec_id, cyl)

                expected_len = 1 + self.img_bps + 1
                data_search_start = header_bit + 3*8
                data_search_end = min(
                    len(bits_ext),
                    data_search_start + data_sync_window_bits
                )
                data_syncs = find_syncs(
                    bits_ext, data_search_start, data_search_end,
                    data_sync_bits
                )

                if not data_syncs:
                    print("[ERROR][tr %d.%d][rev %d] no data sync found "
                          "for header sec_id %d" %
                          (self.cyl, self.head, rev, sec_id))
                    if self.dbg() or self.dbg_bits():
                        dump_bits_window(bits_ext, data_search_start,
                                         before_bits=64,
                                         after_bits=512,
                                         label="data_sync_error_no_sync")
                    # Move past where data would have been if sync were present.
                    search_pos = data_search_start + 4*8 + expected_len * 8
                    if next_slot_floor is not None:
                        search_pos = max(search_pos, next_slot_floor)
                    continue

                data_sync_start = data_syncs[0]
                data_bit = data_sync_start + 4*8
                dat = bytes_from_bits_no_pad(bits_ext, data_bit, expected_len)

                if self.dbg():
                    print("[Debug] data_sync_distance",
                          "rev", rev,
                          "sec", sec_id,
                          "header_to_data_sync",
                          data_sync_start - header_sync_start)

                if self.dbg_bits() and rev == 0:
                    dump_bits_window(bits_ext, data_sync_start,
                                     before_bits=0,
                                     after_bits=256,
                                     label="data_sync")

                if len(dat) < expected_len:
                    is_last_rev = (rev == len(rev_bits) - 1)
                    if not is_last_rev or self.dbg() or self.dbg_bits():
                        print("[WARNING][tr %d.%d][rev %d] data too short for "
                              "logical CS %d:%d got %d bytes, expected %d" %
                              (self.cyl, self.head, rev,
                               cyl, sec_id, len(dat), expected_len))
                    self.sector_data_error[sec_id] = True
                    search_pos = data_bit + expected_len * 8
                    if next_slot_floor is not None:
                        search_pos = max(search_pos, next_slot_floor)
                    continue
                elif dat[0] != 0xff:
                    print("[ERROR][tr %d.%d][rev %d] data sync byte not "
                          "found at expected location for sec_id %d" %
                          (self.cyl, self.head, rev, sec_id))
                    if self.dbg() or self.dbg_bits():
                        dump_bits_window(bits_ext, data_sync_start,
                                         before_bits=64,
                                         after_bits=256,
                                         label="data_sync_error_bad_ff")
                    self.sector_data_error[sec_id] = True
                    search_pos = data_bit + expected_len * 8
                    if next_slot_floor is not None:
                        search_pos = max(search_pos, next_slot_floor)
                    continue
                else:
                    payload = dat[1:1+self.img_bps]
                    csum = dat[1+self.img_bps]
                    calc = alcyane_csum(payload)
                    if self.dbg():
                        print("DBG csum",
                              "rev", rev,
                              "sec", sec_id,
                              "csum", csum,
                              "calc", calc,
                              "payload0", payload[:8].hex(),
                              "after", dat[1+self.img_bps:
                                           1+self.img_bps+8].hex())

                    if calc != csum:
                        print("[ERROR][tr %d.%d][rev %d] checksum mismatch "
                              "for sec_id %d: got %02x, expected %02x" %
                              (self.cyl, self.head, rev, sec_id, csum, calc))
                        if self.dbg() or self.dbg_bits():
                            dump_bits_window(bits_ext, data_sync_start,
                                             before_bits=64,
                                             after_bits=256,
                                             label="data_sync_error_bad_csum")

                        dat2 = bytes_from_bits_no_pad(
                            bits_ext, data_bit, 1+self.img_bps+1+2
                        )
                        csum_idx = 1 + self.img_bps
                        if (self.dbg() or self.dbg_bits()) and len(dat2) >= csum_idx + 3 and csum_idx >= 2:
                            print("       data around crc: "
                                  f"{dat2[csum_idx-2]:02x} "
                                  f"{dat2[csum_idx-1]:02x} "
                                  f"[{dat2[csum_idx]:02x}] "
                                  f"{dat2[csum_idx+1]:02x} "
                                  f"{dat2[csum_idx+2]:02x} "
                                  )
                        if self.sector[sec_id] is None:
                            self.add(sec_id, payload)
                        self.sector_data_error[sec_id] = True
                        search_pos = data_bit + expected_len * 8
                        if next_slot_floor is not None:
                            search_pos = max(search_pos, next_slot_floor)
                        continue

                    if self.dbg():
                        print("FOUND data phyCH=%d:%d, logiCS=%d:%d "
                              "(rev=%d, sync_offset=%d)" %
                              (self.cyl, self.head, cyl, sec_id,
                               rev, data_sync_start))
                        print("      first data=", dat[:32].hex())
                    self.add_or_compare(sec_id, payload, rev)
                    sectors_decoded += 1
                    search_pos = data_bit + expected_len * 8
                    if next_slot_floor is not None:
                        search_pos = max(search_pos, next_slot_floor)
                    continue

                search_pos = data_bit + expected_len * 8
                if next_slot_floor is not None:
                    search_pos = max(search_pos, next_slot_floor)
            found_order_by_rev.append(found_order)
            # Incremental merge by revolution:
            # - rev0 builds the baseline track order
            # - each next rev is merged left-to-right with a cursor
            if not merged_order:
                merged_order = list(found_order)
            else:
                cursor = 0
                for sec_id in found_order:
                    if cursor < len(merged_order) and merged_order[cursor] == sec_id:
                        cursor += 1
                        continue
                    try:
                        idx = merged_order.index(sec_id, cursor)
                    except ValueError:
                        merged_order.insert(cursor, sec_id)
                        print("[INFO][tr %d.%d][rev %d] merged sector %d at pos %d"
                              % (self.cyl, self.head, rev, sec_id, cursor))
                        cursor += 1
                    else:
                        cursor = idx + 1

        # Keep all sectors represented for emitters that expect fixed count.
        seen = set()
        deduped = []
        for sec_id in merged_order:
            if sec_id not in seen:
                seen.add(sec_id)
                deduped.append(sec_id)
        merged_order = deduped
        for sec_id in range(self.nsec):
            if sec_id not in seen:
                merged_order.append(sec_id)
        self.physical_sector_order = merged_order

    def master_track(self) -> MasterTrack:
        t = bytes()
        slen = int(self.time_per_rev / self.clock / self.nsec / 16)

        for sec_id in range(self.nsec):
            payload = self.sector[sec_id]
            if payload is None:
                payload = self.bad_sector()

            dat = bytearray()
            dat += bytes(16)
            dat += b'\xff'
            dat += bytes([self.cyl])
            dat += bytes([sec_id])
            dat += bytes(8)
            dat += bytes(4)
            dat += b'\xff'
            dat += payload
            dat += bytes([alcyane_csum(payload)])
            dat += b'\x00'

            s = encode(bytes(dat))

            if len(s) // 2 < slen:
                s += encode(bytes(slen - len(s)//2))

            t += s

        t = fm_encode(t)

        track = MasterTrack(bits=t, time_per_rev=self.time_per_rev,
                            hardsector_bits=[slen*16] * self.nsec)
        track.verify = self
        return track

    def verify_track(self, flux):
        readback_track = self.__class__(self.cyl, self.head, self.config)
        readback_track.decode_flux(flux)
        return (readback_track.nr_missing() == 0
                and self.sector == readback_track.sector)


class AlcyaneDef(codec.TrackDef):
    default_revs = default_revs

    def __init__(self, format_name: str):
        self.secs: Optional[int] = None
        self.img_bps: Optional[int] = None
        self.debug: bool = False
        self.debug_bits: bool = False
        self.finalised = False

    def add_param(self, key: str, val) -> None:
        if key == 'secs':
            self.secs = int(val)
        elif key == 'img_bps':
            self.img_bps = int(val)
            error.check(self.img_bps == 162,
                        f'bad img_bps {self.img_bps}')
        elif key == 'debug':
            self.debug = True
        elif key == 'debug_bits':
            self.debug_bits = True
        else:
            raise error.Fatal('unrecognised track option %s' % key)

    def finalise(self) -> None:
        if self.finalised:
            return
        error.check(self.secs is not None,
                    'number of sectors not specified')
        error.check(self.img_bps is not None,
                    'img_bps not specified')
        self.finalised = True

    def mk_track(self, cyl: int, head: int) -> Alcyane:
        return Alcyane(cyl, head, self)

# Local variables:
# python-indent: 4
# End:
