from typing import List, Optional, Tuple

from bitarray import bitarray

from greaseweazle import error
from greaseweazle.codec import codec
from greaseweazle.flux import Flux
from greaseweazle.track import PLL, MasterTrack
from greaseweazle.flux import HasFlux


header_sync_bits = bitarray('11111011', endian='big')


def _runs_of_byte(buf: bytes, value: int) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    i = 0
    n = len(buf)
    while i < n:
        if buf[i] != value:
            i += 1
            continue
        j = i + 1
        while j < n and buf[j] == value:
            j += 1
        runs.append((i, j))
        i = j
    return runs


def _find_payload_window(buf: bytes, min_zero_run: int) -> Optional[Tuple[int, int]]:
    zero_runs = [(s, e) for (s, e) in _runs_of_byte(buf, 0x00)
                 if (e - s) >= min_zero_run]
    if len(zero_runs) < 2:
        return None

    best: Optional[Tuple[int, int]] = None
    best_score = -1
    for (a_s, a_e), (b_s, b_e) in zip(zero_runs, zero_runs[1:]):
        if b_s <= a_e:
            continue
        s, e = a_e, b_s
        if e - s < 16:
            continue
        payload = buf[s:e]
        nz = sum(1 for x in payload if x != 0x00)
        ff = sum(1 for x in payload if x == 0xFF)
        score = nz + min(ff, 32)
        if score > best_score:
            best = (s, e)
            best_score = score

    return best


def _score_payload(payload: bytes) -> int:
    if not payload:
        return -10_000
    nz = sum(1 for x in payload if x != 0x00)
    ff = sum(1 for x in payload if x == 0xFF)
    printable = sum(1 for x in payload if 0x20 <= x <= 0x7E)
    return nz + min(ff, 32) + printable // 4


def _flux_intervals_to_data_bits(flux_list, sample_freq,
                                 saw_4us_in: bool = False) -> tuple[bitarray, bool]:
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


def _split_flux_by_index(flux_list, index_list):
    if not index_list:
        return [list(flux_list)]

    revs = []
    i = 0
    carry = None

    for rev_ticks in index_list:
        remaining = rev_ticks
        rev = []

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
                remaining = 0

        if rev:
            revs.append(rev)

    return revs


def _group_hardsector_indexes(index_list: List[float], nsec: int) -> Tuple[List[float], List[List[float]]]:
    rev_index: List[float] = []
    rev_slots: List[List[float]] = []
    if nsec <= 0:
        return rev_index, rev_slots
    full = len(index_list) // nsec
    for r in range(full):
        slots = index_list[r*nsec:(r+1)*nsec]
        rev_slots.append(slots)
        rev_index.append(sum(slots))
    return rev_index, rev_slots


def _bytes_from_bits_no_pad(bits: bitarray, start: int, nbytes: int) -> bytes:
    raw_bits = bits[start:min(len(bits), start + nbytes * 8)]
    valid_bits = (len(raw_bits) // 8) * 8
    return raw_bits[:valid_bits].tobytes()


def _find_syncs(bits: bitarray, start: int, end: int, pat: bitarray) -> List[int]:
    return [start + x for x in bits[start:end].search(pat)]


def _find_zero_run_bytes(buf: bytes, start: int = 0,
                         min_zero_bytes: int = 8) -> Optional[int]:
    run = bytes(min_zero_bytes)
    idx = buf.find(run, start)
    return None if idx < 0 else idx


def _find_first_zero_byte(buf: bytes, start: int = 0) -> Optional[int]:
    idx = buf.find(b'\x00', start)
    return None if idx < 0 else idx


def _max_run(bits: bitarray, value: int) -> int:
    best = 0
    cur = 0
    target = bool(value)
    for b in bits:
        if b == target:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def _first_one_run(bits: bitarray, min_one_bits: int = 8
                   ) -> Optional[Tuple[int, int, int]]:
    i = 0
    e = len(bits)
    prev_zero_len = 0
    while i < e:
        while i < e and not bits[i]:
            prev_zero_len += 1
            i += 1
        if i >= e:
            break
        o0 = i
        while i < e and bits[i]:
            i += 1
        olen = i - o0
        if olen >= min_one_bits:
            return o0, olen, prev_zero_len
        prev_zero_len = 0
    return None


def _bits_excerpt(bits: bitarray, start: int, before: int = 32,
                  after: int = 96) -> str:
    s = max(0, start - before)
    e = min(len(bits), start + after)
    return bits[s:e].to01()


def _slot_starts_from_ticks(slots: List[float]) -> List[float]:
    starts = [0.0]
    acc = 0.0
    for t in slots[:-1]:
        acc += t
        starts.append(acc)
    return starts


def _xor8(buf: bytes) -> int:
    x = 0
    for b in buf:
        x ^= b
    return x


def _sum8(buf: bytes) -> int:
    return sum(buf) & 0xff


def _neg_sum8(buf: bytes) -> int:
    return (-sum(buf)) & 0xff


def _sum8_end_around(buf: bytes) -> int:
    s = sum(buf)
    while s > 0xff:
        s = (s & 0xff) + (s >> 8)
    return s & 0xff


def _neg_sum8_end_around(buf: bytes) -> int:
    return (-_sum8_end_around(buf)) & 0xff


def _ones_comp8(x: int) -> int:
    return (~x) & 0xff


def _rol8(x: int, n: int = 1) -> int:
    n &= 7
    return ((x << n) | (x >> (8 - n))) & 0xff


def _ror8(x: int, n: int = 1) -> int:
    n &= 7
    return ((x >> n) | (x << (8 - n))) & 0xff


def _rot_xor8(buf: bytes, rotate: str = "rol") -> int:
    x = 0
    for b in buf:
        x = _rol8(x, 1) if rotate == "rol" else _ror8(x, 1)
        x ^= b
    return x


def _rot_sum8(buf: bytes, rotate: str = "rol") -> int:
    x = 0
    for b in buf:
        x = _rol8(x, 1) if rotate == "rol" else _ror8(x, 1)
        x = (x + b) & 0xff
    return x


def _sum16_words(buf: bytes, little_endian: bool = True) -> int:
    total = 0
    i = 0
    n = len(buf)
    while i < n:
        lo = buf[i]
        hi = buf[i+1] if i+1 < n else 0
        word = (lo | (hi << 8)) if little_endian else ((lo << 8) | hi)
        total = (total + word) & 0xffff
        i += 2
    return total


def _fold16_to8(x: int) -> int:
    x = (x & 0xff) + (x >> 8)
    x = (x & 0xff) + (x >> 8)
    return x & 0xff


def _sum16_fold8(buf: bytes, little_endian: bool = True) -> int:
    return _fold16_to8(_sum16_words(buf, little_endian))


def _neg_sum16_fold8(buf: bytes, little_endian: bool = True) -> int:
    return (-_sum16_fold8(buf, little_endian)) & 0xff


def _xor_columns(buf: bytes) -> int:
    x = 0
    for bit in range(8):
        parity = 0
        mask = 1 << bit
        for b in buf:
            parity ^= 1 if (b & mask) else 0
        if parity:
            x |= mask
    return x


def _crc8_msb(buf: bytes, poly: int, init: int = 0, xorout: int = 0) -> int:
    crc = init & 0xff
    for b in buf:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ poly) & 0xff
            else:
                crc = (crc << 1) & 0xff
    return crc ^ xorout


def _bitrev8(x: int) -> int:
    y = 0
    for _ in range(8):
        y = (y << 1) | (x & 1)
        x >>= 1
    return y


def _crc8_lsb(buf: bytes, poly: int, init: int = 0, xorout: int = 0) -> int:
    crc = init & 0xff
    for b in buf:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = ((crc >> 1) ^ poly) & 0xff
            else:
                crc = (crc >> 1) & 0xff
    return crc ^ xorout


def _crc8_candidates(header: bytes, payload: bytes) -> List[Tuple[str, int]]:
    hp = header + payload
    return [
        ("xor8(data)", _xor8(payload)),
        ("xor8(hdr+data)", _xor8(hp)),
        ("xor_columns(data)", _xor_columns(payload)),
        ("xor_columns(hdr+data)", _xor_columns(hp)),
        ("sum8(data)", _sum8(payload)),
        ("sum8(hdr+data)", _sum8(hp)),
        ("neg_sum8(data)", _neg_sum8(payload)),
        ("neg_sum8(hdr+data)", _neg_sum8(hp)),
        ("onescomp_sum8(data)", _ones_comp8(_sum8(payload))),
        ("onescomp_sum8(hdr+data)", _ones_comp8(_sum8(hp))),
        ("sum8_eac(data)", _sum8_end_around(payload)),
        ("sum8_eac(hdr+data)", _sum8_end_around(hp)),
        ("neg_sum8_eac(data)", _neg_sum8_end_around(payload)),
        ("neg_sum8_eac(hdr+data)", _neg_sum8_end_around(hp)),
        ("rolxor8(data)", _rot_xor8(payload, "rol")),
        ("rolxor8(hdr+data)", _rot_xor8(hp, "rol")),
        ("rorxor8(data)", _rot_xor8(payload, "ror")),
        ("rorxor8(hdr+data)", _rot_xor8(hp, "ror")),
        ("rolsum8(data)", _rot_sum8(payload, "rol")),
        ("rolsum8(hdr+data)", _rot_sum8(hp, "rol")),
        ("rorsum8(data)", _rot_sum8(payload, "ror")),
        ("rorsum8(hdr+data)", _rot_sum8(hp, "ror")),
        ("sum16le_fold8(data)", _sum16_fold8(payload, True)),
        ("sum16le_fold8(hdr+data)", _sum16_fold8(hp, True)),
        ("neg_sum16le_fold8(data)", _neg_sum16_fold8(payload, True)),
        ("neg_sum16le_fold8(hdr+data)", _neg_sum16_fold8(hp, True)),
        ("sum16be_fold8(data)", _sum16_fold8(payload, False)),
        ("sum16be_fold8(hdr+data)", _sum16_fold8(hp, False)),
        ("neg_sum16be_fold8(data)", _neg_sum16_fold8(payload, False)),
        ("neg_sum16be_fold8(hdr+data)", _neg_sum16_fold8(hp, False)),
        ("crc8/poly07/data", _crc8_msb(payload, 0x07, 0x00, 0x00)),
        ("crc8/poly07/hdr+data", _crc8_msb(hp, 0x07, 0x00, 0x00)),
        ("crc8/poly07/initff/data", _crc8_msb(payload, 0x07, 0xff, 0x00)),
        ("crc8/poly07/initff/hdr+data", _crc8_msb(hp, 0x07, 0xff, 0x00)),
        ("crc8/itu/data", _crc8_msb(payload, 0x07, 0x00, 0x55)),
        ("crc8/itu/hdr+data", _crc8_msb(hp, 0x07, 0x00, 0x55)),
        ("crc8/j1850/data", _crc8_msb(payload, 0x1d, 0xff, 0xff)),
        ("crc8/j1850/hdr+data", _crc8_msb(hp, 0x1d, 0xff, 0xff)),
        ("crc8/j1850z/data", _crc8_msb(payload, 0x1d, 0xff, 0x00)),
        ("crc8/j1850z/hdr+data", _crc8_msb(hp, 0x1d, 0xff, 0x00)),
        ("crc8/cdma2000/data", _crc8_msb(payload, 0x9b, 0xff, 0x00)),
        ("crc8/cdma2000/hdr+data", _crc8_msb(hp, 0x9b, 0xff, 0x00)),
        ("crc8/maxim/data", _crc8_lsb(payload, 0x8c, 0x00, 0x00)),
        ("crc8/maxim/hdr+data", _crc8_lsb(hp, 0x8c, 0x00, 0x00)),
        ("bitrev(xor8(data))", _bitrev8(_xor8(payload))),
        ("bitrev(xor8(hdr+data))", _bitrev8(_xor8(hp))),
    ]


def _crc8_total_checks(header: bytes, payload: bytes, crc_byte: int
                       ) -> List[Tuple[str, bool]]:
    data_crc = payload + bytes([crc_byte])
    hdr_data_crc = header + payload + bytes([crc_byte])
    return [
        ("xor8(data+crc)==00", _xor8(data_crc) == 0x00),
        ("xor8(hdr+data+crc)==00", _xor8(hdr_data_crc) == 0x00),
        ("sum8(data+crc)==00", _sum8(data_crc) == 0x00),
        ("sum8(hdr+data+crc)==00", _sum8(hdr_data_crc) == 0x00),
        ("sum8(data+crc)==ff", _sum8(data_crc) == 0xff),
        ("sum8(hdr+data+crc)==ff", _sum8(hdr_data_crc) == 0xff),
        ("sum8_eac(data+crc)==00", _sum8_end_around(data_crc) == 0x00),
        ("sum8_eac(hdr+data+crc)==00", _sum8_end_around(hdr_data_crc) == 0x00),
        ("sum8_eac(data+crc)==ff", _sum8_end_around(data_crc) == 0xff),
        ("sum8_eac(hdr+data+crc)==ff", _sum8_end_around(hdr_data_crc) == 0xff),
        ("sum16le_fold8(data+crc)==00", _sum16_fold8(data_crc, True) == 0x00),
        ("sum16le_fold8(hdr+data+crc)==00", _sum16_fold8(hdr_data_crc, True) == 0x00),
        ("sum16be_fold8(data+crc)==00", _sum16_fold8(data_crc, False) == 0x00),
        ("sum16be_fold8(hdr+data+crc)==00", _sum16_fold8(hdr_data_crc, False) == 0x00),
    ]


class Logabax(codec.Codec):

    time_per_rev = 0.2

    def __init__(self, cyl: int, head: int, config):
        self.clock = 4e-6  # FM expected
        self.cyl, self.head = cyl, head
        self.config = config
        self.imd_mode = 2
        self.sector: List[Optional[bytes]] = [None] * self.nsec
        self.sector_ids: List[int] = list(range(1, self.nsec + 1))
        self.physical_sector_order: List[int] = list(range(self.nsec))
        self.sector_seen: List[List[Tuple[int, bytes]]] = [
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

    def summary_string(self) -> str:
        ok = self.nsec - self.nr_missing()
        return f"Logabax LX500 ({ok}/{self.nsec} sectors)"

    def has_sec(self, sec_id: int) -> bool:
        return self.sector[sec_id] is not None

    def nr_missing(self) -> int:
        return len([x for x in self.sector if x is None])

    def add_or_compare(self, sec_id: int, payload: bytes, rev: int) -> None:
        seen = self.sector_seen[sec_id]
        if seen:
            first_rev, first_payload = seen[0]
            if payload != first_payload:
                diff_at = None
                for i, (a, b) in enumerate(zip(first_payload, payload)):
                    if a != b:
                        diff_at = i
                        break
                if diff_at is None and len(payload) != len(first_payload):
                    diff_at = min(len(payload), len(first_payload))
                msg = (f"[logabax] T{self.cyl}.{self.head} S{self.sector_ids[sec_id]} "
                       f"data mismatch between rev {first_rev} and rev {rev}")
                if diff_at is not None:
                    a = first_payload[diff_at] if diff_at < len(first_payload) else None
                    b = payload[diff_at] if diff_at < len(payload) else None
                    msg += f" at byte {diff_at}"
                    if a is not None and b is not None:
                        msg += f" ({a:02x}!={b:02x})"
                print(msg)
                if self.dbg():
                    print(f"[logabax]   rev{first_rev}_head="
                          f"{first_payload[:16].hex(' ')}")
                    print(f"[logabax]   rev{rev}_head="
                          f"{payload[:16].hex(' ')}")
            elif self.dbg():
                print(f"[logabax] T{self.cyl}.{self.head} "
                      f"S{self.sector_ids[sec_id]} rev {rev} matches rev {first_rev}")
        seen.append((rev, payload))
        if self.sector[sec_id] is None:
            self.sector[sec_id] = payload

    def get_img_track(self) -> bytearray:
        tdat = bytearray()
        for sec in self.sector:
            tdat += sec if sec is not None else bytes(self.img_bps)
        return tdat

    def set_img_track(self, tdat: bytes) -> int:
        totsize = self.nsec * self.img_bps
        if len(tdat) < totsize:
            tdat += bytes(totsize - len(tdat))
        for sec in range(self.nsec):
            self.sector[sec] = tdat[sec*self.img_bps:(sec+1)*self.img_bps]
        return totsize

    def _normalise_hard_sectors(
            self, flux: Flux) -> Tuple[List[float], List[List[float]]]:
        if flux.sector_list is not None:
            return list(flux.index_list), [list(x) for x in flux.sector_list]

        if len(flux.index_list) > self.nsec:
            try:
                flux.identify_hard_sectors()
            except error.Fatal:
                pass
            else:
                assert flux.sector_list is not None
                return list(flux.index_list), [list(x) for x in flux.sector_list]

        idx = list(flux.index_list)
        if not idx:
            return idx, []

        avg_idx = sum(idx) / len(idx)
        rev_ticks = self.time_per_rev * flux.sample_freq
        if len(idx) >= self.nsec * 2 and avg_idx < (rev_ticks * 0.5):
            return _group_hardsector_indexes(idx, self.nsec)

        return idx, []

    def decode_flux(self, track: HasFlux, pll: Optional[PLL] = None) -> None:
        del pll

        flux = track.flux()
        flux.cue_at_index()
        rev_index, rev_slots = self._normalise_hard_sectors(flux)

        rev_fluxes = _split_flux_by_index(flux.list, rev_index)
        rev_bits: List[bitarray] = []
        saw_4us = False
        for rev_flux in rev_fluxes:
            bits, saw_4us = _flux_intervals_to_data_bits(
                rev_flux, flux.sample_freq, saw_4us
            )
            rev_bits.append(bits)

        printed_order_summary = False
        track_reference_phys_sector_ids: Optional[List[int]] = None
        for rev, bits in enumerate(rev_bits):
            found_order: List[int] = []
            found_order_seen = set()
            found_header_ids: List[int] = []
            if rev_slots and rev < len(rev_slots):
                slots = rev_slots[rev]
                if len(slots) == self.nsec and sum(slots) > 0:
                    slot_tick_starts = _slot_starts_from_ticks(slots)
                    total_ticks = sum(slots)
                    slot_starts = [
                        round(t * len(bits) / total_ticks)
                        for t in slot_tick_starts
                    ]
                else:
                    total_ticks = rev_index[rev] if rev < len(rev_index) else 0
                    slots = ([total_ticks / self.nsec] * self.nsec
                             if total_ticks > 0 else [0.0] * self.nsec)
                    slot_tick_starts = _slot_starts_from_ticks(slots)
                    slot_starts = [len(bits) * i // self.nsec
                                   for i in range(self.nsec)]
            else:
                total_ticks = rev_index[rev] if rev < len(rev_index) else 0
                slots = ([total_ticks / self.nsec] * self.nsec
                         if total_ticks > 0 else [0.0] * self.nsec)
                slot_tick_starts = _slot_starts_from_ticks(slots)
                slot_starts = [len(bits) * i // self.nsec for i in range(self.nsec)]

            search_pos = 0
            while search_pos < len(bits):
                header_syncs = _find_syncs(
                    bits, search_pos, len(bits), header_sync_bits
                )
                if not header_syncs:
                    if self.dbg():
                        slot_idx = min(
                            max(search_pos * self.nsec // max(len(bits), 1), 0),
                            self.nsec - 1
                        )
                        slot_tick_start = slot_tick_starts[slot_idx]
                        slot_tick_end = slot_tick_start + slots[slot_idx]
                        slot_us_start = slot_tick_start * 1e6 / flux.sample_freq
                        slot_us_end = slot_tick_end * 1e6 / flux.sample_freq
                        print(f"[logabax] T{self.cyl}.{self.head} R{rev}: "
                              f"no header sync from bit {search_pos} "
                              f"(slot S{slot_idx} "
                              f"ticks[{slot_tick_start:.0f}:{slot_tick_end:.0f}] "
                              f"us[{slot_us_start:.1f}:{slot_us_end:.1f}])")
                        print(f"[logabax]   first200={bits[:200].to01()}")
                    break

                header_sync_start = header_syncs[0]
                header_bit = header_sync_start
                data_start_bit = header_bit + 3 * 8
                slot_idx = min(
                    max(search_pos * self.nsec // max(len(bits), 1), 0),
                    self.nsec - 1
                )
                for i, start in enumerate(slot_starts):
                    if start <= header_sync_start:
                        slot_idx = i
                    else:
                        break
                sec_id = slot_idx
                slot_end_bit = (slot_starts[slot_idx + 1]
                                if slot_idx + 1 < len(slot_starts)
                                else len(bits))

                header = _bytes_from_bits_no_pad(bits, header_bit, 3)
                if len(header) < 3:
                    search_pos = header_sync_start + 1
                    continue
                if header[0] != 0xfb:
                    search_pos = header_sync_start + 1
                    continue
                track_id = header[1]
                sector_id = header[2]
                logical_sec_id = sector_id - 1
                if track_id != self.cyl:
                    if self.dbg():
                        print(f"[logabax] T{self.cyl}.{self.head} R{rev}: "
                              f"unexpected track id {track_id} at "
                              f"sync_bit={header_sync_start}")
                    search_pos = header_sync_start + 1
                    continue
                if not (0 <= logical_sec_id < self.nsec):
                    if self.dbg():
                        print(f"[logabax] T{self.cyl}.{self.head} R{rev}: "
                              f"unexpected sector id {sector_id} at "
                              f"sync_bit={header_sync_start}")
                    search_pos = header_sync_start + 1
                    continue
                sec_id = logical_sec_id
                found_header_ids.append(sector_id)
                if sec_id not in found_order_seen:
                    found_order_seen.add(sec_id)
                    found_order.append(sec_id)

                slot_payload_bits = bits[data_start_bit:slot_end_bit]
                slot_payload = slot_payload_bits[
                    :(len(slot_payload_bits) // 8) * 8
                ].tobytes()

                if len(slot_payload) < self.img_bps:
                    payload = slot_payload
                    trailer = b""
                    post_crc_byte: Optional[int] = None
                    data_end_bit = data_start_bit + len(slot_payload) * 8
                else:
                    payload = slot_payload[:self.img_bps]
                    trailer_search = slot_payload[self.img_bps:]
                    first_zero_idx = _find_first_zero_byte(trailer_search, 0)
                    zero_run_idx = _find_zero_run_bytes(trailer_search, 0, 8)
                    if first_zero_idx == 0:
                        trailer = trailer_search[:1]
                        post_crc_byte = trailer_search[1] if len(trailer_search) > 1 else None
                        data_end_bit = data_start_bit + (self.img_bps + 1) * 8
                    elif first_zero_idx is not None and first_zero_idx <= 8:
                        trailer = trailer_search[:first_zero_idx]
                        post_crc_byte = (
                            trailer_search[first_zero_idx]
                            if len(trailer_search) > first_zero_idx else None
                        )
                        data_end_bit = (
                            data_start_bit
                            + (self.img_bps + first_zero_idx) * 8
                        )
                    elif zero_run_idx is not None:
                        trailer = trailer_search[:zero_run_idx]
                        post_crc_byte = (
                            trailer_search[zero_run_idx]
                            if len(trailer_search) > zero_run_idx else None
                        )
                        data_end_bit = (
                            data_start_bit
                            + (self.img_bps + zero_run_idx) * 8
                        )
                    else:
                        trailer = trailer_search
                        post_crc_byte = None
                        data_end_bit = data_start_bit + len(slot_payload) * 8

                if len(payload) >= self.img_bps:
                    final_payload = payload[:self.img_bps]
                else:
                    final_payload = payload + bytes(self.img_bps - len(payload))
                self.add_or_compare(sec_id, final_payload, rev)

                head_hex = final_payload[:16].hex(' ')
                tail_hex = final_payload[-16:].hex(' ') if final_payload else ""
                trailer_hex = trailer[:16].hex(' ')
                trailer_tail_hex = trailer[-16:].hex(' ') if trailer else ""
                if self.dbg():
                    print(f"[logabax] T{self.cyl}.{self.head} R{rev} S{sec_id} "
                          f"sync_bit={header_sync_start} header={header.hex(' ')} "
                          f"data_bit={data_start_bit} payload_end_bit={data_start_bit + len(final_payload) * 8} "
                          f"data_end_bit={data_end_bit} "
                          f"slot_end_bit={slot_end_bit} "
                          f"len={len(final_payload)} trailer_len={len(trailer)}")
                    print(f"[logabax]   data_start: {head_hex}")
                    print(f"[logabax]   data_end:   {tail_hex}")
                    print(f"[logabax]   trailer_start: {trailer_hex}")
                    print(f"[logabax]   trailer_end:   {trailer_tail_hex}")
                if len(final_payload) == self.img_bps and final_payload and final_payload.count(final_payload[0]) == len(final_payload):
                    crc_txt = f"{trailer[0]:02x}" if trailer else "--"
                    post_crc_txt = f"{post_crc_byte:02x}" if post_crc_byte is not None else "--"
                    if self.dbg():
                        print(f"[logabax] uniform T{self.cyl}.{self.head} "
                              f"R{rev} S{sector_id} header={header.hex(' ')} "
                              f"value={final_payload[0]:02x} "
                              f"crc={crc_txt} next={post_crc_txt} "
                              f"sync_bit={header_sync_start}")
                if trailer and self.dbg():
                    crc_byte = trailer[0]
                    print(f"[logabax]   crc_byte: {crc_byte:02x}")
                    for name, val in _crc8_candidates(header, final_payload):
                        mark = " MATCH" if val == crc_byte else ""
                        print(f"[logabax]   crc_test {name}={val:02x}{mark}")
                    for name, ok in _crc8_total_checks(header, final_payload, crc_byte):
                        mark = " MATCH" if ok else ""
                        print(f"[logabax]   crc_total_test {name}{mark}")
                next_slot_floor = (slot_starts[slot_idx + 1]
                                   if slot_idx + 1 < len(slot_starts)
                                   else len(bits))
                search_pos = max(header_sync_start + 1, next_slot_floor)

            if found_order:
                seen = set()
                phys_order = []
                for sec_id in found_order:
                    if sec_id not in seen:
                        seen.add(sec_id)
                        phys_order.append(sec_id)
                for sec_id in range(self.nsec):
                    if sec_id not in seen:
                        phys_order.append(sec_id)
                self.physical_sector_order = phys_order
                phys_sector_ids = [self.sector_ids[x] for x in phys_order]
                if getattr(self.config, 'reference_phys_sector_ids', None) is None:
                    self.config.reference_phys_sector_ids = list(phys_sector_ids)
                if track_reference_phys_sector_ids is None:
                    track_reference_phys_sector_ids = list(phys_sector_ids)
                if not printed_order_summary:
                    shift = 0
                    reference_phys_sector_ids = getattr(
                        self.config, 'reference_phys_sector_ids', None
                    )
                    if reference_phys_sector_ids:
                        n = len(reference_phys_sector_ids)
                        for s in range(n):
                            if phys_sector_ids == (
                                    reference_phys_sector_ids[s:]
                                    + reference_phys_sector_ids[:s]):
                                shift = s
                                break
                    if self.dbg():
                        print(f"[logabax] T{self.cyl}.{self.head}: "
                              f"phys_ids={phys_sector_ids} shift={shift}")
                    printed_order_summary = True
                else:
                    if track_reference_phys_sector_ids is None:
                        continue
                    ref_ids = list(track_reference_phys_sector_ids)
                    if phys_sector_ids != ref_ids:
                        if self.dbg():
                            print(f"[logabax] T{self.cyl}.{self.head} R{rev}: "
                                  f"physical order differs: ref={ref_ids} cur={phys_sector_ids}")

        seen_parts = []
        for sec_id, seen in enumerate(self.sector_seen):
            revs = ",".join(str(r) for r, _ in seen)
            seen_parts.append(f"S{self.sector_ids[sec_id]}=[{revs}]")
        if self.dbg():
            print(f"[logabax] T{self.cyl}.{self.head}: rev_coverage " +
                  " ".join(seen_parts))

    def master_track(self) -> MasterTrack:
        raise error.Fatal('logabax: write/encode not implemented')

    def verify_track(self, flux) -> bool:
        readback = self.__class__(self.cyl, self.head, self.config)
        readback.decode_flux(flux)
        return self.sector == readback.sector


class LogabaxDef(codec.TrackDef):

    default_revs = 1

    def __init__(self, format_name: str):
        self.secs: Optional[int] = None
        self.img_bps: Optional[int] = None
        self.min_zero_run: int = 16
        self.debug: bool = False
        self.finalised = False

    def add_param(self, key: str, val) -> None:
        if key == 'secs':
            self.secs = int(val)
        elif key == 'img_bps':
            self.img_bps = int(val)
        elif key == 'min_zero_run':
            self.min_zero_run = int(val)
        elif key == 'debug':
            self.debug = (str(val).lower() == 'true')
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

    def mk_track(self, cyl: int, head: int) -> Logabax:
        return Logabax(cyl, head, self)
