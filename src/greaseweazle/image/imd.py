# greaseweazle/image/imd.py
#
# Written & released by Keir Fraser <keir.xen@gmail.com>
# 
# This is free and unencumbered software released into the public domain.
# See the file COPYING for more details, or visit <http://unlicense.org>.

from typing import Dict, Tuple, Optional, List

import datetime, struct

from greaseweazle import __version__
from greaseweazle import error
from greaseweazle.codec.ibm import ibm
from .image import Image

class IMDMode:
    FM_500kbps = 0
    FM_300kbps = 1
    FM_250kbps = 2
    MFM_500kbps = 3
    MFM_300kbps = 4
    MFM_250kbps = 5


def size_to_n(size: int) -> Optional[int]:
    for n in range(7):
        if size == (128 << n):
            return n
    return None


class IMDSector:
    def __init__(self, c: int, h: int, r: int, data: Optional[bytes],
                 deleted: bool = False, data_error: bool = False):
        self.c = c
        self.h = h
        self.r = r
        self.data = data
        self.deleted = deleted
        self.data_error = data_error


class IMDGenericTrack:
    def __init__(self, mode: int, sectors: List[IMDSector]):
        self.mode = mode
        self.sectors = sectors


class IMD(Image):

    def __init__(self, name: str, _fmt):
        self.to_track: Dict[Tuple[int,int],object] = dict()
        self.filename = name


    def from_bytes(self, dat: bytes) -> None:

        # Check and strip the header
        sig, = struct.unpack('4s', dat[:4])
        error.check(sig == b'IMD ', 'Unrecognised IMD file: bad signature')
        for i,x in enumerate(dat):
            if x == 0x1a:
                break
        error.check(x == 0x1a, 'IMD: No comment terminator found')

        # We will adjust this as we go
        rpm = 300

        i += 1
        while i < len(dat)-5:
            mode, cyl, head, nsec, sec_n = struct.unpack('5B', dat[i:i+5])
            i += 5

            has_cyl_map = (head & 0x80) != 0
            has_head_map = (head & 0x40) != 0
            head &= 0x3f
            error.check(0 <= head <= 1, 'IMD: Bad head value %x' % head)

            if sec_n == 0xff:
                error.check(i + 2*nsec <= len(dat),
                            'IMD: Truncated sector size table')
                sec_sizes = [struct.unpack('<H', dat[i+2*j:i+2*j+2])[0]
                             for j in range(nsec)]
                i += 2*nsec
            else:
                error.check(0 <= sec_n <= 6, 'IMD: Bad sector size %x' % sec_n)
                sec_sizes = [128 << sec_n] * nsec

            if mode == IMDMode.FM_250kbps or mode == IMDMode.FM_300kbps:
                fmt = ibm.IBMTrack_FixedDef('ibm.fm')
                fmt.rate = 125
            elif mode == IMDMode.FM_500kbps:
                fmt = ibm.IBMTrack_FixedDef('ibm.fm')
                if nsec == 26: rpm = 360 # 8-inch disk
                fmt.rate = 250
            elif mode == IMDMode.MFM_250kbps or mode == IMDMode.MFM_300kbps:
                fmt = ibm.IBMTrack_FixedDef('ibm.mfm')
                fmt.rate = 250
            elif mode == IMDMode.MFM_500kbps:
                fmt = ibm.IBMTrack_FixedDef('ibm.mfm')
                if nsec == 26: rpm = 360 # 8-inch disk
                fmt.rate = 500
            else:
                raise error.Fatal('IMD: Unrecognised track mode %x' % mode)

            sz_n_list: List[int] = []
            for secsz in sec_sizes:
                n = size_to_n(secsz)
                error.check(n is not None,
                            'IMD: Unsupported sector size %d for IBM track'
                            % secsz)
                sz_n_list.append(n)

            fmt.rpm = rpm
            fmt.secs, fmt.sz = nsec, sz_n_list
            fmt.finalise()
            t = fmt.mk_track(cyl, head)

            rmap = dat[i:i+nsec]
            i += nsec
            if has_cyl_map:
                cmap = dat[i:i+nsec]
                i += nsec
            if has_head_map:
                hmap = dat[i:i+nsec]
                i += nsec

            for nr,s in enumerate(t.sectors):
                s.crc = s.idam.crc = s.dam.crc = 0
                s.idam.r = rmap[nr]
                if has_cyl_map:
                    s.idam.c = cmap[nr]
                if has_head_map:
                    s.idam.h = hmap[nr]
                error.check(i < len(dat), 'IMD: Truncated sector record')
                rec = dat[i]
                i += 1
                error.check(0 <= rec <= 8,
                            'IMD: Unexpected sector code %x' % rec)
                if rec == 0:
                    # Data unavailable (header only)
                    s.dam.crc = s.crc = 0xffff
                    continue

                secsz = sec_sizes[nr]
                compressed = (rec % 2) == 0
                deleted = rec in (3, 4, 7, 8)
                data_error = rec in (5, 6, 7, 8)

                if compressed:
                    error.check(i < len(dat),
                                'IMD: Truncated compressed sector byte')
                    s.dam.data = bytes([dat[i]] * secsz)
                    i += 1
                else:
                    error.check(i + secsz <= len(dat),
                                'IMD: Truncated sector data')
                    s.dam.data = dat[i:i+secsz]
                    i += secsz

                if deleted:
                    s.dam.mark = ibm.Mark.DDAM
                if data_error:
                    s.dam.crc = s.crc = 0xffff

            self.to_track[cyl,head] = t


    def get_track(self, cyl: int, side: int) -> Optional[ibm.IBMTrack_Fixed]:
        if (cyl,side) not in self.to_track:
            return None
        return self.to_track[cyl,side] # type: ignore


    def emit_track(self, cyl: int, side: int, track) -> None:
        if isinstance(track, ibm.IBMTrack_Scan):
            track = track.track
        if isinstance(track, ibm.IBMTrack):
            if not isinstance(track, ibm.IBMTrack_Empty):
                self.to_track[cyl,side] = track
            return

        # Generic path: Any codec exposing sector-oriented data.
        error.check(hasattr(track, 'nsec'),
                    'IMD: Cannot create T%d.%d: Unsupported track type %s'
                    % (cyl, side, type(track).__name__))
        nsec = int(track.nsec)
        sectors: List[IMDSector] = []
        if hasattr(track, 'physical_sector_order'):
            order = [int(x) for x in track.physical_sector_order]
        else:
            order = list(range(nsec))
        seen = set()
        phys_order = []
        for sec_id in order:
            if 0 <= sec_id < nsec and sec_id not in seen:
                seen.add(sec_id)
                phys_order.append(sec_id)
        for sec_id in range(nsec):
            if sec_id not in seen:
                phys_order.append(sec_id)

        for r in phys_order:
            data: Optional[bytes]
            data_error = False
            present = None
            sector_c = cyl
            sector_h = side
            sector_r = r
            if hasattr(track, 'sector_cyls'):
                try:
                    sector_c = int(track.sector_cyls[r])
                except Exception:
                    sector_c = cyl
            if hasattr(track, 'sector_heads'):
                try:
                    sector_h = int(track.sector_heads[r])
                except Exception:
                    sector_h = side
            if hasattr(track, 'sector_ids'):
                try:
                    sector_r = int(track.sector_ids[r])
                except Exception:
                    sector_r = r
            if hasattr(track, 'sector_present'):
                try:
                    present = bool(track.sector_present[r])
                except Exception:
                    present = None
            if present is None and hasattr(track, 'has_sec') and callable(track.has_sec):
                present = bool(track.has_sec(r))

            if present is not None:
                if not present:
                    data = None
                else:
                    sec = track.sector[r]
                    data = bytes(sec) if sec is not None else None
            elif hasattr(track, 'sector'):
                sec = track.sector[r]
                data = bytes(sec) if sec is not None else None
            else:
                data = None
            if hasattr(track, 'sector_data_error'):
                try:
                    data_error = bool(track.sector_data_error[r])
                except Exception:
                    data_error = False
            sectors.append(IMDSector(sector_c, sector_h, sector_r, data,
                                     data_error=data_error))

        mode = IMDMode.MFM_250kbps
        if hasattr(track, 'imd_mode'):
            mode = int(track.imd_mode)
        elif hasattr(track, 'clock'):
            clock = float(track.clock)
            mode = IMDMode.MFM_500kbps if clock < 1.5e-6 else IMDMode.MFM_250kbps
        self.to_track[cyl,side] = IMDGenericTrack(mode, sectors)


    def ibm_mode_to_imd_mode(self, t) -> int:
        if t.mode is ibm.Mode.FM:
            if t.clock < 3.0e-6:
                return IMDMode.FM_500kbps # High Rate
            return IMDMode.FM_250kbps # 300 RPM
        assert t.mode is ibm.Mode.MFM
        if t.clock < 1.5e-6:
            return IMDMode.MFM_500kbps # High Rate
        return IMDMode.MFM_250kbps # 300 RPM


    def emit_imd_track(self, tdat: bytearray, c: int, h: int, mode: int,
                       sectors: List[IMDSector]) -> None:
        nsec = len(sectors)
        rmap, cmap, hmap = [], [], []
        sizes = []

        for s in sectors:
            rmap.append(s.r)
            cmap.append(s.c)
            hmap.append(s.h)
            sizes.append(len(s.data) if s.data is not None else 0)

        head = h
        for i in range(nsec):
            if cmap[i] != c:
                head |= 0x80
            if hmap[i] != h:
                head |= 0x40

        # Choose standard IMD size encoding or size table (0xff).
        non_zero_sizes = [x for x in sizes if x != 0]
        nominal = non_zero_sizes[0] if non_zero_sizes else 128
        sec_n = size_to_n(nominal)
        has_size_table = sec_n is None
        if not has_size_table:
            for dlen in non_zero_sizes:
                if dlen != nominal or size_to_n(dlen) is None:
                    has_size_table = True
                    break

        sec_n_hdr = 0xff if has_size_table else sec_n
        assert sec_n_hdr is not None

        tdat += struct.pack('5B', mode, c, head, nsec, sec_n_hdr)
        tdat += bytes(rmap)
        if head & 0x80:
            tdat += bytes(cmap)
        if head & 0x40:
            tdat += bytes(hmap)
        if has_size_table:
            for dlen in sizes:
                tdat += struct.pack('<H', dlen)

        for s in sectors:
            data = s.data if s.data is not None else bytes()
            dlen = len(data)
            deleted = s.deleted
            data_error = s.data_error

            if dlen == 0:
                rec = 0
                tdat += bytes([rec])
                continue

            compressed = data.count(data[0]) == dlen

            if not deleted and not data_error and not compressed:
                rec = 1
            elif not deleted and not data_error and compressed:
                rec = 2
            elif deleted and not data_error and not compressed:
                rec = 3
            elif deleted and not data_error and compressed:
                rec = 4
            elif not deleted and data_error and not compressed:
                rec = 5
            elif not deleted and data_error and compressed:
                rec = 6
            elif deleted and data_error and not compressed:
                rec = 7
            else:
                rec = 8

            tdat += bytes([rec])
            if compressed:
                tdat += data[:1]
            else:
                tdat += data


    def ibm_track_to_imd_sectors(self, t) -> List[IMDSector]:
        sectors = []
        for s in t.sectors:
            if not isinstance(s, ibm.Sector):
                continue
            data = s.dam.data if s.dam.data is not None else None
            sectors.append(IMDSector(
                c=s.idam.c, h=s.idam.h, r=s.idam.r, data=data,
                deleted=(s.dam.mark == ibm.Mark.DDAM),
                data_error=(s.dam.crc != 0)
            ))
        return sectors


    def get_image(self) -> bytes:

        tdat = bytearray()

        now = datetime.datetime.now()
        sig = ('IMD 1.17: %s\r\nGreaseweazle %s\r\n\x1a'
               % (now.strftime('%d/%m/%Y %H:%M:%S'), __version__))
        tdat += sig.encode()

        for (c,h),t in sorted(self.to_track.items()):
            if isinstance(t, ibm.IBMTrack):
                mode = self.ibm_mode_to_imd_mode(t)
                sectors = self.ibm_track_to_imd_sectors(t)
                self.emit_imd_track(tdat, c, h, mode, sectors)
            else:
                assert isinstance(t, IMDGenericTrack)
                self.emit_imd_track(tdat, c, h, t.mode, t.sectors)

        return tdat


# Local variables:
# python-indent: 4
# End:
