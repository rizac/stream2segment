"""Python-only Mini-SEED module with limited functionality.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

   :Copyright:
       2005-2025 GFZ Helmholtz Centre for Geosciences
   :License:
       GPLv3
   :Platform:
       Linux
"""

from __future__ import absolute_import, division, print_function

import datetime
import struct
import sys
import re
import io
import json

_FIXHEAD_LEN = 48
_BLKHEAD_LEN = 4
_BLK1000_LEN = 4
_BLK1001_LEN = 4
_MAX_RECLEN = 4096

_doy = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365)


def _is_leap(y):
    """True if y is a leap year."""
    return (y % 400 == 0) or (y % 4 == 0 and y % 100 != 0)


def _ldoy(y, m):
    """The day of the year of the first day of month m, in year y.

    Note: for January, m=1; for December, m=12.
    Examples:
    _ldoy(1900, 4) = 90
    _ldoy(1900, 1) = 0
    _ldoy(1999, 4) = 90
    _ldoy(2004, 4) = 91
    _ldoy(2000, 4) = 91

    """
    return _doy[m - 1] + (_is_leap(y) and m >= 3)


def _dy2mdy(doy, year):
    month = 1
    while doy > _ldoy(year, month + 1):
        month += 1

    mday = doy - _ldoy(year, month)
    return (month, mday)


def _mdy2dy(month, day, year):
    return _ldoy(year, month) + day


class EndOfData(Exception):
    """."""


class MSeedError(Exception):
    """."""


class MSeedNoData(MSeedError):
    """."""


_rx_id = re.compile(r"^FDSN:([^_]*)_([^_]*)_([^_]*)_([^_]*)_([^_]*)_([^_]*)$")


class Record(object):
    """Mini-SEED record."""

    def __init__(self, src):
        """Create a Mini-SEED record from a file handle or a bitstream."""
        if type(src) == bytes:
            fd = io.BytesIO(src)
        elif hasattr(src, "read"):
            fd = src
        else:
            raise TypeError("argument is neither bytes nor a file object")

        # self.header = ""
        self.header = bytes()
        fixhead = fd.read(_FIXHEAD_LEN)

        if len(fixhead) == 0:
            # FIXME Check if there is no better option, but NOT StopIteration!
            raise EndOfData

        if fixhead[0:2] == b"MS":
            if fixhead[2] == 3:
                self.__init_ms3(fd, fixhead)

            else:
                raise MSeedError(f"unsupported version: {fixhead[2]}")

        else:
            self.formatversion = 2
            self.__init_ms2(fd, fixhead)

        if self.encoding not in (10, 11):
            return

        (self.X0, self.Xn) = struct.unpack(">ll", self.data[4:12])

        (w0,) = struct.unpack(">L", self.data[:4])
        (w3,) = struct.unpack(">L", self.data[12:16])
        c3 = (w0 >> 24) & 0x3
        d0 = None

        if self.encoding == 10:
            # """STEIM (1) Compression?"""
            if c3 == 1:
                d0 = (w3 >> 24) & 0xFF
                if d0 > 0x7F:
                    d0 -= 0x100
            elif c3 == 2:
                d0 = (w3 >> 16) & 0xFFFF
                if d0 > 0x7FFF:
                    d0 -= 0x10000
            elif c3 == 3:
                d0 = w3 & 0xFFFFFFFF
                if d0 > 0x7FFFFFFF:
                    d0 -= 0xFFFFFFFF
                    d0 -= 1

        elif self.encoding == 11:
            # """STEIM (2) Compression?"""
            if c3 == 1:
                d0 = (w3 >> 24) & 0xFF
                if d0 > 0x7F:
                    d0 -= 0x100
            elif c3 == 2:
                dnib = (w3 >> 30) & 0x3
                if dnib == 1:
                    d0 = w3 & 0x3FFFFFFF
                    if d0 > 0x1FFFFFFF:
                        d0 -= 0x40000000
                elif dnib == 2:
                    d0 = (w3 >> 15) & 0x7FFF
                    if d0 > 0x3FFF:
                        d0 -= 0x8000
                elif dnib == 3:
                    d0 = (w3 >> 20) & 0x3FF
                    if d0 > 0x1FF:
                        d0 -= 0x400
            elif c3 == 3:
                dnib = (w3 >> 30) & 0x3
                if dnib == 0:
                    d0 = (w3 >> 24) & 0x3F
                    if d0 > 0x1F:
                        d0 -= 0x40
                elif dnib == 1:
                    d0 = (w3 >> 25) & 0x1F
                    if d0 > 0xF:
                        d0 -= 0x20
                elif dnib == 2:
                    d0 = (w3 >> 24) & 0xF
                    if d0 > 0x7:
                        d0 -= 0x10

        if d0 is not None:
            self.X_minus1 = self.X0 - d0
        else:
            self.X_minus1 = None

        if (self.nframes is None) or (self.nframes == 0):
            i = 0
            self.nframes = 0
            while i < len(self.data):
                if self.data[i] == "\0":
                    break

                i += 64
                self.nframes += 1

    def __init_ms3(self, fd, fixhead):
        (
            self.formatversion,
            self.flags,
            bt_ns,
            bt_year,
            bt_doy,
            bt_hour,
            bt_minute,
            bt_second,
            self.encoding,
            self.fsamp,
            self.nsamp,
            self.crc,
            self.dataversion,
            id_len,
            extra_len,
            payload_len,
        ) = struct.unpack("<xxBBLHHBBBBdLLBBHL", fixhead[:40])

        self.size = 40 + id_len + extra_len + payload_len

        if self.size > len(fixhead):
            data = fixhead + fd.read(self.size - len(fixhead))

        else:
            data = fixhead

        if len(data) < 40 + id_len + extra_len:
            raise MSeedError(
                f"record size inconsistency: {len(data)} < {40 + id_len + extra_len}"
            )

        self.header = data[: 40 + id_len + extra_len]
        self.data = data[40 + id_len + extra_len :]

        if self.size != len(self.header) + len(self.data):
            raise MSeedError(
                f"record size inconsistency: {self.size} != {len(self.header) + len(self.data)}"
            )

        self.id = data[40 : 40 + id_len].decode("utf-8")

        try:
            self.extra = json.loads(
                data[40 + id_len : 40 + id_len + extra_len].decode("utf-8")
            )

        except Exception as e:
            raise MSeedError(f"invalid extra header: {self.extra}")

        m = _rx_id.match(self.id)

        if m is None:
            raise MSeedError(f"unsupported ID: {self.id}")

        # MS2 compatibility
        self.recno = 0
        self.rectype = "D"
        self.net = m.group(1)
        self.sta = m.group(2)
        self.loc = m.group(3)
        self.cha = m.group(4) + m.group(5) + m.group(6)
        self.aflgs = 0
        self.cflgs = 0
        self.qflgs = 0
        self.byteorder = 1
        self.nframes = None
        self.time_correction = 0

        (self.samprate_num, self.samprate_denom) = float.as_integer_ratio(self.fsamp)

        if self.fsamp < 0:  # negative sample rate represents period
            self.fsamp = -1 / self.fsamp
            (self.samprate_num, self.samprate_denom) = (
                self.samprate_denom,
                -self.samprate_num,
            )

        if self.samprate_num == 0:
            self.sr_factor = 0
            self.sr_mult = 0

        elif self.samprate_denom == 1:
            self.sr_factor = self.samprate_num
            self.sr_mult = 1

        else:
            self.sr_factor = -self.samprate_denom
            self.sr_mult = self.samprate_num

        try:
            self.time_quality = self.extra["FDSN"]["Time"]["Quality"]

        except KeyError:
            self.time_quality = -1

        # quick fix to avoid exception from datetime
        if bt_second > 59:
            self.leap = bt_second - 59
            bt_second = 59
        else:
            self.leap = 0

        try:
            (month, day) = _dy2mdy(bt_doy, bt_year)
            self.begin_time = datetime.datetime(
                bt_year, month, day, bt_hour, bt_minute, bt_second, bt_ns // 1000
            )

            self.nanoseconds = bt_ns % 1000

            if (self.nsamp != 0) and (self.fsamp != 0):
                msAux = 1000000 * self.nsamp / self.fsamp
                self.end_time = self.begin_time + datetime.timedelta(microseconds=msAux)
            else:
                self.end_time = self.begin_time

        except ValueError as e:
            raise MSeedError(f"invalid time: {str(e)}")

    def __init_ms2(self, fd, fixhead):
        if len(fixhead) < _FIXHEAD_LEN:
            raise MSeedError("unexpected end of header")

        (
            recno_str,
            self.rectype,
            sta,
            loc,
            cha,
            net,
            bt_year,
            bt_doy,
            bt_hour,
            bt_minute,
            bt_second,
            bt_tms,
            self.nsamp,
            self.sr_factor,
            self.sr_mult,
            self.aflgs,
            self.cflgs,
            self.qflgs,
            self.__num_blk,
            self.time_correction,
            self.__pdata,
            self.__pblk,
        ) = struct.unpack(">6scx5s2s3s2s2H3Bx2H2h4Bl2H", fixhead)

        if sys.version_info[0] > 2:
            recno_str = recno_str.decode("utf-8")
            self.rectype = self.rectype.decode("utf-8")
            sta = sta.decode("utf-8")
            loc = loc.decode("utf-8")
            cha = cha.decode("utf-8")
            net = net.decode("utf-8")

        self.header += fixhead

        if self.rectype not in ("D", "R", "Q", "M"):
            fd.read(_MAX_RECLEN - _FIXHEAD_LEN)
            raise MSeedNoData("non-data record")

        if self.__pdata >= _MAX_RECLEN:
            raise MSeedError(
                f"invalid pointer at {net.strip()}.{sta.strip()}.{loc.strip()}.{cha.strip()}: "
                f"record size ({self.__pdata}) >= {_MAX_RECLEN}"
            )
        if self.__pdata < _FIXHEAD_LEN or (
            self.__pblk != 0
            and ((self.__pblk < _FIXHEAD_LEN) or (self.__pblk >= self.__pdata))
        ):
            raise MSeedError(
                f"invalid pointer at {net.strip()}.{sta.strip()}.{loc.strip()}.{cha.strip()}"
            )

        if self.__pblk == 0:
            blklen = 0
        else:
            blklen = self.__pdata - self.__pblk
            gaplen = self.__pblk - _FIXHEAD_LEN
            gap = fd.read(gaplen)
            if len(gap) < gaplen:
                raise MSeedError("unexpected end of data")

            self.header += gap

        # defaults
        self.encoding = 11
        self.byteorder = 1
        rec_len_exp = 12
        self.time_quality = -1
        micros = 0
        self.nframes = None
        self.__rec_len_exp_idx = None
        self.__micros_idx = None
        self.__nframes_idx = None

        pos = 0
        while pos < blklen:
            blkhead = fd.read(_BLKHEAD_LEN)
            if len(blkhead) < _BLKHEAD_LEN:
                raise MSeedError(f"unexpected end of blockettes at{pos}{len(blkhead)}")

            (blktype, nextblk) = struct.unpack(">2H", blkhead)
            self.header += blkhead
            pos += _BLKHEAD_LEN

            if blktype == 1000:
                blk1000 = fd.read(_BLK1000_LEN)
                if len(blk1000) < _BLK1000_LEN:
                    raise MSeedError(
                        f"unexpected end of blockettes at {pos}{len(blk1000)}"
                    )

                (self.encoding, self.byteorder, rec_len_exp) = struct.unpack(
                    ">3Bx", blk1000
                )

                self.__rec_len_exp_idx = self.__pblk + pos + 2
                self.header += blk1000
                pos += _BLK1000_LEN

            elif blktype == 1001:
                blk1001 = fd.read(_BLK1001_LEN)
                if len(blk1001) < _BLK1001_LEN:
                    raise MSeedError(
                        f"unexpected end of blockettes at {pos}{len(blk1001)}"
                    )

                (self.time_quality, micros, self.nframes) = struct.unpack(
                    ">BbxB", blk1001
                )

                self.__micros_idx = self.__pblk + pos + 1
                self.__nframes_idx = self.__pblk + pos + 3
                self.header += blk1001
                pos += _BLK1001_LEN

            if nextblk == 0:
                break

            if nextblk < self.__pblk + pos or nextblk >= self.__pdata:
                raise MSeedError("invalid pointers")

            gaplen = nextblk - (self.__pblk + pos)
            gap = fd.read(gaplen)
            if len(gap) < gaplen:
                raise MSeedError("unexpected end of data")

            self.header += gap
            pos += gaplen

        if pos > blklen:
            raise MSeedError("corrupt record")

        gaplen = self.__pdata - len(self.header)
        gap = fd.read(gaplen)
        if len(gap) < gaplen:
            raise MSeedError("unexpected end of data")

        self.header += gap
        pos += gaplen

        self.recno = int(recno_str)
        self.net = net.strip()
        self.sta = sta.strip()
        self.loc = loc.strip()
        self.cha = cha.strip()

        if (self.sr_factor > 0) and (self.sr_mult > 0):
            self.samprate_num = self.sr_factor * self.sr_mult
            self.samprate_denom = 1
        elif (self.sr_factor > 0) and (self.sr_mult < 0):
            self.samprate_num = self.sr_factor
            self.samprate_denom = -self.sr_mult
        elif (self.sr_factor < 0) and (self.sr_mult > 0):
            self.samprate_num = self.sr_mult
            self.samprate_denom = -self.sr_factor
        elif (self.sr_factor < 0) and (self.sr_mult < 0):
            self.samprate_num = 1
            self.samprate_denom = self.sr_factor * self.sr_mult
        else:
            self.samprate_num = 0
            self.samprate_denom = 1

        self.fsamp = float(self.samprate_num) / float(self.samprate_denom)

        # quick fix to avoid exception from datetime
        if bt_second > 59:
            self.leap = bt_second - 59
            bt_second = 59
        else:
            self.leap = 0

        try:
            (month, day) = _dy2mdy(bt_doy, bt_year)
            self.begin_time = datetime.datetime(
                bt_year, month, day, bt_hour, bt_minute, bt_second
            )

            self.begin_time += datetime.timedelta(microseconds=bt_tms * 100 + micros)

            if (self.nsamp != 0) and (self.fsamp != 0):
                msAux = 1000000 * self.nsamp / self.fsamp
                self.end_time = self.begin_time + datetime.timedelta(microseconds=msAux)
            else:
                self.end_time = self.begin_time

        except ValueError as e:
            raise MSeedError(f"invalid time: {str(e)}")

        self.size = 1 << rec_len_exp
        if (self.size < len(self.header)) or (self.size > _MAX_RECLEN):
            raise MSeedError("invalid record size")

        datalen = self.size - self.__pdata
        self.data = fd.read(datalen)
        if len(self.data) < datalen:
            raise MSeedError("unexpected end of data")

        if len(self.header) + len(self.data) != self.size:
            raise MSeedError("internal error")

    def merge(self, rec):
        """Caller is expected to check for contiguity of data.

        Check if rec.nframes * 64 <= len(data)?
        """
        if self.formatversion != 2:
            raise MSeedError(
                f"merging version {self.formatversion} records is not implemented"
            )

        (self.Xn,) = struct.unpack(">l", rec.data[8:12])
        self.data += rec.data[: rec.nframes * 64]
        self.nframes += rec.nframes
        self.nsamp += rec.nsamp
        self.size = len(self.header) + len(self.data)
        self.end_time = rec.end_time

    def write(self, fd, rec_len_exp):
        """Write the record to an already opened file."""
        if self.formatversion != 2:
            raise MSeedError(
                f"writing version {self.formatversion} records is not implemented"
            )

        if self.size > (1 << rec_len_exp):
            raise MSeedError(
                f"record is larger than requested write size: {self.size} > {1 << rec_len_exp}"
            )

        recno_str = bytes(("%06d" % (self.recno,)).encode("utf-8"))
        sta = bytes(("%-5.5s" % (self.sta,)).encode("utf-8"))
        loc = bytes(("%-2.2s" % (self.loc,)).encode("utf-8"))
        cha = bytes(("%-3.3s" % (self.cha,)).encode("utf-8"))
        net = bytes(("%-2.2s" % (self.net,)).encode("utf-8"))
        bt_year = self.begin_time.year
        bt_doy = _mdy2dy(
            self.begin_time.month, self.begin_time.day, self.begin_time.year
        )
        bt_hour = self.begin_time.hour
        bt_minute = self.begin_time.minute
        bt_second = self.begin_time.second + self.leap
        bt_tms = self.begin_time.microsecond // 100
        micros = self.begin_time.microsecond % 100

        # This is just to make it Python 2 AND 3 compatible (str vs. bytes)
        rectype = (
            self.rectype.encode("utf-8") if sys.version_info[0] > 2 else self.rectype
        )

        buf = struct.pack(
            ">6s2c5s2s3s2s2H3Bx2H2h4Bl2H",
            recno_str,
            rectype,
            b" ",
            sta,
            loc,
            cha,
            net,
            bt_year,
            bt_doy,
            bt_hour,
            bt_minute,
            bt_second,
            bt_tms,
            self.nsamp,
            self.sr_factor,
            self.sr_mult,
            self.aflgs,
            self.cflgs,
            self.qflgs,
            self.__num_blk,
            self.time_correction,
            self.__pdata,
            self.__pblk,
        )
        fd.write(buf)

        buf = list(self.header[_FIXHEAD_LEN:])

        if self.__rec_len_exp_idx is not None:
            buf[self.__rec_len_exp_idx - _FIXHEAD_LEN] = struct.pack(">B", rec_len_exp)

        if self.__micros_idx is not None:
            buf[self.__micros_idx - _FIXHEAD_LEN] = struct.pack(">b", micros)

        if self.__nframes_idx is not None:
            buf[self.__nframes_idx - _FIXHEAD_LEN] = struct.pack(">B", self.nframes)

        ba = bytearray()
        for b in buf:
            try:
                ba.append(b)
            except Exception:
                ba.append(int.from_bytes(b, byteorder="big"))
        fd.write(ba)

        buf = (
            self.data[:4]
            + struct.pack(">ll", self.X0, self.Xn)
            + self.data[12:]
            + ((1 << rec_len_exp) - self.size) * b"\0"
        )

        fd.write(buf)


class Input(object):
    """Iterate over the available Mini-SEED records."""

    def __init__(self, fd):
        """Create the iterable from the file handle passed as parameter."""
        self.__fd = fd

    def __iter__(self):
        """Define the iterator."""
        while True:
            try:
                yield Record(self.__fd)

            except EndOfData:
                # This change follows new PEP-479, where it is explicitly forbidden to
                # use StopIteration
                # raise StopIteration
                return

            except MSeedNoData:
                pass
