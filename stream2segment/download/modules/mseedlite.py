"""Python-only Mini-SEED module with limited functionality.

.. moduleauthor:: Andres Heinloo <andres@gfz-potsdam.de>, GEOFON, GFZ Potsdam
.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""

from __future__ import annotations

import datetime
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from math import log
from io import BytesIO


_FIXHEAD_LEN = 48
_BLKHEAD_LEN = 4
_BLK1000_LEN = 4
_BLK1001_LEN = 4
_MAX_RECLEN = 4096

_doy = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365)


def _is_leap(y):
    """True if y is a leap year."""
    return ((y % 400 == 0) or (y % 4 == 0 and y % 100 != 0))


def _ldoy(y, m):
    """The day of the year of the first day of month m, in year y.

    Note: for January, m=0; for December, m=11.
    Examples:
    _ldoy(1900, 3) = 90
    _ldoy(1900, 0) = 0
    _ldoy(1999, 3) = 90
    _ldoy(2004, 3) = 91
    _ldoy(2000, 3) = 91

    """
    return _doy[m] + (_is_leap(y) and m >= 2)


def _dy2mdy(doy, year):
    month = 1
    while doy > _ldoy(year, month):
        month += 1

    mday = doy - _ldoy(year, month - 1)
    return (month, mday)


def _mdy2dy(month, day, year):
    return _ldoy(year, month - 1) + day


class MSeedError(Exception):
    """Custom mseed exception"""
    def __init__(self, message):
        super(MSeedError, self).__init__(message)


class Record:
    """Mini-SEED record."""

    def __init__(self, fd):
        """Create a Mini-SEED record from a file handle or a bitstream.

        :param fd: any object (File descriptor, BytesIO) with a 'read' attribute
        """

        # fd is the file pointer to a sequence of bytes of miniSEED records
        # (not necessarily from a single waveform data). We have 2 types of
        # errors
        # 1) because we reached the EOF -> store the error message in
        #    self.error and return. The miniSEED of the current record will be
        #    malformed, all other miniSEED previously read are unaffected
        # 2) Any other error preventing us to move to the start of the next
        #    record -> raise MseedError. All miniSEED will be malformed. This
        #    means skipping successfully read records, but it is safer to do so
        #    than saving potentially partial data
        # 3) Any other error NOT preventing to move to the next record
        #    -> store the error message in self.error and continue.
        #    The miniSEED of the current record
        #    will be malformed, all other miniSEED are unaffected

        # self.header = ""
        self.header = bytes()
        fixhead = fd.read(_FIXHEAD_LEN)

        self.error = ''
        self.EOF = False
        if len(fixhead) == 0:
            self.EOF = True
            return

        if len(fixhead) < _FIXHEAD_LEN:
            raise MSeedError("unexpected end of header")

        try:
            (recno_str, self.rectype, sta, loc, cha, net, bt_year, bt_doy, bt_hour,
             bt_minute, bt_second, bt_tms, self.nsamp, self.sr_factor,
             self.sr_mult, self.aflgs, self.cflgs, self.qflgs, self.__num_blk,
             self.time_correction, self.__pdata, self.__pblk) = \
                struct.unpack(">6scx5s2s3s2s2H3Bx2H2h4Bl2H", fixhead)
        except struct.error as serr:
            raise MSeedError(str(serr))

        try:
            self.recno = int(recno_str)
        except (TypeError, ValueError) as exc:
            self.error = 'recno not integer'

        self.net = net.strip()
        self.sta = sta.strip()
        self.loc = loc.strip()
        self.cha = cha.strip()

        try:
            # self.record_id = _get_id(net, sta, loc, cha)
            self.record_id = (
                b"%s.%s.%s.%s" % (net.strip(), sta.strip(), loc.strip(), cha.strip())
            ).decode('utf8')
        except UnicodeDecodeError as exc:
            # raise MSeedError so it will be caught
            raise MSeedError(str(exc))

        self.header += fixhead

        if ((self.rectype != b'D') and (self.rectype != b'R') and
                (self.rectype != b'Q') and (self.rectype != b'M')):
            # what do we do here below? seems we know how to move to the next
            # block, how?
            fd.read(_MAX_RECLEN - _FIXHEAD_LEN)
            self.error = "non-data record"
            return

        if ((self.__pdata < _FIXHEAD_LEN) or (self.__pdata >= _MAX_RECLEN) or
            ((self.__pblk != 0) and ((self.__pblk < _FIXHEAD_LEN) or
                                     (self.__pblk >= self.__pdata)))):
            # what do we do here below? seems we know how to move to the next
            # block, how?
            fd.read(_MAX_RECLEN - _FIXHEAD_LEN)
            self.error = "invalid pointers"
            return

        if self.__pblk == 0:
            blklen = 0
        else:
            blklen = self.__pdata - self.__pblk
            gaplen = self.__pblk - _FIXHEAD_LEN
            gap = fd.read(gaplen)
            if len(gap) < gaplen:
                self.error = "unexpected end of data"
                return

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
                self.error = "unexpected end of blockettes at %d" % \
                    (pos + len(blkhead))
                return

            try:
                (blktype, nextblk) = struct.unpack(">2H", blkhead)
            except struct.error as serr:
                raise MSeedError(str(serr))

            self.header += blkhead
            pos += _BLKHEAD_LEN

            if blktype == 1000:
                blk1000 = fd.read(_BLK1000_LEN)
                if len(blk1000) < _BLK1000_LEN:
                    self.error = "unexpected end of blockettes at %d" % \
                        (pos + len(blk1000))
                    return

                try:
                    (self.encoding, self.byteorder, rec_len_exp) = \
                        struct.unpack(">3Bx", blk1000)
                except struct.error as serr:
                    raise MSeedError(str(serr))

                self.__rec_len_exp_idx = self.__pblk + pos + 2
                self.header += blk1000
                pos += _BLK1000_LEN

            elif blktype == 1001:
                blk1001 = fd.read(_BLK1001_LEN)
                if len(blk1001) < _BLK1001_LEN:
                    self.error = "unexpected end of blockettes at %d" % \
                        (pos + len(blk1001))
                    return

                try:
                    (self.time_quality, micros, self.nframes) = \
                        struct.unpack(">BbxB", blk1001)
                except struct.error as serr:
                    self.error = str(serr)
                    # do not return, try to reach next Record start

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
                self.error = "unexpected end of data"
                return

            self.header += gap
            pos += gaplen

        if pos > blklen:
            raise MSeedError("corrupt record")

        gaplen = self.__pdata - len(self.header)
        gap = fd.read(gaplen)
        if len(gap) < gaplen:
            self.error = "unexpected end of data"
            return

        self.header += gap
        pos += gaplen

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

        self.fsamp = self.samprate_num / self.samprate_denom

        # quick fix to avoid exception from datetime
        if bt_second > 59:
            self.leap = bt_second - 59
            bt_second = 59
        else:
            self.leap = 0

        try:
            (month, day) = _dy2mdy(bt_doy, bt_year)
            self.begin_time = datetime.datetime(bt_year, month, day, bt_hour,
                                                bt_minute, bt_second)

            self.begin_time += \
                datetime.timedelta(microseconds=bt_tms*100+micros)

            if (self.nsamp != 0) and (self.fsamp != 0):
                msAux = 1000000 * (self.nsamp - 1) / self.fsamp
                self.end_time = self.begin_time + datetime.timedelta(microseconds=msAux)
            else:
                self.end_time = self.begin_time

        except ValueError as verr:
            self.error = "invalid time: %s" % str(verr)  # err type 2
            # do not return, try to reach next Record start

        self.size = 1 << rec_len_exp
        if (self.size < len(self.header)) or (self.size > _MAX_RECLEN):
            raise MSeedError("invalid record size")  # err type 1

        datalen = self.size - self.__pdata
        self.data = fd.read(datalen)

        # we got to the next Record start. From now on, all error types are 2
        # and we can return to avoid unnecessary operations. In any case,
        # the record's miniseed will not be marked as malformed

        if len(self.data) < datalen:
            self.error = "unexpected end of data"  # err type 2
            return

        if len(self.header) + len(self.data) != self.size:
            self.error = "internal error"  # err type 2
            return

        if self.error:
            # we might have an error set, we reached the next block, just return
            return  # err type 2

        try:
            (self.X0, self.Xn) = struct.unpack(">ll", self.data[4:12])
            (w0,) = struct.unpack(">L", self.data[:4])
            (w3,) = struct.unpack(">L", self.data[12:16])
        except struct.error as serr:
            self.error = str(serr)  # err type 2
            return

        c3 = (w0 >> 24) & 0x3
        d0 = None

        if self.encoding == 10:  # STEIM (1) Compression?
            if c3 == 1:
                d0 = (w3 >> 24) & 0xff
                if d0 > 0x7f:
                    d0 -= 0x100
            elif c3 == 2:
                d0 = (w3 >> 16) & 0xffff
                if d0 > 0x7fff:
                    d0 -= 0x10000
            elif c3 == 3:
                d0 = w3 & 0xffffffff
                if d0 > 0x7fffffff:
                    d0 -= 0xffffffff
                    d0 -= 1

        elif self.encoding == 11:  # STEIM (2) Compression?
            if c3 == 1:
                d0 = (w3 >> 24) & 0xff
                if d0 > 0x7f:
                    d0 -= 0x100
            elif c3 == 2:
                dnib = (w3 >> 30) & 0x3
                if dnib == 1:
                    d0 = w3 & 0x3fffffff
                    if d0 > 0x1fffffff:
                        d0 -= 0x40000000
                elif dnib == 2:
                    d0 = (w3 >> 15) & 0x7fff
                    if d0 > 0x3fff:
                        d0 -= 0x8000
                elif dnib == 3:
                    d0 = (w3 >> 20) & 0x3ff
                    if d0 > 0x1ff:
                        d0 -= 0x400
            elif c3 == 3:
                dnib = (w3 >> 30) & 0x3
                if dnib == 0:
                    d0 = (w3 >> 24) & 0x3f
                    if d0 > 0x1f:
                        d0 -= 0x40
                elif dnib == 1:
                    d0 = (w3 >> 25) & 0x1f
                    if d0 > 0xf:
                        d0 -= 0x20
                elif dnib == 2:
                    d0 = (w3 >> 24) & 0xf
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

#     def merge(self, rec):
#         """Caller is expected to check for contiguity of data.
#
#         Check if rec.nframes * 64 <= len(data)?
#         """
#         (self.Xn,) = struct.unpack(">l", rec.data[8:12])
#         self.data += rec.data[:rec.nframes * 64]
#         self.nframes += rec.nframes
#         self.nsamp += rec.nsamp
#         self.size = len(self.header) + len(self.data)
#         self.end_time = rec.end_time

    def write(self, fd, rec_len_exp):
        """Write the record to an already opened file."""
        if self.size > (1 << rec_len_exp):
            raise MSeedError("record is larger than requested write size")

        recno_str = bytes(b"%06d" % (self.recno,))
        sta = bytes(b"%-5.5s" % (self.sta,))
        loc = bytes(b"%-2.2s" % (self.loc,))
        cha = bytes(b"%-3.3s" % (self.cha,))
        net = bytes(b"%-2.2s" % (self.net,))
        bt_year = self.begin_time.year
        bt_doy = _mdy2dy(self.begin_time.month, self.begin_time.day,
                         self.begin_time.year)
        bt_hour = self.begin_time.hour
        bt_minute = self.begin_time.minute
        bt_second = self.begin_time.second + self.leap
        bt_tms = self.begin_time.microsecond // 100
        micros = self.begin_time.microsecond % 100

        buf = struct.pack(">6s2c5s2s3s2s2H3Bx2H2h4Bl2H", recno_str,
                          self.rectype, b' ', sta, loc, cha, net, bt_year,
                          bt_doy, bt_hour, bt_minute, bt_second, bt_tms,
                          self.nsamp, self.sr_factor, self.sr_mult, self.aflgs,
                          self.cflgs, self.qflgs, self.__num_blk,
                          self.time_correction, self.__pdata, self.__pblk)
        fd.write(buf)

        buf = list(self.header[_FIXHEAD_LEN:])

        if self.__rec_len_exp_idx is not None:
            buf[self.__rec_len_exp_idx - _FIXHEAD_LEN] = \
                struct.pack(">B", rec_len_exp)

        if self.__micros_idx is not None:
            buf[self.__micros_idx - _FIXHEAD_LEN] = struct.pack(">b", micros)

        if self.__nframes_idx is not None:
            buf[self.__nframes_idx - _FIXHEAD_LEN] = \
                struct.pack(">B", self.nframes)

        ba = bytearray()
        for b in buf:
            try:
                ba.append(b)
            except:
                ba.append(int.from_bytes(b, byteorder='big'))
        fd.write(ba)

        buf = self.data[:4] + struct.pack(">ll", self.X0, self.Xn) + \
            self.data[12:] + ((1 << rec_len_exp) - self.size) * b'\0'

        fd.write(buf)

# FIXME REMOVE
# def _get_id(net, sta, loc, cha):
#     """Return the id in the format ```net.sta.loc.cha```: all arguments should
#     be bytes. The four arguments are network, station, location and channel
#     code as read from the miniSEED bytes.
#
#     :return: a string (unicode in python2)
#
#     :raise: UnicodeDecodeError if any character cannot be decoded
#     """
#     return (b"%s.%s.%s.%s" %
#             (net.strip(), sta.strip(), loc.strip(), cha.strip())).decode('utf8')


def unpack(data: bytes | BytesIO) -> Iterable[MiniSeedInfo]:
    """
    Unpack data into its "traces" (time series). Returns an iterable of MiniSeedInfo
    """
    stream = data
    close_stream = False
    if not isinstance(stream, BytesIO):
        stream = BytesIO(data)
        close_stream = True

    mseeds = {}
    mseeds_error_ids = set()
    try:
        while True:
            rec = Record(stream)
            if rec.EOF:
                break

            seed_id = rec.record_id

            if seed_id in mseeds_error_ids:
                continue

            if rec.error:
                mseeds_error_ids.add(seed_id)
                yield MiniSeedInfo(seed_id, MSeedError(rec.error))
                continue

            mseeds.setdefault(seed_id, []).append(rec)

            # # check time bounds, and discard if chunk COMPLETELY out-of bound:
            # if (starttime is not None and starttime > rec.end_time) or \
            #         (endtime is not None and endtime < rec.begin_time):
            #     chunks_out_of_bounds.add(seed_id)
            #     continue

    finally:
        if close_stream:
            stream.close()

    # for miniseed_info in mseeds_errors.values():
    #     yield miniseed_info

    for seed_id, records in mseeds.items():

        try:
            if not records:  # for safety
                raise MSeedError('No data')

            # get records and sort ascending by time
            records.sort(key=lambda elm: elm.begin_time)
            fsamp = records[0].fsamp
            max_gap_overlap_ratios: list[float] = []
            bytesio = BytesIO()

            for i, record in enumerate(records):

                if record.fsamp != fsamp:
                    raise MSeedError("records sample rate mismatch")

                try:
                    record.write(bytesio, int(log(record.size) / log(2)))
                except Exception:
                    raise MSeedError("error packing miniseed records")

                # if i == 0:
                #     continue

                # curr_max_gap_ratio = distance between end_time of this chunk
                # and begin_time of next chunk.
                # curr_max_gap_ratio is in number of samples, thus
                # curr_max_gap_ratio *= fsamp.
                # If curr_max_gap_ratio == 1, then no gaps.
                # If > 1, possible gaps,
                # If < 1 possible overlaps.
                # Subtract 1 as we want 0 for no gaps/overlaps,
                # >0 for possible gaps, and <0 for possible overlaps:
                go_ratio = (
                    (record.begin_time - records[i-1].end_time).total_seconds()
                    * fsamp - 1
                )
                max_gap_overlap_ratios.append(go_ratio)
                # if abs(curr_max_gap_ratio) > abs(max_gap_overlap_ratio):
                #     max_gap_overlap_ratio = curr_max_gap_ratio

            yield MiniSeedInfo(
                seed_id,
                bytesio.getvalue(),
                fsamp,
                records[0].begin_time,
                records[-1].end_time,
                max(max_gap_overlap_ratios, key=abs)
            )

            bytesio.close()

        except MSeedError as ms_err:
            yield MiniSeedInfo(seed_id, ms_err)


@dataclass(slots=True, frozen=True)
class MiniSeedInfo:
    seed_id: str | int
    data: bytes | Exception
    fsamp: float | None = None
    start: datetime.datetime | None = None
    end: datetime.datetime | None = None
    maxgap_overlap_ratio: float | None = None

    @property
    def is_ok(self):
        return not isinstance(self.data, Exception)
