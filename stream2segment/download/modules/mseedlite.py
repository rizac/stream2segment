"""Python-only Mini-SEED module with limited functionality.

.. moduleauthor:: Andres Heinloo <andres@gfz-potsdam.de>, GEOFON, GFZ Potsdam
.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
import datetime
import struct
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
            self.record_id = _get_id(net, sta, loc, cha)
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


class Input:
    """Iterate over the available Mini-SEED records. Keeps track of record ids
    in case of errors on a single mseed
    """

    def __init__(self, fd):
        """Create the iterable from the file handle passed as parameter.

        :param fd: either a bytes sequence or a BytesIO object. The bytes
            sequence must represent downloaded data **related to the same time
            span** (e.g. issued from a query with any network, station,
            location, channel but the same 'start' and 'end' parameters)
        """
        # Avoid creating a BytesIO in each Record otherwise we will infinitely
        # read the first chunk each time. This is not DRY (don't repeat
        # yourself) but avoids modifying the original code:
        if not hasattr(fd, "read"):
            fd = BytesIO(fd)
        self.__fd = fd

    def __iter__(self):
        """Define the iterator. Yields the tuple (Record, is_exc). Record is
        the record  read (if is_exc is False) or the MiniseedError raised
        (is_exc = True). In any  case, the Record object has the attribute
        ```record_id = "[network].[station].[location].[channel]" != None```
        This method raises for any kind of non-MiniseedError exception, or for
        MiniseedError occurred during header reading, i.e. for which the
        exception attribute "record_id" would be None.
        """
        while True:
            rec = Record(self.__fd)
            if rec.EOF:
                self.__fd.close()
                break
            yield rec


def _get_id(net, sta, loc, cha):
    """Return the id in the format ```net.sta.loc.cha```: all arguments should
    be bytes. The four arguments are network, station, location and channel
    code as read from the miniSEED bytes.

    :return: a string (unicode in python2)

    :raise: UnicodeDecodeError if any character cannot be decoded
    """
    return (b"%s.%s.%s.%s" %
            (net.strip(), sta.strip(), loc.strip(), cha.strip())).decode('utf8')


def unpack(data, starttime=None, endtime=None):
    """Unpack data into its "traces" (time series). Returns a dict where keys
    are the seed id as strings:
    "network.station.location.channel"
    mapped to a tuple
    ```
    (is_err, bytes_or_exc, s_rate, max_gap_overlap_ratio, start_time, end_time,
        out_of_bounds_chunks_found)
    ```
    where:
    exc is the exception raised while reading the miniseed (or None)
    data is the bytes data of the miniSEED (if exc is None) or None (if exc is
    not None) s_rate* is the sample rate (float)
    max_gap_overlap_ratio* is a float indicating the maximum gap (positive) or
        overlap (negative) found between all miniseed records. If zero, no
        gaps/ overlaps where found
    start_time*: (datetime) the miniseed start time (time of the first sample)
    end_time*: (datetime) the miniseed end_time (time of the last sample)
    out_of_bounds_chunks_found*: boolean, if either `starttime` or `endtime`
        are provided, reutrns True if some records of `data` where
        out-of-bounds and thus where discarded (not returned in `bytes`)

    * rely on these values only if `exc=None`. Otherwise, these values are None:
      so basically the user should check first and handle the error, and
      otherwise handle `data` and all other elements

    When no error occurs (exc=None) this method assures that:
    ```
    Stream(obspy.read(BytesIO(data)))
    ```
    and
    ```
    Stream([obspy.read(BytesIO(d[0]))[0] for d in unpack(data).values()])
    ```
    return the same object
    :param data: the bytes (or str in python2) representing waveform data as,
        e.g., returned from a query response
    :param starttime: the *expected* starttime (`datetime` object) or None (do
        not check for time bounds): if not None, all records completely before
        this value will not be returned.
    :param endtime: the *expected* endtime (`datetime` object) or None (do not
        check for time bounds): if not None, all records completely after this
        value will not be returned
    :return: a dictionary of keys tuples `(network, station location, channel)`
        mapped to the tuple representing the record read
    :raise MiniseedError if some error is raised, that is 'not' recoverable
        (e.g., header error, or bad file length for some record causing all
        subsequent records to be mis-aligned)
    """
    mseeds_to_read = {}

    # values of preocessed_mseed below are:
    #     exc (Exception or Npne)
    #     data (list of Records) or Exception (if is_exc is True)
    #     sample_rate (float),
    #     max_gap_overlap_ratio (float),
    #     starttime (datetime),
    #     endtime (datetime),
    #     out_of_time_chunks_found (boolean)
    # Example: [None, b'...', None, None, None, None, False]
    processed_mseeds = {}
    chunks_out_of_bounds = set()
    for rec in Input(data):
        id_ = rec.record_id

        if id_ in processed_mseeds:
            continue
        elif rec.error:
            processed_mseeds[id_] = \
                (MSeedError(rec.error), None, None, None, None, None, False)
            mseeds_to_read.pop(id_, None)
            continue

        if id_ not in mseeds_to_read:
            mseeds_to_read[id_] = []

        # check time bounds, and discard if chunk COMPLETELY out-of bound:
        if (starttime is not None and starttime > rec.end_time) or \
                (endtime is not None and endtime < rec.begin_time):
            chunks_out_of_bounds.add(id_)
            continue

        mseeds_to_read[id_].append(rec)

    for id_, records in mseeds_to_read.items():
        if not records:
            processed_mseeds[id_] = \
                (None, b'', None, None, None, None, id_ in chunks_out_of_bounds)
            continue
        # get records and sort ascending by time
        records.sort(key=lambda elm: elm.begin_time)
        fsamp = records[0].fsamp
        max_gap_overlap_ratio = 0
        bytesio = BytesIO()
        try:
            for i, record in enumerate(records):

                if record.fsamp != fsamp:
                    raise MSeedError("records sample rate mismatch")
                record.write(bytesio, int(log(record.size) / log(2)))

                if i == 0:
                    continue

                # curr_max_gap_ratio = distance between end_time of this chunk
                # and begin_time of next chunk.
                # curr_max_gap_ratio is in number of samples, thus
                # curr_max_gap_ratio *= fsamp.
                # If curr_max_gap_ratio == 1, then no gaps.
                # If > 1, possible gaps,
                # If < 1 possible overlaps.
                # Subtract 1 as we want 0 for no gaps/overlaps,
                # >0 for possible gaps, and <0 for possible overlaps:
                curr_max_gap_ratio = \
                    (record.begin_time - records[i-1].end_time).total_seconds() * fsamp - 1
                if abs(curr_max_gap_ratio) > abs(max_gap_overlap_ratio):
                    max_gap_overlap_ratio = curr_max_gap_ratio

            processed_mseeds[id_] = (None,
                                     bytesio.getvalue(),
                                     fsamp,
                                     max_gap_overlap_ratio,
                                     records[0].begin_time,
                                     records[-1].end_time,
                                     id_ in chunks_out_of_bounds)
            bytesio.close()

        except MSeedError as mserr:
            processed_mseeds[id_] = (mserr, None, None, None, None, None, False)

    return processed_mseeds
