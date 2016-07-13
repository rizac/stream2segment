# -*- coding: utf-8 -*-

from __future__ import print_function  # , unicode_literals

"""
    Some utilities which share common functions which I often re-use across projects. Most of the
    functions here are already implemented in many libraries, but none of which has all of them.
    NOTES:
      - Concerning type checking, for worshippers of duck-typing this might be blaspheme, but life
        is to complex let handle all its circumstances in few rules
      - Several functions are checking for string types and doing string conversion (a big complex
        matter when migrating from python2 to 3). As a reminder, we write it here once:
            ======= ============ ===============
                    byte strings unicode strings
            ======= ============ ===============
            Python2 "abc" [*]    u"abc"
            Python3 b"abc"       "abc" [*]
            ======= ============ =================

         [*]=default string object for the given python version
"""
try:
    import numpy as np

    def isnumpy(val):
        """
        :return: True if val is a numpy object (regarldess of its shape and dimension)
        """
        return type(val).__module__ == np.__name__
except ImportError as ierr:
    def isnumpy(val):
        raise ierr

import sys
import datetime as dt
import re
import os
from os import strerror, errno
import shutil
import time
import bisect
import signal

if 2 <= sys.version_info[0] < 3:
    def ispy2():
        """:return: True if the current python version is 2"""
        return True

    def ispy3():
        """:return: True if the current python version is 3"""
        return False
elif 3 <= sys.version_info[0] < 4:
    def ispy2():
        """:return: True if the current python version is 2"""
        return False

    def ispy3():
        """:return: True if the current python version is 3"""
        return True
else:
    def ispy2():
        """:return: True if the current python version is 2"""
        return False

    def ispy3():
        """:return: True if the current python version is 3"""
        return False

# Python 2 and 3: we might try and catch along the lines of:
# http://python-future.org/compatible_idioms.html
# BUT THIS HAS CONFLICTS WITH libraries importing __future__ (see e.g. obspy),
# if those libraries are imported BEFORE this module. this is safer:
if ispy3():
    from urllib.parse import urlparse, urlencode  # @UnresolvedImport
    from urllib.request import urlopen, Request  # @UnresolvedImport
    from urllib.error import HTTPError  # @UnresolvedImport
else:
    from urlparse import urlparse  # @Reimport
    from urllib import urlencode  # @Reimport
    from urllib2 import urlopen, Request, HTTPError  # @Reimport


def isstr(val):
    """
    :return: True if val denotes a string (`basestring` in python < 3 and `str` otherwise).
    """
    if ispy2():
        return isinstance(val, basestring)
    else:
        return isinstance(val, str)


def isunicode(val):
    """
    :return: True if val denotes a unicode string (`unicode` in python < 3 and `str` otherwise)
    """
    isstr
    if ispy2():
        return isinstance(val, unicode)
    else:
        return isinstance(val, str)


def isbytes(val):
    """
    :return: True if val denotes a byte string (`bytes` in both python 2 and 3). In py3, this means
    val is a sequence of bytes (not necessarily to be treated as string)
    """
    return isinstance(val, bytes)


def tobytes(unicodestr, encoding='utf-8'):
    """
        Converts unicodestr to a byte sequence, with the given encoding. Python 2-3 compatible.
        :param unicodestr: a unicode string. If already byte string, this method just returns it
        :param encoding: the encoding used. Defaults to 'utf-8' when missing
        :return: a `bytes` object (same as `str` in python2) resulting from encoding unicodestr
    """
    if isbytes(unicodestr):
        return unicodestr
    return unicodestr.encode(encoding)


def tounicode(bytestr, decoding='utf-8'):
    """
        Converts bytestr to unicode string, with the given decoding. Python 2-3 compatible.
        :param bytestr: a `bytes` object. If already unicode string, this method just returns it
        :param decoding: the decoding used. Defaults to 'utf-8' when missing
        :return: a string (`unicode` string in python2) resulting from decoding bytestr
    """
    if isstr(bytestr):
        return bytestr
    return bytestr.decode(decoding)


def isiterable(obj, include_strings=False):
    """Returns True if obj is an iterable and not a string (or both, if the second argument is True)
    Py2-3 compatible method
    :param obj: a python object
    :param include_strings: boolean, False by default: if obj is a string, returns false
    :Example:
    isiterable([]) -> True
    isiterable((x for x in xrange(3))) -> True
    isiterable(numpy.array(['a'])) -> True
    isiterable('a') -> False
    isiterable('a', True) or isiterable('a', include_strings=True) -> True
    """
    return hasattr(obj, '__iter__') and (include_strings or not isstr(obj))


def isre(val):
    """Returns true if val is a compiled regular expression"""
    return isinstance(val, re.compile(".").__class__)


def regex(arg, retval_if_none=re.compile(".*")):
    """Returns a regular expression built as follows:
        - if arg is already a regular expression, returns it
        - if arg is None, returns retval_if_none, which by default is ".*" (matches everything)
        - Returns the regular expression escaping str(arg) EXCEPT "?" and "*" which will be
        converted to their regexp equivalent (thus arg might be a string with wildcards, as in many
        string processing arguments)
        :return: A regular expression from arg
    """
    if isre(arg):
        return arg

    if arg is None:
        return retval_if_none

    return re.compile(re.escape(str(arg)).replace("\\?", ".").replace("\\*", ".*"))


def ensure(filepath, mode, mkdirs=False, error_type=OSError):
    """checks for filepath according to mode, raises an Exception instanceof error_type if the check
    returns false
    :param mode: either 'd', 'dir', 'r', 'fr', 'w', 'fw' (case insensitive). Checks if file_name is,
        respectively:
            - 'd' or 'dir': an existing directory
            - 'fr', 'r': file for reading (an existing file)
            - 'fw', 'w': file for writing (basically, an existing file or a path whose dirname
            exists)
    :param mkdirs: boolean indicating, when mode is 'file_w' or 'dir', whether to attempt to
        create the necessary path. Ignored when mode is 'r'
    :param error_type: The error type to be raised in case (defaults to OSError. Some libraries
        such as ArgumentPArser might require their own error
    :type error_type: any class extending BaseException (OsError, TypeError, ValueError etcetera)
    :raises: SyntaxError if some argument is invalid, or error_type if filepath is not valid
        according to mode and mkdirs
    :return: True if mkdir has been called
    """
    # to see OsError error numbers, see here
    # https://docs.python.org/2/library/errno.html#module-errno
    # Here we use two:
    # errno.EINVAL ' invalid argument'
    # errno.errno.ENOENT 'no such file or directory'
    if not filepath:
        raise error_type("{0}: '{1}' ({2})".format(strerror(errno.EINVAL),
                                                   str(filepath),
                                                   str(type(filepath))
                                                   )
                         )

    keys = ('fw', 'w', 'fr', 'r', 'd', 'dir')

    # normalize the mode argument:
    if mode.lower() in keys[2:4]:
        mode = 'r'
    elif mode.lower() in keys[:2]:
        mode = 'w'
    elif mode.lower() in keys[4:]:
        mode = 'd'
    else:
        raise error_type('{0}: mode argument must be in {1}'.format(strerror(errno.EINVAL),
                                                                    str(keys)))

    if errmsgfunc is None:  # build custom errormsgfunc if None
        def errmsgfunc(filepath, mode):
            if mode == 'w' or (mode == 'r' and not os.path.isdir(os.path.dirname(filepath))):
                return "{0}: '{1}' ({2}: '{3}')".format(strerror(errno.ENOENT),
                                                        os.path.basename(filepath),
                                                        strerror(errno.ENOTDIR),
                                                        os.path.dirname(filepath)
                                                        )
            elif mode == 'd':
                return "{0}: '{1}'".format(strerror(errno.ENOTDIR), filepath)
            elif mode == 'r':
                return "{0}: '{1}'".format(strerror(errno.ENOENT), filepath)

    if mode == 'w':
        to_check = os.path.dirname(filepath)
        func = os.path.isdir
        mkdir_ = mkdirs
    elif mode == 'd':
        to_check = filepath
        func = os.path.isdir
        mkdir_ = mkdirs
    else:  # mode == 'r':
        to_check = filepath
        func = os.path.isfile
        mkdir_ = False

    exists_ = func(to_check)
    mkdirdone = False
    if not func(to_check):
        if mkdir_:
            mkdirdone = True
            os.makedirs(to_check)
            exists_ = func(to_check)

    if not exists_:
        raise error_type(errmsgfunc(filepath, mode))

    return mkdirdone


def url_read(url, blockSize=1024*1024, decoding=None, raise_exc=True):
    """
        Reads and return data from the given url. Note that in case of IOException, the  data
        read until the exception is returned
        :param url: a valid url
        :type url: string
        :param blockSize: the block size while reading, defaults to 1024 ** 2
            (at most in chunks of 1 MB)
        :type blockSize: integer
        :param: decoding: the string used for decoding to string (e.g., 'utf8'). If None
        (the default), the result is returned as it is (byte string, note that in Python2 this is
        equivalent to string), otherwise as unicode string
        :type: decoding: string, or None
        :param raise_exc: if True (the default when missing) an exception is raised while reading
        blocks of data (an exception is ALWAYS raised while creating urlopen, prior to reading
        blocks of data). Otherwise, an exception is returned as second argument in the tuple, whose
        first argument is the bytes of data (or unicode string) read until that exception
        :return the data read, or empty string if None
        :rtype bytes of data (equivalent to string in python2), or unicode string, or the tuple
        bytes of data or unicode string, exception (the latter might be None)
        :raise: IOError, ValueError or TypeError in case of errors
    """
    dcResult = b''

    try:
        urlopen_ = urlopen(Request(url))
    except (IOError, OSError) as e:
        # note: urllib2 raises urllib2.URLError (subclass of IOError),
        # in python3 raises urllib.errorURLError (subclass of OSError)
        # in both cases there might be a reason or code attributes, which we
        # want to print
        str_ = ''
        if hasattr(e, 'reason'):
            str_ = '%s (Reason: %s)' % (e.__class__.__name__, e.reason)  # pylint:disable=E1103
        elif hasattr(e, 'code'):
            str_ = '%s (The server couldn\'t fulfill the request. Error code: %s)' % \
                    (e.__class__.__name__, e.code)  # pylint:disable=E1103
        else:
            str_ = '%s (%s)' % (e.__class__.__name__, str(e))

        raise IOError(str_)

    except (TypeError, ValueError) as _:
        # see https://docs.python.org/2/howto/urllib2.html#handling-exceptions
        raise

    # Read the data in blocks of predefined size
    # Note the read() method, if the size argument is omitted or negative, may not read until the
    # end of the data stream; there is no good way to determine that the entire stream from a socket
    # has been read in the general case.
    # See https://docs.python.org/2/library/urllib.html

    exc = None
    while True:
        try:
            buf = urlopen_.read(blockSize)
        except IOError as ioexc:  # urlopen behaves as a file-like obj.
            # Thus we catch the file-like exception IOError,
            # see https://docs.python.org/2.4/lib/bltin-file-objects.html
            if raise_exc:
                urlopen_.close()
                raise
            else:
                exc = ioexc
                buf = ''  # for safety (break the loop here below)

        if not buf:
            break
        dcResult += buf

    # Close the connection to avoid overloading the server
    urlopen_.close()

    # logging.debug('%s bytes read from %s', dcBytes, url)
    body = tounicode(dcResult, decoding) if decoding is not None else dcResult

    return body if raise_exc else (body, exc)

# these methods are implemented to avoid complex workarounds in testing.
# See http://blog.xelnor.net/python-mocking-datetime/
_datetime_now = dt.datetime.now
_datetime_utcnow = dt.datetime.utcnow
_datetime_strptime = dt.datetime.strptime


def datetime(string, formats=None, on_err=ValueError):
    """
        Converts a date in string format into a datetime python object. The inverse can be obtained
        by calling dt.isoformat() (which returns 'T' as date time separator, and optionally
        microseconds if they are not zero). This method is mainly used in argument parser from
        command line emulating an easy version of dateutil.parser.parse (without the need of that
        dependency) and assuming string is an iso-like datetime format (e.g. fdnsws standard)
        :param: string: if a datetime object, returns it. If date object, converts to datetime
        and returns it. Otherwise must be a string representing a datetime
        :type: string: a string, a date or a datetime object (in that case just returns it)
        :param formats: if list or iterable, it holds the strings denoting the formats to be used
        to convert string (in the order they are declared). If None (the default), the datetime
        format will be guessed from the string length among the following:
           - '%Y-%m-%dT%H:%M:%S.%fZ'
           - '%Y-%m-%dT%H:%M:%SZ'
           - '%Y-%m-%dZ'
        Note: once a candidate format is chosen, 'T' might be replaced by ' ' if string does not
        have 'T', and ''Z' (the zulu timezone) will be appended if string ends with 'Z'
        :param: on_err: if subclass of Exception, raises it in case of failure. Otherwise it is the
        return value in case of failure (e.g., on_err=ValueError, on_err=None)
        :type: on_err_return_none: object or Exception
        :return: a datetime object
        :Example:
        to_datetime("2016-06-01T09:04:00.5600Z")
        to_datetime("2016-06-01T09:04:00.5600")
        to_datetime("2016-06-01 09:04:00.5600Z")
        to_datetime("2016-06-01 09:04:00.5600Z")
        to_datetime("2016-06-01")
    """
    if isinstance(string, dt.datetime):
        return string

    if formats is None:
        len_ = len(string)
        if len_ <= 9:
            formats = []  # alias as: raiseon_err or return it
        else:
            # string search is faster: try to guess the format instead of looping
            end_ = 'Z' if string[-1] == 'Z' else ''
            sep_ = 'T' if 'T' in string else ' '
            if len_ > 19:
                formats = ['%Y-%m-%d' + sep_ + '%H:%M:%S.%f' + end_]
            elif len_ > 10:
                formats = ['%Y-%m-%d' + sep_ + '%H:%M:%S' + end_]
            else:
                formats = ['%Y-%m-%d' + end_]

    for dtformat in formats:
        try:
            return _datetime_strptime(string, dtformat)
        except ValueError:  # as exce:
            pass
        except TypeError as terr:
            try:
                raise on_err(str(terr))
            except TypeError:
                return on_err

    try:
        raise on_err("%s: invalid date time" % string)
    except TypeError:
        return on_err


def parsedb(string):
    p = re.compile(r'^(?P<dialect>.*?)(?:\+(?P<driver>.*?))?\:\/\/(?:(?P<username>.*?)\:'
                   '(?P<password>.*?)\@(?P<host>.*?)\:(?P<port>.*?))?\/(?P<database>.*?)$')
    m = p.match(string)
    return m


class EstRemTimer():
    """
        An object used to calculate the estimated remaining time (ert) in loops.
        There are two options:
        1) initialize ert=EstRemTimer(N) and then call ert.get(), ert.percent() or
        ert.percent_str() at the *start* of each loop. In this case note that ert.get() will be
        None the first time is called, and ert.get() will never show 0 time remaining as the last
        time is called the last loop has to be done, or:
        2) initialize ert=EstRemTimer(N, start_now=True) and then call ert.get(), ert.percent() or
        ert.percent_str() at the *end* of each loop. In this case, note that ert.get() will never
        be None and will show 0 time remaining the last time is called

        :Example: 
            etr = EstRemTimer(N)  # N number of iterations
            for i in xrange(N):
                etr.get()   # returns the ert as timedelta object (the first time jsut starts the
                            # internal timer returning None)
                            # and increments the internal counter. Call get(False) to return
                            # the last ert without incrementing the counter
                etr.done    # returns the number of iterations done (if get() has not been called
                            # at least TWICE, returns 0)
                etr.total   # returns N
                etr.progress(None)  # returns a float representing the percent done
                                    # (see note of etr.done)
                etr.progress()      # returns the formatted string of the percent done, e.g.
                                    # "  1%", " 15%", "100%"
                ... loop code here ...
    """
    def __init__(self, total_iterations, start_now=False, approx_to_seconds=True, use="mean"):
        """
            Initializes an EstRemTimer for calculating the estimated remaining time (ert)
            :param: total_iterations the total iterations this object is assumed to use for
            calculate the ert
            :type: total_iterations integer
            :param: start_now: False by default, set to True if the object methods get or
            print_progress are called at the end of the loop (i.e., when the first loop work has
            already been done)
            :param: approx_to_seconds: when True (the default) approximate the ert
            (timedelta object) to seconds
            :type: approx_to_seconds: boolean
            :param: use: if 'median' (case insensitive) calculates the estimated remaining time
            using the median of all durations of the iterations done. For any other string,
            use the mean. The default is "median" because it is less sensitive to skewed
            distributions, so basically iterations which take far more (or less) time than the
            average weight less in the computation of the ert.
        """
        self.total = total_iterations
        self.done = 0
        self._start_time = None if not start_now else time.time()
        self.ert = None
        self.approx_to_seconds = approx_to_seconds
        self._times = [] if use.lower() == "median" else None

    def percent(self, formatstr="{:>3.0f}%"):
        """
            Returns the percent done according to the internal counter (which is incremented by
            calling self.get(). I.e., calling this method several times returns always the same
            value until get() is called again).
            :param: format: if None, returns the float representing the percent done (in [0,1]).
            If string, returns a formatted string of the percent done (float within [0,1]). By
            default is "{:>3.0f}%", which means the percent done is rounded to the int in [0, 100]
            and displayed as, e.g., "  2%", " 58%", "100%" etcetera.
            :return: the percent done
            :rtype: string or float in [0,1] according to the argument (default: string)
        """
        num = 1 if not self.total else float(self.done) / self.total
        if formatstr is None:
            return num
        return formatstr.format(100 * num)

    def get(self, increment=True, approx_to_seconds=None):
        """
            Gets the estimated remaing time etr. If increment is True, the returned object is None
            the first time this method is called, at subsequent calls it will be a timedelta object.
            If increment is False, returns the last calculated ert (which might be None)
            :param: increment: (True by default) returns the ert and increments the internal counter
            :type: increment: boolean
            :param: approx_to_seconds: sets whether the ert is approximated to seconds. If None (the
            default) the value of the argument approx_to_seconds passed in the constructor (True
            by default) is used
            :type: approx_to_seconds: boolean, or None
            :return: the estimated remaining time, or None
            :rtype: timedelta object, or None
        """
        if increment:
            if self._start_time is None:
                self._start_time = time.time()  # start now timing
                # first iteration, leave done to zero so that user can query the 'done' attribute
                # and it correctly displays the done iteration
            else:
                self.done += 1
                if approx_to_seconds is None:
                    approx_to_seconds = self.approx_to_seconds
                if self.done >= self.total:
                    ret = dt.timedelta()
                else:
                    elapsed_time = time.time() - self._start_time
                    if self._times is not None:  # use median
                        # Find rightmost value less than or equal to ret:
                        i = bisect.bisect_right(self._times, elapsed_time)
                        self._times.insert(i, elapsed_time)
                        idx = len(self._times) / 2
                        ret = self._times[idx] if len(self._times) % 2 == 1 else \
                            (self._times[idx] + self._times[idx-1]) / 2
                        ret *= (self.total - self.done)
                        ret = dt.timedelta(seconds=int(ret + 0.5) if approx_to_seconds else ret)
                        self._start_time = time.time()  # re-start timer (for next iteration)
                    else:
                        ret = estremttime(elapsed_time, self.done, self.total, approx_to_seconds)
                self.ert = ret
        return self.ert

#     def signal_handler(signal, frame):
#         print 'You pressed Ctrl+C!'
#         sys.exit(0)
# 
#         signal.signal(signal.SIGINT, signal_handler)
#         print 'Press Ctrl+C'
#         while True:
#             time.sleep(1)


class Progress(object):
    bar_chars_length = 12
    empty_fill = u'∙'
    fill = u'█'
    bar_prefix = u'|'
    bar_fuffix = '|'
    out = sys.stdout
    show_percentage = True
    show_ert = True,
#     preamble = ''
#     epilog = ''
    clear_terminal_cursor = True
    start_time_immediately = False

    def __init__(self, number_of_iterations, **kwargs):
        for name, value in kwargs.iteritems():
            setattr(self, name, value)
        self.ert = EstRemTimer(number_of_iterations, start_now=self.start_time_immediately)
        self._uninit = True

    @staticmethod
    def clear_cursor(out=sys.stdout):
        print('\x1b[?25l', end='', file=out)

    @staticmethod
    def show_cursor(out=sys.stdout):
        print('\x1b[?25h', end='', file=out)

    @staticmethod
    def clear_line(out=sys.stdout):
        print('\r\x1b[K', end='', file=out)  # clear line

    def echo(self, preamble='', epilog=''):

        bar_chars_length = self.bar_chars_length
        empty_fill = self.empty_fill
        fill = self.fill
        bar_prefix = self.bar_prefix
        bar_fuffix = self.bar_fuffix
        out = self.out
        show_percentage = self.show_percentage
        show_ert = self.show_ert
        clear_terminal_cursor = self.clear_terminal_cursor

        if self._uninit and clear_terminal_cursor:  # first round
            self._uninit = False
            Progress.clear_cursor(out)
            # add a listener for restoring the cursor if keyboardinterrupt is pressed. See:
            # http://stackoverflow.com/questions/4205317/capture-keyboardinterrupt-in-python-without-try-except

            def signal_handler(signal, frame):
                Progress.show_cursor(out)
            signal.signal(signal.SIGINT, signal_handler)

        Progress.clear_line(out)
        ert = self.ert.get()
        percent = self.ert.percent(None)
        fill_len = max(0, min(bar_chars_length, int(percent * bar_chars_length + 0.5)))
        empty_len = bar_chars_length - fill_len
        line = '' if not preamble else preamble + " "
        line += bar_prefix + (fill * fill_len) + (empty_fill * empty_len) + bar_fuffix
        if show_percentage:
            line += self.ert.percent()
        if show_ert and ert is not None:
            line += u" ≈ %ss remaining." % str(ert)
        if epilog:
            line += " " + epilog
        # calculate the number of columns otherwise the line is NOT properly deleted
        # NOTE: this works when terminal size is too small, so we do not produce several lines
        # and it's fine when resizing UP. Unfortunately, when resizing DOWN in such a way that
        # line will be wrapped the problem is still persist and all rows up to (and not including)
        # the last row will stay on the terminal. For such a case, we should add a listener
        # for terminal resize, which is too much effort
        cols = get_terminal_cols()
        if cols is not None and cols < len(line):
            line = line[:max(0, cols-3)]
            if line:
                line += "..."
        print(line, end='', file=out)
        out.flush()
        if self.ert.done >= self.ert.total:
            print(file=out)  # print new line
            if clear_terminal_cursor:
                Progress.show_cursor(out)  # show cursor


def estremttime(elapsed_time, iteration_number, total_iterations, approx_to_seconds=True):
    """Called within a set of N=total_iterations "operations" (usually in a for loop) started since
    elapsed_time, this method returns a timedelta object representing the ESTIMATED remaining time
    when the iteration_number-th operation has been finished.
    Estimated means that the remaining time is calculated as if each of the remaining operations
    will take in average the average time taken for the operations done, which might not always be
    the case
    :Example:
    import time
    start_time = time.time()
    for i, elm in enumerate(events):  # events being e.g., a list / tuple or whatever
        elapsed = time.time() - start_time
        est_rt = str(estremttime(elapsed, i, len(events))) if i > 0 else "unwknown"
        ... your code here ...

    :param: elapsed_time: the time elapsed since the first operation (operation 0) started
    :type: elapsed_time a timedelta object, or any type castable to float (int, floats, numeric
        strings)
    :param: iteration_number: the number of operations done
    :type: iteration_number: a positive int
    :param: total_iterations: self-explanatory, specifies the total number of operations expected
    :type: total_iterations: a positive int greater or equal than iteration_number
    :param: approx_to_seconds: True by default if missing, returns the remaining time aproximated
        to seconds, which is sufficient for the typical use case of a process remaining time
        which must be shown to the user
    :type: approx_to_seconds: boolean
    :return: the estimated remaining time according to elapsed_time, which is the time taken to
        process iteration_number operations of a total number of total_iterations operations
    :rtype: timedelta object. Note that it's string value (str function) can be called to display
    the text of the estimated remaining time
    """
    if isinstance(elapsed_time, dt.timedelta):
        elapsed_time = elapsed_time.total_seconds()  # is a float
    else:
        elapsed_time = float(elapsed_time)  # to avoid rounding below (FIXME: use true division?)
    remaining_seconds = (total_iterations - iteration_number) * (elapsed_time / iteration_number)
    dttd = dt.timedelta(seconds=int(remaining_seconds+0.5)
                        if approx_to_seconds else remaining_seconds)
    return dttd


def get_terminal_size():
    return [get_terminal_rows(), get_terminal_cols()]


def get_terminal_cols():
    try:
        columns = int(os.popen('tput cols', 'r').read().strip())
        return columns
    except ValueError:
        try:
            _, columns = os.popen('rstty size', 'r').read().split()
            return int(columns)
        except ValueError:
            pass
    return None


def get_terminal_rows():
    try:
        rows = int(os.popen('tput lines', 'r').read().strip())
        return rows
    except ValueError:
        try:
            rows, _ = os.popen('rstty size', 'r').read().split()
            return int(rows)
        except ValueError:
            pass
    return None
