# -*- coding: utf-8 -*-

from __future__ import print_function # , unicode_literals
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
import errno
from os import strerror
import os
import time
import bisect

# Python 2 and 3: alternative 4
# see here:
# http://python-future.org/compatible_idioms.html
try:
    from urllib.parse import urlparse, urlencode
    from urllib.request import urlopen, Request
    from urllib.error import HTTPError
    raise ImportError()
except ImportError:
    from urlparse import urlparse
    from urllib import urlencode
    from urllib2 import urlopen, Request, HTTPError

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


def load_module(filepath, name=None):
    """
        Loads a python module indicated by filepath, returns an object where global variables
        and classes can be accessed as attributes
        See: http://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
        :param filepath: the path of the module
        :param name: defaults to None (implying that the filepath basename, without extension, will
            be taken) and it's only used to set the .__name__ of the returned module. It doesn't
            affect loading
    """
    if name is None:
        name = os.path.splitext(os.path.basename(filepath))[0]
    # name only sets the .__name__ of the returned module. it doesn't effect loading

    if ispy2():  # python 2
        import imp
        return imp.load_source(name, filepath)
    elif ispy3() and sys.version_info[1] >= 5:  # Python 3.5+:
        import importlib.util  # @UnresolvedImport
        spec = importlib.util.spec_from_file_location(name, filepath)
        modul = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modul)
        return modul
    else:  # actually, for Python 3.3 and 3.4, but we assume is the case also for 3.2 3.1 etcetera
        from importlib.machinery import SourceFileLoader  # @UnresolvedImport
        return SourceFileLoader(name, filepath).load_module()
        # (Although this has been deprecated in Python 3.4.)

    # raise SystemError("unsupported python version: "+ str(sys.version_info))


def url_read(url, blockSize=1024*1024, decoding=None):
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
        :return the data read, or empty string if None
        :rtype bytes of data (equivalent to string in python2), or unicode string
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

    except (TypeError, ValueError) as e:
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
            exc = ioexc
            buf = ''  # for safety (break the loop here below)

        if not buf:
            break
        dcResult += buf

    # Close the connection to avoid overloading the server
    urlopen_.close()

    if exc is not None:
        raise exc

    # logging.debug('%s bytes read from %s', dcBytes, url)
    return tounicode(dcResult, decoding) if decoding is not None else dcResult


# def prepare_datestr(string, ignore_z=True, allow_spaces=True):
#     """
#         "Prepares" string trying to make it datetime iso standard. This method basically gives the
#         opportunity to remove the 'Z' at the end (denoting the zulu timezone) and replaces spaces
#         with 'T'. NOTE: this methods returns the same string argument if any TypeError, IndexError
#         or AttributeError is found.
#         :param ignore_z: if True (the default), removes any 'Z' at the end of string, as 'Z' denotes
#             the "zulu" timezone
#         :param allow_spaces: if True (the default) all spaces of string will be replaced with 'T'.
#         :return a new string according to the arguments or the same string object
#     """
#     # kind of redundant but allows unit testing
#     try:
#         if ignore_z and string[-1] == 'Z':
#             string = string[:-1]
# 
#         if allow_spaces:
#             string = string.replace(' ', 'T')
#     except (TypeError, IndexError, AttributeError):
#         pass
# 
#     return string


# these methods are implemented to avoid complex workarounds in testing.
# See http://blog.xelnor.net/python-mocking-datetime/
_datetime_now = dt.datetime.now
_datetime_utcnow = dt.datetime.utcnow
_datetime_strptime = dt.datetime.strptime


def datetime(string, ignore_z=True,
             formats=['%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%d',
                      '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f'],
             on_err=ValueError):
    """
        Converts a date in string format into
        a datetime python object. The inverse can be obtained by calling
        dt.isoformat() (which returns 'T' as date time separator, and optionally microseconds
        if they are not zero). This method is mainly used in argparser from command line and it's
        not intended to be optimized for performance
        :param: string: if a datetime object, returns it. If date object, converts to datetime
        and returns it. Otherwise must be a string representing a datetime
        :type: string: a string, a date or a datetime object
        :param ignore_z: if True (the default), removes any 'Z' at the end of string, as 'Z' denotes
            the "zulu" timezone
        :type: ignore_z: boolean
        :param allow_spaces: if True (the default) for each string in formats an attempt will be done
        by replacing all 'T' with 
        all spaces of string will be replaced with 'T'.
        :type: allow_spaces: boolean
        :param: on_err_return_none: if True, does what it says (None is returned). Otherwise raises
        either a ValueError or a TypeError
        :type: on_err_return_none: boolean
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

    if ignore_z and string[-1] == 'Z':
        string = string[:-1]

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


def datetime2(string, formats=None, on_err=ValueError):
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

    def print_progress(self, length=12, empty_fill='∙', fill='█', bar_prefix='|',
                       bar_fuffix='|', out=sys.stdout, show_percentage=True, show_ert=True,
                       preamble='', epilog='', clear_cursor=False):
        if clear_cursor and (self._start_time is None or self.done == 0):  # first round
            print('\x1b[?25l', end='', file=out)
        print('\r\x1b[K', end='', file=out)  # clear line
        self.get()
        percent = self.percent(None)
        fill_len = max(0, min(length, int(percent * length + 0.5)))
        empty_len = length - fill_len
        line = '' if not preamble else preamble + " "
        line += bar_prefix + (fill * fill_len) + (empty_fill * empty_len) + bar_fuffix
        if show_percentage:
            line += self.percent()
        if show_ert and self.ert is not None:
            line += " ≈ %ss remaining." % str(self.ert)
        if epilog:
            line += " " + epilog
        # calculate the number of columns otherwise the line is NOT properly deleted
        cols = get_terminal_cols()
        if cols is not None and cols < len(line):
            line = line[:max(0, cols-3)]
            if line:
                line += "..."
        print(line, end='', file=out)
        out.flush()
        if self.done >= self.total:
            print(file=out)  # print new line
            if clear_cursor:
                print('\x1b[?25h', end='', file=out)  # show cursor


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

if __name__ == "__main__":
    
    dates = ["2006-01-01", 
             "2006-01-0g", 
             "2006-01-02 11:34:56",
             "2006-01-02T11:34:56",
             "2006-01-02 11:34:56.123",
             "2006-01-02T11:34:56.123"
             "2006-01-02 11:34:56Z",
             "2006-01-02T11:34:56Z",
             "2006-01-02 11:34:56.123Z",
             "2006-01-02T11:34:56.123Z"
             "2006-01-02Tsdf11:34:56.123Z"]
    
    C = 500
    import time
    import numpy as np
    
    print("datetime:")
    timez1 = []
    for d in dates:
        t = time.time()
        for i in xrange(C):
            datetime(d, ignore_z=True, on_err=None)
        timez1.append(time.time()-t)
        print("%s: %f" % (d, timez1[-1]))
        
    print("datetime2:")
    timez2 = []
    for d in dates:
        t = time.time()
        for i in xrange(C):
            datetime2(d, on_err=None)
        timez2.append(time.time()-t)
        print("%s: %f" % (d, timez2[-1]))
                
    print("datetime: %f" % np.mean(timez1))
    print("datetime2: %f" % np.mean(timez2))
    