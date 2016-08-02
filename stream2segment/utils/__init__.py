# -*- coding: utf-8 -*-

from __future__ import print_function  # , unicode_literals
import yaml

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
import pandas as pd
from sqlalchemy.engine import create_engine
from stream2segment.s2sio.db.models import Base
from sqlalchemy.orm.session import sessionmaker


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


def pd_str(dframe):
    """Returns a dataframe to string with all rows and all columns, used for printing to log"""
    with pd.option_context('display.max_rows', len(dframe),
                           'display.max_columns', len(dframe.columns),
                           'max_colwidth', 50, 'expand_frame_repr', False):
        return str(dframe)


class DataFrame(pd.DataFrame):
    """An extension of pandas DataFrame, where indexing with [] (a.k.a. __getitem__ for those
       familiar with implementing class behavior in Python), i.e. selecting out lower-dimensional
       slices, works ignoring the case. This works obviously only if the argument of the slice
       is either a string or an iterable of strings. Thus, given a DataFrame `d` with columns
       'a' and 'B', d['A'] returns the same Series as d['a'] (using a pandas DataFrame, a KeyError
       would be raised), and d[['a', 'b']] returns the same pandas DataFrame as d[['a', 'B']]
       When the slicing would return a pandas DataFrame, a new DataFrame is returned so that the
       same ignoring-case slicing works on the returned DataFrame.

       For info see: http://pandas.pydata.org/pandas-docs/stable/indexing.html#basics
    """

    @staticmethod
    def _isstr__(elm):
        return isinstance(elm, basestring)

    def __getitem__(self, key):
        try:
            dfm = pd.DataFrame.__getitem__(self, key)
        except KeyError as kerr:
            # try to see if key is a string:
            reg = None
            if self._isstr__(key):
                reg = re.escape(key)
                expected_col_num = 1
            else:
                regarray = []
                try:
                    for k in key:
                        if not self._isstr__(k):
                            raise kerr
                        regarray.append("(?:%s)" % re.escape(k))
                    expected_col_num = len(regarray)
                    reg = "|".join([r for r in regarray])
                except TypeError:
                    raise kerr

            if reg is None:  # for safety ...
                raise kerr

            dfm = self.filter(regex=re.compile(r"^"+reg+"$", re.IGNORECASE))
            cols = len(dfm.columns)
            if cols != expected_col_num:
                raise kerr

            # original pandas slicing returns a series when single string:
            if cols == 1 and self._isstr__(key):
                dfm = dfm[dfm.columns[0]]  # returns a Series

        return DataFrame(dfm) if isinstance(dfm, pd.DataFrame) else dfm


def get_default_cfg_filepath():
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    return os.path.normpath(os.path.abspath(os.path.join(config_dir, "config.yaml")))


def load_def_cfg(filepath=None, raw=False):
    """Loads default config from yaml file"""
    if filepath is None:
        filepath = get_default_cfg_filepath()
    with open(filepath, 'r') as stream:
        ret = yaml.safe_load(stream) if not raw else stream.read()
    # load config file. This might be better implemented in the near future
    if not raw:
        configfilepath = os.path.abspath(os.path.dirname(filepath))
        # convert sqlite path to relative to the config
        sqlite_prefix = 'sqlite:///'
        newdict = {}
        for k, v in ret.iteritems():
            try:
                if v.startswith(sqlite_prefix):
                    dbpath = v[len(sqlite_prefix):]
                    if os.path.isabs(filepath):
                        newdict[k] = sqlite_prefix + \
                            os.path.normpath(os.path.join(configfilepath, dbpath))
            except AttributeError:
                pass
        if newdict:
            ret.update(newdict)
    return ret


def get_default_dbpath(config_filepath=None):
    if config_filepath is None:
        config_filepath = get_default_cfg_filepath()
    return load_def_cfg(config_filepath)['dburi']


def get_session(dbpath=None):
    """
    Create an sql alchemy session for IO db operations
    :param dbpath: the path to the database, e.g. sqlite:///path_to_my_dbase.sqlite
    if None or missing, it defaults to the 'dburi' field in config.yaml
    """
    if dbpath is None:
        dbpath = get_default_dbpath()
    # init the session:
    engine = create_engine(dbpath)
    Base.metadata.create_all(engine)
    # create a configured "Session" class
    Session = sessionmaker(bind=engine)
    # create a Session
    session = Session()
    return session

# ==end== 
