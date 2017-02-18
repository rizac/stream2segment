# -*- coding: utf-8 -*-
"""
    Common utilities for the program
"""
# from __future__ import print_function  # , unicode_literals
import os
import yaml
import re
import datetime as dt
from sqlalchemy.orm.scoping import scoped_session
from sqlalchemy.engine import create_engine
from sqlalchemy.orm.session import sessionmaker
from stream2segment.io.db.models import Base
# from click import progressbar as click_progressbar
from click._termui_impl import ProgressBar as ClickProgressBar
from click.globals import resolve_color_default
from click import progressbar as click_progressbar
import time
from collections import defaultdict


def isstr(val):
    """
    :return: True if val denotes a string (`basestring` in python2 and `str` otherwise).
    """
    try:
        return isinstance(val, basestring)
    except NameError:  # python3
        return isinstance(val, str)


def isunicode(val):
    """
    :return: True if val denotes a unicode string (`unicode` in python2 and `str` otherwise)
    """
    try:
        if isinstance(val, basestring):
            return isinstance(val, unicode)
    except NameError:  # python3
        return isinstance(val, str)


def tobytes(unicodestr, encoding='utf-8'):
    """
        Converts unicodestr to a byte sequence, with the given encoding. Python 2-3 compatible.
        :param unicodestr: a unicode string. If already byte string, this method just returns it
        :param encoding: the encoding used. Defaults to 'utf-8' when missing
        :return: a `bytes` object (same as `str` in python2) resulting from encoding unicodestr
    """
    if isinstance(unicodestr, bytes):  # works for both py2 and py3
        return unicodestr
    return unicodestr.encode(encoding)


def tounicode(bytestr, decoding='utf-8'):
    """
        Converts bytestr to unicode string, with the given decoding. Python 2-3 compatible.
        :param bytestr: a `bytes` object. If already unicode string (`unicode` in python2,
        `str` in python3) this method just returns it
        :param decoding: the decoding used. Defaults to 'utf-8' when missing
        :return: a string (`str` in python3, `unicode` string in python2) resulting from decoding
        `bytestr`
    """
    return bytestr.decode(decoding) if isinstance(bytestr, bytes) else bytestr


def strptime(string, formats=None):
    """
        Converts a date in string format into a datetime python object. The inverse can be obtained
        by calling dt.isoformat() (which returns 'T' as date time separator, and optionally
        microseconds if they are not zero). This function is an easy version of
        `dateutil.parser.parse` for parsing iso-like datetime format (e.g. fdnsws standard)
        without the need of a module import
        :param: string: if a datetime object, returns it. If date object, converts to datetime
        and returns it. Otherwise must be a string representing a datetime
        :type: string: a string, a date or a datetime object (in that case just returns it)
        :param formats: if list or iterable, it holds the strings denoting the formats to be used
        to convert string (in the order they are declared). If None (the default), the datetime
        format will be guessed from the string length among the following (with optional 'Z', and
        with 'T' replaced by space as vaild option):
           - '%Y-%m-%dT%H:%M:%S.%fZ'
           - '%Y-%m-%dT%H:%M:%SZ'
           - '%Y-%m-%dZ'
        :raise: ValueError if the string cannot be parsed
        :type: on_err_return_none: object or Exception
        :return: a datetime object
        :Example:
        ```
            strptime("2016-06-01T09:04:00.5600Z")
            strptime("2016-06-01T09:04:00.5600")
            strptime("2016-06-01 09:04:00.5600Z")
            strptime("2016-06-01T09:04:00Z")
            strptime("2016-06-01T09:04:00")
            strptime("2016-06-01 09:04:00Z")
            strptime("2016-06-01")
        ```
    """
    if isinstance(string, dt.datetime):
        return string

    string = string.strip()

    if formats is None:
        has_z = string[-1] == 'Z'
        has_t = 'T' in string
        if has_t or has_z or ' ' in string:
            t_str, z_str = 'T' if has_t else ' ', 'Z' if has_z else ''
            formats = ['%Y-%m-%d{}%H:%M:%S.%f{}'.format(t_str, z_str),
                       '%Y-%m-%d{}%H:%M:%S{}'.format(t_str, z_str)]
        else:
            formats = ['%Y-%m-%d']

    for dtformat in formats:
        try:
            return dt.datetime.strptime(string, dtformat)
        except ValueError:  # as exce:
            pass

    raise ValueError("%s: invalid date time" % string)


def get_proc_template_files():
    """Returns the tuple (pyton file, yaml config file) to be used for a processing template"""
    proc_template_dir = os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                  'process'), "templates")
    return os.path.abspath(os.path.join(proc_template_dir, "template1.py")),\
        os.path.abspath(os.path.join(proc_template_dir, "template1.conf.yaml"))


def get_default_cfg_filepath(filename='config.yaml'):
    """Returns the configuration file path (absolute path)"""
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    return os.path.normpath(os.path.abspath(os.path.join(config_dir, filename)))


def yaml_load(filepath=None, raw=False):
    """Loads default config from yaml file, normalizing relative sqlite file paths if any"""
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


def yaml_load_doc(filepath=None):
    """Loads the doc from a yaml. The doc is intended to be all consecutive commented lines (i.e.,
    without blank lines) before each top-level variable (nested variables are not considered).
    The returned dict is a defaultdict which returns an empty string for non-found documented
    variables
    :param filepath: if None, it defaults to `config.example.yaml`. Otherwise, the yaml file to
    read the doc from
    """
    if filepath is None:
        filepath = get_default_cfg_filepath('config.example.yaml')
    last_comment = ''
    prev_line = None
    reg = re.compile("([^:]+):.*")
    reg_comment = re.compile("\\s*#+(.*)")
    ret = defaultdict(str)
    with open(filepath, 'r') as stream:
        while True:
            line = stream.readline()
            if not line:
                break
            m = reg_comment.match(line)
            if m and m.groups():  # set comment (append or new if previous was a newline)
                comment = m.groups()[0]
                if prev_line == '\n':
                    last_comment = comment
                else:
                    last_comment += comment
            elif line in ('\n', '\r') or line[:2] == '\r\n':  # normalize newlines
                line = '\n'
            else:  # try to see if it's a variable, and in case set the doc (if any)
                m = reg.match(line)
                if m and m.groups():
                    ret[m.groups()[0]] = last_comment
                last_comment = ''
            prev_line = line
    return ret


def get_default_dbpath(config_filepath=None):
    if config_filepath is None:
        config_filepath = get_default_cfg_filepath()
    return yaml_load(config_filepath)['dburl']


def get_session(dbpath=None, scoped=False):
    """
    Create an sql alchemy session for IO db operations
    :param dbpath: the path to the database, e.g. sqlite:///path_to_my_dbase.sqlite
    if None or missing, it defaults to the 'dburi' field in config.yaml
    :param scoped: boolean (False by default) if the session must be scoped session
    """
    if dbpath is None:
        dbpath = get_default_dbpath()
    # init the session:
    engine = create_engine(dbpath)
    Base.metadata.create_all(engine)  # @UndefinedVariable
    if not scoped:
        # create a configured "Session" class
        session = sessionmaker(bind=engine)
        # create a Session
        return session()
    # return session
    else:
        session_factory = sessionmaker(bind=engine)
        return scoped_session(session_factory)


def get_progressbar(isterminal=False):
    """Returns a click progressbar if isterminal is True or, if False, a mock class which supports
    `with` statement and the .update(N) method (no-op). In this latter case the returned object
    is NOT an iterator (contrarily to click.progressbar) so you should use it like this:
    ```
        pbar = utils.progressbar(isterminal)  # <-- class instance, not object!
        with pbar(length=..., ...) as bar:
            # do your stuff in iterators and call
            bar.update(num_increments)  # will update the terminal with a progressbar, or
                                        # do nothing (no-op) if isterminal=True
    ```
    """
    if not isterminal:
        class DPB(object):
            # support for iteration, if iterator is given, support for update and __enter__ __exit__
            def __init__(self, *a, **v):
                pass

            def __enter__(self, *a, **v):
                return self

            def __exit__(self, *a, **v):
                pass

            def update(self, *a, **v):
                pass

        return DPB
    else:
        return click_progressbar
