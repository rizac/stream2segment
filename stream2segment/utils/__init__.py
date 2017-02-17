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


# def parsedb(string):
#     p = re.compile(r'^(?P<dialect>.*?)(?:\+(?P<driver>.*?))?\:\/\/(?:(?P<username>.*?)\:'
#                    '(?P<password>.*?)\@(?P<host>.*?)\:(?P<port>.*?))?\/(?P<database>.*?)$')
#     m = p.match(string)
#     return m


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
    without blank lines) before each variable.
    The returned dict is a defaultdict which returns the empty string for non-found documented
    variables"""
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


# FIXME: stuff below should be cleaned!!!

# def get_progressbar(isterminal=False):
#     """Returns a class that will be display a progressbar (see `click.progressbar`), or a subclass
#     of click.progressbar if isterminal=True. The subclass implements all superclass methods
#     but it actually does not render anything nor performs any calculation (e.g. eta, percent
#     etcetera)
#     ```
#         pbar = utils.progressbar(isterminal)  # <-- class instance, not object!
#         with pbar(...) as bar:
#             # do your stuff in iterators and call
#             bar.update(num_increments)  # will update the terminal with a progressbar, or do nothing
#                                         # if isterminal=True
#     ```
#     """
#     if not isterminal:
#         # return a "mock" object which supports iteration, with statement and .update method
#         # NOTE: THIS HAS BEEN TESTED FOR with statements and update, so this should be safe:
#         # with progressbar(False) as bar:
#         #    ...
#         #    bar.update(N)  # no-op
# 
#         class DPB(object):
#             # support for iteration, if iterator is given, support for update and __enter__ __exit__
#             def __init__(self, *a, **v):
#                 if len(a):
#                     iterable = a[0]
#                 else:
#                     iterable = v.get('iterable', None)
#                 if len(a) > 1:
#                     length = a[1]
#                 else:
#                     length = v.get('length', None)
# 
#                 if iterable is None and length is None:
#                     raise TypeError('iterable or length is required')  # copied from ProgressBar
# 
#                 self._i = iterable
# 
#             def __enter__(self, *a, **v):
#                 return self
# 
#             def __iter__(self):   # @DontTrace # pylint:disable=non-iterator-returned
#                 if not self._i:
#                     raise TypeError('object is not iterable, you should provide an iterator '
#                                     'in the constructor')
#                 return self
# 
#             def __next__(self):
#                 return next(self.i)
# 
#             def __exit__(self, *a, **v):
#                 pass
# 
#             def update(self, *a, **v):
#                 pass
# 
#         return DPB
#     else:
#         return click_progressbar

# # copied and pasted from click progressbar
# def _progressbar(iterable=None, length=None, label=None, show_eta=True,
#                 show_percent=None, show_pos=False,
#                 item_show_func=None, fill_char='\xe2\x96\x88', empty_char='-',
#                 bar_template='%(label)s  |%(bar)s|  %(info)s',
#                 info_sep='  ', width=31, file=None, color=None):
#     """This function creates an iterable context manager that can be used
#     to iterate over something while showing a progress bar.  It will
#     either iterate over the `iterable` or `length` items (that are counted
#     up).  While iteration happens, this function will print a rendered
#     progress bar to the given `file` (defaults to stdout) and will attempt
#     to calculate remaining time and more.  By default, this progress bar
#     will not be rendered if the file is not a terminal.
# 
#     The context manager creates the progress bar.  When the context
#     manager is entered the progress bar is already displayed.  With every
#     iteration over the progress bar, the iterable passed to the bar is
#     advanced and the bar is updated.  When the context manager exits,
#     a newline is printed and the progress bar is finalized on screen.
# 
#     No printing must happen or the progress bar will be unintentionally
#     destroyed.
# 
#     Example usage::
# 
#         with progressbar(items) as bar:
#             for item in bar:
#                 do_something_with(item)
# 
#     Alternatively, if no iterable is specified, one can manually update the
#     progress bar through the `update()` method instead of directly
#     iterating over the progress bar.  The update method accepts the number
#     of steps to increment the bar with::
# 
#         with progressbar(length=chunks.total_bytes) as bar:
#             for chunk in chunks:
#                 process_chunk(chunk)
#                 bar.update(chunks.bytes)
# 
#     .. versionadded:: 2.0
# 
#     .. versionadded:: 4.0
#        Added the `color` parameter.  Added a `update` method to the
#        progressbar object.
# 
#     :param iterable: an iterable to iterate over.  If not provided the length
#                      is required.
#     :param length: the number of items to iterate over.  By default the
#                    progressbar will attempt to ask the iterator about its
#                    length, which might or might not work.  If an iterable is
#                    also provided this parameter can be used to override the
#                    length.  If an iterable is not provided the progress bar
#                    will iterate over a range of that length.
#     :param label: the label to show next to the progress bar.
#     :param show_eta: enables or disables the estimated time display.  This is
#                      automatically disabled if the length cannot be
#                      determined.
#     :param show_percent: enables or disables the percentage display.  The
#                          default is `True` if the iterable has a length or
#                          `False` if not.
#     :param show_pos: enables or disables the absolute position display.  The
#                      default is `False`.
#     :param item_show_func: a function called with the current item which
#                            can return a string to show the current item
#                            next to the progress bar.  Note that the current
#                            item can be `None`!
#     :param fill_char: the character to use to show the filled part of the
#                       progress bar.
#     :param empty_char: the character to use to show the non-filled part of
#                        the progress bar.
#     :param bar_template: the format string to use as template for the bar.
#                          The parameters in it are ``label`` for the label,
#                          ``bar`` for the progress bar and ``info`` for the
#                          info section.
#     :param info_sep: the separator between multiple info items (eta etc.)
#     :param width: the width of the progress bar in characters, 0 means full
#                   terminal width
#     :param file: the file to write to.  If this is not a terminal then
#                  only the label is printed.
#     :param color: controls if the terminal supports ANSI colors or not.  The
#                   default is autodetection.  This is only needed if ANSI
#                   codes are included anywhere in the progress bar output
#                   which is not the case by default.
#     """
#     color = resolve_color_default(color)
#     return ProgressBar(iterable=iterable, length=length, show_eta=show_eta,
#                        show_percent=show_percent, show_pos=show_pos,
#                        item_show_func=item_show_func, fill_char=fill_char,
#                        empty_char=empty_char, bar_template=bar_template,
#                        info_sep=info_sep, file=file, label=label,
#                        width=width, color=color)
# 
# 
# class ProgressBar(ClickProgressBar):
#     RENDER_TIMEDELTA_IN_SEC = 3
# 
#     def __init__(self, *a, **kw):
#         super(ProgressBar, self).__init__(*a, **kw)
#         self._last_render = None
# 
#     def render_progress(self):
#         if self._last_render is not None and not self.finished:
#             s = time.time()
#             if s - self._last_render < ProgressBar.RENDER_TIMEDELTA_IN_SEC:
#                 return
#         ClickProgressBar.render_progress(self)
#         if not self.finished:
#             self._last_render = time.time()
# # ==end==
