# -*- coding: utf-8 -*-
"""
Common utilities for the whole program

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
# make open py2-3 compatible. Call 'from stream2segment.utils import open'
# (http://python-future.org/imports.html#explicit-imports):
from builtins import open as compatible_open

from future.utils import string_types, itervalues, PY2

# this can not apparently be fixed with the future package:
# The problem is io.StringIO accepts unicode in python2 and strings in Py3:
if PY2:
    from cStringIO import StringIO  # noqa
else:
    from io import StringIO  # noqa

import os
import sys
import re
# import time
from itertools import chain
from datetime import datetime, timedelta
# from collections import defaultdict
import inspect
from contextlib import contextmanager
from dateutil import parser as dateparser
from dateutil.tz import tzutc

import yaml
from click import progressbar as click_progressbar


def _getmodulename(pyfilepath):
    """Return a (most likely) unique module name for a python source file
    loaded as a module
    """
    # In both python2 and 3, the builtin function importing a module from file
    # needs two arguments, a 'file path' and a 'name'. It's not clear why the
    # letter is necessary and does not default to, e.g., the filepath 's name.
    # We build the name here following these conventions:
    # 1. The name must be UNIQUE: otherwise when importing the second file the
    #    module of the former is actually returned
    # 2. Names should NOT contain dots, as otherwise a
    #    `RuntimeWarning: Parent module ... not found` is issued.
    return os.path.abspath(os.path.realpath(pyfilepath)).replace(".", "_dot_").\
        replace(os.path.sep, "_pathsep_")
    # note above: os.path.sep returns '/' on mac, os.pathsep returns ':'


# python 2 and 3 compatible code:
if sys.version_info[0] > 2:  # python 3+ (FIXME: what if Python4?)
    import importlib.util  # noqa

    def load_source(pyfilepath):
        """Load a source python file and returns it"""
        name = _getmodulename(pyfilepath)
        spec = importlib.util.spec_from_file_location(name, pyfilepath)  # noqa
        mod_ = importlib.util.module_from_spec(spec)  # noqa
        spec.loader.exec_module(mod_)
        return mod_

    def is_mod_function(pymodule, func, include_classes=False):
        """Return True if the python function `func` is a function (or class if
        `include_classes` is True) defined (and not imported) in the Python
        module `pymodule`
        """
        is_candidate = inspect.isfunction(func) or \
            (include_classes and inspect.isclass(func))
        # check that the source file is the module (i.e. not imported). NOTE that
        # getsourcefile might raise (not the case for functions or classes)
        return is_candidate and os.path.abspath(inspect.getsourcefile(pymodule)) == \
            os.path.abspath(inspect.getsourcefile(func))

else:
    import imp  # noqa

    def load_source(pyfilepath):
        """Load a source python file and returns it"""
        name = _getmodulename(pyfilepath)
        return imp.load_source(name, pyfilepath)  # noqa

    def is_mod_function(pymodule, func, include_classes=False):
        """Return True if the python function `func` is a function (or class if
        `include_classes` is True) defined (and not imported) in the Python
        module `pymodule`
        """
        is_candidate = inspect.isfunction(func) or \
            (include_classes and inspect.isclass(func))
        # check that the source file is the module (i.e. not imported). NOTE that
        # getsourcefile might raise (not the case for functions or classes)
        return is_candidate and inspect.getmodule(func) == pymodule


def iterfuncs(pymodule, include_classes=False):
    """Return an iterator over all functions (or classes if `include_classes`
    is True) defined (and not imported) in the given python module `pymodule`
    """
    for func in itervalues(pymodule.__dict__):
        if is_mod_function(pymodule, func, include_classes):
            yield func


class strconvert(object):
    """String conversion utilities from sql-LIKE operator's wildcards,
    Filesystem's wildcards, and regular expressions
    """
    @staticmethod
    def sql2wild(text):
        """Return a new string from `text` by replacing all sql-LIKE-operator's
        wildcard characters ('sql') with their filesystem's counterparts
        ('wild'):

        === ==== === ===============================
        sql wild re  meaning
        === ==== === ===============================
        %   *    .*  matches zero or more characters
        _   ?    .   matches exactly one character
        === ==== === ===============================

        :return: string. Note that this function performs a simple
            replacement: wildcard characters in the input string will result in
            a string that is not the perfect translation of the input
        """
        return text.replace("%", "*").replace("_", "?")

    @staticmethod
    def wild2sql(text):
        """Return a new string from `text` by replacing all filesystem's wildcard
        characters ('wild') with their sql-LIKE-operator's counterparts ('sql'):

        === ==== === ===============================
        sql wild re  meaning
        === ==== === ===============================
        %   *    .*  matches zero or more characters
        _   ?    .   matches exactly one character
        === ==== === ===============================

        :return: string. Note that this function performs a simple replacement:
            sql special characters in the input string will result in a string
            that is not the perfect translation of the input
        """
        return text.replace("*", "%").replace("?", "_")

    @staticmethod
    def wild2re(text):
        """Return a new string from `text` by replacing all filesystem's wildcard
        characters ('wild') with their regular expression's counterparts ('re'):

        === ==== === ===============================
        sql wild re  meaning
        === ==== === ===============================
        %   *    .*  matches zero or more characters
        _   ?    .   matches exactly one character
        === ==== === ===============================

        :return: string. Note that this function performs a simple replacement:
            regexp special characters in the input string will result in a
            string that is not the perfect translation of the input
        """
        return re.escape(text).replace(r"\*", ".*").replace(r"\?", ".")

    @staticmethod
    def sql2re(text):
        """Return a new string from `text` by replacing all sql-LIKE-operator's
        wildcard characters ('sql') with their regular expression's
        counterparts ('re'):

        === ==== === ===============================
        sql wild re  meaning
        === ==== === ===============================
        %   *    .*  matches zero or more characters
        _   ?    .   matches exactly one character
        === ==== === ===============================

        :return: string. Note that this function performs a simple replacement:
            regexp special characters in the input string will result in a
            string that is not the perfect translation of the input
        """
        if PY2 or sys.version_info[1] < 3:
            # py2 and py3.3- escape "_" (insert '\' before) AND '%':
            percent, underscore = r"\%", r"\_"
        elif sys.version_info[1] < 7:
            # versions up to 3.7 do not escape anymore "_":
            percent, underscore = r"\%", "_"
        else:
            # from version 3.7, only special characters are escaped,
            # thus neither "%" nor "_" are escaped:
            percent, underscore = "%", "_"
        return re.escape(text).replace(percent, ".*").replace(underscore, ".")


def tounicode(string, decoding='utf-8'):
    """Convert string to 'text' (unicode in python2, str in Python3). Function
    Python 2-3 compatible. If string is already a 'text' type, returns it

    :param string: a `str`, 'bytes' or (in py2) 'unicode' object.
    :param decoding: the decoding used if `string` has to be converted to text.
        Defaults to 'utf-8' when missing
    :return: the text (`str` in python3, `unicode` string in Python2)
        representing `string`
    """
    # Curiously, future.utils has no such a simple method. So instead of
    # checking when string is text, let's check when it is NOT, i.e. when it's
    # instance of bytes (str in py2 is instanceof bytes):
    return string.decode(decoding) if isinstance(string, bytes) else string


def strptime(obj):
    """Convert `obj` to a `datetime` object **in UTC without tzinfo**. This
    function should be used within this program as the opposite of
    `datetime.isoformat()` for parsing date times from, e.g. web service
    queries or command line inputs, under the assumption that no time zone
    means UTC.

    If `obj` is string, creates a `datetime` object by parsing it. If `obj`
    is not a date-time object, raises TypeError. Otherwise, uses `obj` as
    `datetime` object. Then, if the datetime object has a tzinfo supplied,
    converts it to UTC and removes the tzinfo attribute. Finally, returns the
    datetime object

    Implementation details: `datetime.strptime`does not keep time zone
    information in the parsed date-time, nor it recognizes 'Z' as 'UTC' (raises
    instead). The library `dateutil`, on the other hand, is too permissive and
    has too many false "positives" (e.g. integers or strings such as  '5-7' are
    successfully parsed into date-time). We choose `dateutil` as the code is
    shorter, cleaner, and a single hack is needed: we simply check, after a
    string `obj` is succesfully parsed into `dtime`, that `obj` contains at
    least the string `dtime.strftime(format='%Y-%m-%d')` (such as e,g,
    '2006-01-31')

    :param obj: `datetime` object or string in ISO format (see examples below)

    :return: a datetime object in UTC, with the tzinfo removed
    :raise: TypeError or ValueError
    :Example. These are all equivalent:
    ```
    strptime("2016-06-01T00:00:00.000000Z")
    strptime("2016-06-01T00.01.00CET")
    strptime("2016-06-01 00:00:00.000000Z")
    strptime("2016-06-01 00:00:00.000000")
    strptime("2016-06-01 00:00:00")
    strptime("2016-06-01 00:00:00Z")
    strptime("2016-06-01")

    This raises ValueError:
    strptime("2016-06-01Z")

    This raises TypeError:
    strptime(45.5)
    ```
    """
    dtime = obj
    if isinstance(obj, string_types):
        try:
            dtime = dateparser.parse(obj, fuzzy=False, fuzzy_with_tokens=False)
            # now, dateperser is quite hacky on purpose, guessing too much.
            # datetime.strptime, on the other hand, does not parse Z as UTC
            # (raises in case) and does not include the timezone in the parsed
            # date. The best (hacky) solution is to assert the bare minimum:
            # that %Y-%m-%d is in dtime:
            assert dtime.strftime('%Y-%m-%d') in obj
        except Exception as exc:
            raise ValueError(str(exc))

    if not isinstance(dtime, datetime):
        raise TypeError('string or datetime required, found %s' %
                        str(type(obj)))

    if dtime.tzinfo is not None:
        # if a time zone is specified, convert to utc and remove the timezone
        dtime = dtime.astimezone(tzutc()).replace(tzinfo=None)

    # the datetime has no timezone provided AND is in UTC:
    return dtime


def secure_dburl(dburl):
    """Return a printable database name by removing passwords, if any

    :param dburl: database path as string in the format:
        dialect+driver://username:password@host:port/database
        For info see:
        http://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls
    """
    return re.sub(r"://(.*?):(.*)@", r"://\1:***@", dburl)


def ascii_decorate(string):
    """Decorate the string with a frame in unicode decoration characters,
    and returns the decorated string

    :param string: a signle- or multi-line string
    """
    if not string:
        return ''

    # defined the frame characters:
    # (topleft, top, topright, left, right, bottomleft, bottom, bottomright):
    # note that top and bottom must be 1-length strings, and
    # topleft+left=bottomleft must have the same length, as well as
    # topright+right+bottomright

    frame = "╔", "═", "╗", "║", "║", "╚", "═", "╝"
    # frame = "###", "#", "###", "###", "###", "###", "#", "###"

    linez = string.splitlines()
    maxlen = max(len(l) for l in linez)
    frmt = "%s {:<%d} %s" % (frame[3], maxlen, frame[4])
    hline_top = frame[0] + frame[1] * (maxlen + 2) + frame[2]
    hline_bottom = frame[-3] + frame[-2] * (maxlen + 2) + frame[-1]

    return "\n".join(chain([hline_top],
                           (frmt.format(l) for l in linez),
                           [hline_bottom]))


class Nop(object):
    """Dummy class (no-op), used to yeild a contextmanager where each method
    is no-op. Used in `get_progressbar`
    """
    # https://stackoverflow.com/a/24946360
    def __init__(self, *a, **kw):
        pass

    @staticmethod
    def __nop(*args, **kw):
        pass

    def __getattr__(self, _):
        return self.__nop


@contextmanager
def get_progressbar(show, **kw):
    """Return a `click.progressbar` if `show` is True, otherwise a No-op
    class, so that we can run programs by simply doing:
    ```
    isterminal = True  # or False for no-op class
    with get_progressbar(isterminal, length=..., ...) as bar:
        # do your stuff ... and then:
        bar.update(num_increments)  # this is no-op if `isterminal` is False
    ```
    """
    if not show or kw.get('length', 1) == 0:
        yield Nop(**kw)
    else:
        # some custom setup if missing:
        # (note that progressbar characters render differently across OSs:
        # after some attempts, I found out the best for mac - which is the
        # default - and Ubuntu):
        is_linux = sys.platform.startswith('linux')
        kw.setdefault('fill_char', "▮" if is_linux else "●")
        kw.setdefault('empty_char', "▯" if is_linux else "○")
        kw.setdefault('bar_template', '%(label)s %(bar)s %(info)s')
        with click_progressbar(**kw) as pbar:
            yield pbar


def urljoin(*urlpath, **query_args):
    """Join urls and appends to it the query string obtained by kwargs
    Note that this function is intended to be simple and fast: No check is made
    about white-spaces in strings, no encoding is done, and if some value of
    `query_args` needs special formatting (e.g., "%1.1f"), that needs to be
    done before calling this function

    :param urlpath: portion of urls which will build the query url Q. For more
        complex url functions see `urlparse` library: this function builds the
        url path via a simple join stripping slashes:
        ```
        '/'.join(url.strip('/') for url in urlpath)
        ```
        So to preserve slashes (e.g., at the beginning) pass "/" or "" as
        arguments (e.g. as first argument to preserve relative paths).
    :query_args: keyword arguments which will build the query string

    :return: a query url built from arguments (string)

    Examples:
    ```
    >>> urljoin("https://abc", start='2015-01-01T00:05:00', mag=5.1, arg=True)
    'https://abc?start=2015-01-01T00:05:00&mag=5.1&arg=True'

    >>> urljoin("http://abc", "data", start='2015-01-01', mag=5.459, arg=True)
    'http://abc/data?start=2015-01-01&mag=5.459&arg=True'

    Note how slashes are handled in urlpath. These two examples give the
    same url path:

    >>> urljoin("http://www.domain", "data")
    'http://www.domain/data?'

    >>> urljoin("http://www.domain/", "/data")
    'http://www.domain/data?'

    # leading and trailing slashes on each element of urlpath are removed:

    >>> urljoin("/www.domain/", "/data")
    'www.domain/data?'

    # so if you want to preserve them, provide an empty argument or a slash:

    >>> urljoin("", "/www.domain/", "/data")
    '/www.domain/data?'

    >>> urljoin("/", "/www.domain/", "/data")
    '/www.domain/data?'
    ```
    """
    # For a discussion, see https://stackoverflow.com/q/1793261
    return "{}?{}".format('/'.join(url.strip('/') for url in urlpath),
                          "&".join("{}={}".format(k, v)
                                   for k, v in query_args.items()))


def open2writetext(file, **kw):
    """Python 2+3 compatible function for writing **text** files with `str`
    types to file (i.e., object of `<type str>` in *both* python2 and 3).
    This function should be used with the csv writer or when we provide an
    input string which is `str` type in both python2 and 3 (e.g., by writing
    a = 'abc'). This function basically returns the python3 `open` function
    where the 'mode' argument is 'wb' in Python2 and 'w' in Python3. In the
    latter case, 'errors' and 'encoding' will be removed from `kw`, if any,
    because not compatible with 'wb' mode.
    Using `io.open(mode='w',..)` in py2 and `open(mode='w', ...)` in py3
    provides compatibility across function **signatures**, but the user must
    provide `unicodes` in python2 and `str` in py3. If this is not the case
    (e.g., we created a string such as a="abc" and we write it to a file, or we
    use the csv module) this function takes care of using the correct 'mode' in
    `open`

    :param file: the file. It is the first argument of the builtin `open`
        function
    :param kw: keyword arguments as for the python3 open function. 'mode' will
        be replaced if present ('wb' for Python2, 'w' for Python 3). An
        optional 'append' argument (True or False) will ad 'a' to the 'mode'
        (i.e., 'wba' for Python2, 'wa' for Python 3). If python2, 'encoding',
        'newline' and 'errors' will be removed as not compatible with the 'wb'
        mode (they raise if present)
    :return: the python3 open function for writing `str` types into text file
    """
    append = kw.pop('append', False)
    if PY2:
        kw.pop('encoding', None)
        kw.pop('errors', None)
        kw.pop('newline', None)
        kw['mode'] = 'wb'
    else:
        kw['mode'] = 'w'
    if append:
        kw['mode'] = kw['mode'].replace('w', 'a')
    return compatible_open(file, **kw)


def yaml_safe_dump(data, stream=None, default_flow_style=False,
                   sort_keys=False, **kwds):
    """Wrapper around `yaml.safe_dump` with convenient defaults:
    try to dump dicts preserving the dict keys order, and handle PyYaml
    versions (< 5.1) where sort_keys was not supported and defaulted to True
    (the passed `sort_keys` value will be ignored in case)

    :return: None (if stream is not None). **If stream is None, returns
        the produced string instead**
    """
    kwds['default_flow_style'] = default_flow_style
    kwds['sort_keys'] = sort_keys
    try:
        return yaml.safe_dump(data, stream, **kwds)
    except TypeError:
        # we might have a PyYaml version < 5.1 where sort_keys was not
        # supported: try to remove the argument. Note however that in that
        # case safe_dump will sort dict keys
        kwds.pop('sort_keys', None)
        return yaml.safe_dump(data, stream, **kwds)
