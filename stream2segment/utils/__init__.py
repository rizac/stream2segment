# -*- coding: utf-8 -*-
"""
Common utilities for the whole program

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import object

from future.utils import string_types

import os
import yaml
import re
import time
from datetime import datetime, timedelta
from dateutil import parser as dateparser
from dateutil.tz import tzutc
import sys
from collections import defaultdict
import inspect
from contextlib import contextmanager
from future.utils import itervalues

from sqlalchemy.orm.scoping import scoped_session
from sqlalchemy.engine import create_engine
from sqlalchemy.orm.session import sessionmaker
from click import progressbar as click_progressbar

from stream2segment.io.db.models import Base


def _getmodulename(pyfilepath):
    '''returns a most likely unique module name for a python source file loaded as a module'''
    # In both python2 and 3, the function importing a module from file needs a file path and 
    # a 'name' argument. It's not clear why it's necessary and e.g., it does not default
    # to the filepath 's name. However, keep in mind that:
    # 1. The name must be UNIQUE: otherwise when importing the second file the module of the
    # former is actually returned
    # 2. Names should NOT contain dots, as otherwise a `RuntimeWarning: Parent module ... not
    # found` is issued.
    # To achieve 1. and 2. this function returns a string that is the full absolute path of
    # `pyfilepath`, replacing file-path separators and dots with '_pathsep_' and '_dot_', repsectiyely
    return os.path.abspath(os.path.realpath(pyfilepath)).replace(".", "_dot_").\
        replace(os.path.sep, "_pathsep_")
    # note above: os.path.sep returns '/' on mac, os.pathsep returns ':'


# python 2 and 3 compatible code:
if sys.version_info[0] > 2:  # python 3+ (if py4 will not be compliant we'll fix that when needed)
    import importlib.util  # @UnresolvedImport

    def load_source(pyfilepath):
        """Loads a source python file and returns it"""
        name = _getmodulename(pyfilepath)
        spec = importlib.util.spec_from_file_location(name, pyfilepath)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        return foo

    def is_mod_function(pymodule, func, include_classes=False):
        '''returns True if the python function `func` is a function (or class if `include_classes`
        is True) defined (and not imported) in the python module `pymodule`
        '''
        is_candidate = inspect.isfunction(func) or (include_classes and inspect.isclass(func))
        # check that the source file is the module (i.e. not imported). NOTE that
        # getsourcefile might raise (not the case for functions or classes)
        return is_candidate and os.path.abspath(inspect.getsourcefile(pymodule)) == \
            os.path.abspath(inspect.getsourcefile(func))

else:
    import imp

    def load_source(pyfilepath):
        """Loads a source python file and returns it"""
        name = _getmodulename(pyfilepath)
        return imp.load_source(name, pyfilepath)

    def is_mod_function(pymodule, func, include_classes=False):
        '''returns True if the python function `func` is a function (or class if `include_classes`
        is True) defined (and not imported) in the python module `pymodule`
        '''
        is_candidate = inspect.isfunction(func) or (include_classes and inspect.isclass(func))
        # check that the source file is the module (i.e. not imported). NOTE that
        # getsourcefile might raise (not the case for functions or classes)
        return is_candidate and inspect.getmodule(func) == pymodule


def iterfuncs(pymodule, include_classes=False):
    '''Returns an iterator over all functions (or classes if `include_classes`
    is True) defined (and not imported) in the given python module `pymodule`
    '''
    for func in itervalues(pymodule.__dict__):
        if is_mod_function(pymodule, func, include_classes):
            yield func


class strconvert(object):
    '''String conversion utilities from sql-LIKE operator's wildcards, Filesystem's wildcards, and
    regular expressions'''
    @staticmethod
    def sql2wild(text):
        """
        Returns a new string from `text` by replacing all sql-LIKE-operator's wildcard characters
        ('sql') with their filesystem's counterparts ('wild'):

        === ==== === ===============================
        sql wild re  meaning
        === ==== === ===============================
        %   *    .*  matches zero or more characters
        _   ?    .   matches exactly one character
        === ==== === ===============================

        :return: string. Note that this function performs a simple replacement: wildcard
            characters in the input string will result in a string that is not the perfect
            translation of the input
        """
        return text.replace("%", "*").replace("_", "?")

    @staticmethod
    def wild2sql(text):
        """
        Returns a new string from `text` by replacing all filesystem's wildcard characters ('wild')
        with their sql-LIKE-operator's counterparts ('sql'):

        === ==== === ===============================
        sql wild re  meaning
        === ==== === ===============================
        %   *    .*  matches zero or more characters
        _   ?    .   matches exactly one character
        === ==== === ===============================

        :return: string. Note that this function performs a simple replacement: sql special
            characters in the input string will result in a string that is not the perfect
            translation of the input
        """
        return text.replace("*", "%").replace("?", "_")

    @staticmethod
    def wild2re(text):
        """
        Returns a new string from `text` by replacing all filesystem's wildcard characters ('wild')
        with their regular expression's counterparts ('re'):

        === ==== === ===============================
        sql wild re  meaning
        === ==== === ===============================
        %   *    .*  matches zero or more characters
        _   ?    .   matches exactly one character
        === ==== === ===============================

        :return: string. Note that this function performs a simple replacement: regexp special
            characters in the input string will result in a string that is not the perfect
            translation of the input
        """
        return re.escape(text).replace(r"\*", ".*").replace(r"\?", ".")

    @staticmethod
    def sql2re(text):
        """
        Returns a new string from `text` by replacing all sql-LIKE-operator's wildcard characters
        ('sql') with their regular expression's counterparts ('re'):

        === ==== === ===============================
        sql wild re  meaning
        === ==== === ===============================
        %   *    .*  matches zero or more characters
        _   ?    .   matches exactly one character
        === ==== === ===============================

        :return: string. Note that this function performs a simple replacement: regexp special
            characters in the input string will result in a string that is not the perfect
            translation of the input
        """
        return re.escape(text).replace(r"\%", ".*").replace("_", ".")


def tounicode(string, decoding='utf-8'):
    """
        Converts string to text (unicode in python2, str in python3). Function python 2-3
        compatible. If string is already a text, returns it
        :param string: a `str`, 'bytes' or (in py2) 'unicode' object.
        :param decoding: the decoding used if `string` has to be converted to text.
            Defaults to 'utf-8' when missing
        :return: the text (`str` in python3, `unicode` string in python2) representing `string`
    """
    # Curiously, future.utils has no such a simple method. So instead of checking
    # when string is text, let's check when it is NOT, i.e. when it's instance of bytes
    # (str in py2 is instanceof bytes):
    return string.decode(decoding) if isinstance(string, bytes) else string


def strptime(obj):
    """
        Converts `obj` to a `datetime` object in UTC without tzinfo.
        
        If `obj` is string, creates a `datetime` object by parsing it with `dateutil`, otherwise
        uses `obj` as `datetime` object. Then, if the datetime object has a tzinfo supplied,
        converts it to UTC and removes the tzinfo attribute. Finally, returns the datetime object
        
        :param obj: `datetime` object or string.
        
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
            dtime = dateparser.parse(obj)
        except Exception as exc:
            raise ValueError(str(exc))

    if not isinstance(dtime, datetime):
        raise TypeError('string or datetime required, found %s' % str(type(obj)))

    if dtime.tzinfo is not None:
        # if a time zone is specified, convert to utc and remove the timezone
        dtime = dtime.astimezone(tzutc()).replace(tzinfo=None)
    
    # the datetime has no timezone provided AND is in UTC:
    return dtime


def get_session(dbpath, scoped=False):  # , enable_fk_if_sqlite=True):
    """
    Create an sql alchemy session for IO db operations
    :param dbpath: the path to the database, e.g. sqlite:///path_to_my_dbase.sqlite
    :param scoped: boolean (False by default) if the session must be scoped session
    """
    # init the session:
    engine = create_engine(dbpath)
    Base.metadata.create_all(engine)  # @UndefinedVariable

    # enable fkeys if sqlite. This can be added also as event listener as outlined here:
    # http://stackoverflow.com/questions/13712381/how-to-turn-on-pragma-foreign-keys-on-in-sqlalchemy-migration-script-or-conf
    # NOT implemented YET. See models.py

    if not scoped:
        # create a configured "Session" class
        session = sessionmaker(bind=engine)
        # create a Session
        return session()
    # return session
    else:
        session_factory = sessionmaker(bind=engine)
        return scoped_session(session_factory)


def secure_dburl(dburl):
    """Returns a printable database name by removing passwords, if any
    :param dbpath: string, in the format:
    dialect+driver://username:password@host:port/database
    For infor see:
    http://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls
    """
    return re.sub(r"://(.*?):(.*)@", r"://\1:***@", dburl)


# https://stackoverflow.com/questions/24946321/how-do-i-write-a-no-op-or-dummy-class-in-python
class Nop(object):
    def __init__(self, *a, **kw):
        pass

    @staticmethod
    def __nop(*args, **kw):
        pass

    def __getattr__(self, _):
        return self.__nop


@contextmanager
def get_progressbar(show, **kw):
    """Returns a `click.progressbar` if `show` is True, otherwise a No-op class, so that we can
    run programs by simply doing:
    ```
        isterminal = True  # or False for no-op class
        with get_progressbar(isterminal, length=..., ...) as bar:
            # do your stuff in iterators and call
            bar.update(num_increments)  # will update the terminal with a progressbar, or
                                        # do nothing (no-op) if isterminal=False
    ```
    """
    if not show or kw.get('length', 1) == 0:
        yield Nop(**kw)
    else:
        # some custom setup if missing:
        if 'fill_char' not in kw:
            kw['fill_char'] = "●"
        if 'empty_char' not in kw:
            kw['empty_char'] = '○'
        if 'bar_template' not in kw:
            kw['bar_template'] = '%(label)s %(bar)s %(info)s'
        with click_progressbar(**kw) as bar:
            yield bar


def urljoin(*urlpath, **query_args):
    """Joins urls and appends to it the query string obtained by kwargs
    Note that this function is intended to be simple and fast: No check is made about white-spaces
    in strings, no encoding is done, and if some value of `query_args` needs special formatting
    (e.g., "%1.1f"), that needs to be done before calling this function
    :param urls: portion of urls which will build the query url Q. For more complex url functions
    see `urlparse` library: this function builds the url path via a simple join stripping slashes:
    ```'/'.join(url.strip('/') for url in urlpath)```
    So to preserve slashes (e.g., at the beginning) pass "/" or "" as arguments (e.g. as first
    argument to preserve relative paths).
    :query_args: keyword arguments which will build the query string
    :return: a query url built from arguments

    :Example:
    ```
    >>> urljoin("http://www.domain", start='2015-01-01T00:05:00', mag=5.455559, arg=True)
    'http://www.domain?start=2015-01-01T00:05:00&mag=5.455559&arg=True'

    >>> urljoin("http://www.domain", "data", start='2015-01-01T00:05:00', mag=5.455559, arg=True)
    'http://www.domain/data?start=2015-01-01T00:05:00&mag=5.455559&arg=True'

    # Note how slashes are handled in urlpath. These two examples give the same url path:

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
    # http://stackoverflow.com/questions/1793261/how-to-join-components-of-a-path-when-you-are-constructing-a-url-in-python
    return "{}?{}".format('/'.join(url.strip('/') for url in urlpath),
                          "&".join("{}={}".format(k, v) for k, v in query_args.items()))
