# -*- coding: utf-8 -*-
"""
    Common utilities for the program
"""
# from __future__ import print_function  # , unicode_literals
from builtins import object
import os
import yaml
import re
import time
from datetime import datetime, timedelta
import sys
from collections import defaultdict
import inspect

from contextlib import contextmanager
from sqlalchemy.orm.scoping import scoped_session
from sqlalchemy.engine import create_engine
from sqlalchemy.orm.session import sessionmaker
from click import progressbar as click_progressbar

from stream2segment.io.db.models import Base


def _getmodulename(pyfilepath):
    '''returns a most likely unique module name for a python source file loaded as a module'''
    # We need to supply a model nameFirst let's define a module name. What the name does and why is necessary is not well
    # documented. However, we cannot supply whatever we want for two reasons following
    # python import mechanism (tested in python2):
    # 1. two different `pyfilepath`s must have different `name`s, otherwise when importing the
    # second file the module of the former is actually returned
    # 2. Names should contain dots, as otherwise a `RuntimeWarning: Parent module ... not found`
    # is issued
    # So, make the name equal to the pathname to avoid as much as possible dupes in names, by replacing
    # pathseps with underscores:
    return os.path.abspath(os.path.realpath(pyfilepath)).replace(".", "_").\
        replace(os.path.sep, "_")  # note: os.path.sep returns '/' on mac, os.pathsep returns ':'


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

    def is_mod_function(pymodule, func):
        '''returns True if the python function `func` is a function defined (and not imported) in
        the python module `pymodule`
        '''
        sourcefile = inspect.getsourcefile  # just to make next line fit in max line-width
        return inspect.isfunction(func) and \
            os.path.abspath(sourcefile(pymodule)) == os.path.abspath(sourcefile(func))

    # and also make itervalues a function
    def itervalues(dict_obj):
        return dict_obj.values()
else:
    import imp

    def load_source(pyfilepath):
        """Loads a source python file and returns it"""
        name = _getmodulename(pyfilepath)
        return imp.load_source(name, pyfilepath)

    def is_mod_function(pymodule, func):
        '''returns True if the python function `func` is a function defined (and not imported) in
        the python module `pymodule`
        '''
        return inspect.isfunction(func) and inspect.getmodule(func) == pymodule

    # and also make itervalues a function
    def itervalues(dict_obj):
        return dict_obj.values()


def iterfuncs(pymodule):
    '''Returns an iterator over all functions defined (and not imported)
    in the given python module `pymodule`
    '''
    for func in itervalues(pymodule.__dict__):
        if is_mod_function(pymodule, func):
            yield func


class strconvert(object):

    @staticmethod
    def sql2wild(text):
        """
        :return: a string by replacing in `text` all sql 'like' wildcards ('%', '_') with text
        search equivalent ('*', '?')
        """
        return text.replace("%", "*").replace("_", "?")

    @staticmethod
    def wild2sql(text):
        """
        :return: a string by replacing in `text` all text search wildcards ('*', '?') with
        sql 'like' equivalent ('%', '_')
        """
        return text.replace("*", "%").replace("?", "_")

    @staticmethod
    def wild2re(text):
        """
        :return: a string by replacing in `text` all text search wildcards ('*', '?') with
        regular expression equivalent ('.*', '.')
        """
        return re.escape(text).replace(r"\*", ".*").replace(r"\?", ".")

    @staticmethod
    def sqld2re(text):
        """
        :return: a string by replacing in `text` all sql 'like' wildcards ('%', '_') with
        regular expression equivalent ('.*', '.')
        """
        return re.escape(text).replace(r"\%", ".*").replace(r"\_", ".")


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
        by calling datetime.isoformat() (which returns 'T' as date time separator, and optionally
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
    if isinstance(string, datetime):
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
            return datetime.strptime(string, dtformat)
        except ValueError:  # as exce:
            pass

    raise ValueError("%s: invalid date time" % string)


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


def timedeltaround(tdelta):
    """Rounds a timedelta to seconds"""
    add = 1 if tdelta.microseconds >= 500000 else 0
    return timedelta(days=tdelta.days, seconds=tdelta.seconds+add, microseconds=0)


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
    run programs from code (do not print progress) and from terminal (print progress) by simply
    doing:
    ```
        isterminal = True  # or False for no-op class
        with get_progressbar(isterminal, length=..., ...) as bar:
            # do your stuff in iterators and call
            bar.update(num_increments)  # will update the terminal with a progressbar, or
                                        # do nothing (no-op) if isterminal=True
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


def indent(string, n_chars=3):
    """Indents the given string (or each line of string if multi-line)
    with n_chars spaces.
    :param n_chars: int or string: the number of spaces to use for indentation. If 'tab',
    indents using the tab character"""
    reg = re.compile("^", re.MULTILINE)
    return reg.sub("\t" if n_chars == 'tab' else " " * n_chars, string)


# def printfunc(isterminal=False):
#     """Returns the print function if isterminal is True else a no-op function"""
#     if isterminal:
#         return print
#     else:
#         return lambda *a, **v: None
