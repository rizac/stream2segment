# -*- coding: utf-8 -*-
"""
Common utilities for the whole program

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
# make open py2-3 compatible. Call 'from stream2segment.utils import open'
# (http://python-future.org/imports.html#explicit-imports):

from future.utils import itervalues

import os
import sys
# import time
from itertools import chain
import inspect
from contextlib import contextmanager

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
    """Dummy class (no-op), used to yield a contextmanager where each method
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


