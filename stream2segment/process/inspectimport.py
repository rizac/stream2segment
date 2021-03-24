"""
inspect+importlib functions for stream2segment

March 22, 2020

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""

from future.utils import itervalues

import os
import sys
import inspect


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


