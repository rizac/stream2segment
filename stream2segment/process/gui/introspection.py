"""inspect+importlib functions for stream2segment"""
# March 22, 2020

import os
import importlib.util
from inspect import getsourcefile, isfunction, isclass
from types import ModuleType


def load_source(py_file_path: str) -> ModuleType:
    """Load and return a Python module from file"""
    spec = importlib.util.spec_from_file_location(
        os.path.abspath(py_file_path), py_file_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scan_module(
    py_module: ModuleType, functions: bool, classes: bool, exclude_imported=True
):
    """
    Return an iterator over all functions (or classes if `include_classes`
    is True) defined (and not imported) in the given python module `pymodule`
    """
    module_file = os.path.abspath(getsourcefile(py_module))

    for member in py_module.__dict__.values():
        if (functions and isfunction(member)) or (classes and isclass(member)):
            member_file = os.path.abspath(getsourcefile(member))
            if not exclude_imported or member_file == module_file:
                yield member


