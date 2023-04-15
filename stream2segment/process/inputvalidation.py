"""
Input validation for the process routine
"""
import os
import inspect

from stream2segment.process.inspectimport import load_source


def valid_filewritable(filepath):
    """Check that the file is writable, i.e. that is a string and its
    directory exists"""
    if not isinstance(filepath, str):
        raise TypeError('string required, found %s' % str(type(filepath)))

    if not os.path.isdir(os.path.dirname(filepath)):
        raise ValueError('cannot write file: parent directory does not exist')

    return filepath


def valid_default_processing_funcname():
    """Return 'main', the default function name for processing, when such
    a name is not given"""
    return 'main'


def valid_pyfile(pyfile):
    """Return the Python module from the given Python file
    An optional double semicolon separates the python module path and the function
    name implemented therein. If missing the function name to search defaults to
    :func:`valid_default_processing_funcname`
    """
    if not isinstance(pyfile, str):
        raise TypeError('Python file must be given as string, not %s' %
                        str(type(pyfile)))

    funcname = valid_default_processing_funcname()
    sep = '::'
    idx = pyfile.rfind(sep)
    if idx >= 0:
        funcname = pyfile[idx + len(sep):]
        pyfile = pyfile[:idx]

    if not pyfile or not os.path.isfile(pyfile):
        raise Exception('File does not exist: "%s"' % pyfile)

    pymoduledict = load_source(pyfile).__dict__

    # check for new style module: SkipSegment instead of ValueError
    if 'SkipSegment' not in pymoduledict:
        raise ValueError('The provided Python module looks outdated.\nYou first need to '
                         'import SkipSegment ("from stream2segment.process import '
                         'SkipSegment") to suppress this warning, and\n'
                         'check your code: to skip a segment, please type '
                         '"raise SkipSegment(.." instead of "raise ValueError(..."')

    if funcname not in pymoduledict:
        raise Exception('Function "%s" not found in %s' %
                        (str(funcname), pyfile))
    return valid_pyfunc(pymoduledict[funcname])


def valid_pyfunc(pyfunc):
    """Checks if the argument is a valid processing Python function by inspecting its
    signature"""
    params = inspect.signature(pyfunc).parameters # dict[str, inspect.Parameter]
    # less than two arguments? then function invalid:
    if len(params) < 2:
        raise ValueError('Python function should have 2 arguments '
                         '`(segment, config)`, %d found' % len(params))
    # more than 2 args? then we need to have them with a default set:
    for pname, param in list(params.items())[2:]:
        if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            # it is not *args or **kwargs, does it have a default?
            if not param.default == param.empty:
                raise ValueError('Python function argument "%s" should have a '
                                 'default, or be removed' % pname)
    # i = 0
    # for i, (pname, pval) in enumerate(inspect.signature(pyfunc).parameters.items(), 1):
    #     if i > 2:
    #         # ops, more than two arguments? maybe is variable length *args **kwargs?
    #         if not pval.kind in (pval.VAR_POSITIONAL, pval.VAR_KEYWORD):
    #             # it is not *args or **kwargs, does it have a default?
    #             if not pval.default == pval.empty:
    #                 raise ValueError('Python function argument "%s" should have a '
    #                                  'default, or be removed' % pname)
    # if i < 2:
    #     raise ValueError('Python function should have 2 arguments '
    #                      '`(segment, config)`, %d found' % i)

    return pyfunc
