"""
Input validation for the process routine
"""
import os
import inspect

from future.utils import string_types

from stream2segment.io import yaml_load
from stream2segment.io.inputvalidation import (validate_param, get_param,
                                               pop_param, valid_between)
from stream2segment.process.inspectimport import load_source


SEGMENT_SELECT_PARAM_NAMES = ('segments_selection', 'segment_select')


def _extract_segments_selection(config):
    """Return the dict in `config` denoting the selection of segment. Validators
    should all call this method so that the valid parameter names are implemented in
    one place and can be easily modified.

    :param config: the config `dict` (e.g. resulting from a YAML config file used for
        processing, or visualization)
    """
    return pop_param(config, SEGMENT_SELECT_PARAM_NAMES, {})[1]


def check_pyfunc_or_pyfile(pyfunc):
    """Check pyfunc or pyfile, returns the tuple (pyfunc, func_path)
    where func_path has the form <module_path>::<function_name>
    """
    pname = 'pyfunc'
    if not callable(pyfunc):
        pname = 'pyfile'
        pyfunc = validate_param(pname, pyfunc, valid_pyfunc)

    return pyfunc, validate_param(pname, pyfunc, _get_func_path)


def _get_func_path(pyfunc):
    return inspect.getsourcefile(pyfunc) + '::' + pyfunc.__name__


def load_p_config(config=None, **param_overrides):
    """Loads a YAML configuration file for processing, returning a tuple of 5 elements:

    config:dict
    segments_selection:dict
    multi_process:Union[bool, int]
    chunksize:Union[None, int]
    writer_options:Union[None, dict]

    :param config: file path to a YAMl file or dict
    :param param_overrides: additional parameter(s) for the YAML `config`. The
        value of existing config parameters will be overwritten, e.g. if
        `config` is {'a': 1} and `param_overrides` is `a=2`, the result is
        {'a': 2}. **Note however that when both parameters are dictionaries, the
        result will be merged**. E.g. if `config` is {'a': {'b': 1, 'c': 1}} and
        `param_overrides` is `a={'c': 2, 'd': 2}`, the result is
        {'a': {'b': 1, 'c': 2, 'd': 2}}
    """
    config = validate_param("config", config or {}, yaml_load, **param_overrides)
    seg_sel = _extract_segments_selection(config)
    multi_process, chunksize = _get_process_advanced_settings(config,
                                                              'advanced_settings')
    writer_options = config.get('advanced_settings', {}).get('writer_options', {})
    return config, seg_sel, multi_process, chunksize, writer_options


def _get_process_advanced_settings(config, adv_settings_key):
    """Return the tuple `(multi_process, chunksize)` validated from
    `config[advanced)_settings_key]`. Raise :class:`BadParam` if any param is invalid
    """
    prefix = adv_settings_key + '.'  # 'advanced_settings.'

    pname, multi_process = get_param(config, prefix + 'multi_process', default=False)
    if multi_process is True:
        # Backward compatibility: if multi_process is True, there
        # was a separate parameter to set the Pool processes: num_processes.
        # (now just set multi_process as int)
        num_processes = get_param(config, prefix + 'num_processes',
                                  default=multi_process)[1]
        # the line below is no-op if num_process was not found (new config):
        multi_process = num_processes
    if multi_process not in (True, False):
        multi_process = validate_param(pname, multi_process,
                                       valid_between, 1, None)

    pname, chunksize = get_param(config, prefix + 'segments_chunksize', None)
    if chunksize is not None:
        chunksize = validate_param(pname, chunksize, valid_between, 1, None)

    return multi_process, chunksize


def valid_filewritable(filepath):
    """Check that the file is writable, i.e. that is a string and its
    directory exists"""
    if not isinstance(filepath, string_types):
        raise TypeError('string required, found %s' % str(type(filepath)))

    if not os.path.isdir(os.path.dirname(filepath)):
        raise ValueError('cannot write file: parent directory does not exist')

    return filepath


def valid_default_processing_funcname():
    """Return 'main', the default function name for processing, when such
    a name is not given"""
    return 'main'


def valid_pyfunc(pyfile):
    """Return the Python module from the given Python file
    An optional double semicolon separates the python module path and the function
    name implemented therein. If missing the function name to search defaults to
    :func:`valid_default_processing_funcname`
    """
    if not isinstance(pyfile, string_types):
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
    return pymoduledict[funcname]