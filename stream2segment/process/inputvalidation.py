"""
Input validation for the process routine
"""
import os

from future.utils import string_types

from stream2segment.io import yaml_load
from stream2segment.io.inputvalidation import (validate_param, get_param,
                                               pop_param, valid_between)
# import get_session as 'valid_session' for compatibility with the download package:
from stream2segment.process.db import get_session
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


# def load_config_for_process(pyfile, funcname=None, config=None,
#                             outfile=None, **param_overrides):
#     """Check process arguments. Returns the tuple session, pyfunc, config_dict,
#     where session is the dql alchemy session from `dburl`, `funcname` is the
#     Python function loaded from `pyfile`, and config_dict is the dict loaded
#     from `config` which must denote a path to a yaml file, or None (config_dict
#     will be empty in this latter case)
#     """
#     funcname = validate_param("funcname", funcname, valid_funcname)
#     pyfunc = validate_param("pyfile", pyfile, valid_pyfunc, funcname)
#     config = validate_param("config", config or {}, yaml_load, **param_overrides)
#     if outfile is not None:
#         validate_param('outfile', outfile, valid_filewritable)
#         # (ignore return value of filewritable: it's outfile, we already have it)
#     seg_sel = _extract_segments_selection(config)
#
#     multi_process, chunksize = _get_process_advanced_settings(config,
#                                                               'advanced_settings')
#
#     return pyfunc, config, seg_sel, multi_process, chunksize, writer_options


def load_pyfunc_for_process(pyfile, funcname=None):
    """Loads the Python function"""
    funcname = validate_param("funcname", funcname, valid_funcname)
    return validate_param("pyfile", pyfile, valid_pyfunc, funcname)


def load_config(config=None, **param_overrides):
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


def valid_funcname(funcname=None):
    """Return the Python module from the given python file"""
    if funcname is None:
        funcname = valid_default_processing_funcname()

    if not isinstance(funcname, string_types):
        raise TypeError('string required, not %s' % str(type(funcname)))

    return funcname


def valid_default_processing_funcname():
    """Return 'main', the default function name for processing, when such
    a name is not given"""
    return 'main'


def valid_pyfunc(pyfile, funcname):
    """Return the Python module from the given python file"""
    if not isinstance(pyfile, string_types):
        raise TypeError('string required, not %s' % str(type(pyfile)))

    if not os.path.isfile(pyfile):
        raise Exception('file does not exist')

    pymoduledict = load_source(pyfile).__dict__

    # check for new style module: SkipSegment instead of ValueError
    if 'SkipSegment' not in pymoduledict:
        raise ValueError('The provided Python module looks outdated.\nYou first need to '
                         'import SkipSegment ("from stream2segment.process import '
                         'SkipSegment") to suppress this warning, and\n'
                         'check your code: to skip a segment, please type '
                         '"raise SkipSegment(.." instead of "raise ValueError(..."')

    if funcname not in pymoduledict:
        raise Exception('function "%s" not found in %s' %
                        (str(funcname), pyfile))
    return pymoduledict[funcname]