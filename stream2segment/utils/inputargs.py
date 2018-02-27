'''
Module with utilities for checking / parsing / setting input arguments.
Functions decorated with @argchecker provide a way, when called with a dict as first argument,
to raise explicative exceptions whise message format is similar to click, and which are
caught from the cli to print the message and exit instead of raising  

:date: Feb 27, 2018

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
from datetime import datetime, timedelta
import os
from stream2segment.utils.resources import yaml_load, get_ttable_fpath
from future.utils import string_types
from stream2segment.utils import get_session, strptime, load_source
from stream2segment.traveltimes.ttloader import TTTable
from stream2segment.download.utils import nslc_param_value_aslist


class BadArgument(Exception):
    '''An exception whose string method is similar to click formatted output. It
    supports sub-classes for most common argument errors
    '''
    def __init__(self, param_name, error, msg_preamble=''):
        '''init method
        
        The formatted output, depending on the truthy value of the arguments will be:

        "%(msg_preamble) %(param_name): %(error)"
        "%(param_name): %(error)"
        "%(msg_preamble) %(param_name)"
        "%(param_name)"
        
        :param param_name: the parameter name (string)
        :param error: the original exception, or a string message
        :param msg_preamble: the optional message preamble, as string
        '''
        super(BadArgument, self).__init__(str(error))
        self.msg_preamble = msg_preamble
        self.param_name = str(param_name) if param_name else None

    @property
    def message(self):
        msg = '%s' if not self.msg_preamble else self.msg_preamble.strip() + " %s"
        err_msg = self.args[0]  # in ValueError, is the error_msg passed in the constructor
        pname = ('"%s"' % self.param_name) if self.param_name else \
            'unknown parameter (check input arguments)'
        ret = (msg % pname) + (": " + err_msg if err_msg else '')
        return ret[0:1].upper() + ret[1:]

    def __str__(self):
        ''''''
        return "Error: %s" % self.message


class MissingArg(BadArgument):
    
    def __init__(self, param_name):
        '''A BadArgument notifying a missing value of some argument'''
        super(MissingArg, self).__init__(param_name, '', "Missing value for")

   
class BadValueArg(BadArgument):
    
    def __init__(self, param_name, error):
        '''A BadArgument notifying a bad value of some argument'''
        super(BadValueArg, self).__init__(param_name, error, "Invalid value for")


class BadTypeArg(BadArgument):
    
    def __init__(self, param_name, error):
        '''A BadArgument notifying a bad type of some argument'''
        super(BadTypeArg, self).__init__(param_name, error, "Invalid type for")


class ConflictingArgs(BadArgument):
    
    def __init__(self, *param_names):
        '''A BadArgument notifying conflicting argument names'''
        # little hack: build a string wiothout first and last quote (will be added in super-class)
        param_name = " / ".join('"%s"' % p for p in param_names)[1:-1]
        super(ConflictingArgs, self).__init__(param_name, '', "Conflicting names")


def argchecker(pname, *optional_pnames, ifdict=None, default=None):
    '''decorator that makes a function f(value, *args, **kwargs) be called also with `value` as
    dict. If dict, several options are given to manioulate the dict and raise explicative
    Exceptions in case `pname` is not found. When not called as dict, the function is run as-it-is.
    
    Example:
    
    @argchecker('paramname'
    
     
    In the latter case value[pname] will be first got and passed to `f`.
    Moreover, the decorated function raises always a BadArgument exception:
    If dict is provided and the key is not found, a MissingArg exception is raised.
    From within `f` any TypeError raised will be re-raised with a BadTypeArg exception. In any
    other case, a BadValueArg exception is raised''' 
    def wrap(f):
        def wrapped_f(value_or_dict, *args, **kwargs):
            try:
                isdict = isinstance(value_or_dict, dict)
                if isdict:
                    keys_in =[]
                    for p in set([pname] + list(optional_pnames)):
                        if p in value_or_dict:
                            keys_in.append(p)
                    if len(keys_in) > 1:
                        raise ConflictingArgs(*keys_in)
                    pname2get = pname if not keys_in else keys_in[0]
                    if pname2get not in value_or_dict and default is not None:
                        value_or_dict[pname2get] = default
                    if ifdict.lower() in ('pop', 'set', 'replace'):
                        value = value_or_dict.pop(pname2get)
                    else:
                        value = value_or_dict[pname2get]
                else:
                    value = value_or_dict
            except KeyError as _:
                raise MissingArg(pname)
            try:
                newvalue = f(value, *args, **kwargs)
                if isdict: 
                    if ifdict.lower() in ('set', 'replace'):
                        value_or_dict[pname] = newvalue
                return newvalue
            except TypeError as terr:
                raise BadTypeArg(pname, terr)
            except Exception as exc:
                raise BadValueArg(pname, exc)
        return wrapped_f
    return wrap


@argchecker('configfile')
def load_configfile(configfile, **param_overrides):
    if not os.path.isfile(configfile):
        raise Exception('file does not exist')
    
    return yaml_load(configfile, **param_overrides)


@argchecker('dburl')
def extract_dburl(db_url):
    return db_url


def extract_dburl_if_yamlpath(value):
    """
    For all non-download click Options, returns the database path from 'value':
    'value' can be a file (in that case is assumed to be a yaml file with the
    'dburl' key in it) or the database path otherwise
    """
    return extract_dburl(yaml_load(value)) \
        if (value and isinstance(value, string_types) and os.path.isfile(value)) else value


def keyval_list_to_dict(value):
    """parses optional event query args (when the 'd' command is issued) into a dict"""
    # use iter to make a dict from a list whose even indices = keys, odd ones = values
    # https://stackoverflow.com/questions/4576115/convert-a-list-to-a-dictionary-in-python
    itr = iter(value)
    return dict(zip(itr, itr))

@argchecker('dburl', ifdict='pop')
def create_session(dburl):
    if not isinstance(dburl, string_types):
        raise TypeError('string required, %s found' % str(type(dburl)))
    return get_session(dburl, scoped=False)


@argchecker('traveltimes_model', ifdict='pop')
def load_tt_table(file_or_name):
    if not isinstance(file_or_name, string_types):
        raise TypeError('string required, not %s' % str(type(file_or_name)))
    filepath = get_ttable_fpath(file_or_name)
    if not os.path.isfile(filepath):
        filepath = file_or_name
    if not os.path.isfile(filepath):
        raise Exception('file or builtin model name not found')
    return TTTable(filepath)


@argchecker('start', ifdict='replace')
def adjust_start(start):
    return valid_date(start)


@argchecker('end', ifdict='replace')
def adjust_end(end):
    return valid_date(end)


def valid_date(obj):
    try:
        return strptime(obj)  # if obj is datetime, returns obj
    except ValueError as _:
        try:
            days = int(obj)
            now = datetime.utcnow()
            endt = datetime(now.year, now.month, now.day, 0, 0, 0, 0)
            return endt - timedelta(days=days)
        except Exception:
            pass
    raise ValueError("date-time or an integer required")


@argchecker('networks', 'net', 'network', ifdict='replace', default=[])
def adjust_net(networks):
    '''
    '''
    return nslc_param_value_aslist(0, networks)


@argchecker('stations', 'sta', 'station', ifdict='replace', default=[])
def adjust_sta(stations):
    '''
    '''
    return nslc_param_value_aslist(1, stations)


@argchecker('locations', 'loc', 'location', ifdict='replace', default=[])
def adjust_loc(locations):
    '''
    '''
    return nslc_param_value_aslist(2, locations)


@argchecker('channels', 'cha', 'channel', ifdict='replace', default=[])
def adjust_cha(channels):
    '''
    '''
    return nslc_param_value_aslist(3, channels)


@argchecker('pyfile')
def load_pyfunc(pyfile, funcname):
    '''Returns the python module from the given python file'''
    if not isinstance(pyfile, string_types):
        raise TypeError('string required, not %s' % str(type(pyfile)))

#     reg = re.compile("^(.*):([a-zA-Z_][a-zA-Z_0-9]*)$")
#     m = reg.match(pyfile)
#     if m and m.groups():
#         pyfile = m.groups()[0]
#         funcname = m.groups()[1]
#     elif funcname is None:
#         funcname = default_processing_funcname()

    try:
        if not os.path.isfile(pyfile):
            raise Exception('file does not exist')
    
        return load_source(pyfile).__dict__[funcname]
    except KeyError as _:
        raise Exception("function '%s' not found in %s" % (str(funcname), pyfile))


@argchecker('funcname')
def get_funcname(funcname=None):
    '''Returns the python module from the given python file'''
    if funcname is None:
        funcname = default_processing_funcname()
    
    if not isinstance(funcname, string_types):
        raise TypeError('string required, not %s' % str(type(funcname)))

    return funcname


def default_processing_funcname():
    '''returns 'main', the default function name for processing, when such a name is not given'''
    return 'main'
