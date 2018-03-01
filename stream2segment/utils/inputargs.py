'''
Module with utilities for checking / parsing / setting input arguments from the cli
and for the main functionalities (download, process).

:date: Feb 27, 2018

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
import os
import sys
import re
from datetime import datetime, timedelta

from future.utils import string_types

from stream2segment.utils.resources import yaml_load, get_ttable_fpath, yaml_load_doc, \
    get_templates_fpath
from stream2segment.utils import get_session, strptime, load_source, iterfuncs
from stream2segment.traveltimes.ttloader import TTTable


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
        

class UnknownArg(BadArgument):
    
    def __init__(self, param_name):
        '''A BadArgument notifying an unknown argument'''
        super(UnknownArg, self).__init__(param_name, '', "Unknown argument")


_DEFAULTDOC = yaml_load_doc(get_templates_fpath("download.yaml"))
_ORIGCONFIG = yaml_load(get_templates_fpath("download.yaml"))


def get_doc(download_param):
    return _DEFAULTDOC[download_param]


def arg(pname, *optional_pnames, ifmissing=None, newname=None, newvalue=lambda v: v):
    def wrapped_f(dic):
        try:
            keys_in =[]
            for p in set([pname] + list(optional_pnames)):
                if p in dic:
                    keys_in.append(p)
            if len(keys_in) > 1:
                raise ConflictingArgs(*keys_in)
            pname2get = pname if not keys_in else keys_in[0]
            if pname2get not in dic and ifmissing is not None:
                value = ifmissing
            else:
                value = dic.pop(pname2get)
            
        except KeyError as _:
            raise MissingArg(pname)
        try:
            newval = newvalue(value) if hasattr(newvalue, '__call__') else newvalue
            dic[pname if newname is None else newname] = newval
            return newval
        except TypeError as terr:
            raise BadTypeArg(pname, terr)
        except Exception as exc:
            raise BadValueArg(pname, exc)
    wrapped_f.argnames = [pname] + list(optional_pnames)
    return wrapped_f


def typesmatch(value, other_value):
    if not issubclass(value.__class__, other_value.__class__):
        raise TypeError("%s expected, found %s" % (str(type(value)), str(type(other_value))))
    return value


def nslc_param_value_aslist(value):
    '''Returns a nslc (network/station/location/channel) parameter value converted as list.
    This method cleans-up and checks `value` splitting each of its string elements
    with the comma "," and aggregating all the string chunks into a single list, after performing
    some sanity check. The resulting list is also sorted alphabetically
    (for unit testing and readibility).
    Raises ValueError in case some sanity checks fail (e.g., conflicts, syntax errors)

    Examples:
    
    nslc_param_value_aslist 
    arguments (any means:
    any value in [0,1,2,3])   Result (with comment)            
    ========================= =================================================================
    (['A','D','C','B'])  ['A', 'B', 'C', 'D']  # note result is sorted
    ('B,C,D,A')          ['A', 'B', 'C', 'D']  # same as above
    ('A*, B??, C*')      ['A*', 'B??', 'C*']  # fdsn wildcards accepted
    ('!A*, B??, C*')     ['!A*', 'B??', 'C*']  # we support negations: !A* means "not A*"
    (' A, B ')           ['A', 'B']  # leading and trailing spaces ignored
    ('*')                []  # if any chunk is '*', then [] (=match all) is returned
    ([])                 []  # same as above
    ('  ')               ['']  # this means: match the empty string (strip the string)
    ("")                 [""]  # same as above
    ("!")                ['!']  # match any non empty string
    ("!*")               this raises (you cannot specify "discard all")
    ("!H*, H*")          this raises (it's a paradox)
    (" A B,  CD")        this raises ('A B' invalid: only leading and trailing spaces allowed)
    

    :param value: string or iterable of strings: (iterable in this context means python iterable
        EXCEPT strings). If string, the argument will be converted
        to the list [value] to make it iterable before processing it
    :param index: integer in [0,1,2,3]: the index of the parameter associated to `valie`,
        where 0 means 'network', 1 'station', 2 'location' and 3 'channel'.
        It is used for converting '--' to empty
        strings in case of location(s), and to output the correct parameter name in the body of
        Exceptions, if any has occurred or been raised
    ''' 
    try:
        strings = set()
    
        # we assume, when arg is not list, that arg is str in both python2 and python3, i.e.
        # it is NOT bytes in python2. The line below checks if is an iterable first:
        # in python2, it is sufficient to say it's not a string
        # in python3, we need to check that is no str also
        if not hasattr(value, "__iter__") or isinstance(value, str):
            # it’s an iterable not a string
            value = [value]
            
        for string in value:
            splitted = string.split(",")
            for s in splitted:
                s = s.strip()
                if ' ' in s:
                    raise Exception("invalid space char(s): '%s'" % s)
                # if i == 3 (location) convert '--' to '':
                strings.add(s)

        # some checks:
        if "!*" in strings:  # discard everything is not valid
            raise ValueError("'!*' (=discard all) invalid")
        elif "*" in  strings:  # accept everything and X: X is redundant
            strings = set(_ for _ in strings if _[0:1] == '!')
        else: 
            for string in strings:  # accept A end discard A is not valid
                opposite = "!%s" % string
                if opposite in strings:
                    raise Exception("conflicting values: '%s' and '%s'" % (string, opposite))
        
        return sorted(strings)

    except Exception as exc:
        raise ValueError(str(exc))


def load_config(config, **param_overrides):
    try:
        return yaml_load(config, **param_overrides)
    except Exception as exc:
        raise BadValueArg('config', exc)


def extract_dburl_if_yamlpath(value):
    """
    For all non-download click Options, returns the database path from 'value':
    'value' can be a file (in that case is assumed to be a yaml file with the
    'dburl' key in it) or the database path otherwise
    """
    return yaml_load(value)['dburl'] \
        if (value and isinstance(value, string_types) and os.path.isfile(value)) else value


def keyval_list_to_dict(value):
    """parses optional event query args (when the 'd' command is issued) into a dict"""
    # use iter to make a dict from a list whose even indices = keys, odd ones = values
    # https://stackoverflow.com/questions/4576115/convert-a-list-to-a-dictionary-in-python
    itr = iter(value)
    return dict(zip(itr, itr))


def create_session(dburl):
    if not isinstance(dburl, string_types):
        raise TypeError('string required, %s found' % str(type(dburl)))
    return get_session(dburl, scoped=False)


def load_tt_table(file_or_name):
    if not isinstance(file_or_name, string_types):
        raise TypeError('string required, not %s' % str(type(file_or_name)))
    filepath = get_ttable_fpath(file_or_name)
    if not os.path.isfile(filepath):
        filepath = file_or_name
    if not os.path.isfile(filepath):
        raise Exception('file or builtin model name not found')
    return TTTable(filepath)


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



def valid_fdsn(url):
    if url.lower() in ('eida', 'iris'):
        return url
    reg = re.compile("^.*/fdsnws/(?P<service>[^/]+)/(?P<majorversion>\\d+)/query$")
    match = reg.match(url)
    if not match or not match.group('service') or not match.group('majorversion'):
        raise ValueError("No FDSN url: <site>/fdsnws/<service>/<majorversion>/")
    if match.group('service') not in ('dataselect', 'station', 'event'):
        raise ValueError("No FDSN url: service not in 'station', 'event' or 'dataselect'")
    return url



def checkdownloadinput(dic):
    '''checks download arguments'''

    remainingkeys = set(dic.keys())
    args = (arg('dburl', newname='session', newvalue=create_session),
            arg('traveltimes_model', newname='tt_table', newvalue=load_tt_table),
            arg('start', newvalue=valid_date),
            arg('end', newvalue=valid_date),
            arg('eventws', newvalue=valid_fdsn),
            arg('dataws', newvalue=valid_fdsn),
            arg('networks', 'net', 'network', ifmissing=[], newvalue=nslc_param_value_aslist),
            arg('stations', 'sta', 'station', ifmissing=[], newvalue=nslc_param_value_aslist),
            arg('locations', 'loc', 'location', ifmissing=[], newvalue=nslc_param_value_aslist),
            arg('channels', 'cha', 'channel', ifmissing=[], newvalue=nslc_param_value_aslist))

    for func in args:
        func(dic)
        remainingkeys -= set(func.argnames)
    
    # Note: dic here is passed with the arguments 9keys) not implemented in the decorator
    # above. We perform a simple check: the argument must be present in the default config
    # shipped with this package and the types must match. Note that you should call arg
    # because 1. it raises the appropriate BadArgument exception which is caught and print in cli,
    # and 2. it modifies the dic value with the rpoper value (which in this case is the same)
    orig_config = _ORIGCONFIG

    for key in remainingkeys:  # iterate over list(keys) cause we will modify the dict inplace
        if key not in orig_config:
            raise UnknownArg(key)
        arg(key, newvalue=lambda value: typesmatch(value, orig_config[key]))(dic)
    
    # now process the arguments in checker
    return dic['session'], dic['tt_table']


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

    if not os.path.isfile(pyfile):
        raise Exception('file does not exist')

    pymoduledict = load_source(pyfile).__dict__
    if funcname not in pymoduledict:
        raise Exception("function '%s' not found in %s" % (str(funcname), pyfile))
    return pymoduledict[funcname]
    

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


def checkprocessinput(dic):
    '''checks download arguments'''
    session = arg('dburl', newname='session', newvalue=create_session)(dic)
    arg('funcname', newvalue=get_funcname)(dic)
    arg('config', newname='config_dict', newvalue=lambda c: {} if c is None else load_config(c))(dic)
    funcname = dic.pop('funcname')
    arg('pyfile', newvalue=lambda val: load_pyfunc(val, funcname))(dic)
    # nothing more to process
    return session



