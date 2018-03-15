'''
Module with utilities for checking / parsing / setting input arguments from the cli
(download, process).

:date: Feb 27, 2018

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
import os
import sys
import re
from datetime import datetime, timedelta

from future.utils import string_types

from stream2segment.utils.resources import yaml_load, get_ttable_fpath, \
    get_templates_fpath
from stream2segment.utils import get_session, strptime, load_source
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
        param_name = self.formatnames(*param_names)
        super(ConflictingArgs, self).__init__(param_name, '', "Conflicting names")

    @staticmethod
    def formatnames(*param_names):
        # little hack: build a string wiothout first and last quote (will be added in super-class)
        return " / ".join('"%s"' % p for p in param_names)[1:-1]


class UnknownArg(BadArgument):

    def __init__(self, param_name):
        '''A BadArgument notifying an unknown argument'''
        super(UnknownArg, self).__init__(param_name, '', "no such option")


class S2SArgument(object):

    def __init__(self, name, *optional_names):
        '''Creates a new stream2segment argument, which represents an application input
        argument (sort of `click.Option`). An object of this class allows retrieving values from a
        dict (methods `popfrom` and `getfrom`) with validation options via a 'callback' argument.
        `parse`, `popfrom` and `getfrom` all raise BadArgument
        exception, which are useful because can print a meaningful message with the
        parameter name. Moreover, program cli commands like 'download' and 'process'
        capture BadArguments in order to print to the console the message without the
        the stack trace.

            :param name: the argument name
            :param optional_names: the argument optional names
        '''
        self._names = set([name] + list(optional_names))
        self._name = name

    def _get(self, dic, pop=False, ifmissing=None):
        try:
            keys_in =[p for p in self.names if p in dic]
            if len(keys_in) > 1:
                raise ConflictingArgs(*keys_in)
            elif not keys_in:
                if ifmissing is not None:
                    return self.name, ifmissing
                raise KeyError()
            name = keys_in[0]
            return name, dic.pop(name) if pop else dic[name]

        except KeyError as _:
            raise MissingArg(ConflictingArgs.formatnames(self.names))

    def getfrom(self, dic, default=None, callback=None, **callback_kwargs):
        '''Gets and returns the value mapped to this argument from `dic`.
        Raises BadArgument exceptions (e.g., no key found in `dic` among `self.names`, or more
        than one key found, or `callback` raising exceptions)

        :param dic: a python dic
        :param default: what to return if no key of `dic` matches any of `self.names`.
            If None (default when missing), then a 'MissingArgument' is raised if `dic` does
            not contain any key among `self.names`
        :param callback: None or function. If None, does nothing on `dic[name]`,
            where `name` is the first element of `self.names` which is found as key of `dic`.
            Otherwise returns
            `S2SArgument.parse(name, dic[name], callback, **callback_kwargs)`
        '''
        name, value = self._get(dic, False, default)
        return value if callback is None else \
            S2SArgument.parse(name, value, callback, **callback_kwargs)

    def popfrom(self, dic, default=None, callback=None, **callback_kwargs):
        '''Pops and returns the value mapped to this argument from `dic`.
        Raises BadArgument exceptions (e.g., no key found in `dic` among `self.names`, or more
        than one key found, or `callback` raising exceptions)

        :param dic: a python dic
        :param default: what to return if no key of `dic` matches any of `self.names`.
            If None (default when missing), then a 'MissingArgument' is raised if `dic` does
            not contain any key among `self.names`
        :param callback: None or function. If None, does nothing on `dic.pop(name)`,
            where `name` is the first element of `self.names` which is found as key of `dic`.
            Otherwise returns
            `S2SArgument.parse(name, dic[name], callback, **callback_kwargs)`
        '''
        name, value = self._get(dic, True, default)
        return value if callback is None else \
            S2SArgument.parse(name, value, callback, **callback_kwargs)

    @staticmethod
    def parse(name, value, func, *args, **kwargs):
        '''Calls `func` on the given value, and returns the result of `func`
        Raises BadArgument exceptions (e.g., ValueError, TypeErrors) with the given name
        in the exception str representation.
        This function can be called from `popfrom` and `getfrom`

        :param value: any python object.
        :param func: the function to be called with `value` as first argument. If not
            callable (function) if will return `func` ignoring *args and **kwargs, if any.
        :param args: optional positional arguments to be passed to `func`
        :param args: optional keyword arguments to be passed to `func`
        '''
        try:
            return func(value, *args, **kwargs) if hasattr(func, '__call__') else func
        except TypeError as terr:
            raise BadTypeArg(name, terr)
        except Exception as exc:
            raise BadValueArg(name, exc)

    @property
    def name(self):
        '''returns the name of this argument (string). `self.names` contains the returned string
        (not necessarily at the first position when iterating over it)'''
        return self._name
    
    @property
    def names(self):
        '''returns the names of this argument (set of strings). `self.name` is an element of
        the returned set (not necessarily at the first position when iterating over the set)'''
        return self._names


def typesmatch(value, other_value):
    '''checks that value is of the same type (same class, or subclass) of `other_value`.
    Raises TypeError if that's not the case
    
    :param value: a python object
    :param other_value: a python object
    
    :return: value
    '''
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
    '''
    try:
        strings = set()

        # we assume, when parsearg is not list, that parsearg is str in both python2 and python3,
        # i.e. it is NOT bytes in python2. The line below checks if is an iterable first:
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


def extract_dburl_if_yamlpath(value, param_name='dburl'):
    """
    Returns the database path from 'value':
    'value' can be a file (in that case is assumed to be a yaml file with the
    `param_name` key in it, which must denote a db path) or the database path otherwise
    """
    if not isinstance(value, string_types) or not value:
        raise TypeError('please specify a string denoting either a path to a yaml file with the '
                        '`dburl` parameter defined, or a valid db path')
    return yaml_load(value)[param_name] if (os.path.isfile(value)) else value


def keyval_list_to_dict(value):
    """parses optional event query args (when the 'd' command is issued) into a dict"""
    # use iter to make a dict from a list whose even indices = keys, odd ones = values
    # https://stackoverflow.com/questions/4576115/convert-a-list-to-a-dictionary-in-python
    itr = iter(value)
    return dict(zip(itr, itr))


def create_session(dburl):
    '''Creates an asql-alchemy session from dburl. Raises TypeError if dburl os not
    a string, or any SqlAlchemy exception if the session could not be created
    
    :param dburl: string denoting a database url (currently postgres and sqlite supported
    '''
    if not isinstance(dburl, string_types):
        raise TypeError('string required, %s found' % str(type(dburl)))
    return get_session(dburl, scoped=False)


def load_tt_table(file_or_name):
    '''Loads the given TTTable object from the given file path or name. If name (string)
    it must match any of the builtin TTTable .npz files defined in this package
    Raises TypeError or any Exception that TTTable might raise (including when the file is not
    found)
    '''
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
    except (TypeError, ValueError) as _:
        try:
            days = int(obj)
            now = datetime.utcnow()
            endt = datetime(now.year, now.month, now.day, 0, 0, 0, 0)
            return endt - timedelta(days=days)
        except Exception:
            pass
        if isinstance(_, TypeError):
            raise TypeError(("iso-formatted datetime string, datetime "
                             "object or int required, found %s") % str(type(obj)))
        else:
            raise _


def valid_fdsn(url):
    '''Returns url if it matches a FDSN service (valid strings are 'eida' and 'iris'),
    raises ValueError or TypeError otherwise'''
    if not isinstance(url, string_types):
        raise TypeError('string required')
    if url.lower() in ('eida', 'iris'):
        return url
    reg = re.compile("^.*/fdsnws/(?P<service>[^/]+)/(?P<majorversion>\\d+)/query$")
    match = reg.match(url)
    if not match or not match.group('service') or not match.group('majorversion'):
        raise ValueError("No FDSN url, the format needs to be: "
                         "<site>/fdsnws/<service>/<majorversion>/query")
    if match.group('service') not in ('dataselect', 'station', 'event'):
        raise ValueError("No FDSN url: service not in ('station', 'event', 'dataselect')")
    return url


def load_config_for_download(config, parseargs, **param_overrides):
    '''loads download arguments from the given config (yaml file or dict) after parsing and
    checking some of the dict keys.

    :return: a dict loaded from the given `config` and with parseed arguments (dict keys)

    Raises BadArgument in case of parsing errors, missisng arguments, conflicts etcetera
    '''
    try:
        dic = yaml_load(config, **param_overrides)
    except Exception as exc:
        raise BadValueArg('config', exc)

    # normalize eventws_query_args: the sub-dict is correctly updated. The function
    # yaml_load updates nested sub-dict values, so that if both dic['eventws_query_args']
    # and param_overrides['eventws_query_args'] contain, e.g. the key 'minlat', the key
    # is overridden in dic['eventws_query_args']. But
    # param_overrides['eventws_query_args'] might contain 'minlatitude' instead of 'minlat'
    # which should override 'minlat' in dic['eventws_query_args'] as well.
    # Check these cases of double names:
    overrides_eventdic = param_overrides.get('eventws_query_args', {})
    yaml_eventdic = dic['eventws_query_args']
    for par in overrides_eventdic:
        for find, rep in (('latitude', 'lat'), ('longitude', 'lon'), ('magnitude', 'mag')):
            twinpar = par.replace(find, rep)
            if twinpar == par:
                twinpar.replace(rep, find)
            if twinpar != par and twinpar in yaml_eventdic:
                # rename the overridden par with the previously set config par:
                yaml_eventdic[twinpar] = yaml_eventdic.pop(par)
                break

    if parseargs:
        remainingkeys = set(dic.keys())

        # parse arguments (dic keys):

        # First, two arguments which have to be replaced (pop=True)
        # and assigned to new dic key:
        argument = S2SArgument('dburl')
        dic['session'] = argument.popfrom(dic, callback=create_session)
        # now remove the already processed dic keys:
        remainingkeys -= argument.names  # argument.names is simply set(['dburl]) in this case
    
        argument = S2SArgument('traveltimes_model')
        dic['tt_table'] = argument.popfrom(dic, callback=load_tt_table)
        remainingkeys -= argument.names
    
        # parse "simple" arguments where we must only parse a value and replace it in the dict
        # (no key replacement / no variable num of arg names):
        for argument, func in  [(S2SArgument('start', 'starttime'), valid_date),
                                (S2SArgument('end', 'endtime'), valid_date),
                                (S2SArgument('eventws'), valid_fdsn),
                                (S2SArgument('dataws'), valid_fdsn)]:
            dic[argument.name] = argument.getfrom(dic, callback=func)
            remainingkeys -= argument.names
    
        # then, network channel ... arguments:
        for argument in (S2SArgument('networks', 'net', 'network'),
                         S2SArgument('stations', 'sta', 'station'),
                         S2SArgument('locations', 'loc', 'location'),
                         S2SArgument('channels', 'cha', 'channel'),):
            dic[argument.name] = argument.popfrom(dic, default=[], callback=nslc_param_value_aslist)
            remainingkeys -= argument.names
    
        # For all remaining arguments, just check the type as it should match the
        # default download config shipped with this package:
        orig_config = yaml_load(get_templates_fpath("download.yaml"))
        for key in remainingkeys:
            try:
                other_value = orig_config[key]
            except KeyError:
                raise UnknownArg(key)
            S2SArgument.parse(key, dic[key], typesmatch, other_value)

    return dic


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
        raise Exception('function "%s" not found in %s' % (str(funcname), pyfile))
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


def filewritable(filepath):
    '''checks that the file is writable, i.e. that is a string and its directory exists'''
    if not isinstance(filepath, string_types):
        raise TypeError('string required, found %s' % str(type(filepath)))

    if not os.path.isdir(os.path.dirname(filepath)):
        raise ValueError('cannot write file: parent directory does not exist')

    return filepath


def load_config_for_process(dburl, pyfile, funcname=None, config=None, outfile=None):
    '''checks process arguments.
    Returns the tuple session, pyfunc, config_dict,
    where session is the dql alchemy session from `dburl`,
    pyfunc is the python function loaded from `pyfile`, and config_dict is the dict loaded from
    `config` which must denote a path to a yaml file, or None (config_dict will be empty
    in this latter case)
    '''
    session = S2SArgument.parse('dburl', dburl, create_session)
    funcname = S2SArgument.parse('funcname', funcname, get_funcname)
    if config is not None:
        config = S2SArgument.parse('config', config, yaml_load)
    else:
        config = {}
    pyfunc = S2SArgument.parse('pyfile', pyfile, load_pyfunc, funcname)
    if outfile is not None:
        S2SArgument.parse('outfile', outfile, filewritable)
    # nothing more to process
    return session, pyfunc, funcname, config



