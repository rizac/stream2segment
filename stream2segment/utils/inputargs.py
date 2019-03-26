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
    get_templates_fpath, normalizedpath
from stream2segment.utils import strptime, load_source
from stream2segment.traveltimes.ttloader import TTTable
from stream2segment.io.db.models import Fdsnws
from stream2segment.download.utils import Authorizer, EVENTWS_MAPPING,\
    EVENTWS_SAFE_PARAMS


class BadArgument(Exception):
    '''An exception whose string method is similar to click formatted output. It
    supports sub-classes for most common argument errors.
    Typical usage for modules importing it:
    ```
        param = 'my_param_name'
        try:
            ... do operations
        except Exception as exc:
            raise BadArgument(param, exc)
    ```
    '''
    def __init__(self, param_name, error, msg_preamble=''):
        '''init method. Formats the message according to the given parameters

        The formatted output, depending on the truthy value of the arguments will be:

        'Error: %(msg_preamble) "%(param_name)": %(error)'
        'Error: "%(param_name)": %(error)"
        'Error: "%(param_name)"'

        :param param_name: the parameter name (string), or a list of parameter names
            (if the parameter supports several optional names)
        :param error: the original exception, or a string message. If exception
            in (TypeError, KeyError), it will determine the message preamble,
            if not explicitly passed (see below)
        :param msg_preamble: the optional message preamble, as string. If not
            provided (empty string by default), it will default to a string
            according to `error` type
        '''
        if not msg_preamble:
            msg_preamble = "Invalid value for"
            if isinstance(error, KeyError):
                msg_preamble = "Missing value for"
                error = ''
            elif isinstance(error, TypeError):
                msg_preamble = "Invalid type for"

        super(BadArgument, self).__init__(error.message
                                          if isinstance(error, BadArgument)
                                          else str(error))
        self.msg_preamble = msg_preamble
        self.param_name = param_name

    @property
    def message(self):
        msg = '%s' if not self.msg_preamble else self.msg_preamble.strip() + " %s"
        # Access the parent message (works in py2 and 3):
        err_msg = self.args[0]  # pylint: disable=unsubscriptable-object
        pname = '"%s"' % (" / ".join('"%s"' % p for p in self.param_name)[1:-1]
                          if isinstance(self.param_name, (list, tuple)) else
                          str(self.param_name))
#         ('"%s"' % self.param_name) if self.param_name else \
#             'unknown parameter (check input arguments)'
        ret = (msg % pname) + (": " + err_msg if err_msg else '')
        return ret[0:1].upper() + ret[1:]

    def __str__(self):
        ''''''
        return "Error: %s" % self.message


def parse_arguments(yaml_dic, *params):
    '''Parses yaml_dic parameters according to `params`. Modifies in-place `yaml_dic` and
    returns the set of `yaml_dic` parameters not parsed.

    :param params: a list of dicts. Each dict defines how to parse the given parameter and can
        have the keys and values:
        'names': (mandatory) list / tuple of the parameter name(s): the first parameter name
            found in `yaml_dic` will be used, and  a :class:`ConflictingArgs` exception is raised
            if any of the other names is also found in `yaml_dic` keys. It can be also a string,
            in which case the function behaves as if `names` was a list with that string as only
            element
        'defvalue': (optional) when provided, and no parameter name is found in `yaml_dic`,
            this is used as value. If not provided and no name in `names` is in
            in `yaml_dic`, a :class:`MissingArg` exception is raised
        'newname': (optional) string denoting the new parameter name which will replace the
            old one. When missing, it defaults to `names[0]`
        'newvalue': (optional) a callable which accepts the parameter value as argument and
            returns a new value. The function can safely raise: its exception(s) will be converted
            to :class:BadTypeArg or :class:BadValueArg depending on the cause

        For each element of `params`, this function parses the given argument and raises the
        appropriate `BadArgument` exceptions
    :raise: BadArgument

    :return: the set of names of `yaml_dic` not parsed

    '''
    remainingkeys = set(yaml_dic)
    for param in params:
        names = param['names']
        if not isinstance(names, (list, tuple)):
            names = (names,)
        name, value = get(yaml_dic, names, param['defvalue']) \
            if 'defvalue' in param else get(yaml_dic, names)
        parsefunc = param.get('newvalue', lambda val: val)
        try:
            newvalue = parsefunc(value)
        except Exception as exc:
            raise BadArgument(name, exc)
        # names[0] is the key that will be set on yaml_dct, if newname is missing:
        newname = param.get('newname', names[0])
        # if the newname is not names[0], remove name (not names[0]) from yanl_dic:
        if newname != name:
            yaml_dic.pop(name, None)
        # set new name and new (parsed) value:
        yaml_dic[newname] = newvalue
        # remove the parsed keys from remainingkeys:
        remainingkeys -= set(names)
    return remainingkeys


# to make None a passable argument to the next function
# (See https://stackoverflow.com/a/14749388/3526777):
_DEF_GET_MISSING_ARG_ = object()


def get(dic, names, default_ifmissing=_DEF_GET_MISSING_ARG_):
    '''Similar to `dic.get` with optinal (multi) keys. I.e., it calls iteratively
    `dic.get(key)` for each key in `names` and stops at the first key found `n`.
    Returns the tuple `(n, dic[n])` and raises :class:`BadArgument` in case

    :param dic: the source dict
    :param names: list/tuple of `dic` keys to be searched.. It can be also a string,
        in which case the function behaves as if `names` was a list with that string as only
        element
    :param default_if_missing: if provided and not None (the default), then this is the
        value returned if no name is found. If not provided, and no name is found in
        `dic`, :class:`MissingArg` is raised
    '''
    if not isinstance(names, (list, tuple)):
        names = (names,)

    try:
        keys_in = [par for par in names if par in dic]
        if len(keys_in) > 1:
            raise BadArgument(keys_in, '', "Conflicting names")
        elif not keys_in:
            if default_ifmissing is not _DEF_GET_MISSING_ARG_:
                return names[0], default_ifmissing
            raise KeyError()  # handled in the except below. Note that a KeyError
                              # will prprend the 'Missing value' in BadArgument
        name = keys_in[0]
        return name, dic[name]

    except BadArgument:
        raise
    except Exception as _:  # for safety
        raise BadArgument(names, _)


def typesmatch(value, *other_values):
    '''checks that value is of the same type (same class, or subclass) of *any* `other_value`
    (at least one). Raises TypeError if that's not the case

    :param value: a python object
    :param other_values: python objects. This function raises if value is NOT of the same type of
        any other_values types

    :return: value
    '''
    for other_value in other_values:
        if issubclass(value.__class__, other_value.__class__):
            return value
    raise TypeError("%s expected, found %s" % (" or ".join(str(type(_)) for _ in other_values),
                                               str(type(value))))


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
            # it's an iterable not a string
            value = [value]

        for string in value:
            splitted = string.split(",")
            for chunk in splitted:
                chunk = chunk.strip()
                if ' ' in chunk:
                    raise Exception("invalid space char(s): '%s'" % chunk)
                # if i == 3 (location) convert '--' to '':
                strings.add(chunk)

        # some checks:
        if "!*" in strings:  # discard everything is not valid
            raise ValueError("'!*' (=discard all) invalid")
        elif "*" in strings:  # accept everything or X => X is redundant
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


def get_session(dburl, for_process=False):
    '''Creates an asql-alchemy session from dburl. Raises TypeError if dburl os not
    a string, or any SqlAlchemy exception if the session could not be created

    :param dburl: string denoting a database url (currently postgres and sqlite supported
    '''
    if not isinstance(dburl, string_types):
        raise TypeError('string required, %s found' % str(type(dburl)))
    if for_process:
        from stream2segment.process.db import get_session as sess_func
    else:
        from stream2segment.io.db import get_session as sess_func
    return sess_func(dburl, scoped=False)


def create_auth(restricted_data, dataws, configfile=None):
    '''Creates an Auth class (handling authentication/authorization)
    from the given restricted_data

    :param restricted_data: either file path, to token, token data in bytes, or
        tuple (user, password). If None, or the empty string, None is returned
    '''
    if restricted_data in ('', None, b''):
        restricted_data = None
    elif isinstance(restricted_data, string_types) and configfile is not None:
        restricted_data = normalizedpath(restricted_data, os.path.dirname(configfile))
    ret = Authorizer(restricted_data)
    # here we have 4 cases: two ok ('eida' + token, any other fdsn + username & password)
    # Bad cases: eida + username & password: raise
    # any other fdsn + token: return normally, we might have provided a single eida datacenter
    #    in which case the parameter set is fine.
    if dataws.lower() == 'eida' and ret.userpass:
        raise ValueError('downloading from EIDA requires a token, not username and password')
    return ret


def parse_inventory(inventory):
    '''parses inventory returning True, False or 'only'
    '''
    inv = inventory
    if isinstance(inventory, string_types):
        if inventory.lower() == 'true':
            inventory = True
        elif inventory.lower() == 'false':
            inventory = False
        else:
            inventory = inventory.lower()
    if inventory not in (True, False, 'only'):
        raise ValueError('value can be true, false or only, %s provided' % str(inv))
    return inventory


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


def valid_fdsn(url, is_eventws, configfile=None):
    '''Returns url if it matches a FDSN service (valid strings are 'eida' and 'iris'),
    raises ValueError or TypeError otherwise'''
    if not isinstance(url, string_types):
        raise TypeError('string required')

    if (is_eventws and url.lower() in EVENTWS_MAPPING) or \
            (not is_eventws and url.lower() in ('eida', 'iris')):
        return url.lower()

    if is_eventws:
        fpath = url if configfile is None else normalizedpath(url, os.path.dirname(configfile))
        if os.path.isfile(fpath):
            return fpath
        try:
            return Fdsnws(url).url()
        except Exception:
            raise ValueError('Invalid FDSN url or file path, check typos')

    return Fdsnws(url).url()


def between(min, max, include_start=True, include_end=True, pass_if_none=True):
    def func(val):
        if val is None:
            if pass_if_none:
                return True
            raise ValueError('value is None/null')
        is_ok = min is None or val > min or (include_start and val >= min)
        if not is_ok:
            raise ValueError('%s must be %s %s' %
                             (str(val), '>=' if include_start else '>', str(min)))
        is_ok = max is None or val < max or (include_start and val <= max)
        if not is_ok:
            raise ValueError('%s must be %s %s' %
                             (str(val), '<=' if include_end else '<', str(max)))
        return val

    return func


def parse_download_advanced_settings(advanced_settings):
    paramname = 'download_blocksize'
    try:
        if advanced_settings[paramname] <= 0:
            advanced_settings[paramname] = -1
        paramname = 'max_thread_workers'
        if advanced_settings[paramname] <= 0:
            advanced_settings[paramname] = None
        paramname = 'db_buf_size'
        advanced_settings[paramname] = max(advanced_settings[paramname], 1)
    except Exception as exc:
        raise BadArgument(paramname, exc)
    return advanced_settings


def load_config_for_download(config, parseargs, **param_overrides):
    '''loads download arguments from the given config (yaml file or dict) after parsing and
    checking some of the dict keys.

    :return: a dict loaded from the given `config` and with parseed arguments (dict keys)

    Raises BadArgument in case of parsing errors, missisng arguments, conflicts etcetera
    '''
    try:
        config_dict = yaml_load(config, **param_overrides)
    except Exception as exc:
        raise BadArgument('config', exc)

    if parseargs:
        # few variables:
        configfile = config if (isinstance(config, string_types) and os.path.isfile(config))\
            else None

        # define first default event params in order to avoid typos
        def_evt_params = EVENTWS_SAFE_PARAMS

        # now, what we want to do here is basically convert config_dict keys
        # into suitable arguments for stream2segment functions: this includes
        # renaming params, parsing/converting their values, raising
        # BadArgument exceptions and so on

        # Let's configure a 'params' list, a list of dicts where each dict is a 'param checker'
        # with the following keys (at least one should be provided):
        # names: list of strings. provide it in order to check for optional names,
        #        check that only one param is provided, and
        #        replace whatever is found with the first item in the list
        # newname: string, provide it if you want to replace names above with this value
        #          instead first item in 'names'
        # defvalue: if provided, then the parameter is optional and will be set to this value
        #           if not provided, then the parameter is mandatory (BadArgument is raised in case)
        # newvalue: function accepting a value (the parameter value) raising whatever is
        #           needed if the parameter is invalid, and returning the correct parameter value
        params = [
            {
             'names': def_evt_params[:2],  # ['minlatitude', 'minlat'],
             'defvalue': None,
             'newvalue': between(-90.0, 90.0)
            },
            {
             'names': def_evt_params[2:4],  # ['maxlatitude', 'maxlat'],
             'defvalue': None,
             'newvalue': between(-90.0, 90.0)
            },
            {
             'names': def_evt_params[4:6],  # ['minlongitude', 'minlon'],
             'defvalue': None,
             'newvalue': between(-180.0, 180.0)
            },
            {
             'names': def_evt_params[6:8],  # ['maxlongitude', 'maxlon'],
             'defvalue': None,
             'newvalue': between(-180.0, 180.0)
            },
            {
             'names': def_evt_params[8:10],  # ['minmagnitude', 'minmag'],
             'defvalue': None
            },
            {
             'names': def_evt_params[10:12],  # ['maxmagnitude', 'maxmag'],
             'defvalue': None
            },
            {
             'names': def_evt_params[12:13],  # ['mindepth'],
             'defvalue': None
            },
            {
             'names': def_evt_params[13:14],  # ['maxdepth'],
             'defvalue': None
            },
            {
             'names': ['inventory'],
             'newvalue': parse_inventory
             },
            {
             'names': ['restricted_data'],
             'newname': 'authorizer',
             'newvalue': lambda val: create_auth(val, config_dict['dataws'], configfile)
            },
            {
             'names': ['dburl'],
             'newname': 'session',
             'newvalue': get_session
            },
            {
             'names': ['traveltimes_model'],
             'newname': 'tt_table',
             'newvalue': load_tt_table
            },
            {
             'names': ('starttime', 'start'),
             'newvalue': valid_date
            },
            {
             'names': ('endtime', 'end'),
             'newvalue': valid_date
            },
            {
             'names': ['eventws'],
             'newvalue': lambda url: valid_fdsn(url, is_eventws=True, configfile=configfile)
            },
            {
             'names': ['dataws'],
             'newvalue': lambda url: valid_fdsn(url, is_eventws=False)
            },
            {
             'names': ('network', 'net', 'networks'),
             'defvalue': [],
             'newvalue': nslc_param_value_aslist
            },
            {
             'names': ('station', 'sta', 'stations'),
             'defvalue': [],
             'newvalue': nslc_param_value_aslist
            },
            {
             'names': ('location', 'loc', 'locations'),
             'defvalue': [],
             'newvalue': nslc_param_value_aslist
            },
            {
             'names': ('channel', 'cha', 'channels'),
             'defvalue': [],
             'newvalue': nslc_param_value_aslist
            },
            {'names': ['eventws_params', 'eventws_query_args']},
            {
             'names': ['advanced_settings'],
             'newvalue': parse_download_advanced_settings
            }
            ]

        remainingkeys = parse_arguments(config_dict, *params)

        # check that we did not implement any conflicting arg in
        # eventws_params

        # remove all event-related parameters (except starttime and endtime):
        # and put them in the eventws_params dict:
        # pop all stuff from 'eventws_params' and put it into 'event_query_params':
        eventsearchparams = config_dict['eventws_params']
        # eventsearchparams might be none
        if not eventsearchparams:
            config_dict['eventws_params'] = eventsearchparams = {}
        for par in def_evt_params:
            if par in eventsearchparams:  # conflict:
                raise BadArgument('eventws_params',
                                  'conflicting parameter "%s"' % par)
            value = config_dict.pop(par, None)
            if value is not None:
                eventsearchparams[par] = value

        # For all remaining arguments, just check the type as it should match the
        # default download config shipped with this package:
        orig_config = yaml_load(get_templates_fpath("download.yaml"))
        for key in remainingkeys:
            try:
                other_value = orig_config[key]
            except KeyError:
                raise BadArgument(key, '', 'No such option')
            try:
                typesmatch(config_dict[key], other_value)
            except Exception as exc:
                raise BadArgument(key, exc)

    return config_dict


def load_pyfunc(pyfile, funcname):
    '''Returns the python module from the given python file'''
    if not isinstance(pyfile, string_types):
        raise TypeError('string required, not %s' % str(type(pyfile)))

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


def load_config_for_process(dburl, pyfile, funcname=None, config=None, outfile=None,
                            **param_overrides):
    '''checks process arguments.
    Returns the tuple session, pyfunc, config_dict,
    where session is the dql alchemy session from `dburl`,
    pyfunc is the python function loaded from `pyfile`, and config_dict is the dict loaded from
    `config` which must denote a path to a yaml file, or None (config_dict will be empty
    in this latter case)
    '''
    try:
        session = get_session(dburl, True)
    except Exception as exc:
        raise BadArgument('dburl', exc)

    try:
        funcname = get_funcname(funcname)
    except Exception as exc:
        raise BadArgument('funcname', exc)

    try:
        # yaml_load accepts a file name or a dict
        config = yaml_load({} if config is None else config, **param_overrides)
    except Exception as exc:
        raise BadArgument('config', exc)

    # NOTE: contrarily to the download routine, we cannot check the types of the config because
    # no parameter is mandatory, and thus they might NOT be present in the config.

    try:
        pyfunc = load_pyfunc(pyfile, funcname)
    except Exception as exc:
        raise BadArgument('pyfile', exc)

    if outfile is not None:
        try:
            filewritable(outfile)
        except Exception as exc:
            raise BadArgument('outfile', exc)

    # nothing more to process
    return session, pyfunc, funcname, config


def load_session_for_dinfo(dburl):
    try:
        return get_session(dburl)
    except Exception as exc:
        raise BadArgument('dburl', exc)


