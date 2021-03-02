"""
Module with utilities for checking / parsing / setting input arguments from
the command line interface (cli), e.g. download and process.

:date: Feb 27, 2018

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
import inspect
import os
# import sys
# import re
import sys
from datetime import datetime, timedelta
from itertools import chain
import importlib

from future.utils import string_types
from stream2segment.process import SkipSegment

from stream2segment.utils.resources import yaml_load, get_ttable_fpath, \
    get_templates_fpath, normalizedpath
from stream2segment.utils import strptime, load_source
from stream2segment.traveltimes.ttloader import TTTable
from stream2segment.io.db.models import Fdsnws
from stream2segment.download.utils import Authorizer, EVENTWS_MAPPING,\
    EVENTWS_SAFE_PARAMS


class BadArgument(Exception):
    """Exception describing a bad configuration parameter, as
    :class:`click.exceptions.BadParameter`, provides similar messages when
    output in the terminal but it can be used outside a command line interface
    environment
    """
    def __init__(self, param_name, error, msg_preamble=''):
        """Initialize a BadArgument object. Formats the message according to
        the given parameters. The formatted output string, depending on the
        value of the arguments will be:

        'Error: %(msg_preamble) "%(param_name)": %(error)'
        'Error: "%(param_name)": %(error)"'
        'Error: "%(param_name)"'

        :param param_name: the parameter name (string), or a list of parameter
            names (if the parameter supports several optional names)
        :param error: the original exception, or a string message. If exception
            in (TypeError, KeyError), it will determine the message preamble,
            if not explicitly passed (see below)
        :param msg_preamble: the optional message preamble, as string. If not
            provided (empty string by default), it will default to a string
            according to `error` type, e.g. KeyError => 'Missing value for'
        """
        if not msg_preamble:
            msg_preamble = "Invalid value for"
            if isinstance(error, KeyError):
                msg_preamble = "Missing value for"
                error = ''
            elif isinstance(error, TypeError):
                msg_preamble = "Invalid type for"

        if isinstance(error, BadArgument):
            err_msg = error.message
        else:
            err_msg = str(error)

        super(BadArgument, self).__init__(err_msg)
        self.msg_preamble = msg_preamble
        self.param_name = param_name

    @property
    def message(self):
        # msg = '%s' if not self.msg_preamble else \
        #     self.msg_preamble.strip() + " %s"
        # Access the parent message (works in py2 and 3):
        if isinstance(self.param_name, (list, tuple)):
            p_name = " / ".join('"%s"' % p for p in self.param_name)[1:-1]
        else:
            p_name = str(self.param_name)
        p_name = '"' + p_name + '"'
        msg_preamble = self.msg_preamble
        if msg_preamble:
            msg_preamble += ' '
        err_msg = str(self.args[0] or '')  # noqa
        if err_msg:
            # lower case first letter as err_msg will follow a ":"
            err_msg = ": " + err_msg[0].lower() + err_msg[1:]
        # ret = (msg % p_name) + ": " + err_msg
        ret = msg_preamble + p_name + err_msg
        return ret[0:1].upper() + ret[1:]

    def __str__(self):
        """String representation of this object"""
        return "Error: %s" % self.message


def parse_arguments(yaml_dic, *params):
    """Parse `yaml_dic` parameters according to `params`, updating the
    `yaml_dict`. Raises `BadArgument` exception in case of parameter errors.
    WARNING: this method modifies `yaml_dic` in-place!

    :param params: a list of dicts. Each dict defines how to parse the given
        parameter and can have the keys and values:
        'names': (mandatory) list / tuple of the parameter name(s): if list or
            tuple, a `BadArgument` is raised in case of conflicts (more than
            one name is found in `yaml_dic` keys). If string, it is converted
            to a list with that string as only element
        'defvalue': (optional) when provided, and no parameter name is found in
            `yaml_dic`, this is the parameter value. If not provided and no
            name in `names` is in `yaml_dic`, a `BadArgument` is raised
        'newname': (optional) string denoting the new parameter name which will
            replace the old one in `yaml_dict`. When missing, it defaults to
            `names[0]`
        'newvalue': (optional) a callable which accepts as argument the
            parameter value and returns a new value.
            `yaml_dic[newname] = newvalue` will be then called. The callable
            can safely raise, any exception will be converted to `BadArgument`
            and re-raised

    :raise: BadArgument
    """
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
        # names[0] is the key that will be set on yaml_dct,
        # if newname is missing:
        newname = param.get('newname', names[0])
        # if the newname is not names[0], remove name (not names[0])
        # from yanl_dic:
        if newname != name:
            yaml_dic.pop(name, None)
        # set new name and new (parsed) value:
        yaml_dic[newname] = newvalue


# to make None a passable argument to the next function
# (See https://stackoverflow.com/a/14749388/3526777):
_DEF_GET_MISSING_ARG_ = object()


def get(dic, names, default_ifmissing=_DEF_GET_MISSING_ARG_):
    """Similar to `dic.get` with optional (multi) keys. I.e., it calls
    iteratively `dic.get(key)` for each key in `names` and stops at the first
    key found `n`. Returns the tuple `(n, dic[n])` and raises
    :class:`BadArgument` in case

    :param dic: the source dict
    :param names: list/tuple of `dic` keys to be searched.. It can be also a
        string, in which case the function behaves as if `names` was a list
        with that string as only element
    :param default_ifmissing: if provided and not None (the default), then
        this is the value returned if no name is found. If not provided, and no
        name is found in `dic`, :class:`MissingArg` is raised
    """
    if not isinstance(names, (list, tuple)):
        names = (names,)

    try:
        keys_in = [par for par in names if par in dic]
        if len(keys_in) > 1:
            raise BadArgument(keys_in, '', "Conflicting names")
        elif not keys_in:
            if default_ifmissing is not _DEF_GET_MISSING_ARG_:
                return names[0], default_ifmissing
            raise KeyError()
            # KeyError caught below. Note that a KeyError will
            # prepend the 'Missing value' in BadArgument
        name = keys_in[0]
        return name, dic[name]

    except BadArgument:
        raise
    except Exception as _:  # for safety
        raise BadArgument(names, _)


def typesmatch(value, *other_values):
    """Check that value is of the same type (same class, or subclass) of *any*
    `other_value` (at least one). Raises TypeError if that's not the case

    :param value: a python object
    :param other_values: python objects. This function raises if value is NOT
        of the same type of any other_values types

    :return: value
    """
    for other_value in other_values:
        if issubclass(value.__class__, other_value.__class__):
            return value
    raise TypeError("%s expected, found %s" %
                    (" or ".join(str(type(_)) for _ in other_values),
                     str(type(value))))


def nslc_param_value_aslist(value):
    """Return a nslc (network/station/location/channel) parameter value
    converted as list. This method cleans-up and checks `value` splitting each
    of its string elements with the comma "," and aggregating all the string
    chunks into a single list, after performing some sanity check. The
    resulting list is also sorted alphabetically (for unit testing and
    readability). Raises ValueError in case some sanity checks fail (e.g.,
    conflicts, syntax errors)

    Examples:

    Func. arguments      Result (with comment)
    =================== =================================================
    (['A','D','C','B'])  ['A', 'B', 'C', 'D']  # note result is sorted
    ('B,C,D,A')          ['A', 'B', 'C', 'D']  # same as above
    ('A*, B??, C*')      ['A*', 'B??', 'C*']  # FDSN wildcards accepted
    ('!A*, B??, C*')     ['!A*', 'B??', 'C*']  # in s2s, !A* means "not A*"
    (' A, B ')           ['A', 'B']  # leading and trailing spaces ignored
    ('*')                []  # [] means "match all"
    ([])                 []  # same as above
    ('  ')               ['']  # string is stripped: match the empty string
    ("")                 [""]  # match the empty string
    ("!")                ['!']  # match any non empty string
    ("!*")               this raises (you cannot specify "discard all")
    ("!H*, H*")          this raises (it's a paradox)
    (" A B,  CD")        this raises ('A B' invalid: only leading and trailing
                                      spaces allowed)

    :param value: string or iterable of strings: (iterable in this context
        means Python iterable EXCEPT strings). If string, the argument will be
        converted to the list [value] to make it iterable before processing it
    """
    try:
        strings = set()

        # we assume, when parsearg is not list, that parsearg is str in both
        # py2 and py3, i.e. it is NOT bytes in python2. The line below checks
        # if is an iterable first:
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
                    raise Exception("conflicting values: '%s' and '%s'" %
                                    (string, opposite))

        return sorted(strings)

    except Exception as exc:
        raise ValueError(str(exc))


def extract_dburl_if_yamlpath(value, param_name='dburl'):
    """Return the database path from 'value': 'value' can be a file (in that
    case is assumed to be a yaml file with the `param_name` key in it, which
    must denote a db path) or the database path otherwise
    """
    if not isinstance(value, string_types) or not value:
        raise TypeError('please specify a string denoting either a path to a '
                        'yaml file with the `dburl` parameter defined, or a '
                        'valid db path')
    return yaml_load(value)[param_name] if (os.path.isfile(value)) else value


def keyval_list_to_dict(value):
    """Parse optional event query args (when the 'd' command is issued) into
    a dict"""
    # use iter to make a dict from a list whose even indices = keys, odd
    # ones = values (https://stackoverflow.com/a/4576128)
    itr = iter(value)
    return dict(zip(itr, itr))


def get_session(dburl, for_process=False, raise_bad_argument=False,
                scoped=False, **engine_kwargs):
    """Create an SQL-Alchemy session from dburl. Raises if `dburl` is
    not a string, or any SqlAlchemy exception if the session could not be
    created. If `raise_bad_argument` is True (default False), raises
    wraps any exception into a `BadArgument` associated to the parameter
    'dburl'.

    :param dburl: string denoting a database url (currently postgres and sqlite
        supported
    :param for_process: boolean (default: False) whether the session should be
        used for processing, i.e. the database is supposed to exist already and
        the `Segment` model has ObsPy method such as `Segment.stream()`
    :param raise_bad_argument: boolean (default: False)if any exception should
        be wrapped into a `BadArgument` exception,whose message will be
        prefixed with the parameter name 'dburl'
    :param scoped: boolean (False by default) if the session must be scoped
        session
    :param engine_args: optional keyword argument values for the
        `create_engine` method. E.g., let's provide two engine arguments,
        `echo` and `connect_args`:
        ```
        get_session(dbpath, ..., echo=True, connect_args={'connect_timeout': 10})
        ```
        For info see:
        https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine.params.connect_args
    """
    try:
        if not isinstance(dburl, string_types):
            raise TypeError('string required, %s found' % str(type(dburl)))
        # import in function to speed up module imports from cli:
        if for_process:
            # important, rename otherwise conflicts with this function name:
            from stream2segment.process.db import get_session as sess_func
        else:
            # important, rename otherwise conflicts with this function name:
            from stream2segment.io.db import get_session as sess_func

        sess = sess_func(dburl, scoped=scoped, **engine_kwargs)

        # Check if database exist, which should not always be done (e.g.
        # for_processing=True). Among other methods
        # (https://stackoverflow.com/a/3670000
        # https://stackoverflow.com/a/59736414) this seems to do what we need
        # (we might also not check if that the tables length > 0 sometime):
        sess.bind.engine.table_names()
        return sess
    except Exception as exc:
        if raise_bad_argument:
            raise BadArgument('dburl', exc)
        raise


def create_auth(restricted_data, dataws, configfile=None):
    """Create an Auth class (handling authentication/authorization)
    from the given restricted_data

    :param restricted_data: either file path, to token, token data in bytes, or
        tuple (user, password). If None, or the empty string, None is returned
    """
    if restricted_data in ('', None, b''):
        restricted_data = None
    elif isinstance(restricted_data, string_types) and configfile is not None:
        restricted_data = normalizedpath(restricted_data, os.path.dirname(configfile))
    ret = Authorizer(restricted_data)
    # check dataws is single element list:
    if len(dataws) != 1:
        raise ValueError('downloading restricted data requires '
                         'a single URL in `dataws`')
    dataws = dataws[0]
    # Here we have 4 cases:
    # 1 'eida' + token: OK
    # 2. Any other fdsn + username & password: OK
    # 3. eida + username & password: BAD. raise ValueError
    # 4. Any other fdsn + token: OK (we might have provided a single eida
    #                                datacenter in which case it's fine)
    if dataws.lower() == 'eida' and ret.userpass:
        raise ValueError('downloading from EIDA requires a token, '
                         'not username and password')
    return ret


def parse_update_metadata(value):
    """Parse parse_update_metadata returning True, False or 'only'"""
    val_str = str(value).lower()
    if val_str == 'true':
        return True
    if val_str == 'false':
        return False
    if val_str == 'only':
        return val_str
    raise ValueError('value can be true, false or only, %s provided' % val_str)


def load_tt_table(file_or_name):
    """Load the given TTTable object from the given file path or name. If name
    (string) it must match any of the builtin TTTable .npz files defined in
    this package. Raises TypeError or any Exception that TTTable might raise
    (including when the file is not found)
    """
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
                             "object or int required, found %s") %
                            str(type(obj)))
        else:
            raise _


def valid_fdsn(url, is_eventws, configfile=None):
    """Return url if it matches a FDSN service (valid strings are 'eida' and
    'iris'), raises ValueError or TypeError otherwise
    """
    if not isinstance(url, string_types):
        raise TypeError('string required')

    if (is_eventws and url.lower() in EVENTWS_MAPPING) or \
            (not is_eventws and url.lower() in ('eida', 'iris')):
        return url.lower()

    if is_eventws:
        fpath = url if configfile is None else \
            normalizedpath(url, os.path.dirname(configfile))
        if os.path.isfile(fpath):
            return fpath
        try:
            return Fdsnws(url).url()
        except Exception:
            raise ValueError('Invalid FDSN url or file path, check typos')

    return Fdsnws(url).url()


def dict_or_none(value):
    """Check that value is a dict and returns `value`.
    Returns {} if the value is None. In any other cases, raise ValueError
    """
    if value is None:
        value = {}
    if isinstance(value, dict):
        return value
    raise ValueError('dict/None required, found: %s' % str(type(value)))


def between(min, max, include_start=True, include_end=True, pass_if_none=True):
    def func(val):
        if val is None:
            if pass_if_none:
                return val
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

        paramname = 'max_concurrent_downloads'
        if paramname not in advanced_settings:
            # try to search old parameter "max_thread_workers"
            # (maybe an old download config)
            old_paramname = 'max_thread_workers'
            if old_paramname not in advanced_settings:
                raise KeyError()  # (will raise paramname error)
            # When old_paramname<=0, it defaulted to None (= max thread workers
            # automatically set by threadPool):
            if advanced_settings[old_paramname] <= 0:
                advanced_settings[old_paramname] = None
            advanced_settings[paramname] = advanced_settings.pop(old_paramname)

        if advanced_settings[paramname] is not None:
            advanced_settings[paramname] = int(advanced_settings[paramname])

        paramname = 'db_buf_size'
        advanced_settings[paramname] = max(advanced_settings[paramname], 1)
    except Exception as exc:
        raise BadArgument(paramname, exc)
    return advanced_settings


def check_search_radius(search_radius):
    """Check the validity of the 'search_radius' argument (dict)"""
    args = [
        search_radius.get('minmag'),
        search_radius.get('maxmag'),
        search_radius.get('minmag_radius'),
        search_radius.get('maxmag_radius'),
        search_radius.get('min'),
        search_radius.get('max')
    ]
    magdep_args = args[:4]
    magindep_args = args[4:]
    magdep_argscount = sum(_ is not None for _ in magdep_args)
    magindep_argscount = sum(_ is not None for _ in magindep_args)
    is_mag_dep = magdep_argscount == len(magdep_args) and not magindep_argscount
    is_mag_indep = magindep_argscount == len(magindep_args) and not magdep_argscount

    if is_mag_dep == is_mag_indep:
        raise ValueError("provide either 'min', 'max' or 'minmag', 'maxmag', "
                         "'minmag_radius', 'maxmag_radius'")

    # check errors:
    nofloaterr = ValueError('numeric values expected')
    if is_mag_dep:
        if not all(isinstance(_, (int, float)) for _ in magdep_args):
            raise nofloaterr
        if args[0] > args[1]:  # minmag > maxmag
            raise ValueError('minmag should not be greater than maxmag')
        if args[2] <= 0 or args[3] <= 0:  # minmag_radius or maxmag_radius <=0
            raise ValueError('minmag_radius and maxmag_radius should be '
                             'greater than 0')
        if args[0] == args[1] and args[2] == args[3]:
            # minmag == maxmag, minmag_radius == maxmag_radius => error
            raise ValueError('To supply a constant radius, set "min: 0" and '
                             'specify the radius with the "max" argument')
    else:
        if not all(isinstance(_, (int, float)) for _ in magindep_args):
            raise nofloaterr
        if args[-2] < 0:
            raise ValueError('min should not be lower than 0')
        if args[-1] <= 0:
            raise ValueError('max should be greater than 0')
        if args[-2] >= args[-1]:
            raise ValueError('min should be lower than max')

    return search_radius


def load_config_for_download(config, parseargs, **param_overrides):
    """Load download arguments from the given config (yaml file or dict) after
    parsing and checking some of the dict keys.

    :return: a dict loaded from the given `config` and with parsed arguments
        (dict keys)

    Raises `BadArgument` in case of parsing errors, missing arguments,
    conflicts and so on
    """
    try:
        config_dict = yaml_load(config, **param_overrides)
    except Exception as exc:
        raise BadArgument('config', exc)

    if parseargs:
        # few variables:
        configfile = None
        if isinstance(config, string_types) and os.path.isfile(config):
            configfile = config

        # define first default event params in order to avoid typos
        def_evt_params = EVENTWS_SAFE_PARAMS

        # now, what we want to do here is basically convert config_dict keys
        # into suitable arguments for stream2segment functions: this includes
        # renaming params, parsing/converting their values, raising
        # BadArgument exceptions and so on

        # Let's configure a 'params' list, a list of dicts where each dict is a
        # 'param checker' with the following keys (at least one should be
        # provided):
        # names: list of strings. provide it in order to check for optional
        #        names, check that only one param is provided, and
        #        replace whatever is found with the first item in the list
        # newname: string, provide it if you want to replace names above with
        #          this value instead first item in 'names'
        # defvalue: if provided, then the parameter is optional and will be set
        #           to this value if not provided, then the parameter is
        #           mandatory (BadArgument is raised in case)
        # newvalue: function accepting a value (the parameter value) raising
        #           whatever is needed if the parameter is invalid, and
        #           returning the correct parameter value
        params = [
            {
                # dataws is a list of strings, but for backward compatibility
                # we must accept strings too. Convert `dataws` to list AS FIRST
                # ARGUMENT (I.E., this must be the FIRST dict of the list), so
                # that any other parameter check below requiring dataws can
                # safely work with lists:
                'names': ['dataws'],
                'newvalue': lambda _: [_] if isinstance(_, string_types) else _
            },
            {
                'names': def_evt_params[:2],  # ['minlatitude', 'minlat'],
                'defvalue': None,  # None: param not added (see below)
                'newvalue': between(-90.0, 90.0)
            },
            {
                'names': def_evt_params[2:4],  # ['maxlatitude', 'maxlat'],
                'defvalue': None,  # None: param not added (see below)
                'newvalue': between(-90.0, 90.0)
            },
            {
                'names': def_evt_params[4:6],  # ['minlongitude', 'minlon'],
                'defvalue': None,  # None: param not added (see below)
                'newvalue': between(-180.0, 180.0)
            },
            {
                'names': def_evt_params[6:8],  # ['maxlongitude', 'maxlon'],
                'defvalue': None,  # None: param not added (see below)
                'newvalue': between(-180.0, 180.0)
            },
            {
                'names': def_evt_params[8:10],  # ['minmagnitude', 'minmag'],
                'defvalue': None  # None: param not added (see below)
            },
            {
                'names': def_evt_params[10:12],  # ['maxmagnitude', 'maxmag'],
                'defvalue': None  # None: param not added (see below)
            },
            {
                'names': def_evt_params[12:13],  # ['mindepth'],
                'defvalue': None  # None: param not added (see below)
            },
            {
                'names': def_evt_params[13:14],  # ['maxdepth'],
                'defvalue': None  # None: param not added (see below)
            },
            {
                'names': ['update_metadata'],
                'newvalue': parse_update_metadata
            },
            {
                'names': ['restricted_data'],
                'newname': 'authorizer',
                'newvalue': lambda val: create_auth(val,
                                                    config_dict['dataws'],
                                                    configfile)
            },
            {
                 'names': ['eventws'],
                 'newvalue': lambda url: valid_fdsn(url, is_eventws=True,
                                                    configfile=configfile)
            },
            {
                 'names': ['dataws'],
                 'newvalue': lambda urls: [valid_fdsn(url, is_eventws=False)
                                           for url in urls]
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
            {
                'names': ['eventws_params', 'eventws_query_args'],
                'defvalue': {},
                'newvalue': dict_or_none
            },
            {
                 'names': ['advanced_settings'],
                 'newvalue': parse_download_advanced_settings
            },
            {
                 'names': ['search_radius'],
                 'newvalue': check_search_radius
            },
            {
                'names': ['min_sample_rate'],
                'defvalue': 0,
                'newvalue': int
            }
        ]

        # store all keys now because we might change them (see below):
        all_keys = set(config_dict)
        # do the check (this MODIFIES config_dict in place!):
        parse_arguments(config_dict, *params)

        # Now check for:
        # 1a. parameter supplied here NOT in the default config
        # 1b. parameter supplied here with different type of the default config
        # 2. Parameters in the default config not supplied here

        # First, create some sets of params names:
        # the parsed keys (all names defined above):
        parsed_keys = set(chain(*(_['names'] for _ in params)))
        # load original configuration (default in this package):
        orig_config = yaml_load(get_templates_fpath("download.yaml"))

        # Check 1a. and 1b.:
        for key in all_keys - parsed_keys:
            try:
                other_value = orig_config[key]
            except KeyError:
                raise BadArgument(key, '', 'No such option')
            try:
                typesmatch(config_dict[key], other_value)
            except Exception as exc:
                raise BadArgument(key, exc)

        # Check 2. :
        missing_keys = set(orig_config) - all_keys - parsed_keys
        if missing_keys:
            raise BadArgument(list(missing_keys), KeyError())

        # At last, put all event-related parameters (except starttime and
        # endtime): and in the eventws_params dict (the latter is an OPTIONAL
        # dict which can be set in the config for ADDITIONAL eventws
        # parameters) and check for conflicts:
        # IF A PRAMETER IS NONE IT IS NOT ADDED
        _esp = 'eventws_params'
        eventsearchparams = config_dict[_esp]
        for par in def_evt_params:
            if par in eventsearchparams:  # conflict:
                raise BadArgument(_esp, 'conflicting parameter "%s"' % par)
            value = config_dict.pop(par, None)
            if value is not None:
                eventsearchparams[par] = value

    return config_dict


def load_pyfunc(pyfile, funcname):
    """Return the Python module from the given python file"""
    if not isinstance(pyfile, string_types):
        raise TypeError('string required, not %s' % str(type(pyfile)))

    if not os.path.isfile(pyfile):
        raise Exception('file does not exist')

    pymoduledict = load_source(pyfile).__dict__

    # check for new style module: SkipSegment instead of ValueError
    if 'SkipSegment' not in pymoduledict:
        raise ValueError('The module seems to be outdated. You need to import '
                         'SkipSegment\n("from stream2segment.process import '
                         'SkipSegment") and check your code:\nevery time you '
                         'write "raise ValueError(..." to skip a segment, replace'
                         'it with "raise SkipSegment(..."')

    if funcname not in pymoduledict:
        raise Exception('function "%s" not found in %s' %
                        (str(funcname), pyfile))
    return pymoduledict[funcname]


def get_funcname(funcname=None):
    """Return the Python module from the given python file"""
    if funcname is None:
        funcname = default_processing_funcname()

    if not isinstance(funcname, string_types):
        raise TypeError('string required, not %s' % str(type(funcname)))

    return funcname


def default_processing_funcname():
    """Return 'main', the default function name for processing, when such
    a name is not given"""
    return 'main'


def filewritable(filepath):
    """Check that the file is writable, i.e. that is a string and its
    directory exists"""
    if not isinstance(filepath, string_types):
        raise TypeError('string required, found %s' % str(type(filepath)))

    if not os.path.isdir(os.path.dirname(filepath)):
        raise ValueError('cannot write file: parent directory does not exist')

    return filepath


def load_config_for_process(dburl, pyfile, funcname=None, config=None,
                            outfile=None, **param_overrides):
    """Check process arguments. Returns the tuple session, pyfunc, config_dict,
    where session is the dql alchemy session from `dburl`, `funcname` is the
    Python function loaded from `pyfile`, and config_dict is the dict loaded
    from `config` which must denote a path to a yaml file, or None (config_dict
    will be empty in this latter case)
    """
    try:
        session = get_session(dburl, for_process=True)
    except Exception as exc:
        raise BadArgument('dburl', exc)

    try:
        funcname = get_funcname(funcname)
    except Exception as exc:
        raise BadArgument('funcname', exc)

    try:
        # yaml_load accepts a file name or a dict
        config = yaml_load({} if config is None else config, **param_overrides)

        adv_settings = config.get('advanced_settings', {})  # dict
        multi_process = adv_settings.get('multi_process', False)
        # the number of Pool processes can now be set directly to multi_process (which
        # accepts bool or int). Previously, there was a separate parameter for that,
        # num_processes. Let' implement backward compatibility here:
        if multi_process is True and 'num_processes' in adv_settings:
            multi_process = adv_settings['num_processes']

        # check parameters:
        if multi_process not in (True, False):
            try:
                int(multi_process)
                assert multi_process >= 0
            except:
                raise ValueError('advanced_settings.multi_process must be '
                                 'boolean or non negative int')
        chunksize = adv_settings.get('segments_chunksize', None)
        if chunksize is not None:
            try:
                int(chunksize)
                assert chunksize > 0
            except:
                raise ValueError('advanced_settings.chunksize must be '
                                 'null or positive int')

        exceptions = [SkipSegment]
        # In case we want to customize skip exceptions from config, below the
        # code snippet (this is momentarily hidden in the config documentation):
        exc_names = adv_settings.get('skip_exceptions', [])
        if exc_names is None or isinstance(exc_names, string_types):
            exc_names = [exc_names]
        for exc_name in exc_names:
            if exc_name is None:
                continue
            exc_class = __builtins__[exc_name]
            if not issubclass(exc_class, Exception):
                raise ValueError('%s can be a Python exception name or null '
                                 '(in advanced_settings.continue_execution_on)'
                                 % exc_name)
            exceptions.append(exc_class)

    except Exception as exc:
        raise BadArgument('config', exc)

    # NOTE: contrarily to the download routine, we cannot check the types of
    # the config because no parameter is mandatory, and thus they might NOT be
    # present in the config.

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
    segment_selection = config.get('segment_selection', config.get('segment_select', {}))

    return session, pyfunc, funcname, config, segment_selection, chunksize,\
        tuple(exceptions)
