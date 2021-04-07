"""
Input validation for the download routine
"""

import os
from datetime import datetime, timedelta

from future.utils import string_types

from stream2segment.download.modules.utils import (EVENTWS_SAFE_PARAMS, Authorizer,
                                                   strptime, EVENTWS_MAPPING)
from stream2segment.io import yaml_load, absrelpath, Fdsnws
from stream2segment.io.db import DbNotFound, get_session, is_sqlite
from stream2segment.io.inputvalidation import (validate_param, pop_param,
                                               get_param, BadParam, valid_between)
from stream2segment.resources import get_templates_fpath, get_ttable_fpath
from stream2segment.traveltimes.ttloader import TTTable


def load_config_for_download(config, validate, **param_overrides):
    """Load download arguments from the given config (yaml file or dict) after
    parsing and checking some of the dict keys.

    :return: a dict loaded from the given `config` and with parsed arguments
        (dict keys)

    Raise `BadParam` in case of parsing errors, missing arguments,
    conflicts and so on
    """
    config_dict = validate_param("config", config, yaml_load, **param_overrides)

    if not validate:
        return config_dict

    # few variables:
    configfile = None
    if isinstance(config, string_types) and os.path.isfile(config):
        configfile = config

    # =====================
    # Parameters validation
    # =====================

    # put validated params into a new dict, which will be eventually returned:
    old_config, new_config = config_dict, {}
    validated_params = set()

    # validate dataws FIRST because it is used by other params later
    pname, pval = pop_param(old_config, 'dataws')
    validated_params.add(pname)
    if isinstance(pval, string_types):  # backward compatibility
        pval = [pval]
    dataws = validate_param(pname, pval,
                            lambda urls: [valid_fdsn(url, is_eventws=False)
                                          for url in urls])
    new_config[pname] = dataws

    pname, pval = pop_param(old_config, 'update_metadata')
    validated_params.add(pname)
    new_config[pname] = validate_param(pname, pval, valid_updatemetadata_param)

    pname, pval = pop_param(old_config, 'eventws')
    validated_params.add(pname)
    new_config[pname] = validate_param(pname, pval, valid_fdsn,
                                       is_eventws=True, configfile=configfile)

    pname, pval = pop_param(old_config, 'search_radius')
    validated_params.add(pname)
    new_config[pname] = validate_param(pname, pval, valid_search_radius)

    pname, pval = pop_param(old_config, 'min_sample_rate', default=0)
    validated_params.add(pname)
    new_config[pname] = validate_param(pname, pval, int)

    # parameters whose validation changes completely their type and should
    # returned separately from the new confg dict:

    pname, pval = pop_param(old_config, 'restricted_data')
    validated_params.add(pname)
    authorizer = validate_param(pname, pval, valid_authorizer, dataws, configfile)

    pname, pval = pop_param(old_config, 'dburl')
    validated_params.add(pname)
    session = validate_param(pname, pval, valid_session)

    pname, pval = pop_param(old_config, 'traveltimes_model')
    validated_params.add(pname)
    tt_table = validate_param(pname, pval,valid_tt_table)

    # parameters with multiple allowed names (use get_param_tuple to get which
    # param name is implemented among the allowed ones)

    pnames = ('starttime', 'start')
    validated_params.update(pnames)
    pname, pval = pop_param(old_config, pnames)
    new_config[pnames[0]] = validate_param(pname, pval, valid_date)

    pnames = ('endtime', 'end')
    validated_params.update(pnames)
    pname, pval = pop_param(old_config, pnames)
    new_config[pnames[0]] = validate_param(pname, pval, valid_date)

    pnames = ('network', 'net', 'networks')
    validated_params.update(pnames)
    pname, pval = pop_param(old_config, pnames, default=[])
    new_config[pnames[0]] = validate_param(pname, pval, valid_nslc)

    pnames = ('station', 'sta', 'stations')
    validated_params.update(pnames)
    pname, pval = pop_param(old_config, pnames, default=[])
    new_config[pnames[0]] = validate_param(pname, pval, valid_nslc)

    pnames = ('location', 'loc', 'locations')
    validated_params.update(pnames)
    pname, pval = pop_param(old_config, pnames, default=[])
    new_config[pnames[0]] = validate_param(pname, pval, valid_nslc)

    pnames = ('channel', 'cha', 'channels')
    validated_params.update(pnames)
    pname, pval = pop_param(old_config, pnames, default=[])
    new_config[pnames[0]] = validate_param(pname, pval, valid_nslc)

    # validate advanced_settings:
    pname = 'advanced_settings'
    validated_params.add(pname)
    # Call get_param with no default just for raising if param is missing:
    new_config[pname] = pop_param(old_config, pname)[1]
    _validate_download_advanced_settings(new_config, pname)

    # Validate eventws (event web service) params. These parameters can be supplied
    # in the main config but also in the eventws_params dict, which was formerly
    # named eventws_query_args:

    pnames = ('eventws_params', 'eventws_query_args')
    validated_params.update(pnames)
    # get eventws_param dict:
    pname, evt_params = get_param(old_config, pnames, default=None)
    # validate it (null is allowed and should be converted to {}):
    evt_params = validate_param(pnames, evt_params or {}, valid_type, {})
    if evt_params:
        # evt_params is not empty, validate some parameters:
        evt_params.update(_pop_event_params(old_config, pname))
    # Now remove evt_params dict from the old config:
    pop_param(old_config, pnames, default=None)
    # Ok, now do the same above on the main config:
    evt_params2 = _pop_event_params(old_config)
    # Now merge (but check conflicts beforehand):
    _conflicts = set(evt_params) & set(evt_params2)
    if _conflicts:
        # Issue a general warning (printing all conflicting params might be too verbose,
        # and we should print them as they were input,e.g. minlatitude or minlat?)
        raise BadParam(BadParam.P_CONFLICT.replace('names', 'name(s)'), _conflicts,
                       message='parameter(s) can be provided globally or in "%s", '
                               'not in both' % pname)
    evt_params.update(evt_params2)  # merge
    # now put evt_params into new_config:
    new_config[pnames[0]] = evt_params

    # =========================================================
    # Done with parameter validation. Just perform final checks
    # =========================================================

    # load original config (default in this package) to perform some checks:
    orig_config = yaml_load(get_templates_fpath("download.yaml"))

    unknown_keys = set(old_config) - set(orig_config)
    if unknown_keys:
        raise BadParam(BadParam.P_UNKNOWN, unknown_keys)

    # Now check for params supplied here NOT in the default config, and supplied
    # here but with different type in the original config
    validated_params.update(old_config)
    for pname in list(old_config.keys()):
        pval = old_config.pop(pname)
        new_config[pname] = validate_param(pname, pval,
                                           valid_type, orig_config[pname])

    # And finally, check for params in the default config not supplied here:
    missing_keys = set(orig_config) - validated_params - set(EVENTWS_SAFE_PARAMS)
    if missing_keys:
        raise BadParam(BadParam.P_MISSING, missing_keys)

    return new_config, session, authorizer, tt_table


def _pop_event_params(config, prefix=None):
    """pop / move event params from the given config (`dict`) into a new dict and return
    the new dict. Raise :class:`BadParam` if any event parameter is invalid
    """
    # define first default event params in order to avoid typos
    def_evt_params = EVENTWS_SAFE_PARAMS

    if prefix:
        def_evt_params = [prefix + '.' + _ for _ in def_evt_params]

    # returned dict:
    evt_params = {}

    pnames = def_evt_params[:2]  # ['minlatitude', 'minlat']
    pname, pval = pop_param(config, pnames, None)
    if pval is not None:
        new_pname = pnames[0].split('.')[-1]  # remove prefix, if any
        evt_params[new_pname] = validate_param(pname, pval,
                                               valid_between, -90.0, 90.0)

    pnames = def_evt_params[2:4]  # ['maxlatitude', 'maxlat'],
    pname, pval = pop_param(config, pnames, None)
    if pval is not None:
        new_pname = pnames[0].split('.')[-1]  # remove prefix, if any
        evt_params[new_pname] = validate_param(pname, pval,
                                               valid_between, -90.0, 90.0)

    pnames = def_evt_params[4:6]  # ['minlongitude', 'minlon'],
    pname, pval = pop_param(config, pnames, None)
    if pval is not None:
        new_pname = pnames[0].split('.')[-1]  # remove prefix, if any
        evt_params[new_pname] = validate_param(pname, pval,
                                               valid_between, -180.0, 180.0)

    pnames = def_evt_params[6:8]  # ['maxlongitude', 'maxlon'],
    pname, pval = pop_param(config, pnames, None)
    if pval is not None:
        new_pname = pnames[0].split('.')[-1]  # remove prefix, if any
        evt_params[new_pname] = validate_param(pname, pval,
                                               valid_between, -180.0, 180.0)

    pnames = def_evt_params[8:10]  # ['minmagnitude', 'minmag'],
    pname, pval = pop_param(config, pnames, None)
    if pval is not None:
        newp_name = pnames[0].split('.')[-1]  # remove prefix, if any
        evt_params[newp_name] = validate_param(pname, pval, float)

    pnames = def_evt_params[10:12]  # ['maxmagnitude', 'maxmag'],
    pname, pval = pop_param(config, pnames, None)
    if pval is not None:
        newp_name = pnames[0].split('.')[-1]  # remove prefix, if any
        evt_params[newp_name] = validate_param(pname, pval, float)

    pnames = def_evt_params[12:13]  # ['mindepth'],
    pname, pval = pop_param(config, pnames, None)
    if pval is not None:
        newp_name = pnames[0].split('.')[-1]  # remove prefix, if any
        evt_params[newp_name] = validate_param(pname, pval, float)

    pnames = def_evt_params[13:14]  # ['maxdepth'],
    pname, pval = pop_param(config, pnames, None)
    if pval is not None:
        newp_name = pnames[0].split('.')[-1]  # remove prefix, if any
        evt_params[newp_name] = validate_param(pname, pval, float)

    return evt_params


def _validate_download_advanced_settings(config, adv_settings_key):
    """Validate the advanced settings of the given download config. Modifies
     config.advanced_settings` keys inplace (do not return the validated `dict`)
     """
    adv_settings_dict = config[adv_settings_key]
    prefix = adv_settings_key + '.'  # 'advanced_settings'

    pname = 'download_blocksize'
    pval = validate_param(prefix + pname, pop_param(config, prefix + pname)[1],
                          valid_type, int)
    adv_settings_dict[pname] = pval if pval > 0 else -1

    pname = 'db_buf_size'
    adv_settings_dict[pname] = validate_param(prefix + pname,
                                              pop_param(config, prefix + pname)[1],
                                              valid_between, 1, None,
                                              pass_if_none=False)

    pnames = [prefix + _ for _ in ('max_concurrent_downloads', 'max_thread_workers')]
    pname, pval = pop_param(config, pnames)
    if pname == pnames[1]:
        # When max_thread_workers<0, it defaulted to None:
        pval = validate_param(pname, pval, valid_type, int, None)
        if pval <= 0:
            pval = None
    else:
        pval = validate_param(pname, pval, valid_between, 1, None, pass_if_none=True)
    adv_settings_dict[pnames[0].split('.')[-1]] = pval


def valid_session(dburl, scoped=False, **engine_kwargs):
    try:
        sess = get_session(dburl, scoped, check_db_existence=not is_sqlite(dburl),
                           **engine_kwargs)
    except DbNotFound as dbnf:
        raise ValueError('%s, it needs to be created first' % str(dbnf))

    # Note: this creates the SCHEMA, not the database
    # the import below is in the function because slightly time consuming:
    from stream2segment.download.db.models import Base
    try:
        Base.metadata.create_all(sess.get_bind())
    except Exception as exc:
        raise ValueError('Error creating tables. Possible reason: tables created '
                         'with an older version or with a different program '
                         '(original error: %s)' % str(exc))
    return sess


def valid_type(value, *other_values_or_types):
    """Return value if it is of the same type (same class, or subclass) of *any*
    other value type (at least one). Raises TypeError otherwise.

    :param value: a python object
    :param other_values_or_types: Python objects or classes. In the first case,
        it will compare the value class vs. the object class.

    :return: value
    """
    value_type = value.__class__
    other_types = tuple(_ if isinstance(_, type) else _.__class__
                        for _ in other_values_or_types)

    if issubclass(value_type, other_types):
        return value

    raise TypeError("%s expected, found %s" %
                    (" or ".join(str(_) for _ in other_types), str(value_type)))


def valid_nslc(value):
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


def valid_authorizer(restricted_data, dataws, configfile=None):
    """Create an :class:`stream2segment.download.utils.Authorizer`
    (handling authentication/authorization) from the given restricted_data

    :param restricted_data: either file path, to token, token data in bytes, or
        tuple (user, password). If None, or the empty string, None is returned
    """
    if restricted_data in ('', None, b''):
        restricted_data = None
    elif isinstance(restricted_data, string_types) and configfile is not None:
        restricted_data = absrelpath(restricted_data, configfile)
    ret = Authorizer(restricted_data)
    # check dataws is single element list:
    if (ret.token or ret.userpass) and len(dataws) != 1:
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


def valid_updatemetadata_param(value):
    """Parse parse_update_metadata returning True, False or 'only'"""
    val_str = str(value).lower()
    if val_str == 'true':
        return True
    if val_str == 'false':
        return False
    if val_str == 'only':
        return val_str
    raise ValueError('value can be true, false or only, %s provided' % val_str)


def valid_tt_table(file_or_name):
    """Load the given TTTable object from the given file path or name. If name
    (string) it must match any of the builtin TTTable .npz files defined in
    this package. Raise TypeError or any Exception that TTTable might raise
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
    'iris'), raise ValueError or TypeError otherwise
    """
    if not isinstance(url, string_types):
        raise TypeError('string required')

    if (is_eventws and url.lower() in EVENTWS_MAPPING) or \
            (not is_eventws and url.lower() in ('eida', 'iris')):
        return url.lower()

    if is_eventws:
        fpath = url if configfile is None else absrelpath(url, configfile)
        if os.path.isfile(fpath):
            return fpath
        try:
            return Fdsnws(url).url()
        except Exception:
            raise ValueError('Invalid FDSN url or file path, check typos')

    return Fdsnws(url).url()


def valid_search_radius(search_radius):
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