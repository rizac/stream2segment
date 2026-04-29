"""
Input validation for the download routine
"""

import os
import re
from collections.abc import Sequence
from datetime import datetime, timedelta, UTC, date
from os.path import isabs, abspath, join, dirname, isfile
from typing import Any

from stream2segment.download.modules.utils import fdsn_url
from stream2segment.download.modules.events import EVENTWS_MAPPING
from stream2segment.io import yaml_load
from stream2segment.io.cli import BadParam
from stream2segment.io.db import get_engine
from stream2segment.resources import get_templates_fpath, get_ttable_fpath
from stream2segment.traveltimes.ttloader import TTTable


def extract_download_args(config: dict, config_file_path: str) -> dict:
    """Load download arguments from the given config (yaml file or dict) after
    parsing and checking some of the dict keys.

    :return: a dict loaded from the given `config` and with parsed arguments
        (dict keys)

    Raise `BadParam` in case of parsing errors, missing arguments,
    conflicts and so on
    """

    kwargs = {}
    params = tuple()

    try:
        params = ('data_url', 'dataws')  # decalre explicitly (see Except below)
        # validate dataws FIRST because it is used by other params later
        val = pop_param(params, config)
        if isinstance(val, str):  # backward compatibility
            val = [val]
        kwargs['data_url'] = [valid_fdsn(url, is_eventws=False) for url in val]

        params = ('events_url', 'eventws')
        val = pop_param(params, config)
        if isinstance(val, str):  # backward compatibility
            val = [val]
        kwargs['events_url'] = [
            valid_fdsn(u, is_eventws=True, configfile=config_file_path) for u in val
        ]

        params = 'search_radius'
        kwargs['search_radius'] = valid_search_radius(pop_param(params, config))

        params = 'min_sample_rate'
        kwargs['min_sample_rate'] = int(pop_param(params, config, default=0))

        # parameters whose validation changes completely their type and should
        # return separately from the new config dict:

        params = ('credentials', 'restricted_data')
        kwargs['credentials'] = valid_credentials(
            pop_param(params, config),
            dataws=kwargs['data_url'],
            configfile=config_file_path
        )

        params = 'dburl'
        kwargs['engine'] = get_engine(pop_param(params, config))

        params = ('starttime', 'start')
        kwargs['start'] = valid_date(pop_param(params, config))

        params = ('endtime', 'end')
        kwargs['end'] = valid_date(pop_param(params, config))

        params = ('network', 'net', 'networks')
        kwargs['network'] = valid_nslc(pop_param(params, config, default=[]))

        params = ('station', 'sta', 'stations')
        kwargs['station'] = valid_nslc(pop_param(params, config, default=[]))

        params = ('location', 'loc', 'locations')
        kwargs['location'] = valid_nslc(pop_param(params, config, default=[]))

        params = ('channel', 'cha', 'channels')
        kwargs['channel'] = valid_nslc(pop_param(params, config, default=[]))

        params = ('time_window', 'segment_window', 'timespan')
        is_timespan = 'timespan' in config
        val = pop_param(params, config)
        if is_timespan:
            # a positive timespan[0] is now the same as a negative time_window[0]:
            val[0] = -float(val[0])
        kwargs['time_window'] = [float(val[0]), float(val[1])]

        params = ('stationxml', 'inventory')
        kwargs['stationxml'] = bool(pop_param(params, config))

        params = ('quakeml',)
        kwargs['quakeml'] = bool(pop_param(params, config, default=False))

        # validate advanced_settings:
        params = 'advanced_settings'
        # old configs had traveltimes_model as top-level param (now in advanced_settings):
        val = config.pop('traveltimes_model', None)
        if val is not None:
            config[params]['traveltimes_model'] = val
        kwargs['advanced_settings'] = _validate_download_advanced_settings(
            pop_param(params, config)
        )

        # Validate eventws (event web service) params. These parameters can be supplied
        # in the main config but also in the eventws_params dict, which was formerly
        # named eventws_query_args:

        pnames = ('events_extra_params', 'eventws_params', 'eventws_query_args')
        val = pop_param(params, config, default={})


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
        legacy_keys = {
            'retry_client_err', 'retry_mseed_err', 'retry_seg_not_found',
            'retry_server_err', 'retry_timespan_err', 'retry_url_err', 'update_metadata'
        }
        if unknown_keys - legacy_keys:
            raise BadParam(BadParam.P_UNKNOWN, unknown_keys - legacy_keys)

        return kwargs

    except (Exception, ) as err:
        if not isinstance(params, str):
            params = " / ".join(params)
        raise BadParam(f'{params}: {err}')


def pop_param(param: str | Sequence[str], cfg: dict, default: Any=None):
    """
    Pop the given `param` from the dict `cfg` and return the associated value.
    Raises if:
     - param is not found and `default` is missing or None (otherwise, return `default`)
     - param is a Sequence of strings and more than one string is a key of `cfg` (all
       strings are popped from `cfg` before raising)
    """
    params = param
    if isinstance(param, str):
        params = [param]
    params = set(params) & set(cfg.keys())
    if len(params) > 1:
        for p in params:
            cfg.pop(p)
        raise ValueError(f"conflicting parameters, please use {params[0]}")
    elif len(params) == 0:
        if default is not None:
            return default
        raise ValueError(f"parameter not found")
    return cfg.pop(list(params)[0])


def _validate_event_params(config, extra_event_params):
    """pop / move event params from the given config (`dict`) into a new dict and return
    the new dict. Raise :class:`BadParam` if any event parameter is invalid
    """
    # returned dict:
    evt_params = {}

    pnames = ['minlatitude', 'minlat']
    pname, pval = pop_param(pnames, config)
    if pval is not None:
        new_pname = pnames[0].split('.')[-1]  # remove prefix, if any
        evt_params[new_pname] = validate_param(pname, pval,
                                               valid_between, -90.0, 90.0)

    pnames = ['maxlatitude', 'maxlat'],
    pname, pval = pop_param(config, pnames, None)
    if pval is not None:
        new_pname = pnames[0].split('.')[-1]  # remove prefix, if any
        evt_params[new_pname] = validate_param(pname, pval,
                                               valid_between, -90.0, 90.0)

    pnames = ['minlongitude', 'minlon'],
    pname, pval = pop_param(config, pnames, None)
    if pval is not None:
        new_pname = pnames[0].split('.')[-1]  # remove prefix, if any
        evt_params[new_pname] = validate_param(pname, pval,
                                               valid_between, -180.0, 180.0)

    pnames = ['maxlongitude', 'maxlon'],
    pname, pval = pop_param(config, pnames, None)
    if pval is not None:
        new_pname = pnames[0].split('.')[-1]  # remove prefix, if any
        evt_params[new_pname] = validate_param(pname, pval,
                                               valid_between, -180.0, 180.0)

    pnames = ['minmagnitude', 'minmag'],
    pname, pval = pop_param(config, pnames, None)
    if pval is not None:
        newp_name = pnames[0].split('.')[-1]  # remove prefix, if any
        evt_params[newp_name] = validate_param(pname, pval, float)

    pnames = ['maxmagnitude', 'maxmag'],
    pname, pval = pop_param(config, pnames, None)
    if pval is not None:
        newp_name = pnames[0].split('.')[-1]  # remove prefix, if any
        evt_params[newp_name] = validate_param(pname, pval, float)

    pnames = ['mindepth'],
    pname, pval = pop_param(config, pnames, None)
    if pval is not None:
        newp_name = pnames[0].split('.')[-1]  # remove prefix, if any
        evt_params[newp_name] = validate_param(pname, pval, float)

    pnames = ['maxdepth'],
    pname, pval = pop_param(config, pnames, None)
    if pval is not None:
        newp_name = pnames[0].split('.')[-1]  # remove prefix, if any
        evt_params[newp_name] = validate_param(pname, pval, float)

    return evt_params


def _validate_download_advanced_settings(adv_settings: dict):
    """
    Validate the advanced settings of the given download config, returning a new dict
    """
    advanced_settings = dict(adv_settings)
    pname = ''

    try:
        pname = 'traveltimes_model'
        advanced_settings[pname] = valid_tt_table(adv_settings[pname])

        pname = 'download_blocksize'
        val = int(adv_settings[pname])
        advanced_settings[pname] = val if val > 0 else -1

        pname = 'db_buf_size'
        val = int(adv_settings[pname])
        assert val > 0
        advanced_settings[pname] = val

        pname = 'routing_service_url'
        advanced_settings[pname] = adv_settings[pname]
        if not isinstance(advanced_settings[pname], (list, tuple)):
            advanced_settings[pname] = [advanced_settings[pname]]

        pname = 'max_concurrent_downloads'
        val = adv_settings.get(pname, adv_settings['max_thread_workers'])
        if val is None:
            val = 1
        assert val > 0
        advanced_settings[pname] = val

        return advanced_settings

    except Exception as e:
        raise ValueError(f'error in {pname}: {e}')


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


def valid_credentials(
    credentials, dataws, configfile=None
) -> tuple[str, str] | bytes | None:
    """Create an :class:`stream2segment.download.utils.Authorizer`  # FIXME DOCSTRING!
    (handling authentication/authorization) from the given restricted_data

    :param credentials: either file path, to token, token data in bytes, or
        tuple (user, password). If None, or the empty string, None is returned
    """
    if credentials in ('', None, b''):
        return None

    if len(dataws) != 1:
        raise ValueError(
            'downloading restricted data requires a single URL in `dataws`'
        )

    if isinstance(credentials, (tuple, list)):
        if len(credentials) != 2 or not all(isinstance(_, str) for _ in credentials):
            raise ValueError(
                'provide username and password as list/tuple of two strings'
            )
        if dataws.lower() == 'eida':
            raise ValueError(
                'downloading from EIDA requires a token, not username and password'
            )
        return str(credentials[0]), str(credentials[1])

    if isinstance(credentials, str):
        token_path = credentials
        if not isfile(token_path) and not isabs(token_path) and configfile is not None:
            token_path = abspath(join(dirname(configfile), token_path))

        if isfile(token_path):
            with open(token_path, 'rb') as fhd:
                credentials = fhd.read()

        if not re.search(
            pattern=rb'\bBEGIN PGP\b', string=credentials, flags=re.IGNORECASE
        ):
            raise ValueError(
                "Invalid token. If you passed a file path, "
                "check that the file is a valid token"
            )
        return credentials

    raise ValueError(f'Invalid "restricted data" parameter: {credentials}')
    #
    # if isinstance(credentials, str) and configfile is not None:
    #     if not isabs(credentials):
    #         credentials = abspath(join(dirname(configfile), credentials))
    #
    #
    #
    #
    # ret = Authorizer(credentials)
    #
    # # check dataws is single element list:
    # dataws = dataws[0]
    # # Here we have 4 cases:
    # # 1 'eida' + token: OK
    # # 2. Any other fdsn + username & password: OK
    # # 3. eida + username & password: BAD. raise ValueError
    # # 4. Any other fdsn + token: OK (we might have provided a single eida
    # #                                datacenter in which case it's fine)
    # if dataws.lower() == 'eida' and ret.userpass:
    #     raise ValueError('downloading from EIDA requires a token, '
    #                      'not username and password')
    # return ret


def valid_tt_table(file_or_name):
    """Load the given TTTable object from the given file path or name. If name
    (string) it must match any of the builtin TTTable .npz files defined in
    this package. Raise TypeError or any Exception that TTTable might raise
    (including when the file is not found)
    """
    if not isinstance(file_or_name, str):
        raise TypeError('string required, not %s' % str(type(file_or_name)))
    filepath = get_ttable_fpath(file_or_name)
    if not os.path.isfile(filepath):
        filepath = file_or_name
    if not os.path.isfile(filepath):
        raise Exception('file or builtin model name not found')
    return TTTable(filepath)


def valid_date(obj):
    dtime = None
    try:
        dtime = datetime.fromisoformat(obj)  # if obj is datetime, returns obj
    except Exception as _:
        try:
            import dateutil.parser
            # https://stackoverflow.com/a/15228038
            try:
                dtime = dateutil.parser.isoparse(obj)
            except (TypeError, ValueError):
                try:
                    days = int(obj)
                    dtime = datetime.now(UTC).replace(
                        hour=0, minute=0, second=0, microsecond=0,
                    ) + timedelta(days=days)
                except (TypeError, ValueError):
                    pass
        except ImportError:
            pass

    if isinstance(dtime, date):
        dtime = datetime(year=dtime.year, month=dtime.month, day=dtime.day)

    if not isinstance(dtime, datetime):
        raise TypeError(("iso-formatted datetime string, datetime or date "
                         "object, non-positive int required, found %s") %
                        str(type(obj)))

    # convert datetime in UTC, and remove tzinfo (so no 'Z' in its string repr):
    return dtime.astimezone(UTC).replace(tzinfo=None)


def valid_fdsn(url, is_eventws, configfile=None):
    """Return url if it matches a FDSN service (valid strings are 'eida' and
    'iris'), raise ValueError or TypeError otherwise
    """
    if not isinstance(url, str):
        raise TypeError('string required')

    if (is_eventws and url.lower() in EVENTWS_MAPPING) or \
            (not is_eventws and url.lower() in ('eida', 'iris')):
        return url.lower()

    if is_eventws:
        if configfile is None:
            fpath = url
        else:
            fpath = abspath(join(dirname(configfile), url))
        if os.path.isfile(fpath):
            return fpath
        else:
            raise ValueError('Invalid file path, check typos')
    try:
        return fdsn_url(url)
    except Exception:
        raise ValueError('Invalid FDSN url or file path, check typos')


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