"""
Input validation for the download routine
"""

import os
import re
from collections.abc import Sequence
from datetime import datetime, timedelta, UTC, date
from os.path import isabs, abspath, join, dirname, isfile
from pathlib import Path
from typing import Any

import yaml
from numpy import inf
from stream2segment.download.utils import fdsn_url
from stream2segment.download.events import EVENTWS_MAPPING
from stream2segment.io.utils import BadParam
from stream2segment.io.db import (
    create_engine, resolve_db_path, s2s_db_version, is_sqlite
)
from stream2segment.resources import get_ttable_fpath
from stream2segment.traveltimes.ttloader import TTTable


def load_input(
    config_file_path: str, **override_params
) -> tuple[dict, dict]:
    """
    Load config for download, returning the tuple
    (config, kwargs)
    denoting the clean config and a kwargs dict for the
    download routine. Raise BadParm
    """
    kwargs = {}
    config = {}

    params = ""

    try:
        params = 'configuration file'
        with open(config_file_path, 'r') as stream:
            config = yaml.safe_load(stream)

        params = ('data_url', 'dataws')  # declare explicitly (see Except below)
        # validate dataws FIRST because it is used by other params later
        val = get_param(params, config, override_params)
        if isinstance(val, str):  # backward compatibility
            val = [val]
        kwargs['data_url'] = [valid_fdsn(url, is_eventws=False) for url in val]

        params = ('events_url', 'eventws', 'event_url')
        val = get_param(params, config, override_params)
        if isinstance(val, str):  # backward compatibility
            val = [val]
        kwargs['events_url'] = [
            valid_fdsn(u, is_eventws=True, configfile=config_file_path) for u in val
        ]

        params = ('search_radius',)
        kwargs['search_radius'] = valid_search_radius(
            get_param(params, config, override_params)
        )

        params = ('min_sample_rate',)
        kwargs['min_sample_rate'] = int(
            get_param(params, config, override_params, default=0)
        )

        # parameters whose validation changes completely their type and should
        # return separately from the new config dict:

        params = ('credentials', 'restricted_data')
        kwargs['credentials'] = valid_credentials(
            get_param(params, config, override_params),
            dataws=kwargs['data_url'],
            configfile=config_file_path
        )

        params = ('db', 'dburl', 'db_url')
        val = get_param(params, config, override_params)
        kwargs['engine'] = get_engine(
            resolve_db_path(val, dirname(config_file_path))
        )

        params = ('starttime', 'start')
        kwargs['start'] = valid_date(get_param(params, config, override_params))

        params = ('endtime', 'end')
        kwargs['end'] = valid_date(get_param(params, config, override_params))

        params = ('network', 'net', 'networks')
        kwargs['network'] = valid_nslc(
            get_param(params, config, override_params, default=[])
        )

        params = ('station', 'sta', 'stations')
        kwargs['station'] = valid_nslc(
            get_param(params, config, override_params, default=[])
        )

        params = ('location', 'loc', 'locations')
        kwargs['location'] = valid_nslc(
            get_param(params, config, override_params, default=[])
        )

        params = ('channel', 'cha', 'channels')
        kwargs['channel'] = valid_nslc(
            get_param(params, config, override_params, default=[])
        )

        params = ('time_window', 'segment_window', 'timespan')
        is_timespan = 'timespan' in config
        val = get_param(params, config, override_params)
        if is_timespan:
            # a positive timespan[0] is now the same as a negative time_window[0]:
            val[0] = -float(val[0])
        kwargs['time_window'] = [float(val[0]), float(val[1])]

        params = ('stationxml', 'inventory')
        kwargs['stationxml'] = bool(get_param(params, config, override_params))

        params = ('quakeml',)
        kwargs['quakeml'] = bool(
            get_param(params, config, override_params, default=False)
        )

        params = ('advanced_settings',)
        # old configs had traveltimes_model as top-level param (now in advanced_settings):
        val = config.pop('traveltimes_model', None)
        if val is not None:
            config[params[0]]['traveltimes_model'] = val
        kwargs['advanced_settings'] = _validate_download_advanced_settings(
            get_param(params, config, override_params)
        )

        # event parameters:

        params = ('events_extra_params', 'eventws_params', 'eventws_query_args')
        evt_params = get_param(params, config, override_params, default={})
        if evt_params is None:
            evt_params = {}
        invalid = set(evt_params.keys()) & set(config)
        if invalid:
            raise ValueError(f"invalid duplicated parameter(s): {', '.join(invalid)}")
        kwargs['event_params'] = evt_params

        params = ('minlatitude', 'minlat')
        val = get_param(params, config, override_params)
        if val is not None:
            assert -90.0 <= float(val) <= 90.0, "not in [-90, 90]"
            evt_params[params[0]] = float(val)

        params = ('maxlatitude', 'maxlat')
        val = get_param(params, config, override_params)
        if val is not None:
            assert -90.0 <= float(val) <= 90.0, "not in [-90, 90]"
            evt_params[params[0]] = float(val)

        assert (
            evt_params.get('minlatitude', evt_params.get('minlat', -inf)) <
            evt_params.get('maxlatitude', evt_params.get('maxlat', inf))
        ), "the value must be > minlatitude / minlat"

        params = ('minlongitude', 'minlon')
        val = get_param(params, config, override_params)
        if val is not None:
            assert -180.0 <= float(val) <= 180.0, "not in [-180, 180]"
            evt_params[params[0]] = float(val)

        params = ('maxlongitude', 'maxlon')
        val = get_param(params, config, override_params)
        if val is not None:
            assert -180.0 <= float(val) <= 180.0, "not in [-180, 180]"
            evt_params[params[0]] = float(val)

        assert (
            evt_params.get('minlongitude', evt_params.get('minlon', -inf)) <
            evt_params.get('maxlongitude', evt_params.get('maxlon', inf))
        ), "the value must be > minlongitude / minlon"

        params = ('minmagnitude', 'minmag')
        val = get_param(params, config, override_params)
        if val is not None:
            evt_params[params[0]] = float(val)

        params = ('maxmagnitude', 'maxmag')
        val = get_param(params, config, override_params)
        if val is not None:
            evt_params[params[0]] = float(val)

        assert (
            evt_params.get('minmagnitude', evt_params.get('minmag', -inf)) <
            evt_params.get('maxmagnitude', evt_params.get('maxmag', inf))
        ), "the value must be > minmagnitude / minmag"

        params = ('mindepth',)
        val = get_param(params, config, override_params)
        if val is not None:
            evt_params[params[0]] = float(val)

        params = ('maxdepth',)
        val = get_param(params, config, override_params)
        if val is not None:
            evt_params[params[0]] = float(val)

        assert (
            evt_params.get('mindepth', -inf) < evt_params.get('maxdepth', inf)
        ), "the value must be > mindepth"


        # =========================================================
        # Done with parameter validation. Just perform final checks
        # =========================================================
        # params = ("Unknown config. parameter(s)",)  # just for the error msg (see below)
        legacy_keys = {
            'retry_client_err', 'retry_mseed_err', 'retry_seg_not_found',
            'retry_server_err', 'retry_timespan_err', 'retry_url_err', 'update_metadata'
        }
        # invalid = set(config.keys()) - legacy_keys
        # if invalid:
        #     raise ValueError(', '.join(invalid))

        # remove from original config
        for p in legacy_keys:
            config.pop(p, None)

        return config, kwargs

    except (Exception, ) as err:
        raise BadParam(f'{params[0]}: {err}') from None


def get_param(
    params: Sequence[str],
    cfg: dict,
    override_params: dict,
    *,
    default: Any=None
):
    """
    Get the value of the given param from either `cfg` or `override_params`, raising
    in case of conflicts or missing parameter. The returned value is assured to be in
    `cfg`, keyed by its primary name (`param[0]`). As such, `cfg` might be modified
    inplace
    """
    param = params[0]

    p1 = [p for p in params if p in cfg]
    p2 = [p for p in params if p in override_params]

    if len(p1) == len(p2) == 0:
        if default is None:
            raise ValueError(f"parameter not found")
        val = default
    elif len(p1) > 1 or len(p2) > 1:
        raise ValueError(
            f"names conflict ({', '.join(p1 + p2)}), please use {repr(param)}"
        )

        # if len(p2) == 1:  # FIXME REMOVE
        #     val = override_params[p2[0]]
        #     if p1:
        #         cfg.pop(p1[0])
        #     cfg[param] = val
        # else:
        #     val = cfg[p1[0]]
        #     if p1[0] != param:
        #         cfg.pop(p1[0])
        #         cfg[param] = val
    elif p2:
        val = override_params[p2[0]]
    else:
        p = p1[0]
        val = cfg[p]
        if p != param:
            cfg.pop(p, None)

    cfg[param] = val
    return val


def _validate_download_advanced_settings(adv_settings: dict):
    """
    Validate the advanced settings of the given download config, returning a new dict
    """
    advanced_settings = dict(adv_settings)
    pname = ''

    try:
        pname = 'traveltimes_model'
        advanced_settings[pname] = valid_tt_table(adv_settings[pname])

        # this param is now settable but hidden (keep config simple):
        pname = 'download_blocksize'
        val = int(adv_settings.pop(pname, 1024 * 1024))
        advanced_settings[pname] = val if val > 0 else -1

        # this param is not settable anymore and set automatically (keep config simple):
        pname = 'db_buf_size'
        adv_settings.pop(pname, -1)
        # advanced_settings[pname] = int(val)

        pname = 'routing_service_url'
        advanced_settings[pname] = adv_settings[pname]
        if not isinstance(advanced_settings[pname], (list, tuple)):
            advanced_settings[pname] = [advanced_settings[pname]]

        pname = 'max_concurrent_downloads'
        val = adv_settings.get(pname, adv_settings.get('max_thread_workers'))
        assert val is None or val > 0
        advanced_settings[pname] = val

        pname = 'text_download_timeout'
        val = adv_settings.get(
            pname,
            adv_settings.get(
                'e_timeout', adv_settings.get('s_timeout')
            )
        )
        assert val > 0
        advanced_settings[pname] = val

        pname = 'data_download_timeout'
        val = adv_settings.get(
            pname,
            adv_settings.get(
                'w_timeout', adv_settings.get('i_timeout')
            )
        )
        assert val > 0
        advanced_settings[pname] = val

        pname = 'event_overlap_tolerance'
        val = adv_settings.get(
            pname, {
                'lat':5, 'lon': 5, 'time': 10, 'depth': 5
            }
        )
        advanced_settings[pname] = val

        pname = 'on_event_conflict'
        advanced_settings[pname] = adv_settings.get(pname,"keep")

        return advanced_settings

    except Exception as e:
        raise ValueError(f'error in {pname}: {e}')


def get_engine(db_url):
    engine = create_engine(db_url, check_db_existence=not is_sqlite(db_url))
    if s2s_db_version(engine) < 5:
        raise ValueError(
            'Invalid DB schema (possible cause: trying to save on an old s2s DB)'
        )
    from stream2segment.io.db.models import Base
    Base.metadata.create_all(engine)
    return engine


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
            for chunk in string.split(","):
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

    if isinstance(credentials, (str, Path)):
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

    raise ValueError(f'unable to read file path "{credentials}"')
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


def valid_date(obj) -> datetime:
    dtime = obj
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

    if not isinstance(dtime, datetime):
        if isinstance(dtime, date):
            dtime = datetime(year=dtime.year, month=dtime.month, day=dtime.day)

    if not isinstance(dtime, datetime):
        raise TypeError(("iso-formatted datetime string, datetime or date "
                         "object, non-positive int required, found %s") %
                        str(type(obj)))

    # convert datetime in UTC, and remove tzinfo
    # (so no 'Z' or "+00:00" in its ISO repr):
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

    try:
        return fdsn_url(url)
    except Exception:
        if is_eventws:
            if configfile is None:
                fpath = url
            else:
                fpath = abspath(join(dirname(configfile), url))
            if os.path.isfile(fpath):
                return fpath
            else:
                raise ValueError('Invalid FDSN url or file path, check typos')
        raise ValueError('Invalid FDSN url, check typos')


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