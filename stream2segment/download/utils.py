"""
Utilities for the download routine
"""
# date Nov 25, 2016
import re
from io import IOBase
from datetime import datetime, date
import logging
from typing import Literal
from urllib.parse import urlencode, urlsplit, urlunsplit

import pandas as pd
from obspy.clients.fdsn.header import service


# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial):
# logger = logging.getLogger(__name__)


class IdOnceLogFilter(logging.Filter):
    """
    logging Filter that expects an 'ID' attribute on each record
    and filters out already processed record (comparing by same 'ID').

    # Usage:

    log_filter = IdOnceLogFilter()
    logger.addFilter(log_filter)
    logger.warn('message', extra={'ID': (1, 'geofon.gfz.de')})  # logged
    logger.warn('another message', extra={'ID': (1, 'geofon.gfz.de')})  # not logged
    # eventually, you can optionally remove the filter (freeing memory):
    logger.removeFilter(log_filter)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processed_ids = set()

    def filter(self, record):
        _id = getattr(record, "ID", None)
        if _id is None:
            return True

        if _id in self.processed_ids:
            return False

        self.processed_ids.add(_id)
        return True


def fdsn_response_text_to_df(response: IOBase, **csv_kwargs) -> pd.DataFrame:
    """
    Convert a response content obtained from a FDSN webservice with format=text
    into a pandas DataFrame of type str (no casting performed)
    """
    params = dict(
        sep='|',
        header=None,
        comment='#',
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
        na_values=[  # cannot set '' otherwise location is cast as float FIXME?
            '#N/A', '#NA', '-NaN', '-nan', '<NA>', 'N/A',
            'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null'
        ]
    )
    params |= csv_kwargs

    return pd.read_csv(response, **params)


def fdsn_url(
    url: str,
    *,
    new_service: Literal['station', 'dataselect', 'event'] | None = None,
    new_method: Literal['query', 'queryauth', 'auth', 'version', 'application.wadl'] | None = None,
):
    """
    Check the validity of the given FDSN URL, raising ValueError if invalid, returning
    unchanged or modified according to the arguments.

    :param url: a valid FDSN url, with or without query string (path suffix after '?').
        If scheme is missing, but URL starts withg "www.", then "https://" is prefixed
        in the returned URL. Otherwise, if scheme is missing, raise ValueError.
    :param new_service: the new service. None will leave the service of `url`. Must be
        a string in ('station', 'dataselect', 'event')
    :param new_method: the new method, None will leave the url method. Must be a string
        in ('query', 'queryauth', 'auth', 'version', 'application.wadl')

    :return: a valid FDSN URL
    """
    services = {'station', 'dataselect', 'event'}
    methods = {'query', 'queryauth', 'auth', 'version', 'application.wadl'}

    split_url = urlsplit(url)
    scheme = new_scheme = split_url.scheme or ''
    if not scheme:
        if url.startswith('www.'):
            split_url = urlsplit(f'https://{url}')
            new_scheme = split_url.scheme
        if not new_scheme:
            raise ValueError('missing URL scheme (e.g. "https://")')

    if not split_url.netloc:
        raise ValueError('missing URL domain (e.g. "geofon.gfz.de")')

    if split_url.fragment:
        raise ValueError(f'unsupported URL fragment (#{split_url.fragment})')

    path = split_url.path  # the query string is in split_url.query

    # Check the path:
    reg = re.match(
        "^/fdsnws/(?P<service>[^/]+)/(?P<majorversion>[^/]+)/(?P<method>.+)$", path
    )

    if not reg:
        raise ValueError(
            'URL path does not match "fdsnws/<service>/<majorversion>/<method>'
        )

    service = reg.group('service')
    if service not in services:
        raise ValueError(f"invalid URL service: {service}")

    majorversion = reg.group('majorversion')
    try:
        float(majorversion)
    except ValueError:
        raise ValueError(f"invalid URL major version: {majorversion}")

    method = reg.group('method')
    if method not in methods:
        raise ValueError(f"invalid URL method: {method}")

    if new_service is None:
        new_service = service
    elif new_service not in services:
        raise ValueError(f'invalid service: {new_service}')

    if new_method is None:
        new_method = method
    elif new_method not in methods:
        raise ValueError(f'invalid method: {new_method}')

    if (
        new_scheme != scheme or
        new_service != service or
        new_method != method
    ):
        new_path = f'/fdsnws/{new_service}/{majorversion}/{new_method}'
        new_parsed = split_url._replace(scheme=new_scheme, path=new_path)
        url = urlunsplit(new_parsed)

    return url


def fdsn_url_qs(base_url: str, **query_args):
    """
    Build a valid FDSN URL with a query string from a base URL and a dictionary of
    parameters. If `query_args` is empty, the original `base_url` is returned without
    its query string, if any.

    :param base_url: the base FDSN URL. Its well-formation is not checked for here
        (See `fdsn_url`). If a query string is present, it will be replaced.
        As such, with no query_args provided, this method effectively removes the query
        string from `base_url`
    :param query_args: the query string to append to the base FDSN URL, in form of dict.
        dict keys (param names) mapped to None will be skipped (not appended), any other
        value will be encoded using its `str` representation or using their
        `isoformat()` method (parameters `start` /`starttime`, `end` / `endtime` that
        are expected to be `datetime`s). Parameters net, sta, loc, cha (and their
        long-name variants, e.g. network) will be ignored if '*'
        Duplicates (e.g. 'mag', 'magnitude') are not checked for and will be both
        encoded, whether the encoded URL is valid depends on the server, but in
        principle it should be invalid
    """
    qs = {}
    # date and time params:
    d_params = {
        'start', 'starttime', 'end', 'endtime', 'startbefore', 'startafter',
        'endbefore', 'endafter'
    }
    # special text params (needing * treated specially):
    c_params = {'net', 'network', 'sta', 'station', 'cha', 'channel', 'loc', 'location'}
    # Numeric params. Keep track of them just for ref (maybe needed in future):
    n_params = {
        'lat', 'latitude', 'minlat', 'minlatitude', 'maxlat', 'maxlatitude',
        'lon', 'longitude', 'minlon', 'minlongitude', 'maxlon', 'maxlongitude',
        'mag', 'magnitude', 'minmag', 'minmagnitude', 'maxmag', 'maxmagnitude',
        'mindepth', 'maxdepth', 'minradius', 'maxradius'
    }
    safe_chars = set()  # avoid encoding these chars (improve debug and copy/paste url)
    for k, v in query_args.items():
        if v is None or (k in c_params and v.strip() == '*'):
            continue
        if k in d_params and isinstance(v, (date, datetime)):
            if not isinstance(v, datetime):
                # do not check if it's date, cause a datetime is also a date!
                v = datetime(v.year, v.month, v.day)
            v = v.isoformat('T')
            safe_chars.update(set('-:') & set(v))  # don't encode ":-" if in v
        else:
            v = str(v)
            if k in c_params:
                safe_chars.update(set(',?*') & set(v))  # don't encode ',?*' if in v
            elif k in n_params:
                safe_chars.update(set('.') & set(v))  # don't encode '.' if in v

        qs[k] = v

    query = ""
    if qs:
        query = urlencode(qs, safe="".join(safe_chars))
    return urlunsplit(urlsplit(base_url)._replace(query=query))


class QuitDownload(Exception):
    """Base abstract-like Exception denoting a quit download action"""


class NoSegmentsToDownload(QuitDownload):
    """Exception that should be raised whenever the download process has no
    segments to download according to the user's settings (no error). See
    `download.main.py` for details
    """
    prefix = 'No segments to download'

    def __str__(self):
        return f'{self.prefix}; {super().__str__().lower()}'


class FailedDownload(QuitDownload):
    """Exception that should be raised whenever the download process could not
    proceed for some error (e.g., download error). See `download.main.py`
    for details
    """
    prefix = 'Download failed'

    def __str__(self):
        return f'{self.prefix}; {super().__str__().lower()}'
