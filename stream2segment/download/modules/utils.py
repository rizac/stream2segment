"""
Utilities for the download routine
"""
# date Nov 25, 2016
import re
import psutil
from io import StringIO
from datetime import datetime, date
import logging
from typing import Literal
from urllib.parse import urlencode, urlparse, urlunparse

import pandas as pd

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


def fdsn_response_text_to_df(response: str):
    """
    Convert a response content obtained from a FDSN webservice with format=text
    into a pandas DataFrame of type str (no casting performed)
    """
    return pd.read_csv(
        StringIO(response),
        sep='|',
        header=None,
        comment='#',
        dtype=str,
        keep_default_na=False,
        na_values=[
            '#N/A', '#NA', '-NaN', '-nan', '<NA>', 'N/A',
            'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null'
        ]
    )


def fdsn_url(
    url: str,
    new_service: Literal['station', 'dataselect', 'event'] | None = None,
    new_method: Literal['query', 'queryauth', 'auth', 'version', 'application.wadl'] | None = None,   # noqa
    check_scheme=True
):
    """
    Check that the given url is a valid FDSN URL and return it (with new service and
    method substrings, if given). Raise ValueError if the url is invalid. The URL query
    string, if present, will not be checked and returned as it is

    :param url: a valid FDSN url, with or without query string or schema
    :param new_service: the new service. None will leave the service of `url`. Must be
        a string in ('station', 'dataselect', 'event')
    :param new_method: the new method, None will leave the url method. Must be a string
        in ('query', 'queryauth', 'auth', 'version', 'application.wadl')
    :param check_scheme: if True (the default) check that the url is prefixed with a
        scheme (e.g. https://), raising ValueError if missing. If False, urls can also
        miss the schema
    """
    services = {'station', 'dataselect', 'event'}
    if new_service is not None and new_service not in services:
        raise ValueError(f'Invalid argument service: {new_service}')

    methods = {'query', 'queryauth', 'auth', 'version', 'application.wadl'}
    if new_method is not None and new_method not in methods:
        raise ValueError(f'Invalid argument method: {new_method}')

    parsed_url = urlparse(url)
    if not parsed_url.scheme and check_scheme:
        raise ValueError('url starts with no scheme, e.g. "https://")')

    if not parsed_url.netloc:
        raise ValueError('url has no valid domain, e.g. "geofon.gfz.de"')

    path = parsed_url.path
    # urlparse has already removed query char '?' and params and fragment
    # from the path (which starts with '/'). Now check the path:
    reg = re.match(
        "^/fdsnws/(?P<service>[^/]+)/(?P<majorversion>[^/]+)/(?P<method>.+)$", path
    )

    if not reg:
        raise ValueError('url has no path "fdsnws/<service>/<majorversion>/<method>')

    service = reg.group('service')
    if service not in services:
        raise ValueError(f"Invalid service in url: {service}")

    majorversion = reg.group('majorversion')
    try:
        float(majorversion)
    except ValueError:
        raise ValueError(f"Invalid major version in url: {majorversion}")

    method = reg.group('method')
    if method not in methods:
        raise ValueError(f"Invalid method in url: {method}")

    change_service = new_service is not None and new_service != service
    change_method = new_method is not None and new_method != method
    if change_service or change_method:
        path2 = f'/fdsnws/{new_service}/{majorversion}/{new_method}'
        new_parsed = parsed_url._replace(path=path2)
        url = urlunparse(new_parsed)

    return url


def fdsn_url_qs(base_url: str, **query_args):
    """Build a valid FDSN URL appending to `base_url` the query string (qs) built
    from the FDSN parameters in `query_args`.

    Note: any query parameter (keys of `query_args`) mapped to None will be removed.
    Any other value will be encoded using its string representation, except for the
    following parameters:
    - net, sta, loc, cha (and their long-name variants, e.g. network)
      will be ignored if '*'.
    - start, end (and their long-name variants) will be converted to ISO format strings
      if Python datetime or date
    Duplicates (e.g. 'mag', 'magnitude') are not checked for and will be both encoded,
    whether the encoded URL is valid depends on the server, but in principle it
    should be rejected
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
            if isinstance(v, date):
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
    return f'{base_url}?{urlencode(qs, safe="".join(safe_chars))}'


class QuitDownload(Exception):
    """Base abstract-like Exception denoting a quit download action"""


class NothingToDownload(QuitDownload):
    """Exception that should be raised whenever the download process has no
    segments to download according to the user's settings (no error). See
    `download.main.py` for details
    """
    pass


class FailedDownload(QuitDownload):
    """Exception that should be raised whenever the download process could not
    proceed for some error (e.g., download error). See `download.main.py`
    for details
    """
    pass


def compute_db_buf_size(
    item_avg_size_mb: float,
    max_mem_fraction: float = 0.2,
    hard_cap_mb: int = 4096
) -> int:
    mem = psutil.virtual_memory()

    available_mb = mem.available / (1024 ** 2)
    usable_mb = min(available_mb * max_mem_fraction, hard_cap_mb)

    return max(1, int(usable_mb // item_avg_size_mb))