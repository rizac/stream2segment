"""
Events download
"""
import os
from collections.abc import Iterable
from datetime import timedelta, datetime
import logging
from itertools import product

import numpy as np
import pandas as pd
from obspy.geodetics import kilometers2degrees
from sqlalchemy import Engine, select

from stream2segment.io.cli import get_progressbar
from stream2segment.io.db.pdsql import (
    apply_table_dtypes, select_df, insert_df, sync_pkey
)
from stream2segment.io.db.models import Event, WebService
from stream2segment.download.url import urlread, CustomResponseCode
from stream2segment.download.modules.utils import (
    fdsn_url_qs, fdsn_response_text_to_df, FailedDownload, NothingToDownload
)

# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial):
logger = logging.getLogger(__name__)


url_file_prefix = "file://"


EVENTWS_MAPPING = {
    'emsc':  'http://www.seismicportal.eu/fdsnws/event/1/query',
    'isc':   'http://www.isc.ac.uk/fdsnws/event/1/query',
    'iris':  'http://service.iris.edu/fdsnws/event/1/query',
    'ncedc': 'http://service.ncedc.org/fdsnws/event/1/query',
    'scedc': 'http://service.scedc.caltech.edu/fdsnws/event/1/query',
    'usgs':  'http://earthquake.usgs.gov/fdsnws/event/1/query',
}


def get_events(
    engine: Engine,
    urls: str | Iterable[str],
    evt_query_args: dict,
    start: datetime,
    end: datetime,
    download_timeout = None,
    show_progress=True
) -> pd.DataFrame:
    """Return the event data frame from the given url or local file"""

    if isinstance(urls, str):
        urls = [urls]

    # urls = [
    #     url_file_prefix + os.path.abspath(u)
    #     if is_local_file(u) else EVENTWS_MAPPING.get(u, u)
    #     for u in urls
    # ]

    # local_file = is_local_file(url)

    # event_ws_id = None

    # if not local_file:
    #     event_ws_id = configure_ws_fk(url, engine)

    dfr_iter = download_events(
        urls, evt_query_args, start, end, download_timeout, show_progress
    )
    # pd_df_list surely not empty (otherwise we raised FailedDownload)
    events = pd.concat(dfr_iter, axis=0, ignore_index=True, copy=False)
    events[WebService.url.key] = events[WebService.url.key].astype("category")
    sync_webservice_ids_with_db(events, engine, merge_on=Event.webservice_id.key)
    events = save_events(events, engine)

    return events[[
        Event.id.key,
        Event.magnitude.key,
        Event.latitude.key,
        Event.longitude.key,
        Event.depth_km.key,
        Event.time.key
    ]]


def download_events(
    urls: Iterable[str],
    evt_query_args: dict,
    start: datetime,
    end: datetime,
    timeout=120,
    show_progress=False
):
    """Return a list of pandas dataframe(s) from the event url or file

    :param url: a valid url, a mappings string, or a local file (fdsn 'text'
        formatted)
    """
    for url in urls:
        try:
            if is_local_file(url):
                yield read_events_file(url)
            else:
                url = EVENTWS_MAPPING.get(url, url)
                yield from download_from_url(
                    url, evt_query_args, start, end, timeout, show_progress
                )
        except Exception as exc:
            raise FailedDownload(f"Error downloading from {url}: {exc}")


def read_events_file(file_path: str) -> pd.DataFrame:

    with open(file_path, "r") as f:
        first_line = f.readline()

    pipe_count = first_line.count("|")
    comma_count = first_line.count(",")
    semi_count = first_line.count(";")

    sep = "|" if first_line.count("|") > first_line.count(",") else ","

    if pipe_count > semi_count >= comma_count:
        with open(file_path, "r") as f:
            return fdsn_event_response_text_to_df(f.read())
        return pd.read_csv(file_path, sep=sep, comment="#", header=None)

    dfr = pd.read_csv(
        file_path, comment="#", sep="," if comma_count >= semi_count else ";"
    )
    # restore "normal" fdsn dataframe
    # EventID|Time|Latitude|Longitude|Depth/km|Author|Catalog|Contributor|
    # ContributorID|MagType|Magnitude|MagAuthor|EventLocationName|EventType

    col_names = {
        # ("id", "event_id", "eventid",  "evt_id"): Event.id.key,
        ("time",): Event.time.key,
        ("latitude", "lat"): Event.latitude.key,
        ("longitude", "lon"): Event.longitude.key,
        ("depth_km", "depth"): Event.depth_km.key,
        ("magnitude", "mag"): Event.magnitude.key,
        ("mag_type", "magtype"): Event.mag_type.key,
    }
    rename = {}

    dfr_columns = set(dfr.columns)
    for names, sql_col_name in col_names.items():
        keys = set(names) & dfr_columns
        if not keys:
            raise KeyError(f"No column named {names[0]} in {file_path}")
        elif len(keys) != 1 and names[0] != 'id':
            raise KeyError(f"Conflict: Multiple column named {names} in {file_path}")
        rename[list(keys)[0]] = sql_col_name

    dfr = dfr.rename(columns=rename)[list(rename.values())]
    return apply_table_dtypes(Event, dfr, drop_non_nullable=True)


def download_from_url(
    base_url,
    evt_query_args,
    start: datetime,
    end: datetime,
    timeout,
    show_progress=False
):
    """Yield an iterator of tuples (url, data), where both are strings denoting
    the URL and the corresponding response body. The returned iterator has
    length > 1 if the request was too large and had to be split
    """
    for key in ['start', 'end', 'starttime', 'endtime']:
        evt_query_args.pop(key, None)
    evt_query_args['start'] = start
    evt_query_args['end'] = end

    request_too_large_codes = {413, 504, 503, CustomResponseCode.TIMEOUT_ERROR}

    total = 1000000  # 10000 set arbitrarily high (max request should be ~= 200:
    # 0.1 min mag step and 6 month min time step, then cannot be more than 100 * 200)
    step = total
    done = 0
    with get_progressbar(0 if not show_progress else total) as pbar:
        downloads = [evt_query_args]

        while downloads:
            evt_query_args = downloads.pop(0)
            url = fdsn_url_qs(base_url, **evt_query_args)
            response = urlread(url, timeout)

            if response.is_ok:
                if len(downloads) == 0:
                    step =  total - done
                pbar.update(step)
                done += 1
                try:
                    if response.status_code == 204:
                        raise Exception("No data (Http code 204)")
                    yield fdsn_event_response_text_to_df(response.data)
                except Exception as exc:
                    logger.warning(f"Error downloading from {url}: {exc}")
            elif response.status_code not in request_too_large_codes:
                logger.warning(f"Error downloading from {url}: {response.data}")
            else:
                downloads.extend(_split_request(evt_query_args))
                step = (total - done) // len(downloads)


def fdsn_event_response_text_to_df(response: str):
    """
    Convert a response content obtained from a FDSN event webservice with format=text
    into a pandas DataFrame with proper dtypes associated to the SQL mapped class
    """
    dframe = fdsn_response_text_to_df(response)

    if not dframe.empty:
        # EventID|Time|Latitude|Longitude|Depth/km|Author|Catalog|Contributor|
        # ContributorID|MagType|Magnitude|MagAuthor|EventLocationName|EventType
        columns = {
            dframe.columns[0]: Event.event_id.key,
            dframe.columns[1]: Event.time.key,
            dframe.columns[2]: Event.latitude.key,
            dframe.columns[3]: Event.longitude.key,
            dframe.columns[4]: Event.depth_km.key,
            # skip Author (Rarely used, memory-intensive text field)
            # dframe.columns[6]: Event.catalog.key,
            # skip Contributor (Rarely used, memory-intensive text field)
            # skip ContributorID (Rarely used, memory-intensive text field)
            dframe.columns[9]: Event.mag_type.key,
            dframe.columns[10]: Event.magnitude.key
            # skip MagAuthor (Rarely used, memory-intensive text field)
            # skip EventLocationName (Rarely used, memory-intensive text field)
            # skip EventType (Rarely used, memory-intensive text field)
        }

        # rename and set order:
        dframe = dframe.rename(columns=columns)[list(columns.values())]
        dframe = apply_table_dtypes(Event, dframe, drop_non_nullable=True)

    if dframe.empty:
        raise ValueError("Malformed data (e.g., no data, type mismatch, NaN)")
    return dframe


# def normalize_url(base_url, evt_query_args, start, end):
#     """Return the normalized URL string of url:
#     1. Converts base_url to a normal URL if the former is a key of EVENTWS_MAPPING
#     2. Set event_query_args 'starttime' and 'endtime' equal to the provided arguments
#        `start` and `end` (handling duplicate names such as 'start' / 'startime')
#     3. Converts 'minmag' 'maxmag' in `evt_query_args` to 'minmagnitude', 'maxmagnitude'
#     4. Adds a custom format 'text' unless the base_url is not EVENTWS_MAPPING['isc']
#     """
#     _url, _query_args = _normalize(base_url, evt_query_args, start, end)
#     return fdsn_url_qs(_url, **_query_args)


def is_local_file(url):
    """Return whether url denotes a local file path, existing on the computer
    machine
    """
    return url not in EVENTWS_MAPPING and os.path.isfile(url)


# def _normalize(base_url, evt_query_args, start, end):
#     """Return the normalized tuple (url, evt_query_args):
#     1. Converts base_url to a normal URL if the former is a key of EVENTWS_MAPPING
#     2. Set event_query_args 'starttime' and 'endtime' equal to the provided arguments
#        `start` and `end` (handling duplicate names such as 'start' / 'startime')
#     3. Converts 'minmag' 'maxmag' in `evt_query_args` to 'minmagnitude', 'maxmagnitude'
#     4. Adds a custom format 'text'
#     """
#     # This should never happen but let's be safe: override start and end
#     if 'start' in evt_query_args:
#         evt_query_args.pop('start')
#     evt_query_args['starttime'] = start
#     if 'end' in evt_query_args:
#         evt_query_args.pop('end')
#     evt_query_args['endtime'] = end
#     # assure that we have 'minmagnitude' and 'maxmagnitude' as mag parameters,
#     # if any:
#     if 'minmag' in evt_query_args:
#         minmag = evt_query_args.pop('minmag')
#         if 'minmagnitude' not in evt_query_args:
#             evt_query_args['minmagnitude'] = minmag
#     if 'maxmag' in evt_query_args:
#         maxmag = evt_query_args.pop('maxmag')
#         if 'maxmagnitude' not in evt_query_args:
#             evt_query_args['maxmagnitude'] = maxmag
#
#     url = EVENTWS_MAPPING.get(base_url, base_url)
#     evt_query_args.setdefault('format', "text")
#
#     return url, evt_query_args


# _SUSPECTED_REQUEST_TOO_ARGE = type('suspected_request_too_large', (object,), {})()
#
#
# def _urlread(url, timeout=None):
#     """Wrapper around `urlread` but returns None if the url should be split
#     because of a too long request
#     """
#     raw_data, exc, code = urlread(url, decode='utf8', timeout=timeout)
#
#     if exc is not None:
#         if isinstance(exc, socket.timeout) or \
#                 (isinstance(exc, HTTPError) and exc.code in (413, 504)):  # noqa
#             return _SUSPECTED_REQUEST_TOO_ARGE
#         raise exc
#
#     if code == 204:
#         raw_data = ''
#
#     return raw_data

def _split_request(evt_query_args: dict):
    query_args = dict(evt_query_args)
    minmag = query_args.pop('minmagnitude', query_args.pop('minmag', 0))
    maxmag = query_args.pop('maxmagnitude', query_args.pop('maxmag', 10))
    start = query_args.pop('start', query_args.pop('starttime'))
    end = query_args.pop('end', query_args.pop('endtime'))

    mags = np.array([minmag, maxmag])

    if maxmag - minmag > 0.5:
        low = minmag if minmag > 0 else np.finfo(float).tiny
        mags = np.unique(
            np.around(
                np.logspace(
                    np.log10(low), np.log10(maxmag),6, endpoint=True
                ),
                1
            )
        )

    # split mags in halves
    times = [start, end]
    if abs(end - start) >= timedelta(days=300):
        mid = (start + (end - start) / 2).replace(microsecond=0)
        times = [start, mid, end]

    if len(mags) < 3 and len(times) < 3:
        raise FailedDownload(
            "Maximum recursion reached, cannot split request bounds further"
        )

    for (m1, m2), (t1, t2) in product(
        zip(mags[:-1], mags[1:]),
        zip(times[:-1], times[1:])
    ):
        yield query_args | {
            'minmag': m1,
            'maxmag': m2,
            'start': t1,
            'end': t2
        }

# def _split_request(evt_query_args):
#     """Split the event query issued with the given `event_query_args` (dict)
#     and returns a two-element list:
#     (event_query_args1, event_query_args2)
#     of event query parameters (dicts) resulting from splitting `evt_query_args`
#     """
#     minmag, deltamag, evtfreq_freq_mag_dist = _get_freq_mag_distrib(evt_query_args)
#     if len(evtfreq_freq_mag_dist) < 2:  # max recusrion on magnitudes, split by time:
#         start = strptime(evt_query_args['starttime'])
#         end = strptime(evt_query_args['endtime'])
#         days_diff = int((end - start).days / 2.0)
#         if days_diff < 1:
#             raise ValueError('maximum recursion depth reached')
#         half_dtime_str = (start + timedelta(days=days_diff)).isoformat()
#         evt_query_args1 = dict(evt_query_args)
#         evt_query_args2 = dict(evt_query_args)
#         evt_query_args1['endtime'] = half_dtime_str
#         evt_query_args2['starttime'] = half_dtime_str
#     else:
#         half = evtfreq_freq_mag_dist.sum() / 2.0
#         idx = 1
#         while evtfreq_freq_mag_dist[:idx + 1].sum() < half:
#             idx += 1
#         mag_half = minmag + idx * deltamag
#         evt_query_args1 = dict(evt_query_args)
#         evt_query_args2 = dict(evt_query_args)
#         evt_query_args1['maxmagnitude'] = str(round(mag_half, 1))
#         evt_query_args2['minmagnitude'] = str(round(mag_half, 1))
#
#     return evt_query_args1, evt_query_args2


# def _get_freq_mag_distrib(evt_query_args):
#     """Return the tuple minmag, step, distrib, where minmag is a float
#     representing `func` first point (magnitude), step is the magnitude
#     distance two adjacent points of `distrib`, and `distrib` is a a numpy array
#     (dtype=int) representing the theoretical events distribution from a given
#     magnitude `mag`:
#     ```
#     f(mag) = 10 ** (9-mag)
#     ```
#     """
#     default_min, step, default_max = 0, .1, 9
#
#     # create the function:
#     ret = ((10 ** (default_max - np.arange(default_min, default_max, step))) + 0.5). \
#         astype(int)
#     # set all points of magnitude <1 equal to the frequency at magnitude 1
#     # (no frequency increase after that threshold)
#     index_of_mag_1 = int(0.5 + ((1.0 - default_min) / step))
#     if index_of_mag_1 > 0:
#         ret[:index_of_mag_1] = ret[index_of_mag_1]
#
#     # trim ret if maxmagnitude is given:
#     if 'maxmagnitude' in evt_query_args:
#         maxmag = float(evt_query_args['maxmagnitude'])
#         index_of_maxmag = int(0.5 + ((maxmag - default_min) / step))
#         if index_of_maxmag < len(ret):
#             ret = ret[:index_of_maxmag]
#
#     minmag = default_min
#     # trim ret if minmagnitude is given:
#     if 'minmagnitude' in evt_query_args:
#         minmag = float(evt_query_args['minmagnitude'])
#         index_of_minmag = int(0.5 + ((minmag - default_min) / step))
#         if index_of_minmag > 0:
#             ret = ret[index_of_minmag:]
#
#     return minmag, step, ret


def sync_webservice_ids_with_db(
    dfr: pd.DataFrame, engine, merge_on:str
) -> pd.DataFrame:

    url_col = WebService.url.key

    if not pd.api.types.is_categorical_dtype(dfr[url_col]):
        dfr[url_col] = dfr[url_col].astype("category")
    ws_df = pd.DataFrame([{url_col: dfr[url_col].cat.categories}])

    id_col = WebService.id.key
    ws_df = sync_pkey(
        ws_df,
        engine,
        WebService,
        id_col,
        [url_col]
    )

    id_na = ws_df[id_col].isna()
    if id_na.any():
        inserted, failed = insert_df(ws_df[id_na], engine, WebService)
        if (~id_na).any():
            ws_df = pd.concat([inserted, ws_df[~id_na]], ignore_index=True)
        else:
            ws_df = inserted
        if failed:
            logger.warning(
                f"Failed to insert {len(failed):,} "
                f"WebService url(s) to DB, discarding"
            )

    # avoid conflicts (remove merge_on column, if any):
    dfr.drop(columns=[merge_on], errors="ignore", inplace=True)

    # now assign:
    return dfr.merge(
        ws_df.rename(columns={id_col: merge_on}), on=url_col, how="left"
    )


def save_events(
    events: pd.DataFrame,
    engine: Engine,
    lon_tol_km=5,
    lat_tol_km=5,
    depth_tol_km=5,
    time_tol_sec=30,
    preferred_mag_types: list | None = None,
    show_progress=False
):
    events.reset_index(drop=True, inplace=True)

    lat_col = Event.latitude.key
    lon_col = Event.longitude.key
    depth_col = Event.depth.key
    time_col = Event.time.key

    _suf = ".-"

    def round(series, abs_tol):
        epsilon = np.finfo(float).tiny  # smallest float (for safety instead of 0)
        return series if abs_tol <= epsilon else (series / abs_tol).round().astype(int)

    # first check equal events in the current dataframe:
    events[lat_col + _suf] = round(events[lat_col], kilometers2degrees(lat_tol_km))
    events[lon_col + _suf] = round(events[lon_col], kilometers2degrees(lon_tol_km))
    events[depth_col + _suf] = round(events[depth_col], depth_tol_km)
    events[time_col + _suf] = round(events[time_col].dt.timestamp, time_tol_sec)

    cmp_cols = [lat_col, lon_col, depth_col, time_col]
    cmp_cols_round = [_ + _suf for _ in cmp_cols]
    # uid_cols = [Event.eventid.key, Event.catalog.key]

    conflict_ids = []
    drop_ids = []

    for _, ev_df in events[events.duplicated(cmp_cols_round)].groupby([
        lat_col + _suf, lon_col + _suf, depth_col + _suf, time_col + _suf
    ]):
        # different mag_types? If preferred_mag_types provided check, otherwise conflicts
        aval_mag_types = pd.unique(ev_df[Event.mag_type.key])
        if len(aval_mag_types) > 1:
            preferred_mag_type = None
            if preferred_mag_types is not None:
                for ma_type in preferred_mag_types:
                    if ma_type in aval_mag_types:
                        preferred_mag_type = ma_type
                        break
            if preferred_mag_type is None:
                conflict_ids.extend(ev_df.index)
                continue
            else:
                # take first mag type (if we have more than once):
                keep_idx = (
                    ev_df[ev_df[Event.mag_type.key] == preferred_mag_type].index[0]
                )
                drop_ids.extend(ev_df[ev_df.index != keep_idx].index)
        else:
            # same mag type, take first index that has max mag:
            drop_ids.extend(
                ev_df.index.difference([ev_df[Event.magnitude.key].idxmax()])
            )

    if conflict_ids:
        logger.warning(events.loc[conflict_ids].to_string(index=False, na_rep=''))
        raise FailedDownload(
            f'{len(conflict_ids)} spatio-temporal conflict(s) in events, see log for details'
        )

    if drop_ids:
        logger.warning(f"Dropping {len(drop_ids)} duplicated events")
        events = events[~events.index.isin(drop_ids)]


    # duplicated = events.duplicated(uid_cols, keep=False)
    # if duplicated.any():
    #     logger.warning(events.loc[duplicated].to_string(index=False, na_rep=''))
    #     raise FailedDownload(
    #         f'{len(conflict_ids)} '
    #         f'(eventid, catalog) conflict(s) in events, '
    #         f'see log for details'
    #     )

    cols = events.columns
    conflicts = 0
    new_events = []
    # uc_cols = [Event.eventid.key, Event.catalog.key]

    where_stmt = (
        (Event.latitude >= events[Event.latitude.key].min()) &
        (Event.latitude <= events[Event.latitude.key].max()) &
        (Event.longitude >= events[Event.longitude.key].min()) &
        (Event.longitude <= events[Event.longitude.key].max()) &
        (Event.time >= events[Event.time.key].min()) &
        (Event.time <= events[Event.time.key].max()) &
        (Event.depth_km >= events[Event.depth_km.key].min()) &
        (Event.depth_km <= events[Event.depth_km.key].max()) &
        (Event.magnitude >= events[Event.magnitude.key].min()) &
        (Event.magnitude <= events[Event.magnitude.key].max())
    )
    select_stmt = select(Event).where(where_stmt)
    _suf = '_db_'
    for saved_events in select_df(engine, select_stmt):
        saved_events[lat_col + _suf] = round(
            saved_events[lat_col], kilometers2degrees(lat_tol_km)
        )
        saved_events[lon_col + _suf] = round(
            saved_events[lon_col], kilometers2degrees(lon_tol_km)
        )
        saved_events[depth_col + _suf] = round(
            saved_events[depth_col], depth_tol_km
        )
        saved_events[time_col + _suf] = round(
            saved_events[time_col].dt.timestamp, time_tol_sec
        )

        merged = events.merge(
            saved_events,
            how='left',
            on=cmp_cols_round,
            suffixes=('', _suf),
            indicator = True
        )
        suf_cols = [c for c in merged.columns if c.endswith(_suf)]
        merged = merged.drop_duplicates(subset=cmp_cols_round, keep='first')
        matched = merged[merged["_merge"] == "both"]
        matched.rename(columns={c: c.removesuffix(_suf) for c  in suf_cols}, inplace=True)
        new_events.append(matched[cols])

        events = merged[merged["_merge"] == "left_only"]
        events.drop(columns=suf_cols + ["_merge"], inplace=True)

    if not events.empty:  # it might be if we entered the for loop
        events, failed = insert_df(events, engine, Event)
        if not failed.empty:
            logger.warning(f"Unable to insert {len(failed)} event(s)")
            logger.warning(failed.to_string(
                max_rows=30, index=False, na_rep='', show_dimensions=True
            ))

    if new_events:
        if not events.empty:
            new_events.append(events)
        events = pd.concat(new_events, ignore_index=True)
        if not pd.api.types.is_integer_dtype(events[Event.id.key]):
            events[Event.id.key] = events[Event.id.key].astype(int)

    if conflicts > 0:
        logger.info(
            f'Found {conflicts} conflict(s) with saved DB events, '
            f'using the latter instead of downloaded/supplied events'
        )
    return events
