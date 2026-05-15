"""
Events download
"""
import os
from collections.abc import Iterable
from datetime import timedelta, datetime
import logging
from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from obspy.geodetics import kilometers2degrees
from pandas import CategoricalDtype
from sqlalchemy import and_, Select

from stream2segment.io.utils import get_progressbar
from stream2segment.io.db.pdsql import (
    apply_table_dtypes, fetch_df, insert_df, sync_pkey, set_pkeys, select, Engine
)
from stream2segment.io.db.models import Event, WebService
from stream2segment.download.url import read_url, CustomResponseCode, Response
from stream2segment.download.utils import (
    fdsn_url_qs, fdsn_response_text_to_df, FailedDownload, NothingToDownload, fdsn_url
)

# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial):
logger = logging.getLogger(__name__)


url_file_prefix = "file://"
lat_col = Event.latitude.key
lon_col = Event.longitude.key
mag_col = Event.magnitude.key
depth_col = Event.depth_km.key
time_col = Event.time.key
url_col = WebService.url.key
magtype_col = Event.mag_type.key


EVENTWS_MAPPING = {
    'emsc':  'https://www.seismicportal.eu/fdsnws/event/1/query',
    'isc':   'https://www.isc.ac.uk/fdsnws/event/1/query',
    'iris':  'https://service.iris.edu/fdsnws/event/1/query',
    'ncedc': 'https://service.ncedc.org/fdsnws/event/1/query',
    'scedc': 'https://service.scedc.caltech.edu/fdsnws/event/1/query',
    'usgs':  'https://earthquake.usgs.gov/fdsnws/event/1/query',
    'geofon': 'https://geofon.gfz.de/fdsnws/event/1/query'
}


def get_events(
    *,
    engine: Engine,
    urls: Iterable[str],
    evt_query_args: dict,
    start: datetime,
    end: datetime,
    download_timeout,
    event_overlap_tolerance: dict,
    on_event_conflict: Literal["keep", "discard"] | str = "keep",
    show_progress=True,
) -> pd.DataFrame:
    """Return the event data frame from the given url or local file"""

    dfr_iter = download_or_read_events(
        urls, evt_query_args, start, end, download_timeout, show_progress
    )
    # pd_df_list surely not empty (otherwise we raised FailedDownload)
    events = pd.concat(dfr_iter, axis=0, ignore_index=True, copy=False)
    if events.empty:
        raise NothingToDownload('No valid event downloaded or read from file')

    if not pd.api.types.is_categorical_dtype(events[url_col]):
        events[url_col] = events[url_col].astype("category")
    ws_id_col = Event.webservice_id.key
    rows = len(events)
    events = sync_webservice_urls_and_assign_ids(
        events, engine, ids_column_name=ws_id_col
    )
    events.dropna(subset=[ws_id_col], inplace=True)
    if events.empty:
        raise FailedDownload("No events left after failed DB URLs insertion")
    elif rows > len(events):
        logger.warning(
            f"Discarding {rows - len(events)} events(s) "
            f"(associated URL not saved to DB)"
        )
    events[ws_id_col] = events[ws_id_col].astype(int)

    events = save_events(
        events,
        engine,
        lat_tol_km=event_overlap_tolerance['lat'],
        lon_tol_km=event_overlap_tolerance['lon'],
        depth_tol_km=event_overlap_tolerance['depth'],
        time_tol_sec=event_overlap_tolerance['time'],
        on_event_conflict=on_event_conflict
    )

    if events.empty:
        raise FailedDownload(
            'No events left after failing to save them'
        )

    return events[[
        Event.id.key,
        mag_col,
        lat_col,
        lon_col,
        depth_col,
        time_col
    ]]


def download_or_read_events(
    urls: Iterable[str],
    evt_query_args: dict,
    start: datetime,
    end: datetime,
    timeout=120,
    show_progress=False
) -> Iterable[pd.DataFrame]:
    """
    Yield pandas dataframe(s) from the event url or file
    """
    harmonized_urls = {
        url_file_prefix + u if is_local_file(u) else fdsn_url(EVENTWS_MAPPING.get(u, u))
        for u in urls
    }
    cat_type = CategoricalDtype(categories=list(harmonized_urls))

    for url in harmonized_urls:
        if url.startswith(url_file_prefix):
            iterable = [Path(url.removeprefix(url_file_prefix))]
        else:
            iterable = download_events(
                url, evt_query_args, start, end, timeout, show_progress
            )

        for obj in iterable:
            url = None
            data = obj
            if isinstance(obj, Response):  # is s Response object
                url = fdsn_url(obj.request, new_query_string="")  # FIXME unnecessary fdsn_url check
                data = obj.data
            try:
                dfr = read_events(data)
                if dfr.empty:
                    raise Exception(
                        'No rows left after type conversion and filtering'
                    )
                dfr[url_col] = pd.Series(url, index=dfr.index, dtype=cat_type)
                yield dfr
            except Exception as exc:
                if url is None:  # file passed, stop download
                    raise FailedDownload(exc)
                else:
                    logger.warning(f"Unable to read data downloaded from {url}: {exc}")


def read_events(content: str | Path) -> pd.DataFrame:

    sep = ','
    if isinstance(content, Path):  # file path
        if not content.is_file():
            raise Exception('file does not exists')

        with open(content, "r") as f:
            first_line = f.readline()
        if not first_line:
            raise Exception('file empty')

        pipe_count = first_line.count("|")
        comma_count = first_line.count(",")
        semi_count = first_line.count(";")

        if pipe_count <= 1 and comma_count <= 1 and semi_count <= 1:
            raise Exception('file not in CSV format')

        if pipe_count > comma_count:
            if pipe_count > semi_count:
                sep = "|"
        elif semi_count > comma_count:
            sep = ';'

        if sep == '|':
            with open(content, "rb") as f:
                content = f.read()

    if isinstance(content, (str, bytes)):
        if not content:
            raise Exception("no data")
        dfr = fdsn_event_response_text_to_df(content)
        if dfr.empty:
            return dfr
    else:
        dfr = pd.read_csv(content, comment="#", sep=sep)
        if dfr.empty:
            return dfr
        # restore "normal" fdsn dataframe
        # EventID|Time|Latitude|Longitude|Depth/km|Author|Catalog|Contributor|
        # ContributorID|MagType|Magnitude|MagAuthor|EventLocationName|EventType

        col_names = {
            # ("id", "event_id", "eventid",  "evt_id"): Event.id.key,
            ("time",): time_col,
            ("latitude", "lat"): lat_col,
            ("longitude", "lon"): lon_col,
            ("depth_km", "depth"): depth_col,
            ("magnitude", "mag"): mag_col,
            ("mag_type", "magtype"): magtype_col,
        }
        rename = {}

        dfr_columns = set(dfr.columns)
        for names, sql_col_name in col_names.items():
            keys = set(names) & dfr_columns
            if not keys:
                raise Exception(f"No column named {names[0]} in {content}")
            elif len(keys) != 1 and names[0] != 'id':
                raise Exception(
                    f"Conflict: Multiple column named {names} in {content}"
                )
            rename[list(keys)[0]] = sql_col_name

        dfr = dfr.rename(columns=rename)[list(rename.values())]

    return apply_table_dtypes(Event, dfr, drop_non_nullable=True)


def download_events(
    base_fdsn_url,
    evt_query_args,
    start: datetime,
    end: datetime,
    timeout,
    show_progress=False
) -> Iterable[Response]:
    """Yield an iterator of tuples (url, data), where both are strings denoting
    the URL and the corresponding response body. The returned iterator has
    length > 1 if the request was too large and had to be split
    """
    for key in ['start', 'end', 'starttime', 'endtime']:
        evt_query_args.pop(key, None)
    evt_query_args['start'] = start
    evt_query_args['end'] = end
    evt_query_args['format'] = 'text'

    request_too_large_codes = {413, 504, 503, CustomResponseCode.TIMEOUT_ERROR}

    total = 1000000  # 10000 set arbitrarily high (max request should be ~= 200:
    # 0.1 min mag step and 6 month min time step, then cannot be more than 100 * 200)
    step = total
    done = 0
    with get_progressbar(0 if not show_progress else total) as pbar:
        downloads = [evt_query_args]

        while downloads:
            evt_query_args = downloads.pop(0)
            url = fdsn_url_qs(base_fdsn_url, **evt_query_args)
            response = read_url(url, timeout)

            if response.is_ok:
                if len(downloads) == 0:
                    step =  total - done
                pbar.update(step)
                done += 1
                try:
                    if response.status_code == 204:
                        raise Exception("No data (Http code 204)")  # fallback below
                    yield response
                except Exception as exc:
                    logger.warning(f"Unable to read data downloaded from {url}: {exc}")
            elif response.status_code not in request_too_large_codes:
                logger.warning(
                    f"Unable to download data from {url}: "
                    f"{response.data}"
                )
                logger.warning(f"Error downloading from {url}: {response.data}")
            else:
                downloads.extend(_split_request(evt_query_args))
                step = (total - done) // len(downloads)


def fdsn_event_response_text_to_df(response: str):
    """
    Convert a response content obtained from a FDSN event webservice with format=text
    into a pandas DataFrame with proper dtypes associated to the SQL mapped class
    """
    # EventID|Time|Latitude|Longitude|Depth/km|Author|Catalog|Contributor|
    # ContributorID|MagType|Magnitude|MagAuthor|EventLocationName|EventType
    columns = {
        0: Event.eventid.key,
        1: time_col,
        2: lat_col,
        3: lon_col,
        4: depth_col,
        # skip Author (Rarely used, memory-intensive text field)
        # dframe.columns[6]: Event.catalog.key,
        # skip Contributor (Rarely used, memory-intensive text field)
        # skip ContributorID (Rarely used, memory-intensive text field)
        9: magtype_col,
        10: mag_col
        # skip MagAuthor (Rarely used, memory-intensive text field)
        # skip EventLocationName (Rarely used, memory-intensive text field)
        # skip EventType (Rarely used, memory-intensive text field)
    }
    dframe = fdsn_response_text_to_df(
        response, usecols=list(columns.keys()), names=list(columns.values())
    )

    if not dframe.empty:
        dframe = apply_table_dtypes(Event, dframe, drop_non_nullable=True)

    return dframe


def is_local_file(url):
    """Return whether url denotes a local file path, existing on the computer
    machine
    """
    return url not in EVENTWS_MAPPING and os.path.isfile(url)


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


def sync_webservice_urls_and_assign_ids(
    dfr: pd.DataFrame, engine, ids_column_name:str
) -> pd.DataFrame:

    if not pd.api.types.is_categorical_dtype(dfr[url_col]):
        dfr[url_col] = dfr[url_col].astype("category")
    ws_df = pd.DataFrame({url_col: dfr[url_col].cat.categories})

    id_col = WebService.id.key
    ws_df = sync_pkey(
        ws_df,
        engine,
        WebService,
        [url_col]
    )

    to_insert = set_pkeys(ws_df[ws_df[id_col].isna()], engine, WebService)
    inserted = insert_df(to_insert, engine, WebService)
    if not inserted.empty:
        ws_df[id_col] = inserted[id_col]  # assignment is index aligned

    # for safety remove merge_on column, if any:
    dfr.drop(columns=[ids_column_name], errors="ignore", inplace=True)
    # now assign:
    return dfr.merge(
        ws_df.rename(columns={id_col: ids_column_name}), on=url_col, how="left"
    )


def save_events(
    events: pd.DataFrame,
    engine: Engine,
    lon_tol_km,
    lat_tol_km,
    depth_tol_km,
    time_tol_sec,
    on_event_conflict: Literal['keep', 'discard'] | str,
    show_progress=False
):
    events.reset_index(drop=True, inplace=True)

    _round_suf = "_.round._"

    def round(series, abs_tol):
        epsilon = np.finfo(float).eps  # (for safety instead of 0)
        if abs_tol <= epsilon:
            return series
        if pd.api.types.is_datetime64_any_dtype(series):
            series = series.astype("datetime64[ms]").astype("int64")
            abs_tol *= 10 ** 3
        return (series / abs_tol).round().astype("int64")

    # first check equal events in the current dataframe:
    events[lat_col + _round_suf] = round(events[lat_col], kilometers2degrees(lat_tol_km))
    events[lon_col + _round_suf] = round(events[lon_col], kilometers2degrees(lon_tol_km))
    events[depth_col + _round_suf] = round(events[depth_col], depth_tol_km)
    events[time_col + _round_suf] = round(events[time_col], time_tol_sec)

    cmp_cols_round = [_ + _round_suf for _ in [lat_col, lon_col, depth_col, time_col]]

    drop_indices = []

    for _, ev_df in events[events.duplicated(cmp_cols_round, keep=False)].groupby(
        cmp_cols_round
    ):
        drop_ids = []
        if ev_df[magtype_col].nunique(dropna=True) == 1:
            # same mag type, take first index that has max mag:
            drop_ids.extend(
                ev_df.index.difference([ev_df[mag_col].idxmax()])
            )
        elif on_event_conflict == 'discard':
            drop_ids.extend(ev_df.index)
        elif on_event_conflict != 'keep':
            for ma_type in on_event_conflict.split(','):
                mask = ev_df[magtype_col].str.lower() == ma_type.strip().lower()
                if mask.any():
                    drop_ids.extend(
                        ev_df.index.difference([mask.idxmax()])  # take first true
                    )
                    break
            else:
                # no break loop hit:
                drop_ids.extend(ev_df.index)
        if drop_ids:
            logger.warning(f"Discarding {len(drop_ids)} overlapping events: "
                           f"{to_urls(ev_df.loc[drop_ids])}")
            drop_indices.extend(drop_ids)


    if drop_indices:
        logger.info(f"{len(drop_indices):,} overlapping event(s) discarded")
        events.drop(list(drop_indices), errors='ignore', inplace=True)

    id_col = Event.id.key

    select_stmt = get_db_select_statement(events)

    events[id_col] = pd.Series(pd.NA, index=events.index, dtype="Int64")
    _suf = '_.db._'

    for saved_events in fetch_df(engine, select_stmt):
        saved_events[lat_col + _round_suf] = round(
            saved_events[lat_col], kilometers2degrees(lat_tol_km)
        )
        saved_events[lon_col + _round_suf] = round(
            saved_events[lon_col], kilometers2degrees(lon_tol_km)
        )
        saved_events[depth_col + _round_suf] = round(
            saved_events[depth_col], depth_tol_km
        )
        saved_events[time_col + _round_suf] = round(
            saved_events[time_col], time_tol_sec
        )
        events = events.merge(
            saved_events, how='left', on=cmp_cols_round, suffixes=('', _suf)
        )
        on_db = events[id_col + _suf].notna()
        mismatches = on_db & (
            (events[lat_col] != events[lat_col + _suf]) |
            (events[lon_col] != events[lon_col + _suf]) |
            (events[depth_col] != events[depth_col + _suf]) |
            (events[time_col] != events[time_col + _suf])
        )
        if mismatches.any():
            # write to dataframe and log FIXME log!
            events.loc[mismatches, lat_col] = events.loc[mismatches, lat_col + _suf]
            events.loc[mismatches, lon_col] = events.loc[mismatches, lon_col + _suf]
            events.loc[mismatches, depth_col] = events.loc[mismatches, depth_col + _suf]
            events.loc[mismatches, time_col] = events.loc[mismatches, time_col + _suf]

        events[id_col] = events[id_col].fillna(events[id_col + _suf])
        events.drop(
            columns=[c for c in events.columns if c.endswith(_suf)], inplace=True
        )
    events.drop(columns=cmp_cols_round, inplace=True)

    to_insert = set_pkeys(events[events[id_col].isna()], engine, Event)
    inserted = insert_df(to_insert, engine, Event)

    if not inserted.empty:
        events[id_col] = inserted[id_col]  # assignment is index aligned

    id_na = events[id_col].isna()
    if id_na.any():
        logger.warning(
            f"{id_na.sum():,} "
            f"events(s) discarded (likely error while inserting to DB)\n" +
            events[id_na].to_string(
                max_rows=30, index=False, na_rep='', show_dimensions=True
            )
        )
        events.dropna(subset=[id_col], inplace=True)

    # for safety:
    if not events.empty:
        events[id_col] = events[id_col].astype(int)

    return events


def get_db_select_statement(events) -> Select:
    conditions = []
    for name, col in {
        lat_col: Event.latitude,
        lon_col: Event.longitude,
        depth_col: Event.depth_km,
        mag_col: Event.magnitude,
        time_col: Event.time,
    }.items():
        if pd.notna(events[name].max()):
            conditions.append(col <= events[name].max())
        if pd.notna(events[name].min()):
            conditions.append(col >= events[name].min())

    select_stmt = select(
        Event.id, Event.latitude, Event.longitude, Event.time, Event.depth_km,
    )
    if conditions:
        select_stmt = select_stmt.where(and_(*conditions))
    return select_stmt


def to_urls(dfr:pd.DataFrame, max_rows=3):
    ret = []
    _i = 0
    if max_rows is None:
        max_rows = np.inf
    for row in dfr.itertuples(index=True):
        url = getattr(row, url_col, None)
        ev_id = getattr(row, Event.eventid.key, None)
        if url is None or ev_id is None:
            line = (
                f'event #{row.Index + 1} ('
                f'mag: {getattr(row, mag_col, "N/A")}, '
                f'lat: {getattr(row, lat_col, "N/A")}, '
                f'lon: {getattr(row, lon_col, "N/A")},'
                f'time: {getattr(row, time_col, "N/A")})'
            )
        else:
            line = fdsn_url_qs(url, eventid=ev_id)
        ret.append(line)
        _i += 1
        if _i >= max_rows:
            ret.append(f'(showing first {max_rows:,} of {len(dfr):,})')
            break

    return "\n".join(ret)