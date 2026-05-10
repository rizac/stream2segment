"""
Events download
"""
import os
from collections.abc import Iterable
from datetime import timedelta, datetime
import logging
from itertools import product
from typing import Literal

import numpy as np
import pandas as pd
from obspy.geodetics import kilometers2degrees
from sqlalchemy import Engine, select

from stream2segment.io.utils import get_progressbar
from stream2segment.io.db.pdsql import (
    apply_table_dtypes, select_df, insert_df, sync_pkey
)
from stream2segment.io.db.models import Event, WebService
from stream2segment.download.url import read_url, CustomResponseCode
from stream2segment.download.modules.utils import (
    fdsn_url_qs, fdsn_response_text_to_df, FailedDownload, NothingToDownload
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
    'emsc':  'http://www.seismicportal.eu/fdsnws/event/1/query',
    'isc':   'http://www.isc.ac.uk/fdsnws/event/1/query',
    'iris':  'http://service.iris.edu/fdsnws/event/1/query',
    'ncedc': 'http://service.ncedc.org/fdsnws/event/1/query',
    'scedc': 'http://service.scedc.caltech.edu/fdsnws/event/1/query',
    'usgs':  'http://earthquake.usgs.gov/fdsnws/event/1/query',
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

    dfr_iter = download_events(
        urls, evt_query_args, start, end, download_timeout, show_progress
    )
    # pd_df_list surely not empty (otherwise we raised FailedDownload)
    events = pd.concat(dfr_iter, axis=0, ignore_index=True, copy=False)

    events[url_col] = events[url_col].astype("category")
    ws_id_col = Event.webservice_id.key
    sync_webservice_ids_with_db(events, engine, merge_on=ws_id_col)
    wsid_na = pd.isna(events[ws_id_col])
    if wsid_na.any():
        logger.warning(
            f"Discarding {wsid_na.sum()} event(s) (associated URL not saved to DB)"
        )
    events = save_events(
        events,
        engine,
        lat_tol_km=event_overlap_tolerance['lat'],
        lon_tol_km=event_overlap_tolerance['lon'],
        depth_tol_km=event_overlap_tolerance['depth'],
        time_tol_sec=event_overlap_tolerance['time'],
        on_event_conflict=on_event_conflict
    )

    return events[[
        Event.id.key,
        mag_col,
        lat_col,
        lon_col,
        depth_col,
        time_col
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
                yield from download_events_from_url(
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
            raise KeyError(f"No column named {names[0]} in {file_path}")
        elif len(keys) != 1 and names[0] != 'id':
            raise KeyError(f"Conflict: Multiple column named {names} in {file_path}")
        rename[list(keys)[0]] = sql_col_name

    dfr = dfr.rename(columns=rename)[list(rename.values())]
    return apply_table_dtypes(Event, dfr, drop_non_nullable=True)


def download_events_from_url(
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
            response = read_url(url, timeout)

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
            dframe.columns[1]: time_col,
            dframe.columns[2]: lat_col,
            dframe.columns[3]: lon_col,
            dframe.columns[4]: depth_col,
            # skip Author (Rarely used, memory-intensive text field)
            # dframe.columns[6]: Event.catalog.key,
            # skip Contributor (Rarely used, memory-intensive text field)
            # skip ContributorID (Rarely used, memory-intensive text field)
            dframe.columns[9]: magtype_col,
            dframe.columns[10]: mag_col
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


def sync_webservice_ids_with_db(
    dfr: pd.DataFrame, engine, merge_on:str
) -> pd.DataFrame:

    if not pd.api.types.is_categorical_dtype(dfr[url_col]):
        dfr[url_col] = dfr[url_col].astype("category")
    ws_df = pd.DataFrame({url_col: dfr[url_col].cat.categories})

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
        inserted, failed = insert_df(
            ws_df[id_na].drop(columns=id_col), engine, WebService
        )
        if id_na.all():
            ws_df = inserted
        else:
            ws_df = pd.concat([inserted, ws_df[~id_na]], ignore_index=True)
        if not failed.empty:
            logger.warning(
                f"Discarding {len(failed):,} "
                f"WebService URL(s) (error while inserting to DB)"
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
        return series if abs_tol <= epsilon else (series / abs_tol).round().astype(int)

    # first check equal events in the current dataframe:
    events[lat_col + _round_suf] = round(events[lat_col], kilometers2degrees(lat_tol_km))
    events[lon_col + _round_suf] = round(events[lon_col], kilometers2degrees(lon_tol_km))
    events[depth_col + _round_suf] = round(events[depth_col], depth_tol_km)
    events[time_col + _round_suf] = round(events[time_col].dt.timestamp, time_tol_sec)

    cmp_cols_round = [_ + _round_suf for _ in [lat_col, lon_col, depth_col, time_col]]

    drop_ids = []

    for _, ev_df in events[events.duplicated(cmp_cols_round)].groupby(cmp_cols_round):
        _drop_ids = []
        if ev_df[magtype_col].nunique(dropna=True) == 1:
            # same mag type, take first index that has max mag:
            _drop_ids.extend(
                ev_df.index.difference([ev_df[mag_col].idxmax()])
            )
        elif on_event_conflict == 'discard':
            _drop_ids.extend(ev_df.index)
        elif on_event_conflict != 'keep':
            for ma_type in on_event_conflict.split(','):
                mask = ev_df[magtype_col].str.lower() == ma_type.strip().lower()
                if mask.any():
                    _drop_ids.extend(
                        ev_df.index.difference([mask.idxmax()])  # take first true
                    )
                    break
            else:
                # no break loop hit:
                _drop_ids.extend(ev_df.index)
        if _drop_ids:
            logger.warning(f"Discarding {len(_drop_ids)} overlapping events: "
                           f"{to_urls(ev_df.loc[_drop_ids])}")
            drop_ids.extend(_drop_ids)


    if drop_ids:
        logger.info(f"{len(drop_ids):,} overlapping event(s) discarded")
        events = events[~events.index.isin(drop_ids)]

    cols = events.columns
    new_events = []
    # uc_cols = [Event.eventid.key, Event.catalog.key]

    where_stmt = (
        (Event.latitude >= events[lat_col].min()) &
        (Event.latitude <= events[lat_col].max()) &
        (Event.longitude >= events[lon_col].min()) &
        (Event.longitude <= events[lon_col].max()) &
        (Event.time >= events[time_col].min()) &
        (Event.time <= events[time_col].max()) &
        (Event.depth_km >= events[depth_col].min()) &
        (Event.depth_km <= events[depth_col].max()) &
        (Event.magnitude >= events[mag_col].min()) &
        (Event.magnitude <= events[mag_col].max())
    )
    select_stmt = select(Event).where(where_stmt)
    _suf = '_.db._'
    id_col = Event.id.key

    for saved_events in select_df(engine, select_stmt):
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
            saved_events[time_col].dt.timestamp, time_tol_sec
        )

        events = events.merge(
            saved_events,
            how='left',
            on=cmp_cols_round,
            suffixes=('', _suf)
        )
        on_db = events[id_col + _suf].notna()
        mismatches = on_db & (
            (events[lat_col + _round_suf] != events[lat_col + _round_suf + _suf]) |
            (events[lon_col + _round_suf] != events[lon_col + _round_suf + _suf]) |
            (events[depth_col + _round_suf] != events[depth_col + _round_suf + _suf]) |
            (events[time_col + _round_suf] != events[time_col + _round_suf + _suf])
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

    inserted, failed = insert_df(
        events[events[id_col].isna()], engine, Channel
    )
    if not failed.empty:
        logger.warning(
            f"{len(failed)} events(s) discarded (error while inserting to DB)\n" +
            failed.to_string(
                max_rows=30, index=False, na_rep='', show_dimensions=True
            )
        )
    if not inserted.empty:
        # put id_col into events:
        events = events.merge(
            inserted[uc_cols + [id_col]], how='left', on=uc_cols,  suffixes = ('', _suf)
        )
        events.drop(
            columns=[c for c in events.columns if c.endswith(_suf)], inplace=True
        )

    events.drop(columns=cmp_cols_round, inplace=True)

    # for safety:
    events[id_col] = events[id_col].astype(int)
    return events


def to_urls(dfr:pd.DataFrame, max_rows=3):
    ret = []
    _i = 0
    if max_rows is None:
        max_rows = np.inf
    for row in dfr.itertuples(index=True):
        url = getattr(row, url_col, None)
        ev_id = getattr(row, Event.eventid.key, None)
        if url is None and ev_id is None:
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