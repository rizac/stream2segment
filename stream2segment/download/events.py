"""
Events download
"""
import os
from collections.abc import Iterable
from datetime import timedelta, datetime
import logging
from io import BytesIO
from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from obspy.geodetics import kilometers2degrees
from pandas import CategoricalDtype
from sqlalchemy import and_, Select
from sqlalchemy.orm import DeclarativeBase

from stream2segment.io.utils import get_progressbar
from stream2segment.io.db.pdsql import (
    apply_table_dtypes, fetch_df, insert_df, sync_pkey, set_pkeys, select, Engine
)
from stream2segment.io.db.models import Event, WebService
from stream2segment.download.url import read_url, CustomResponseCode, Response
from stream2segment.download.utils import (
    fdsn_url_qs,
    fdsn_response_text_to_df,
    FailedDownload,
    NoSegmentsToDownload,
    fdsn_url
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

# catalog shortcuts (remember to keep the download.yml doc in sync with this list):
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
    download_timeout: int,
    event_overlap_tolerance: dict,
    on_event_conflict: Literal["keep", "discard"] | str = "keep",
    show_progress=True,
) -> pd.DataFrame:
    """Return the event data frame from the given url or local file"""

    events = download_or_read_events(
        urls, evt_query_args, start, end, download_timeout, show_progress
    )
    if events.empty:
        raise FailedDownload(
            f'no events downloaded; check your configuration, '
            'web services status, your internet connection. See log for details'
        )
    else:
        logger.info(
            f"{len(events):,} event(s) downloaded; "
            f"checking duplicates, conflicts, and saving"
        )

    ws_id_col = Event.webservice_id.key
    events = sync_webservice_urls_and_assign_ids(
        events, engine, ids_column_name=ws_id_col
    )
    rem = events[ws_id_col].isna() & events[url_col].notna()
    if rem.any():
        events = events[~rem]
        if events.empty:
            raise FailedDownload(
                f"All events ({rem.sum()}) discard (associated URL not saved to DB)"
            )
        else:
            logger.warning(
                f"Discarding {rem.sum()} events(s) (associated URL not saved to DB)"
            )

    events[ws_id_col] = events[ws_id_col].astype('Int64')  # there might be Nones
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
            'No events saved; this is likely due to a Database I/O error. '
            'See log for details'
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
) -> pd.DataFrame:
    """
    Yield pandas dataframe(s) from the event url or file
    """

    url_col_dtype = CategoricalDtype(categories=list({
        fdsn_url(EVENTWS_MAPPING.get(u, u)) for u in urls if not is_local_file(u)
    }))
    events = []

    for url in urls:
        if is_local_file(url):
            try:
                events.append(read_events(Path(url)))
                if url_col not in events[-1].columns:
                    events[-1][url_col] =  pd.Series(
                        None, index=events[-1].index, dtype=url_col_dtype
                    )
            except Exception as exc:
                raise FailedDownload(f"{url}: {exc}")
            continue

        url = EVENTWS_MAPPING.get(url, url)
        for resp in download_events(
            url, evt_query_args, start, end, timeout, show_progress
        ):
            try:
                dfr = read_events(resp.data)
                dfr[url_col] = pd.Series(url, index=dfr.index, dtype=url_col_dtype)
                events.append(dfr)
            except Exception as exc:
                logger.warning(
                    f"Unable to read data downloaded from {resp.request}: {exc}"
                )

    ret = pd.DataFrame()
    if events:
        ret = pd.concat(events, axis=0, ignore_index=True, copy=False)
    return ret


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
    split_request = False  # flag denoting when we split the firt time

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
                yield response
            elif response.status_code not in request_too_large_codes:
                logger.warning(str(response))
            else:
                if not split_request:
                    logger.warning('Request too large; splitting into smaller chunks')
                    split_request = True
                try:
                    downloads.extend(_split_request(evt_query_args))
                except RecursionError:
                    FailedDownload(
                        "Recursion limit reached, cannot split request bounds further. "
                        "Narrow down your events search or try again later"
                    )
                step = (total - done) // len(downloads)
                continue

            if len(downloads) == 0:
                step = total - done
            pbar.update(step)
            done += 1


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
        dfr = fdsn_event_response_text_to_df(BytesIO(content))
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
                raise Exception(f"No column named {repr(names[0])} event catalog")
            elif len(keys) != 1 and names[0] != 'id':
                raise Exception(
                    f"Conflict: Multiple column(s) for {repr(names)} in event catalog"
                )
            rename[list(keys)[0]] = sql_col_name

        dfr = dfr.rename(columns=rename)[list(rename.values())]

    dfr = apply_table_dtypes(Event, dfr, drop_non_nullable=True)
    if dfr.empty:
        raise Exception('no event with valid data')
    return dfr


def fdsn_event_response_text_to_df(response: BytesIO):
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
        raise RecursionError("Recursion limit reached")  # <- message is likely useless

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
    ws_df = sync_pkey(ws_df, engine, WebService,[url_col])

    to_insert = set_pkeys(ws_df[ws_df[id_col].isna()], engine, WebService)
    inserted = insert_df(to_insert, engine, WebService)
    if not inserted.empty:
        ws_df[id_col] = inserted[id_col]  # assignment is index aligned

    # for safety remove merge_on column, if any:
    dfr.drop(columns=[ids_column_name], errors="ignore", inplace=True)
    # now assign (might change dtype of url so check this beforehand):
    dfr = dfr.merge(
        ws_df.rename(columns={id_col: ids_column_name}), on=url_col, how="left"
    )
    # restore categorical if needed (merge might convert to StringDtype):
    if not pd.api.types.is_categorical_dtype(dfr[url_col]):
        dfr[url_col] = dfr[url_col].astype("category")
    return dfr



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
    num_events = len(events)

    uc_cols = [lat_col, lon_col, depth_col, time_col]
    # for safety:
    events.drop_duplicates(uc_cols + [mag_col, magtype_col], inplace=True)

    lat_tol_deg = kilometers2degrees(lat_tol_km)
    lon_tol_deg = kilometers2degrees(lon_tol_km)
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
    events[lat_col + _round_suf] = round(events[lat_col], lat_tol_deg)
    events[lon_col + _round_suf] = round(events[lon_col], lon_tol_deg)
    events[depth_col + _round_suf] = round(events[depth_col], depth_tol_km)
    events[time_col + _round_suf] = round(events[time_col], time_tol_sec)

    cmp_cols_round = [_ + _round_suf for _ in uc_cols]

    # if events share same coordinates and mag type, take the largest magnitude:
    idx = events.groupby(cmp_cols_round + [magtype_col])[mag_col].idxmax()
    events = events.loc[idx]

    if on_event_conflict == 'discard':
        events.drop_duplicates(cmp_cols_round, keep=False, inplace=True)

    elif on_event_conflict != 'keep':
        preferred_mag_types = on_event_conflict.split(',')
        drop_indices = []
        for _, ev_df in events[events.duplicated(cmp_cols_round, keep=False)].groupby(
            cmp_cols_round
        ):
            drop_ids = []
            for ma_type in preferred_mag_types:
                mask = ev_df[magtype_col].str.lower() == ma_type.strip().lower()
                if mask.sum() == 1:
                    drop_ids.extend(
                        ev_df.index.difference([mask.idxmax()])  # take first true
                    )
                    break
            else:
                # no break loop hit:
                drop_ids.extend(ev_df.index)

            if drop_ids:
                logger.warning(
                    f"Discarding the following overlapping events: "
                    f"{to_urls(ev_df.loc[drop_ids])}"
                )
                drop_indices.extend(drop_ids)

        if drop_indices:
            events.drop(list(drop_indices), errors='ignore', inplace=True)

    if len(events) < num_events:
        dropped = num_events - len(events)
        logger.info(f"{dropped:,} overlapping event(s) discarded")

    if events.empty:
        raise NoSegmentsToDownload('All downloaded events overlap')

    id_col = Event.id.key

    # check values on db and assign their spatio-temporal coords and ids,
    select_stmt = get_db_select_statement(
        events, lat_tol_deg, lon_tol_deg, depth_tol_km, time_tol_sec
    )

    events[id_col] = pd.Series(pd.NA, index=events.index, dtype="Int64")
    _suf = '_.db._'
    mismatches = 0
    for saved_events in fetch_df(engine, select_stmt):
        saved_events[lat_col + _round_suf] = round(
            saved_events[lat_col], lat_tol_deg
        )
        saved_events[lon_col + _round_suf] = round(
            saved_events[lon_col], lon_tol_deg
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

        # mismatch events are those with very similar coordinates to some already
        # saved event. We find them by checking that id_col is notna (already saved)
        # and the coordinates (lat_col downloaded one, lat_col + _suf saved one, and
        # so on) are not equal:
        mismatch = events[id_col + _suf].notna() & (
            (events[lat_col] != events[lat_col + _suf]) |
            (events[lon_col] != events[lon_col + _suf]) |
            (events[depth_col] != events[depth_col + _suf]) |
            (events[time_col] != events[time_col + _suf])
        )
        _mismatches = mismatch.sum()
        if _mismatches > 0:
            mismatches += _mismatches
            logger.warning(
                f'Replacing the following events with matching database records '
                f'(magnitude might differ):\n'
                f'{to_urls(events[mismatch])}'
            )
            events.loc[mismatch, lat_col] = events.loc[mismatch, lat_col + _suf]
            events.loc[mismatch, lon_col] = events.loc[mismatch, lon_col + _suf]
            events.loc[mismatch, depth_col] = events.loc[mismatch, depth_col + _suf]
            events.loc[mismatch, time_col] = events.loc[mismatch, time_col + _suf]

        events[id_col] = events[id_col].fillna(events[id_col + _suf])
        events.drop(
            columns=[c for c in events.columns if c.endswith(_suf)], inplace=True
        )
    events.drop(columns=cmp_cols_round, inplace=True)
    if mismatches > 0:
        logger.warning(
            f'{mismatches} event(s) replaced with matching database records '
            f'(magnitudes might differ)'
        )
    events = insert_id_col_na_values_to_db(engine, Event, events, id_col)

    if not events.empty:
        events.drop_duplicates(id_col, keep='first', inplace=True) # for safety

    return events


def insert_id_col_na_values_to_db(
    engine: Engine, table: type[DeclarativeBase], dfr: pd.DataFrame, id_col: str
) -> pd.DataFrame:

    item_name = table.__name__.lower()
    to_insert = set_pkeys(dfr[dfr[id_col].isna()], engine, table)
    inserted = insert_df(to_insert, engine, table)
    if not inserted.empty:
        dfr[id_col] = inserted[id_col]  # assignment is index aligned

    id_na = dfr[id_col].isna()
    if id_na.any():
        logger.warning(
            f"{id_na.sum():,} {item_name}(s)"
            f"discarded (likely error while inserting to DB)\n" +
            dfr[id_na].to_string(
                max_rows=30, index=False, na_rep='', show_dimensions=True
            )
        )
        dfr.dropna(subset=[id_col], inplace=True)

    if not dfr.empty:
        # for safety:
        dfr[id_col] = dfr[id_col].astype(int)

    logger.info(f'{len(to_insert):,} new {item_name}(s) saved to database')

    return dfr


def get_db_select_statement(
    events, lat_tol_deg, lon_tol_deg, depth_tol_km, time_tol_sec
) -> Select:
    conditions = []
    # select according to current events; use loose deltas to avoid
    # fetching and comparing db events unnecessarily:
    for name, col, delta in [
        (lat_col, Event.latitude, lat_tol_deg + 0.0001),
        (lon_col, Event.longitude, lon_tol_deg + 0.0001),
        (depth_col, Event.depth_km, depth_tol_km + 0.01),
    ]:
        if pd.notna(events[name].max()):
            conditions.append(col <= float(events[name].max() + delta))
        if pd.notna(events[name].min()):
            conditions.append(col >= float(events[name].min() - delta))

    # handle datetimes separately:
    delta = timedelta(seconds=time_tol_sec + 1)
    col = Event.time
    name = time_col
    if pd.notna(events[name].max()):
        conditions.append(col <= (events[name].max() + delta).to_pydatetime())
    if pd.notna(events[name].min()):
        conditions.append(col >= (events[name].min() - delta).to_pydatetime())

    select_stmt = select(
        Event.id, Event.latitude, Event.longitude, Event.time, Event.depth_km,
    )
    if conditions:
        select_stmt = select_stmt.where(and_(*conditions))
    return select_stmt


def to_urls(dfr:pd.DataFrame, max_rows=5):
    ret = []
    for row in dfr[:max_rows].itertuples(index=True):
        url = getattr(row, url_col, None)
        ev_id = getattr(row, Event.eventid.key, None)
        if url is None or ev_id is None:
            line = (
                f'event('
                f'mag={getattr(row, mag_col, "N/A")}, '
                f'lat={getattr(row, lat_col, "N/A")}, '
                f'lon={getattr(row, lon_col, "N/A")}, '
                f'time={getattr(row, time_col, "N/A")})'
            )
        else:
            line = fdsn_url_qs(url, eventid=ev_id, format='text')
        ret.append(line)

    if max_rows is not None:
        ret.append(f'(showing first {max_rows:,} of {len(dfr):,})')

    return "\n".join(ret)