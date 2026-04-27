"""
Events download
"""
import os
from datetime import timedelta, datetime, UTC
import logging

import numpy as np
import pandas as pd
from obspy.geodetics import kilometers2degrees
from sqlalchemy import Engine, select

from stream2segment.io.cli import get_progressbar
from stream2segment.io.db.pdsql import (
    apply_table_dtypes, get_col_max, execute_sql, create_insert_statement,
    select_df, insert_df
)
from stream2segment.download.exc import FailedDownload, NothingToDownload
from stream2segment.io.db.models import Event, WebService
from stream2segment.download.url import urlread, socket, HTTPError
from stream2segment.download.modules.utils import (
    formatmsg, EVENTWS_MAPPING, strptime, fdsn_url_qs, fdsn_response_text_to_df
)

# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial):
logger = logging.getLogger(__name__)


def get_events(
    engine: Engine,
    url: str,
    evt_query_args: dict,
    start: datetime,
    end: datetime,
    show_progress=True
) -> pd.DataFrame:
    """Return the event data frame from the given url or local file"""
    local_file = is_local_file(url)

    event_ws_id = None

    if not local_file:
        event_ws_id = configure_ws_fk(url, engine)

    pd_df_list = events_df_list(url, evt_query_args, start, end, 120, show_progress)
    # pd_df_list surely not empty (otherwise we raised FailedDownload)
    events_df = pd.concat(pd_df_list, axis=0, ignore_index=True, copy=False)
    if local_file:
        # support for Nones:
        events_df[Event.webservice_id.key] = (
            events_df[Event.webservice_id.key].astype('Int64')
        )
    events_df[Event.webservice_id.key] = event_ws_id

    events_df = save_events(events_df, engine)

    return events_df[[Event.id.key, Event.magnitude.key, Event.latitude.key,
                      Event.longitude.key, Event.depth_km.key, Event.time.key]]


def configure_ws_fk(event_ws_url, engine: Engine):
    """Configure the web service foreign key creating such a db row if it does
    not exist and returning its id"""
    if event_ws_url in EVENTWS_MAPPING:
        event_ws_url = EVENTWS_MAPPING[event_ws_url]

    with engine.begin() as conn:  # noqa
        event_ws_id = conn.execute(
            select(WebService.id).where(WebService.url == event_ws_url)
        ).scalar_one_or_none()

    if event_ws_id is None:  # write url to table
        event_ws_id = get_col_max(engine, WebService.id) + 1
        oks = list(execute_sql(
            engine,
            [create_insert_statement(WebService)],
            [{WebService.id.key: event_ws_id, WebService.url.key: event_ws_url}]
        ))
        if not oks:
            raise FailedDownload(f'Cannot update DB entry {event_ws_url}')

    return event_ws_id


# error string (constants, used in test so we can change them with no problem, hopefully)
ERR_FETCH = "Unable to fetch events"
ERR_FETCH_FDSN = "Unable to fetch events, data not in the supported FDSN format"
ERR_READ_FDSN = "Unable to read events, data not in the supported FDSN format"
ERR_FETCH_NODATA = "No event received, search parameters might be too strict"


def events_df_list(url, evt_query_args, start, end, timeout=120, show_progress=False):
    """Return a list of pandas dataframe(s) from the event url or file

    :param url: a valid url, a mappings string, or a local file (fdsn 'text'
        formatted)
    """
    urls_and_data = []
    local_file = is_local_file(url)
    if local_file:
        try:
            with open(url, encoding='utf-8') as opn:
                data = opn.read()
                if not data:
                    raise ValueError('Empty file')
                urls_and_data.append(data)
        except Exception as exc:
            raise FailedDownload(formatmsg(ERR_READ_FDSN, exc,
                                           f"file:///.../{os.path.basename(url)}"))
    else:
        try:
            urls_and_data = list(events_iter_from_url(url, evt_query_args, start, end,
                                                      timeout, show_progress))
        except NothingToDownload:
            raise
        except Exception as exc:
            _url_ = normalize_url(url, evt_query_args, start, end)
            raise FailedDownload(formatmsg(ERR_FETCH, exc, _url_))

    pd_df_list = []
    for url_, data in urls_and_data:
        # data surely not empty, FDSN formatted
        try:
            dframe = fdsn_event_response_text_to_df(data)
            pd_df_list.append(dframe)
            discarded = dframe.attrs.pop('discarded', 0)
            if discarded > 0:
                logger.warning(
                    formatmsg(f"{discarded} row(s) discarded","malformed text data", url)
                )
        except Exception as exc:
            msg = ERR_READ_FDSN if local_file else ERR_FETCH_FDSN
            if local_file or len(urls_and_data) == 1:  # raise:
                raise FailedDownload(formatmsg(msg, exc, url_))
            else:
                logger.warning(formatmsg(msg, exc, url_))

    if not pd_df_list:
        raise FailedDownload(formatmsg(ERR_FETCH_FDSN, 'details in log file',
                                       normalize_url(url, evt_query_args, start, end)))

    return pd_df_list


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
            dframe.columns[6]: Event.catalog.key,
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


def normalize_url(base_url, evt_query_args, start, end):
    """Return the normalized URL string of url:
    1. Converts base_url to a normal URL if the former is a key of EVENTWS_MAPPING
    2. Set event_query_args 'starttime' and 'endtime' equal to the provided arguments
       `start` and `end` (handling duplicate names such as 'start' / 'startime')
    3. Converts 'minmag' 'maxmag' in `evt_query_args` to 'minmagnitude', 'maxmagnitude'
    4. Adds a custom format 'text' unless the base_url is not EVENTWS_MAPPING['isc']
    """
    _url, _query_args = _normalize(base_url, evt_query_args, start, end)
    return fdsn_url_qs(_url, **_query_args)


def is_local_file(url):
    """Return whether url denotes a local file path, existing on the computer
    machine
    """
    return url not in EVENTWS_MAPPING and os.path.isfile(url)


def events_iter_from_url(base_url, evt_query_args, start, end, timeout,
                         show_progress=False):
    """Yield an iterator of tuples (url, data), where both are strings denoting
    the URL and the corresponding response body. The returned iterator has
    length > 1 if the request was too large and had to be split
    """
    base_url, evt_query_args = _normalize(base_url, evt_query_args, start, end)
    end_iso = evt_query_args['endtime']

    url = fdsn_url_qs(base_url, **evt_query_args)
    result = _urlread(url, timeout)

    if result is not _SUSPECTED_REQUEST_TOO_ARGE:
        if not result:
            raise NothingToDownload(formatmsg(ERR_FETCH_NODATA, "", url))
        yield url, result  # then result is the tuple (url, raw_data)
    else:
        logger.info("Request seems to be too large, splitting into "
                    "sub-requests")

        # control that at least one subrequest returned non empty data:
        yielded = False

        # the tricky part below is actually the progressbar part. It must:
        # 1 not be linear, thus advance "more" at lower magnitudes (where
        #   events are denser)
        # 2 consider that, when the maximum magnitude depth is reached, we split
        #   by time and in this case only the last sub-request should advance the
        #   progress bar
        total_pbar_steps = _get_freq_mag_distrib(evt_query_args)[2].sum()
        with get_progressbar(total_pbar_steps if show_progress else 0) as pbar:
            downloads = [evt_query_args]

            while downloads:
                evt_q_args = _split_request(downloads.pop(0))
                for i, evt_q_arg in enumerate(evt_q_args):
                    url = fdsn_url_qs(base_url, **evt_q_arg)
                    result = _urlread(url, timeout)

                    if result is not _SUSPECTED_REQUEST_TOO_ARGE:
                        # update pbar only if the end of the request equals
                        # the global end_iso (when recursion is done on time, it
                        # updates only on the first time chunk):
                        if evt_q_arg['endtime'] == end_iso:
                            steps = _get_freq_mag_distrib(evt_q_arg)[2].sum()
                            pbar.update(steps)
                        if result:  # do not yield empty data
                            yield url, result  # (url, raw_data)
                            yielded = True
                    else:
                        downloads.insert(i, evt_q_arg)
        if not yielded:
            raise ValueError("no sub-request returned data")


def _normalize(base_url, evt_query_args, start, end):
    """Return the normalized tuple (url, evt_query_args):
    1. Converts base_url to a normal URL if the former is a key of EVENTWS_MAPPING
    2. Set event_query_args 'starttime' and 'endtime' equal to the provided arguments
       `start` and `end` (handling duplicate names such as 'start' / 'startime')
    3. Converts 'minmag' 'maxmag' in `evt_query_args` to 'minmagnitude', 'maxmagnitude'
    4. Adds a custom format 'text'
    """
    # This should never happen but let's be safe: override start and end
    if 'start' in evt_query_args:
        evt_query_args.pop('start')
    evt_query_args['starttime'] = start
    if 'end' in evt_query_args:
        evt_query_args.pop('end')
    evt_query_args['endtime'] = end
    # assure that we have 'minmagnitude' and 'maxmagnitude' as mag parameters,
    # if any:
    if 'minmag' in evt_query_args:
        minmag = evt_query_args.pop('minmag')
        if 'minmagnitude' not in evt_query_args:
            evt_query_args['minmagnitude'] = minmag
    if 'maxmag' in evt_query_args:
        maxmag = evt_query_args.pop('maxmag')
        if 'maxmagnitude' not in evt_query_args:
            evt_query_args['maxmagnitude'] = maxmag

    url = EVENTWS_MAPPING.get(base_url, base_url)
    evt_query_args.setdefault('format', "text")

    return url, evt_query_args


_SUSPECTED_REQUEST_TOO_ARGE = type('suspected_request_too_large', (object,), {})()


def _urlread(url, timeout=None):
    """Wrapper around `urlread` but returns None if the url should be split
    because of a too long request
    """
    raw_data, exc, code = urlread(url, decode='utf8', timeout=timeout)

    if exc is not None:
        if isinstance(exc, socket.timeout) or \
                (isinstance(exc, HTTPError) and exc.code in (413, 504)):  # noqa
            return _SUSPECTED_REQUEST_TOO_ARGE
        raise exc

    if code == 204:
        raw_data = ''

    return raw_data


def _split_request(evt_query_args):
    """Split the event query issued with the given `event_query_args` (dict)
    and returns a two-element list:
    (event_query_args1, event_query_args2)
    of event query parameters (dicts) resulting from splitting `evt_query_args`
    """
    minmag, deltamag, evtfreq_freq_mag_dist = _get_freq_mag_distrib(evt_query_args)
    if len(evtfreq_freq_mag_dist) < 2:  # max recusrion on magnitudes, split by time:
        start = strptime(evt_query_args['starttime'])
        end = strptime(evt_query_args['endtime'])
        days_diff = int((end - start).days / 2.0)
        if days_diff < 1:
            raise ValueError('maximum recursion depth reached')
        half_dtime_str = (start + timedelta(days=days_diff)).isoformat()
        evt_query_args1 = dict(evt_query_args)
        evt_query_args2 = dict(evt_query_args)
        evt_query_args1['endtime'] = half_dtime_str
        evt_query_args2['starttime'] = half_dtime_str
    else:
        half = evtfreq_freq_mag_dist.sum() / 2.0
        idx = 1
        while evtfreq_freq_mag_dist[:idx + 1].sum() < half:
            idx += 1
        mag_half = minmag + idx * deltamag
        evt_query_args1 = dict(evt_query_args)
        evt_query_args2 = dict(evt_query_args)
        evt_query_args1['maxmagnitude'] = str(round(mag_half, 1))
        evt_query_args2['minmagnitude'] = str(round(mag_half, 1))

    return evt_query_args1, evt_query_args2


def _get_freq_mag_distrib(evt_query_args):
    """Return the tuple minmag, step, distrib, where minmag is a float
    representing `func` first point (magnitude), step is the magnitude
    distance two adjacent points of `distrib`, and `distrib` is a a numpy array
    (dtype=int) representing the theoretical events distribution from a given
    magnitude `mag`:
    ```
    f(mag) = 10 ** (9-mag)
    ```
    """
    default_min, step, default_max = 0, .1, 9

    # create the function:
    ret = ((10 ** (default_max - np.arange(default_min, default_max, step))) + 0.5). \
        astype(int)
    # set all points of magnitude <1 equal to the frequency at magnitude 1
    # (no frequency increase after that threshold)
    index_of_mag_1 = int(0.5 + ((1.0 - default_min) / step))
    if index_of_mag_1 > 0:
        ret[:index_of_mag_1] = ret[index_of_mag_1]

    # trim ret if maxmagnitude is given:
    if 'maxmagnitude' in evt_query_args:
        maxmag = float(evt_query_args['maxmagnitude'])
        index_of_maxmag = int(0.5 + ((maxmag - default_min) / step))
        if index_of_maxmag < len(ret):
            ret = ret[:index_of_maxmag]

    minmag = default_min
    # trim ret if minmagnitude is given:
    if 'minmagnitude' in evt_query_args:
        minmag = float(evt_query_args['minmagnitude'])
        index_of_minmag = int(0.5 + ((minmag - default_min) / step))
        if index_of_minmag > 0:
            ret = ret[index_of_minmag:]

    return minmag, step, ret


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
        epsilon = np.nextafter(0, 1)  # smallest float (for safety instead of 0)
        return series if abs_tol <= epsilon else (series / abs_tol).round().astype(int)

    # first check equal events in the current dataframe:
    events[lat_col + _suf] = round(events[lat_col], kilometers2degrees(lat_tol_km))
    events[lon_col + _suf] = round(events[lon_col], kilometers2degrees(lon_tol_km))
    events[depth_col + _suf] = round(events[depth_col], depth_tol_km)
    events[time_col + _suf] = round(events[time_col].dt.timestamp, time_tol_sec)

    cmp_cols = [lat_col, lon_col, depth_col, time_col]
    cmp_cols_round = [_ + _suf for _ in cmp_cols]
    uid_cols = [Event.eventid.key, Event.catalog.key]

    conflict_ids = []
    drop_ids = []

    for _, ev_df in events[events.duplicated(cmp_cols_round)].groupby([
        lat_col + _suf, lon_col + _suf, depth_col + _suf, time_col + _suf
    ]):
        # same (eventid, catalog)? If yes, go on, otherwise conflicts
        dupes = ev_df.duplicated(uid_cols, keep=False)
        if not dupes.all():
            conflict_ids.extend(ev_df.index)
            continue

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


    duplicated = events.duplicated(uid_cols, keep=False)
    if duplicated.any():
        logger.warning(events.loc[duplicated].to_string(index=False, na_rep=''))
        raise FailedDownload(
            f'{len(conflict_ids)} '
            f'(eventid, catalog) conflict(s) in events, '
            f'see log for details'
        )

    cols = events.columns
    conflicts = 0
    new_events = []
    uc_cols = [Event.eventid.key, Event.catalog.key]

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
        matched = merged[merged["_merge"] == "both"]
        matched.rename(columns={c + _suf: c for c  in cols}, inplace=True)
        new_events.append(matched[cols] + [Event.id.key])

        events = merged[merged["_merge"] == "left_only"]
        events.drop(
            columns=[c for c  in events.columns if c.endswith(_suf)] + ["_merge"],
            inplace=True
        )

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
