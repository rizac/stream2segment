"""
Events download
"""
import os
from datetime import timedelta
import logging

import numpy as np
import pandas as pd
from obspy.geodetics import degrees2kilometers, kilometers2degrees
from sqlalchemy import func, Engine, select

from stream2segment.io.cli import get_progressbar
from stream2segment.io.db.pdsql import df2db, apply_table_dtypes, get_row_count, db2dfs
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
    url,
    evt_query_args,
    start,
    end,
    # db_bufsize=30,
    # timeout=15,
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

    if local_file:
        check_events_df_from_local_file(events_df, engine, show_progress)

    events_df, failed_i, _ = df2db(
        events_df,
        Event,
        engine,
        'id',
        [Event.eventid.key, Event.catalog.key],
        # chunksize=db_bufsize,
    )

    # try to release memory for unused columns (FIXME: NEEDS TO BE TESTED)
    return events_df[[Event.id.key, Event.magnitude.key, Event.latitude.key,
                      Event.longitude.key, Event.depth_km.key, Event.time.key]].copy()


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
        dfr = pd.DataFrame((event_ws_url,), columns=[WebService.url.key])

        dfr, i_err, _ = df2db(
            dfr,
            WebService,
            engine,
            'id',
            [WebService.url.key],
        )
        event_ws_id = dfr.iloc[0][WebService.id.key]

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
    if not dframe.empty:
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


def check_duplicates(
    events: pd.DataFrame,
    engine: Engine,
    lon_tol_km=10,
    lat_tol_km=10,
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
    # first check equal events in the current dataframe:
    events[lat_col + _suf] = (
        (events[lat_col] / kilometers2degrees(lat_tol_km)).round().astype(int)
    )
    events[lon_col + _suf] = (
        (events[lon_col] / kilometers2degrees(lon_tol_km)).round().astype(int)
    )
    events[depth_col + _suf] = (events[depth_col] / depth_tol_km).round().astype(int)
    events[time_col + _suf] = (
        (events[time_col].dt.timestamp / time_tol_sec).round().astype(int)
    )
    cmp_cols = [lat_col, lon_col, depth_col, time_col]
    cmp_cols_round = [_ + _suf for _ in cmp_cols]
    uid_cols = [Event.eventid.key, Event.catalog.key]

    conflict_ids = []
    drop_ids = []

    for _, ev_df in events[events.duplicated(cmp_cols_round)].groupby([
        Event.latitude.key + _suf,
        Event.longitude.key + _suf,
        Event.depth_km.key + _suf,
        Event.latitude.key + _suf
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


    cols = [
        Event.latitude,
        Event.longitude,
        Event.depth_km,
        Event.time,
        Event.magnitude,
        Event.eventid,
        Event.catalog
    ]
    if preferred_mag_types is not None:
        cols += [Event.mag_type]

    if get_row_count(engine, Event) > 0:
        uc_cols = [Event.eventid.key, Event.catalog.key]
        for saved_events in pd.concat(db2dfs(select(cols), engine)):
            events = events.merge(
                saved_events, how='left', on=uc_cols, suffixes=('', _suf)
            )
            events.merge(saved_events, on=[Event.eventid.key, Event.catalog.key])



def check_events_df_from_local_file(
    events: pd.DataFrame, engine: Engine, show_progress=False
):
    suffix_msg = "Check events file"
    dupes = events[[Event.eventid.key]].duplicated().sum()
    if dupes:
        raise FailedDownload(f'Events file contains {dupes:,} events with same '
                             f'`eventid` (1st column). {suffix_msg}')
    # check duplicated
    mag, time, lat, lon = Event.magnitude, Event.time, Event.latitude, Event.longitude  # noqa
    tmp_df = pd.DataFrame({
        # mag: events_df[mag.key],
        time: events[time.key].dt.round('s'),
        lat: events[lat.key].round(3),
        lon: events[lon.key].round(3)
    })
    dupes = tmp_df.duplicated().sum()
    if dupes:
        raise FailedDownload(f'Events file contains {dupes:,} events with similar '
                             f'spatio-temporal coordinates. {suffix_msg}')
    events_from_file = 0
    ws_ids = {
        row[0] for row in
        session.query(WebService).filter(WebService.url.like("file:/%"))
    }
    if ws_ids:
        events_from_file = session.query(func.count()).select_from(Event).\
            filter(Event.webservice_id.in_(ws_ids)).scalar()

    if not events_from_file:
        return

    logger.info("Checking events in local file vs. db")
    with get_progressbar(len(events) if show_progress else 0) as pbar:

        for ev_id, m_, t_, la_, lo_ in zip(
                events[Event.eventid.key],
                events[mag.key],
                events[time.key],
                events[lat.key],
                events[lon.key]
        ):
            pbar.update(1)
            db_val = session.query(mag, time, lat, lon).filter(
                Event.webservice_id.in_(ws_ids) & (Event.eventid == ev_id)
            ).first()
            if db_val is None or tuple(db_val) == (m_, t_, la_, lo_):
                continue
            raise FailedDownload(f'Event eventid={ev_id} is already stored in the '
                                 f'database with different magnitude or coordinates. '
                                 f'{suffix_msg}')
