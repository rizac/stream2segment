# -*- coding: utf-8 -*-
# from __future__ import print_function

"""module holding the basic functionalities of the download routine
   :Platform:
       Mac OSX, Linux
   :Copyright:
       Deutsches GFZ Potsdam <XXXXXXX@gfz-potsdam.de>
   :License:
       To be decided!
"""
import logging
import re
from collections import defaultdict
from datetime import timedelta  #, datetime
from urlparse import urlparse
from itertools import izip, imap  #, cycle
from urllib2 import Request
import concurrent.futures
import numpy as np
import pandas as pd
from sqlalchemy import or_, and_
from sqlalchemy.exc import SQLAlchemyError
# from obspy.taup.tau import TauPyModel
# from obspy.taup.helper_classes import TauModelError, SlownessModelError
from stream2segment.utils.url import urlread, read_async, URLException
from stream2segment.io.db.models import Class, Event, DataCenter, Segment, Station,\
    dc_get_other_service_url, Channel
from stream2segment.io.db.pd_sql_utils import commit, withdata, dfrowiter, dfupdate, add2db,\
    dbquery2df, syncnullpkeys, sync
from stream2segment.download.utils import empty, urljoin, response2df, normalize_fdsn_dframe,\
    get_search_radius, UrlStats, stats2str, get_inventory_url, save_inventory,\
    get_events_list, locations2degrees, get_arrival_time
from stream2segment.utils import mseedlite3, strconvert, get_progressbar
from stream2segment.utils.msgs import MSG

logger = logging.getLogger(__name__)

ADDBUFSIZE = 10  # FIXME: add this as parameter!!!


def dbsync(method, items_name, *a, **kw):
    if method == 'add2db':
        df, discarded, new = add2db(*a, **kw)
    elif method == 'syncnullpkeys':
        df, discarded, new = syncnullpkeys(*a, **kw)
    elif method == 'sync':
        df, discarded, new = sync(*a, **kw)
    else:
        raise ValueError('%s invalid method. Use "add2db", "syncnullpkeys" or "sync"', method)

    if discarded:
        logger.warning(MSG("syncing database (%s)" % items_name, "%d item(s) not saved",
                           "duplicates or violating SQL constraints"), discarded)
    if new:
        logger.info(MSG("syncing database (%s)" % items_name, "%d new item(s) saved"), new)

    return df


def get_events_df(session, eventws, **args):
    """
        Returns the events from an event ws query. Splits the results into smaller chunks
        (according to 'start' and 'end' parameters, if they are not supplied in **args they will
        default to `datetime(1970, 1, 1)` and `datetime.utcnow()`, respectively)
        In case of errors just raise, the caller is responsible of displaying messages to the
        logger, which is used in this function only for all those messages which should not stop
        the program
    """
    url = urljoin(eventws, format='text', **args)
    ret = []
    try:
        datalist = get_events_list(eventws, **args)
        if len(datalist) > 1:
            logger.info(MSG("events query",
                            "Request was splitted into sub-queries, aggregating the results",
                            "Original request entity too large", url))
        for data, msg, url in datalist:
            if not data and msg:
                logger.warning(MSG("events query", "discarding result", msg, url))
            elif data:
                try:
                    events_df = response2normalizeddf(url, data, "event")
                    ret.append(events_df)
                except ValueError as exc:
                    logger.warning(MSG("events query", "discarding query result", exc, url))
        events_df = pd.concat(ret, axis=0, ignore_index=True, copy=False)
        if empty(events_df):
            raise Exception("No events fetched")

        events_df = dbsync("add2db", "events", events_df, session, [Event.id],
                           add_buf_size=ADDBUFSIZE, query_first=True, drop_duplicates=True)

        # try to release memory for unused columns (FIXME: NEEDS TO BE TESTED)
        ret = events_df[[Event.id.key, Event.magnitude.key, Event.latitude.key, Event.longitude.key,
                         Event.depth_km.key, Event.time.key]].copy()
        del events_df
        return ret
    except Exception as exc:
        if hasattr(exc, 'exc'):  # stream2segment URLException
            exc = exc.exc
        msg = MSG("events query", "Quitting", exc, url)
        # logger.error(msg)
        raise ValueError(msg)


def response2normalizeddf(url, raw_data, dbmodel_key):
    """Returns a normalized and harmonized dataframe from raw_data. dbmodel_key can be 'event'
    'station' or 'channel'. Raises ValueError if the resulting dataframe is empty or if
    a ValueError is raised from sub-functions"""

    dframe = response2df(raw_data)
    oldlen, dframe = len(dframe), normalize_fdsn_dframe(dframe, dbmodel_key)
    # stations_df surely not empty:
    if oldlen > len(dframe):
        logger.warning(MSG(dbmodel_key + "s", "%d item(s) discarded",
                           "malformed server response data, e.g. NaN's", url),
                       oldlen - len(dframe))
    return dframe


def get_datacenters_df(session, channels, **query_args):
    """Queries 'http://geofon.gfz-potsdam.de/eidaws/routing/1/query' for all datacenters
    available
    :param query_args: any key value pair for the url. Note that 'service' and 'format' will
    be overiidden in the code with values of 'station' and 'format', repsecively
    :param channels: the channels to query. Can be None (=all channels) or a list of FDSN channels
    format (e.g. `['HH?', BHZ']`)
    :return: the tuple datacenters_df, post_requests where the latter is a list of strings
    (same length as `datacenters_df`) usable for requesting station or channels to the
    given data center (via e.g. `urllib2.Request(datacenter_url, data=post_request)`). Some
    elements of `post_requests` might be None. If `post_requests` is None, then the RS encountered
    a problem and we have some channels saved. If we do not have channels saved, an exception
    is raised. In this latter case, the caller is responsible of displaying messages to the
    logger, which is used in this function only for all those messages which should not stop
    the program
    """
    channels_re = None if not channels else \
        re.compile("|".join("^%s$" % strconvert.wild2re(c) for c in channels))  # re.macth must be exact match
    DC_SQU = DataCenter.station_query_url.key
    DC_DQU = DataCenter.dataselect_query_url.key

    # do not return only new datacenters, return all of them
    query_args['service'] = 'dataselect'
    query_args['format'] = 'post'
    url = urljoin('http://geofon.gfz-potsdam.de/eidaws/routing/1/query', **query_args)

    dc_df = dbquery2df(session.query(DataCenter.id, DataCenter.station_query_url,
                                     DataCenter.dataselect_query_url)).reset_index(drop=True)

    # add list for post data, to be populated in the loop below
    dc_postdata = [None] * len(dc_df)

    try:
        dc_result, status, msg = urlread(url, decode='utf8', raise_http_err=True)

        current_dc_url = None
        dc_data_buf = []

        dc_split = dc_result.split("\n")
        lastline = len(dc_split) - 1
        for i, line in enumerate(dc_split):
            is_dc_line = False

            if line:
                is_dc_line = line[:7] == "http://"

                if not is_dc_line:
                    accept_it = channels_re is None
                    if not accept_it:
                        spl = line.split(' ')
                        if len(spl) > 3:
                            cha = spl[3].strip()
                            if spl[3] == '*' or channels_re.match(cha):
                                accept_it = True

                    if accept_it:
                        dc_data_buf.append(line)

            if (i == lastline or is_dc_line) and current_dc_url is not None:
                # get index of given dataselect url:
                _ = dc_df.index[(dc_df[DC_DQU] == current_dc_url)].values
                idx = _[0] if len(_) else None
                # index not found? add the item:
                if idx is None:
                    idx = len(dc_df)
                    dc_df = dc_df.append([{DC_DQU: current_dc_url,
                                           DC_SQU: dc_get_other_service_url(current_dc_url)}],
                                         ignore_index=True)  # this re-index the dataframe
                    dc_postdata.append(None)

                if dc_data_buf:
                    dc_postdata[idx] = "\n".join(dc_data_buf)

                dc_data_buf = []

            if is_dc_line:
                current_dc_url = line

        dc_df = dbsync("sync", "data-centers", dc_df, session, [DataCenter.station_query_url],
                       DataCenter.id, add_buf_size=ADDBUFSIZE)

        # dc_postdata, dc_df = dc_df[DC_POSTDATA], dc_df.drop(DC_POSTDATA, axis=1)
        return dc_df, dc_postdata

    except URLException as urlexc:
        if empty(dc_df):
            msg = MSG("data-centers", "EIDA routing service error and no data-center in cache",
                      urlexc.exc, url)
            # logger.error(msg)
            raise ValueError(msg)
        else:
            msg = MSG("data-centers", "EIDA routing service error, working with already saved (%d)",
                      urlexc.exc, url)
            logger.warning(msg, len(dc_df))
            logger.info(msg, len(dc_df))
            return dc_df, None


def get_channels_df(session, datacenters_df, post_data, channels, min_sample_rate,
                    max_thread_workers, timeout, blocksize,
                    notify_progress_func=lambda *a, **v: None):
    """Returns a dataframe representing a query to the eida services (or the internal db
    if `post_data` is None) with the given argument.  The
    dataframe will have as columns the `key` attribute of any of the following db columns:
    ```
    [Channel.id, Station.latitude, Station.longitude, Station.datacenter_id]
    ```
    :param datacenters_df: the first item resulting from `get_datacenters_df` (pandas DataFrame)
    :param post_data: the second item resulting from `get_datacenters_df` (list of strings or None)
    If None, the internal db is queried with the given arguments
    :param channels: a list of string denoting the channels, or None for no filtering
    (all channels). Each string follows FDSN specifications (e.g. 'BHZ', 'H??'). This argument
    is not used if `post_data` is given (not None)
    :param min_sample_rate: minimum sampling rate, set to negative value for no-filtering
    (all channels)
    """
    CHA_ID = Channel.id
    CHA_STAID = Channel.station_id
    STA_LAT = Station.latitude
    STA_LON = Station.longitude
    STA_STIME = Station.start_time
    STA_ETIME = Station.end_time
    STA_DCID = Station.datacenter_id

    COLS_DB = [CHA_ID, CHA_STAID, STA_LAT, STA_LON, STA_DCID, STA_STIME, STA_ETIME]
    COLS_DF = [c.key for c in COLS_DB]

    ret_df = empty()

    if post_data is None:
        expr1 = or_(*[Channel.channel.like(strconvert.wild2sql(cha)) for cha in channels]) \
            if channels else None
        expr2 = Channel.sample_rate >= min_sample_rate if min_sample_rate > 0 else None
        expr = expr1 if expr2 is None else expr2 if expr1 is None else and_(expr1, expr2)
        qry = session.query(*COLS_DB).join(Channel.station)
        if expr is not None:
            qry = qry.filter(expr)
        ret_df = dbquery2df(qry)
    else:
        ret = []

        def ondone(obj, result, exc, url):  # pylint:disable=unused-argument
            """function executed when a given url has successfully downloaded data"""
            notify_progress_func(1)
            dcen_id, fullurl = obj[0], url.get_full_url()
            if exc:
                logger.warning(MSG("channels", "unable to perform request", exc, fullurl))
            else:
                try:
                    df = response2normalizeddf(fullurl, result[0], "channel")
                except ValueError as exc:
                    logger.warning(MSG("channels", "discarding response data", exc, fullurl))
                    df = empty()
                if not empty(df):
                    df[Station.datacenter_id.key] = dcen_id
                    ret.append(df)

        iterable = ((id_, Request(url, data='format=text\nlevel=channel\n'+post_data_str))
                    for url, id_, post_data_str in
                    izip(datacenters_df[DataCenter.station_query_url.key],
                         datacenters_df[DataCenter.id.key], post_data) if post_data_str)

        read_async(iterable, ondone, urlkey=lambda obj: obj[-1], blocksize=blocksize,
                   max_workers=max_thread_workers, decode='utf8', timeout=timeout)

        if ret:  # pd.concat complains for empty list
            channels_df = pd.concat(ret, axis=0, ignore_index=True, copy=False)

            # remove unmatching sample rates:
            if min_sample_rate > 0:
                srate_col = Channel.sample_rate.key
                oldlen, channels_df = len(channels_df), \
                    channels_df[channels_df[srate_col] >= min_sample_rate]
                reason = "sample rate < %s Hz" % str(min_sample_rate)
                if oldlen > len(channels_df):
                    logger.warning(MSG("channels", "discarding %d channels", reason))

            if not empty(channels_df):
                # logger.info("Saving channels (%d) and relative stations", len(channels_df))
                channels_df = save_stations_and_channels(channels_df, session)

                if not empty(channels_df):
                    ret_df = channels_df[COLS_DF].copy()
                    # does this free up memory? FIXME: check
                    del channels_df

    if empty(ret_df):
        msg = MSG("channels", 'No channel found, unable to continue',
                  'possible causes: eida routing service down and no channels in cache (db), '
                  'eida routing service not providing post data, '
                  'client/server errors in all data-center queries, ',
                  'no channel with matching sample rate')
        # logger.error(msg)
        raise ValueError(msg)
    return ret_df


def save_stations_and_channels(channels_df, session):
    """
        Saves to db channels (and their stations) and returns a dataframe with only channels saved
        The returned data frame will have the column 'id' (`Station.id`) renamed to
        'station_id' (`Channel.station_id`) and a new 'id' column referring to the Channel id
        (`Channel.id`)
        :param channels_df: pandas DataFrame resulting from `get_channels_df`
    """
    STA_COLS_DB = [Station.network, Station.station, Station.start_time]
    STA_COLS_DF = [c.key for c in STA_COLS_DB]
    STA_ID_DB = Station.id
    STA_ID_DF = STA_ID_DB.key
    CHA_COLS_DB = [Channel.station_id, Channel.location, Channel.channel]
    CHA_ID_DB = Channel.id
    CHA_STAID_DF = Channel.station_id.key

    # Important: we do not need to allocate the pkey columns like:
    # channels_df[STA_ID_DF] = None
    # Because dbsync('sync'...) and
    # dfupdate do that for us, also with the correct dtype (the line above allocates a dtype=object
    # by default)

    # attempt to write only unique stations
    stas_df = dbsync("sync", "stations", channels_df, session, STA_COLS_DB, STA_ID_DB,
                     add_buf_size=ADDBUFSIZE, drop_duplicates=True)
    if empty(stas_df):
        return empty(), empty()
    channels_df = dfupdate(channels_df, stas_df, STA_COLS_DF, [STA_ID_DF])
    channels_df.rename(columns={STA_ID_DF: CHA_STAID_DF}, inplace=True)
    # add to db:
    channels_df = dbsync("sync", "channels", channels_df, session, CHA_COLS_DB, CHA_ID_DB,
                         add_buf_size=ADDBUFSIZE, drop_duplicates=True)
    return channels_df


def save_inventories(session, stations, max_thread_workers, timeout,
                     download_blocksize, notify_progress_func=lambda *a, **v: None):
    def ondone(obj, result, exc, cancelled):
        notify_progress_func(1)
        sta, url = obj
        if exc:
            logger.warning(MSG("station inventories", "discarding query result", exc, url))
        else:
            if not result:
                logger.warning(MSG("station inventories", "discarding query result",
                                   "empty response", url))
                return
            try:
                save_inventory(result, sta)
            except (TypeError, SQLAlchemyError) as _:
                session.rollback()
                logger.warning(MSG("station inventories", "item (station id=%s) not saved", _, url),
                               str(sta.id))

    iterable = izip(stations, (get_inventory_url(sta) for sta in stations))
    read_async(iterable, ondone, urlkey=lambda obj: obj[1],
               max_workers=max_thread_workers,
               blocksize=download_blocksize, timeout=timeout)


def merge_events_stations(events_df, channels_df, sradius_minmag,
                          sradius_maxmag, sradius_minradius, sradius_maxradius):
    """
        Merges `events_df` and `channels_df` by returning a new dataframe representing all
        channels within a specific search radius. *Each row of the resturned data frame is
        basically a segment to be potentially donwloaded*.
        The returned dataframe will be the same as `channels_df` with one or more rows repeated
        (some channels might be in the search radius of several events), plus a column
        "event_id" (`Segment.event_id`) representing the event associated to that channel
        and two columns 'event_distance_deg', 'time' (representing the *event* time) and
        'depth_km' (representing the event depth in km)
        :param channels_df: pandas DataFrame resulting from `get_channels_df`
        :param events_df: pandas DataFrame resulting from `get_events_df`
    """
    CHA_STAID = Channel.station_id.key

    channels_df = channels_df.rename(columns={Channel.id.key: Segment.channel_id.key})
    # get unique stations, rename Channel.id into Segment.channel_id now so we do not bother later
    stations_df = channels_df.drop_duplicates(subset=[CHA_STAID])
    stations_df.is_copy = False
    # tmp col for channels to take
    OK_TMP_COL = "__i'm.ok!__"

    STA_LAT = Station.latitude.key
    STA_LON = Station.longitude.key
    EVT_LAT = Event.latitude.key
    EVT_LON = Event.longitude.key
    EVT_TIME = Event.time.key
    STA_STIME = Station.start_time.key
    EVT_DEPTH = Event.depth_km.key
    STA_ETIME = Station.end_time.key
    EVT_MAG = Event.magnitude.key
    EVT_ID = Event.id.key
    SEG_EVID = Segment.event_id.key
    SEG_EVDIST = Segment.event_distance_deg.key

    ret = []
    max_radia = get_search_radius(events_df[EVT_MAG].values, sradius_minmag, sradius_maxmag,
                                  sradius_minradius, sradius_maxradius)

    for max_radius, evt_dic in izip(max_radia, dfrowiter(events_df, [EVT_ID, EVT_LAT, EVT_LON,
                                                                     EVT_TIME, EVT_DEPTH])):
        l2d = locations2degrees(stations_df[STA_LAT], stations_df[STA_LON],
                                evt_dic[EVT_LAT], evt_dic[EVT_LON])
        condition = (l2d <= max_radius) & (stations_df[STA_STIME] <= evt_dic[EVT_TIME]) & \
                    (pd.isnull(stations_df[STA_ETIME]) |
                     (stations_df[STA_ETIME] >= evt_dic[EVT_TIME] + timedelta(days=1)))

        if not np.any(condition):
            continue

        stations_df[OK_TMP_COL] = condition
        stations_df[SEG_EVDIST] = l2d

        cha_df = dfupdate(channels_df, stations_df, [CHA_STAID], [OK_TMP_COL, SEG_EVDIST],
                          drop_df_new_duplicates=False)  # dupes already dropped
        cha_df = cha_df[cha_df[OK_TMP_COL]]
        cha_df.is_copy = False
        cha_df[SEG_EVID] = evt_dic[EVT_ID]
        cha_df[EVT_DEPTH] = evt_dic[EVT_DEPTH]
        cha_df[EVT_TIME] = evt_dic[EVT_TIME]
        ret.append(cha_df.drop([STA_STIME, STA_ETIME, STA_LAT, STA_LON, OK_TMP_COL], axis=1))

    return pd.concat(ret, axis=0, ignore_index=True, copy=True)


def set_saved_arrivaltimes(session, segments_df):
    """
        Set on `segments_df` the already calculated arrival times, i.e. the arrival time
        of saved segments with the same 'event_id' and 'station_id'.
        Adds a new column 'arrival_time' with NaT values for those rows (segments) where arrival
        time has not been calculated.

        :param segments_df: pandas DataFrame resulting from `merge_events_stations`
        :param session: sql-alchemy session asociated to the database

        Note: by comparing with station id and event id, we increase a lot performances
        (because we might avoid calculating the P-arrival time for many segments)
        but as drawback we cannot change P-arrival time configuration values
        for already saved segments. However, by
        re-calculating the P-arrival time for all segments when changing P-arrival time config
        we might end in slightly different time-spans for the same segment, which is not what we
        might want to, as it would save several copies of the - basically same - segment)
    """
    # define col labels (strings):
    SEG_EVID = Segment.event_id
    CHA_STAID = Channel.station_id
    SEG_EVDIST = Segment.event_distance_deg
    SEG_ATIME = Segment.arrival_time
    cols = [SEG_EVDIST, SEG_ATIME, CHA_STAID, SEG_EVID]
    query = session.query(*cols).join(Segment.channel).distinct()
    df_repl = dbquery2df(query)
    segments_df[SEG_ATIME.key] = pd.NaT  # necessary to coerce values to date later
    return segments_df if empty(df_repl) else dfupdate(segments_df, df_repl,
                                                       [CHA_STAID.key, SEG_EVID.key],
                                                       [SEG_EVDIST.key, SEG_ATIME.key])


def get_arrivaltimes(segments_df, wtimespan, traveltime_phases, taup_model,
                     notify_progress_func=lambda *a, **v: None):
    """
        Calculates the arrival times for those rows of `segments_df` wich are NaT

        :param segments_df: pandas DataFrame resulting from `set_saved_arrivaltimes`
    """
    # define col labels (strings):
    SEG_EVID = Segment.event_id.key
    SEG_EVDIST = Segment.event_distance_deg.key
    SEG_ATIME = Segment.arrival_time.key
    SEG_STIME = Segment.start_time.key
    SEG_ETIME = Segment.end_time.key
    CHA_STAID = Channel.station_id.key
    EVT_DEPTH = Event.depth_km.key
    EVT_TIME = Event.time.key

    ptimes2calculate_df = segments_df[pd.isnull(segments_df[SEG_ATIME])]
    patime_data = {SEG_ATIME: [], CHA_STAID: [], SEG_EVID: []}
    # We can use a with statement to ensure threads are cleaned up promptly
    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:

        future_to_evtid = {}
        for stadict in dfrowiter(ptimes2calculate_df, [SEG_EVDIST, EVT_DEPTH, EVT_TIME, SEG_EVID,
                                                       CHA_STAID]):
            future_to_evtid[executor.submit(get_arrival_time,
                                            stadict[SEG_EVDIST],
                                            stadict[EVT_DEPTH],
                                            stadict[EVT_TIME],
                                            traveltime_phases,
                                            taup_model)] = stadict[SEG_EVID], stadict[CHA_STAID]

        for future in concurrent.futures.as_completed(future_to_evtid):
            notify_progress_func(1)
            atime = None
            evt_id, sta_id = future_to_evtid[future]
            try:
                atime = future.result()
                # set arrival time only if non-null
                patime_data[SEG_ATIME].append(atime)
                patime_data[CHA_STAID].append(sta_id)
                patime_data[SEG_EVID].append(evt_id)
            except Exception as exc:
                # evt_id = atime = None
                logger.warning(MSG("segments (P-) arrival time calculation", "discarding segment",
                                   exc))

    # assign data to segments:
    df_repl = pd.DataFrame(data=patime_data)
    segments_df = dfupdate(segments_df, df_repl, [SEG_EVID, CHA_STAID], [SEG_ATIME])

    # drop errors in arrival time:
    oldlen = len(segments_df)
    segments_df.dropna(subset=[SEG_ATIME, SEG_EVDIST], inplace=True)
    if oldlen > len(segments_df):
        logger.info(MSG("segments (P-) arrival time calculation", "%d of %d segments discarded"),
                    oldlen-len(segments_df), len(segments_df))
    # set start time and end time:
    td0, td1 = timedelta(minutes=wtimespan[0]), timedelta(minutes=wtimespan[1])
    segments_df[SEG_STIME] = (segments_df[SEG_ATIME] - td0).dt.round('s')
    segments_df[SEG_ETIME] = (segments_df[SEG_ATIME] + td1).dt.round('s')
    # drop unnecessary columns and return:
    return segments_df.drop([EVT_DEPTH, EVT_TIME], axis=1)


# def create_segments_df(channels_df, evts_stations_df):
#     seg_df = channels_df.merge(evts_stations_df, how='left', on=[Channel.station_id.key])
#     seg_df.drop([Channel.station_id.key], inplace=True)
#     seg_df.rename(columns={Channel.id.key: Segment.channel_id.key}, inplace=True)
#     return seg_df


def prepare_for_download(session, segments_df, retry_no_code, retry_url_errors,
                         retry_mseed_errors, retry_4xx, retry_5xx):
    """
        Drops the segments which are already present on the database and updates the primary
        keys for those not present (adding them to the db).
        Adds three new columns to the returned Data frame:
        `Segment.id` and `Segment.download_status_code`

        :param session: the sql-alchemy session bound to an existing database
        :param segments_df: pandas DataFrame resulting from `get_arrivaltimes`
    """
    # init col labels:
    SEG_STIME = Segment.start_time.key
    SEG_ETIME = Segment.end_time.key
    SEG_CHID = Segment.channel_id.key
    SEG_ID = Segment.id.key
    SEG_DSC = Segment.download_status_code.key
    SEG_RETRY = "__do.download__"
    SEG_SEEDID = Segment.seed_identifier
    SEG_ID_DBCOL = Segment.id
    SEG_STIME_DBCOL = Segment.start_time
    SEG_ETIME_DBCOL = Segment.end_time
    SEG_CHID_DBCOL = Segment.channel_id
    SEG_DSC_DBCOL = Segment.download_status_code

    # we might use dbsync('sync', ...) which sets pkeys and updates non-existing, but then we
    # would issue a second db query to check which segments should be re-downloaded (retry).
    # As the segments table might be big (hundred of thousands of records) we want to optimize
    # db queries, thus we first "manually" set the existing pkeys with a SINGLE db query which
    # gets ALSO the status codes (whereby we know what to re-download), and AFTER we call we
    # call dbsync('syncpkeys',..) which sets the null pkeys.
    # This function is basically what dbsync('sync', ...) does with the addition that we set whcch
    # segments have to be re-downloaded, if any

    # query relevant data into data frame:
    db_seg_df = dbquery2df(session.query(SEG_ID_DBCOL, SEG_CHID_DBCOL, SEG_STIME_DBCOL,
                                         SEG_ETIME_DBCOL, SEG_DSC_DBCOL))

    # filter already downloaded:
    mask = None
    if retry_no_code:
        _mask = pd.isnull(db_seg_df[SEG_DSC])
        mask = _mask if mask is None else mask | _mask
    if retry_url_errors:
        _mask = db_seg_df[SEG_DSC] == -1
        mask = _mask if mask is None else mask | _mask
    if retry_mseed_errors:
        _mask = db_seg_df[SEG_DSC] == -2
        mask = _mask if mask is None else mask | _mask
    if retry_4xx:
        _mask = db_seg_df[SEG_DSC].between(400, 499.9999, inclusive=True)
        mask = _mask if mask is None else mask | _mask
    if retry_5xx:
        _mask = db_seg_df[SEG_DSC].between(500, 599.9999, inclusive=True)
        mask = _mask if mask is None else mask | _mask

    if mask is None:
        mask = False
    db_seg_df[SEG_RETRY] = mask

    # update existing dataframe:
    segments_df[SEG_ID] = None
    segments_df[SEG_RETRY] = True
    segments_df[SEG_DSC] = None
    segments_df = dfupdate(segments_df, db_seg_df, [SEG_CHID, SEG_STIME, SEG_ETIME],
                           [SEG_ID, SEG_RETRY])

    oldlen, segments_df = len(segments_df), segments_df[segments_df[SEG_RETRY]]
    if oldlen != len(segments_df):
        reason = ", ".join("%s=%s" % (k, str(v)) for k, v in locals().iteritems()
                           if k.startswith("retry_"))
        logger.info(MSG("segments", "%d segments discarded", reason), oldlen-len(segments_df))
    # segments_df.is_copy = False
    # drop unnecessary columns:
    segments_df.drop([SEG_RETRY], axis=1, inplace=True)

    # for safety, remove dupes (we should not have them, however...). FIXME: # add msg???
    segments_df = segments_df.drop_duplicates(subset=[SEG_CHID, SEG_STIME, SEG_ETIME])
    segments_df = dbsync("syncnullpkeys", "waveform segments", segments_df, session,
                         SEG_ID_DBCOL, add_buf_size=ADDBUFSIZE)

    return segments_df


def _get_download_dict(grouped_df, dc_key, seedid_key, stime_key, etime_key,
                       id_key):
    stime = grouped_df[stime_key].iloc[0].isoformat()
    etime = grouped_df[etime_key].iloc[0].isoformat()
    dcen = grouped_df[dc_key].iloc[0]
    ret = {'url': dcen}
    ret['mseedid2id'] = {}
    ret['post_lines'] = []
    for seedid, id_ in izip(grouped_df[seedid_key], grouped_df[id_key]):
        post_str = "%s %s %s" % (" ".join("--" if not _ else _ for _ in seedid.split(".")), stime,
                                 etime)
        ret['post_lines'].append(post_str)
        ret['mseedid2id'][seedid] = id_
    return ret


def get_download_dicts(segments_df, datacenters_df):
    SEG_STIME = Segment.start_time.key
    SEG_ETIME = Segment.end_time.key
    SEG_DCID = Segment.datacenter_id.key
    DC_ID = DataCenter.id.key
    DC_DSURL = DataCenter.dataselect_query_url.key
    STA_NET = Station.network.key
    STA_STA = Station.station.key
    CHA_LOC = Channel.location.key
    CHA_CHA = Channel.channel.key
    SEG_SEEDID = "__" + Segment.seed_identifier.key
    # POST_REQ_KEY = "__post_request__" + SEG_SEEDID

    # make a seedid key by concatenating n, s, l, c:
    segments_df[SEG_SEEDID] = \
        segments_df[STA_NET].str.cat(segments_df[STA_STA], sep='.', na_rep='').\
        str.cat(segments_df[CHA_LOC], sep='.', na_rep='').\
        str.cat(segments_df[CHA_CHA], sep='.', na_rep='')

    datacenters_df[SEG_DCID] = datacenters_df[DC_ID]
    segments_df = dfupdate(segments_df, datacenters_df, [SEG_DCID], [DC_DSURL],
                           drop_df_new_duplicates=True)
    keys = [SEG_DCID, SEG_STIME, SEG_ETIME]
    # group dataframe by?? channel? location? station?
    seg_groups = segments_df.groupby(keys).apply(_get_download_dict, DC_DSURL, SEG_SEEDID,
                                                 SEG_STIME, SEG_ETIME)

    return seg_groups


def download_segments(seg_dicts, max_error_count, max_thread_workers,
                      timeout, download_blocksize,
                      notify_progress_func=lambda *a, **v: None):
    # define column(s) string labels:
    SEG_DATA = Segment.data.key
    SEG_DCID = Segment.datacenter_id.key

    stats = defaultdict(lambda: UrlStats())
    if empty(seg_dicts):
        return stats
    errors = defaultdict(int)
    results = {}
    count = [0, len(seg_dicts)]
    total_saved = 0

    def ondone(obj, result, exc, url):
        """function executed when a given url has succesfully downloaded `data`"""
        notify_progress_func(1)
        count[0] += 1
        seedid2dbid = obj['mseedid2id']
        code = None
        if not exc:
            code, bytes_data = result[1], result[0]
            try:
                result.extend((seedid2dbid[seedid], seedid, data, code) for seedid, data in
                              mseedlite3.unpack(bytes_data).iteritems() if seedid in seedid2dbid)
            except mseedlite3.MSeedError as _:
                code = -2  # MSEED ERROR
                exc = _
        if exc:
            if code is None:
                code = -1
            logger.warning(MSG("segments", "discarding query result", exc, url))
            stats[url.get_host()][exc] += len(seedid2dbid)
        else:
            if len(results) >= ADDBUFSIZE or count[0] == count[1]:
                saved, discarded = save_wdata(results)
                if discarded:
                    logger.warning("%d segments not updated (db error)", discarded)
                total_saved += saved

    # now download Data:
    # we use the index as urls cause it's much faster when locating a dframe row compared to
    # df[df[df_urls_colname] == some_url]). We zip it with the datacenters for faster search
    # REMEMBER: iterating over series values is FASTER BUT USES underlying numpy types, and for
    # datetime's is a problem cause pandas sublasses datetime, numpy not
    read_async(seg_dicts,
               urlkey=lambda obj: Request(url=obj['url'], data='\n'.join(obj['post_lines'])),
               ondone=ondone, max_workers=max_thread_workers,
               timeout=timeout, blocksize=download_blocksize)

    if total_saved:
        logger.info("%d segments updated", total_saved)

    # reset index (we do not need urls anymore):
    return stats


def save_wdata(session, seg_list, run_id, sync_session_on_update='evaluate'):
    SEG_ID = Segment.id.key
    SEG_RUNID = Segment.run_id.key
    SEG_DATA = Segment.data.key
    # SEG_DCID = Segment.datacenter_id.key
    SEG_SEEDID = Segment.seed_identifier
    SEG_DSC = Segment.download_status_code

    for id_, seed_id, data, code in seg_list:
        session.query(Segment).filter(Segment.id == id_).\
            update({SEG_DATA: data, SEG_RUNID: run_id, SEG_DSC: code, SEG_SEEDID: seed_id},
                   synchronize_session=sync_session_on_update)

    if commit(session):
        return len(seg_list), 0
    else:
        return 0, len(seg_list)

# def save_segments(session, segments_df, run_id, sync_session_on_update='evaluate'):
#     SEG_ID = Segment.id.key
#     SEG_RUNID = Segment.run_id.key
#     SEG_DATA = Segment.data.key
#     SEG_DCID = Segment.datacenter_id.key
# 
#     if empty(segments_df):
#         logger.info("No segment to save")
# 
#     ids_set = segments_df[SEG_ID].notnull()  # or pd.isnull(segments_df)
#     # add new segments
#     segments_to_add_df = segments_df[~ids_set]
#     segments_to_add_df.is_copy = False
#     segments_to_add_df[SEG_RUNID] = run_id
#     tot = len(segments_to_add_df)
#     adder = Adder(session, Segment.id, ADDBUFSIZE)
#     # hack to avoid checking if exists:
#     adder.existing_keys = set()
#     adder.add(segments_to_add_df)
#     if adder.discarded:
#         logger.warning("%d of %d new segments not saved (db error)", adder.discarded, tot)
#     # update new segments:
#     segments_to_update_df = segments_df[ids_set]
#     tot = len(segments_to_update_df)  # get now the num of segment sot update, and filter again:
#     # count where data is none, and # update stats
#     segments_to_update_df = segments_to_update_df[segments_df[SEG_DATA].notnull()]
#     if tot - len(segments_to_update_df):
#         logger.warning("%d segments discarded (retried, but still zero bytes "
#                        "received or client/server error)", tot - len(segments_to_update_df))
#     # write to db:
#     last = len(segments_to_update_df) - 1
#     buf = []
#     updated = 0
#     errors = 0
#     for i, rowdict in enumerate(dfrowiter(segments_to_update_df)):
#         # now we will either update or add the new segment. Set the run_id first:
#         # already downloaded, but this time data was found
#         # (if we attempted an already downloaded, it means segment.data was empty or None):
#         # note that we do not update run_id
#         session.query(Segment).filter(Segment.id == rowdict[SEG_ID]).\
#             update({SEG_DATA: rowdict[SEG_DATA], SEG_RUNID: run_id},
#                    synchronize_session=sync_session_on_update)
#         buf.append([rowdict[SEG_DCID], True if rowdict[SEG_DATA] else False])
#         if (i == last and buf) or len(buf) == ADDBUFSIZE:
#             if commit(session):
#                 updated += len(buf)
#             else:
#                 # we rolled back. We should actually count which has been really written!
#                 errors += len(buf)
#             buf = []
#     if errors:
#         logger.warning("%d segments not updated (db error)", errors)
#     if updated:
#         # also add as info (which by default prints to screen, if run from terminal):
#         logger.info("%d segments updated", updated)
#     if adder.new:
#         logger.info("%d new segments saved", adder.new)


def add_classes(session, class_labels):
    if class_labels:
        cdf = pd.DataFrame(data=[{Class.label.key: k, Class.description.key: v}
                           for k, v in class_labels.iteritems()])
        dbsync("add2db", "classes", cdf, session, [Class.label], ADDBUFSIZE, True)


def main(session, run_id, start, end, eventws, eventws_query_args,
         sradius_minmag, sradius_maxmag, sradius_minradius, sradius_maxradius,
         channels, min_sample_rate, download_s_inventory, traveltime_phases,
         wtimespan, retry_no_code, retry_url_errors, retry_mseed_errors, retry_4xx, retry_5xx,
         advanced_settings, class_labels=None, isterminal=False):
    """
        Downloads waveforms related to events to a specific path
        :param eventws: Event WS to use in queries. E.g. 'http://seismicportal.eu/fdsnws/event/1/'
        :type eventws: string
        :param eventws_query_args: a dict of fdsn arguments to be passed to the eventws query. E.g.
        {'minmag' : 3.0}
        :param search_radius_args: The arguments required to get the search radius R whereby all
            stations within R will be queried from a given event location E_lat, E_lon
        :type search_radius_args: list or iterable of numeric values:
            (min_magnitude, max_magnitude, min_distance, max_distance)
        :param datacenters_dict: a dict of data centers as a dictionary of the form
            {name1: url1, ..., nameN: urlN} where url1, url2,... are strings
        :type datacenters_dict dict of key: string entries
        :param channels: iterable (e.g. list) of channels (as strings), e.g.
            ['HH?', 'SH?', 'BH?', 'HN?', 'SN?', 'BN?']
        :type channels: iterable of strings
        :param start: Limit to events on or after the specified start time
            E.g. (date.today() - timedelta(days=1))
        :type start: datetime
        :param end: Limit to events on or before the specified end time
            E.g. date.today().isoformat()
        :type end: datetime
        :param ptimespan: the minutes before and after P wave arrival for the waveform query time
            span
        :type ptimespan: iterable of two float
        :param min_sample_rate: the minimum sample rate required to download data
        channels with a field 'SampleRate' lower than this value (in Hz) will be discarded and
        relative data not downloaded
        :type min_sample_rate: float
        :param session: sql alchemy session object
        :type outpath: string
    """
    # set blocksize if zero:
    if advanced_settings['download_blocksize'] <= 0:
        advanced_settings['download_blocksize'] = -1
    if advanced_settings['max_thread_workers'] <= 0:
        advanced_settings['max_thread_workers'] = None

    progressbar = get_progressbar(isterminal)

    __steps = 9 if download_s_inventory else 8
    stepiter = imap(lambda i: "%d of %d" % (i+1, __steps), xrange(__steps))

    # write the class labels:
    add_classes(session, class_labels)

    startiso = start.isoformat()
    endiso = end.isoformat()

    # events and datacenters should raise exceptions so that we can print the message in case.
    # Delegating the log only does not help a user when the program stops so quickly
    try:
        # Get events, store them in the db, returns the event instances (db rows) correctly added:
        logger.info("")
        logger.info("STEP %s: Querying events", next(stepiter))
        events_df = get_events_df(session, eventws, start=startiso, end=endiso,
                                  **eventws_query_args)
        # Get datacenters, store them in the db, returns the dc instances (db rows) correctly added:
        logger.info("")
        logger.info("STEP %s: Querying datacenters", next(stepiter))
        datacenters_df, postdata = get_datacenters_df(session, start=startiso, end=endiso)
    except Exception as exc:
        if isterminal:
            print str(exc)
        return 1

    logger.info("")
    if postdata is None:
        logger.info(("STEP %s: Getting stations from db"), next(stepiter),
                    len(datacenters_df))
        _pbar = get_progressbar(False)
    else:
        logger.info(("STEP %s: Querying stations from %d datacenter(s)"), next(stepiter),
                    len(datacenters_df))
        _pbar = progressbar
    with _pbar(length=len(datacenters_df)) as bar:
        channels_df = get_channels_df(session, datacenters_df, postdata, channels, min_sample_rate,
                                      advanced_settings['max_thread_workers'],
                                      advanced_settings['s_timeout'],
                                      advanced_settings['download_blocksize'], bar.update)

    if download_s_inventory:
        stations = session.query(Station).filter(~withdata(Station.inventory_xml)).all()
        logger.info("")
        logger.info(("STEP %s: Downloading %d stations inventories"), next(stepiter), len(stations))
        with progressbar(length=len(stations)) as bar:
            save_inventories(session, stations,
                             advanced_settings['max_thread_workers'],
                             advanced_settings['i_timeout'],
                             advanced_settings['download_blocksize'], bar.update)

    logger.info(("STEP %s: Selecting stations within search radius from %d events"), next(stepiter),
                len(events_df))
    segments_df = merge_events_stations(channels_df, events_df, sradius_minmag,
                                        sradius_maxmag, sradius_minradius, sradius_maxradius)

    logger.info("")
    logger.info(("STEP %s: Calculating P-arrival times "
                 "and time ranges"), next(stepiter))
    segments_df = set_saved_arrivaltimes(session, segments_df)
    with progressbar(length=segments_df[Segment.arrival_time.key].isnull().sum()) as bar:
        # rename dataframe to make clear that now we have segments:
        segments_df = get_arrivaltimes(segments_df, wtimespan, traveltime_phases, 'ak135',
                                       bar.update)

    # merging into channels_df:
    # segments_df = create_segments_df(channels_df, segments_df)

    logger.info("")
    logger.info(("STEP %s: Checking already downloaded segments"), next(stepiter))
    segments_df = prepare_for_download(session, segments_df, retry_no_code, retry_url_errors,
                                       retry_mseed_errors, retry_4xx, retry_5xx)

    seg_groups = get_download_dicts(segments_df, datacenters_df)

    segments_count = len(segments_df)
    logger.info("")
    logger.info("STEP %s: Querying Datacenter WS for %d segments and saving to db",
                next(stepiter), segments_count)

    with progressbar(length=seg_groups) as bar:
        d_stats = download_segments(seg_groups,
                                    advanced_settings['w_maxerr_per_dc'],
                                    advanced_settings['max_thread_workers'],
                                    advanced_settings['w_timeout'],
                                    advanced_settings['download_blocksize'],
                                    bar.update)

#     logger.info("")
#     logger.info("STEP %s: Saving waveform segments to db", next(stepiter))
#     save_segments(session, segments_df, run_id, sync_session_on_update=False)
#     logger.info("")

    # define functions to represent stats:
    def rfunc(row):
        """function for modifying each row display"""
        url_ = datacenters_df[datacenters_df[DataCenter.id.key] ==
                              row][DataCenter.station_query_url.key].iloc[0]
        return urlparse(url_).netloc

    def cfunc(col):
        """function for modifying each col display"""
        return col if col.find(":") < 0 else col[:col.find(":")]

#     logger.info("Summary Station WS query info:")
#     logger.info(stats2str(s_stats, fillna=0, transpose=True, lambdarow=rfunc, lambdacol=cfunc,
#                           sort='col'))
#     logger.info("")
    logger.info("Summary of Data-center 'dataselect' requests:")
    logger.info(stats2str(d_stats, fillna=0, transpose=True, lambdarow=rfunc, lambdacol=cfunc,
                          sort='col'))
    logger.info("")

    return 0



# def get_datacenters_df(session, **query_args):
#     """Queries 'http://geofon.gfz-potsdam.de/eidaws/routing/1/query' for all datacenters
#     available
#     Rows already existing (comparing by datacenter station_query_url) are returned as well,
#     but not added again
#     :param query_args: any key value pair for the url. Note that 'service' and 'format' will
#     be overiidden in the code with values of 'station' and 'format', repsecively
#     """
#     # do not return only new datacenters, return all of them
#     query_args['service'] = 'station'
#     query_args['format'] = 'post'
#     url = urljoin('http://geofon.gfz-potsdam.de/eidaws/routing/1/query', **query_args)
#     # add datacenters. Sql-alchemy way, cause we want the ids be autogenerated by the db
#     sta2ids = {k.station_query_url: k for k in session.query(DataCenter)}
#     try:
#         dc_result = urlread(url, decode='utf8')
#         # add to db the datacenters read. Two little hacks:
#         # 1) parse dc_result string and assume any new line starting with http:// is a valid station
#         # url url
#         # 2) When adding the datacenter, the table column dataselect_query_url (when not provided,
#         # as in this case) is assumed to be the same as station_query_url by replacing "/station"
#         # with "/dataselect". See https://www.fdsn.org/webservices/FDSN-WS-Specifications-1.1.pdf
#         datacenters = [DataCenter(station_query_url=dcen) for dcen in dc_result.split("\n")
#                        if dcen[:7] == "http://"]
#         dropped = 0
#         new = 0
#         for d in datacenters:
#             if d.station_query_url not in sta2ids:
#                 try:
#                     session.add(d)
#                     session.commit()
#                     new += 1
#                     sta2ids[d.station_query_url] = d
#                 except SQLAlchemyError as _:
#                     session.rollback()
#                     dropped += 1
# 
#         if dropped:
#             logger.warning(msgs.format(msgs.DB_ITEM_DISCARDED_(dropped), url))
#         if new:
#             logger.info(msgs.DB_ITEM_DISCARDED_(new))
# 
#     except URLException as urlexc:
#         msg = msgs.format(urlexc.exc, url)
#         if not sta2ids:
#             logger.error(msg)
#             raise ValueError(msg)
#         else:
#             logger.warning(msg)
#             logger.info(msgs.format("No datacenter downloaded, working with already "
#                                     "saved datacenters (%d)" % len(datacenters)))
#     dc_df = pd.DataFrame(data=[{DataCenter.id.key: x.id,
#                                 DataCenter.station_query_url.key: x.station_query_url,
#                                 DataCenter.dataselect_query_url.key: x.dataselect_query_url}
#                                for x in sta2ids.values()])
# 
#     return dc_df




# def get_fdsn_channels_df(session, events_df, datacenters_df, sradius_minmag, sradius_maxmag,
#                          sradius_minradius, sradius_maxradius, station_timespan, channels,
#                          min_sample_rate, max_thread_workers, timeout, blocksize,
#                          notify_progress_func=lambda *a, **v: None):
#     """Returns dict {event_id: stations_df} where stations_df is an already normalized and
#     harmonized dataframe with the stations saved or already present, and the JOINT fields
#     of Station and Channel. The id column values refers to Channel id's
#     though"""
#     stats = defaultdict(lambda: UrlStats())
#     ret = []
# 
#     def ondone(obj, result, exc, cancelled):  # pylint:disable=unused-argument
#         """function executed when a given url has successfully downloaded data"""
#         notify_progress_func(1)
#         evt_id, dcen_id, url = obj[0], obj[1], obj[2]
#         if cancelled:  # should never happen, however...
#             msg = "Download cancelled"
#             logger.warning(msgs.format(msg, url))
#             stats[dcen_id][msg] += 1
#         elif exc:
#             logger.warning(msgs.format(exc, url))
#             stats[dcen_id][exc] += 1
#         else:
#             df = get_stations_df(url, result, min_sample_rate)
#             if empty(df):
#                 if result:
#                     stats[dcen_id]['Malformed'] += 1
#                 else:
#                     stats[dcen_id]['Empty'] += 1
#             else:
#                 stats[dcen_id]['OK'] += 1
#                 df[Station.datacenter_id.key] = dcen_id
#                 df[Segment.event_id.key] = evt_id
#                 ret.append(df)
# 
#     iterable = evdcurl_iter(events_df, datacenters_df, sradius_minmag, sradius_maxmag,
#                             sradius_minradius, sradius_maxradius, station_timespan, channels)
# 
#     read_async(iterable, ondone, urlkey=lambda obj: obj[-1], blocksize=blocksize,
#                max_workers=max_thread_workers, decode='utf8', timeout=timeout)
# 
#     return pd.concat(ret, axis=0, ignore_index=True, copy=False), stats
# 
# 
# def evdcurl_iter(events_df, datacenters_df, sradius_minmag, sradius_maxmag, sradius_minradius,
#                  sradius_maxradius, station_timespan, channels):
#     """returns an iterable of tuple (event, datacenter, station_query_url) where the last element
#     is build with sradius_* arguments and station_timespan and channels"""
#     # calculate search radia:
#     max_radia = get_search_radius(events_df[Event.magnitude.key].values, sradius_minmag,
#                                   sradius_maxmag, sradius_minradius, sradius_maxradius)
# 
#     # dfrowiter yields dicts with pythojn objects
#     python_dcs = list(dfrowiter(datacenters_df, [DataCenter.id.key,
#                                                  DataCenter.station_query_url.key]))
# 
#     for max_radius, evt_dict in izip(max_radia, dfrowiter(events_df)):
#         evt_time = evt_dict[Event.time.key]
#         evt_lat = evt_dict[Event.latitude.key]
#         evt_lon = evt_dict[Event.longitude.key]
#         start = evt_time - timedelta(hours=station_timespan[0])
#         end = evt_time + timedelta(hours=station_timespan[1])
#         evt_id = evt_dict[Event.id.key]
#         for dc_dict in python_dcs:
#             dcen_station_query_url = dc_dict[DataCenter.station_query_url.key]
#             url = urljoin(dcen_station_query_url,
#                           latitude="%3.3f" % evt_lat,
#                           longitude="%3.3f" % evt_lon,
#                           maxradius=max_radius,
#                           start=start.isoformat(), end=end.isoformat(),
#                           channel=','.join(channels), format='text', level='channel')
#             yield (evt_id, dc_dict[DataCenter.id.key], url)
# 
# 
# def get_stations_df(url, raw_data, min_sample_rate):
#     """FIXME: write doc! """
#     try:
#         stations_df = query2df(url, raw_data, "channel")
#     except ValueError as exc:
#         logger.warning(msgs.format(exc, url))
#         return empty()
# 
#     if min_sample_rate > 0:
#         srate_col = Channel.sample_rate.key
#         oldlen, stations_df = len(stations_df), \
#             stations_df[stations_df[srate_col] >= min_sample_rate]
#         reason = "sample rate < %s Hz" % str(min_sample_rate)
#         if oldlen > len(stations_df):
#             logger.warning("%s: %s", reason,
#                            msgs.format(msgs.DB_ITEM_DISCARDED_(oldlen-len(stations_df)), url))
# 
#         if empty(stations_df):
#             return stations_df
#         # http://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
#         stations_df.is_copy = False
# 
#     return stations_df  # might be empty
# 
# 
