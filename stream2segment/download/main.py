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
from multiprocessing import cpu_count
import concurrent.futures
import numpy as np
import pandas as pd
from sqlalchemy import or_, and_
from sqlalchemy.exc import SQLAlchemyError
# from obspy.taup.tau import TauPyModel
# from obspy.taup.helper_classes import TauModelError, SlownessModelError
from stream2segment.utils.url import urlread, read_async, URLException
from stream2segment.io.db.models import Class, Event, DataCenter, Segment, Station,\
    dc_get_other_service_url, Channel, WebService
from stream2segment.io.db.pd_sql_utils import withdata, dfrowiter, mergeupdate,\
    dbquery2df, insertdf, syncdf, insertdf_napkeys, updatedf
from stream2segment.download.utils import empty, urljoin, response2df, normalize_fdsn_dframe,\
    get_search_radius, UrlStats, stats2str, get_inventory_url, save_inventory,\
    get_events_list, locations2degrees, get_arrival_time, get_url_mseed_errorcodes, get_taumodel
from stream2segment.utils import strconvert, get_progressbar, yaml_load
from stream2segment.utils.mseedlite3 import MSeedError, unpack as mseedunpack
from stream2segment.utils.msgs import MSG
from stream2segment.utils.resources import get_ws_fpath
from stream2segment.io.utils import dumps_inv
from datetime import datetime
import psutil
import os
import gc

logger = logging.getLogger(__name__)

# ADDBUFSIZE = 10  # FIXME: add this as parameter!!!


class QuitDownload(Exception):
    """
    This is an exception that should be raised from each function of this module, when their OUTPUT
    is empty and thus would prevent the continuation of the program.
    Any function here THUS EXPECTS THEIR DATAFRAME INPUT TO BE NON-EMPTY.

    There are two causes for that:
    - no data because of a download error (no data fetched)
    - no data because of current settings (e.g., no channels with given sample rate, all segments
    already downloaded with current retry settings, etcetera)
    In both cases, the program should exit. Thus we implement an Exception to be able to try-catch
    when we cannot continue. But:
     - In the first case the program should `log.error` the message and return nonzero
     - In the second case the program should `log.info` the message and return zero
    Note that with the current settings defined in stream2segment/main, log.info and log.error both
    print to stdout (only `log.warning` and `log.debug` do not).

    So the usage from within each method is to raise a QuitDownload any time they would return
    empty data, i.e. empty DataFrame or None.
    The usage from within `run` is to try.. catch each method, and in case of QuitDownload
    simply write a method that prints to log (info or error according to the constructor) and
    returns the right program exit code:
    ```
        exit_code = exc.log()
    ```
    or if you want to return:
    ```
        return exc.log()
    ```
    """
    def __init__(self, exc_or_msg):
        """Creates a new QuitDownload instance
        :param exc_or_msg: if Exception, then this object will log.error in the `log()` method
        and return a nonzero exit code (error), otherwise (if string) this object will log.info
        inthere and return 0
        """
        super(QuitDownload, self).__init__(str(exc_or_msg))
        self._iserror = isinstance(exc_or_msg, Exception)

    def log(self):
        if self._iserror:
            logger.error(self)
            return 1  # that's the program return
        else:
            # use str(self) although MSG does not care
            # but in case the formatting will differ, as we are here not for an error,
            # we might be ready to distinguish the cases
            logger.info(str(self))
            return 0  # that's the program return, 0 means ok anyway


def get_eventws_url(session, service):
    """Returns the event web service url from a given service name (string),
    currently either "eida" or "iris"
    """
    dic = yaml_load(get_ws_fpath())
    try:
        return dic['seismicportal' if service == 'eida' else service]['event']
    except (IndexError, KeyError):
        raise ValueError("Service '%s'\\'s web service not found" % service)


def add_classes(session, class_labels, db_bufsize):
    """Inserts the given classes to the relative table
    :param class_labels: a dict of class_label mapped to the class description
    (e.g. "low_snr": "segment with low signal-to-noise ratio")
    """
    if class_labels:
        cdf = pd.DataFrame(data=[{Class.label.key: k, Class.description.key: v}
                           for k, v in class_labels.iteritems()])
        dbinsertdf(cdf, session, [Class.label], db_bufsize)


# FIXME: this method is just called from add_classes above. Worth using?
# We might move to dbsyncdf below but then the program would stop if no classes are added
# Is what we want? Or maybe move classes to GUI?
def dbinsertdf(dataframe, session, matching_columns, buf_size=10, query_first=True,
               drop_duplicates=True, return_df=True):
    """Calls `insertdf` and writes to the logger before returning the
    new dataframe."""
    oldlen = len(dataframe)
    df, new = insertdf(dataframe, session, matching_columns, buf_size, query_first,
                       drop_duplicates, return_df)
    discarded = oldlen - len(df)
    dblog(matching_columns[0].class_, new, discarded)
    return df


def dbsyncdf(dataframe, session, matching_columns, autoincrement_pkey_col, buf_size=10,
             drop_duplicates=True, return_df=True, cols_to_print_on_err=None):
    """Calls `syncdf` and writes to the logger before returning the
    new dataframe. Raises `QuitDownload` if the returned dataframe is empty (no items saved)"""
    oldlen = len(dataframe)
    df, new = syncdf(dataframe, session, matching_columns, autoincrement_pkey_col, buf_size,
                     drop_duplicates, return_df)
    table = autoincrement_pkey_col.class_
    if empty(df):
        raise QuitDownload(Exception(MSG("",
                                         "No item saved to table '%s'" % table.__tablename__,
                                         "unknown error, check database connection")))
    discarded = oldlen - len(df)
    dblog(table, new, discarded)
    if discarded:
        _df = dataframe[~dataframe.index.isin(df.index)]
        if cols_to_print_on_err is not None:
            _df = _df[cols_to_print_on_err]
        if not empty(_df):
            N = 50
            msg = (" (showing first %d only)" % N) if len(_df) > N else ""
            _df = _df.iloc[:N] if len(_df) > N else _df
            logger.warning("%s%s:\n%s" % ("Discarded items", msg, _df.to_string(index=False)))
    return df


def dblog(table, inserted, not_inserted, updated=-1, not_updated=-1):
    """Prints to log the result of a database wrtie operation.
    Use this function to harmonize the message format and make it more readable in log or
    terminal"""

    def item(num):
        return "item" if num == 1 else "items"
    _header = "Db table '%s'"
    _errmsg = "sql error, e.g. unique / not null constraint violation"
    _noerrmsg = "no sql errors"

    if inserted or not_inserted:
        if not_inserted:
            msgs = ["%d new %s inserted, %d discarded", _errmsg]
            args = [inserted, item(inserted), not_inserted]
        else:
            msgs = ["%d new %s inserted", _noerrmsg]
            args = [inserted, item(inserted)]
    else:
        msgs = ["no new item to insert", ""]
        args = []

    logger.info(MSG(_header, msgs[0], msgs[1]), table.__tablename__, *args)

    if not updated == -1 and not_updated == -1:  # do not log if we did not updated stuff
        return

    if updated or not_updated:
        if not_updated:
            msgs = ["%d %s updated, %d discarded", _errmsg]
            args = [updated, item(updated), not_updated]
        else:
            msgs = ["%d %s updated", _noerrmsg]
            args = [updated, item(updated)]
    else:
        msgs = ["no item to update", ""]
        args = []

    logger.info(MSG(_header, msgs[0], msgs[1]), table.__tablename__, *args)


def get_events_df(session, eventws_url, db_bufsize, **args):
    """
        Returns the events from an event ws query. Splits the results into smaller chunks
        (according to 'start' and 'end' parameters, if they are not supplied in **args they will
        default to `datetime(1970, 1, 1)` and `datetime.utcnow()`, respectively)
        In case of errors just raise, the caller is responsible of displaying messages to the
        logger, which is used in this function only for all those messages which should not stop
        the program
    """
    eventws_id = session.query(WebService.id).filter(WebService.url == eventws_url).scalar()
    if eventws_id is None:  # write url to table
        data = [("event", eventws_url)]
        df = pd.DataFrame(data, columns=[WebService.type.key, WebService.url.key])
        df = dbsyncdf(df, session, [WebService.url], WebService.id, db_bufsize)
        eventws_id = df.iloc[0][WebService.id.key]

    url = urljoin(eventws_url, format='text', **args)
    ret = []
    try:
        datalist = get_events_list(eventws_url, **args)
    except ValueError as exc:
        raise QuitDownload(exc)

    if len(datalist) > 1:
        logger.info(MSG("",
                        "Request was split into sub-queries, aggregating the results",
                        "Original request entity too large", url))

    for data, msg, url in datalist:
        if not data and msg:
            logger.warning(MSG("", "discarding request", msg, url))
        elif data:
            try:
                events_df = response2normalizeddf(url, data, "event")
                ret.append(events_df)
            except ValueError as exc:
                logger.warning(MSG("", "discarding response", exc, url))

    if not ret:  # pd.concat below raise ValueError if ret is empty:
        raise QuitDownload(Exception(MSG("", "",
                                         "No events found. Check config and log for details",
                                         url)))

    events_df = pd.concat(ret, axis=0, ignore_index=True, copy=False)
    events_df[Event.webservice_id.key] = eventws_id
    events_df = dbsyncdf(events_df, session,
                         [Event.eventid, Event.webservice_id], Event.id, db_bufsize,
                         cols_to_print_on_err=[Event.eventid.key])

    # try to release memory for unused columns (FIXME: NEEDS TO BE TESTED)
    return events_df[[Event.id.key, Event.magnitude.key, Event.latitude.key, Event.longitude.key,
                     Event.depth_km.key, Event.time.key]].copy()


def response2normalizeddf(url, raw_data, dbmodel_key):
    """Returns a normalized and harmonized dataframe from raw_data. dbmodel_key can be 'event'
    'station' or 'channel'. Raises ValueError if the resulting dataframe is empty or if
    a ValueError is raised from sub-functions
    :param url: url (string) or `urllib2.Request` object. Used only to log the specified
    url in case of wranings
    """

    dframe = response2df(raw_data)
    oldlen, dframe = len(dframe), normalize_fdsn_dframe(dframe, dbmodel_key)
    # stations_df surely not empty:
    if oldlen > len(dframe):
        logger.warning(MSG(dbmodel_key + "s", "%d item(s) discarded",
                           "malformed server response data, e.g. NaN's", url),
                       oldlen - len(dframe))
    return dframe


def get_dc_filterfunc(service):
    """returns a function for filtering datacenters based on service.
    Returns a function that accepts an url (datacenter or station url) and returns a boolean
    indicating whether or not that url matches the given service.
    :param service: string (case insensitive) denoting the service: "iris" returns True for
    iris nodes, any other string returns True for EIDA nodes"""
    if service:
        service = service.lower()
        if service == 'iris':
            return lambda x: "service.iris.edu" in x
        elif service in ('eida', 'seismicportal'):
            return lambda x: "service.iris.edu" not in x

    return lambda x: True


def get_datacenters_df(session, routing_service_url, service, channels, db_bufsize, **query_args):
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
    # remember: re.macth must be exact match
    channels_re = None if not channels else \
        re.compile("|".join("^%s$" % strconvert.wild2re(c) for c in channels))

    # define columns (sql-alchemy model attrs) and their string names (pandas col names) once:
    DC_SURL_COL = DataCenter.station_url
    DC_DURL_COL = DataCenter.dataselect_url
    DC_SURL_NAME = DC_SURL_COL.key
    DC_DURL_NAME = DC_DURL_COL.key
    DC_ID_COL = DataCenter.id

    # do not return only new datacenters, return all of them
    query_args['service'] = 'dataselect'
    query_args['format'] = 'post'
    url = urljoin(routing_service_url, **query_args)

    accept_dc = get_dc_filterfunc(service)
    dc_df = dbquery2df(session.query(DC_ID_COL, DC_SURL_COL, DC_DURL_COL)).reset_index(drop=True)
    # filter by service:
    dc_df = dc_df[dc_df.apply(lambda x: accept_dc(x[DC_DURL_NAME]), axis=1)]

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
                if any('KEA00' in k for k in dc_data_buf):
                    dfg = 9
                # get index of given dataselect url:
                if accept_dc(current_dc_url):
                    indices = dc_df.index[(dc_df[DC_DURL_NAME] == current_dc_url)].values
                    idx = indices[0] if len(indices) else None

                    # index not found? add the item:
                    if idx is None:
                        idx = len(dc_df)
                        current_dc_station_url = dc_get_other_service_url(current_dc_url)
                        dc_df = dc_df.append([{DC_DURL_NAME: current_dc_url,
                                               DC_SURL_NAME: current_dc_station_url}],
                                             ignore_index=True)  # this re-index the dataframe
                        dc_postdata.append(None)

                    if dc_data_buf:
                        dc_postdata[idx] = "\n".join(dc_data_buf)

                dc_data_buf = []

            if is_dc_line:
                current_dc_url = line

        dc_df = dbsyncdf(dc_df, session, [DC_SURL_COL], DataCenter.id, db_bufsize)
        return dc_df, dc_postdata

    except URLException as urlexc:
        if empty(dc_df):
            msg = MSG("", "routing service error, no data-center saved in database",
                      urlexc.exc, url)
            raise QuitDownload(Exception(msg))
        else:
            msg = MSG("",
                      "routing service error, trying to work with %d already saved data-centers",
                      urlexc.exc, url)
            logger.warning(msg, len(dc_df))
            logger.info(msg, len(dc_df))
            return dc_df, None


def get_channels_df(session, datacenters_df, post_data, channels, min_sample_rate,
                    max_thread_workers, timeout, blocksize, db_bufsize,
                    show_progress=False):
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
    # define columns (sql-alchemy model attrs) and their string names (pandas col names) once:
    CHA_ID = Channel.id
    CHA_STAID = Channel.station_id
    STA_LAT = Station.latitude
    STA_LON = Station.longitude
    STA_STIME = Station.start_time
    STA_ETIME = Station.end_time
    STA_DCID = Station.datacenter_id
    STA_NET = Station.network
    STA_STA = Station.station
    CHA_LOC = Channel.location
    CHA_CHA = Channel.channel
    COLS_DB = [CHA_ID, CHA_STAID, STA_LAT, STA_LON, STA_DCID, STA_STIME, STA_ETIME,
               STA_NET, STA_STA, CHA_LOC, CHA_CHA]
    COLS_DF = [c.key for c in COLS_DB]

    ret_df = empty()

    if post_data is None:
        logger.info("Getting already-saved stations and channels from db")
        expr1 = or_(*[Channel.channel.like(strconvert.wild2sql(cha)) for cha in channels]) \
            if channels else None
        expr2 = Channel.sample_rate >= min_sample_rate if min_sample_rate > 0 else None
        expr = expr1 if expr2 is None else expr2 if expr1 is None else and_(expr1, expr2)
        qry = session.query(*COLS_DB).join(Channel.station)
        if expr is not None:
            qry = qry.filter(expr)
        ret_df = dbquery2df(qry)
        if empty(ret_df):
            raise QuitDownload("No channel found in database according to given parameters")
    else:
        ret = []

        iterable = ((id_, Request(url, data='format=text\nlevel=channel\n'+post_data_str))
                    for url, id_, post_data_str in
                    izip(datacenters_df[DataCenter.station_url.key],
                         datacenters_df[DataCenter.id.key], post_data) if post_data_str)

        with get_progressbar(show_progress, length=len(datacenters_df)) as bar:
            for obj, result, exc, url in read_async(iterable, urlkey=lambda obj: obj[-1],
                                                    blocksize=blocksize,
                                                    max_workers=max_thread_workers,
                                                    decode='utf8', timeout=timeout):
                bar.update(1)
                dcen_id = obj[0]
                if exc:
                    logger.warning(MSG("", "unable to perform request", exc, url))
                else:
                    try:
                        df = response2normalizeddf(url, result[0], "channel")
                    except ValueError as exc:
                        logger.warning(MSG("", "discarding response data", exc, url))
                        df = empty()
                    if not empty(df):
                        df[Station.datacenter_id.key] = dcen_id
                        ret.append(df)

        if ret:  # pd.concat complains for empty list
            channels_df = pd.concat(ret, axis=0, ignore_index=True, copy=False)

            # remove unmatching sample rates:
            if min_sample_rate > 0:
                srate_col = Channel.sample_rate.key
                oldlen, channels_df = len(channels_df), \
                    channels_df[channels_df[srate_col] >= min_sample_rate]
                discarded_sr = oldlen - len(channels_df)
                if discarded_sr:
                    logger.warning(MSG("", "discarding %d channels",
                                       "sample rate < %s Hz" % str(min_sample_rate)),
                                   discarded_sr)

            if not empty(channels_df):
                # logger.info("Saving channels (%d) and relative stations", len(channels_df))
                channels_df = save_stations_and_channels(session, channels_df, db_bufsize)
                ret_df = channels_df[COLS_DF].copy()
            else:
                raise QuitDownload("No channel found with sample rate >= %f" % min_sample_rate)
        else:
            raise QuitDownload(Exception(MSG("", "No channel found",
                                             "server / client errors in server responses: "
                                             "check config and log for details")))

    return ret_df


def save_stations_and_channels(session, channels_df, db_bufsize):
    """
        Saves to db channels (and their stations) and returns a dataframe with only channels saved
        The returned data frame will have the column 'id' (`Station.id`) renamed to
        'station_id' (`Channel.station_id`) and a new 'id' column referring to the Channel id
        (`Channel.id`)
        :param channels_df: pandas DataFrame resulting from `get_channels_df`
    """
    # define columns (sql-alchemy model attrs) and their string names (pandas col names) once:
    STA_COLS_DB = [Station.network, Station.station, Station.start_time]
    STA_COLS_DF = [c.key for c in STA_COLS_DB]
    STA_ID_DB = Station.id
    STA_ID_DF = STA_ID_DB.key
    CHA_COLS_DB = [Channel.station_id, Channel.location, Channel.channel]
    CHA_COLS_DF = [c.key for c in [Channel.station_id, Station.network, Station.station,
                                   Channel.location, Channel.channel, Station.start_time,
                                   Station.datacenter_id]]
    CHA_ID_DB = Channel.id
    CHA_STAID_DF = Channel.station_id.key

    # Important: we do not need to allocate the pkey columns like:
    # channels_df[STA_ID_DF] = None
    # Because dbsync('sync'...) and
    # mergeupdate do that for us, also with the correct dtype (the line above allocates a
    # dtype=object  by default)

    # remember: dbsyncdf raises a QuitDownload, so no need to check for empty(dataframe)
    # attempt to write only unique stations. Purge stations now otherwise we get misleading
    # error message: "xxx stations discarded"
    stas_df = dbsyncdf(channels_df.drop_duplicates(subset=STA_COLS_DF).copy(), session, STA_COLS_DB,
                       STA_ID_DB, db_bufsize, cols_to_print_on_err=STA_COLS_DF)
    channels_df = mergeupdate(channels_df, stas_df, STA_COLS_DF, [STA_ID_DF])
    # add channels to db:
    channels_df = dbsyncdf(channels_df.rename(columns={STA_ID_DF: CHA_STAID_DF}),
                           session, CHA_COLS_DB, CHA_ID_DB, db_bufsize,
                           cols_to_print_on_err=CHA_COLS_DF)
    return channels_df


def merge_events_stations(events_df, channels_df, minmag, maxmag, minmag_radius, maxmag_radius):
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
    # define columns (sql-alchemy model attrs) and their string names (pandas col names) once:
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
    CHA_STAID = Channel.station_id.key
    OK_TMP_COL = "__i'm.ok!__"  # tmp col for channels to take

    channels_df = channels_df.rename(columns={Channel.id.key: Segment.channel_id.key})
    # get unique stations, rename Channel.id into Segment.channel_id now so we do not bother later
    stations_df = channels_df.drop_duplicates(subset=[CHA_STAID])
    stations_df.is_copy = False

    ret = []
    max_radia = get_search_radius(events_df[EVT_MAG].values, minmag, maxmag,
                                  minmag_radius, maxmag_radius)

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

        cha_df = mergeupdate(channels_df, stations_df, [CHA_STAID], [OK_TMP_COL, SEG_EVDIST],
                             drop_df_new_duplicates=False)  # dupes already dropped
        cha_df = cha_df[cha_df[OK_TMP_COL]]
        cha_df.is_copy = False
        cha_df[SEG_EVID] = evt_dic[EVT_ID]
        cha_df[EVT_DEPTH] = evt_dic[EVT_DEPTH]
        cha_df[EVT_TIME] = evt_dic[EVT_TIME]
        ret.append(cha_df.drop([STA_STIME, STA_ETIME, STA_LAT, STA_LON, OK_TMP_COL],
                               axis=1))

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
    # define columns (sql-alchemy model attrs) and their string names (pandas col names) once:
    SEG_EVID = Segment.event_id
    CHA_STAID = Channel.station_id
    SEG_EVDIST = Segment.event_distance_deg
    SEG_ATIME = Segment.arrival_time
    cols = [SEG_EVDIST, SEG_ATIME, CHA_STAID, SEG_EVID]
    query = session.query(*cols).join(Segment.channel).distinct()
    df_repl = dbquery2df(query)
    segments_df[SEG_ATIME.key] = pd.NaT  # necessary to coerce values to date later
    return segments_df if empty(df_repl) else mergeupdate(segments_df, df_repl,
                                                          [CHA_STAID.key, SEG_EVID.key],
                                                          [SEG_EVDIST.key, SEG_ATIME.key])


def get_arrivaltimes(segments_df, wtimespan, traveltime_phases, tau_model,
                     mp_max_workers=None,
                     show_progress=False):
    """
        Calculates the arrival times for those rows of `segments_df` wich are NaT

        :param segments_df: pandas DataFrame resulting from `set_saved_arrivaltimes`
    """
    # define columns (sql-alchemy model attrs) and their string names (pandas col names) once:
    SEG_EVID = Segment.event_id.key
    SEG_EVDIST = Segment.event_distance_deg.key
    SEG_ATIME = Segment.arrival_time.key
    SEG_STIME = Segment.start_time.key
    SEG_ETIME = Segment.end_time.key
    CHA_STAID = Channel.station_id.key
    EVT_DEPTH = Event.depth_km.key
    EVT_TIME = Event.time.key

    ptimes2calculate_df = segments_df[pd.isnull(segments_df[SEG_ATIME])].\
        drop_duplicates(subset=[SEG_EVID, CHA_STAID])
    patime_data = {SEG_ATIME: [], CHA_STAID: [], SEG_EVID: []}

    num_to_calculate = len(ptimes2calculate_df)

    # pre-allocate model (improves perfs)
    model = get_taumodel(tau_model)

    with get_progressbar(show_progress, length=num_to_calculate) as bar:
        # We can use a with statement to ensure threads are cleaned up promptly
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count() if not mp_max_workers
                                                    else mp_max_workers) as executor:
            future_to_evtid = {}
            for stadict in dfrowiter(ptimes2calculate_df, [SEG_EVDIST, EVT_DEPTH, EVT_TIME,
                                                           SEG_EVID, CHA_STAID]):
                future_to_evtid[executor.submit(get_arrival_time,
                                                stadict[SEG_EVDIST],
                                                stadict[EVT_DEPTH],
                                                stadict[EVT_TIME],
                                                traveltime_phases,
                                                0.0,
                                                model)] = stadict[SEG_EVID], stadict[CHA_STAID]

            for future in concurrent.futures.as_completed(future_to_evtid):
                bar.update(1)
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
                    logger.warning(MSG("", "discarding segment (event_id: %d, station_id: %d)"
                                       % (evt_id, sta_id), exc))

    if num_to_calculate == 0:
        logger.info("All P-arrival times and time ranges already calculated (fetched from db)")

    # assign data to segments:
    df_repl = pd.DataFrame(data=patime_data)
    segments_df = mergeupdate(segments_df, df_repl, [SEG_EVID, CHA_STAID], [SEG_ATIME])

    # drop errors in arrival time:
    oldlen = len(segments_df)
    segments_df.dropna(subset=[SEG_ATIME, SEG_EVDIST], inplace=True)
    segdf_len = len(segments_df)
    if not segdf_len:
        raise QuitDownload(Exception(MSG("", "No arrival times calculated",
                                             "internal error: "
                                             "check config and log for details")))
    if oldlen > segdf_len:
        logger.info(MSG("", "%d of %d segments discarded"), oldlen-segdf_len, oldlen)
    # set start time and end time:
    td0, td1 = timedelta(minutes=wtimespan[0]), timedelta(minutes=wtimespan[1])
    segments_df[SEG_STIME] = (segments_df[SEG_ATIME] - td0).dt.round('s')
    segments_df[SEG_ETIME] = (segments_df[SEG_ATIME] + td1).dt.round('s')
    # drop unnecessary columns and return:
    return segments_df.drop([EVT_DEPTH, EVT_TIME, CHA_STAID], axis=1)


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
    # define columns (sql-alchemy model attrs) and their string names (pandas col names) once:
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
    SEG_DCID_COL = Segment.datacenter_id
    SEG_DCID_NAME = SEG_DCID_COL.key

    URLERR_CODE, MSEEDERR_CODE = get_url_mseed_errorcodes()
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
        _mask = db_seg_df[SEG_DSC] == URLERR_CODE
        mask = _mask if mask is None else mask | _mask
    if retry_mseed_errors:
        _mask = db_seg_df[SEG_DSC] == MSEEDERR_CODE
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
    segments_df[SEG_ID] = np.nan  # coerce to valid type (should be int, however allow nans)
    segments_df[SEG_RETRY] = True  # coerce to valid type
    # segments_df[SEG_DSC] = None
    # segments_df[SEG_RUNID] = run_id
    segments_df = mergeupdate(segments_df, db_seg_df, [SEG_CHID, SEG_STIME, SEG_ETIME],
                              [SEG_ID, SEG_RETRY])

    oldlen, segments_df = len(segments_df), segments_df[segments_df[SEG_RETRY]]  # FIXME: copy???
    if oldlen != len(segments_df):
        reason = ", ".join("%s=%s" % (k, str(v)) for k, v in locals().iteritems()
                           if k.startswith("retry_"))
        logger.info(MSG("", "%d segments discarded", reason), oldlen-len(segments_df))

    if empty(segments_df):
        raise QuitDownload("Nothing to download: all segments already downloaded according to "
                           "the current configuration")

    # DO NOT REMOVE dupes anymore (commented out):
    # for safety, remove dupes (we should not have them, however...). FIXME: # add msg???
    # segments_df = segments_df.drop_duplicates(subset=[SEG_CHID, SEG_STIME, SEG_ETIME])

    # set index with post data line (string)
    segments_df = segments_df.set_index(_strcat(segments_df))
    segments_df.index.name = None  # just for better str display...
    # keep relevant data only. inplace should help memory management
    STA_NET = Station.network.key
    STA_STA = Station.station.key
    CHA_LOC = Channel.location.key
    CHA_CHA = Channel.channel.key
    segments_df.drop([STA_NET, STA_STA, CHA_LOC, CHA_CHA, SEG_RETRY], axis=1, inplace=True)
    return segments_df


def _strcat(segments_df):
    STA_NET = Station.network.key
    STA_STA = Station.station.key
    CHA_LOC = Channel.location.key
    CHA_CHA = Channel.channel.key
    n = segments_df[STA_NET].str.cat
    s = segments_df[STA_STA].str.cat
    l = segments_df[CHA_LOC].str.cat
    c = segments_df[CHA_CHA]
    return n(s(l(c, sep='.', na_rep=''), sep='.', na_rep=''), sep='.', na_rep='')


def _get_seg_request(segments_df, datacenter_url):
    """returns a Request object from the given segments_df"""
    SEG_STIME = Segment.start_time.key
    SEG_ETIME = Segment.end_time.key
    stime = segments_df[SEG_STIME].iloc[0].isoformat()
    etime = segments_df[SEG_ETIME].iloc[0].isoformat()
    frmt_str = "{} {} {} {} %s %s" % (stime, etime)
    post_data = "\n".join(frmt_str.format(*("--" if not _ else _ for _ in k.split(".")))
                          for k in segments_df.index)
    return Request(url=datacenter_url, data=post_data)


def _mem_percent(process=None):
    return 0 if process is None else process.memory_percent()


def download_save_segments(session, segments_df, datacenters_df, run_id,
                           max_thread_workers, timeout, download_blocksize, db_bufsize,
                           show_progress=False):

    """Downloads and saves the segments. segments_df MUST not be empty (this is not checked for)
        :param segments_df: the dataframe resulting from `prepare_for_download`
    """
    # define columns (sql-alchemy model attrs) and their string names (pandas col names) once:
    SEG_DCID_COL = Segment.datacenter_id
    SEG_DCID_NAME = SEG_DCID_COL.key
    DC_ID_COL = DataCenter.id
    DC_ID_NAME = DC_ID_COL.key
    DC_DSQU_COL = DataCenter.dataselect_url
    DC_DSQU_NAME = DC_DSQU_COL.key
    SEG_ID_COL = Segment.id
    # SEG_ID_NAME = SEG_ID_COL.key
    SEG_STIME_COL = Segment.start_time
    SEG_ETIME_COL = Segment.end_time
    SEG_STIME_NAME = SEG_STIME_COL.key
    SEG_ETIME_NAME = SEG_ETIME_COL.key
    # CHA_CHA_COL = Channel.channel
    # CHA_CHA_NAME = CHA_CHA_COL.key
    SEG_DATA_COL = Segment.data
    SEG_DSC_COL = Segment.download_status_code
    SEG_SEEDID_COL = Segment.seed_identifier
    SEG_MGR_COL = Segment.max_gap_ovlap_ratio
    SEG_SRATE_COL = Segment.sample_rate
    SEG_RUNID_COL = Segment.run_id

    SEG_DATA_NAME = SEG_DATA_COL.key
    SEG_DSC_NAME = SEG_DSC_COL.key
    SEG_SEEDID_NAME = SEG_SEEDID_COL.key
    SEG_MGR_NAME = SEG_MGR_COL.key
    SEG_SRATE_NAME = SEG_SRATE_COL.key
    SEG_RUNID_NAME = SEG_RUNID_COL.key

    URLERR_CODE, MSEEDERR_CODE = get_url_mseed_errorcodes()

    stats = defaultdict(lambda: UrlStats())

    datcen_id2url = datacenters_df.set_index([DC_ID_NAME])[DC_DSQU_NAME].to_dict()

    segmanager = DbManager(session, SEG_ID_COL, [SEG_RUNID_COL, SEG_DATA_COL, SEG_MGR_COL,
                                                 SEG_SRATE_COL, SEG_DSC_COL], db_bufsize)

    seg_groups = segments_df.groupby([SEG_DCID_NAME, SEG_STIME_NAME, SEG_ETIME_NAME], sort=False)

    # we assume it's the terminal, thus allocate the current process to track
    # memory overflows
    process = psutil.Process(os.getpid()) if show_progress else None

    with get_progressbar(show_progress, length=len(segments_df)) as bar:
        # seg group is an iterable of 2 element tuples. The first element is the tuple of keys[:idx]
        # values, and the second element is the dataframe
        itr = read_async(seg_groups,
                         urlkey=lambda obj: _get_seg_request(obj[1], datcen_id2url[obj[0][0]]),
                         raise_http_err=False,
                         max_workers=max_thread_workers,
                         timeout=timeout, blocksize=download_blocksize)

        while itr is not None:
            retries = []
            for df, result, exc, request in itr:

                # first check if we are not overflowing:
                if _mem_percent(process) > 90:
                    raise QuitDownload(Exception(MSG("", "QUITTING: please re-try downloading segments",
                                                     "too much memory used: %.2f%%" %
                                                     process.memory_percent())))
                _ = df[0]  # not used. NOTE that the type of this var changes according to
                # whether we are retriying from a  previous 413 http code or not
                df = df[1]  # copy data so that we do not have refs to the old dataframe
                # and hopefully the gc works better
                url = request.get_host()
                data, code, msg = result if not exc else (None, None, None)
                if code == 413 and len(df) > 1:
                    for i, postdatarow in enumerate(request.data.split("\n")):
                        retries.append((Request(url=request.get_full_url(), data=postdatarow),
                                        pd.DataFrame(df.iloc[i]).T))
                    continue
                # now copy, we are about to write the df:
                # for numpy types, see
                # https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html#specifying-and-constructing-data-types
                df = df.copy()
                df[SEG_DATA_NAME] = None
                df[SEG_MGR_NAME] = float('nan')
                df[SEG_SRATE_NAME] = float('nan')
                df[SEG_SEEDID_NAME] = None
                df[SEG_DSC_NAME] = float('nan')
                df[SEG_RUNID_NAME] = run_id
                if exc:
                    code = URLERR_CODE
                else:
                    if code >= 400:
                        exc = "%d: %s" % (code, msg)
                    else:
                        # init with empty data. We got no errors so
                        # if we have empty data skip below, and we already have all values set
                        if not data:
                            df.loc[:, SEG_DATA_NAME] = b''
                            df.loc[:, SEG_DSC_NAME] = code
                        else:
                            try:
                                resdict = mseedunpack(data)
                                oks = 0
                                values = []
                                for key, (data, s_rate, max_gap_ratio, err) in resdict.iteritems():
                                    if err is not None:
                                        values.append((None, None, None, None, MSEEDERR_CODE))
                                        stats[url][err] += 1
                                    else:
                                        values.append((data, s_rate, max_gap_ratio, key, code))
                                        oks += 1
                                df.loc[resdict.keys(), [SEG_DATA_NAME, SEG_SRATE_NAME, SEG_MGR_NAME,
                                                        SEG_SEEDID_NAME, SEG_DSC_NAME]] = values
                                stats[url]["%d: %s" % (code, msg)] += oks
                            except MSeedError as mseedexc:
                                code = MSEEDERR_CODE
                                exc = mseedexc
                            except Exception as unknown_exc:
                                code = None
                                exc = unknown_exc

                if exc is not None:
                    df.loc[:, SEG_DSC_NAME] = code
                    stats[url][exc] += len(df)
                    logger.warning(MSG("", "Unable to get waveform data", exc, request))

                segmanager.add(df)
                bar.update(len(df))
                del df  # remove ref to df (helps gc when done in DbMAnager??)

            if retries:
                itr = read_async(retries,
                                 urlkey=lambda obj: obj[0],
                                 raise_http_err=False,
                                 max_workers=max_thread_workers,
                                 timeout=timeout, blocksize=download_blocksize)
            else:
                break

    segmanager.close()  # flush remaining stuff to insert / update, if any
    return stats


def _get_sta_request(datacenter_url, network, station, start_time, end_time):
    """
    returns a Request object from the given station arguments to download the inventory xml"""
    # we need a endtime (ingv does not accept * as last param)
    # note :pd.isnull(None) is true, as well as pd.isnull(float('nan')) and so on
    et = datetime.utcnow().isoformat() if pd.isnull(end_time) else end_time.isoformat()
    post_data = " ".join("*" if not x else x for x in[network, station, "*", "*",
                                                      start_time.isoformat(), et])
    return Request(url=datacenter_url, data="level=response\n{}".format(post_data))


def save_inventories(session, stations_df, max_thread_workers, timeout,
                     download_blocksize, db_bufsize, show_progress=False):
    """Save inventories. Stations_df must not be empty (this is not checked for)"""

    _msg = "Unable to save inventory (station id=%d)"

    downloaded, errors, empty = 0, 0, 0
    dbmanager = DbManager(session, Station.id, [Station.inventory_xml], db_bufsize)
    with get_progressbar(show_progress, length=len(stations_df)) as bar:
        iterable = izip(stations_df[Station.id.key],
                        stations_df[DataCenter.station_url.key],
                        stations_df[Station.network.key],
                        stations_df[Station.station.key],
                        stations_df[Station.start_time.key],
                        stations_df[Station.end_time.key])
        for obj, result, exc, request in read_async(iterable,
                                                    urlkey=lambda obj: _get_sta_request(*obj[1:]),
                                                    max_workers=max_thread_workers,
                                                    blocksize=download_blocksize, timeout=timeout,
                                                    raise_http_err=True):
            bar.update(1)
            sta_id = obj[0]
            if exc:
                logger.warning(MSG("", _msg, exc, request), sta_id)
                errors += 1
            else:
                data, code, msg = result  # @UnusedVariable
                if not data:
                    empty += 1
                    logger.warning(MSG("", _msg, "empty response", request), sta_id)
                else:
                    downloaded += 1
                    dbmanager.add(pd.DataFrame({Station.id.key: [sta_id],
                                                Station.inventory_xml.key: [dumps_inv(data)]}))

    logger.info(("Summary of web service responses for station inventories:\n"
                 "downloaded %d\n"
                 "discarded: %d (empty response)\n"
                 "not downloaded: %d (client/server errors: check log for details)") %
                (downloaded, empty, errors))
    dbmanager.close()


class DbManager(object):
    """Class managing the insertion of table rows into db. As insertion/updates should
    be happening during download for not losing data in case of unexpected error, this class
    manages the buffer size for the insertion/ updates on the db"""

    def __init__(self, session, id_col, update_cols, bufsize):
        self.info = [0, 0, 0, 0]  # new, total_new, updated, updated_new
        self.inserts = []
        self.updates = []
        self.bufsize = bufsize
        self._num2insert = 0
        self._num2update = 0
        self.session = session
        self.id_col = id_col
        self.update_cols = update_cols
        self.table = id_col.class_
#         self.SEG_ID_COL = Segment.id
#         self.SEG_ID_NAME = self.SEG_ID_COL.key
#         self.UPD_COLS = [Segment.run_id, Segment.data, Segment.max_gap_ratio, Segment.sample_rate,
#                          Segment.download_status_code]

    def add(self, df):
        bufsize = self.bufsize
        mask = pd.isnull(df[self.id_col.key])
        if mask.any():
            if mask.all():
                dfinsert = df
                dfupdate = None
            else:
                dfinsert = df[mask]
                dfupdate = df[~mask]
        else:
            dfinsert = None
            dfupdate = df

        if dfinsert is not None:
            self.inserts.append(dfinsert)
            self._num2insert += len(dfinsert)
            if self._num2insert >= bufsize:
                self.insert()

        if dfupdate is not None:
            self.updates.append(dfupdate)
            self._num2update += len(dfupdate)
            if self._num2update >= bufsize:
                self.update()

    def insert(self):
        df = pd.concat(self.inserts, axis=0, ignore_index=True, copy=False, verify_integrity=False)
        total, new = insertdf_napkeys(df, self.session, self.id_col, len(df), return_df=False)
        info = self.info
        info[0] += new
        info[1] += total
        # cleanup:
        self._num2insert = 0
        self.inserts = []

    def update(self):
        df = pd.concat(self.updates, axis=0, ignore_index=True, copy=False, verify_integrity=False)
        total = len(df)
        updated = updatedf(df, self.session, self.id_col, self.update_cols, total, return_df=False)
        info = self.info
        info[2] += updated
        info[3] += total
        # cleanup:
        self._num2update = 0
        self.updates = []

    def close(self):
        """flushes remaining stuff to insert/ update, if any, prints to log updates and inserts"""
        if self.inserts:
            self.insert()
        if self.updates:
            self.update()
        new, ntot, upd, utot = self.info
        dblog(self.table, new, ntot - new, upd, utot - upd)


def print_stats(stats_dict, datacenters_df):
    # STATS PRINTING:
    # define functions to represent stats:
    def rfunc(row):
        """function for modifying each row display"""
        url_ = datacenters_df[datacenters_df[DataCenter.id.key] ==
                              row][DataCenter.station_url.key].iloc[0]
        return urlparse(url_).netloc

    def cfunc(col):
        """function for modifying each col display"""
        return col if col.find(":") < 0 else col[:col.find(":")]

    logger.info("Summary of web service responses for waveform segments:\n%s" %
                (stats2str(stats_dict, fillna=0, transpose=True, lambdacol=cfunc, sort='col') or
                 "(Nothing to show)"))


def run(session, run_id, start, end, service, eventws_query_args,
        search_radius,
        channels, min_sample_rate, inventory, traveltime_phases,
        wtimespan, retry_no_code, retry_url_errors, retry_mseed_errors, retry_4xx, retry_5xx,
        advanced_settings, class_labels=None, isterminal=False):
    """
        Downloads waveforms related to events to a specific path. FIXME: improve doc
    """
    # set blocksize if zero:
    if advanced_settings['download_blocksize'] <= 0:
        advanced_settings['download_blocksize'] = -1
    if advanced_settings['max_thread_workers'] <= 0:
        advanced_settings['max_thread_workers'] = None
    dbbufsize = min(advanced_settings['db_buf_size'], 1)

    __steps = 7 + inventory  # bool substraction works: 8 - True == 7
    stepiter = imap(lambda i: "%d of %d" % (i+1, __steps), xrange(__steps))

    # write the class labels:
    add_classes(session, class_labels, dbbufsize)

    startiso = start.isoformat()
    endiso = end.isoformat()

    # events and datacenters should print meaningful stuff
    # cause otherwise is unclear why the program stop so quickly
    logger.info("")
    logger.info("STEP %s: Requesting events", next(stepiter))
    eventws_url = get_eventws_url(session, service)
    try:
        events_df = get_events_df(session, eventws_url, dbbufsize, start=startiso, end=endiso,
                                  **eventws_query_args)
    except QuitDownload as dexc:
        return dexc.log()

    # Get datacenters, store them in the db, returns the dc instances (db rows) correctly added:
    logger.info("")
    logger.info("STEP %s: Requesting data-centers", next(stepiter))
    try:
        datacenters_df, postdata = get_datacenters_df(session,
                                                      advanced_settings['routing_service_url'],
                                                      service, channels, dbbufsize,
                                                      start=startiso, end=endiso)
    except QuitDownload as dexc:
        return dexc.log()

    logger.info("")
    logger.info(("STEP %s: Requesting stations and channels from %d %s"), next(stepiter),
                len(datacenters_df),
                'data-center' if len(datacenters_df) == 1 else 'data-centers')
    try:
        channels_df = get_channels_df(session, datacenters_df, postdata, channels,
                                      min_sample_rate,
                                      advanced_settings['max_thread_workers'],
                                      advanced_settings['s_timeout'],
                                      advanced_settings['download_blocksize'], dbbufsize,
                                      isterminal)
    except QuitDownload as dexc:
        return dexc.log()

    logger.info("")
    logger.info(("STEP %s: Selecting stations within search radius from %d events"), next(stepiter),
                len(events_df))
    try:
        segments_df = merge_events_stations(events_df, channels_df, search_radius['minmag'],
                                            search_radius['maxmag'], search_radius['minmag_radius'],
                                            search_radius['maxmag_radius'])
    except QuitDownload as dexc:
        return dexc.log()

    # help gc by deleting the (only) refs to unused dataframes
    del events_df
    del channels_df
    # session.expunge_all()  # for memory: https://stackoverflow.com/questions/30021923/how-to-delete-a-sqlalchemy-mapped-object-from-memory

    logger.info("")
    logger.info(("STEP %s: Calculating P-arrival times and time ranges"), next(stepiter))
    segments_df = set_saved_arrivaltimes(session, segments_df)
    # rename dataframe to make clear that now we have segments:
    try:
        segments_df = get_arrivaltimes(segments_df, wtimespan, traveltime_phases, 'ak135',
                                       None, isterminal)
    except QuitDownload as dexc:
        return dexc.log()

    logger.info("")
    logger.info(("STEP %s: Checking already downloaded segments"), next(stepiter))
    exit_code = 0
    try:
        segments_df = prepare_for_download(session, segments_df, retry_no_code,
                                           retry_url_errors, retry_mseed_errors, retry_4xx,
                                           retry_5xx)
        # download_save_segments raises a QuitDownload if there is no data, remember its
        # exitcode
        logger.info("")
        if empty(segments_df):
            logger.info("STEP %s: Skipping: No segment to download", next(stepiter))
        else:
            logger.info("STEP %s: Downloading %d segments and saving to db", next(stepiter),
                        len(segments_df))

        d_stats = download_save_segments(session, segments_df, datacenters_df, run_id,
                                         advanced_settings['max_thread_workers'],
                                         advanced_settings['w_timeout'],
                                         advanced_settings['download_blocksize'],
                                         dbbufsize,
                                         isterminal)
        # help gc by deleting the (only) refs to unused dataframes
        del segments_df
        gc.collect()  # this should free some memory usage
        logger.info("")
        print_stats(d_stats, datacenters_df)

    except QuitDownload as dexc:
        # we are here if:
        # 1) we didn't have segments in prepare_for... (quit download with string message)
        # 2) we ran out of memory in download_... (quitdownload with exception message

        # in any case store exit code and return it after we downloaded inventories, if any
        exit_code = dexc.log()
        if exit_code != 0:
            return exit_code

    if inventory:
        gc.collect()  # this should free some memory usage. Do it before segments downloading
        # so maybe we have better performances

        # query station id, network station, datacenter_url
        # for those stations with empty inventory_xml
        # AND at least one segment non emtpy/ull
        # Download inventories for those segments
        qry = session.query(Station.id, Station.network, Station.station, DataCenter.station_url,
                            Station.start_time, Station.end_time).\
            join(Station.datacenter).\
            filter((~withdata(Station.inventory_xml)) &
                   (Station.segments.any(withdata(Segment.data))))  # @UndefinedVariable
        sta_df = dbquery2df(qry)
        # stations = session.query(Station).filter(~withdata(Station.inventory_xml)).all()
        logger.info("")
        if empty(sta_df):
            logger.info(("STEP %s: Skipping: No station inventory to download"), next(stepiter))
        else:
            logger.info(("STEP %s: Downloading %d station inventories"), next(stepiter),
                        len(sta_df))
            save_inventories(session, sta_df,
                             advanced_settings['max_thread_workers'],
                             advanced_settings['i_timeout'],
                             advanced_settings['download_blocksize'], dbbufsize, isterminal)

    return exit_code
