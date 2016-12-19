# -*- coding: utf-8 -*-
# from __future__ import print_function

"""query_utils: utilities of the package

   :Platform:
       Mac OSX, Linux
   :Copyright:
       Deutsches GFZ Potsdam <XXXXXXX@gfz-potsdam.de>
   :License:
       To be decided!
"""
import logging
import urllib2
import httplib
import socket
from datetime import timedelta
import numpy as np
import pandas as pd
from click import progressbar as click_pbar
from obspy.taup.tau import TauPyModel
from obspy.geodetics.base import locations2degrees
from obspy.taup.helper_classes import TauModelError, SlownessModelError
from stream2segment.utils import msgs, get_progressbar
from stream2segment.utils.url import url_read
from stream2segment.classification import class_labels_df
from stream2segment.io.db import models
from stream2segment.io.db.pd_sql_utils import df2dbiter, get_or_add_iter, commit, colnames
from stream2segment.utils.url import read_async
from stream2segment.download.utils import empty, get_query, query2dframe, normalize_fdsn_dframe,\
    get_search_radius, appenddf, get_arrival_time, UrlStats, stats2str, get_inventory_query
from urlparse import urlparse
from itertools import izip, imap, repeat
from sqlalchemy.exc import SQLAlchemyError
from stream2segment.io.dataseries import dumps_inv
import concurrent.futures


logger = logging.getLogger(__name__)


def get_events(session, eventws, minmag, minlat, maxlat, minlon, maxlon, startiso, endiso):
    evt_query = get_query(eventws, minmagnitude=minmag, minlat=minlat, maxlat=maxlat,
                          minlon=minlon, maxlon=maxlon, start=startiso,
                          end=endiso, format='text')
    raw_data = url_read(evt_query, decode='utf8',
                        on_exc=lambda exc: logger.error(msgs.format(exc, evt_query)))

    empty_result = {}  # Create once an empty result consistent with the excpetced return value
    if raw_data is None:
        return empty_result

    try:
        events_df = query2dframe(raw_data)
    except ValueError as exc:
        logger.error(msgs.format(exc, evt_query))
        return empty_result

    # events_df surely not empty
    try:
        olddf, events_df = events_df, normalize_fdsn_dframe(events_df, "event")
    except ValueError as exc:
        logger.error(msgs.format(exc, evt_query))
        return empty_result

    # events_df surely not empty
    if len(olddf) > len(events_df):
        logger.warning(msgs.query.dropped_evt(len(olddf) - len(events_df), evt_query,
                                              "malformed/invalid data, e.g.: NaN"))

    events = {}  # loop below a bit verbose, but better for debug
    for inst, _ in get_or_add_iter(session, df2dbiter(events_df, models.Event),
                                   on_add='commit'):
        if inst is not None:
            events[inst.id] = inst

    if not events:
        logger.error(msgs.db.dropped_evt(len(events_df), evt_query))
        return empty_result
    elif len(events) < len(events_df):
        logger.warning(msgs.db.dropped_evt(len(events_df) - len(events), evt_query))

    return events


def get_datacenters(session, **query_args):
    """Queries all datacenters and returns the local db model rows correctly added
    Rows already existing (comparing by datacenter station_query_url) are returned as well,
    but not added again
    :param query_args: any key value pair for the query. Note that 'service' and 'format' will
    be overiidden in the code with values of 'station' and 'format', repsecively
    """
    empty_result = {}  # Create once an empty result consistent with the excpetced return value
    query_args['service'] = 'station'
    query_args['format'] = 'post'
    query = get_query('http://geofon.gfz-potsdam.de/eidaws/routing/1/query', **query_args)
#     query = ('http://geofon.gfz-potsdam.de/eidaws/routing/1/query?service=station&'
#              'start=%s&end=%s&format=post') % (start_time.isoformat(), end_time.isoformat())
    dc_result = url_read(query, decode='utf8',
                         on_exc=lambda exc: logger.error(msgs.format(exc, query)))

    if not dc_result:
        return empty_result
    # add to db the datacenters read. Two little hacks:
    # 1) parse dc_result string and assume any new line starting with http:// is a valid station
    # query url
    # 2) When adding the datacenter, the table column dataselect_query_url (when not provided, as
    # in this case) is assumed to be the same as station_query_url by replacing "/station" with
    # "/dataselect". See https://www.fdsn.org/webservices/FDSN-WS-Specifications-1.1.pdf

    datacenters = [models.DataCenter(station_query_url=dcen) for dcen in dc_result.split("\n")
                   if dcen[:7] == "http://"]

    new = 0
    err = 0
    for dcen, isnew in get_or_add_iter(session, datacenters, [models.DataCenter.station_query_url],
                                       on_add='commit'):
        if isnew:
            new += 1
        elif dcen is None:
            err += 1

    if err > 0:
        logger.warning(msgs.db.dropped_dc(err, query))

    dcenters = session.query(models.DataCenter).all()
#     logger.debug("%d datacenters found, %d newly added, %d skipped (internal db error)\nurl: %s",
#                  len(dcenters), new, err, query)
    # do not return only new datacenters, return all of them
    return {dcen.id: dcen for dcen in dcenters}


def get_stations_df(url, raw_data, min_sample_rate):
    """FIXME: write doc! """
    if not raw_data:
        logger.warning(msgs.query.empty())  # query2dframe below handles empty data,
        return empty()
    # but we want meaningful log
    try:
        stations_df = query2dframe(raw_data)
    except ValueError as exc:
        logger.warning(msgs.format(exc, url))
        return empty()

    # stations_df surely not empty:
    try:
        olddf, stations_df = stations_df, normalize_fdsn_dframe(stations_df, "channel")
    except ValueError as exc:
        logger.warning(msgs.format(exc, url))
        return empty()

    # stations_df surely not empty:
    if len(olddf) > len(stations_df):
        logger.warning(msgs.query.dropped_sta(len(olddf)-len(stations_df), url,
                                              "malformed/invalid data, e.g.: NaN"))

    if min_sample_rate > 0:
        srate_col = models.Channel.sample_rate.key
        olddf, stations_df = stations_df, stations_df[stations_df[srate_col] >= min_sample_rate]
        reason = "sample rate < %s Hz" % str(min_sample_rate)
        if len(olddf) > len(stations_df):
            logger.warning(msgs.query.dropped_sta(len(olddf)-len(stations_df), url, reason))
        if empty(stations_df):
            return stations_df
        # http://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
        stations_df.is_copy = False

#     olddf, stations_df = stations_df, save_stations_df(session, stations_df)
#     if len(olddf) > len(stations_df):
#         logger.warning(msgs.db.dropped_sta(len(olddf) - len(stations_df), url))

    return stations_df  # might be empty


def make_ev2sta(session, events, datacenters, sradius_minmag, sradius_maxmag, sradius_minradius,
                sradius_maxradius, station_timespan, channels,
                min_sample_rate, max_thread_workers, timeout, blocksize,
                notify_progress_func=lambda *a, **v: None):
    """Returns dict {event_id: stations_df} where stations_df is an already normalized and
    harmonized dataframe with the stations saved or already present, and the JOINT fields
    of models.Station and models.Channel. The id column values refers to models.Channel id's
    though"""
    # calculate search radia:
    magnitudes = np.array([evt.magnitude for evt in events.itervalues()])
    max_radia = get_search_radius(magnitudes, sradius_minmag, sradius_maxmag,
                                  sradius_minradius, sradius_maxradius)

    urls2tuple = {}
    for dcen in datacenters.itervalues():
        for max_radius, evt in izip(max_radia, events.itervalues()):
            start = evt.time - timedelta(hours=station_timespan[0])
            end = evt.time + timedelta(hours=station_timespan[1])
            url = get_query(dcen.station_query_url,
                            latitude="%3.3f" % evt.latitude,
                            longitude="%3.3f" % evt.longitude,
                            maxradius=max_radius,
                            start=start.isoformat(), end=end.isoformat(),
                            channel=','.join(channels), format='text', level='channel')
            urls2tuple[url] = (evt, dcen)

    stats = {d.station_query_url: UrlStats() for d in datacenters.itervalues()}

    ret = {}

    def onsuccess(data, url, index):  # pylint:disable=unused-argument
        """function executed when a given url has successfully downloaded data"""
        notify_progress_func(1)
        tup = urls2tuple[url]
        evt, dcen = tup[0], tup[1]
        df = get_stations_df(url, data, min_sample_rate)
        if empty(df):
            if data:
                stats[dcen.station_query_url]['malformed'] += 1
            else:
                stats[dcen.station_query_url]['empty'] += 1
        else:
            stats[dcen.station_query_url]['OK'] += 1
            all_channels = len(df)
            df[models.Station.datacenter_id.key] = dcen.id
            df, new_sta, new_cha = save_stations_and_channels(session, df)
            if new_cha:
                stats[dcen.station_query_url]['Channels: number of new channels saved'] += new_cha
            if new_sta:
                stats[dcen.station_query_url]['Stations: number of new stations saved'] += new_sta
            if all_channels - len(df):
                stats[dcen.station_query_url][('DB Error: local database '
                                               'errors while saving data')] += 1  # FIXME
            ret[evt.id] = appenddf(ret.get(evt.id, empty()), df)

    def onerror(exc, url, index):  # pylint:disable=unused-argument
        """function executed when a given url has failed downloading data"""
        notify_progress_func(1)
        logger.warning(msgs.format(exc, url))
        dcen_station_query = urls2tuple[url][1].station_query_url
        stats[dcen_station_query][exc] += 1

    read_async(urls2tuple.keys(), onsuccess, onerror, blocksize=blocksize,
               max_workers=max_thread_workers, decode='utf8',
               timeout=timeout)

    return ret, stats


def save_stations_and_channels(session, stations_df):
    """
        stations_df is already harmonized. If saved, it is appended a column 
        `models.Channel.station_id.key` with nonNull values
    """
    new_stations = new_channels = 0
    sta_ids = []
    for sta, isnew in get_or_add_iter(session,
                                      df2dbiter(stations_df, models.Station, False, False),
                                      [models.Station.network, models.Station.station],
                                      on_add='commit'):
        if isnew:
            new_stations += 1
        sta_ids.append(None if sta is None else sta.id)

    stations_df[models.Channel.station_id.key] = sta_ids
    old_len = len(stations_df)
    stations_df.dropna(subset=[models.Channel.station_id.key], inplace=True)

    if old_len > len(stations_df):
        logger.warning(msgs.db.dropped_sta(old_len - len(stations_df), url=None,
                                           msg_or_exc=None))
    if empty(stations_df):
        return stations_df

    channels_df = stations_df  # rename just for making clear what we are handling from here on...
    cha_ids = []
    for cha, isnew in get_or_add_iter(session,
                                      df2dbiter(channels_df, models.Channel, False, False),
                                      [models.Channel.station_id, models.Channel.location,
                                       models.Channel.channel],
                                      on_add='commit'):
        if isnew:
            new_channels += 1
        cha_ids.append(None if cha is None else cha.id)

    # channels_df = channels_df.drop(models.Channel.station_id.key, axis=1)  # del station_id column
    channels_df[models.Channel.id.key] = cha_ids
    old_len = len(channels_df)
    channels_df.dropna(subset=[models.Channel.id.key], inplace=True)
    if old_len > len(channels_df):
        logger.warning(msgs.db.dropped_cha(old_len - len(channels_df), url=None,
                                           msg_or_exc=None))

    channels_df.reset_index(drop=True, inplace=True)  # to be safe
    return channels_df, new_stations, new_channels


def save_inventories(session, stations, max_thread_workers, timeout,
                     download_blocksize, notify_progress_func=lambda *a, **v: None):
    urls = {get_inventory_query(sta): sta for sta in stations}

    def onsuccess(data, url, index):
        notify_progress_func(1)
        if not data:
            logger.warning(msgs.format("Empty inventory", url))
            return
        try:
            urls[url].inventory_xml = dumps_inv(data)
            session.commit()
        except SQLAlchemyError as exc:
            session.rollback()
            logger.warning(msgs.format(exc, url))

    def onerror(err, url, index):
        notify_progress_func(1)
        logger.warning(msgs.format(err, url))


    read_async(urls, onsuccess, onerror, max_workers=max_thread_workers,
               blocksize=download_blocksize, timeout=timeout)



# def get_segments_REMOVE(session, events, datacenters, stations, wtimespan, traveltime_phases,
#                         notify_progress_func=lambda *a, **v: None):
#     segments = {dcen_id: empty() for dcen_id in datacenters}
#     skipped_already_d = {dcen_id: 0 for dcen_id in datacenters}
#     distcache_dict = {}
#     timecache_dict = {}
#     tau_p_model = TauPyModel('ak135')
#     for (evt_id, dcen_id), stations_df in stations.iteritems():
#         notify_progress_func(1)
#         if empty(stations_df):  # for safety
#             continue
#         segments_df = calculate_times(stations_df, events[evt_id], wtimespan, traveltime_phases,
#                                       tau_p_model, distcache_dict, timecache_dict)
#         segments_df[models.Segment.channel_id.key] = stations_df[models.Channel.id.key]
#         segments_df[models.Segment.event_id.key] = evt_id
# 
#         # we will purge already downloaded segments, and use the index of the purged segments
#         # to filter out stations, too. For this, we need to be sure they have the same index
#         # before these operations:
#         stations_df.reset_index(drop=True, inplace=True)
#         segments_df.reset_index(drop=True, inplace=True)
#         oldsegments_df, segments_df = segments_df, purge_already_downloaded(session, segments_df)
#         skipped_already_d[dcen_id] += (len(oldsegments_df) - len(segments_df))
#         # purge stations, too (see comment above):
#         stations_df = stations_df[stations_df.index.isin(segments_df.index.values)]
#         # set the wav query as dataframe index:
#         segments_df = set_wav_queries(datacenters[dcen_id], stations_df, segments_df,
#                                       _SEGMENTS_DATAURL_COLNAME)
#         segments[dcen_id] = appenddf(segments[dcen_id], segments_df)
# 
#     return segments, skipped_already_d


def make_dc2seg(session, events, datacenters, evt2stations, wtimespan, traveltime_phases,
                notify_progress_func=lambda *a, **v: None):
    segments = {dc_id: empty() for dc_id in datacenters}
    skipped_already_d = {dc_id: 0 for dc_id in datacenters}
    distcache_dict = {}
    timecache_dict = {}
    # tau_p_model = TauPyModel('ak135')

    ev2segs = {}
    # We can use a with statement to ensure threads are cleaned up promptly
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        # Start the load operations and mark each future with its URL
        future_to_segments = {executor.submit(calculate_times, stations_df, events[evt_id],
                                              wtimespan, traveltime_phases,
                                              TauPyModel('ak135'), distcache_dict,
                                              timecache_dict): evt_id
                              for evt_id, stations_df in evt2stations.iteritems()}
        for future in concurrent.futures.as_completed(future_to_segments):
            notify_progress_func(1)
            ev_id = future_to_segments[future]
            try:
                segments_df = future.result()
                ev2segs[ev_id] = segments_df
            except Exception as exc:
                ev2segs[ev_id] = empty()
                logger.warning(msgs.calc.dropped_sta(len(evt2stations[ev_id]),
                                                     "calculating arrival time", exc))


    # create a columns which we know not being in the segments model
    cnames = colnames(models.Segment)
    _SEGMENTS_DATAURL_COLNAME = '__wquery__'
    while _SEGMENTS_DATAURL_COLNAME in cnames:
        _SEGMENTS_DATAURL_COLNAME += "_"
    for evt_id, segments_df in ev2segs.iteritems():
        if empty(segments_df):  # for safety
            continue
        stations_df = evt2stations[evt_id]
#         segments_df = calculate_times(stations_df, events[evt_id], wtimespan, traveltime_phases,
#                                       tau_p_model, distcache_dict, timecache_dict)
        where_valid = ~pd.isnull(segments_df[models.Segment.arrival_time.key])
        stations_df, segments_df = stations_df[where_valid], segments_df[where_valid]
        segments_df[models.Segment.channel_id.key] = stations_df[models.Channel.id.key]
        segments_df[models.Segment.event_id.key] = evt_id
        segments_df[models.Segment.datacenter_id.key] = stations_df[models.Station.datacenter_id.key]

        # build queries, None's for already downloaded:
        wqueries = []
        for sta, seg in izip(stations_df.itertuples(), segments_df.itertuples()):
            cha_id = getattr(seg, models.Segment.channel_id.key)
            start_time = getattr(seg, models.Segment.start_time.key)
            end_time = getattr(seg, models.Segment.end_time.key)
            if session.query(models.Segment).\
                filter((models.Segment.channel_id == cha_id) &
                       (models.Segment.start_time == start_time) &
                       (models.Segment.end_time == end_time)).\
                    first():
                wqueries.append(None)
            else:
                dc_id = getattr(seg, models.Segment.datacenter_id.key)
                wqueries.append(get_query(datacenters[dc_id].dataselect_query_url,
                                          network=getattr(sta, models.Station.network.key),
                                          station=getattr(sta, models.Station.station.key),
                                          location=getattr(sta, models.Channel.location.key),
                                          channel=getattr(sta, models.Channel.channel.key),
                                          start=start_time.isoformat(),
                                          end=end_time.isoformat()))

        segments_df[_SEGMENTS_DATAURL_COLNAME] = wqueries
        dc_ids = pd.unique(segments_df[models.Segment.datacenter_id.key])
        for dc_id in dc_ids:
            dc_segments_df = segments_df[segments_df[models.Segment.datacenter_id.key] == dc_id]
            if empty(dc_segments_df):
                continue
            old_dc_segments_df, dc_segments_df = dc_segments_df, \
                dc_segments_df.dropna(subset=[_SEGMENTS_DATAURL_COLNAME], axis=0)
            segments[dc_id] = appenddf(segments[dc_id], dc_segments_df)
            skip_downld = len(old_dc_segments_df) - len(dc_segments_df)
            if skip_downld:
                skipped_already_d[dc_id] += skip_downld

    # sets as indices the query urls, drop the relative column
    for dc_id in segments:
        dataframe = segments[dc_id]
        if not empty(dataframe):
            # it should never happen that we have duplicates. However, in case,
            # the segments downloaded will not be correctly placed at the right index.
            # Thus,in case, remove all duplicates
            lendf = len(dataframe)
            dataframe.drop_duplicates(subset=[_SEGMENTS_DATAURL_COLNAME], keep=False, inplace=True)
            if lendf != len(dataframe):
                logger.warning(msgs.format(("datacenter (id=%d): Can not handle "
                                            "%d duplicates in segments urls: discarded")),
                               dc_id, lendf-len(dataframe))
            dataframe.set_index(_SEGMENTS_DATAURL_COLNAME, inplace=True)  # drops the column
            # In order to remove the index name which is now _SEGMENTS_DATAURL_COLNAME:
            dataframe.index.name = None

    return segments, skipped_already_d


def calculate_times(stations_df, evt, timespan, traveltime_phases, tau_p_model='ak135',
                    distcache_dict=None, timecache_dict=None):
    event_distances_degrees = []
    arrival_times = []
    latstr, lonstr = models.Station.latitude.key, models.Station.longitude.key
    # cache already calculated results:
    cache = {}
    for sta in stations_df.itertuples():
        sta_id = getattr(sta, models.Channel.station_id.key)
        if sta_id in cache:
            degrees, arr_time = cache[sta_id]
        else:
            stalat, stalon = getattr(sta, latstr), getattr(sta, lonstr)
            degrees = locations2degrees(evt.latitude, evt.longitude, stalat, stalon)
            try:
                arr_time = get_arrival_time(degrees, evt.depth_km, evt.time, traveltime_phases,
                                            tau_p_model)
            except (TauModelError, ValueError, SlownessModelError) as exc:
                logger.warning(msgs.calc.dropped_sta(sta_id, "arrival time calculation", exc))
                arr_time = None
            cache[sta_id] = (degrees, arr_time)
        event_distances_degrees.append(degrees)
        arrival_times.append(arr_time)

    ret = pd.DataFrame({models.Segment.event_distance_deg.key: event_distances_degrees,
                        models.Segment.arrival_time.key: arrival_times})
    td0, td1 = timedelta(minutes=timespan[0]), timedelta(minutes=timespan[1])
    # this works also if arrival time is None (sets NaT):
    ret[models.Segment.start_time.key] = ret[models.Segment.arrival_time.key] - td0
    ret[models.Segment.end_time.key] = ret[models.Segment.arrival_time.key] + td1

    return ret



def download_segments(session, segments_df, run_id, max_error_count, max_thread_workers,
                      timeout, download_blocksize,
                      notify_progress_func=lambda *a, **v: None):

    stats = UrlStats()
    if empty(segments_df):
        return stats
    errors = [0]  # http://stackoverflow.com/questions/2609518/python-nested-function-scopes
    segments_df[models.Segment.data.key] = None
    # set_index as urls (this is much faster when locating a dframe row compared to
    # df[df[df_urls_colname] == some_url]):
    # segments_df.set_index(_SEGMENTS_DATAURL_COLNAME, inplace=True)  # FIXME: check drop!
    urls = segments_df.index.values

    def onsuccess(data, url, index):  # pylint:disable=unused-argument
        """function executed when a given url has succesfully downloaded `data`"""
        notify_progress_func(1)
        segments_df.loc[url, models.Segment.data.key] = data  # avoid pandas SettingWithCopyWarning

    def onerror(exc, url, index):  # pylint:disable=unused-argument
        """function executed when a given url has failed"""
        notify_progress_func(1)
        logger.warning(msgs.query.dropped_seg(1, url, exc))
        stats[exc] += 1
        errors[0] += 1
        if max_error_count > 0 and errors[0] >= max_error_count:
            return False  # stop next downloads (well, skip them, it will be faster anyways)

    # now download Data:
    read_async(urls, onsuccess, onerror, max_workers=max_thread_workers, timeout=timeout,
               blocksize=download_blocksize)

    old_df, segments_df = segments_df, segments_df.dropna(subset=[models.Segment.data.key])
    null_data_count = len(old_df) - len(segments_df)

    # get empty data, then remove it:
    segments_df[models.Segment.data.key].replace(b'', np.nan, inplace=True)
    old_df, segments_df = segments_df, segments_df.dropna(subset=[models.Segment.data.key])
    stats['Empty'] = len(old_df) - len(segments_df)

    if errors[0] >= max_error_count:
        discarded_after_max_error_count = null_data_count - errors[0]
        key_skipped = ("Discarded: Remaining queries from the given domain ignored "
                       "after %d previous errors") % max_error_count
        stats[key_skipped] = discarded_after_max_error_count
        notify_progress_func(discarded_after_max_error_count)

    # segments = []
    if not empty(segments_df):
        # http://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
        segments_df.is_copy = False
        segments_df[models.Segment.run_id.key] = run_id
        for model_instance in df2dbiter(segments_df, models.Segment, False, False):
            session.add(model_instance)
            if commit(session):
                # segments.append(model_instance)
                stats['Saved'] += 1
            else:
                stats['DB Error: Local database error while saving data'] += 1

        # reset_index as integer. This might not be the old index if the old one was not a
        # RangeIndex (0,1,2 etcetera). But it shouldn't be an issue
        # Note: 'drop=False' to restore 'df_urls_colname' column:
        segments_df.reset_index(drop=False, inplace=True)

    return stats


def main(session, run_id, eventws, minmag, minlat, maxlat, minlon, maxlon, start, end, stimespan,
         sradius_minmag, sradius_maxmag, sradius_minradius, sradius_maxradius,
         channels, min_sample_rate, download_s_inventory, traveltime_phases,
         wtimespan, advanced_settings, isterminal=False):
    """
        Downloads waveforms related to events to a specific path
        :param eventws: Event WS to use in queries. E.g. 'http://seismicportal.eu/fdsnws/event/1/'
        :type eventws: string
        :param minmaa: Minimum magnitude. E.g. 3.0
        :type minmaa: float
        :param minlat: Minimum latitude. E.g. 30.0
        :type minlat: float
        :param maxlat: Maximum latitude E.g. 80.0
        :type maxlon: float
        :param minlon: Minimum longitude E.g. -10.0
        :type minlon: float
        :param maxlon: Maximum longitude E.g. 60.0
        :type maxlon: float
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
    if advanced_settings['download_blocksize'] == 0:
        advanced_settings['download_blocksize'] = -1

    progressbar = get_progressbar(isterminal)

    stepiter = imap(lambda i, m: "%d of %d" % (i+1, m), xrange(5 if download_s_inventory else 4),
                    repeat(5 if download_s_inventory else 4))

    # write the class labels:
    for _, _ in get_or_add_iter(session, df2dbiter(class_labels_df,
                                                   models.Class,
                                                   harmonize_cols_first=True,
                                                   harmonize_rows_first=True), on_add='commit'):
        pass

    startiso = start.isoformat()
    endiso = end.isoformat()

    logger.info("")
    logger.info("STEP %s: Querying events and datacenters", next(stepiter))
    # Get events, store them in the db, returns the event instances (db rows) correctly added:

    events = get_events(session, eventws, minmag, minlat, maxlat, minlon, maxlon, startiso, endiso)
    if not events:
        return 1

    # Get datacenters, store them in the db, returns the dc instances (db rows) correctly added:
    datacenters = get_datacenters(session, start=startiso, end=endiso)
    if not datacenters:
        return 1

    logger.info("")
    logger.info(("STEP %s: Querying stations (level=channel, datacenter(s): %d) "
                 "nearby %d event(s) found"), next(stepiter), len(datacenters), len(events))

    with progressbar(length=len(events)*len(datacenters)) as bar:
        evtid2stations, s_stats = make_ev2sta(session, events, datacenters,
                                              sradius_minmag, sradius_maxmag,
                                              sradius_minradius, sradius_maxradius,
                                              stimespan, channels, min_sample_rate,
                                              advanced_settings['max_thread_workers'],
                                              advanced_settings['s_timeout'],
                                              advanced_settings['download_blocksize'],
                                              bar.update)

    if download_s_inventory:
        stations = session.query(models.Station).filter(models.Station.inventory_xml == None).all()
        logger.info("")
        logger.info(("STEP %s: Downloading %d stations inventories"), next(stepiter), len(stations))
        with progressbar(length=len(stations)) as bar:
            save_inventories(session, stations,
                             advanced_settings['max_thread_workers'],
                             advanced_settings['i_timeout'],
                             advanced_settings['download_blocksize'], bar)

    logger.info("")
    logger.info(("STEP %s: Preparing segments download: calculating P-arrival times "
                 "and time ranges"), next(stepiter))

    with progressbar(length=len(evtid2stations)) as bar:
        dcid2seg, skipped_already_d = make_dc2seg(session, events, datacenters, evtid2stations,
                                                  wtimespan, traveltime_phases, bar.update)

    segments_count = sum([len(seg_df) for seg_df in dcid2seg.itervalues()])
    logger.info("")
    logger.info("STEP %s: Querying Datacenter WS for %d segments", next(stepiter), segments_count)

    d_stats = {}
    with progressbar(length=segments_count) as bar:
        for dcen_id, segments_df in dcid2seg.iteritems():
            stats = download_segments(session, segments_df, run_id,
                                      advanced_settings['w_maxerr_per_dc'],
                                      advanced_settings['max_thread_workers'],
                                      advanced_settings['w_timeout'],
                                      advanced_settings['download_blocksize'],
                                      bar.update)
            stats['Discarded: Already saved'] = skipped_already_d[dcen_id]
            # Note above: provide ":" to auto split column if too long
            d_stats[datacenters[dcen_id].dataselect_query_url] = stats

    logger.info("")

    # define functions to represent stats:
    def rfunc(row):
        """function for modifying each row display"""
        return urlparse(row).netloc

    def cfunc(col):
        """function for modifying each row display"""
        return col if col.find(":") < 0 else col[:col.find(":")]

    logger.info("Summary Station WS query info:")
    logger.info(stats2str(s_stats, fillna=0, transpose=True, lambdarow=rfunc, lambdacol=cfunc,
                          sort='col'))
    logger.info("")
    logger.info("Summary Datacenter WS info :")
    logger.info(stats2str(d_stats, fillna=0, transpose=True, lambdarow=rfunc, lambdacol=cfunc,
                          sort='col'))
    logger.info("")

    return 0
