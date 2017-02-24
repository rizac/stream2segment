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
from collections import defaultdict
from datetime import timedelta
from urlparse import urlparse
from itertools import izip, imap, repeat
import concurrent.futures
import numpy as np
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError
from obspy.taup.tau import TauPyModel
from obspy.geodetics.base import locations2degrees
from obspy.taup.helper_classes import TauModelError, SlownessModelError
from stream2segment.utils import msgs, get_progressbar
from stream2segment.utils.url import urlread, read_async, URLException
from stream2segment.io.db.models import Class, Event, DataCenter, Segment, Channel, Station
from stream2segment.io.db.pd_sql_utils import df2dbiter, get_or_add_iter, commit, colnames,\
    get_or_add, withdata
from stream2segment.download.utils import empty, get_query, query2dframe, normalize_fdsn_dframe,\
    get_search_radius, get_arrival_time, UrlStats, stats2str, get_inventory_query
from stream2segment.io.utils import dumps_inv


logger = logging.getLogger(__name__)


def get_events(session, eventws, **args):
    evt_query = get_query(eventws, format='text', **args)
    try:
        raw_data = urlread(evt_query, decode='utf8')
    except URLException as urlexc:
        logger.error(msgs.format(urlexc.exc, evt_query))
        raw_data = None

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
    for inst, _ in get_or_add_iter(session, df2dbiter(events_df, Event),
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
    try:
        dc_result = urlread(query, decode='utf8')
    except URLException as urlexc:
        logger.error(msgs.format(urlexc.exc, query))
        dc_result = None

    if not dc_result:
        return empty_result
    # add to db the datacenters read. Two little hacks:
    # 1) parse dc_result string and assume any new line starting with http:// is a valid station
    # query url
    # 2) When adding the datacenter, the table column dataselect_query_url (when not provided, as
    # in this case) is assumed to be the same as station_query_url by replacing "/station" with
    # "/dataselect". See https://www.fdsn.org/webservices/FDSN-WS-Specifications-1.1.pdf

    datacenters = [DataCenter(station_query_url=dcen) for dcen in dc_result.split("\n")
                   if dcen[:7] == "http://"]

    new = 0
    err = 0
    for dcen, isnew in get_or_add_iter(session, datacenters, [DataCenter.station_query_url],
                                       on_add='commit'):
        if isnew:
            new += 1
        elif dcen is None:
            err += 1

    if err > 0:
        logger.warning(msgs.db.dropped_dc(err, query))

    dcenters = session.query(DataCenter).all()
    # do not return only new datacenters, return all of them
    return {dcen.id: dcen for dcen in dcenters}


def make_ev2sta(session, events, datacenters, sradius_minmag, sradius_maxmag, sradius_minradius,
                sradius_maxradius, station_timespan, channels,
                min_sample_rate, max_thread_workers, timeout, blocksize,
                notify_progress_func=lambda *a, **v: None):
    """Returns dict {event_id: stations_df} where stations_df is an already normalized and
    harmonized dataframe with the stations saved or already present, and the JOINT fields
    of Station and Channel. The id column values refers to Channel id's
    though"""
    stats = {d.station_query_url: UrlStats() for d in datacenters.itervalues()}

    ret = defaultdict(list)

    def ondone(obj, result, exc, cancelled):  # pylint:disable=unused-argument
        """function executed when a given url has successfully downloaded data"""
        notify_progress_func(1)
        evt, dcen, url = obj[0], obj[1], obj[2]
        if cancelled:  # should never happen, however...
            msg = "Download cancelled"
            logger.warning(msgs.format(msg, url))
            stats[dcen.station_query_url][msg] += 1
        elif exc:
            logger.warning(msgs.format(exc, url))
            stats[dcen.station_query_url][exc] += 1
        else:
            df = get_stations_df(url, result, min_sample_rate)
            if empty(df):
                if result:
                    stats[dcen.station_query_url]['Malformed'] += 1
                else:
                    stats[dcen.station_query_url]['Empty'] += 1
            else:
                stats[dcen.station_query_url]['OK'] += 1
                all_channels = len(df)
                df[Station.datacenter_id.key] = dcen.id
                df, new_sta, new_cha = save_stations_and_channels(session, df)
                if new_cha:
                    stats[dcen.station_query_url]['Channels: new channels saved'] += new_cha
                if new_sta:
                    stats[dcen.station_query_url]['Stations: new stations saved'] += new_sta
                if all_channels - len(df):
                    stats[dcen.station_query_url][('DB Error: local database '
                                                   'errors while saving data')] += 1  # FIXME
                ret[evt.id].append(df)

    iterable = evdcurl_iter(events, datacenters, sradius_minmag, sradius_maxmag, sradius_minradius,
                            sradius_maxradius, station_timespan, channels)

    read_async(iterable, ondone, urlkey=lambda obj: obj[-1], blocksize=blocksize,
               max_workers=max_thread_workers, decode='utf8', timeout=timeout)

    return {eid: pd.concat(ret[eid], axis=0, ignore_index=True, copy=False) for eid in ret}, stats


def evdcurl_iter(events, datacenters, sradius_minmag, sradius_maxmag, sradius_minradius,
                 sradius_maxradius, station_timespan, channels):
    """returns an iterable of tuple (event, datacenter, station_query_url) where the last element
    is build with sradius arguments and station_timespan and channels"""
    # calculate search radia:
    magnitudes = np.array([evt.magnitude for evt in events.itervalues()])
    max_radia = get_search_radius(magnitudes, sradius_minmag, sradius_maxmag,
                                  sradius_minradius, sradius_maxradius)
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
            yield (evt, dcen, url)


def get_stations_df(url, raw_data, min_sample_rate):
    """FIXME: write doc! """
    if not raw_data:
        logger.warning(msgs.query.empty(url))  # query2dframe below handles empty data,
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
        srate_col = Channel.sample_rate.key
        olddf, stations_df = stations_df, stations_df[stations_df[srate_col] >= min_sample_rate]
        reason = "sample rate < %s Hz" % str(min_sample_rate)
        if len(olddf) > len(stations_df):
            logger.warning(msgs.query.dropped_sta(len(olddf)-len(stations_df), url, reason))
        if empty(stations_df):
            return stations_df
        # http://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
        stations_df.is_copy = False

    return stations_df  # might be empty


def save_stations_and_channels(session, stations_df):
    """
        stations_df is already harmonized. If saved, it is appended a column
        `Channel.station_id.key` with nonNull values
    """
    new_stations = new_channels = 0
    sta_ids = []
    for sta, isnew in get_or_add_iter(session,
                                      df2dbiter(stations_df, Station, False, False),
                                      [Station.network, Station.station],
                                      on_add='commit'):
        if isnew:
            new_stations += 1
        sta_ids.append(None if sta is None else sta.id)

    stations_df[Channel.station_id.key] = sta_ids
    old_len = len(stations_df)
    stations_df.dropna(subset=[Channel.station_id.key], inplace=True)

    if old_len > len(stations_df):
        logger.warning(msgs.db.dropped_sta(old_len - len(stations_df), url=None,
                                           msg_or_exc=None))
    if empty(stations_df):
        return stations_df

    channels_df = stations_df  # rename just for making clear what we are handling from here on...
    cha_ids = []
    for cha, isnew in get_or_add_iter(session,
                                      df2dbiter(channels_df, Channel, False, False),
                                      [Channel.station_id, Channel.location,
                                       Channel.channel],
                                      on_add='commit'):
        if isnew:
            new_channels += 1
        cha_ids.append(None if cha is None else cha.id)

    channels_df[Channel.id.key] = cha_ids
    old_len = len(channels_df)
    channels_df.dropna(subset=[Channel.id.key], inplace=True)
    if old_len > len(channels_df):
        logger.warning(msgs.db.dropped_cha(old_len - len(channels_df), url=None,
                                           msg_or_exc=None))

    channels_df.reset_index(drop=True, inplace=True)  # to be safe
    return channels_df, new_stations, new_channels


def save_inventories(session, stations, max_thread_workers, timeout,
                     download_blocksize, notify_progress_func=lambda *a, **v: None):
    def ondone(obj, result, exc, cancelled):
        notify_progress_func(1)
        sta, url = obj
        if exc:
            logger.warning(msgs.format(exc, url))
        else:
            if not result:
                logger.warning(msgs.query.empty(url))
                return
            try:
                sta.inventory_xml = dumps_inv(result)
                session.commit()
            except SQLAlchemyError as sqlexc:
                session.rollback()
                logger.warning(msgs.db.dropped_inv(sta.id, url, sqlexc))

    iterable = izip(stations, (get_inventory_query(sta) for sta in stations))
    read_async(iterable, ondone, urlkey=lambda obj: obj[1],
               max_workers=max_thread_workers,
               blocksize=download_blocksize, timeout=timeout)


def get_segments_df(session, events, datacenters, evt2stations, wtimespan, traveltime_phases,
                    taup_model, retry_empty_segments,
                    notify_progress_func=lambda *a, **v: None):
    # create a columns which we know not being in the segments model
    cnames = colnames(Segment)
    _SEGMENTS_DATAURL_COLNAME = '__wquery__'
    while _SEGMENTS_DATAURL_COLNAME in cnames:
        _SEGMENTS_DATAURL_COLNAME += "_"

    stationslist = []
    segmentslist = []
    # We can use a with statement to ensure threads are cleaned up promptly
    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        # Start the load operations and mark each future with its URL
        future_to_segments = {executor.submit(calculate_times, stations_df, events[evt_id],
                                              traveltime_phases,
                                              taup_model): evt_id
                              for evt_id, stations_df in evt2stations.iteritems()}
        for future in concurrent.futures.as_completed(future_to_segments):
            notify_progress_func(1)
            try:
                evt_id = future_to_segments[future]
                segments_df = future.result()
                if empty(segments_df):  # for safety
                    continue
                stations_df = evt2stations[evt_id]
                stationslist.append(stations_df)
                segments_df[Segment.event_id.key] = evt_id
                segmentslist.append(segments_df)
            except Exception as exc:
                logger.warning(msgs.calc.dropped_sta(len(evt2stations[evt_id]),
                                                     "calculating arrival time", exc))

    if not stationslist or not segmentslist:
        return empty(), {dc_id: 0 for dc_id in datacenters}

    # concat is faster than DataFrame append
    stations_df = pd.concat(stationslist, axis=0, ignore_index=True, copy=False)
    segments_df = pd.concat(segmentslist, axis=0, ignore_index=True, copy=False)
    # remove invalid arrival times:
    where_valid = ~pd.isnull(segments_df[Segment.arrival_time.key])
    # set segments attributes from relative stations
    stations_df, segments_df = stations_df[where_valid], segments_df[where_valid]
    segments_df[Segment.channel_id.key] = stations_df[Channel.id.key]
    segments_df[Segment.datacenter_id.key] = stations_df[Station.datacenter_id.key]
    # set start time and end time. Round to seconds cause arrival time has rounding errors
    # NOTE THAT - although we don't have NaT's now - THERE IS A BUG in pandas (0.18.1) with NATs
    # when using e.g. .dt.round('5min') instead of .dt.round('s')
    td0, td1 = timedelta(minutes=wtimespan[0]), timedelta(minutes=wtimespan[1])
    segments_df[Segment.start_time.key] = \
        (segments_df[Segment.arrival_time.key] - td0).dt.round('s')
    segments_df[Segment.end_time.key] = (segments_df[Segment.arrival_time.key] + td1).dt.round('s')
    # build queries, None's for "do not download'em":
    wqueries, ids, skipped_already_d = prepare_for_wdownload(session, datacenters, segments_df,
                                                             stations_df, retry_empty_segments)
    segments_df[_SEGMENTS_DATAURL_COLNAME] = wqueries
    segments_df[Segment.id.key] = ids
    # remove already downloaded:
    segments_df.dropna(subset=[_SEGMENTS_DATAURL_COLNAME], axis=0, inplace=True)

    # remove duplicated queries: it should never happen. However, in case,
    # the segments downloaded will not be correctly placed at the right index.
    # Thus,in case, remove all duplicates
    lendf = len(segments_df)
    segments_df.drop_duplicates(subset=[_SEGMENTS_DATAURL_COLNAME], keep=False, inplace=True)
    if lendf != len(segments_df):
        logger.warning(msgs.format(("Cannot handle %d duplicates in segments urls: discarded")),
                       lendf-len(segments_df))
    # set index as the queries to download. Search by index is faster. Note that the column
    # _SEGMENTS_DATAURL_COLNAME is removed by default in set_index, but the index name keeps
    # the column name. That's harmless, but we remove it in `segments_df.index.name = None` below
    segments_df.set_index(_SEGMENTS_DATAURL_COLNAME, inplace=True)
    segments_df.index.name = None
    segments_df.is_copy = False
    return segments_df, skipped_already_d


def calculate_times(stations_df, evt, traveltime_phases, taup_model='ak135'):
    event_distances_degrees = []
    arrival_times = []
    taupmodel_obj = TauPyModel(taup_model)  # create the taupmodel once
    cache = {}  # cache already calculated results:
    # iteration over dframe columns is faster than DataFrame.itertuples
    # and is more readable as we only need a bunch of columns.
    # Note: we zip using dataframe[columname] iterables. Using
    # dataframe[columname].values (underlying pandas numpy array) is even faster,
    # BUT IT DOES NOT RETURN pd.TimeStamp objects for date-time-like columns but np.datetim64
    # instead. As the former subclasses python datetime (so it's sqlalchemy compatible) and the
    # latter does not, we go for the latter ONLY BECAUSE WE DO NOT HAVE DATETIME LIKE OBJECTS:
    for sta_id, stalat, stalon in izip(stations_df[Channel.station_id.key].values,
                                       stations_df[Station.latitude.key].values,
                                       stations_df[Station.longitude.key].values):
        if sta_id in cache:
            degrees, arr_time = cache[sta_id]
        else:
            # stalat, stalon = getattr(sta, latstr), getattr(sta, lonstr)
            degrees = locations2degrees(evt.latitude, evt.longitude, stalat, stalon)
            try:
                arr_time = get_arrival_time(degrees, evt.depth_km, evt.time, traveltime_phases,
                                            taupmodel_obj)
            except (TauModelError, ValueError, SlownessModelError) as exc:
                logger.warning(msgs.calc.dropped_sta(sta_id, "arrival time calculation", exc))
                arr_time = None
            cache[sta_id] = (degrees, arr_time)
        event_distances_degrees.append(degrees)
        arrival_times.append(arr_time)

    ret = pd.DataFrame({Segment.event_distance_deg.key: event_distances_degrees,
                        Segment.arrival_time.key: arrival_times})
    return ret


def prepare_for_wdownload(session, datacenters, segments_df, stations_df, retry_empty_segments):
    """returns the wave queries and sets them to null if segments are already downloaded.
    Returns also the ids of each segment which indicates if the segment already exists"""
    # build queries, None's for already downloaded:
    wqueries = []
    ids = []
    alreadydownloadeddict = defaultdict(int)
    withdata_flt = None if not retry_empty_segments else withdata(Segment.data)
    # iteration over dframe columns is faster than
    # DataFrame.itertuples (as it does not have to convert each row to tuple)
    # and is more readable. Note: we zip using dataframe[columname] iterables. Using
    # dataframe[columname].values (underlying numpy array) is even faster, BUT IT DOES NOT RETURN
    # pd.TimeStamp objects for date-time-like columns (returns np.datetim64 instead).
    # As the former subclasses python datetime (so it's sqlalchemy compatible) and the second not,
    # we do not go for this solution, nor
    # for calling pd.TimeStamp(numpy_datetime) inside the loop (which is less readable):
    for cha_id, start_time, end_time, dc_id, net, sta, loc, cha \
        in izip(segments_df[Segment.channel_id.key],
                segments_df[Segment.start_time.key],
                segments_df[Segment.end_time.key],
                segments_df[Segment.datacenter_id.key],
                stations_df[Station.network.key],
                stations_df[Station.station.key],
                stations_df[Channel.location.key],
                stations_df[Channel.channel.key]
                ):
        flt = (Segment.channel_id == cha_id) & (Segment.start_time == start_time) & \
            (Segment.end_time == end_time)
        existing_seg_id = session.query(Segment.id).filter(flt).first()
        already_downloaded = True if existing_seg_id else False
        # already_downloaded is set to False if retry_empty_segments is True and the segment.data
        # is empty:
        if existing_seg_id and retry_empty_segments:
            already_downloaded = True if session.query(Segment.id).\
                filter(flt & withdata_flt).first() else False

        ids.append(existing_seg_id[0] if existing_seg_id else None)
        if already_downloaded:
            wqueries.append(None)
            alreadydownloadeddict[dc_id] += 1
            continue
        wqueries.append(get_query(datacenters[dc_id].dataselect_query_url,
                                  network=net,
                                  station=sta,
                                  location=loc,
                                  channel=cha,
                                  start=start_time.isoformat(),
                                  end=end_time.isoformat()))
    return wqueries, ids, alreadydownloadeddict


def download_segments(session, segments_df, run_id, max_error_count, max_thread_workers,
                      timeout, download_blocksize, sync_session_on_update='evaluate',
                      notify_progress_func=lambda *a, **v: None):

    stats = defaultdict(lambda: UrlStats())
    if empty(segments_df):
        return stats
    errors = defaultdict(int)
    segments_df[Segment.data.key] = None

    def ondone(obj, result, exc, cancelled):
        """function executed when a given url has succesfully downloaded `data`"""
        notify_progress_func(1)
        url, dcen_id = obj[0], obj[1]
        if cancelled:
            key_skipped = ("Discarded: Cancelled remaining downloads from the given data-center "
                           "after %d previous errors") % max_error_count
            stats[dcen_id][key_skipped] += 1
        elif exc:
            logger.warning(msgs.query.dropped_seg(1, url, exc))
            stats[dcen_id][exc] += 1
            errors[dcen_id] += 1
            if max_error_count > 0 and errors[dcen_id] == max_error_count:
                # skip remaining datacenters
                return lambda obj: obj[1] == dcen_id
        else:
            # avoid pandas SettingWithCopyWarning:
            segments_df.loc[url, Segment.data.key] = result  # might be empty: b''

    # now download Data:
    # we use the index as urls cause it's much faster when locating a dframe row compared to
    # df[df[df_urls_colname] == some_url]). We zip it with the datacenters for faster search
    # REMEMBER: iterating over series values is FASTER BUT USES underlying numpy types, and for
    # datetime's is a problem cause pandas sublasses datetime, numpy not
    read_async(izip(segments_df.index.values,
                    segments_df[Segment.datacenter_id.key].values),
               urlkey=lambda obj: obj[0],
               ondone=ondone, max_workers=max_thread_workers,
               timeout=timeout, blocksize=download_blocksize)

    # messages:
    RETRY_NODATA_MSG = "Discarded: retry failed (zero bytes received or query error)"
    RETRY_WITHDATA_MSG = "Saved: retry successful (waveform data received)"
    NEW_NODATA_MSG = "Saved, no waveform data: zero bytes received or query error"
    NEW_WITHDATA_MSG = "Saved, with waveform data"
    DB_FAILED_MSG = "Discarded, error while saving data: local db error"

    if not empty(segments_df):
        # http://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
        segments_df.is_copy = False  # FIXME: still need it?
        segments_df[Segment.run_id.key] = run_id
        for model_instance in df2dbiter(segments_df, Segment, False, False):
            already_downloaded = model_instance.id is not None
            # if we have tried an already saved instance, and no data is found
            # (empty or error), skip it:
            if already_downloaded and not model_instance.data:
                stats[model_instance.datacenter_id][(RETRY_NODATA_MSG)] += 1
                continue
            elif already_downloaded:
                # already downloaded, but this time data was found
                # (if we attempted an already downloaded, it means segment.data was empty or None):
                # note that we do not update run_id
                stats[model_instance.datacenter_id][RETRY_WITHDATA_MSG] += 1
                session.query(Segment).filter(Segment.id == model_instance.id).\
                    update({Segment.data.key: model_instance.data},
                           synchronize_session=sync_session_on_update)
            else:
                session.add(model_instance)
            if commit(session):
                msg = NEW_WITHDATA_MSG if model_instance.data else NEW_NODATA_MSG
                stats[model_instance.datacenter_id][msg] += 1
            else:
                # we rolled back
                stats[model_instance.datacenter_id][DB_FAILED_MSG] += 1

        # reset_index as integer. This might not be the old index if the old one was not a
        # RangeIndex (0,1,2 etcetera). But it shouldn't be an issue
        # Note: 'drop=False' to restore 'df_urls_colname' column:
        # segments_df.reset_index(drop=False, inplace=True)

    return stats


def add_classes(session, class_labels):
    if class_labels:
        get_or_add(session, (Class(label=lab, description=desc) for lab, desc
                             in class_labels.iteritems()), Class.label, on_add='commit')


def main(session, run_id, start, end, eventws, eventws_query_args, stimespan,
         sradius_minmag, sradius_maxmag, sradius_minradius, sradius_maxradius,
         channels, min_sample_rate, download_s_inventory, traveltime_phases,
         wtimespan,
         retry_empty_segments,
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

    stepiter = imap(lambda i, m: "%d of %d" % (i+1, m), xrange(5 if download_s_inventory else 4),
                    repeat(5 if download_s_inventory else 4))

    # write the class labels:
    add_classes(session, class_labels)

    startiso = start.isoformat()
    endiso = end.isoformat()

    logger.info("")
    logger.info("STEP %s: Querying events and datacenters", next(stepiter))
    # Get events, store them in the db, returns the event instances (db rows) correctly added:

    events = get_events(session, eventws, start=startiso, end=endiso, **eventws_query_args)
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
        stations = session.query(Station).filter(~withdata(Station.inventory_xml)).all()
        logger.info("")
        logger.info(("STEP %s: Downloading %d stations inventories"), next(stepiter), len(stations))
        with progressbar(length=len(stations)) as bar:
            save_inventories(session, stations,
                             advanced_settings['max_thread_workers'],
                             advanced_settings['i_timeout'],
                             advanced_settings['download_blocksize'], bar.update)

    logger.info("")
    logger.info(("STEP %s: Preparing segments download: calculating P-arrival times "
                 "and time ranges"), next(stepiter))

    with progressbar(length=len(evtid2stations)) as bar:
        seg_df, skipped_already_d = get_segments_df(session, events, datacenters, evtid2stations,
                                                    wtimespan, traveltime_phases, 'ak135',
                                                    retry_empty_segments,
                                                    bar.update)

    segments_count = len(seg_df)
    logger.info("")
    logger.info("STEP %s: Querying Datacenter WS for %d segments", next(stepiter), segments_count)

    with progressbar(length=segments_count) as bar:
        stats = download_segments(session, seg_df, run_id,
                                  advanced_settings['w_maxerr_per_dc'],
                                  advanced_settings['max_thread_workers'],
                                  advanced_settings['w_timeout'],
                                  advanced_settings['download_blocksize'],
                                  False,  # do not sync on update
                                  bar.update)
        for dcen_id in skipped_already_d:
            stats[dcen_id]['Discarded: Already saved'] = skipped_already_d[dcen_id]
        # Note above: provide ":" to auto split column if too long
        # now converts keys to datacenters urls instead than datacenters ids:
        d_stats = {datacenters[dcen_id].dataselect_query_url: val
                   for dcen_id, val in stats.iteritems()}

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
