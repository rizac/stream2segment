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
from stream2segment.async import url_read
from stream2segment.classification import class_labels_df
from stream2segment.s2sio.db import models
from stream2segment.s2sio.db.pd_sql_utils import df2dbiter, get_or_add_iter, commit
from stream2segment.async import read_async
from stream2segment.utils import dc_stats_str
from stream2segment.download.utils import empty, get_query, query2dframe, normalize_fdsn_dframe,\
    get_search_radius, save_stations_df, purge_already_downloaded,\
    set_wav_queries, appenddf, get_arrival_time
from obspy.taup.tau import TauPyModel
from obspy.geodetics.base import locations2degrees
from obspy.taup.helper_classes import TauModelError
import time

logger = logging.getLogger(__name__)

MAX_WORKERS = 7  # define the max thread workers
S_TIMEOUT = 30  # timeout (in seconds) for urllib2.urlopen when downloading stations
D_TIMEOUT = 30  # timeout (in seconds) for urllib2.urlopen when downloading stations

_STATION_LEVEL = 'channel'  # do not change this
_SEGMENTS_DATAURL_COLNAME = ' data.url '  # do not modify (must be a name not present in the
# models.Segment columns


def get_events(session, eventws, minmag, minlat, maxlat, minlon, maxlon, startiso, endiso):
    evt_query = get_query(eventws, minmagnitude=minmag, minlat=minlat, maxlat=maxlat,
                          minlon=minlon, maxlon=maxlon, start=startiso,
                          end=endiso, format='text')
    try:
        events_df = query2dframe(url_read(evt_query, decode='utf8'))

        if empty(events_df):
            raise ValueError("No events found")

        _ = len(events_df)
        events_df = normalize_fdsn_dframe(events_df, "event")
        if empty(events_df):
            raise ValueError("malformed data (NaNs, mismatching column or column names)")
        elif _ > len(events_df):
            logger.warning("%d events skipped (malformed data)\nurl:%s", _ - len(events_df),
                           evt_query)

        events = {}  # loop below a bit verbose, but better for debug
        for inst, _ in get_or_add_iter(session, df2dbiter(events_df, models.Event),
                                       on_add='commit'):
            if inst is not None:
                events[inst.id] = inst

        if len(events) < len(events_df):
            logger.warning("Processing %d of %d events (%d discarded due to internal db error)",
                           len(events), len(events_df), len(events_df) - len(events))
        if not events:
            raise ValueError("%d events found, no event saved (internal db error)" %
                             len(events_df))
        return events
    except (ValueError, urllib2.HTTPError, urllib2.URLError, httplib.HTTPException,
            socket.error) as exc:
        logger.error("%s\nURL:%s", str(exc), evt_query)
        return 1


def get_datacenters(session, **query_args):
    """Queries all datacenters and returns the local db model rows correctly added
    Rows already existing (comparing by datacenter station_query_url) are returned as well,
    but not added again
    :param query_args: any key value pair for the query. Note that 'service' and 'format' will
    be overiidden in the code with values of 'station' and 'format', repsecively
    """
    query_args['service'] = 'station'
    query_args['format'] = 'post'
    query = get_query('http://geofon.gfz-potsdam.de/eidaws/routing/1/query', **query_args)
#     query = ('http://geofon.gfz-potsdam.de/eidaws/routing/1/query?service=station&'
#              'start=%s&end=%s&format=post') % (start_time.isoformat(), end_time.isoformat())
    dc_result = url_read(query, decode='utf8', on_exc=lambda exc: logger.error("%s\nurl: %s",
                                                                               str(exc), query))

    # add to db the datacenters read. Two little hacks:
    # 1) parse dc_result string and assume any new line starting with http:// is a valid station
    # query url
    # 2) When adding the datacenter, the table column dataselect_query_url (when not provided, as
    # in this case) is assumed to be the same as station_query_url by replacing "/station" with
    # "/dataselect"

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

    dcenters = session.query(models.DataCenter).all()
#     logger.debug("%d datacenters found, %d newly added, %d skipped (internal db error)\nurl: %s",
#                  len(dcenters), new, err, query)
    # do not return only new datacenters, return all of them
    return {dcen.id: dcen for dcen in dcenters}


def get_stations_df(session, url, raw_data, min_sample_rate):
    """FIXME: write doc!"""
    try:
        if not raw_data:
            raise ValueError("No data")  # query2dframe below handles empty data,
        # but we want meaningful log
        stations_df = query2dframe(raw_data)
        if empty(stations_df):
            raise ValueError("Invalid data")
        _ = len(stations_df)
        stations_df = normalize_fdsn_dframe(stations_df, _STATION_LEVEL)
        if empty(stations_df):
            raise ValueError("malformed data")
        elif _ > len(stations_df):
            logger.warning("%d stations skipped (malformed data)\nurl: %s",
                           _ - len(stations_df),
                           url)
        if min_sample_rate > 0:
            srate_col = models.Channel.sample_rate.key
            tmp = stations_df[stations_df[srate_col] >= min_sample_rate]
            msg = "%d of %d stations discarded (sample rate < %s Hz)" % \
                (len(stations_df) - len(tmp), len(stations_df), str(min_sample_rate))
            if empty(tmp):
                raise ValueError(msg)
            elif len(stations_df) > len(tmp):
                logger.warning("%s\nurl: %s", msg, url)
            stations_df = tmp

        _ = len(stations_df)
        stations_df = save_stations_df(session, stations_df)
        if empty(stations_df):
            raise ValueError("Unable to save data to database, skipping station")
        elif _ > len(stations_df):
            logger.warning("%d stations skipped (internal db error)\nurl: %s",
                           _ - len(stations_df),
                           url)
        return stations_df
    except (ValueError,) as exc:
        logger.warning("%s\nurl: %s", str(exc), url)
        return empty()


def get_stations(session, events, datacenters, search_radius_args, station_timespan, channels,
                 min_sample_rate, bar):
        # calculate search radia:
    magnitudes = np.array([evt.magnitude for evt in events.itervalues()])
    max_radia = get_search_radius(magnitudes, *search_radius_args)

    STATION_LEVEL = 'channel'
    urls2tuple = {}
    for dcen_id, dcen in datacenters.iteritems():
        for max_radius, (evt_id, evt) in zip(max_radia, events.iteritems()):
            start = evt.time - timedelta(hours=station_timespan[0])
            end = evt.time + timedelta(hours=station_timespan[1])
            url = get_query(dcen.station_query_url,
                            latitude="%3.3f" % evt.latitude,
                            longitude="%3.3f" % evt.longitude,
                            maxradius=max_radius,
                            start=start.isoformat(), end=end.isoformat(),
                            channel=','.join(channels), format='text', level=STATION_LEVEL)
            urls2tuple[url] = (evt, dcen)

    stations_stats_df = pd.DataFrame(columns=[d.station_query_url
                                              for d in datacenters.itervalues()],
                                     index=['OK', 'N/A empty',
                                            'N/A server_error', 'N/A discarded'],
                                     data=0)
    ret = {}

    def onsuccess(data, url, index):  # pylint:disable=unused-argument
        """function executed when a given url has successfully downloaded data"""
        bar.update(1)
        tup = urls2tuple[url]
        evt, dcen = tup[0], tup[1]
        if data:
            stations_stats_df.loc['OK', dcen.station_query_url] += 1
            df = get_stations_df(session, url, data, min_sample_rate)
            if empty(df):
                stations_stats_df.loc['N/A malformed', dcen.station_query_url] += 1
            else:
                ret[(evt.id, dcen.id)] = df
        else:
            logger.warning("%s\nurl:%s", "empty data", url)
            stations_stats_df.loc['N/A empty', dcen.station_query_url] += 1

    def onerror(exc, url, index):  # pylint:disable=unused-argument
        """function executed when a given url has failed downloading data"""
        bar.update(1)
        logger.warning("%s\nurl:%s", str(exc), url)
        dcen_station_query = urls2tuple[url][1].station_query_url
        stations_stats_df.loc['N/A server_error', dcen_station_query] += 1

    read_async(urls2tuple.keys(), onsuccess, onerror, max_workers=MAX_WORKERS, decode='utf8',
               timeout=S_TIMEOUT)

    return ret, stations_stats_df


def calculate_times(stations_df, evt, ptimespan, distances_cache_dict={}, times_cache_dict={}):
    event_distances_degrees = []
    arrival_times = []
    model = TauPyModel('ak135')
    for _, sta in stations_df.iterrows():
        coordinates = (evt.latitude, evt.longitude,
                       sta[models.Station.latitude.key], sta[models.Station.longitude.key])
        degrees = distances_cache_dict.get(coordinates, None)
        if degrees is None:
            degrees = locations2degrees(*coordinates)
            distances_cache_dict[coordinates] = degrees
        event_distances_degrees.append(degrees)

        coordinates = (degrees, evt.depth_km, evt.time)
        arr_time = times_cache_dict.get(coordinates, None)
#         if arr_time is None:
#             # get_arrival_time is ... time consuming. Use session to query for an already calculated
#             # value:
#             if session:
#                 # Note on the query below: the filter on the Event class is made database side
#                 # on the Event associated to the Segment thanks to sqlAlchemy relationships
#                 # (see models.py).
#                 # Thus, if seg is not None, we will have:
#                 # seg.event_distance_deg == degrees (trivial)
#                 # seg.event.time == evt.time
#                 # seg.event.depth_km == evt.depth_km
#                 # For info see:
#                 # http://stackoverflow.com/questions/16589208/attributeerror-while-querying-neither-instrumentedattribute-object-nor-compa
#                 seg = session.query(models.Segment).\
#                         filter(and_(models.Segment.event_distance_deg == degrees,
#                                     models.Event.time == evt.time,
#                                     models.Event.depth_km == evt.depth_km)).first()
#                 if seg:
#                     arr_time = seg.arrival_time
        if arr_time is None:
            try:
                arr_time = get_arrival_time(*(coordinates + (model,)))
                times_cache_dict[coordinates] = arr_time
            except (TauModelError, ValueError) as exc:
                logger.warning("skipping station %s, error in arrival time calculation: %s",
                               str(sta.id), str(exc))
        arrival_times.append(arr_time)

    atime_colname = models.Segment.arrival_time.key  # limit length of next lines ...
    ret = pd.DataFrame({models.Segment.event_distance_deg.key: event_distances_degrees,
                        atime_colname: arrival_times}).dropna(subset=[atime_colname], axis=0)
    ret[models.Segment.start_time.key] = ret[atime_colname] - timedelta(minutes=ptimespan[0])
    ret[models.Segment.end_time.key] = ret[atime_colname] + timedelta(minutes=ptimespan[1])

    return ret


def get_segments(session, events, datacenters, stations, ptimespan, bar):
    segments = {dcen_id: empty() for dcen_id in datacenters}
    skipped_already_d = {dcen_id: 0 for dcen_id in datacenters}
    distances_cache_dict = {}
    arrivaltimes_cache_dict = {}

    for (evt_id, dcen_id), stations_df in stations.iteritems():
        bar.update(1)
        if empty(stations_df):  # for safety
            continue
        segments_df = calculate_times(stations_df, events[evt_id], ptimespan,
                                      distances_cache_dict, arrivaltimes_cache_dict)
        segments_df[models.Segment.channel_id.key] = stations_df[models.Channel.id.key]
        segments_df[models.Segment.event_id.key] = evt_id

#             segments_df = get_segments_df(session, stations_df, events[evt_id], ptimespan,
#                                           distances_cache_dict, arrivaltimes_cache_dict)

        # we will purge already downloaded segments, and use the index of the purged segments
        # to filter out stations, too. For this, we need to be sure they have the same index
        # before these operations:
        stations_df.reset_index(drop=True, inplace=True)
        segments_df.reset_index(drop=True, inplace=True)
        oldsegcount = len(segments_df)
        segments_df = purge_already_downloaded(session, segments_df)
        skipped_already_d[dcen_id] += (oldsegcount - len(segments_df))
        # purge stations, too:
        stations_df = stations_df[stations_df.index.isin(segments_df.index.values)]
        # set the wav query as dataframe index:
        segments_df = set_wav_queries(datacenters[dcen_id], stations_df, segments_df,
                                      _SEGMENTS_DATAURL_COLNAME)
        segments[dcen_id] = appenddf(segments[dcen_id], segments_df)

    return segments, skipped_already_d


def download_segments(session, segments_df, max_error_count, stats, bar):

    stat_keys = ['OK saved', 'N/A empty', 'N/A server_error', 'N/A other_reason',
                 'N/A localdb_error']

    if stats is None:
        stats = pd.Series(index=stat_keys, data=0)
    else:
        for key in stat_keys:
            if key not in stats:
                stats[key] = 0

    if empty(segments_df):
        return stats

    segments_df[models.Segment.data.key] = None

    # set_index as urls. this is much faster when locating a dframe row compared to
    # df[df[df_urls_colname] == some_url]
    segments_df.set_index(_SEGMENTS_DATAURL_COLNAME, inplace=True)
    urls = segments_df.index.values

    def onsuccess(data, url, index):  # pylint:disable=unused-argument
        """function executed when a given url has succesfully downloaded `data`"""
        bar.update(1)
        segments_df.loc[url, models.Segment.data.key] = data  # avoid pandas SettingWithCopyWarning

    def onerror(exc, url, index):  # pylint:disable=unused-argument
        """function executed when a given url has failed"""
        bar.update(1)
        logger.warning("%s\nurl:%s", str(exc), url)
        stats['N/A server_error'] += 1
        if stats['N/A server_error'] >= max_error_count:
            return False

    # now download Data:
    read_async(urls, onsuccess, onerror, max_workers=MAX_WORKERS, timeout=D_TIMEOUT)

    tmp_df = segments_df.dropna(subset=[models.Segment.data.key])
    null_data_count = len(segments_df) - len(tmp_df)
    segments_df = tmp_df
    # get empty data, then remove it:
    segments_df[models.Segment.data.key].replace('', np.nan, inplace=True)
    tmp_df = segments_df.dropna(subset=[models.Segment.data.key])
    stats['N/A empty'] = len(segments_df) - len(tmp_df)
    segments_df = tmp_df

    stats['N/A other_reason'] = null_data_count - stats['N/A server_error']
    if stats['N/A other_reason']:
        bar.update(stats['N/A other_reason'])

    if not empty(segments_df):
        for model_instance in df2dbiter(segments_df, models.Segment, False, False):
            session.add(model_instance)
            if commit(session):
                stats['OK saved'] += 1
            else:
                stats['N/A localdb_error'] += 1

        # reset_index as integer. This might not be the old index if the old one was not a
        # RangeIndex (0,1,2 etcetera). But it shouldn't be an issue
        # Note: 'drop=False' to restore 'df_urls_colname' column:
        segments_df.reset_index(drop=False, inplace=True)

    return stats


class dummyprogressbar(object):

    def __init__(self, *a, **kw):
        pass

    def update(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a, **kw):
        pass

# from click import progressbar
from click._termui_impl import ProgressBar
from click.globals import resolve_color_default
def progressbar(iterable=None, length=None, label=None, show_eta=True,
                show_percent=None, show_pos=False,
                item_show_func=None, fill_char='#', empty_char='-',
                bar_template='%(label)s  [%(bar)s]  %(info)s',
                info_sep='  ', width=36, file=None, color=None):
    color = resolve_color_default(color)
    return Pbar(iterable=iterable, length=length, show_eta=show_eta,
                       show_percent=show_percent, show_pos=show_pos,
                       item_show_func=item_show_func, fill_char=fill_char,
                       empty_char=empty_char, bar_template=bar_template,
                       info_sep=info_sep, file=file, label=label,
                       width=width, color=color)
class Pbar(ProgressBar):
    def __init__(self, iterable=None, **v):
        v['width'] = 25
        super(Pbar, self).__init__(iterable, **v)
        self._s_buflen = int(self.length / float(self.width))
        self._s_buf = 0

    def update(self, n_steps):
        self._s_buf += n_steps
        if (self._s_buf >= self._s_buflen) or \
           (self.length_known and (self.pos+self._s_buf) >= self.length):
            super(Pbar, self).update(self._s_buf)
            self._s_buf = 0


def main(session, run_id, eventws, minmag, minlat, maxlat, minlon, maxlon, search_radius_args,
         channels, start, end, ptimespan, station_timespan, min_sample_rate, isterminal=False):
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

    # progressbar = click.progressbar if isterminal else dummyprogressbar

    STEPS = 4

    # write the class labels:
    for _, _ in get_or_add_iter(session, df2dbiter(class_labels_df,
                                                   models.Class,
                                                   harmonize_columns_first=True,
                                                   harmonize_rows=True), on_add='commit'):
        pass

    startiso = start.isoformat()
    endiso = end.isoformat()

    logger.info("")
    logger.info("STEP 1/%d: Querying events and datacenters", STEPS)
    # Get events, store them in the db, returns the event instances (db rows) correctly added:

    events = get_events(session, eventws, minmag, minlat, maxlat, minlon, maxlon, startiso, endiso)
    if not events:
        return 1

    # Get datacenters, store them in the db, returns the dc instances (db rows) correctly added:
    datacenters = get_datacenters(session, start=startiso, end=endiso)
    if not datacenters:
        return 1

    logger.info("")
    logger.info(("STEP 2/%d: Querying %d datacenter(s) for stations (level=channel) "
                 "nearby %d event(s) found"), STEPS, len(datacenters), len(events))

    with progressbar(length=len(events)*len(datacenters)) as bar:
        stations, s_stats_df = get_stations(session, events, datacenters, search_radius_args,
                                            station_timespan, channels, min_sample_rate, bar)

    logger.info("")
    logger.info(("STEP 3/%d: Preparing segments download: calculating P-arrival times "
                 "and time ranges"), STEPS)

    with progressbar(length=len(stations)) as bar:
        segments, skipped_already_d = get_segments(session, events, datacenters, stations,
                                                   ptimespan, bar)

    segments_count = sum([len(seg_df) for seg_df in segments.itervalues()])
    logger.info("")
    logger.info("STEP 3/%d: Querying Datacenter WS for %d segments", STEPS, segments_count)

    max_error_count = 5
    d_stats_df = pd.DataFrame()
    with progressbar(length=segments_count) as bar:
        for dcen_id, segments_df in segments.iteritems():
            if not empty(segments_df):
                segments_df[models.Segment.datacenter_id.key] = dcen_id
                segments_df[models.Segment.run_id.key] = run_id
            stats = download_segments(session, segments_df, max_error_count, None, bar)
            stats['Ok already_downloaded'] = skipped_already_d[dcen_id]
            # append dataframe column
            d_stats_df[datacenters[dcen_id].dataselect_query_url] = stats

    logger.info("")
    logger.info("Summary Station WS query info:")
    logger.info(dc_stats_str(s_stats_df, transpose=True))
    logger.info("")
    logger.info("Summary Datacenter WS info :")
    logger.info(dc_stats_str(d_stats_df, transpose=True))
    logger.info("")

    return 0
