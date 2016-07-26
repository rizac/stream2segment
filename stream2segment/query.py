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

# standard imports:
from StringIO import StringIO
import sys
import logging
from datetime import timedelta, datetime
# third party imports:
import numpy as np
import pandas as pd
import yaml
from click import progressbar

from stream2segment.utils import url_read, tounicode  # , Progress
from stream2segment.s2sio import db
from stream2segment import __version__ as program_version
from stream2segment.classification import UNKNOWN_CLASS_ID
from stream2segment.classification import class_labels_df
from pandas.compat import zip
# IMPORT OBSPY AT END! IT MESSES UP WITH IMPORTS!
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees
from obspy.taup.helper_classes import TauModelError
from stream2segment.s2sio.db import DbHandler, models
# from stream2segment.s2sio.db import DbHandler

# from stream2segment.utils import DataFrame  # overrides DataFrame to allow case-insensitive
from pandas import DataFrame
from stream2segment.s2sio.db.pd_sql_utils import add_or_get, harmonize_columns,\
    harmonize_rows, df_to_table_iterrows, get_or_add_all, get_or_add, flush, commit
from sqlalchemy.exc import IntegrityError
from sqlalchemy.engine import create_engine
from sqlalchemy.orm.session import sessionmaker
from stream2segment.s2sio.db.models import Base
from stream2segment.processing import process
# slicing by columns. Some datacenters are not returning the same columns (concerning case. E.g.
# 'latitude' vs 'Latitude')


def get_min_travel_time(source_depth_in_km, distance_in_degree, model='ak135'):
    """
        Assess and return the travel time of P phases.
        Uses obspy.getTravelTimes
        :param source_depth_in_km: Depth in kilometer.
        :type source_depth_in_km: float
        :param distance_in_degree: Distance in degrees.
        :type distance_in_degree: float
        :param model: Either ``'iasp91'`` or ``'ak135'`` velocity model.
         Defaults to 'ak135'.
        :type model: str, optional
        :return the number of seconds of the assessed arrival time, or None in case of error
        :raises: ValueError (wrapping TauModel error in case)
    """
    taupmodel = TauPyModel(model)
    try:
        tt = taupmodel.get_travel_times(source_depth_in_km, distance_in_degree)
        # return min((ele['time'] for ele in tt if (ele.get('phase_name') or ' ')[0] == 'P'))

        # Arrivals are returned already sorted by time!
        return tt[0].time

        # return min(tt, key=lambda x: x.time).time
        # return min((ele.time for ele in tt))
    except (TauModelError, ValueError, AttributeError) as err:
        raise ValueError(("Unable to find minimum travel time (dist=%s, depth=%s, model=%s). "
                          "Source error: %s: %s"),
                         str(distance_in_degree), str(source_depth_in_km), str(model),
                         err.__class__.__name__, str(err))


def get_arrival_time(distance_in_degrees, ev_depth_km, ev_time):
    """
        Returns the _pwave arrival time, as float
        :param distance_in_degrees: the distance in degrees between station and event
        :type distance_in_degrees: float. See obspy.locations2degrees
        :param ev_depth_km: the event depth in km
        :type ev_depth_km: numeric
        :param ev_time: the event time
        :type ev_time: datetime object
        :return: the P-wave arrival time
    """
    travel_time = get_min_travel_time(ev_depth_km, distance_in_degrees)
    arrival_time = ev_time + timedelta(seconds=float(travel_time))
    return arrival_time


def get_time_range(orig_time, days=0, hours=0, minutes=0, seconds=0):
    """
        Returns the tuple (orig_time - timeDeltaBefore, orig_time + timeDeltaAfter), where the deltas
        are built according to the given parameters. Any of the parameters can be an int
        OR an iterable (list, tuple) of two elements specifying the days before and after,
        respectively

        :Example:
            - get_time_range(t, seconds=(1,2)) returns the tuple with elements:
                - t minus 1 second
                - t plus 2 seconds
            - get_time_range(t, minutes=4) returns the tuple with elements:
                - t minus 4 minutes
                - t plus 4 minutes
            - get_time_range(t, days=1, seconds=(1,2)) returns the tuple with elements:
                - t minus 1 day and 1 second
                - t plus 1 day and 2 seconds

        :param days: the day shift from orig_time
        :type days: integer or tuple of positive integers (of length 2)
        :param minutes: the minutes shift from orig_time
        :type minutes: integer or tuple of positive integers (of length 2)
        :param seconds: the second shift from orig_time
        :type seconds: integer or tuple of positive integers (of length 2)
        :return: the tuple (timeBefore, timeAfter)
        :rtype: tuple of datetime objects (timeBefore, timeAfter)
    """
    td1 = []
    td2 = []
    for val in (days, hours, minutes, seconds):
        try:
            td1.append(val[0])
            td2.append(val[1])
        except TypeError:
            td1.append(val)
            td2.append(val)

    start = orig_time - timedelta(days=td1[0], hours=td1[1], minutes=td1[2], seconds=td1[3])
    endt = orig_time + timedelta(days=td2[0], hours=td2[1], minutes=td2[2], seconds=td2[3])
    return start, endt


def get_search_radius(mag, mmin=3, mmax=7, dmin=1, dmax=5):
    """From a given magnitude, determines and returns the max radius (in degrees).
        Given dmin and dmax and mmin and mmax (FIXME: TO BE CALIBRATED!),
        this function returns D from the f below:

             |
        dmax +                oooooooooooo
             |              o
             |            o
             |          o
        dmin + oooooooo
             |
             ---------+-------+------------
                    mmin     mmax

    """
    if mag < mmin:
        radius = dmin
    elif mag > mmax:
        radius = dmax
    else:
        radius = dmin + (dmax - dmin) / (mmax - mmin) * (mag - mmin)
    return radius


def get_events_df(**kwargs):
    """
        Returns a tuple of two elements: the first one is the DataFrame representing the stations
        read from the specified arguments. The second is the the number of rows (denoting stations)
        which where dropped from the url query due to errors in parsing
        :param kwargs: a variable length list of arguments, including:
            eventws (string): the event web service
            minmag (float): the minimum magnitude
            start (string): the event start, in string format (e.g., datetime.isoformat())
            end (string): the event end, in string format (e.g., datetime.isoformat())
            minlon (float): the event min longitude
            maxlon (float): the event max longitude
            minlat (float): the event min latitude
            maxlat (float): the event max latitude
        :raise: ValueError, TypeError, IOError
    """
    event_query = ('%(eventws)squery?minmagnitude=%(minmag)1.1f&start=%(start)s'
                   '&minlon=%(minlon)s&maxlon=%(maxlon)s&end=%(end)s'
                   '&minlat=%(minlat)s&maxlat=%(maxlat)s&format=text') % kwargs

    result = url_read(event_query, decoding='utf8')

    return query2dframe(result)


# def evt_to_dframe(event_query_result):
#     """
#         :return: the tuple dataframe, dropped_rows (int >=0)
#         raises: ValueError
#     """
#     dfr = query2dframe(event_query_result)
#     oldlen = len(dfr)
#     if not dfr.empty:
#         for key, cast_func in {'Time': pd.to_datetime,
#                                'Depth/km': pd.to_numeric,
#                                'Latitude': pd.to_numeric,
#                                'Longitude': pd.to_numeric,
#                                'Magnitude': pd.to_numeric,
#                                }.iteritems():
#             dfr[key] = cast_func(dfr[key], errors='coerce')
# 
#         dfr.dropna(inplace=True)
# 
#     return dfr, oldlen - len(dfr)


def get_stations_df(datacenter, channels_list, orig_time, lat, lon, max_radius, level='channel'):
    """
        Returns a tuple of two elements: the first one is the DataFrame representing the stations
        read from the specified arguments. The second is the the number of rows (denoting stations)
        which where dropped from the url query due to errors in parsing
        :param datacenter: the datacenter, e.g.: "http://ws.resif.fr/fdsnws/station/1/query"
        :type datacenter: string
        :param channels_list: the list of channels, e.g. ['HL?', 'SL?', 'BL?'].
        :type channels_list: iterable (e.g., list)
        :param orig_time: the origin time. The request will be built with a time start and end of
            +-1 day from orig_time
        :type orig_time: date or datetime
        :param lat: the latitude
        :type lat: float
        :param lon: the longitude
        :type lon: float
        :param max_radius: the radius distance from lat and lon, in degrees FIXME: check!
        :type max_radius: float
        :return: the DataFrame representing the stations, and the stations dropped (int)
        :raise: ValueError, TypeError, IOError
    """

    start, endt = get_time_range(orig_time, days=1)
    station_query = ('%s?latitude=%3.3f&longitude=%3.3f&'
                     'maxradius=%3.3f&start=%s&end=%s&channel=%s&format=text&level=%s')
    aux = station_query % (datacenter, lat, lon, max_radius, start.isoformat(),
                           endt.isoformat(), ','.join(channels_list), level)
    result = url_read(aux, decoding='utf8')

    return query2dframe(result)


def query2dframe(query_result_str):
    """
        Returns a pandas dataframne fro the given query_result_str
        :param: query_result_str
        :raise: ValueError in case of errors
    """
    if not query_result_str:
        return DataFrame()

    events = query_result_str.splitlines()

    data = None
    columns = [e.strip() for e in events[0].split("|")]
    for evt in events[1:]:
        evt_list = evt.split('|')
        # Use numpy and then build the dataframe
        # For info on other solutions:
        # http://stackoverflow.com/questions/10715965/add-one-row-in-a-pandas-dataframe:
        if data is None:
            data = [evt_list]
        else:
            data = np.append(data, [evt_list], axis=0)

    if data is not None:
        # check that data rows and columns have the same length
        # cause DataFrame otherwise might do some weird stuff (e.g., one
        # column and rows of N>1 elemens, the DataFrame is built with
        # a single column packing those N elements as list in it)
        # Note that if we are here we are sure data rows are the same length
        np.append(data, [columns], axis=0)

    return DataFrame(data=data, columns=columns)


def get_wav_query(datacenter, network, station_name, location, channel, start_time, end_time):
    """Returns the wav query from the arguments, all strings except the last two (datetime)"""
    # qry = '%s/dataselect/1/query?network=%s&station=%s&location=%s&channel=%s&start=%s&end=%s'
    qry = '%s?network=%s&station=%s&location=%s&channel=%s&start=%s&end=%s'
    return qry % (datacenter, network, station_name, location, channel, start_time.isoformat(),
                  end_time.isoformat())


def read_wav_data(query_str):
    """Returns the wav data (mseed binary data) from the given query_str (string)"""
    try:
        return url_read(query_str)
    except (IOError, ValueError, TypeError) as _:
        return None


def pd_str(dframe):
    """Returns a dataframe to string with all rows and all columns, used for printing to log"""
    with pd.option_context('display.max_rows', len(dframe),
                           'display.max_columns', len(dframe.columns),
                           'max_colwidth', 50, 'expand_frame_repr', False):
        return str(dframe)


def search_all_stations(events, datacenters, search_radius_args, channels,
                        min_sample_rate, logger=None, progresslistener=None):
    """
    :param events: a dict of ids mapped to model instances
    :param progresslistener: a function accepting an integer (starting from 1 until
    len(events_df) * len(datacenters)
    denoting the progress of the downloaded segments data
    """

    n_step = 0
    # initialize two empty dataframes (which we will return):
    stations_df = pd.DataFrame()
    segments_df = pd.DataFrame()

    for evt_id in events:
        evt = events[evt_id]
        evt_mag = evt.magnitude
        ev_loc_name = evt.event_location_name
        ev_time = evt.time
        ev_lat = evt.latitude
        ev_lon = evt.longitude

        max_radius = get_search_radius(evt_mag,
                                       search_radius_args[0],
                                       search_radius_args[1],
                                       search_radius_args[2],
                                       search_radius_args[3])

        for dcen in datacenters:

            sta_query = dcen.station_query_url
            # dataselect_query = dcen.dataselect_query_url

            n_step += 1
            if progresslistener:
                progresslistener(n_step)

            msg = ("Event '%s': querying stations within %5.3f deg. "
                   "to %s") % (ev_loc_name, max_radius, sta_query)

            if logger:
                logger.debug("")
                logger.debug(msg)

            try:
                stations_cha_level = get_stations_df(sta_query, channels, ev_time, ev_lat,
                                                     ev_lon, max_radius)
            except (IOError, ValueError, TypeError) as exc:
                if logger:
                    logger.warning(exc.__class__.__name__ + ": " + str(exc))
                continue

            if logger:
                logger.debug('%d stations found (from: %s, channel: %s)',
                             len(stations_cha_level), str(sta_query), str(channels))

            if stations_cha_level.empty:
                continue

            tmp_seg_df = pd.DataFrame(columns=[models.Segment.event_id.key,
                                               models.Segment.datacenter_id.key],
                                      data=[[evt_id, dcen.id]] * len(stations_cha_level))

            if stations_df.empty:
                stations_df = stations_cha_level
                segments_df = tmp_seg_df
            else:
                stations_df = stations_df.append(stations_cha_level, ignore_index=True)
                segments_df = segments_df.append(tmp_seg_df, ignore_index=True)

    # reset indices to be sure: from 0 to dataframe length
    # note that the two dataframes have the same number of rows, so its safe
    stations_df.reset_index(drop=True)
    segments_df.reset_index(drop=True)

    # normalize the dataframe, drop NA etcetera:
    count = len(stations_df)
    if not stations_df.empty:
        # rename columns according to fdsn table defined in models, purge nan columns etcetera:
        stations_df = normalize_fdsn_dframe(models.Station, stations_df, logger)
        stations_df = normalize_fdsn_dframe(models.Channel, stations_df, logger)

    if len(stations_df) != count and logger:
        logger.warning(("%d stations skipped (bad values, e.g. NaN's)") %
                       (count - len(stations_df)))
        # filter out segments also:
        segments_df = segments_df.loc[stations_df.index.values]

    if not stations_df.empty and min_sample_rate > 0:
        srate_col = models.Channel.sample_rate.key
        tmp = stations_df[stations_df[srate_col] >= min_sample_rate]
        if len(tmp) != len(stations_df):
            if logger:
                logger.warning(("%d stations skipped (sample rate < %s Hz)") %
                               (len(stations_df) - len(tmp), str(min_sample_rate)))
            stations_df = tmp
            # filter out segments also:
            segments_df = segments_df.loc[stations_df.index.values]

    if not stations_df.empty:
        # append to stations_df the datacenters ids:
        stations_df[models.Station.datacenter_id.key] = segments_df[models.Segment.datacenter_id.key]
    return stations_df, segments_df


def calculate_times(events, stations_df, segments_df, ptimespan):
    if stations_df.empty or segments_df.empty:
        return

    # init column names (according to db model):
    sta_lat_col = models.Station.latitude.key
    sta_lon_col = models.Station.longitude.key
    ev_lat_col = models.Event.latitude.key
    ev_lon_col = models.Event.longitude.key
    ev_id_col = models.Event.id.key
    ev_depth_km_col = models.Event.depth_km.key
    ev_time_col = models.Event.time.key

    seg_ev2sta_dist_col = models.Segment.event_distance_deg.key
    seg_atime_col = models.Segment.arrival_time.key
    seg_stime_col = models.Segment.start_time.key
    seg_etime_col = models.Segment.end_time.key
    seg_eventid_col = models.Segment.event_id.key

    sta_net_col = models.Station.network.key
    sta_sta_col = models.Station.station.key
    cha_loc_col = models.Channel.location.key
    cha_cha_col = models.Channel.channel.key

    # insert columns we will populate here (order is irrelevant for db output)
    segments_df.insert(0, seg_ev2sta_dist_col, float('nan'))
    segments_df.insert(0, seg_stime_col, pd.NaT)
    segments_df.insert(0, seg_atime_col, pd.NaT)
    segments_df.insert(0, seg_etime_col, pd.NaT)

    # reset indices to be sure: from 0 to dataframe length for both dataframes
    # note that the two dataframes are assumed to have the same number of rows, so its safe
    stations_df.reset_index(drop=True)
    segments_df.reset_index(drop=True)

    # Now calculate. As arrival_times is computationally expensive. We might have
    # DUPLICATED stations so we select only those unique according to Latitude and
    # longitude
    stations_unique = stations_df.drop_duplicates(subset=(sta_lat_col, sta_lon_col))
    # NOTE: the function above calculates duplicated if ALL subset(s) are equal, if any
    # is equal then does not drop them (what we want)

    def loc2degrees(row):
        # get the row index (accessible by .name.. weird)
        row_index = row.name
        ev_id = segments_df.loc[row_index][seg_eventid_col]
        event = events[ev_id]
        ev_lat = event.latitude
        ev_lon = event.longitude
        sta_lat = row[sta_lat_col]
        sta_lon = row[sta_lon_col]
        return locations2degrees(sta_lat, sta_lon, ev_lat, ev_lon)

    stations_unique[seg_ev2sta_dist_col] = stations_unique.apply(loc2degrees, axis=1)

    def get_arr_times(row):
        distance_in_degrees = row[seg_ev2sta_dist_col]
        row_index = row.name
        ev_id = segments_df.loc[row_index][seg_eventid_col]
        event = events[ev_id]
        ev_depth_km = event.depth_km
        ev_time = event.time
        return get_arrival_time(distance_in_degrees, ev_depth_km, ev_time)

    stations_unique[seg_atime_col] = stations_unique.apply(get_arr_times, axis=1)

    def populate_segments(row):
        sta_lat = row[sta_lat_col]
        sta_lon = row[sta_lon_col]
        mask = (stations_df[sta_lat_col] == sta_lat) & (stations_df[sta_lon_col] == sta_lon)
        # d.loc[d['int']<50, 'float'] = 7
        segments_df.loc[mask, seg_atime_col] = row[seg_atime_col]
        segments_df.loc[mask, seg_ev2sta_dist_col] = row[seg_ev2sta_dist_col]

    stations_unique.apply(populate_segments, axis=1)

    segments_df[seg_stime_col] = segments_df[seg_atime_col] - timedelta(minutes=ptimespan[0])
    segments_df[seg_etime_col] = segments_df[seg_atime_col] + timedelta(minutes=ptimespan[1])


def write_and_download_data(session, run_id, stations_cha_level_df,
                            segments_df, logger=None,
                            progresslistener=None):
    """
    :param progresslistener: a function accepting an integer (starting from 1 until len(segments_df)
    denoting the progress of the downloaded segments data
    """

    # NOTE: we already harmonized rows and columns, so we skip it
    station_rows = df_to_table_iterrows(models.Station, stations_cha_level_df,
                                        harmonize_columns_first=False,
                                        harmonize_rows=False)
    channel_rows = df_to_table_iterrows(models.Channel, stations_cha_level_df,
                                        harmonize_columns_first=False,
                                        harmonize_rows=False)
    segment_rows = df_to_table_iterrows(models.Segment, segments_df,
                                        harmonize_columns_first=False,
                                        harmonize_rows=False)

    ret_segs = []
    count = 0

    def sta_getter(row):
        return (models.Station.network == row.network) & (models.Station.station == row.station)

    for sta, cha, seg in zip(station_rows, channel_rows, segment_rows):

        if sta is None or cha is None or seg is None:  # for safety...
            continue

        sta, _ = get_or_add(session, sta, [models.Station.network, models.Station.station])

        if sta is None:
            continue

        cha_tmp = session.query(models.Channel).\
            filter((models.Channel.station_id == sta.id) &
                   (models.Channel.location == cha.location) &
                   (models.Channel.channel == cha.channel)).first()

        if not cha_tmp:
            sta.channels.append(cha)
            if not flush(session):  # FIXME: check if correct
                continue
        else:
            cha = cha_tmp

        seg_tmp = session.query(models.Segment).\
            filter((models.Segment.channel_id == cha.id) &
                   (models.Segment.start_time == seg.start_time) &
                   (models.Segment.end_time == seg.end_time)).first()

        if not seg_tmp:  # FIXME: check if correct
            dcen = session.query(models.DataCenter).\
                filter(models.DataCenter.id == seg.datacenter_id).first()

            if dcen is None:
                fg = 9

            query_url = get_wav_query(dcen.dataselect_query_url, sta.network, sta.station,
                                      cha.location, cha.channel, seg.start_time,
                                      seg.end_time)
            data = read_wav_data(query_url)
            if data:
                seg.data = data
                if logger:
                    logger.debug("%7d bytes downloaded from: %s" % (len(data), query_url))

                seg.run_id = run_id
                cha.segments.append(seg)
                if seg.channel_id == "RO.BAC..HNE":
                    fgh = 9
                if flush(session):
                    ret_segs.append(seg)
                else:
                    g = 9
        else:
            ret_segs.append(seg_tmp)

        if progresslistener:
            count += 1
            # progresslistener(count)

    return ret_segs


def normalize_fdsn_dframe(fdsn_model_class, fdsn_query_dataframe, logger):
    if fdsn_query_dataframe.empty:
        return fdsn_query_dataframe
    # rename columns. Note that for Stations and Channels models their column names MUST NOT OVERLAP
    # this has to be remembered if modifying models ORM (not an issue for now)
    # note that Station table needs an argument 'level' (channel in this case):
    kwargs = {'level': 'channel'} if fdsn_model_class == models.Station else {}
    fdsn_query_dataframe = fdsn_model_class.rename_cols(fdsn_query_dataframe, **kwargs)
    # convert columns to correct dtypes (datetime, numeric etcetera). Values not conforming
    # will be set to NaN or NaT or None, thus detectable via pandas.dropna or pandas.isnull
    fdsn_query_dataframe = harmonize_columns(fdsn_model_class, fdsn_query_dataframe)
    leng = len(fdsn_query_dataframe)
    # drop NA rows (NA for columns which are non- nullable):
    fdsn_query_dataframe = harmonize_rows(fdsn_model_class, fdsn_query_dataframe)

    if logger and leng-len(fdsn_query_dataframe):
        logger.warning("Table '%s': %d items skipped (invalid values for the db schema, e.g., "
                       "NaN)) will not be written to table nor further processed" %
                        (str(fdsn_model_class), leng-len(fdsn_query_dataframe),))

    return fdsn_query_dataframe


def get_events(session, logger=None, **args):
    """Queries all events and returns the local db model rows correctly added
    Rows already existing (comparing by event id) are returned as well, but not added again
    """
    try:
        events_df = get_events_df(**args)
        # rename columns
        events_df = normalize_fdsn_dframe(models.Event, events_df, logger)
    except (IOError, ValueError, TypeError) as err:
        logger.error(str(err))
        events_df = DataFrame(columns=models.Event.get_col_names(), data=[])

    # convert dataframe to records (df_to_table_iterrows),
    # add non existing records to db (get_or_add_all) comparing by events.id
    # return the added rows
    # Note: get_or_add_all has already flushed, so the returned model instances (db rows)
    # have the fields updated, if any
    return get_or_add_all(session, df_to_table_iterrows(models.Event, events_df))


def get_datacenters(session, logger, start_time, end_time):
    """Queries all datacenters and returns the local db model rows correctly added
    Rows already existing (comparing by datacenter station_query_url) are returned as well,
    but not added again
    """
    dcs_query = ('http://geofon.gfz-potsdam.de/eidaws/routing/1/query?service=station&'
                 'start=%s&end=%s&format=post' % (start_time.isoformat(), end_time.isoformat()))
    dc_result = url_read(dcs_query, decoding='utf8')

    # add to db the datacenters read. Two little hacks:
    # 1) parse dc_result string and assume any new line starting with http:// is a valid station
    # query url
    # 2) When adding the datacenter, the table column dataselect_query_url (when not provided, as
    # in this case) is assumed to be the same as station_query_url by replacing "/station" with
    # "/dataselect"
    ret_dcs = []
    for dcen in dc_result.split("\n"):
        if dcen[:7] == "http://":
            dc_row, isnew = get_or_add(session, models.DataCenter(station_query_url=dcen),
                                       [models.DataCenter.station_query_url])
            if dc_row is not None:
                ret_dcs.append(dc_row)
    return ret_dcs


def main(session, run_id, eventws, minmag, minlat, maxlat, minlon, maxlon, search_radius_args,
         channels, start, end, ptimespan, min_sample_rate, logger=None):
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

    # write the class labels:
    get_or_add_all(session,
                   df_to_table_iterrows(models.Class,
                                        class_labels_df,
                                        harmonize_columns_first=True,
                                        harmonize_rows=True))

    # a little bit hacky, but convert to dict as the function gets dictionaries
    # Note: we might want to use dict(locals()) as above but that does NOT
    # preserve order and tests should be rewritten. It's too much pain for the moment
    args = {"eventws": eventws,
            "minmag": minmag,
            "minlat": minlat,
            "maxlat": maxlat,
            "minlon": minlon,
            "maxlon": maxlon,
            "start": start.isoformat(),
            "end": end.isoformat()}

    logger.debug("")
    logger.info("STEP 1/4: Querying Event WS")

    # Get events, store them in the db, returns the event instances (db rows) correctly added:
    events = get_events(session, logger, **args)

    # convert to dict (we need it for faster search, and we might also avoid duplicated events)
    events = {evt.id: evt for evt in events}

    logger.info("STEP 2/4: Querying Datacenters")
    # Get datacenters, store them in the db, returns the dc instances (db rows) correctly added:
    datacenters = get_datacenters(session, logger, start, end)

    logger.debug("")
    msg = "STEP 3/4: Querying Station WS (level=channel)"
    logger.info(msg)

    # commit changes now in order not to loose datacenters and events:
    if not commit(session, on_exc=lambda exc: logger.error(str(exc))):
        return 1

    with progressbar(length=len(events) * len(datacenters)) as bar:
        stations_df, segments_df = search_all_stations(events, datacenters, search_radius_args,
                                                       channels, min_sample_rate, logger,
                                                       progresslistener=lambda i: bar.update(i))

    calculate_times(events, stations_df, segments_df, ptimespan)

    logger.debug("")
    logger.info("STEP 3/3: Querying Datacenter WS")
    segments_rows = []

    with progressbar(length=len(segments_df)) as bar:
        segments_rows = write_and_download_data(session, run_id,
                                                stations_df, segments_df, logger=logger,
                                                progresslistener=lambda i: bar.update(i))

    if not commit(session, on_exc=lambda exc: logger.error(str(exc))):
        return 1

    return 0
