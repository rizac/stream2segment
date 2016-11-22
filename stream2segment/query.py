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

from stream2segment.async import url_read
# from stream2segment.utils import tounicode  # , Progress
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
from stream2segment.s2sio.db.pd_sql_utils import harmonize_columns,\
    harmonize_rows, df2dbiter, get_or_add_iter, flush, commit
from sqlalchemy.exc import IntegrityError
from sqlalchemy import and_
from sqlalchemy.engine import create_engine
from sqlalchemy.orm.session import sessionmaker
from stream2segment.s2sio.db.models import Base
from stream2segment.processing import process
from itertools import count
from stream2segment.async import read_async
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
    isscalar = np.isscalar(mag)  # for converting back to scalar later
    mag = np.array(mag)  # copies data
    mag[mag < mmin] = dmin
    mag[mag > mmax] = dmax
    mag[(mag >= mmin) & (mag <= mmax)] = dmin + (dmax - dmin) / (mmax - mmin) * \
        (mag[(mag >= mmin) & (mag <= mmax)] - mmin)

    return mag[0] if isscalar else mag

#     if mag < mmin:
#         radius = dmin
#     elif mag > mmax:
#         radius = dmax
#     else:
#         radius = dmin + (dmax - dmin) / (mmax - mmin) * (mag - mmin)
#     return radius


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
    logger = kwargs['logger'] if 'logger' in kwargs else None
    query = ('%(eventws)squery?minmagnitude=%(minmag)1.1f&start=%(start)s'
             '&minlon=%(minlon)s&maxlon=%(maxlon)s&end=%(end)s'
             '&minlat=%(minlat)s&maxlat=%(maxlat)s&format=text') % kwargs

    try:
        result = url_read(query, decode='utf8')
    except (IOError, ValueError, TypeError) as exc:
        if logger:
            logger.error("query: '%s': %s" % (query, str(exc)))
        return empty()

    dframe = query2dframe(result)

    query_type = "event"
    try:
        dframe = rename_columns(dframe, query_type)
    except ValueError as exc:
        if logger:
            logger.error("query: '%s': %s" % (query, str(exc)))
        return empty()

    try:
        dframe2 = harmonize_fdsn_dframe(dframe, query_type)
        if logger:
            skipped = len(dframe)-len(dframe2)
            if skipped:
                logger.warning("query: '%s': %d %ss skipped (invalid values, e.g., "
                               "NaNs)) will not be written to table nor further processed" %
                               (query, skipped, query_type))  # query_type = 'event'
        dframe = dframe2
    except ValueError as exc:
        if logger:
            logger.error("query: '%s': %s" % (query, str(exc)))
        return empty()

#     if logger:
#         logger.debug("query: '%s': %d valid %ss found (from: %s, channel: %s)",
#                      query, len(dframe), query_type)  # query_type is "station" or "channel"

    return dframe


def get_station_query_url(datacenter, channels_list, orig_time, lat, lon, max_radius,
                          level='channel'):
    start, endt = get_time_range(orig_time, days=1)
    query = ('%s?latitude=%3.3f&longitude=%3.3f&'
             'maxradius=%3.3f&start=%s&end=%s&channel=%s&format=text&level=%s')
    return query % (datacenter, lat, lon, max_radius, start.isoformat(),
                    endt.isoformat(), ','.join(channels_list), level)


def get_stations_df2(query, raw_test_data, min_sample_rate=None, query_type='channel', logger=None):
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
        :param min_sample_rate: a float denoting the minimum sample rate, in Hz. Only taken into
        account if 'sample_rate' (or better, `models.Channel.sample_rate.key`) is a column
        of the resulting dataframe. This is True if level='channel', not True if level='station'
        (not tested for the other level options). If the parameter value is not greater than zero
        (which is True when None, the default), no filter on sample rate is done.
        :type min_sample_rate: float or None
        :return: the DataFrame representing the stations, and the stations dropped (int)
        :raise: ValueError, TypeError, IOError
    """
    dframe = query2dframe(raw_test_data)

    try:
        dframe = rename_columns(dframe, query_type)
    except ValueError as exc:
        if logger:
            logger.error("query: '%s': %s" % (query, str(exc)))
        return empty()  # FIXME: reachable??? logger error should exit!

    try:
        dframe2 = harmonize_fdsn_dframe(dframe, query_type)
        if logger:
            skipped = len(dframe)-len(dframe2)
            if skipped:
                logger.warning("query: '%s': %d %ss skipped (invalid values, e.g., "
                               "NaNs)) will not be written to table nor further processed" %
                               (query, skipped, query_type))  # query_type is "station" or "channel"
        dframe = dframe2
    except ValueError as exc:
        if logger:
            logger.error("query: '%s': %s" % (query, str(exc)))
        return empty()

    # filter out sampling rate lower than required one:
    if not empty(dframe) and min_sample_rate > 0:
        srate_col = models.Channel.sample_rate.key
        tmp = dframe[dframe[srate_col] >= min_sample_rate]
        if logger:
            skipped = len(dframe) - len(tmp)
            logger.warning(("query: '%s': %d stations skipped (sample rate < %s Hz)") %
                           (query, skipped, str(min_sample_rate)))
        dframe = empty() if empty(tmp) else tmp

#     if logger:
#         logger.debug("query: '%s': %d valid %ss found (from: %s, channel: %s)",
#                      query, len(dframe), query_type)  # query_type is "station" or "channel"
    return dframe


def get_stations_df(datacenter, channels_list, orig_time, lat, lon, max_radius,
                    min_sample_rate=None, level='channel', logger=None):
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
        :param min_sample_rate: a float denoting the minimum sample rate, in Hz. Only taken into
        account if 'sample_rate' (or better, `models.Channel.sample_rate.key`) is a column
        of the resulting dataframe. This is True if level='channel', not True if level='station'
        (not tested for the other level options). If the parameter value is not greater than zero
        (which is True when None, the default), no filter on sample rate is done.
        :type min_sample_rate: float or None
        :return: the DataFrame representing the stations, and the stations dropped (int)
        :raise: ValueError, TypeError, IOError
    """

    start, endt = get_time_range(orig_time, days=1)
    query = ('%s?latitude=%3.3f&longitude=%3.3f&'
             'maxradius=%3.3f&start=%s&end=%s&channel=%s&format=text&level=%s')
    aux = query % (datacenter, lat, lon, max_radius, start.isoformat(),
                   endt.isoformat(), ','.join(channels_list), level)
    result = url_read(aux, decode='utf8')

    dframe = query2dframe(result)

    query_type = level
    try:
        dframe = rename_columns(dframe, query_type)
    except ValueError as exc:
        if logger:
            logger.error("query: '%s': %s" % (query, str(exc)))
        return empty()  # FIXME: reachable??? logger error should exit!

    try:
        dframe2 = harmonize_fdsn_dframe(dframe, query_type)
        if logger:
            skipped = len(dframe)-len(dframe2)
            if skipped:
                logger.warning("query: '%s': %d %ss skipped (invalid values, e.g., "
                               "NaNs)) will not be written to table nor further processed" %
                               (query, skipped, query_type))  # query_type is "station" or "channel"
        dframe = dframe2
    except ValueError as exc:
        if logger:
            logger.error("query: '%s': %s" % (query, str(exc)))
        return empty()

    # filter out sampling rate lower than required one:
    if not empty(dframe) and min_sample_rate > 0:
        srate_col = models.Channel.sample_rate.key
        tmp = dframe[dframe[srate_col] >= min_sample_rate]
        if logger:
            skipped = len(dframe) - len(tmp)
            logger.warning(("query: '%s': %d stations skipped (sample rate < %s Hz)") %
                           (query, skipped, str(min_sample_rate)))
        dframe = empty() if empty(tmp) else tmp

#     if logger:
#         logger.debug("query: '%s': %d valid %ss found (from: %s, channel: %s)",
#                      query, len(dframe), query_type)  # query_type is "station" or "channel"
    return dframe


def save_stations_df(session, stations_df, logger=None):
    """
        stations_df is already harmonized
        FIXME: add logger capabilities!!!
    """
    sta_ids = []
    for sta, _ in get_or_add_iter(session,
                                  df2dbiter(stations_df, models.Station, False, False),
                                  [models.Station.network, models.Station.station],
                                  on_add='commit'):
        sta_ids.append(None if sta is None else sta.id)

    stations_df[models.Channel.station_id.key] = sta_ids
    channels_df = stations_df.dropna(subset=[models.Channel.station_id.key])

    cha_ids = []
    for cha, _ in get_or_add_iter(session,
                                  df2dbiter(channels_df, models.Channel, False, False),
                                  [models.Channel.station_id, models.Channel.location,
                                   models.Channel.channel],
                                  on_add='commit'):
        cha_ids.append(None if cha is None else cha.id)

    channels_df = channels_df.drop(models.Channel.station_id.key, axis=1)  # del station_id column
    channels_df.reset_index(drop=True, inplace=True)  # otherwise with `mask` below we have issues
    channels_ids = pd.DataFrame(data=cha_ids, columns=[models.Segment.channel_id.key])
    mask = ~(channels_ids.isnull().any(axis=1))  # we might omit any() and apply on the series
    # (is a single column dataframe)
    return channels_df[mask], channels_ids[mask]


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


def rename_columns(query_df, query_type):
    """Renames the columns of `query_df` according to the "standard" expected column names given by
    query_type, so that IO operation with the database are not suffering naming mismatch (e.g., non
    matching cases). If the number of columns of `query_df` does not match the number of expected
    columns, a ValueError is raised. The assumption is that any datacenter returns the *same* column
    in the *same* position, as guessing columns by name might be tricky (there is not only a problem
    of case sensitivity, but also of e.g. "#Network" vs "network". <-Ref needed!)
    :param query_df: the DataFrame resulting from an fdsn query, either events station
    (level=station) or station (level=channel)
    :param query_type: a string denoting the query type whereby `query_df` has been generated and
    determining the expected column names, so that `query_df` columns will be renamed accordingly.
    Possible values are "event", "station" (for a station query with parameter level=station) or
    "channel" (for a station query with parameter level=channel)
    :return: a new DataFrame with columns correctly renamed
    """
    if empty(query_df):
        return empty()

    Event, Station, Channel = models.Event, models.Station, models.Channel
    if query_type.lower() == "event" or query_type.lower() == "events":
        columns = Event.get_col_names()
    elif query_type.lower() == "station" or query_type.lower() == "stations":
        # these are the query_df columns for a station (level=station) query:
        #  #Network|Station|Latitude|Longitude|Elevation|SiteName|StartTime|EndTime
        # set this table columns mapping (by name, so we can safely add any new column at any
        # index):
        columns = [Station.network.key, Station.station.key, Station.latitude.key,
                   Station.longitude.key, Station.elevation.key, Station.site_name.key,
                   Station.start_time.key, Station.end_time.key]
    elif query_type.lower() == "channel" or query_type.lower() == "channels":
        # these are the query_df expected columns for a station (level=channel) query:
        #  #Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|
        #  SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
        # Some of them are for the Channel table, so select them:
        columns = [Station.network.key, Station.station.key, Channel.location.key,
                   Channel.channel.key, Station.latitude.key, Station.longitude.key,
                   Station.elevation.key, Channel.depth.key,
                   Channel.azimuth.key, Channel.dip.key, Channel.sensor_description.key,
                   Channel.scale.key, Channel.scale_freq.key, Channel.scale_units.key,
                   Channel.sample_rate.key, Station.start_time.key, Station.end_time.key]
    else:
        raise ValueError("Invalid fdsn_model: supply Events, Station or Channel class")

    oldcolumns = query_df.columns.tolist()
    if len(oldcolumns) != len(columns):
        raise ValueError("Mismatching number of columns in '%s' query.\nExpected:\n%s\nFound:\n%s" %
                         (query_type.lower(), str(oldcolumns), str(columns)))

    return query_df.rename(columns={cold: cnew for cold, cnew in zip(oldcolumns, columns)})


def harmonize_fdsn_dframe(query_df, query_type):
    """harmonizes the query dataframe (convert to dataframe dtypes, removes NaNs etcetera) according
    to query_type
    :param query_df: a query dataframe *on which `rename_columns` has already been called*
    :return: a new dataframe with only the good values
    """
    if empty(query_df):
        return empty()

    if query_type.lower() in ("event", "events"):
        fdsn_model_classes = [models.Event]
    elif query_type.lower() in ("station", "stations"):
        fdsn_model_classes = [models.Station]
    elif query_type.lower() in ("channel", "channels"):
        fdsn_model_classes = [models.Station, models.Channel]

    # convert columns to correct dtypes (datetime, numeric etcetera). Values not conforming
    # will be set to NaN or NaT or None, thus detectable via pandas.dropna or pandas.isnull
    for fdsn_model_class in fdsn_model_classes:
        query_df = harmonize_columns(fdsn_model_class, query_df)
        # we might have NA values (NaNs) after harmonize_columns, now
        # drop the rows with NA rows (NA for columns which are non-nullable):
        query_df = harmonize_rows(fdsn_model_class, query_df)

    return query_df


def empty(*obj):
    if not len(obj):
        return pd.DataFrame()  # this allows us to call len(empty()) without errors
    elif len(obj) > 1:
        raise ValueError("empty can be called with a single argument")
    obj = obj[0]
    return obj is None or (hasattr(obj, 'empty') and obj.empty)


def get_datacenters(session, logger, start_time, end_time):
    """Queries all datacenters and returns the local db model rows correctly added
    Rows already existing (comparing by datacenter station_query_url) are returned as well,
    but not added again
    """
    dcs_query = ('http://geofon.gfz-potsdam.de/eidaws/routing/1/query?service=station&'
                 'start=%s&end=%s&format=post' % (start_time.isoformat(), end_time.isoformat()))
    dc_result = url_read(dcs_query, decode='utf8')

    # add to db the datacenters read. Two little hacks:
    # 1) parse dc_result string and assume any new line starting with http:// is a valid station
    # query url
    # 2) When adding the datacenter, the table column dataselect_query_url (when not provided, as
    # in this case) is assumed to be the same as station_query_url by replacing "/station" with
    # "/dataselect"

    datacenters = [models.DataCenter(station_query_url=dcen) for dcen in dc_result.split("\n")
                   if dcen[:7] == "http://"]

    for dcen, _ in get_or_add_iter(session, datacenters, [models.DataCenter.station_query_url],
                                   on_add='commit'):
        pass
    # do not return only new datacenters, return all of them
    return session.query(models.DataCenter).all()


def get_segments_df(session, dcen, evt, channels, max_radius, min_sample_rate,
                    ptimespan, distances_cache_dict,
                    arrivaltimes_cache_dict, logger=None):
    """
    FIXME: write doc
    Downloads stations and channels, saves them , returns a well formatted pd.DataFrame
    with the segments ready to be downloaded
    """

    stations_df = get_stations_df(dcen.station_query_url, channels,
                                  evt.time, evt.latitude,
                                  evt.longitude, max_radius, min_sample_rate, logger=logger)

    if empty(stations_df):
        return empty(), 0

    stations_df, channel_ids = save_stations_df(session, stations_df, logger)

    if empty(stations_df):
        return empty(), 0

    segments_df = calculate_times(stations_df, evt, ptimespan, distances_cache_dict,
                                  arrivaltimes_cache_dict)

    segments_df[models.Segment.channel_id.key] = channel_ids
    segments_df[models.Segment.event_id.key] = evt.id

    notyet_downloaded_filter =\
        [False if session.query(models.Segment).
         filter((models.Segment.channel_id == seg.channel_id) &
                (models.Segment.start_time == seg.start_time) &
                (models.Segment.end_time == seg.end_time)).first() else True
         for _, seg in segments_df.iterrows()]

    segments_df_newitems = segments_df[notyet_downloaded_filter]
    skipped_already_downloaded = len(segments_df)-len(segments_df_newitems)
    segments_df = segments_df_newitems

    stations_df = stations_df[notyet_downloaded_filter]

    # build a query url:
    query_urls = [get_wav_query(dcen.dataselect_query_url,
                                sta[models.Station.network.key],
                                sta[models.Station.station.key],
                                sta[models.Channel.location.key],
                                sta[models.Channel.channel.key],
                                seg[models.Segment.start_time.key],
                                seg[models.Segment.end_time.key])
                  for (_, sta), (_, seg) in zip(stations_df.iterrows(),
                                                segments_df.iterrows())]

    # set index as urls. Do not use reindex cause it will set everything to NaN
    # (considering new index values as missing). This works:
    segments_df = pd.DataFrame(data=segments_df.values, columns=segments_df.columns,
                               index=query_urls)
    return segments_df, skipped_already_downloaded


def appenddf(olddf, newdf):
    if empty(olddf):
        return newdf
    elif empty(newdf):
        return olddf
    else:
        return olddf.append(newdf, ignore_index=True)


def calculate_times(stations_df, evt, ptimespan, distances_cache_dict={}, times_cache_dict={}):
    event_distances_degrees = []
    arrival_times = []
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
        if arr_time is None:
            arr_time = get_arrival_time(*coordinates)
            times_cache_dict[coordinates] = arr_time
        arrival_times.append(arr_time)

    ret = pd.DataFrame({models.Segment.event_distance_deg.key: event_distances_degrees,
                        models.Segment.arrival_time.key: arrival_times})
    ret[models.Segment.start_time.key] = ret[models.Segment.arrival_time.key] - timedelta(minutes=ptimespan[0])
    ret[models.Segment.end_time.key] = ret[models.Segment.arrival_time.key] + timedelta(minutes=ptimespan[1])

    return ret


def download_data(session, dcen, segments_df, max_error_count, stats, bar):
    if not stats:
        stats = {}

    for key in ['skipped_server_error', 'skipped_other_reason', 'skipped_empty',
                'skipped_localdb_error', 'downloaded']:
        stats.setdefault(key, 0)

    segments_df[models.Segment.datacenter_id.key] = dcen.id
    segments_df[models.Segment.data.key] = None
    urls = segments_df.index.values

    def onsuccess(data, url, index):
        bar.update(1)
        segments_df.loc[url, models.Segment.data.key] = data  # avoid pandas SettingWithCopyWarning

    def onerror(exc, url, index):
        bar.update(1)
        stats['skipped_server_error'] += 1
        if stats['skipped_server_error'] >= max_error_count:
            return False

    # now download Data:
    try:
        read_async(urls, onsuccess, onerror)
    except:
        g = 9

    tmp_df = segments_df.dropna(subset=[models.Segment.data.key])
    stats['skipped_other_reason'] = len(segments_df) - len(tmp_df)
    segments_df = tmp_df
    # get empty data, then remove it:
    segments_df[models.Segment.data.key].replace('', np.nan, inplace=True)
    tmp_df = segments_df.dropna(subset=[models.Segment.data.key])
    stats['skipped_empty'] = len(segments_df) - len(tmp_df)
    segments_df = tmp_df

    remaining_to_update = (stats['skipped_other_reason'] - stats['skipped_server_error'])
    if remaining_to_update:
        bar.update(remaining_to_update)

    if not empty(segments_df):
        for model_instance in df2dbiter(segments_df, models.Segment, False, False):
            session.add(model_instance)
            if commit(session):
                stats['downloaded'] += 1
            else:
                stats['skipped_localdb_error'] += 1

    return stats


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

    STEPS = 4

    # write the class labels:
    for _, _ in get_or_add_iter(session, df2dbiter(class_labels_df,
                                                   models.Class,
                                                   harmonize_columns_first=True,
                                                   harmonize_rows=True), on_add='commit'):
        pass

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
    logger.info("STEP 1/%d: Querying Event WS and datacenters" % STEPS)
    # Get events, store them in the db, returns the event instances (db rows) correctly added:
    events_df = get_events_df(logger=logger, **args)
    if empty(events_df):
        logger.error("No events found")
        return 1
    # convert dataframe to records (df_to_table_iterrows),
    # add non existing records to db (get_or_add_all) comparing by events.id
    # return the added rows
    # Note: get_or_add_all has already flushed, so the returned model instances (db rows)
    # have the fields updated, if any
    events = [inst for inst, _ in get_or_add_iter(session, df2dbiter(events_df, models.Event),
                                                  on_add='commit') if inst is not None]

    if not events:
        logger.error("0 events found")
        return 1

    # Get datacenters, store them in the db, returns the dc instances (db rows) correctly added:
    datacenters = get_datacenters(session, logger, start, end)

    if not datacenters:
        logger.error("0 datacenters found")
        return 1

    logger.debug("")
    logger.info(("STEP 2/%d: Querying Station WS (level=channel) from %d datacenter(s) "
                 "nearby %d event(s) found, calculating P-arrival times and time ranges "
                 "for downloading segments")
                % (STEPS, len(datacenters), len(events)))

    # calculate search radia:
    magnitudes = np.array([evt.magnitude for evt in events])
    max_radia = get_search_radius(magnitudes,  # FIXME: use *search_radius_args?
                                  search_radius_args[0],
                                  search_radius_args[1],
                                  search_radius_args[2],
                                  search_radius_args[3])

#     urls_to_dcindex = {}
#     for i, dcen in enumerate(datacenters):
#         for max_radius, evt in zip(max_radia, events):
#             url = get_station_query_url(dcen.station_query_url, channels,
#                                         evt.time, evt.latitude,
#                                         evt.longitude, max_radius)
#             urls_to_dcindex[url] = i
# 
#     dcenrawdata = [[] for _ in len(datacenters)]
# 
#     def onsuccess(data, url, index):
#         if data:
#             dcenrawdata[urls_to_dcindex[url]].append(data)
# 
#     def onerror(exc, url, index):
#         logger.warning("%s: %s" % (url, str(exc)))
# 
#     read_async(urls_to_dcindex.keys(),onsuccess, onerror, max_workers=5, decode='utf8')
    
    
    dcen_stats = pd.DataFrame(index=[dcen.dataselect_query_url for dcen in datacenters],
                              columns=['skipped_downloaded', 'skipped_empty',
                                       'skipped_server_error', 'skipped_other_reason',
                                       'skipped_localdb_error', 'downloaded'],
                              data=0)
    data = []
    distances_cache_dict = {}
    arrivaltimes_cache_dict = {}
    stats = {}
    with progressbar(length=len(events) * len(datacenters)) as bar:
        for dcen in datacenters:
            dcendata = None
            dcen_stats = {'skipped_already_downloaded': 0}
            stats[dcen.dataselect_query_url] = dcen_stats
            for max_radius, evt in zip(max_radia, events):
                bar.update(1)
                segments_df, skipped = \
                    get_segments_df(session, dcen, evt, channels, max_radius, min_sample_rate,
                                    ptimespan, distances_cache_dict, arrivaltimes_cache_dict,
                                    logger)
                dcen_stats['skipped_already_downloaded'] += skipped
                dcendata = appenddf(dcendata, segments_df)
            data.append(dcendata)

    segments_count = sum([0 if empty(d) else len(d) for d in data])
    logger.debug("")
    logger.info("STEP 3/%d: Querying Datacenter WS for %d segments" % (STEPS, segments_count))

    max_error_count = 5
    dcen_stats_series = []
    with progressbar(length=segments_count) as bar:
        for dcen, segments_df in zip(datacenters, data):
            if empty(segments_df):
                continue
            segments_df[models.Segment.run_id.key] = run_id
            download_data(session, dcen, segments_df, max_error_count,
                          stats[dcen.dataselect_query_url], bar)
            dcen_stats_series.append(pd.Series(name=dcen.dataselect_query_url,
                                               data=stats[dcen.dataselect_query_url]))

    logger.info("Done. Summary info per datacenter:")
    stats_info = pd.DataFrame(dcen_stats_series)
    stats_info = stats_info.append(pd.Series(name='total',
                                             data=stats_info.apply(pd.to_numeric,
                                                                   errors='coerce', axis=0).sum()))
    logger.info(stats_info.to_string(col_space=1, justify='right'))
    logger.info("")
    return 0
