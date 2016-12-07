'''
Created on Nov 25, 2016

@author: riccardo
'''
from datetime import timedelta
import numpy as np
import pandas as pd
from sqlalchemy import and_
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees
from obspy.taup.helper_classes import TauModelError
from stream2segment.utils.url import url_read
from stream2segment.io.db import models
from stream2segment.io.db.pd_sql_utils import harmonize_columns,\
    harmonize_rows, df2dbiter, get_or_add_iter
from obspy.taup.taup_time import TauPTime


def get_min_travel_time(source_depth_in_km, distance_in_degree, model='ak135'):
    """
        Assess and return the minimum travel time of P phases.
        Uses obspy.getTravelTimes
        :param source_depth_in_km: (float) Depth in kilometer.
        :param distance_in_degree: (float) Distance in degrees.
        :param model: string (optional, default 'ak135') the model. Either an internal obspy TauPy
        model name, as string (see https://docs.obspy.org/packages/obspy.taup.html) or an instance
        of :ref:`obspy.taup.tau.TauPyModel
        :return the number of seconds of the assessed arrival time, or None in case of error
        :raises: TauModelError, ValueError (if no travel time was found)
    """
    taupmodel = model if hasattr(model, 'get_travel_times') else TauPyModel(model)

    phase_list = ('P', 'p', 'pP', 'PP')  # , 'pp')
    # ("ttall",)  # FIXME: check if we can restrict!
    # see:
    # https://docs.obspy.org/packages/obspy.taup.html
    receiver_depth_in_km = 0.0
    tt = TauPTime(taupmodel.model, phase_list, source_depth_in_km,
                  distance_in_degree, receiver_depth_in_km)
    tt.run()
    # now instead of doing this (excpensive):
    # return Arrivals(sorted(tt.arrivals, key=lambda x: x.time), model=self.model)
    # we just might just in place and return the time of the first element
    # but it's faster to specify a phase list above (which speeds up calculations)
    # and check for min
    if tt.arrivals:
        return min(tt.arrivals, key=lambda x: x.time).time
    else:
        raise ValueError("No travel times found")


# def _get_min_travel_time(source_depth_in_km, distance_in_degree, model='ak135'):
#     """
#         Assess and return the travel time of P phases.
#         Uses obspy.getTravelTimes
#         :param source_depth_in_km: Depth in kilometer.
#         :type source_depth_in_km: float
#         :param distance_in_degree: Distance in degrees.
#         :type distance_in_degree: float
#         :param model: Either ``'iasp91'`` or ``'ak135'`` velocity model.
#          Defaults to 'ak135'.
#         :type model: str, optional
#         :return the number of seconds of the assessed arrival time, or None in case of error
#         :raises: TauModelError, ValueError (if travel times is empty)
#     """
#     taupmodel = TauPyModel(model)
#     try:
#         tt = taupmodel.get_travel_times(source_depth_in_km, distance_in_degree)
#         # return min((ele['time'] for ele in tt if (ele.get('phase_name') or ' ')[0] == 'P'))
# 
#         # Arrivals are returned already sorted by time!
#         return tt[0].time
# 
#         # return min(tt, key=lambda x: x.time).time
#         # return min((ele.time for ele in tt))
#     except (TauModelError, ValueError, AttributeError) as err:
#         raise ValueError(("Unable to find minimum travel time (dist=%s, depth=%s, model=%s). "
#                           "Source error: %s: %s"),
#                          str(distance_in_degree), str(source_depth_in_km), str(model),
#                          err.__class__.__name__, str(err))


def get_arrival_time(distance_in_degrees, ev_depth_km, ev_time, model='ak135'):
    """
        Returns the arrival time by calculating the minimum travel time of p-waves
        :param distance_in_degrees: (float) the distance in degrees between station and event
        :param ev_depth_km: (numeric) the event depth in km
        :param ev_time: (datetime) the event time
        :param model: string (optional, default 'ak135') the model. Either an internal obspy TauPy
        model name, as string (see https://docs.obspy.org/packages/obspy.taup.html) or an instance
        of :ref:`obspy.taup.tau.TauPyModel
        :return: the P-wave arrival time
    """
    travel_time = get_min_travel_time(ev_depth_km, distance_in_degrees, model)
    arrival_time = ev_time + timedelta(seconds=float(travel_time))
    return arrival_time


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


def get_max_radia(events, *search_radius_args):
    """Returns the max radia for any event in events
    :param events: an iterable of objects with the attribute `time` (numeric). ORM model instances
    of the class `Event` are such objects (see `models.py`)
    """
    magnitudes = np.array([evt.magnitude for evt in events])
    return get_search_radius(magnitudes, *search_radius_args)


# ==========================================
def query2dframe(query_result_str, strip_cells=True):
    """
        Returns a pandas dataframne from the given query_result_str
        :param: query_result_str
        :raise: ValueError in case of errors (mismatching row lengths), including the case
        where the resulting dataframe is empty. Note that query_result_str evaluates to False, then
        `empty()` is returned without raising
    """
    if not query_result_str:
        return empty()
    events = query_result_str.splitlines()
    data = []
    columns = None
    colnum = 0
    # parse text into dataframe. Note that we check the row lengths beforehand cause pandas fills
    # with trailing NaNs which we don't want to handle. E.g.:
    # >>> pd.DataFrame(data=[[1,2], [3,4,5]], columns=['a', 'b', 'g'])
    #    a  b    g
    # 0  1  2  NaN
    # 1  3  4  5.0
    # We use simple list append and not np.append cause np string arrays having fixed lengths
    # sometimes cut strings. np.append(arr1, arr2) seems to handle this, but let's be safe
    for i, evt in enumerate(events):
        evt_list = evt.split('|') if not strip_cells else [e.strip() for e in evt.split("|")]
        if i == 0:
            columns = evt_list
            colnum = len(columns)
        elif len(evt_list) != colnum:
            raise ValueError("Column length mismatch while parsing query result")
        else:
            data.append(evt_list)

    if not data or not columns:
        raise ValueError("Data empty after parsing query result (malformed data)")
    return pd.DataFrame(data=data, columns=columns)


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
        return query_df

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
    :param query_type: either 'event', 'channel', 'station'
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


def normalize_fdsn_dframe(dframe, query_type):
    """Calls `rename_columns` and `harmonize_fdsn_dframe`. The returned
    dataframe has the first N columns with normalized names according to `query` and correct data
    types (rows with unparsable values, e.g. NaNs, are removed). The data frame is ready to be
    saved to the internal database
    :param query_df: the dataframe resulting from the string url `query`
    :param query_type: either 'event', 'channel', 'station'
    :return: a new dataframe, whose length is <= `len(dframe)`
    :raise: ValueError in case of errors (e.g., mismatching column number, or returning
    dataframe is empty, e.g. **all** rows have at least one invalid value: in fact, note that
    invalid cell numbers are removed (their row is removed from the returned data frame)
    """
    dframe = rename_columns(dframe, query_type)
    ret = harmonize_fdsn_dframe(dframe, query_type)
    if empty(ret):
        raise ValueError("Malformed data (invalid values, e.g., NaN's)")
    return ret

_EMPTY = pd.DataFrame()


def empty(*obj):
    """
    Utility function to handle "no-data" dataframes in this module function by providing a
    general check and generation of empty objects.
    Returns True or False if the argument is "empty" (i.e. if obj is None or obj has attribute
    'empty' and `obj.empty` is True). With a single argument, returns an object `obj` which
    evaluates to empty, i.e. for which `empty(obj)` is True (currently, an empty DataFrame, but it
    might be any value for which empty(obj) is True. We prefer a DataFrame over `None` so that
    len(empty()) does not raise Exceptions and correctly returns 0).
    """
    if not len(obj):
        return _EMPTY  # this allows us to call len(empty()) without errors
    elif len(obj) > 1:
        return [empty(o) for o in obj]
    obj = obj[0]
    return obj is None or (hasattr(obj, 'empty') and obj.empty)


def appenddf(df1, df2):
    """
    Merges "vertically" the two dataframes provided as argument, handling empty values without
    errors: if the first dataframe is empty (`empty(df1)==True`) returns the second, if the
    second is empty returns the first. Otherwise calls `df1.append(df2, ignore_index=True)`
    :param df1: the first dataframe
    :param df2: the second dataframe
    """
    if empty(df1):
        return df2
    elif empty(df2):
        return df1
    else:
        return df1.append(df2, ignore_index=True)


def get_query(*urlpath, **query_args):
    """Joins urls and appends to it the query string obtained by kwargs
    Note that this function is intended to be simple and fast: No check is made about white-spaces
    in strings, no encoding is done, and if some value of `query_args` needs special formatting
    (e.g., "%1.1f"), that needs to be done before calling this function
    :param urls: portion of urls which will build the query url Q. For more complex url functions
    see `urlparse` library: this function builds the url path via a simple join stripping slashes:
    ```'/'.join(url.strip('/') for url in urlpath)```
    So to preserve slashes (e.g., at the beginning) pass "/" or "" as arguments (e.g. as first
    argument to preserve relative paths).
    :query_args: keyword arguments which will build the query string
    :return: a query url built from arguments

    :Example:
    ```
    >>> get_query("http://www.domain", start='2015-01-01T00:05:00', mag=5.455559, arg=True)
    'http://www.domain?start=2015-01-01T00:05:00&mag=5.455559&arg=True'

    >>> get_query("http://www.domain", "data", start='2015-01-01T00:05:00', mag=5.455559, arg=True)
    'http://www.domain/data?start=2015-01-01T00:05:00&mag=5.455559&arg=True'

    # Note how slashes are handled in urlpath. These two examples give the same url path:

    >>> get_query("http://www.domain", "data")
    'http://www.domain/data?'

    >>> get_query("http://www.domain/", "/data")
    'http://www.domain/data?'

    # leading and trailing slashes on each element of urlpath are removed:

    >>> get_query("/www.domain/", "/data")
    'www.domain/data?'

    # so if you want to preserve them, provide an empty argument or a slash:

    >>> get_query("", "/www.domain/", "/data")
    '/www.domain/data?'

    >>> get_query("/", "/www.domain/", "/data")
    '/www.domain/data?'
    ```
    """
    # http://stackoverflow.com/questions/1793261/how-to-join-components-of-a-path-when-you-are-constructing-a-url-in-python
    return "{}?{}".format('/'.join(url.strip('/') for url in urlpath),
                          "&".join("{}={}".format(k, v) for k, v in query_args.iteritems()))


def save_stations_df(session, stations_df):
    """
        stations_df is already harmonized. If saved, it is appended a column 
        `models.Channel.station_id.key` with nonNull values
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
    channels_df[models.Channel.id.key] = cha_ids
    channels_df.dropna(subset=[models.Channel.id.key], inplace=True)
    channels_df.reset_index(drop=True, inplace=True)  # to be safe
    return channels_df


# def get_segments_df(session, stations_df, evt, ptimespan,
#                     distances_cache_dict, arrivaltimes_cache_dict):
#     """
#     FIXME: write doc
#     stations_df must have a column named `models.Channel.id.key`
#     Downloads stations and channels, saves them , returns a well formatted pd.DataFrame
#     with the segments ready to be downloaded
#     """
# 
#     segments_df = calculate_times(stations_df, evt, ptimespan, distances_cache_dict,
#                                   arrivaltimes_cache_dict, session=session)
# 
#     segments_df[models.Segment.channel_id.key] = stations_df[models.Channel.id.key]
#     segments_df[models.Segment.event_id.key] = evt.id
#     return segments_df


def purge_already_downloaded(session, segments_df):  # FIXME: use apply?
    """Does what the name says removing all segments aready downloaded. Returns a new DataFrame
    which is equal to segments_df with rows, representing already downloaded segments, removed"""
    notyet_downloaded_filter =\
        [False if session.query(models.Segment).
         filter((models.Segment.channel_id == seg.channel_id) &
                (models.Segment.start_time == seg.start_time) &
                (models.Segment.end_time == seg.end_time)).first() else True
         for _, seg in segments_df.iterrows()]

    return segments_df[notyet_downloaded_filter]


def set_wav_queries(datacenter, stations_df, segments_df, queries_colname=' url '):
    """
    Appends a new column to `stations_df` with name `queries_colname` (which is supposed **not**
    to exist, otherwise data might be overridden or unexpected results might happen): the given
    column will have the datacenter query url for any given row representing a segment to be
    downloaded. The given dataframe must have all necessary columns
    """

    queries = [get_query(datacenter.dataselect_query_url,
                         network=sta[models.Station.network.key],
                         station=sta[models.Station.station.key],
                         location=sta[models.Channel.location.key],
                         channel=sta[models.Channel.channel.key],
                         start=seg[models.Segment.start_time.key].isoformat(),
                         end=seg[models.Segment.end_time.key].isoformat())
               for (_, sta), (_, seg) in zip(stations_df.iterrows(), segments_df.iterrows())]

    segments_df[queries_colname] = queries
    return segments_df
