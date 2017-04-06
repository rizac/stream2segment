'''
Created on Nov 25, 2016

@author: riccardo
'''
import re
from datetime import timedelta
# from collections import defaultdict
import numpy as np
import pandas as pd
# from sqlalchemy import and_
from obspy.taup import TauPyModel
# from obspy.geodetics import locations2degrees
# from obspy.taup.helper_classes import TauModelError, SlownessModelError
# from stream2segment.utils.url import url_read
from stream2segment.io.db import models
from stream2segment.io.db.pd_sql_utils import harmonize_columns,\
    harmonize_rows, colnames
from obspy.taup.taup_time import TauPTime
from obspy.geodetics.base import locations2degrees
from itertools import izip, count
from stream2segment.utils.resources import version
from stream2segment.utils.url import urlread
from stream2segment.io.utils import dumps_inv, loads_inv
from sqlalchemy.orm.session import object_session
# from stream2segment.io.utils import dumps_inv


def run_instance(session=None, **args):
    """Same as models.Run() but sets the "version" field column (unless specified in `args`).
    It also add it to the session
    if the argument session is not None.
    We might want to implement it as attribute default in io.models but that requires importing
    other stuff inthere and we do not want it for the moment
    """
    if 'program_version' not in args:
        args['program_version'] = version()
    run_row = models.Run(**args)
    if session is not None:
        session.add(run_row)
        session.commit()
    return run_row


def get_min_travel_time(source_depth_in_km, distance_in_degree, traveltime_phases, model='ak135'):
    """
        Assess and return the minimum travel time of P phases.
        Uses obspy.getTravelTimes
        :param source_depth_in_km: (float) Depth in kilometer.
        :param distance_in_degree: (float) Distance in degrees.
        :param model: string (optional, default 'ak135') or model. Either an internal obspy TauPy
        model name, as string (see https://docs.obspy.org/packages/obspy.taup.html) or an instance
        of :ref:`obspy.taup.tau.TauPyModel
        :param traveltime_phases: a list of strings specifying the phases to calculate, e.g.
        ['P', 'pP'].See FIXME: update ref
        :return the number of seconds of the assessed arrival time, or None in case of error
        :raises: obspy.taup.helper_classes.TauModelError,
        obspy.taup.helper_classes.SlownessModelError,
        ValueError (if no travel time was found)
    """
    try:
        taupmodel = TauPyModel(model)
    except TypeError:
        taupmodel = model  # ok, we assume the argument is already a TaupModel then

    # see: https://docs.obspy.org/packages/obspy.taup.html
    receiver_depth_in_km = 0.0
    tt = TauPTime(taupmodel.model, traveltime_phases, source_depth_in_km,
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


def get_arrival_time(distance_in_degrees, ev_depth_km, ev_time, traveltime_phases, model='ak135'):
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
    travel_time = get_min_travel_time(ev_depth_km, distance_in_degrees, traveltime_phases, model)
    arrival_time = ev_time + timedelta(seconds=float(travel_time))
    return arrival_time


def get_search_radius(mag, minmag, maxmag, minradius, maxradius):
    """From a given magnitude, determines and returns the max radius (in degrees).
        Given minradius and maxradius and minmag and maxmag (FIXME: TO BE CALIBRATED!),
        this function returns D from the f below:

                  |
        maxradius +                oooooooooooo
                  |              o
                  |            o
                  |          o
        minradius + oooooooo
                  |
                  ---------+-------+------------
                        minmag     maxmag

    :return: the max radius (in degrees)
    :param mag: (numeric or list or numbers/numpy.array) the magnitude
    :param minmag: (int, float) the minimum magnitude
    :param maxmag: (int, float) the maximum magnitude
    :param minradius: (int, float) the minimum distance (in degrees)
    :param maxradius: (int, float) the maximum distance (in degrees)
    :return: a scalar if mag is scalar, or an numpy.array
    """
    mag = np.asarray(mag)  # do NOT copies data for existing arrays
    isscalar = not mag.shape
    if isscalar:
        mag = np.array(mag, ndmin=1)  # copies data, assures an array of dim=1

    if minmag == maxmag:
        dist = np.array(mag)
        dist[mag < minmag] = minradius
        dist[mag > minmag] = maxradius
        dist[mag == minmag] = np.true_divide(minradius+maxradius, 2)
    else:
        dist = minradius + np.true_divide(maxradius - minradius, maxmag - minmag) * (mag - minmag)
        dist[dist < minradius] = minradius
        dist[dist > maxradius] = maxradius

    return dist[0] if isscalar else dist


# ==========================================
def query2dframe(query_result_str, strip_cells=True):
    """
        Returns a pandas dataframe from the given query_result_str
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
        columns = list(colnames(Event))
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


# def appenddf(df1, df2):
#     """
#     Merges "vertically" the two dataframes provided as argument, handling empty values without
#     errors: if the first dataframe is empty (`empty(df1)==True`) returns the second, if the
#     second is empty returns the first. Otherwise calls `df1.append(df2, ignore_index=True)`
#     :param df1: the first dataframe
#     :param df2: the second dataframe
#     """
#     if empty(df1):
#         return df2
#     elif empty(df2):
#         return df1
#     else:
#         return df1.append(df2, ignore_index=True)


def urljoin(*urlpath, **query_args):
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
    >>> urljoin("http://www.domain", start='2015-01-01T00:05:00', mag=5.455559, arg=True)
    'http://www.domain?start=2015-01-01T00:05:00&mag=5.455559&arg=True'

    >>> urljoin("http://www.domain", "data", start='2015-01-01T00:05:00', mag=5.455559, arg=True)
    'http://www.domain/data?start=2015-01-01T00:05:00&mag=5.455559&arg=True'

    # Note how slashes are handled in urlpath. These two examples give the same url path:

    >>> urljoin("http://www.domain", "data")
    'http://www.domain/data?'

    >>> urljoin("http://www.domain/", "/data")
    'http://www.domain/data?'

    # leading and trailing slashes on each element of urlpath are removed:

    >>> urljoin("/www.domain/", "/data")
    'www.domain/data?'

    # so if you want to preserve them, provide an empty argument or a slash:

    >>> urljoin("", "/www.domain/", "/data")
    '/www.domain/data?'

    >>> urljoin("/", "/www.domain/", "/data")
    '/www.domain/data?'
    ```
    """
    # http://stackoverflow.com/questions/1793261/how-to-join-components-of-a-path-when-you-are-constructing-a-url-in-python
    return "{}?{}".format('/'.join(url.strip('/') for url in urlpath),
                          "&".join("{}={}".format(k, v) for k, v in query_args.iteritems()))


def get_inventory(station, save_if_downloaded=False, **urlread_kwargs):
    """Gets the inventory object for the given station, downloading it and saving it
    if not data is empty/None.
    Raises SqlAlchemyError or TypeError if station's session is None, ValueError if inventory data
    is empty
    """
    data = station.inventory_xml
    if not data:
        data = download_inventory(station, **urlread_kwargs)
        if save_if_downloaded and data:
            save_inventory(data, station)

    if not data:
        raise ValueError()

    return loads_inv(data)


def download_inventory(station, **urlread_kwargs):
    query_url = get_inventory_url(station)
    return urlread(query_url, **urlread_kwargs)


def get_inventory_url(station):
    return get_inventory_url_(station.datacenter.station_query_url, station.network,
                              station.station)


def get_inventory_url_(station_query_url, network, station):
    return urljoin(station_query_url, station=station, network=network,
                     level='response')


def save_inventory(downloaded_data, station):
    """Saves the inventory. Raises SqlAlchemyError or TypeError if station's session is None
    """
    session = object_session(station)
    if session is None:
        raise TypeError()
    station.inventory_xml = dumps_inv(downloaded_data)
    session.commit()
    return


def calculate_times(sta_lat, sta_lon, evt_lat, evt_lon, evt_depth_km, evt_time,
                    traveltime_phases, taup_model='ak135'):
    taupmodel_obj = TauPyModel(taup_model)  # create the taupmodel once
    # old comment (REMOVE not used here anymore):
    # iteration over dframe columns is faster than DataFrame.itertuples
    # and is more readable as we only need a bunch of columns.
    # Note: we zip using dataframe[columname] iterables. Using
    # dataframe[columname].values (underlying pandas numpy array) is even faster,
    # BUT IT DOES NOT RETURN pd.TimeStamp objects for date-time-like columns but np.datetim64
    # instead. As the former subclasses python datetime (so it's sqlalchemy compatible) and the
    # latter does not, we go for the latter ONLY BECAUSE WE DO NOT HAVE DATETIME LIKE OBJECTS:
#     for sta_id, stalat, stalon in izip(stations_df[Channel.station_id.key].values,
#                                        stations_df[Station.latitude.key].values,
#                                        stations_df[Station.longitude.key].values):

    # stalat, stalon = getattr(sta, latstr), getattr(sta, lonstr)
    degrees = locations2degrees(evt_lat, evt_lon, sta_lat, sta_lon)
    arr_time = get_arrival_time(degrees, evt_depth_km, evt_time, traveltime_phases,
                                taupmodel_obj)
    return degrees, arr_time


class UrlStats(dict):
    """A subclass of dict to store keys (usually messages, i.e. strings) mapped to their occurrence
    (integers).
    When getting a particular key, an instance of this class returns 0 if the key is not found (as
    `colelctions.defaultdict` does) without raising any exception. When setting or getting a key,
    `Exception`s are first converted to the string format:
    ```
    exception.__class__.__name__+ ": " + str(exception)
    ```
    and then handled normally. The name stems from the fact that this class is used to pass either
    message strings or URL/HTTP/connection errors when querying data.
    Example:
    ```
        s = UrlStats()
        print s['a']
        >>> 0
        s['a'] += 3
        print s
        >>> {'a' : 3}
        s[Exception('a')] = 5
        print s[Exception('a')]
        >>> 5
        print s['Exception: a']
        >>> 5
    ```
    """
    @staticmethod
    def re(value):
        return re.compile("(?<!\\w)%s(?!\\w)" % re.escape(str(value)))

    @staticmethod
    def convert(key):
        if isinstance(key, Exception):
            exc_msg = str(key)
            # be sure to print the 'reason' or 'code' attribute (if any, and if they are
            # not already in exc_msg:
            if hasattr(key, 'code') and not UrlStats.re(key.code).search(exc_msg):
                # code not already in string (it should be the case in general), add it:
                exc_msg += " [code=%s]" % str(key.code)
            elif hasattr(key, 'reason') and not UrlStats.re(key.reason).search(exc_msg):
                # reason not already in string, add it:
                exc_msg += " [reason=%s]" % str(key.reason)
            if not exc_msg.startswith("HTTP Error") or ":" not in exc_msg:
                # HTTPError usually starts with "HTTP Error", so in that case skip this part:
                key = "%s: %s" % (key.__class__.__name__, exc_msg)
            else:
                key = exc_msg
        return key

    def __setitem__(self, key, value):
        super(UrlStats, self).__setitem__(UrlStats.convert(key), value)

    def __missing__(self, key):
        key = UrlStats.convert(key)
        return self.get(key, 0)


def stats2str(data, fillna=None, transpose=False,
              lambdarow=None, lambdacol=None, sort=None,
              totals='all', totals_caption='TOTAL', *args, **kwargs):
    """
        Returns a string representation of `data` nicely formatted in a table. The argument `data`
        is any valid object which can be passed to a pandas.DataFrame as 'data' argument. In the
        most simple and typical case, it is a dict of string keys K mapped to dicts (or pandas
        Series) D: {..., K: D, ....}.
        **Each D will be represented in a column with its key k as column header** (unless transpose
        is True, see below)
        :Example:
        ```
        stats2str(data={
                        'col1': {'a': 9, 'b': 3},
                        'col2': pd.Series({'a': -1, 'c': 0})
                       })
        # result (note missing values filled with zero and totals calculated by default):
                col1  col2  total
        a         9    -1      8
        b         3     0      3
        c         0     0      0
        total    12    -1     11
        :param data: numpy ndarray (structured or homogeneous), dict, or DataFrame
        Dict can contain Series, arrays, constants, or list-like objects. The data to display
        If dict containing sub-dicts D, each D will be
        displayed in columns, and `data.keys()` K will be the column names. In this case, the union
        U of each D key will populate the row headers. If transpose=True, the table/DataFrame will
        be transposed, thus columns become rows, and viceversa (see below). The cell table value
        of a key found in a certain D and missing in another D will be displayed as nan in the
        latter and can be customized with `fillna` (see below)
        :param transpose: if False (the default) nothing happens. If true, before any further
        processing the table/DataFrame will be transposed, thus columns become rows, and viceversa
        :param fillna: (any object, default: None) the value used to fill missing values. This
        method uses internally a pandas DataFrame which will convert the input data into a table,
        filling by default missing values with NaN's. These NaN's can be converted to a suitable
        value specified here (e.g., if the table specifies occurrences of the given keys, then a key
        not found should be 0, and thus fillna=0). If this value is an int (or numpy int),
        then a further attempt is made to **convert all table data to int**.
        The same holds if this value is a float (or numpy float), then the data will be converted
        (if possible) to float (this might result in NaN values which will NOT be converted again
        according to `fillna`). In any other case, no conversion is made
        :param lambdarow: (callable or None. Default:None) Function to customize row header(s).
        If None, this argument is ignored. Otherwise it is a callable which accepts a single
        argument (each row, usually string) and should return the new value to be displayed
        :param lambdacol: (callable or None. Default:None) Function to customize column header(s).
        If None, this argument is ignored. Otherwise it is a callable which accepts a single
        argument (each column, usually string) and should return the new value to be displayed.
        **Note**: contrarily to rows, if this function returns a *different* value than the
        original column, a note is added to the new value and legend will be displayed after the
        table. The legend will display the note and the full original column value
        :param sort: string ('col', 'row', 'all' or None) Sorts data headers before displaying.
        If 'row' or 'all', sorts rows ascending. If 'col' or 'all', sorts column ascending. If any
        other value, this argument is ignored
        :param totals: string ('col', 'row', 'all' or None. Default: 'all'). Display totals
        (sum along table axis). If 'row' or 'all', display totals for each row (adding a further
        column with header `totals_caption`. If 'col' or 'all', display totals for each column
        (adding a further row with header `totals_caption`). If `sort` is enabled, this does not
        affect the totals (which will be always displayed as last row or column). This argument
        forces data numeric conversion, thus NaN's might appear and must be handled beforehand by
        the user.
        :param totals_caption: string, default: 'TOTAL'. Caption for the row or column displaying
        totals. Ignored if `totals` is not in ('row', 'col' or 'all')
        :param args: additional position arguments pased to `pandas.DataFrame.to_string`
        :param kwargs: additional keyword arguments pased to `pandas.DataFrame.to_string`
    """
    dframe = pd.DataFrame(data=data)
    if dframe.empty:
        return ""
    # replace NaNs (for columns not shared) with zeros, and convert to int
    if fillna is not None:
        dframe.fillna(fillna, inplace=True)
        if type(fillna) in (int, np.int8, np.int16, np.int32, np.int64):
            try:
                dframe = dframe.astype(int)
            except ValueError:  # we do not have NaN's, but what if we have other kind of data? skip
                pass
        elif type(fillna) in (float, np.float16, np.float32, np.float64, np.float128):
            try:
                dframe = dframe.astype(float)
            except ValueError:  # we do not have NaN's, but what if we have other kind of data? skip
                pass

    if transpose:
        dframe = dframe.T

    if hasattr(lambdarow, "__call__"):
        dframe.index = dframe.index.map(lambdarow)

    columndetails = []
    if hasattr(lambdacol, "__call__"):
        new_columns = dframe.columns.map(lambdacol)
        counter = 1
        for new, old, i in izip(new_columns, dframe.columns, count()):
            if old != new:
                columndetails.append("[%d] %s" % (counter, old))
                new_columns[i] = "%s[%d]" % (new, counter)
                counter += 1
        if columndetails:
            dframe.columns = new_columns

    if sort in ('col', 'all'):
        dframe.sort_index(axis=1, inplace=True)
    if sort in ('row', 'all'):
        dframe.sort_index(axis=0, inplace=True)

    if totals in ('row', 'all', 'col'):
        # convert to numeric so that sum returns the correct number of rows/columns
        # (with NaNs in case)
        dframe = dframe.apply(pd.to_numeric, errors='coerce', axis=0)  # axis should be irrelevant
        if totals in ('row', 'all'):
            # append a row with sum:
            dframe.loc[totals_caption] = dframe.sum(axis=0)
        if totals in ('col', 'all'):
            # append a column with sums:
            dframe[totals_caption] = dframe.sum(axis=1)

    ret = dframe.to_string(*args, **kwargs)

    if columndetails:
        legendtitle = "Detailed column headers:"
        ret = "%s\n%s\n%s\n%s" % (ret, "-"*len(legendtitle), legendtitle, "\n".join(columndetails))

    return ret

#     with pd.option_context('display.max_rows', len(dframe),
#                            'display.max_columns', len(dframe.columns),
#                            'max_colwidth', 50, 'expand_frame_repr', False):
#         return str(dframe)


def dfupdate(df_old, df_new, matching_columns, set_columns, ondupes=None):
    """
        Kind-of pandas.DataFrame update: sets
        `df_old[set_columns]` = `df_new[set_columns]`
        for those row where `df_old[matching_columns]` = `df_new[matching_columns]` only.
        `df_new` **should** have unique rows under `matching columns` (see argument `ondupes`)
        :param df_old: the pandas DataFrame whose values should be replaced
        :param df_new: the pandas DataFrame which should set the new values to `df_old`
        :param matching_columns: list of strings: the columns to be checked for matches. They must
        be shared between both data frames
        :param set_columns: list of strings denoting the column to be set from `df_new` to
        `df_old` for those rows matching under `matching_cols`
        :param ondupes: as `df_new` should not have duplicated rows for any `matching_columns`, this
        argument tells what to do: None (the default) does not do any check, it might be faster but
        you should issue a ```df_new.drop_duplicates(subset=matching_columns, ...)``` if
        any duplicate is present in `df_new`. 'raise' and `ignore` do a check inside the function:
        the former raises `ValueError` (with a memaingful message) on duplicates, the latter simply
        ignores duplicates returning `df_old` unmodified if any duplicate is present
    """
    if df_new.empty or df_old.empty:  # for safety (avoid useless calculations)
        return df_old

    if ondupes == 'ignore' or ondupes == 'raise':
        if df_new[matching_columns].duplicated().any():
            if ondupes == 'raise':
                raise ValueError("dataframe has duplicated values under %s" %
                                 str(matching_columns))
            else:
                return df_old
    # use df_new[matching_columns + set_columns] only for relevant columns
    # (should speed up merging?):
    mergedf = df_old.merge(df_new[matching_columns + set_columns], how='left',
                           on=list(matching_columns), indicator=True)

    # set values of new_df by means of the _merge column created via the arg indicator=True above:
    # _merge is in ('both', 'right_only', 'left_only'). We should never have 'right_only because of
    # the how='left' above. Skip checking for the moment
    for col in set_columns:
        ser = np.where(mergedf['_merge'] == 'both', mergedf[col+"_y"], mergedf[col+"_x"])
        # ser = mergedf[col+"_y"].where(mergedf['_merge'] == 'both', mergedf[col+"_x"])
        df_old[col] = ser

    return df_old

