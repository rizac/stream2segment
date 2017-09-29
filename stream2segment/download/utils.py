'''
Utilities for the download package

:date: Nov 25, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
from __future__ import division
# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import zip

import re
from datetime import timedelta, datetime
import dateutil
from itertools import count

import numpy as np
import pandas as pd
from sqlalchemy.orm.session import object_session

from stream2segment.io.db.models import Event, Station, Channel
from stream2segment.io.db.pd_sql_utils import harmonize_columns,\
    harmonize_rows, colnames
from stream2segment.utils.url import urlread, URLException
from stream2segment.utils import urljoin


def get_events_list(eventws, **args):
    """Returns a list of tuples (raw_data, status, url_string) elements from an eventws query
    The list is due to the fact that entities too large are split into subqueries
    rasw_data's can be None in case of URLExceptions (the message tells what happened in case)
    :raise: ValueError if the query cannot be firhter splitted (max difference between start and
    end time : 1 second)
    """
    url = urljoin(eventws, format='text', **args)
    arr = []
    try:
        raw_data, code, msg = urlread(url, decode='utf8', raise_http_err=False)
        if code == 413:  # payload too large (formerly: request entity too large)
            start = dateutil.parser.parse(args.get('start', datetime(1970, 1, 1).isoformat()))
            end = dateutil.parser.parse(args.get('end', datetime.utcnow().isoformat()))
            total_seconds_diff = (end-start).total_seconds() / 2
            if total_seconds_diff < 1:
                raise ValueError("%d: %s (maximum recursion reached: time window < 1 sec)" %
                                 (code, msg))
                # arr.append((None, "Cannot futher split start and end time", url))
            else:
                dtime = timedelta(seconds=int(total_seconds_diff))
                bounds = [start.isoformat(), (start+dtime).isoformat(), end.isoformat()]
                arr.extend(get_events_list(eventws, **dict(args, start=bounds[0], end=bounds[1])))
                arr.extend(get_events_list(eventws, **dict(args, start=bounds[1], end=bounds[2])))
        else:
            arr = [(raw_data, msg, url)]
    except URLException as exc:
        arr = [(None, str(exc.exc), url)]
    except:
        raise
    return arr


def get_search_radius(mag, minmag, maxmag, minmag_radius, maxmag_radius):
    """From a given magnitude, determines and returns the max radius (in degrees).
        Given minmag_radius and maxmag_radius and minmag and maxmag (FIXME: TO BE CALIBRATED!),
        this function returns D from the f below:

                      |
        maxmag_radius +                oooooooooooo
                      |              o
                      |            o
                      |          o
        minmag_radius + oooooooo
                      |
                      ---------+-------+------------
                            minmag     maxmag

    :return: the max radius (in degrees)
    :param mag: (numeric or list or numbers/numpy.array) the magnitude
    :param minmag: (int, float) the minimum magnitude
    :param maxmag: (int, float) the maximum magnitude
    :param minmag_radius: (int, float) the radius for `min_mag` (in degrees)
    :param maxmag_radius: (int, float) the radius for `max_mag` (in degrees)
    :return: a scalar if mag is scalar, or an numpy.array
    """
    mag = np.asarray(mag)  # do NOT copies data for existing arrays
    isscalar = not mag.shape
    if isscalar:
        mag = np.array(mag, ndmin=1)  # copies data, assures an array of dim=1

    if minmag == maxmag:
        dist = np.array(mag)
        dist[mag < minmag] = minmag_radius
        dist[mag > minmag] = maxmag_radius
        dist[mag == minmag] = np.true_divide(minmag_radius+maxmag_radius, 2)
    else:
        dist = minmag_radius + \
            np.true_divide(maxmag_radius - minmag_radius, maxmag - minmag) * (mag - minmag)
        dist[dist < minmag_radius] = minmag_radius
        dist[dist > maxmag_radius] = maxmag_radius

    return dist[0] if isscalar else dist


def response2df(response_data, strip_cells=True):
    """
        Returns a pandas dataframe from the given response_data
        :param: response_data the string sequence of data
        :raise: ValueError in case of errors (mismatching row lengths), including the case
        where the resulting dataframe is empty. Note that response_data evaluates to False, then
        `empty()` is returned without raising
    """
    if not response_data:
        raise ValueError("Empty input data")
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
    textlines = response_data.splitlines()
    for i, line in enumerate(textlines):
        items = line.split('|') if not strip_cells else [_.strip() for _ in line.split("|")]
        if i == 0:
            columns = items
            colnum = len(columns)
        elif len(items) != colnum:
            raise ValueError("Column length mismatch")
        else:
            data.append(items)

    if not data or not columns:
        raise ValueError("Data empty after parsing")
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

    if query_type.lower() == "event" or query_type.lower() == "events":
        columns = list(colnames(Event, pkey=False, fkey=False))
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
        fdsn_model_classes = [Event]
    elif query_type.lower() in ("station", "stations"):
        fdsn_model_classes = [Station]
    elif query_type.lower() in ("channel", "channels"):
        fdsn_model_classes = [Station, Channel]

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
        return pd.DataFrame()  # this allows us to call len(empty()) without errors
    elif len(obj) > 1:
        return [empty(o) for o in obj]
    obj = obj[0]
    return obj is None or obj.empty  # (hasattr(obj, 'empty') and obj.empty)


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
        most simple and typical case, it is a dict of string keys K representing the table columns
        headers** mapped to dicts (or pandas Series) D: {..., K: D, ....}.
        **(if transpose is True then each dictionary key is the row header, see below)
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
        new_columns = dframe.columns.map(lambdacol).tolist()
        counter = 1
        for new, old, i in zip(new_columns, dframe.columns, count()):
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


def locations2degrees(lat1, lon1, lat2, lon2):
    """
    Same as obspy `locations2degree` but works with numpy arrays

    From the doc:
    Convenience function to calculate the great circle distance between two
    points on a spherical Earth.

    This method uses the Vincenty formula in the special case of a spherical
    Earth. For more accurate values use the geodesic distance calculations of
    geopy (https://github.com/geopy/geopy).

    :type lat1: numpy numeric array
    :param lat1: Latitude(s) of point 1 in degrees
    :type lon1: numpy numeric array
    :param lon1: Longitude(s) of point 1 in degrees
    :type lat2: numpy numeric array
    :param lat2: Latitude(s) of point 2 in degrees
    :type lon2: numpy numeric array
    :param lon2: Longitude(s) of point 2 in degrees
    :rtype: numpy numeric array
    :return: Distance in degrees as a floating point number.

    """
    # Convert to radians.
    lat1 = np.radians(np.asarray(lat1))
    lat2 = np.radians(np.asarray(lat2))
    lon1 = np.radians(np.asarray(lon1))
    lon2 = np.radians(np.asarray(lon2))
    long_diff = lon2 - lon1
    gd = np.degrees(
        np.arctan2(
            np.sqrt((
                np.cos(lat2) * np.sin(long_diff)) ** 2 +
                (np.cos(lat1) * np.sin(lat2) - np.sin(lat1) *
                    np.cos(lat2) * np.cos(long_diff)) ** 2),
            np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) *
            np.cos(long_diff)))
    return gd


def get_url_mseed_errorcodes():
    """returns the error codes for general url exceptions and mseed errors, respectively"""
    return (-1, -2)
