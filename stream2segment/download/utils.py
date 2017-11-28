'''
Utilities for the download package.

Module implementing all functions not involving IO operations
(logging, url read, db IO operations) in order to cleanup a bit the main module

:date: Nov 25, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
from __future__ import division
# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import zip, range

import re
from datetime import timedelta, datetime
import dateutil
from itertools import count, chain
from collections import defaultdict

import numpy as np
import pandas as pd
from sqlalchemy.orm.session import object_session

from stream2segment.io.db.models import Event, Station, Channel, DataCenter, fdsn_urls
from stream2segment.io.db.pd_sql_utils import harmonize_columns,\
    harmonize_rows, colnames, dbquery2df
from stream2segment.utils.url import urlread, URLException
from stream2segment.utils import urljoin, strconvert

from future.standard_library import install_aliases
install_aliases()
from http.client import responses  # @UnresolvedImport @IgnorePep8


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


class DownloadStats(defaultdict):
    ''':ref:`class``defaultdict` subclass which holds statistics of a download.
        Keys of this dict are the domains (string), and values are `defaultdict`s of
        download codes keys (int) mapped to their occurrences. Typical usage is to add data and
        then print the overall statistics:
        ```
            d = DownloadStats()
            d['domain.org'][200] += 4
            d['domain2.org2'][413] = 4
            print(str(d))
        ```
    '''
    def __init__(self):
        '''initializes a new instance'''
        # apparently, using defaultdict(int) is slightly faster than collections.Count
        super(DownloadStats, self).__init__(lambda: defaultdict(int))

    def normalizecodes(self):
        '''normalizes all values (defaultdict) of this object casting keys to int, and merging
          their values if an int key is present.
          Useful if the codes provided are also instance of `str`'''
        for val in self.values():
            for key in list(val.keys()):
                try:
                    intkey = int(key)
                    if intkey != key:  # e.g. key is str. False if jey is int or np.int
                        val[intkey] += val.pop(key)
                except Exception:
                    pass
        return self

    def __str__(self):
        '''prints a nicely formatted table with the statistics of the download. Returns the
        empty string if this object is empty'''
        resp = dict(responses)
        customcodes = custom_download_codes()
        URLERR, MSEEDERR, OUTTIMEERR, OUTTIMEWARN = customcodes
        resp[URLERR] = 'Url Error'
        resp[MSEEDERR] = 'MSeed Error'
        resp[OUTTIMEERR] = 'Time Span Error'
        resp[OUTTIMEWARN] = 'OK Partially Saved'
        resp[None] = 'Segment Not Found'

        # create a set of unique codes:
        colset = set((k for dic in self.values() for k in dic))
        if not colset:
            return ""

        # create a list of sorted codes. First 200, then OUTTIMEWARN, then
        # all HTTP codes sorted naturally, then unkwnown codes, then Null

        # assure a minimum val between all codes: custom and 0-600 (standard http codes):
        minval = min(min(min(customcodes), 0), 600)
        maxval = max(max(max(customcodes), 0), 600)

        def sortkey(key):
            '''sorts an integer key. 200Ok comes first, then OUTTIMEWARN, Then all other ints
            sorted "normally". At the end all non-int key values, and as really last None's'''
            if key is None:
                return maxval+2
            elif key == 200:
                return minval-2
            elif key == OUTTIMEWARN:
                return minval-1
            else:
                try:
                    return int(key)
                except:
                    return maxval+1

        columns = sorted(colset, key=sortkey)

        # create data matrix
        data = []
        rows = []
        colindex = {c: i for i, c in enumerate(columns)}
        for row, dic in self.items():
            if not dic:
                continue
            rows.append(row)
            datarow = [0]*len(columns)
            data.append(datarow)
            for key, value in dic.items():
                datarow[colindex[key]] = value

        if not rows:
            return ""

        # create dataframe of the data. Columns will be set later
        d = pd.DataFrame(index=rows, data=data)
        d.loc["TOTAL"] = d.sum(axis=0)
        # add last column. Note that by default  (we did not specified it) columns are integers:
        # it is important to provide the same type for any new column
        d[len(d.columns)] = d.sum(axis=1)

        # Set columns and legend. Columns should take the min available space, so stack them
        # in rows via a word wrap. Unfortunately, pandas does not allow this, so we need to create
        # a top dataframe with our col headers. Mopreover, add the legend to be displayed at the
        # bottom after the whole dataframe for non standard http codes
        columns_df = pd.DataFrame(columns=d.columns)
        legend = []
        colwidths = d.iloc[-1].astype(str).str.len().tolist()

        def codetype2str(code):
            code = int(code / 100)
            if code == 1:
                return "Informational response"
            elif code == 2:
                return "Success"
            elif code == 3:
                return "Redirection"
            elif code == 4:
                return "Client error"
            elif code == 5:
                return "Server error"
            else:
                return "Unknown status"

        for i, c in enumerate(chain(columns, ['TOTAL'])):
            if i < len(columns):  # last column is the total string, not a response code
                code = c
                if c not in resp:
                    c = "Unknown %s" % str(c)
                    legend.append("%s: Non-standard response, unknown message (code=%s)" %
                                  (str(c), str(code)))
                else:
                    c = resp[c]
                    if code == URLERR:
                        legend.append("%s: Generic Url error (e.g., timeout, no internet "
                                      "connection, ...)" % str(c))
                    elif code == MSEEDERR:
                        legend.append("%s: Response OK, but data cannot be read as "
                                      "MiniSeed" % str(c))
                    elif code == OUTTIMEERR:
                        legend.append("%s: Response OK, but data completely outside "
                                      "requested time span " % str(c))
                    elif code == OUTTIMEWARN:
                        legend.append("%s: Response OK, data saved partially: some received "
                                      "data chunks where completely outside requested time "
                                      "span" % str(c))
                    elif code is None:
                        legend.append("%s: Response OK, but segment data not found "
                                      "(e.g., after a multi-segment request)" % str(c))
                    else:
                        legend.append("%s: Standard response message indicating %s (code=%d)" %
                                      (str(c), codetype2str(code), code))
            rows = [_ for _ in c.split(" ") if _.strip()]
            rows_to_insert = len(rows) - len(columns_df)
            if rows_to_insert > 0:
                emptyrows = pd.DataFrame(index=['']*rows_to_insert,
                                         columns=d.columns,
                                         data=[['']*len(columns_df.columns)]*rows_to_insert)
                columns_df = pd.concat((emptyrows, columns_df))
            # calculate colmax:
            colmax = max(len(c) for c in rows)
            if colmax > colwidths[i]:
                colwidths[i] = colmax
            # align every row left:
            columns_df.iloc[len(columns_df)-len(rows):, i] = \
                [("{:<%d}" % colwidths[i]).format(r) for r in rows]

        # create column header by setting the same number of rows for each column:
        # create separator lines:
        maxindexwidth = d.index.astype(str).str.len().max()
        linesep_df = pd.DataFrame(data=[["-" * cw for cw in colwidths]],
                                  index=['-' * maxindexwidth])
        d = pd.concat((columns_df, linesep_df, d))

        with pd.option_context('max_colwidth', 50):
            # creating to_string needs max_colwidth as its default (50), otherwise, numbers
            # are left-aligned (just noticed from failing tests. impossible to understand why.
            # Btw, note that d has all dtypes = object, because mixes numeric and string values)
            ret = d.to_string(na_rep='0', justify='right', header=False)

        if legend:
            legend = ["\n\nCOLUMNS DETAILS:"] + legend
            ret += "\n - ".join(legend)
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


def custom_download_codes():
    """returns the tuple (url_err, mseed_err, timespan_err, timespan_warn), i.e. the tuple
    (-1, -2, -204, -200) where each number represents a custom download code
    not included in the standard HTTP status codes:
    * -1 denotes general url exceptions (e.g. no internet conenction)
    * -2 denotes mseed data errors while reading downloaded data, and
    * -204 denotes a timespan error: all response is out of time with respect to the reqeuest's
      time-span
    * -200 denotes a timespan warning: some response data was out of time with respect to the
      request's time-span (only the data intersecting with the time span has been saved)
    """
    return (-1, -2, -204, -200)


def eidarsiter(responsetext):
    """iterator yielding the tuple (url, postdata) for each datacenter found in responsetext
    :param responsetext: the eida routing service response text
    """
    # not really pythonic code, but I enjoyed avoiding copying strings and creating lists
    # so this iterator is most likely really low memory consuming
    start = 0
    textlen = len(responsetext)
    while start < textlen:
        end = responsetext.find("\n\n", start)
        if end < 0:
            end = textlen
        mid = responsetext.find("\n", start, end)  # note: now we set a new value to idx
        if mid > -1:
            url, postdata = responsetext[start:mid].strip(), responsetext[mid:end].strip()
            if url and postdata:
                yield url, postdata
        start = end + 2


class EidaValidator(object):
    '''Class for validating stations duplicates according to the eida routing service
    response text'''
    def __init__(self, datacenters_df, responsetext):
        """Initializes a validator. You can then call `isin` to check if a station is valid
        :param datacenters_df: a dataframe representing the datacenters read from the eida
        routing service
        :param responsetext: the plain response text from the eida routing service
        """
        self.dic = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:
                                                                                           set()))))
        reg = re.compile("^(\\S+) (\\S+) (\\S+) (\\S+) .*$",
                         re.MULTILINE)  # @UndefinedVariable
        for url, postdata in eidarsiter(responsetext):
            _ = datacenters_df[datacenters_df[DataCenter.dataselect_url.key] == url]
            if _.empty:
                _ = datacenters_df[datacenters_df[DataCenter.station_url.key] == url]
            if len(_) != 1:
                continue
            dc_id = _[DataCenter.id.key].iloc[0]
            for match in reg.finditer(postdata):
                try:
                    net, sta, loc, cha = \
                        match.group(1), match.group(2), match.group(3), match.group(4)
                except IndexError:
                    continue
                self.add(dc_id, net, sta, loc, cha)

    @staticmethod
    def _tore(wild_str):
        if wild_str == '--':
            wild_str = ''
        return re.compile("^%s$" % strconvert.wild2re(wild_str))

    def add(self, dc_id, net, sta, loc, cha):
        """adds the tuple datacenter id, network station location channels to the internal dic
        :param dc_id: integer
        :param net: string, the network name
        :param sta: string, the station name. Special cases: '*' (match all), "--" (empty)
        :param sta: string, the location name. Special cases: '*' (match all), "--" (empty)
        :param cha: string, the channel (can contain wildcards like '*' or '?'). Special cases:
                    '*' (match all)
        """
        self.dic[dc_id][net][self._tore(sta)][self._tore(loc)].add(self._tore(cha))

    @staticmethod
    def _get(regexiterable, key, return_bool=False):
        for regex in regexiterable:
            if regex.match(key):
                return True if return_bool else regexiterable[regex]
        return False if return_bool else None

    def isin(self, dc_id, net, sta, loc, cha):
        """Returns a boolean (or a list of booleans) telling if the tuple arguments:
        ```(dc_id, net, sta, loc, cha)```
        match any of the eida response lines of text.
        Returns a list of boolean if the arguments are iterable (not including strings)
        Returns numpy.array if return_np = True
        """
        # dc_id - > {net_re -> //}
        stadic = self.dic.get(dc_id, {}).get(net, None)
        if stadic is None:
            return False
        # sta_re - > {loc_re -> //}
        locdic = self._get(stadic, sta)
        if locdic is None:
            return False
        # loc_re - > set(cha_re,..)
        chaset = self._get(locdic, loc)
        if chaset is None:
            return False
        return self._get(chaset, cha, return_bool=True)
