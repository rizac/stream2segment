"""
Utilities for the download package.

Module implementing all functions not involving IO operations
(logging, url read, db IO operations) in order to cleanup a bit the main module

:date: Nov 25, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
import os
import sys
import re
from io import StringIO
from datetime import datetime
from itertools import chain
from collections import OrderedDict
from functools import cmp_to_key
import logging

import numpy as np
from dateutil import parser as dateparser
from dateutil.tz.tz import tzutc

import pandas as pd

from stream2segment.io.db.models import MINISEED_READ_ERROR_CODE
from stream2segment.io.db.pdsql import harmonize_columns, dropnulls, syncdf
from stream2segment.io.db.inspection import colnames
from stream2segment.download.db.models import Event, Station, Channel
from stream2segment.download.exc import FailedDownload
from stream2segment.download.url import responses, adjust_max_concurrent_downloads

# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial):
logger = logging.getLogger(__name__)


def formatmsg(action=None, errmsg=None, url=None):
    """Format a message in order to have normalized message types across the
    program (e.g., in logging utilities). The argument can contain new
    (e.g., "{}") but also old-style format keywords (such as '%s', '%d') for
    usage within the logging functions, e.g.:
    `logging.warning(msg('%d segments discarded', 'no response'), 3)`.
    The resulting string message will be in any of the following formats
    (according to how many arguments are non-empty):
    ```
        "{action} ({errmsg}). url: {url}"
        "{action} ({errmsg})"
        "{action}"
        "{errmsg}. url: {url}"
        "{errmsg}"
        "{url}"
        ""
    ```
    :param action: string or None: what has been done
        (e.g. "discarded 3 events")
    :param errmsg: string or Exception: the Exception or error message which
        caused the action
    :param url: the url (string) or `urllib2.Request` object: the url
        originating the message, if the latter was issued from a web request
    """
    msg = action.strip()
    if errmsg:
        strerr = err2str(errmsg)
        msg = "{} ({})".format(msg, strerr) if msg else strerr
    if url:
        urlmsg = url2str(url, maxlen=200).strip()
        msg = "{}. url: {}".format(msg, urlmsg) if msg else urlmsg
    return msg


def err2str(err):
    """Return the string representation of `err`

    :param err: string or Exception denoting the error
    """
    # This class basically does two things: convert KeyErrors into
    # "KeyError: 'a'" and not simply "a",
    # and in case of exceptions which produce the empty string, return their
    # class name instead (e.g. socket.timeout returns 'timeout' instead of '')
    errclass = err.__class__
    if errclass == KeyError:
        return "%s: %s" % (str(errclass), str(err))
    if errclass == str:  # if we passed a string, just return it
        return err
    return (str(err) or str(errclass.__name__)).strip()


def url2str(obj, maxlen=None):
    """Convert an url or `urllib2.Request` object to string. In the latter
    case, the format is:
    "{obj.get_full_url()}" if `obj.data` is falsy
    "{obj.get_full_url()}, data: '{obj.get_data()}'"
    if `obj.data` has no newlines, or
    "{obj.get_full_url()}, data: '{obj.get_data()[:I]}'" otherwise
    (I=obj.get_data().find('\n')`)
    """
    try:
        url = str(obj.get_full_url())
        data = str(obj.data or '')
        if data:
            if maxlen is not None and len(data) > maxlen:
                data = ("%s\n...(showing first %d characters only)" %
                        (data[:maxlen], maxlen))
            url = "%s, POST data:\n%s" % (url, data)
    except AttributeError:
        url = str(obj)
    return url


def dbsyncdf(dataframe, session, matching_columns, autoincrement_pkey_col,
             update=False, buf_size=10, keep_duplicates=False, return_df=True,
             cols_to_print_on_err=None):
    """Call `syncdf` and writes to the logger before returning the new
    Dataframe. Raises a :class:`FailedDownload` if the returned Dataframe is
    empty (no row saved)
    this function should be used for bulk insert/updates of metadata (event,
    station, channels). Segments inserts/updates use the underling
    :class:`pdsql.DbManager`
    """
    db_exc_logger = DbExcLogger(cols_to_print_on_err)

    inserted, not_inserted, updated, not_updated, dfr = \
        syncdf(dataframe, session, matching_columns, autoincrement_pkey_col,
               update, buf_size, keep_duplicates,
               db_exc_logger.failed_insert,  # on duplicates callback
               db_exc_logger.failed_insert,   # on insert err callback
               db_exc_logger.failed_update)   # on update err callback

    table = autoincrement_pkey_col.class_
    if dfr.empty:
        # Build a meaningful error message for the FailedDownload exception
        err_count = len(db_exc_logger.exc_history)
        if not err_count:
            first_err = 'Unknown. Try to check log for details'
        else:
            # Take the 1st item of the Set, who cares if it's not the first inserted:
            first_err = next(iter(db_exc_logger.exc_history))
        if err_count > 1:
            err_msg = ("%d errors. Check log for details, first reported error "
                       "is: %s") % (err_count, first_err)
        else:
            err_msg = 'error: ' + first_err
        raise FailedDownload(formatmsg("No row saved to table '%s'" %
                                       table.__tablename__, err_msg))
    dblog(table, inserted, not_inserted, updated, not_updated)
    return dfr


class DbExcLogger:
    """Class handling db I/O error and logging the rows not inserted
    (:meth:`DbExcLogger.failed_insert`) or updated
    (:meth:`DbExcLogger.failed_update`). The rows have to be passed to those
    methods in form of pandas DataFrame"""

    def __init__(self, cols_to_print_on_err, max_row_count=30):
        """initialize a db logger

        :param cols_to_print_on_err: list of strings denoting the dataframe
            columns to be printed in the log'. None will print all columns
        """
        self.cols_to_print_on_err = cols_to_print_on_err
        self.max_row_count = max_row_count
        self.exc_history = set()

    def failed_insert(self, dataframe, exception):
        """logs a failed db insertion

        :param dataframe: the pandas DataFrame with rows not inserted.
        :param exception: the original exception preventing insertion
        """
        self._handledbexc(dataframe, exception, update=False)

    def failed_update(self, dataframe, exception):
        """log a failed db update

        :param dataframe: the pandas DataFrame with rows not updated.
        :param exception: the original exception preventing update
        """
        self._handledbexc(dataframe, exception, update=True)

    def _handledbexc(self, dataframe, exception, update=False):
        """Function to be passed to pdsql functions on error when inserting/
        updating the database. Basically, it prints to log
        """
        try:
            # if SQL-Alchemy exception, try to guess the orig attribute
            # which represents the wrapped exception
            # http://docs.sqlalchemy.org/en/latest/core/exceptions.html
            errmsg = str(exception.orig)
        except AttributeError:
            # just use the string representation of exception
            errmsg = str(exception)
        self.exc_history.add(errmsg)
        if not dataframe.empty:
            len_df = len(dataframe)
            msg = formatmsg("%d database row(s) not %s" %
                            (len_df, "updated" if update else "inserted"),
                            errmsg)
            logwarn_dataframe(dataframe, msg, self.cols_to_print_on_err,
                              self.max_row_count)


def logwarn_dataframe(dataframe, msg, columns=None, max_row_count=30):
    """Log as warning the current dataframe. Does not check if
    Dataframe is empty

    :param columns: the columns to print, if None writes all columns
    """
    if len(dataframe) > max_row_count:
        chunks = ['showing first',
                  'row' if max_row_count == 1 else '%d rows' % max_row_count,
                  'only']
        footer = "\n... (%s)" % " ".join(chunks)
        dataframe = dataframe.iloc[:max_row_count]
    else:
        footer = ""

    if columns is not None and len(columns) < len(dataframe.columns):
        dataframe = dataframe[list(columns)].copy()
        dataframe['...'] = pd.Categorical(('...' for _ in
                                           range(len(dataframe))))

    df_str = dataframe.to_string(na_rep='', index=False)
    msg = "{}:\n{}{}".format(msg, df_str, footer)
    logger.warning(msg)


def dblog(table, inserted, not_inserted, updated=0, not_updated=0):
    """Print to log the result of a database wrtie operation. Use this function
    to harmonize the message format and make it more readable in log or
    terminal
    """
    _header = "Db table '%s'" % table.__tablename__
    if not inserted and not not_inserted and not updated and not not_updated:
        logger.info("%s: no new row to insert, no row to update", _header)
    else:

        def dolog(ok, notok, okstr, nookstr):
            if not ok and not notok:
                return
            _errmsg = "sql errors"
            _noerrmsg = "no sql error"
            msg = okstr % (ok, "row" if ok == 1 else "rows")
            infomsg = _noerrmsg
            if notok:
                msg += nookstr % notok
                infomsg = _errmsg
            logger.info(formatmsg("%s: %s" % (_header, msg), infomsg))

        dolog(inserted, not_inserted, "%d new %s inserted", ", %d discarded")
        dolog(updated, not_updated, "%d %s updated", ", %d discarded")


def get_dataframe_from_fdsn(response_str, query_type, url=''):
    """Return a normalized and harmonized dataframe from raw_data. dbmodel_key
    can be 'event' 'station' or 'channel'. Raises ValueError if the resulting
    dataframe is empty or if a `ValueError` is raised from sub-functions

    :param response_str: the response content of a FDSN query, in str format. For
        info see https://www.fdsn.org/webservices/FDSN-WS-Specifications-1.1.pdf#page=12
    :param query_type: a string denoting the web service type:
        "event", "station" (for a station query with parameter level=station)
        or "channel" (for a station query with parameter level=channel)
    :param url: URL (string) or `Request` object. Optional, used only to log the
        specified URL in case of warnings
    """
    # read csv but do not let pandas infer data types (`_harmonize_columns` does that
    # later): `dtype=str` reads everything as strings (this prevents `event_id`s in
    # catalogs to be inadvertently casted as int), `na_values` + `keep_default_na` reads
    # empty cells as "" (this prevents channels `location` to be NULL and the relative
    # row to be dropped in `_harmonize_columns` because NULL is not allowed)
    dframe = pd.read_csv(StringIO(response_str), sep='|', header=None, comment='#',
                         dtype=str, keep_default_na=False,
                         na_values=['#N/A', '#NA', '-NaN', '-nan', '<NA>', 'N/A', 'NA',
                                    'NULL', 'NaN', 'n/a', 'nan', 'null'])
    # however
    oldlen = len(dframe)
    dframe = _rename_columns(dframe, query_type)
    dframe = _harmonize_fdsn_dframe(dframe, query_type)
    if dframe.empty:
        raise ValueError("Malformed data (e.g., type mismatch, NaN)")
    # stations_df surely not empty:
    if oldlen > len(dframe):
        dframe = dframe.copy()
        logger.warning(formatmsg("%d row(s) discarded",
                                 "malformed text data", url),
                       oldlen - len(dframe))
    return dframe


def _rename_columns(query_df, query_type):
    """Rename the columns of `query_df` according to the ORM model representing
    the FDSN query type ('event', 'station', 'channel') originating the data frame
    """
    if query_df.empty:
        return query_df

    if query_type.lower() == "event" or query_type.lower() == "events":
        model = Event
        columns = list(colnames(Event, pkey=False, fkey=False))
    elif query_type.lower() == "station" or query_type.lower() == "stations":
        model = Station
        # these are the query_df columns for a station (level=station) query:
        # `Network|Station|Latitude|Longitude|Elevation|SiteName|StartTime|
        # EndTime`
        # Set this table columns mapping (by name, so we can safely add any
        # new column at any index):
        columns = [Station.network.key, Station.station.key,
                   Station.latitude.key, Station.longitude.key,
                   Station.elevation.key, Station.site_name.key,
                   Station.start_time.key, Station.end_time.key]
    elif query_type.lower() == "channel" or query_type.lower() == "channels":
        model = Channel
        # these are the query_df expected columns for a station (level=channel)
        # query:
        # `Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|
        # Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|
        # StartTime|EndTime`
        # Some of them are for the Channel table, so select them:
        columns = [Station.network.key, Station.station.key,
                   Channel.location.key, Channel.channel.key,
                   Station.latitude.key, Station.longitude.key,
                   Station.elevation.key, Channel.depth.key,
                   Channel.azimuth.key, Channel.dip.key,
                   Channel.sensor_description.key, Channel.scale.key,
                   Channel.scale_freq.key, Channel.scale_units.key,
                   Channel.sample_rate.key, Station.start_time.key,
                   Station.end_time.key]
    else:
        raise ValueError("Invalid fdsn_model: supply Events, "
                         "Station or Channel class")

    oldcolumns = query_df.columns.tolist()
    if len(oldcolumns) > len(columns):
        # do not provide long messages, the exception is likely to be wrapped
        # also do not print columns, which are often just numbers with no meaning:
        raise ValueError("Data has %d column(s), expected: %d" %
                         (len(oldcolumns), len(columns)))

    ret = query_df.rename(columns={cold: cnew for cold, cnew in
                          zip(oldcolumns, columns)})

    # before returning, add missing nullable columns. this happens when we added new
    # columns in our models, following some new spec, but the server response is still
    # in the old format (if missing columns are not nullable, raise). The procedure below
    # assumes that new columns, if any, are appended at the end of the response
    if len(oldcolumns) < len(columns):
        missing_cols = list(colnames(model, pkey=False, fkey=False))[len(oldcolumns):]
        missing_non_nullable_cols = set(colnames(model, pkey=False, fkey=False,
                                                 nullable=False)) & set(missing_cols)
        if missing_non_nullable_cols:
            raise ValueError("Missing non-nullable column(s) in data: %s" %
                             (", ".join(missing_non_nullable_cols)))
        logger.warning("Adding missing nullable column(s) in data: %s" %
                       (", ".join(missing_cols)))
        for col in missing_cols:
            ret[col] = None

    return ret


def _harmonize_fdsn_dframe(query_df, query_type):
    """Harmonize the query dataframe (convert to dataframe dtypes, removes
    NaNs and so on) according to query_type ('event', 'station', 'channel').
    """
    if query_df.empty:
        return query_df

    if query_type.lower() in ("event", "events"):
        fdsn_model_classes = [Event]
    elif query_type.lower() in ("station", "stations"):
        fdsn_model_classes = [Station]
    elif query_type.lower() in ("channel", "channels"):
        fdsn_model_classes = [Station, Channel]
    else:
        return query_df

    # convert columns to correct dtypes (datetime, numeric etcetera). Values
    # not conforming will be set to NaN or NaT or None, thus detectable via
    # pandas.dropna or pandas.isnull
    for fdsn_model_class in fdsn_model_classes:
        query_df = harmonize_columns(fdsn_model_class, query_df)
        query_df = dropnulls(fdsn_model_class, query_df)

    return query_df


class s2scodes:  # pylint: disable=too-few-public-methods, invalid-name
    """Simple container for download codes"""
    url_err = -1
    mseed_err = MINISEED_READ_ERROR_CODE  # -2
    timespan_err = -204
    timespan_warn = -200
    seg_not_found = None
    # codes and codes which might be returned in case of restricted data access:
    restricted_data = (204, 401, 403, 404)


def get_s2s_responses():
    """Create a default response dict which maps http responses (int-like
    objects) to the tuple ('title', 'legend',  sort_value)

    `sort_value` is a value which controls the order of each http response
    code, as follows:

    code           Meaning             sort value
    =============  =================== ===================================
    2xx            HTTP code success   0xx   (float(code-200))
    -200           out of time warning 0.5   (=> next to 'success')
    -204           out of time error   99.1  (=> after all successful response)
    -2             Mseed err           99.2  (see above)
    -1             url err             99.3  (see above)
    None           seg not found       99.4  (see above)
    4xx            HTTP Client error   1xx   (float(code)-300))
    5xx            HTTP Server error   2xx   (float(code)-300))
    1xx            HTTP Informational
                   Response            3xx   (float(code)+200)
    3xx            HTTP Redirection    4xx   (float(code)+100)
    <any int>      User-defined        User-defined or float(code)
    =============  =================== ===================================

    See also `s2scodes` and `DownloadStats.sortcodes`
    """
    resp = {}
    for code, title in responses.items():
        leg = None
        sortpos = code
        if code >= 500:
            sortpos = code - 300
            leg = ('No data saved (download failed: Server error, '
                   'server response code %d)') % code
        elif code >= 400:
            sortpos = code - 300
            leg = ('No data saved (download failed: Client error, '
                   'server response code %d)') % code
        elif code >= 300:
            sortpos = code + 100
            leg = ('Data status unknown (download completed, server response '
                   'code %d indicates Redirection)') % code
        elif code >= 200:
            sortpos = code - 200
            if code == 200:
                leg = 'Data saved (download completed, no additional warning)'
            elif code == 204:
                leg = ('No data saved (download completed, the server returned '
                       '0 bytes of data)')
            else:
                leg = ('Data status unknown (download completed, server '
                       'response code %d indicates Success)') % code
        elif code >= 100:
            sortpos = code + 200
            leg = ('Data status unknown (download completed, server response '
                   'code %d indicates Informational response)') % code
        if leg is not None:
            resp[code] = title, leg, float(sortpos)
    # custom codes:
    codes = s2scodes
    resp[codes.timespan_warn] = ('OK Partially Saved',
                                 'Data saved (download completed, some data '
                                 'chunks discarded because outside the requested '
                                 'time window)', 0.5)
    resp[codes.timespan_err] = ('Time Span Error',
                                'No data saved (download completed, all data discarded '
                                'because outside the requested time window)', 99.1)
    resp[codes.mseed_err] = ('MSeed Error', 'Data saved (download completed, '
                             'malformed MiniSeed data)', 99.2)
    resp[codes.url_err] = ('Url Error',
                           'No data saved (download failed, generic url '
                           'error: timeout, no internet connection, ...)',
                           99.3)
    resp[codes.seg_not_found] = ('Segment Not Found',
                                 'No data saved (download completed, segment data not '
                                 'found, e.g., in a multi-segment request)',
                                 99.4)

    return resp


class HTTPCodesCounter(dict):
    """A dict, mapping http status codes to the number of times they occurred.
    This class handles string status codes, which are sometimes returned by
    some data center (i.e., treat '200' as if it was 200)
    """
    # implementation note: In a previous version, we used a normal defaultdict
    # as values of DownloadStats, but the above mentioned int/str problem
    # (e.g., treat '200' and 200 as the same key) turned out to be easier to
    # implement on a dict subclass, without performances drop
    def __missing__(self, key):  # @UnusedVariable
        return 0

    def __setitem__(self, key, val):
        try:
            key = int(key)  # e.g. '200' and 200 must be the same key
        except:  # noqa
            # could not convert to int: no ambiguity, use the key as it is
            pass
        # slightly faster than super(...).__setitiem__:
        return dict.__setitem__(self, key, val)

    def __getitem__(self, key):
        try:
            key = int(key)  # e.g. '200' and 200 must be the same key
        except:  # noqa
            # could not convert to int: no ambiguity, use the key as it is
            pass
        # slightly faster than super(...).__setitiem__:
        return dict.__getitem__(self, key)


class DownloadStats(OrderedDict):
    """Class storing statistics during a download routine, and printing them
    nicely formatted to string. You can think of this class as a table where
    each row is a URL (string), and each column a different HTTP status code:
    the cell value is the number of status codes received from that URL. This
    class is a dict (instead of the more natural choice of a pandas DataFrame)
    because dicts are way more efficient with dynamically changing data sizes
    (on a million 'items' inserted, dicts 1-2 hundreds seconds, DataFrames 2-3
    thousands). Typical usage:
    ```
    d = DownloadStats()
    d['domain.org'][200] += 4  # note defaultdict capabilities
    d['domain2.org2'][413] = 4
    # Strings are casted, when possible:
    d['domain2.org2']['413'] = 4 # d['domain2.org2']['413']=8
    ...
    print(str(d))
    ```

    Advanced usage:
    ---------------
    If you want to fill custom codes (non-standard HTTP status code, including
    our application codes -1, -2, -200, -204 and None), you should subclass
    this class. For instance, to add a custom GAP_OVLAP_CODE integer:
    ```
    class DownloadStats2(DownloadStats):
        GAP_OVLAP_CODE = -2000
        resp = dict(DownloadStats.resp,
                    GAP_OVLAP_CODE=('OK Gaps Overlaps',  # title
                                    'Data saved (download completed, '  # legend
                                    'data has gaps or overlaps)',
                                    0.1) # sort order (put it next ot '200 ok')
        )
    ```
    In this case, please note:
    1. titles should be all with first letters capitalized (to conform to HTTP
       messages implemented as values of `stream2segment.utils.url.responses`)
    2. legends should have the format:
       '<Data saved|No data saved> (download <ok|failed|completed><details>)'
       (where <details> is optional)
    3. The last tuple element is a float denoting the column position (order)
       when this class is printed or its `str` method called. The sort values
       for the default codes are described in `get_s2s_responses`
    """
    resp = get_s2s_responses()

    def __missing__(self, key):  # @UnusedVariable
        """Return an new intkeysdict and **sets** it in this dict"""
        # To implement a defaultdict like behaviour, we might simply
        # `return intkeysdict()`, but this would work when setting a key of
        # this dict directly. E.g. `downloadstats['geofon'] += 5`
        # This dict, on the other hand, has assignments of this type:
        # `downloadstats['geofon'][204] += 5`
        # so we need to assign here the intkeysdict() before returning it
        value = HTTPCodesCounter()
        OrderedDict.__setitem__(self, key, value)
        return value

    @classmethod
    def titlelegend(cls, code):
        """Return the title (string), legend (string) and column order (number
        or None) for the given missing / unknown code. If code is not found,
        returns default generic title and legend (see code)
        """
        titleleg = cls.resp.get(code, None)
        if titleleg is None:
            titleleg = ("Code %s" % str(code),
                        "Data status unknown (download completed, server "
                        "response code %s is unknown)" % str(code))
        else:
            titleleg = titleleg[:2]
        return titleleg

    @classmethod
    def sortcodes(cls, codes):
        """Return a list from the iterable `codes`, sorting them ascending
        with the rules described in `get_s2s_responses`. Codes not in the
        default ones (i.e., in `cls.resp`) are pushed to the end. When
        comparing two codes both not in the default ones, the one which is
        castable to int comes first (if both are not castable, the first one is
        chosen, if both are castable, then their natural order as integers is
        chosen)

        :param codes: an iterable of numeric codes, usually but not necessarily
            integers
        """
        def cmp_func(kode1, kode2):
            """sort function"""
            in1, in2 = kode1 in cls.resp, kode2 in cls.resp
            if not in1 and not in2:
                # both codes not default one, i.e. not in `self.resp`: the
                # first one caastable to int has priority. If both castable,
                # sort them as integers. If both not castable, choose kode1
                try:
                    int1 = int(kode1)
                except:  # @IgnorePep8 pylint: disable=bare-except
                    int1 = None
                try:
                    int2 = int(kode2)
                except:  # @IgnorePep8 pylint: disable=bare-except
                    int2 = None
                if int2 is None and int1 is not None:
                    return -1
                elif int1 is not None and int2 is not None:
                    return int1 - int2
                return 1
            elif not in1:
                return 1
            elif not in2:
                return -1
            return cls.resp[kode1][2] - cls.resp[kode2][2]

        return sorted(codes, key=cmp_to_key(cmp_func))

    def __str__(self):
        """Print a nicely formatted table with the statistics of the download.
        Return the empty string if this object is empty
        """
        # create a set of unique codes:
        sorted_codes = self.sortcodes(set((k for dic in self.values()
                                           for k in dic)))
        if not sorted_codes:
            return ""

        # create data matrix
        data = []
        rows = []
        colindex = {c: i for i, c in enumerate(sorted_codes)}
        for row, dic in self.items():
            if not dic:
                continue
            rows.append(row)
            datarow = [0] * len(sorted_codes)
            data.append(datarow)
            for key, value in dic.items():
                datarow[colindex[key]] = value

        if not rows:
            return ""

        # create dataframe of the data. Columns will be set later
        data_df = pd.DataFrame(index=rows, data=data)
        data_df.loc["TOTAL"] = data_df.sum(axis=0)
        # add last column. Note that by default  (we did not specified it)
        # sorted_codes are ints: it is important to provide the same type for
        # any new column
        data_df[len(data_df.columns)] = data_df.sum(axis=1)

        # Set sorted_codes and legend. Columns should take the min available
        # space, so stack them in rows via a word wrap. Unfortunately, pandas
        # does not allow this, so we need to create a top dataframe with our
        # col headers. Moreover, add the legend to be displayed at the
        # bottom after the whole dataframe for non standard http codes
        columns_df = pd.DataFrame(columns=data_df.columns)
        legend = []
        colwidths = data_df.iloc[-1].astype(str).str.len().tolist()

        # adjust all sorted_codes (split into chunks not to make them too wide,
        # set legend, ...etc):
        for i, code in enumerate(chain(sorted_codes, ['TOTAL'])):
            title = code
            if i < len(sorted_codes):
                # last column is the total string, not a response code
                title, leg = self.titlelegend(code)
                legend.append("%s: %s" % (title, leg))

            # make, title, splitting in rows not to make column to wide, and
            # adjust all other columns accordingly if new rows needs to be
            # added:
            rows = [_ for _ in title.split(" ") if _.strip()]
            rows_to_insert = len(rows) - len(columns_df)
            if rows_to_insert > 0:
                _data = [[''] * len(columns_df.columns)] * rows_to_insert
                emptyrows = pd.DataFrame(index=[''] * rows_to_insert,
                                         columns=data_df.columns,
                                         data=_data)
                columns_df = pd.concat((emptyrows, columns_df))
            # calculate colmax:
            colmax = max(len(_) for _ in rows)
            if colmax > colwidths[i]:
                colwidths[i] = colmax
            # align every row left:
            columns_df.iloc[len(columns_df) - len(rows):, i] = \
                [("{:<%d}" % colwidths[i]).format(r) for r in rows]

        # create column header by setting the same number of rows for each
        # column. Create separator lines:
        maxindexwidth = data_df.index.astype(str).str.len().max()
        linesep_df = pd.DataFrame(data=[["-" * cw for cw in colwidths]],
                                  index=['-' * maxindexwidth])
        data_df = pd.concat((columns_df, linesep_df, data_df))

        with pd.option_context('max_colwidth', 50):
            # creating to_string needs max_colwidth as its default (50),
            # otherwise, numbers are left-aligned (just noticed from failing
            # tests. impossible to understand why. Btw, note that d has all
            # dtypes = object, because mixes numeric and string values)
            ret = data_df.to_string(na_rep='0', justify='right', header=False)

        if legend:
            legend = ["\n\nCOLUMNS DETAILS:"] + legend
            ret += "\n - ".join(legend)
        return ret


EVENTWS_MAPPING = {
    'emsc':  'http://www.seismicportal.eu/fdsnws/event/1/query',
    'isc':   'http://www.isc.ac.uk/fdsnws/event/1/query',
    'iris':  'http://service.iris.edu/fdsnws/event/1/query',
    'ncedc': 'http://service.ncedc.org/fdsnws/event/1/query',
    'scedc': 'http://service.scedc.caltech.edu/fdsnws/event/1/query',
    'usgs':  'http://earthquake.usgs.gov/fdsnws/event/1/query',
}


EVENTWS_SAFE_PARAMS = ['minlatitude', 'minlat', 'maxlatitude', 'maxlat',
                       'minlongitude', 'minlon', 'maxlongitude', 'maxlon',
                       'minmagnitude', 'minmag', 'maxmagnitude', 'maxmag',
                       'mindepth', 'maxdepth']


class Authorizer:
    """Class handling authorization/authentication"""

    def __init__(self, token):
        """Initialize a new Authorizer, a class handling authorization and
        authentication for restricted data

        :param token: a filepath (to a token), the token data (bytes),
            or a tuple (username, password). If None, this authorizer is no-op
        """
        self._uname, self._pswd, self._token = None, None, None
        if token is not None:
            token_file = None
            if isinstance(token, (tuple, list)):
                if len(token) != 2 or not all(isinstance(_, str)
                                              for _ in token):
                    raise ValueError('provide username and password as '
                                     'list/tuple of two strings')
                self._uname, self._pswd = token
            else:
                # check if there's a local file that matches the provided str
                token_file = token if os.path.isfile(token) else None
                if token_file is not None:
                    with open(token_file, 'rb') as fhd:
                        token = fhd.read()
                self._token = token
                if not self._validate_eida_token(token.decode()
                                                 if isinstance(token, bytes)
                                                 else token):
                    raise ValueError("Invalid token. "
                                     "If you passed a file path, "
                                     "check that the file is a valid token")

    @staticmethod
    def _validate_eida_token(token):
        """Along the lines of ObsPy: basic check to test that a token is ok"""
        if re.search(pattern=r'\bBEGIN PGP\b', string=token,
                     flags=re.IGNORECASE):  # @UndefinedVariable
            return True
        return False

    @property
    def token(self):
        """Return the token (as bytes), or None. You can safely use this method
        also in an if statement: `if auth.token`, as the token can not be empty
        """
        return self._token

    @property
    def userpass(self):
        """Return the tuple (user, password), or None, You can safely use
        this method also in an if statement: `if auth.userpass`
        """
        if (self._uname, self._pswd) == (None, None):
            return None
        return self._uname, self._pswd


class strconvert:
    """String conversion utilities from sql-LIKE operator's wildcards,
    Filesystem's wildcards, and regular expressions
    """
    @staticmethod
    def sql2wild(text):
        """Return a new string from `text` by replacing all sql-LIKE-operator's
        wildcard characters ('sql') with their filesystem's counterparts
        ('wild'):

        === ==== === ===============================
        sql wild re  meaning
        === ==== === ===============================
        %   *    .*  matches zero or more characters
        _   ?    .   matches exactly one character
        === ==== === ===============================

        :return: string. Note that this function performs a simple
            replacement: wildcard characters in the input string will result in
            a string that is not the perfect translation of the input
        """
        return text.replace("%", "*").replace("_", "?")

    @staticmethod
    def wild2sql(text):
        """Return a new string from `text` by replacing all filesystem's wildcard
        characters ('wild') with their sql-LIKE-operator's counterparts ('sql'):

        === ==== === ===============================
        sql wild re  meaning
        === ==== === ===============================
        %   *    .*  matches zero or more characters
        _   ?    .   matches exactly one character
        === ==== === ===============================

        :return: string. Note that this function performs a simple replacement:
            sql special characters in the input string will result in a string
            that is not the perfect translation of the input
        """
        return text.replace("*", "%").replace("?", "_")

    @staticmethod
    def wild2re(text):
        """Return a new string from `text` by replacing all filesystem's wildcard
        characters ('wild') with their regular expression's counterparts ('re'):

        === ==== === ===============================
        sql wild re  meaning
        === ==== === ===============================
        %   *    .*  matches zero or more characters
        _   ?    .   matches exactly one character
        === ==== === ===============================

        :return: string. Note that this function performs a simple replacement:
            regexp special characters in the input string will result in a
            string that is not the perfect translation of the input
        """
        return re.escape(text).replace(r"\*", ".*").replace(r"\?", ".")

    @staticmethod
    def sql2re(text):
        """Return a new string from `text` by replacing all sql-LIKE-operator's
        wildcard characters ('sql') with their regular expression's
        counterparts ('re'):

        === ==== === ===============================
        sql wild re  meaning
        === ==== === ===============================
        %   *    .*  matches zero or more characters
        _   ?    .   matches exactly one character
        === ==== === ===============================

        :return: string. Note that this function performs a simple replacement:
            regexp special characters in the input string will result in a
            string that is not the perfect translation of the input
        """
        if sys.version_info[0] == 3 and sys.version_info[1] < 7:
            # versions up to 3.7 do not escape anymore "_":
            percent, underscore = r"\%", "_"
        else:
            # from version 3.7, only special characters are escaped,
            # thus neither "%" nor "_" are escaped:
            percent, underscore = "%", "_"
        return re.escape(text).replace(percent, ".*").replace(underscore, ".")


def strptime(obj):
    """Convert `obj` to a `datetime` object **in UTC without tzinfo**. This
    function should be used within this program as the opposite of
    `datetime.isoformat()` for parsing date times from, e.g. web service
    queries or command line inputs, under the assumption that no time zone
    means UTC.

    If `obj` is string, creates a `datetime` object by parsing it. If `obj`
    is not a date-time object, raises TypeError. Otherwise, uses `obj` as
    `datetime` object. Then, if the datetime object has a tzinfo supplied,
    converts it to UTC and removes the tzinfo attribute. Finally, returns the
    datetime object

    Implementation details: `datetime.strptime`does not keep time zone
    information in the parsed date-time, nor it recognizes 'Z' as 'UTC' (raises
    instead). The library `dateutil`, on the other hand, is too permissive and
    has too many false "positives" (e.g. integers or strings such as  '5-7' are
    successfully parsed into date-time). We choose `dateutil` as the code is
    shorter, cleaner, and a single hack is needed: we simply check, after a
    string `obj` is succesfully parsed into `dtime`, that `obj` contains at
    least the string `dtime.strftime(format='%Y-%m-%d')` (such as e,g,
    '2006-01-31')

    :param obj: `datetime` object or string in ISO format (see examples below)

    :return: a datetime object in UTC, with the tzinfo removed
    :raise: TypeError or ValueError
    :Example. These are all equivalent:
    ```
    strptime("2016-06-01T00:00:00.000000Z")
    strptime("2016-06-01T00.01.00CET")
    strptime("2016-06-01 00:00:00.000000Z")
    strptime("2016-06-01 00:00:00.000000")
    strptime("2016-06-01 00:00:00")
    strptime("2016-06-01 00:00:00Z")
    strptime("2016-06-01")

    This raises ValueError:
    strptime("2016-06-01Z")

    This raises TypeError:
    strptime(45.5)
    ```
    """
    dtime = obj
    if isinstance(obj, str):
        try:
            dtime = dateparser.parse(obj, fuzzy=False, fuzzy_with_tokens=False)
            # now, dateperser is quite hacky on purpose, guessing too much.
            # datetime.strptime, on the other hand, does not parse Z as UTC
            # (raises in case) and does not include the timezone in the parsed
            # date. The best (hacky) solution is to assert the bare minimum:
            # that %Y-%m-%d is in dtime:
            assert dtime.strftime('%Y-%m-%d') in obj
        except Exception as exc:
            raise ValueError(str(exc))

    if not isinstance(dtime, datetime):
        raise TypeError('string or datetime required, found %s' %
                        str(type(obj)))

    if dtime.tzinfo is not None:
        # if a time zone is specified, convert to utc and remove the timezone
        dtime = dtime.astimezone(tzutc()).replace(tzinfo=None)

    # the datetime has no timezone provided AND is in UTC:
    return dtime


def urljoin(*urlpath, **query_args):
    """Join urls and appends to it the query string obtained by kwargs
    Note that this function is intended to be simple and fast: No check is made
    about white-spaces in strings, no encoding is done, and if some value of
    `query_args` needs special formatting (e.g., "%1.1f"), that needs to be
    done before calling this function

    :param urlpath: portion of urls which will build the query url Q. For more
        complex url functions see `urlparse` library: this function builds the
        url path via a simple join stripping slashes:
        ```
        '/'.join(url.strip('/') for url in urlpath)
        ```
        So to preserve slashes (e.g., at the beginning) pass "/" or "" as
        arguments (e.g. as first argument to preserve relative paths).
    :query_args: keyword arguments which will build the query string

    :return: a query url built from arguments (string)

    Examples:
    ```
    >>> urljoin("https://abc", start='2015-01-01T00:05:00', mag=5.1, arg=True)
    'https://abc?start=2015-01-01T00:05:00&mag=5.1&arg=True'

    >>> urljoin("http://abc", "data", start='2015-01-01', mag=5.459, arg=True)
    'http://abc/data?start=2015-01-01&mag=5.459&arg=True'

    Note how slashes are handled in urlpath. These two examples give the
    same url path:

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
    # For a discussion, see https://stackoverflow.com/q/1793261
    return "{}?{}".format('/'.join(url.strip('/') for url in urlpath),
                          "&".join("{}={}".format(k, v)
                                   for k, v in query_args.items()))


class _MemoryChecker:
    """Legacy class for checking memory consumption. Initialize outside a loop and then
    call `check_memory_consumption` in the loop. A FailedDownload will be raised if the
    memory consumption exceeds a certain threshold. Memory checks were implemented
    for each request and might be useful especially in the download of segments or
    inventories (see e.g. `download.modules.segments.get_responses`) to prevent the
    program to exit with no clear message. Because these error happened long ago on 8Gb
    machines only, as of 2023 memory checks are removed from the codebase
    """
    def __init__(self, step=500, memory_max_percent=90):
        self.step = step
        self.counter = 0
        self.memory_max_percent = memory_max_percent
        import psutil
        self.process = psutil.Process(os.getpid())

    def check_memory_consumption(self):
        self.counter += 1
        if self.counter % self.step == 0:
            self.counter = 0
            mem_percent = self.process.memory_percent()
            if mem_percent > self.memory_max_percent:
                raise FailedDownload(("Memory overflow: %.2f%% (used) > "
                                      "%.2f%% (threshold)") %
                                     (mem_percent, self.memory_max_percent))
