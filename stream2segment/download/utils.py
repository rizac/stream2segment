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

import os
import re
from itertools import chain
from collections import OrderedDict
from functools import cmp_to_key

from future.utils import viewitems

import pandas as pd
import psutil

from stream2segment.io.db.models import Event, Station, Channel, Fdsnws, DataCenter
from stream2segment.io.db.pdsql import harmonize_columns, \
    harmonize_rows, colnames, syncdf
from stream2segment.utils.url import read_async as original_read_async, urlread, \
    urlparse, Request, get_opener, responses

# logger: do not use logging.getLogger(__name__) but point to stream2segment.download.logger:
# this way we preserve the logging namespace hierarchy
# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial) when calling logging
# functions of stream2segment.download.utils:
from stream2segment.download import logger  # @IgnorePep8


class QuitDownload(Exception):
    """
    This is an abstract-like class representing an Exception to be raised
    as soon as something causes no segments to be downloaded.

    This class should not be called directly. Rather, the user should reaise a
    :class:`NothingToDownload` or :class:`FailedDownload` (see their documentation)
    """

    def __init__(self, exc_or_msg):
        """Creates a new QuitDownload instance

        :param exc_or_msg: an Exception or a message string. If string, it is usually
            passed via the :function:`formatmsg` function in order to provide harmonized
            message formats
        """
        if isinstance(exc_or_msg, KeyError):  # just re-format key errors
            exc_or_msg = 'KeyError: %s' % str(exc_or_msg)
        super(QuitDownload, self).__init__(str(exc_or_msg))


class NothingToDownload(QuitDownload):
    '''This class represents an exception that should be raised whenever the download process
    has no segments to download according to the user's settings.
    Currently, stream2segments catches these Exceptions logging their message as level INFO and
    returning a 0 (=successful) status code

    This class and :class:`FailedDownload` both inherit from :class:`QuitDownload`.
    '''
    pass


class FailedDownload(QuitDownload):
    '''This class represents an exception that should be raised whenever the download process
    could not proceed. E.g., a download error (e.g., no internet connection) prevents to
    fetch any data. Currently, stream2segments catches these Exceptions logging their message as
    level CRITICAL or ERROR and returning a nonzero (=unsuccessful) status code

    This class and :class:`NothingToDownload` both inherit from :class:`QuitDownload`
    '''
    pass


def formatmsg(action=None, errmsg=None, url=None):
    """Function which formats a message in order to have normalized message
    types across the program (e.g., in logging utilities). The argument can contain new
    (e.g., "{}") but also old-style format keywords (such as '%s', '%d') for usage within the
    logging functions, e.g.: `logging.warning(msg('%d segments discarded', 'no response'), 3)`.
    The resulting string message will be in any of the following formats (according to
    how many arguments are non-empty):
    ```
        "{action} ({errmsg}). url: {url}"
        "{action} ({errmsg})"
        "{action}"
        "{errmsg}. url: {url}"
        "{errmsg}"
        "{url}"
        ""
    ```
    :param action: string or None: what has been done (e.g. "discarded 3 events")
    :param errmsg: string or Exception: the Exception or error message which caused the action
    :param url: the url (string) or `urllib2.Request` object: the url originating the message, if
    the latter was issued from a web request
    """
    msg = action.strip()
    if errmsg:
        # sometimes exceptions have no message, append their name
        # (e.g. socket.timeout would now print at least 'timeout')
        strerr = (str(errmsg) or str(errmsg.__class__.__name__)).strip()
        msg = "{} ({})".format(msg, strerr) if msg else strerr
    if url:
        urlmsg = url2str(url).strip()
        msg = "{}. url: {}".format(msg, urlmsg) if msg else urlmsg
    return msg


def url2str(obj):
    """converts an url or `urllib2.Request` object to string. In the latter case, the format is:
    "{obj.get_full_url()}" if `obj.data` is falsy
    "{obj.get_full_url()}, data: '{obj.get_data()}'" if `obj.data` has no newlines, or
    "{obj.get_full_url()}, data: '{obj.get_data()[:I]}'" otherwise (I=obj.get_data().find('\n')`)
    """
    try:
        url = obj.get_full_url()
        data = obj.data
        if data is not None:
            maxnum = 200
            str_data = ("%s\n...(showing first %d characters only)" % (data[:maxnum], maxnum)) \
                if len(data) > maxnum else data
        url = "%s, POST data:\n%s" % (url, str_data)
    except AttributeError:
        url = obj
    return url


def read_async(iterable, urlkey=None, max_workers=None, blocksize=1024 * 1024,
               decode=None, raise_http_err=True, timeout=None, max_mem_consumption=90,
               **kwargs):
    """Wrapper around read_async defined in url which raises a :class:`FailedDownload` in case
    of MemoryError
    :param max_mem_consumption: a value in (0, 100] denoting the threshold in % of the
    total memory after which the program should raise. This should return as fast as possible
    consuming the less memory possible, and assuring the quit-download message will be sent to
    the logger
    """
    do_memcheck = max_mem_consumption > 0 and max_mem_consumption < 100
    process = psutil.Process(os.getpid()) if do_memcheck else None
    count = 0
    step = 100
    for result in original_read_async(iterable, urlkey, max_workers, blocksize, decode,
                                      raise_http_err, timeout, **kwargs):
        yield result
        if do_memcheck:
            count += 1
            if count == step:
                count = 0
                mem_percent = process.memory_percent()
                if mem_percent > max_mem_consumption:
                    raise FailedDownload(("Memory overflow: %.2f%% (used) > "
                                          "%.2f%% (threshold)") %
                                         (mem_percent, max_mem_consumption))


def dbsyncdf(dataframe, session, matching_columns, autoincrement_pkey_col, update=False,
             buf_size=10, keep_duplicates=False, return_df=True, cols_to_print_on_err=None):
    """Calls `syncdf` and writes to the logger before returning the new dataframe.
    Raises a :class:`FailedDownload` if the returned dataframe is empty (no row saved)"""

    oninsert_err_callback = handledbexc(cols_to_print_on_err, update=False)
    onupdate_err_callback = handledbexc(cols_to_print_on_err, update=True)
    onduplicates_callback = oninsert_err_callback

    inserted, not_inserted, updated, not_updated, df = \
        syncdf(dataframe, session, matching_columns, autoincrement_pkey_col, update,
               buf_size, keep_duplicates,
               onduplicates_callback, oninsert_err_callback, onupdate_err_callback)

    table = autoincrement_pkey_col.class_
    if df.empty:
        raise FailedDownload(formatmsg("No row saved to table '%s'" % table.__tablename__,
                                       "unknown error, check log for details and db connection"))
    dblog(table, inserted, not_inserted, updated, not_updated)
    return df


def handledbexc(cols_to_print_on_err, update=False):
    """Returns a **function** to be passed to pdsql functions when inserting/ updating
    the db. Basically, it prints to log"""
    if not cols_to_print_on_err:
        return None

    def hde(dataframe, exception):
        if not dataframe.empty:
            try:
                # if sql-alchemy exception, try to guess the orig atrribute which represents
                # the wrapped exception
                # http://docs.sqlalchemy.org/en/latest/core/exceptions.html
                errmsg = str(exception.orig)
            except AttributeError:
                # just use the string representation of exception
                errmsg = str(exception)
            len_df = len(dataframe)
            msg = formatmsg("%d database rows not %s" %
                            (len_df, "updated" if update else "inserted"), errmsg)
            logwarn_dataframe(dataframe, msg, cols_to_print_on_err)

    return hde


def logwarn_dataframe(dataframe, msg, cols_to_print_on_err, max_row_count=30):
    '''prints (using log.warning) the current dataframe. Does not check if dataframe is empty'''
    len_df = len(dataframe)
    if len_df > max_row_count:
        footer = "\n... (showing first %d rows only)" % max_row_count
        dataframe = dataframe.iloc[:max_row_count]
    else:
        footer = ""
    msg = "{}:\n{}{}".format(msg, dataframe.to_string(columns=cols_to_print_on_err,
                                                      index=False), footer)
    logger.warning(msg)


def dblog(table, inserted, not_inserted, updated=0, not_updated=0):
    """Prints to log the result of a database wrtie operation.
    Use this function to harmonize the message format and make it more readable in log or terminal
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


def response2normalizeddf(url, raw_data, dbmodel_key):
    """Returns a normalized and harmonized dataframe from raw_data. dbmodel_key can be 'event'
    'station' or 'channel'. Raises ValueError if the resulting dataframe is empty or if
    a ValueError is raised from sub-functions

    :param url: url (string) or `Request` object. Used only to log the specified
        url in case of wranings
    :param raw_data: valid FDSN data in text format. For info see:
        https://www.fdsn.org/webservices/FDSN-WS-Specifications-1.1.pdf#page=12
    """

    dframe = response2df(raw_data)
    oldlen, dframe = len(dframe), normalize_fdsn_dframe(dframe, dbmodel_key)
    # stations_df surely not empty:
    if oldlen > len(dframe):
        logger.warning(formatmsg("%d row(s) discarded",
                                 "malformed text data", url),
                       oldlen - len(dframe))
    return dframe


def response2df(response_data, strip_cells=True):
    """Returns a pandas dataframe from the given fdsn text format data `response_data`

    :param: response_data the string sequence of data as text. For info see:
        https://www.fdsn.org/webservices/FDSN-WS-Specifications-1.1.pdf#page=12
    :raise: ValueError in case of errors (mismatching row lengths), including the case
        where the resulting dataframe is empty. Note that response_data evaluates to False, then
        `empty()` is returned without raising
    """
    if not response_data:
        raise ValueError("Empty input data")
    data = []
    expected_length = None
    # parse text into dataframe. Note that we check the row lengths beforehand cause pandas fills
    # with trailing NaNs which we don't want to handle. E.g.:
    # >>> pd.DataFrame(data=[[1,2], [3,4,5]], columns=['a', 'b', 'g'])
    #    a  b    g
    # 0  1  2  NaN
    # 1  3  4  5.0
    # We use simple list append and not np.append cause np string arrays having fixed lengths
    # sometimes cut strings. np.append(arr1, arr2) seems to handle this, but let's be safe
    for line in response_data.splitlines():
        if line[:1] == '#':
            continue

        items = line.split('|') if not strip_cells else [_.strip() for _ in line.split("|")]
        if expected_length is None:
            expected_length = len(items)
        elif expected_length != len(items):
            raise ValueError("Column length mismatch")

        data.append(items)

    if not data:
        raise ValueError("Data empty after parsing")
    return pd.DataFrame(data=data)


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
    if ret.empty:
        raise ValueError("Malformed data (invalid values, e.g., NaN's)")
    return ret


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
    if query_df.empty:
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
        raise ValueError(("Mismatching number of columns in '%s' query."
                          "\nExpected:\n%s\nFound:\n%s") %
                         (query_type.lower(), str(oldcolumns), str(columns)))

    return query_df.rename(columns={cold: cnew for cold, cnew in zip(oldcolumns, columns)})


def harmonize_fdsn_dframe(query_df, query_type):
    """harmonizes the query dataframe (convert to dataframe dtypes, removes NaNs
    etcetera) according to query_type.

    :param query_df: a query dataframe *on which `rename_columns` has already been called*
    :param query_type: either 'event', 'channel', 'station'
    :return: a new dataframe with only the good values
    """
    if query_df.empty:
        return query_df

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


class s2scodes(object):  # pylint: disable=too-few-public-methods, invalid-name
    '''simple container for download codes'''
    url_err = -1
    mseed_err = -2
    timespan_err = -204
    timespan_warn = -200
    seg_not_found = None
    # codes and codes which might be returned in case of restricted data access:
    restricted_data = (204, 401, 403, 404)


def to_fdsn_arg(iterable):
    ''' Converts an iterable of strings denotings networks, stations, locations or channels
    into a valid string argument for an fdsn query,
    This methid basically joins all element of `iterable` with a comma after removing all
    elements starting with '!' (used in this application to denote logical not, and not
    fdsn standard).

    :param iterable: an iterable of strings. This function does not check if any string
        element is invalid for the query (e.g., it contains spaces)
    '''
    return ",".join(v for v in iterable if v[0:1] != '!')


def get_s2s_responses():
    '''Creates a default response dict which maps http responses (int-like objects) to
    the tuple ('title', 'legend',  sort_value)

    `sort_value` is a value which controls the order of each http response code, as follows:

    code           Meaning             sort value
    =============  =================== ===================================
    2xx            HTTP code success   0xx   (float(code-200))
    -200           out of time warning 0.5   (=> next to 'success')
    -204           out of time error   99.1  (=> after all successfull response)
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
    '''
    resp = {}
    for code, title in viewitems(responses):
        leg = None
        sortpos = code
        if code >= 500:
            sortpos = code - 300
            leg = ('No data saved (download failed: Server error, server response code %d)') % code
        elif code >= 400:
            sortpos = code - 300
            leg = ('No data saved (download failed: Client error, server response code %d)') % code
        elif code >= 300:
            sortpos = code + 100
            leg = ('Data status unknown (download completed, server response code %d '
                   'indicates Redirection)') % code
        elif code >= 200:
            sortpos = code - 200
            if code == 200:
                leg = 'Data saved (download ok, no additional warning)'
            elif code == 204:
                leg = 'Data saved but empty (download ok, the server did not return any data)'
            else:
                leg = ('Data probably saved (download completed, server response code %d '
                       'indicates Success)') % code
        elif code >= 100:
            sortpos = code + 200
            leg = ('Data status unknown (download completed, server response code %d '
                   'indicates Informational response)') % code
        if leg is not None:
            resp[code] = title, leg, float(sortpos)
    # custom codes:
    codes = s2scodes
    resp[codes.timespan_warn] = ('OK Partially Saved', 'Data saved (download ok, '
                                 'some received data chunks were completely outside '
                                 'the requested time span and discarded)', 0.5)
    resp[codes.timespan_err] = ('Time Span Error', 'No data saved (download ok, data completely '
                                'outside requested time span)', 99.1)
    resp[codes.mseed_err] = ('MSeed Error', 'No data saved (download ok, '
                             'malformed MiniSeed data)', 99.2)
    resp[codes.url_err] = ('Url Error', 'No data saved (download failed, generic url error: '
                           'timeout, no internet connection, ...)', 99.3)
    resp[codes.seg_not_found] = ('Segment Not Found', 'No data saved (download ok, segment data '
                                 'not found, e.g., after a multi-segment request)', 99.4)

    return resp


class intkeysdict(dict):
    '''a defaultdict -like dict with integer keys (e.g. http status codes) mapped to int
    (occurrences of that status code). This dict tries to cast in all getting / insertion
    operation each key to integer, in order to make keys such as '200' and 200 the same:
    this is important because some datacenters return http codes as strings (such as '200')
    instead of integers (200), and both should be considered the same code'''
    # implementation note: In a previous version, we used a normal defaultdict as values of
    # DownloadStats, but we might account for keys (http codes) passed as strings
    # (e.g. '204' instead of 204): we decided to implement a custom dict with both defaultdict
    # and cast-to-int functionalities because it is more maintainable, clean, and even faster
    # (couple of seconds on 2 millions segments is
    # negligible, but we compensate a bit the overhead of
    # casting to int each time, which raises from 1.5 seconds - normal defaultdict -
    # to about 5.5 seconds. In any case, everything negligible compared to the
    # time spent downloading data)
    def __missing__(self, key):  # @UnusedVariable
        return 0

    def __setitem__(self, key, val):
        try:
            key = int(key)
        except:  # @IgnorePep8 pylint: disable=bare-except
            pass
        return dict.__setitem__(self, key, val)  # slightly faster than super(...).__setitiem__

    def __getitem__(self, key):
        try:
            key = int(key)
        except:  # @IgnorePep8 pylint: disable=bare-except
            pass
        return dict.__getitem__(self, key)  # slightly faster than super(...).__setitiem__


class DownloadStats(OrderedDict):
    ''':ref:`class``OrderedDict` subclass (with defaultdict-like capabilities) which holds
        statistics of a download.
        Keys of this dict are the domains (string), and values are `defaultdict`s of
        download codes keys (usually integers) mapped to their
        occurrences (any code key castable to int will be casted and inserted as int).
        This class is inteded for representation, e.g. typical usage:

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
        If you want to fill custom codes (non-standard HTTP status code, including our
        application codes -1, -2, -200, -204 and None), you should subclass this class.
        For instance, to add a custom GAP_OVLAP_CODE integer:
        ```
            class DownloadStats2(DownloadStats):
                GAP_OVLAP_CODE = -2000
                resp = dict(DownloadStats.resp, GAP_OVLAP_CODE=('OK Gaps Overlaps',  # title
                                                    'Data saved (download ok, '  # legend
                                                    'data has gaps or overlaps)',
                                                    0.1)  # sort order (put it next ot '200 ok')
                )
        ```

        In this case, please note:
        1. titles should be all with first letters capitalized (to conform to HTTP
           messages implemented as values of `stream2segment.utils.url.responses`)
        2. legends should have the format:
           '<Data saved|No data saved> (download <ok|failed|completed><optional details>)'
        3. The last tuple element is a float denoting the column position (order)
           when this class is printed or its `str` method called. The sort values
           for the default codes are described in `get_s2s_responses`
    '''
    resp = get_s2s_responses()

    def __missing__(self, key):  # @UnusedVariable
        '''returns an new intkeysdict and **sets** it in this dict
        '''
        # To implement a defaultdict like behaviour, we might simply `return intkeysdict()`,
        # but this would work when setting a key of this dict directly. E.g.
        # `downloadstats['geofon'] += 5`
        # This dict, on the other hand, has assignements of this type:
        # `downloadstats['geofon'][204] += 5`
        # so we need to assign here the intkeysdict() before returning it
        value = intkeysdict()
        OrderedDict.__setitem__(self, key, value)
        return value

    @classmethod
    def titlelegend(cls, code):
        '''Returns the title (string), legend (string) and column order (number or None) for the
        given missing / unknown code.
        If code is not found, returns the tuple
        ```
        "Status code %s" % str(code),
        "Data potentially saved (download completed with unknown HTTP code %d)" % str(code)
        ```
        '''
        titleleg = cls.resp.get(code, None)
        if titleleg is None:
            titleleg = ("Code %s" % str(code),
                        "Data status unknown "
                        "(download completed, server response code %s is unknown)" % str(code))
        else:
            titleleg = titleleg[:2]
        return titleleg

    @classmethod
    def sortcodes(cls, codes):
        '''Returns a list from the iterable `codes`, sorting them ascending with the rules
        described in `get_s2s_responses`.
        codes not in the default ones (i.e., in `cls.resp`) are pushed to the end. When comparing
        two codes both not in the default ones, the one which is castable to int comes first
        (if both are not castable, the first one is choosen, if both are castable, then their
        natural order as integers is chosen)

        :param codes: an iterable of numeric codes, usually but not necessarily integers
        '''
        def cmp_func(kode1, kode2):
            '''sort func'''
            in1, in2 = kode1 in cls.resp, kode2 in cls.resp
            if not in1 and not in2:
                # both codes not default one, i.e. not in `self.resp`: the first one caastable to
                # int has priority. If both castable, sort them as integers. If both not
                # castable, choose kode1
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
        '''prints a nicely formatted table with the statistics of the download. Returns the
        empty string if this object is empty. Consider calling
        `self.normalizecodes() if the codes whereby we populated this object came from the unsfae
        sources (e.g., some web services might have returned strings instead of integers,
        and without `normalizecodes` they would be displayed in two different columns
        '''
        # create a set of unique codes:
        sorted_codes = self.sortcodes(set((k for dic in self.values() for k in dic)))
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
        # add last column. Note that by default  (we did not specified it) sorted_codes are ints:
        # it is important to provide the same type for any new column
        data_df[len(data_df.columns)] = data_df.sum(axis=1)

        # Set sorted_codes and legend. Columns should take the min available space, so stack them
        # in rows via a word wrap. Unfortunately, pandas does not allow this, so we need to create
        # a top dataframe with our col headers. Mopreover, add the legend to be displayed at the
        # bottom after the whole dataframe for non standard http codes
        columns_df = pd.DataFrame(columns=data_df.columns)
        legend = []
        colwidths = data_df.iloc[-1].astype(str).str.len().tolist()

        # adjust all sorted_codes (split into chunks not to make them too wide, set legend, ...etc):
        for i, code in enumerate(chain(sorted_codes, ['TOTAL'])):
            title = code
            if i < len(sorted_codes):  # last column is the total string, not a response code
                title, leg = self.titlelegend(code)
                legend.append("%s: %s" % (title, leg))

            # make, title, splitting in rows not to make column to wide, and adjust all other
            # cokumns accordingly if new rows needs to be added:
            rows = [_ for _ in title.split(" ") if _.strip()]
            rows_to_insert = len(rows) - len(columns_df)
            if rows_to_insert > 0:
                emptyrows = pd.DataFrame(index=[''] * rows_to_insert,
                                         columns=data_df.columns,
                                         data=[[''] * len(columns_df.columns)] * rows_to_insert)
                columns_df = pd.concat((emptyrows, columns_df))
            # calculate colmax:
            colmax = max(len(_) for _ in rows)
            if colmax > colwidths[i]:
                colwidths[i] = colmax
            # align every row left:
            columns_df.iloc[len(columns_df) - len(rows):, i] = \
                [("{:<%d}" % colwidths[i]).format(r) for r in rows]

        # create column header by setting the same number of rows for each column:
        # create separator lines:
        maxindexwidth = data_df.index.astype(str).str.len().max()
        linesep_df = pd.DataFrame(data=[["-" * cw for cw in colwidths]],
                                  index=['-' * maxindexwidth])
        data_df = pd.concat((columns_df, linesep_df, data_df))

        with pd.option_context('max_colwidth', 50):
            # creating to_string needs max_colwidth as its default (50), otherwise, numbers
            # are left-aligned (just noticed from failing tests. impossible to understand why.
            # Btw, note that d has all dtypes = object, because mixes numeric and string values)
            ret = data_df.to_string(na_rep='0', justify='right', header=False)

        if legend:
            legend = ["\n\nCOLUMNS DETAILS:"] + legend
            ret += "\n - ".join(legend)
        return ret


class Authorizer(object):
    '''Class handling authorization/authentication'''

    def __init__(self, token):
        '''initializes a new Authorizer, a class handling authorization and authentication
        for restricted data

        :param arg: a filepath (to a token), the token data (bytes),
            or a tuple (username, password). If None, this authorizer is no-op
        '''
        self._uname, self._pswd, self._token = None, None, None
        if token is not None:
            token_file = None
            if isinstance(token, (tuple, list)):
                if len(token) != 2 or not all(isinstance(_, str) for _ in token):
                    raise ValueError('provide username and password as list/tuple of two '
                                     'strings')
                self._uname, self._pswd = token
            else:
                # check if there's a local file that matches the provided string
                token_file = token if os.path.isfile(token) else None
                if token_file is not None:
                    with open(token_file, 'rb') as fhd:
                        token = fhd.read()
                self._token = token
                if not self._validate_eida_token(token.decode() if isinstance(token, bytes)
                                                 else token):
                    raise ValueError("Invalid token. If you passed a file path, "
                                     "check also that the file exists")

    @staticmethod
    def _validate_eida_token(token):
        """Along the lines of obspy: basic check to test that a token is ok"""
        if re.search(pattern='BEGIN PGP MESSAGE', string=token,
                     flags=re.IGNORECASE):  # @UndefinedVariable
            return True
        return False

    @property
    def isnoop(self):
        '''Returns if this authorizer does actually nothing (download open data only)'''
        return (self._uname, self._pswd, self._token) == (None, None, None)

    @property
    def token(self):
        '''returns the token (as bytes), or None. You can safely use this method also in an if
        statement: `if auth.token`, as the token can not be empty'''
        return self._token

    @property
    def userpass(self):
        '''Returnss the tuple (user, passowrd), or None, You can safely use this method also
        in an if statement: `if auth.userpass`'''
        return None if (self._uname, self._pswd) == (None, None) else (self._uname, self._pswd)
