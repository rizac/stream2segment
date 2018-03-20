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

import logging
import os
from itertools import chain, cycle
from collections import defaultdict

from future.utils import viewitems

import pandas as pd
import psutil

from stream2segment.io.db.models import Event, Station, Channel
from stream2segment.io.db.pdsql import harmonize_columns, \
    harmonize_rows, colnames, syncdf
from stream2segment.utils.url import read_async as original_read_async
from stream2segment.utils.msgs import MSG

from future.standard_library import install_aliases
install_aliases()
from http.client import responses  # @UnresolvedImport @IgnorePep8

# logger: do not use logging.getLogger(__name__) but point to stream2segment.download.logger:
# this way we preserve the logging namespace hierarchy
# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial) when calling logging
# functions of stream2segment.download.utils:
from stream2segment.download import logger  # @IgnorePep8


class QuitDownload(Exception):
    """
    This is an exception that should be raised from the functions of this package, when something
    prevents the continuation of the download process.

    Passing an Exception in the construcor makes this exception "critical", which is
    handled differntly by the main download process. E.g.:

    - There is no data because of a download error (no data fetched). A function should then:

      ```raise QuitDownload(Exception(...))```

      and the caller function might then log.error the exception, and return non-zero.

    - There is no data because of current settings (e.g., all segments already downloaded
      with current retry settings): the program should then:

      ```raise QuitDownload(string_message)```

      and the caller function might then log.info the message, and return zero.

    The method 'iscritical' of any `QuitDownload` object tells the user if the object has been built
    for indicating a download error (first case).

    Note that in both cases the string messages need most likely to be built with the `MSG`
    function for harmonizing the message outputs.
    (Note also that with the default logging settings defined in stream2segment.main from the
    command line `log.info` and `log.error` both print also to `stdout`, `log.warning` and
    `log.debug` do not).
    """

    def __init__(self, exc_or_msg):
        """Creates a new QuitDownload instance
        :param exc_or_msg: if Exception, then this object will log.error in the `log()` method
        and return a nonzero exit code (error), otherwise (if string) this object will log.info
        inthere and return 0
        """
        super(QuitDownload, self).__init__(str(exc_or_msg))
        self.iscritical = isinstance(exc_or_msg, Exception)


def read_async(iterable, urlkey=None, max_workers=None, blocksize=1024 * 1024,
               decode=None, raise_http_err=True, timeout=None, max_mem_consumption=90,
               **kwargs):
    """Wrapper around read_async defined in url which raises a QuitDownload in case of MemoryError
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
                    raise QuitDownload(MemoryError(("Memory overflow: %.2f%% (used) > "
                                                    "%.2f%% (threshold)") %
                                                   (mem_percent, max_mem_consumption)))


def dbsyncdf(dataframe, session, matching_columns, autoincrement_pkey_col, update=False,
             buf_size=10, drop_duplicates=True, return_df=True, cols_to_print_on_err=None):
    """Calls `syncdf` and writes to the logger before returning the
    new dataframe. Raises `QuitDownload` if the returned dataframe is empty (no row saved)"""

    oninsert_err_callback = handledbexc(cols_to_print_on_err, update=False)
    onupdate_err_callback = handledbexc(cols_to_print_on_err, update=True)
    onduplicates_callback = oninsert_err_callback

    inserted, not_inserted, updated, not_updated, df = \
        syncdf(dataframe, session, matching_columns, autoincrement_pkey_col, update,
               buf_size, drop_duplicates,
               onduplicates_callback, oninsert_err_callback, onupdate_err_callback)

    table = autoincrement_pkey_col.class_
    if df.empty:
        raise QuitDownload(Exception(MSG("No row saved to table '%s'" % table.__tablename__,
                                         "unknown error, check log for details and db connection")))
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
            msg = MSG("%d database rows not %s" % (len_df, "updated" if update else "inserted"),
                      errmsg)
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
            logger.info(MSG("%s: %s" % (_header, msg), infomsg))

        dolog(inserted, not_inserted, "%d new %s inserted", ", %d discarded")
        dolog(updated, not_updated, "%d %s updated", ", %d discarded")


def response2normalizeddf(url, raw_data, dbmodel_key):
    """Returns a normalized and harmonized dataframe from raw_data. dbmodel_key can be 'event'
    'station' or 'channel'. Raises ValueError if the resulting dataframe is empty or if
    a ValueError is raised from sub-functions
    :param url: url (string) or `Request` object. Used only to log the specified
    url in case of wranings
    """

    dframe = response2df(raw_data)
    oldlen, dframe = len(dframe), normalize_fdsn_dframe(dframe, dbmodel_key)
    # stations_df surely not empty:
    if oldlen > len(dframe):
        logger.warning(MSG("%d row(s) discarded",
                           "malformed server response data, e.g. NaN's", url),
                       oldlen - len(dframe))
    return dframe


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


class DownloadStats(defaultdict):
    ''':ref:`class``defaultdict` subclass which holds statistics of a download.
        Keys of this dict are the domains (string), and values are `defaultdict`s of
        download codes keys (numeric, ususally but not necessarily int's) mapped to their
        occurrences. This class is inteded for representation, e.g. typical usage:

        ```
            d = DownloadStats()
            d['domain.org'][200] += 4
            d['domain2.org2'][413] = 4
            ...
            d.normalizecodes()  # optional: if the source of codes is not safe, this merges
                                # string codes with their int, in case e.g. '200' was returned
            print(str(d))
        ```

        If you want to fill custom codes (non-standard HTTP status code, including our
        application codes -1, -2, -200, -204 and None), adds the tuple (title, legend)
        to `self.resp`. E.g.:
        `self.resp(1300) = ('OK with gaps', 'Data saved (download OK, data has gaps)')`
        In this case, please note:
        1. titles should be all with first letter capitalized (to conform to HTTP
           messages implemented in `http.client.responses`)
        1. legends should have the format:
           '<Data saved|No data saved> (download <ok|failed|completed><optional details>)'

        2. The column order (when this class is printed or its `str` method called)
           roughly follows the code natural ordering (using `float(code)`),
           with the following exceptions:

           code       sort value
           =======    ==========
           -200       200.5 (-> so that it is next to 'success')
           -204       399.1 (->so that it is after all successfull response, and before errors)
           -2         399.8 (mseed erros)
           -1         399.9  (url errors)
           <any>      1000  (any code for which float(code) raises, including None of course)
           =========  ==========
    '''

    def __init__(self):
        '''initializes a new instance'''
        # apparently, using defaultdict(int) is slightly faster than collections.Count
        super(DownloadStats, self).__init__(lambda: defaultdict(int))
        # build an internal dict of codes mapped to the tuple title, legend
        resp = {}
        for code, title in viewitems(responses):
            leg = None
            if code == 200:
                leg = 'Data saved (download ok, no additional warning)'
            elif code == 204:
                leg = 'Data saved but empty (download ok, the server did not return any data)'
            elif code >= 500:
                leg = ('No data saved (download failed, HTTP code %d indicates '
                       'Server error)') % code
            elif code >= 400:
                leg = ('No data saved (download failed, HTTP code %d indicates '
                       'Client error)') % code
            elif code >= 300:
                leg = ('Data potentially saved (download completed, HTTP code %d indicates '
                       'Redirection)') % code
            elif code >= 200:
                leg = ('Data potentially saved (download completed, HTTP code %d indicates '
                       'Success)') % code
            elif code >= 100:
                leg = ('Data potentially saved (download completed, HTTP code %d indicates '
                       'Informational response)') % code
            if leg is not None:
                resp[code] = title, leg
        # custom codes:
        customcodes = custom_download_codes()
        URLERR, MSEEDERR, OUTTIMEERR, OUTTIMEWARN = customcodes
        resp[URLERR] = ('Url Error', 'No data saved (download failed due to generic error: '
                        'timeout, no internet connection, ...)')
        resp[MSEEDERR] = ('MSeed Error', 'No data saved (download ok, data malformed could '
                          'not be read as MiniSeed)')
        resp[OUTTIMEERR] = ('Time Span Error', 'No data saved (download ok, data completely '
                            'outside requested time span)')
        resp[OUTTIMEWARN] = ('OK Partially Saved', 'Data saved (download ok, data saved '
                             'partially: some received data chunks where completely outside '
                             'requested time span')
        resp[None] = ('Segment Not Found', 'No data saved (download ok, segment data '
                      'not found, e.g., after a multi-segment request)')

        self.resp = resp

    def normalizecodes(self):
        '''normalizes all values (defaultdict) of this object casting keys to int, and merging
          their values if an int key is present.
          Useful if the codes provided are also instance of `str`'''
        for val in self.values():
            for key in list(val.keys()):  # we will modify val inplace, copy keys
                try:
                    intkey = int(key)
                    if intkey != key:  # e.g. key is str. False if key is int or np.int
                        val[intkey] += val.pop(key)
                except Exception:
                    pass
        return self

    def titlelegend(self, code):
        '''Returns the title (string), legend (string) and column order (number or None) for the
        given missing / unknown code.
        If code is not found, returns the tuple
        ```
        "Status code %s" % str(code),
        "Data potentially saved (download completed with unknown HTTP code %d)" % str(code)
        ```
        '''
        titleleg = self.resp.get(code, None)
        if titleleg is None:
            titleleg = ("Status code %s" % str(code),
                        "Data potentially saved (download completed with unknown HTTP code %d)")
        return titleleg

    @staticmethod
    def sortcodes(codes):
        '''Returns a list from the iterable codes, sorted ascending. Each element of code is
        intended to be any of the standard HTTP codes plus our user defined codes
        (-1, -2, -200, -204, None) or any numeric user-defined code.
        The sort ordering roughly follows the codes natural ordering (using `float(code)`.
        consider it when implementing user defined codes),
        with the following exceptions:

        code       sort value
        =======    ==========
        -200       200.5 (-> so that in principle it is next to 'OK success')
        -204       399.1 (->so that it is after all successfull HTTP response, and before errors)
        -2         399.8 (mseed erros)
        -1         399.9  (url errors)
        <else>     1000  (any code for which float(code) raises)

        :param codes: an iterable of numeric codes. If not numeric, a code is pushed at the end
        '''
        URLERR, MSEEDERR, OUTTIMEERR, OUTTIMEWARN = custom_download_codes()

        def sortkey(kode):
            '''sort func'''
            if kode == OUTTIMEWARN:
                return 200.5
            elif kode == OUTTIMEERR:
                return 399.1
            elif kode == MSEEDERR:
                return 399.8
            elif kode == URLERR:
                return 399.9
            try:
                return float(kode)
            except:  # @IgnorePep8
                return 1000

        return sorted(codes, key=sortkey)

    def __str__(self):
        '''prints a nicely formatted table with the statistics of the download. Returns the
        empty string if this object is empty. Consider calling
        `self.normalizecodes() if the codes whereby we populated this object came from the unsfae
        sources (e.g., some web services might have returned strings instead of integers,
        and without `normalizecodes` they would be displayed in two different columns
        '''
        resp = dict(responses)
        customcodes = custom_download_codes()
        URLERR, MSEEDERR, OUTTIMEERR, OUTTIMEWARN = customcodes
        resp[URLERR] = 'Url Error'
        resp[MSEEDERR] = 'MSeed Error'
        resp[OUTTIMEERR] = 'Time Span Error'
        resp[OUTTIMEWARN] = 'OK Partially Saved'
        resp[None] = 'Segment Not Found'

        # create a set of unique codes:
        sorted_codes = DownloadStats.sortcodes(set((k for dic in self.values() for k in dic)))
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


def custom_download_codes():
    """returns the tuple (url_err, mseed_err, timespan_err, timespan_warn), i.e. the tuple
    (-1, -2, -204, -200) where each number represents a custom download code
    not included in the standard HTTP status codes:
    * -1 denotes general url exceptions (e.g. no internet conenction)
    * -2 denotes mseed data errors while reading downloaded data, and
    * -204 denotes a timespan error: all response is out of time with respect to the
      reqeuest's time-span
    * -200 denotes a timespan warning: some response data was out of time with respect to
      the request's time-span (only the data intersecting with the time span has been
      saved)
    """
    return (-1, -2, -204, -200)


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
