'''
Download module forevents download

:date: Dec 3, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import map, next, zip, range, object

import logging, os, sys
from datetime import timedelta, datetime
from collections import OrderedDict
from io import open  # py2-3 compatible

import pandas as pd

from stream2segment.download.utils import dbsyncdf, FailedDownload, response2normalizeddf, \
    formatmsg, read_async
from stream2segment.io.db.models import WebService, Event
from stream2segment.utils.url import urlread, URLException, socket, HTTPError
from stream2segment.utils import urljoin, strptime, get_progressbar

# logger: do not use logging.getLogger(__name__) but point to stream2segment.download.logger:
# this way we preserve the logging namespace hierarchy
# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial) when calling logging
# functions of stream2segment.download.utils:
from stream2segment.download import logger  # @IgnorePep8


_MAPPINGS = {
    'emsc':  'http://www.seismicportal.eu/fdsnws/event/1/query',
     # 'isc':   'http://www.isc.ac.uk/fdsnws/event/1/query',
    'iris':  'http://service.iris.edu/fdsnws/event/1/query',
    'ncedc': 'http://service.ncedc.org/fdsnws/event/1/query',
    'scedc': 'http://service.scedc.caltech.edu/fdsnws/event/1/query',
    'usgs':  'http://earthquake.usgs.gov/fdsnws/event/1/query',
    }


def get_events_df(session, url, evt_query_args, start, end,
                  db_bufsize=30, max_downloads=30, timeout=15,
                  show_progress=False):
    '''Returns the event data frame from the given url or local file'''

    isfile = url not in _MAPPINGS and os.path.isfile(url)
    if isfile:
        evens_titer = events_iter_from_file(url)
        url = tofileuri(url)
    else:
        evens_titer = events_iter_from_url(_MAPPINGS.get(url, url),
                                           evt_query_args,
                                           start, end,
                                           max_downloads,
                                           timeout, show_progress)

    eventws_id = configure_ws_fk(url, session, db_bufsize)

    pd_df_list = []
    for url_, data in evens_titer:
        try:
            events_df = response2normalizeddf(url_, data, "event")
            pd_df_list.append(events_df)
        except ValueError as exc:
            logger.warning(formatmsg("Discarding response", exc, url_))

    events_df = None
    if pd_df_list:  # pd.concat below raise ValueError if ret is empty:
        # build the data frame:
        events_df = pd.concat(pd_df_list, axis=0, ignore_index=True, copy=False)

    if events_df is None or events_df.empty:
        raise FailedDownload(formatmsg("No events parsed",
                                       ("Malformed response data. "
                                        "Is the %s FDSN compliant?" %
                                        ('file content' if isfile else 'server')), url))

    events_df[Event.webservice_id.key] = eventws_id
    events_df = dbsyncdf(events_df, session,
                         [Event.event_id, Event.webservice_id], Event.id, buf_size=db_bufsize,
                         cols_to_print_on_err=[Event.event_id.key])

    # try to release memory for unused columns (FIXME: NEEDS TO BE TESTED)
    return events_df[[Event.id.key, Event.magnitude.key, Event.latitude.key, Event.longitude.key,
                      Event.depth_km.key, Event.time.key]].copy()


def dataframe_iter(session, url, evt_query_args, start, end,
                   db_bufsize=30, max_downloads=30, timeout=15,
                   show_progress=False):
    '''Returns the event data frame from the given url or local file'''

    isfile = url not in _MAPPINGS and os.path.isfile(url)
    if isfile:
        evens_titer = events_iter_from_file(url)
        url = tofileuri(url)
    else:
        evens_titer = events_iter_from_url(_MAPPINGS.get(url, url),
                                           evt_query_args,
                                           start, end,
                                           max_downloads,
                                           timeout, show_progress)

    pd_df_list = []
    for url_, data in evens_titer:
        try:
            yield response2normalizeddf(url_, data, "event")
        except ValueError as exc:
            logger.warning(formatmsg("Discarding response", exc, url_))

    events_df = None
    if pd_df_list:  # pd.concat below raise ValueError if ret is empty:
        # build the data frame:
        events_df = pd.concat(pd_df_list, axis=0, ignore_index=True, copy=False)

    if events_df is None or events_df.empty:
        raise FailedDownload(formatmsg("No events parsed",
                                       ("Malformed response data. "
                                        "Is the %s FDSN compliant?" %
                                        ('file content' if isfile else 'server')), url))

    events_df[Event.webservice_id.key] = eventws_id
    events_df = dbsyncdf(events_df, session,
                         [Event.event_id, Event.webservice_id], Event.id, buf_size=db_bufsize,
                         cols_to_print_on_err=[Event.event_id.key])

    # try to release memory for unused columns (FIXME: NEEDS TO BE TESTED)
    return events_df[[Event.id.key, Event.magnitude.key, Event.latitude.key, Event.longitude.key,
                      Event.depth_km.key, Event.time.key]].copy()


def events_iter_from_file(file_path):
    """Yields the tuple (filepath, events_data) from a file, which must exist on
    the local computer"""
    try:
        with open(file_path, encoding='utf-8') as opn:
            yield tofileuri(file_path), opn.read()
    except Exception as exc:
        raise FailedDownload(formatmsg("Unable to open events file", exc,
                                       file_path))


def tofileuri(file_path):
    '''returns a file uri form thegiven file, basically file:///+file_path'''
    # https://en.wikipedia.org/wiki/File_URI_scheme#Format
    return 'file:///' + os.path.abspath(os.path.normpath(file_path))


def events_iter_from_url(url, evt_query_args, start, end, max_downloads, timeout,
                         show_progress=False):
    """
    Yields an iterator of tuples (url, data), where bith are strings denoting the
    url and the corresponding response body. The returned iterator has length > 1
    if the request was too large and had to be splitted
    """

    url_, raw_data, urls = compute_urls_chunks(url, evt_query_args, start, end,
                                               timeout, max_downloads)
    yield url_, raw_data

    oks = 0
    if urls:
        logger.info(formatmsg("Request split into %d sub-requests" % (len(urls)+1),
                              "", url))
        with get_progressbar(show_progress, length=len(urls)) as pbar:

            for obj, result, exc, request in read_async(urls,
                                                        timeout=timeout,
                                                        raise_http_err=True):
                pbar.update(1)
                if exc:
                    logger.warning(formatmsg("Error fetching events", str(exc), request))
                else:
                    data, code, msg = result  # @UnusedVariable
                    if not data:
                        logger.warning(formatmsg("Discarding request", msg, request))
                    else:
                        oks += 1
                        yield request, data

    if oks < len(urls):
        logger.info('Some sub-request failed, '
                    'some available events might not have been fetched')


def configure_ws_fk(eventws_url, session, db_bufsize):
    '''configure the web service foreign key creating such a db row if it does not
    exist and returning its id'''
    ws_name = ''
    if eventws_url in _MAPPINGS:
        ws_name = eventws_url
        eventws_url = _MAPPINGS[eventws_url]
    eventws_id = session.query(WebService.id).filter(WebService.url == eventws_url).scalar()
    if eventws_id is None:  # write url to table
        data = [("event", ws_name, eventws_url)]
        dfr = pd.DataFrame(data, columns=[WebService.type.key, WebService.name.key,
                                          WebService.url.key])
        dfr = dbsyncdf(dfr, session, [WebService.url], WebService.id, buf_size=db_bufsize)
        eventws_id = dfr.iloc[0][WebService.id.key]

    return eventws_id
#     urls = build_urls(eventws_url, **args)
#     url = urljoin(eventws_url, format='text', **args)
#     ret = []
#     try:
#         datalist = get_events_list(eventws_url, **args)
#     except ValueError as exc:
#         raise FailedDownload(exc)
# 
#     if len(datalist) > 1:
#         logger.info(formatmsg("Request was split into sub-queries, aggregating the results",
#                               "Original request entity too large", url))
# 
#     for data, msg, url in datalist:
#         if not data and msg:
#             logger.warning(formatmsg("Discarding request", msg, url))
#         elif data:
#             try:
#                 events_df = response2normalizeddf(url, data, "event")
#                 ret.append(events_df)
#             except ValueError as exc:
#                 logger.warning(formatmsg("Discarding response", exc, url))
# 
#     if not ret:  # pd.concat below raise ValueError if ret is empty:
#         raise FailedDownload(formatmsg("",
#                                        ("No events found. Check input config. "
#                                         "or log for details"), url))
# 
#     events_df = pd.concat(ret, axis=0, ignore_index=True, copy=False)
#     events_df[Event.webservice_id.key] = eventws_id
#     events_df = dbsyncdf(events_df, session,
#                          [Event.event_id, Event.webservice_id], Event.id, buf_size=db_bufsize,
#                          cols_to_print_on_err=[Event.event_id.key])
# 
#     # try to release memory for unused columns (FIXME: NEEDS TO BE TESTED)
#     return events_df[[Event.id.key, Event.magnitude.key, Event.latitude.key, Event.longitude.key,
#                       Event.depth_km.key, Event.time.key]].copy()


def compute_urls_chunks(eventws, evt_query_args, start, end, timeout, max_downloads):
    """Returns a  the tuple (url (string), raw_data (string), urls (list of strings))
    where url and raw_data are the url read and the response content, and urls
    is a list of remaining urls to be read after auto-shrinking the request time
    window, if necessary. The list might be empty if no shrinkage was needed

    :param eventws: string denoting the event web service url
    :evt_query_args: dict of event search FDSn params mapped to their values
    :param start: start time (datetime)
    :param end: end time (datetime)
    """
    max_downloads = 0 if not max_downloads or max_downloads < 0 else max_downloads
    evt_query_args['format'] = 'text'
    start_iso = start.isoformat()
    end_iso = end.isoformat()
    time_window = None
    four_years_in_sec = 4 * 365 * 24 * 60 * 60
    urls = []
    while True:
        if not urls:  # first iteration
            timeout_ = 2 * 60  # 1st attempt: relax timeout conditions, might be long
            urls = [urljoin(eventws, **dict(evt_query_args, start=start_iso, end=end_iso))]
        else:
            if time_window is None:  # secodn iteration (1st iteration splitting request)
                logger.info("Calculating the required sub-requests")
                time_window = end - start
            timeout_ = timeout
            # create a divisor factor which is 4 for time windows >= 4 year, 2 otherwise
            # this should optimize a bit the search of the 'optimum' time window:
            factor = 4 if time_window.total_seconds() > four_years_in_sec else 2
            time_window /= factor
            iterations = len(urls) * factor
            if max_downloads and iterations > max_downloads:
                raise FailedDownload('max download (%d) exceeded, restrict your search'
                                     'or change advanced settings' % max_downloads)
            urls = []
            for i in range(iterations):
                tstart = (start + i * time_window).strftime('%Y-%m-%dT%H-%M-%S')
                tend = (start + (i + 1) * time_window).strftime('%Y-%m-%dT%H-%M-%S')
                urls.append(urljoin(eventws, **dict(evt_query_args, start=tstart, end=tend)))

        try:
            raw_data, code, msg = urlread(urls[0], decode='utf8', timeout=timeout_,
                                          raise_http_err=True, wrap_exceptions=False)

            return urls[0], raw_data, urls[1:]

        except Exception as exc:  # pylint: disable=broad-except
            # raise only if we do NOT have timeout or http err in (413, 504)
            if not isinstance(exc, socket.timeout) and not \
                    (isinstance(exc, HTTPError)
                     and exc.code in (413, 504)):  # pylint: disable=no-member
                raise FailedDownload(formatmsg("Unable to fetch events", exc,
                                               urls[0]))

#     try:
#         raw_data, code, msg = urlread(url, decode='utf8', raise_http_err=False)
#         if code == 413:  # payload too large (formerly: request entity too large)
#             start = strptime(args.get('start', datetime(1970, 1, 1)))
#             end = strptime(args.get('end', datetime.utcnow()))
#             total_seconds_diff = (end-start).total_seconds() / 2
#             if total_seconds_diff < 1:
#                 raise ValueError("%d: %s (maximum recursion reached: time window < 1 sec)" %
#                                  (code, msg))
#                 # arr.append((None, "Cannot futher split start and end time", url))
#             else:
#                 dtime = timedelta(seconds=int(total_seconds_diff))
#                 bounds = [start.isoformat(), (start+dtime).isoformat(), end.isoformat()]
#                 arr.extend(get_events_list(eventws, **dict(args, start=bounds[0], end=bounds[1])))
#                 arr.extend(get_events_list(eventws, **dict(args, start=bounds[1], end=bounds[2])))
#         else:
#             arr = [(raw_data, msg, url)]
#     except URLException as exc:
#         arr = [(None, str(exc.exc), url)]
#     except:
#         raise
#     return arr


# def get_events_list(eventws, **args):
#     """Returns a list of tuples (raw_data, status, url_string) elements from an eventws query
#     The list is due to the fact that entities too large are split into subqueries
#     rasw_data's can be None in case of URLExceptions (the message tells what happened in case)
#     :raise: ValueError if the query cannot be firhter splitted (max difference between start and
#     end time : 1 second)
#     """
#     url = urljoin(eventws, format='text', **args)
#     arr = []
#     try:
#         raw_data, code, msg = urlread(url, decode='utf8', raise_http_err=False)
#         if code == 413:  # payload too large (formerly: request entity too large)
#             start = strptime(args.get('start', datetime(1970, 1, 1)))
#             end = strptime(args.get('end', datetime.utcnow()))
#             total_seconds_diff = (end-start).total_seconds() / 2
#             if total_seconds_diff < 1:
#                 raise ValueError("%d: %s (maximum recursion reached: time window < 1 sec)" %
#                                  (code, msg))
#                 # arr.append((None, "Cannot futher split start and end time", url))
#             else:
#                 dtime = timedelta(seconds=int(total_seconds_diff))
#                 bounds = [start.isoformat(), (start+dtime).isoformat(), end.isoformat()]
#                 arr.extend(get_events_list(eventws, **dict(args, start=bounds[0], end=bounds[1])))
#                 arr.extend(get_events_list(eventws, **dict(args, start=bounds[1], end=bounds[2])))
#         else:
#             arr = [(raw_data, msg, url)]
#     except URLException as exc:
#         arr = [(None, str(exc.exc), url)]
#     except:
#         raise
#     return arr
# 
# 
# def build_urls(eventws_url, **args):
#     urls = []
#     start = strptime(args.get('start', datetime(1970, 1, 1)))
#     end = strptime(args.get('end', datetime.utcnow()))
# 
#     tdelta = timedelta(days=92)  # approximately 3 Months
#     breakloop = False
# 
#     while True:
#         args['starttime'] = start.isoformat()
#         etime = start + tdelta
#         if etime >= end:
#             breakloop = True
#             etime = end
#         args['endtime'] = etime.isoformat()
#         urls.append(urljoin(eventws_url, **args))
#         start = etime
#         if breakloop:
#             break
# 
#     return urls

