"""
Events download functions

:date: Dec 3, 2017

.. moduleauthor:: <rizac@gfz-potsdam.de>
"""
import os
from datetime import timedelta
import logging
from io import StringIO
from urllib.request import Request

import numpy as np
import pandas as pd

from stream2segment.io.cli import get_progressbar
from stream2segment.io.db.pdsql import DbManager
from stream2segment.download.exc import FailedDownload, NothingToDownload
from stream2segment.download.db.models import Event, WebService
from stream2segment.download.url import urlread, socket, HTTPError, read_async
from stream2segment.download.modules.utils import (dbsyncdf, get_dataframe_from_fdsn,
                                                   formatmsg,
                                                   EVENTWS_MAPPING,
                                                   strptime,
                                                   urljoin,
                                                   DbExcLogger,
                                                   OneTimeLogger,
                                                   compress)

# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial):
logger = logging.getLogger(__name__)


def get_events_df(session, url, evt_query_args, start, end,
                  db_bufsize=30, timeout=15,
                  show_progress=False):
    """Return the event data frame from the given url or local file"""

    eventws_id = configure_ws_fk(url, session, db_bufsize)

    pd_df_list = events_df_list(url, evt_query_args, start, end, timeout, show_progress)
    # pd_df_list surely not empty (otherwise we raised FailedDownload)
    events_df = pd.concat(pd_df_list, axis=0, ignore_index=True, copy=False)

    events_df[Event.webservice_id.key] = eventws_id
    events_df = dbsyncdf(events_df, session,
                         [Event.event_id, Event.webservice_id], Event.id,
                         buf_size=db_bufsize,
                         cols_to_print_on_err=[Event.event_id.key, Event.magnitude.key,
                                               Event.time.key],
                         keep_duplicates='first')

    # try to release memory for unused columns (FIXME: NEEDS TO BE TESTED)
    return events_df[[Event.id.key, Event.magnitude.key, Event.latitude.key,
                      Event.longitude.key, Event.depth_km.key, Event.time.key]].copy()


def configure_ws_fk(eventws_url, session, db_bufsize):
    """Configure the web service foreign key creating such a db row if it does
    not exist and returning its id"""
    ws_name = ''
    if eventws_url in EVENTWS_MAPPING:
        ws_name = eventws_url
        eventws_url = EVENTWS_MAPPING[eventws_url]
    elif islocalfile(eventws_url):
        eventws_url = tofileuri(eventws_url)

    eventws_id = session.query(WebService.id). \
        filter(WebService.url == eventws_url).scalar()

    if eventws_id is None:  # write url to table
        data = [("event", ws_name, eventws_url)]
        dfr = pd.DataFrame(data, columns=[WebService.type.key,
                                          WebService.name.key,
                                          WebService.url.key])
        dfr = dbsyncdf(dfr, session, [WebService.url], WebService.id,
                       buf_size=db_bufsize)
        eventws_id = dfr.iloc[0][WebService.id.key]

    return eventws_id


# error string (constants, used in test so we can change them with no problem, hopefully)
ERR_FETCH = "Unable to fetch events"
ERR_FETCH_FDSN = "Unable to fetch events, data not in the supported FDSN format"
ERR_READ_FDSN = "Unable to read events, data not in the supported FDSN format"
ERR_FETCH_NODATA = "No event received, search parameters might be too strict"


def events_df_list(url, evt_query_args, start, end, timeout=15, show_progress=False):
    """Return a list of pandas dataframe(s) from the event url or file

    :param url: a valid url, a mappings string, or a local file (fdsn 'text'
        formatted)
    """
    urls_and_data = []
    is_local_file = islocalfile(url)
    if is_local_file:
        try:
            urls_and_data.append(events_data_from_file(url))
        except Exception as exc:
            raise FailedDownload(formatmsg(ERR_READ_FDSN, exc, tofileuri(url)))
    else:
        try:
            urls_and_data = list(events_iter_from_url(url, evt_query_args, start, end,
                                                      timeout, show_progress))
        except NothingToDownload:
            raise
        except Exception as exc:
            raise FailedDownload(formatmsg(ERR_FETCH, exc,
                                           normalize_url(url, evt_query_args, start, end)))

    pd_df_list = []
    for url_, data in urls_and_data:
        # data surely not empty, FDSN formatted
        try:
            pd_df_list.append(get_dataframe_from_fdsn(data, "event", url_))
        except Exception as exc:
            msg = ERR_READ_FDSN if is_local_file else ERR_FETCH_FDSN
            if is_local_file or len(urls_and_data) == 1:  # raise:
                raise FailedDownload(formatmsg(msg, exc, url_))
            else:
                logger.warning(formatmsg(msg, exc, url_))

    if not pd_df_list:
        raise FailedDownload(formatmsg(ERR_FETCH_FDSN, 'details in log file',
                                       normalize_url(url, evt_query_args, start, end)))

    return pd_df_list


def normalize_url(base_url, evt_query_args, start, end):
    """Return the normalized URL string of url:
    1. Converts base_url to a normal URL if the former is a key of EVENTWS_MAPPING
    2. Set event_query_args 'starttime' and 'endtime' equal to the provided arguments
       `start` and `end` (handling duplicate names such as 'start' / 'startime')
    3. Converts 'minmag' 'maxmag' in `evt_query_args` to 'minmagnitude', 'maxmagnitude'
    4. Adds a custom format 'text' unless the base_url is not EVENTWS_MAPPING['isc']
    """
    _url, _query_args = _normalize(base_url, evt_query_args, start, end)
    return urljoin(_url, **_query_args)


def events_data_from_file(file_path):
    """Yield the tuple (filepath, events_data) from a file, which must exist
    on the local computer.
    The only supported format is txt.
    """
    with open(file_path, encoding='utf-8') as opn:
        data = opn.read()
        if not data:
            raise ValueError('Empty file')
        return tofileuri(file_path), data


def tofileuri(file_path):
    """return a file URI form the given file,
    basically file_path:///+basename(file_path)
    """
    # https://en.wikipedia.org/wiki/File_URI_scheme#Format
    # return 'file:///' + os.path.abspath(os.path.normpath(file_path))
    return 'file:///' + os.path.basename(file_path)


def islocalfile(url):
    """Return whether url denotes a local file path, existing on the computer
    machine
    """
    return url not in EVENTWS_MAPPING and os.path.isfile(url)


def events_iter_from_url(base_url, evt_query_args, start, end, timeout,
                         show_progress=False):
    """Yield an iterator of tuples (url, data), where both are strings denoting
    the URL and the corresponding response body. The returned iterator has
    length > 1 if the request was too large and had to be split
    """
    base_url, evt_query_args = _normalize(base_url, evt_query_args, start, end)
    end_iso = evt_query_args['endtime']

    url = urljoin(base_url, **evt_query_args)
    result = _urlread(url, timeout)
    if result is not _SUSPECTED_REQUEST_TOO_ARGE:
        if not result:
            raise NothingToDownload(formatmsg(ERR_FETCH_NODATA, "", url))
        yield url, result  # then result is the tuple (url, raw_data)
    else:
        logger.info("Request seems to be too large, splitting into "
                    "sub-requests")

        # control that at least one subrequest returned non empty data:
        yielded = False

        # the tricky part below is actually the progressbar part. It must:
        # 1 not be linear, thus advance "more" at lower magnitudes (where
        #   events are more dense)
        # 2 consider that, when the maximum magnitude depth is reached, we split
        #   by time and in this case only the last sub-request should advance the
        #   progress bar
        total_pbar_steps = _get_freq_mag_distrib(evt_query_args)[2].sum()
        with get_progressbar(show_progress, length=total_pbar_steps) as pbar:
            downloads = [evt_query_args]

            while downloads:
                evt_q_args = _split_request(downloads.pop(0))
                for i, evt_q_arg in enumerate(evt_q_args):
                    url = urljoin(base_url, **evt_q_arg)
                    result = _urlread(url, timeout)
                    if result is not _SUSPECTED_REQUEST_TOO_ARGE:
                        # update pbar only if the end of the request equals
                        # the global end_iso (when recursion is done on time, it
                        # updates only on the first time chunk):
                        if evt_q_arg['endtime'] == end_iso:
                            steps = _get_freq_mag_distrib(evt_q_arg)[2].sum()
                            pbar.update(steps)
                        if result:  # do not yield empty data
                            yield url, result  # (url, raw_data)
                            yielded = True
                    else:
                        downloads.insert(i, evt_q_arg)
        if not yielded:
            raise ValueError("no sub-request returned data")


def _normalize(base_url, evt_query_args, start, end):
    """Return the normalized tuple (url, evt_query_args):
    1. Converts base_url to a normal URL if the former is a key of EVENTWS_MAPPING
    2. Set event_query_args 'starttime' and 'endtime' equal to the provided arguments
       `start` and `end` (handling duplicate names such as 'start' / 'startime')
    3. Converts 'minmag' 'maxmag' in `evt_query_args` to 'minmagnitude', 'maxmagnitude'
    4. Adds a custom format 'text' unless the base_url is not EVENTWS_MAPPING['isc']
    """
    start_iso = start.isoformat()
    end_iso = end.isoformat()
    # This should never happen but let's be safe: override start and end
    if 'start' in evt_query_args:
        evt_query_args.pop('start')
    evt_query_args['starttime'] = start_iso
    if 'end' in evt_query_args:
        evt_query_args.pop('end')
    evt_query_args['endtime'] = end_iso
    # assure that we have 'minmagnitude' and 'maxmagnitude' as mag parameters,
    # if any:
    if 'minmag' in evt_query_args:
        minmag = evt_query_args.pop('minmag')
        if 'minmagnitude' not in evt_query_args:
            evt_query_args['minmagnitude'] = minmag
    if 'maxmag' in evt_query_args:
        maxmag = evt_query_args.pop('maxmag')
        if 'maxmagnitude' not in evt_query_args:
            evt_query_args['maxmagnitude'] = maxmag

    url = EVENTWS_MAPPING.get(base_url, base_url)
    evt_query_args.setdefault('format', "text")

    return url, evt_query_args


_SUSPECTED_REQUEST_TOO_ARGE = type('suspected_request_too_large', (object,), {})()


def _urlread(url, timeout=None):
    """Wrapper around `urlread` but returns None if the url should be split
    because of a too long request
    """
    raw_data, exc, code = urlread(url, decode='utf8', timeout=timeout)

    if exc is not None:
        if isinstance(exc, socket.timeout) or \
                (isinstance(exc, HTTPError) and exc.code in (413, 504)):  # noqa
            return _SUSPECTED_REQUEST_TOO_ARGE
        raise exc

    if code == 204:
        raw_data = ''

    return raw_data


def _split_request(evt_query_args):
    """Split the event query issued with the given `event_query_args` (dict)
    and returns a two-element list:
    (event_query_args1, event_query_args2)
    of event query parameters (dicts) resulting from splitting `evt_query_args`
    """
    minmag, deltamag, evtfreq_freq_mag_dist = _get_freq_mag_distrib(evt_query_args)
    if len(evtfreq_freq_mag_dist) < 2:  # max recusrion on magnitudes, split by time:
        start = strptime(evt_query_args['starttime'])
        end = strptime(evt_query_args['endtime'])
        days_diff = int((end - start).days / 2.0)
        if days_diff < 1:
            raise ValueError('maximum recursion depth reached')
        half_dtime_str = (start + timedelta(days=days_diff)).isoformat()
        evt_query_args1 = dict(evt_query_args)
        evt_query_args2 = dict(evt_query_args)
        evt_query_args1['endtime'] = half_dtime_str
        evt_query_args2['starttime'] = half_dtime_str
    else:
        half = evtfreq_freq_mag_dist.sum() / 2.0
        idx = 1
        while evtfreq_freq_mag_dist[:idx + 1].sum() < half:
            idx += 1
        mag_half = minmag + idx * deltamag
        evt_query_args1 = dict(evt_query_args)
        evt_query_args2 = dict(evt_query_args)
        evt_query_args1['maxmagnitude'] = str(round(mag_half, 1))
        evt_query_args2['minmagnitude'] = str(round(mag_half, 1))

    return evt_query_args1, evt_query_args2


def _get_freq_mag_distrib(evt_query_args):
    """Return the tuple minmag, step, distrib, where minmag is a float
    representing `func` first point (magnitude), step is the magnitude
    distance two adjacent points of `distrib`, and `distrib` is a a numpy array
    (dtype=int) representing the theoretical events distribution from a given
    magnitude `mag`:
    ```
    f(mag) = 10 ** (9-mag)
    ```
    """
    default_min, step, default_max = 0, .1, 9

    # create the function:
    ret = ((10 ** (default_max - np.arange(default_min, default_max, step))) + 0.5). \
        astype(int)
    # set all points of magnitude <1 equal to the frequency at magnitude 1
    # (no frequency increase after that threshold)
    index_of_mag_1 = int(0.5 + ((1.0 - default_min) / step))
    if index_of_mag_1 > 0:
        ret[:index_of_mag_1] = ret[index_of_mag_1]

    # trim ret if maxmagnitude is given:
    if 'maxmagnitude' in evt_query_args:
        maxmag = float(evt_query_args['maxmagnitude'])
        index_of_maxmag = int(0.5 + ((maxmag - default_min) / step))
        if index_of_maxmag < len(ret):
            ret = ret[:index_of_maxmag]

    minmag = default_min
    # trim ret if minmagnitude is given:
    if 'minmagnitude' in evt_query_args:
        minmag = float(evt_query_args['minmagnitude'])
        index_of_minmag = int(0.5 + ((minmag - default_min) / step))
        if index_of_minmag > 0:
            ret = ret[index_of_minmag:]

    return minmag, step, ret
  
def save_quakeml(session, events_df, max_thread_workers, timeout,
                 download_blocksize, db_bufsize, show_progress=False):
    """Save event's quakeML data. envents_df must not be empty"""

    logger_header = "QuakeML"
    evt_logger = OneTimeLogger(logger_header)

    downloaded, errors, empty = 0, 0, 0

    db_exc_logger = DbExcLogger([Event.id.key])

    dbmanager = DbManager(session, Event.id,
                          update=[Event.quakeml.key],
                          buf_size=db_bufsize,
                          oninsert_err_callback=db_exc_logger.failed_insert,
                          onupdate_err_callback=db_exc_logger.failed_update)

    with get_progressbar(show_progress, length=len(events_df)) as pbar:

        iterable = zip(events_df[Event.id.key],
                       events_df[WebService.url],
                       events_df[Event.event_id.key])

        reader = read_async(iterable,
                            urlkey=lambda obj: _get_evt_request(*obj[1:]),
                            max_workers=max_thread_workers,
                            blocksize=download_blocksize, timeout=timeout)

        for obj, request, data, exc, status_code in reader:
            pbar.update(1)
            evt_id = obj[0]
            if exc:
                evt_logger.warn(request, exc)
                errors += 1
            else:
                if not data:
                    evt_logger.warn(request, "empty response")
                    empty += 1
                else:
                    downloaded += 1
                    dfr = pd.DataFrame({Event.id.key: [evt_id],
                                        Event.quakeml.key: [compress(data)]})
                    dbmanager.add(dfr)

    dbmanager.close()

    return downloaded, empty, errors


def _get_evt_request(evt_url, evt_eventid):
    """Return a Request object from the given event arguments to download the
    QuakeML
    """
    return Request(url=f"{evt_url}?eventid={evt_eventid}")
  