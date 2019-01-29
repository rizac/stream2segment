'''
Download module forevents download

:date: Dec 3, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import map, next, zip, range, object

import logging, os, sys, re
from datetime import timedelta, datetime
from collections import OrderedDict
from io import open  # py2-3 compatible
import math

import numpy as np
import pandas as pd

from stream2segment.utils import StringIO
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
from stream2segment.io.db.pdsql import colnames



_MAPPINGS = {
    'emsc':  'http://www.seismicportal.eu/fdsnws/event/1/query',
    'isc':   'http://www.isc.ac.uk/fdsnws/event/1/query',
    'iris':  'http://service.iris.edu/fdsnws/event/1/query',
    'ncedc': 'http://service.ncedc.org/fdsnws/event/1/query',
    'scedc': 'http://service.scedc.caltech.edu/fdsnws/event/1/query',
    'usgs':  'http://earthquake.usgs.gov/fdsnws/event/1/query',
    }


def get_events_df(session, url, evt_query_args, start, end,
                  db_bufsize=30, timeout=15,
                  show_progress=False):
    '''Returns the event data frame from the given url or local file'''

    eventws_id = configure_ws_fk(url, session, db_bufsize)

    pd_df_list = [dfr for dfr in dataframe_iter(url, evt_query_args, start, end,
                                                timeout,
                                                show_progress)]

    events_df = None
    if pd_df_list:  # pd.concat below raise ValueError if ret is empty:
        # build the data frame:
        events_df = pd.concat(pd_df_list, axis=0, ignore_index=True, copy=False)

    if events_df is None or events_df.empty:
        isfile = islocalfile(url)
        raise FailedDownload(formatmsg("No events parsed",
                                       ("Malformed response data. "
                                        "Is the %s FDSN compliant?" %
                                        ('file content' if isfile else 'server')), url))

    events_df[Event.webservice_id.key] = eventws_id
    events_df = dbsyncdf(events_df, session,
                         [Event.event_id, Event.webservice_id], Event.id, buf_size=db_bufsize,
                         cols_to_print_on_err=[Event.event_id.key], keep_duplicates='first')

    # try to release memory for unused columns (FIXME: NEEDS TO BE TESTED)
    return events_df[[Event.id.key, Event.magnitude.key, Event.latitude.key, Event.longitude.key,
                      Event.depth_km.key, Event.time.key]].copy()


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


def dataframe_iter(url, evt_query_args, start, end,
                   timeout=15,
                   show_progress=False):
    '''Yields pandas dataframe(s) from the event url or file

    :param url: a valid url, a mappings string, or a local file (fdsn 'text' formatted)
    '''

    if islocalfile(url):
        evens_titer = events_iter_from_file(url)
        url = tofileuri(url)
    else:
        evens_titer = events_iter_from_url(_MAPPINGS.get(url, url),
                                           evt_query_args,
                                           start, end,
                                           timeout, show_progress)

    for url_, data in evens_titer:
        try:
            yield response2normalizeddf(url_, data, "event")
        except ValueError as exc:
            logger.warning(formatmsg("Discarding response", exc, url_))


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


def islocalfile(url):
    '''Returns whether url denotes a local file path, existing on the computer machine'''
    return url not in _MAPPINGS and os.path.isfile(url)


def events_iter_from_url(base_url, evt_query_args, start, end, timeout, show_progress=False):
    """
    Yields an iterator of tuples (url, data), where bith are strings denoting the
    url and the corresponding response body. The returned iterator has length > 1
    if the request was too large and had to be splitted
    """
    evt_query_args.setdefault('format', 'isf' if base_url == _MAPPINGS['isc'] else 'text')
    is_isf = evt_query_args['format'] == 'isf'

    start_iso = start.isoformat()
    end_iso = end.isoformat()
    # This should never happen but let's be safe: override start and end
    if 'start' in evt_query_args:
        evt_query_args.pop('start')
    evt_query_args['starttime'] = start_iso
    if 'end' in evt_query_args:
        evt_query_args.pop('end')
    evt_query_args['endtime'] = end_iso
    # assure that we have 'minmagnitude' and 'maxmagnitude' as mag parameters, if any:
    if 'minmag' in evt_query_args:
        minmag = evt_query_args.pop('minmag')
        if 'minmagnitude' not in evt_query_args:
            evt_query_args['minmagnitude'] = minmag
    if 'maxmag' in evt_query_args:
        maxmag = evt_query_args.pop('maxmag')
        if 'maxmagnitude' not in evt_query_args:
            evt_query_args['maxmagnitude'] = maxmag

    total_pbar_steps = _get_evtfreq_freq_mag_dist(evt_query_args)[2].sum()
    downloads = [(evt_query_args, total_pbar_steps)]

    with get_progressbar(show_progress, length=total_pbar_steps) as pbar:
        while downloads:
            evt_q_arg, steps = downloads.pop(0)
            url = urljoin(base_url, **evt_q_arg)
            try:
                raw_data, code, msg = urlread(url, decode='utf8', timeout=timeout,
                                              raise_http_err=True, wrap_exceptions=False)

                pbar.update(steps)
                if raw_data:
                    yield url, isfresponse2txt(raw_data) if is_isf else raw_data
                else:
                    logger.warning(formatmsg("Discarding request", msg, url))

            except Exception as exc:  # pylint: disable=broad-except
                # raise only if we do NOT have timeout or http err in (413, 504)
                if isinstance(exc, socket.timeout) or \
                        (isinstance(exc, HTTPError)
                         and exc.code in (413, 504)):  # pylint: disable=no-member
                    try:
                        downloads = _split_url(evt_q_arg) + downloads
                    except ValueError as verr:
                        raise FailedDownload(formatmsg("Unable to fetch events", verr,
                                                       url))
                else:
                    raise FailedDownload(formatmsg("Unable to fetch events", exc,
                                                   url))


def _split_url(evt_query_args):
    '''Splits the event query issued with the given `event_query_args` (dict)
    and returns a two-element list where each element is the tuple:
    (event_query_args, progressbar_incrementer)
    where the first item is a dict of an event query parameter, and the second
    is an integer representing how much a progress bar should be incremented
    if the query is successful (the number takes into account the theoretical
    frequency of events in the given query)
    '''
    minmag, deltamag, evtfreq_freq_mag_dist = _get_evtfreq_freq_mag_dist(evt_query_args)
    if len(evtfreq_freq_mag_dist) < 2:
        raise ValueError('maximum recursion depth reached, decrease '
                         'spatial-temporal bounds or magnitude range')
    half = evtfreq_freq_mag_dist.sum() / 2.0
    idx = 0
    while evtfreq_freq_mag_dist[:idx+1].sum() < half:
        idx += 1
    mag_half = minmag + idx * deltamag
    evt_query_args1 = dict(evt_query_args)
    evt_query_args2 = dict(evt_query_args)

    evt_query_args1['maxmagnitude'] = str(round(mag_half, 1))
    evt_query_args2['minmagnitude'] = str(round(mag_half, 1))

    return [(evt_query_args1, evtfreq_freq_mag_dist[:idx].sum()),
            (evt_query_args2, evtfreq_freq_mag_dist[idx:].sum())]


def _get_evtfreq_freq_mag_dist(evt_query_args):
    '''Returns the tuple minmag, step, func, where minmag is a float
    representing `func` first point (magnitude), step is the magnitude
    distance two adjacent points of `func`, and `func` is a a numpy array
    representing the theoretical events count from a given magnitude `mag`:
    ```
    f(mag) = 10 ** (9-mag)
    ```
    '''
    default_min, step, default_max = 0, .1, 9

    # create the function:
    ret = ((10 ** (default_max - np.arange(default_min, default_max, step))) + 0.5).astype(int)
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


def isfresponse2txt(nonempty_text, catalog='ISC', contributor='ISC'):
    sio = StringIO(nonempty_text)
    sio.seek(0)
    try:
        return '\n'.join('|'.join(_) for _ in isf2text_iter(sio, catalog, contributor))
    except Exception:  # pylint: disable=broad-except
        return ''


def isf2text_iter(isf_filep, catalog='', contributor=''):
    '''Yields lists of strings representing an event. The yielded list L
    can be passed to a DataFrame: pd.DataFrame[L]) and then converted with
    response2normalizeddf('file:///' + filepath, data, "event")
    '''

    # To have an idea of the text format parsed  See e.g.:
    # http://www.isc.ac.uk/fdsnws/event/1/query?starttime=2011-01-08T00:00:00&endtime=2011-01-08T01:00:00&format=isf

    buf = []
    event_line = 0
    reg2 = re.compile(' +')
    event_col_groups = eventmag_col_groups = None
    while True:
        line = isf_filep.readline()
        if not line or line in ('STOP', 'STOP\n'):  # last line
            if buf:  # remaining unparsed event
                yield buf
            break
        try:
            if line.startswith('Event '):
                if buf:  # remaining unparsed event
                    yield buf
                buf = [''] * 13
                event_line = 0
                elements = reg2.split(line)
                buf[0] = elements[1] if len(elements) > 1 else ''  # event_id
                buf[6] = catalog  # catalog
                buf[7] = contributor  # contributor
                buf[8] = buf[0]  # contributor id
                buf[12] = ' '.join(elements[2:]).strip()  # event location name
            elif event_line == 1 and buf and not event_col_groups:
                event_col_groups = _findgroups(line)
            elif event_line == 2 and buf and event_col_groups:
                elements = _findgroups(line, event_col_groups)
                # elements = reg2.split(line)
                dat, tme = elements[0], elements[1]
                if '/' in dat:
                    dat = dat.replace('/', '-')
                dtime = ''
                try:
                    dtime = strptime(dat + 'T' + tme).strftime('%Y-%m-%dT%H:%M:%S')
                except (TypeError, ValueError):
                    pass
                buf[1] = dtime  # time
                buf[2] = elements[4]  # latitude
                buf[3] = elements[5]  # longitude
                depth = elements[9]
                if depth and depth[-1] == 'f':
                    depth = depth[:-1]
                buf[4] = depth
                buf[5] = elements[17]  # author
            elif event_line == 4 and buf and not eventmag_col_groups:
                eventmag_col_groups = _findgroups(line)
            elif event_line == 5 and buf and eventmag_col_groups:
                elements = _findgroups(line, eventmag_col_groups)
                # we might have '[magtype]   [mag]' or simply '[mag]'
                # to be sure, try cast to float:
                mag, magtype = elements[0], ''
                try:
                    float(elements[0])
                except ValueError:
                    magtype, mag = re.split('\\s+', elements[0].strip())
                buf[9] = magtype  # magnitude type
                buf[10] = mag  # magnitude
                buf[11] = elements[3]  # mag author
                yield buf
                buf = []
            event_line += 1
        except IndexError:
            buf = []


def _findgroups(line, groups=None):
    ret = []
    if groups is None:
        ret = [[_.start(), _.end()] for _ in re.finditer(r'\b\w+\b', line)]
        return ret

    spaces = set([' ', '\t', '\n', '\r', '\f', '\v'])  # spaces chars in python re '\s'
    lastpos = len(line) - 1
    for grp in groups:
        stindex = grp[0]
        while stindex > 0 and line[stindex-1] not in spaces:
            stindex -= 1
        eindex = grp[1]
        while eindex < lastpos and line[eindex] not in spaces:
            eindex += 1
        ret.append(line[stindex:eindex].strip())
    return ret
