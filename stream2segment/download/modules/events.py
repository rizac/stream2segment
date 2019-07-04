'''
Download module forevents download

:date: Dec 3, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import map, next, zip, range, object

import os
import sys
import re
from io import open  # py2-3 compatible
from datetime import timedelta

import numpy as np
import pandas as pd

from stream2segment.utils import StringIO
from stream2segment.download.utils import dbsyncdf, FailedDownload, response2normalizeddf, \
    formatmsg, EVENTWS_MAPPING
from stream2segment.io.db.models import WebService, Event
from stream2segment.utils.url import urlread, socket, HTTPError
from stream2segment.utils import urljoin, strptime, get_progressbar

# logger: do not use logging.getLogger(__name__) but point to stream2segment.download.logger:
# this way we preserve the logging namespace hierarchy
# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial) when calling logging
# functions of stream2segment.download.utils:
from stream2segment.download import logger  # @IgnorePep8


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
        if isfile:
            msg = ("No event found. Check that the file is non emtpy and its content is valid")
        else:
            # we might have supplied a file that the program interprets as url.
            # Check first that we surely did not provide a file and set message
            # accordingly
            if url in EVENTWS_MAPPING or url.lower().startswith('http://') or \
                    url.lower().startswith('https://'):
                msg = ("No event found. Change your search parameters but check first "
                       "that the service is FDSN and its url is correct (e.g., no typos)")
            else:
                # we might have provided a file interpreted as url, set message
                # accordingly:
                msg = ("No event found. If you supplied a file, the file was not found: "
                       "check path and typos. Otherwise, change your search parameters but check "
                       "first that the service is FDSN and its url is correct (e.g., no typos)")
        raise FailedDownload(formatmsg(msg, url))

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
    if eventws_url in EVENTWS_MAPPING:
        ws_name = eventws_url
        eventws_url = EVENTWS_MAPPING[eventws_url]
    elif islocalfile(eventws_url):
        eventws_url = tofileuri(eventws_url)
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
        events_iter = events_iter_from_file(url, evt_query_args.get('format'))
        url = tofileuri(url)
    else:
        events_iter = events_iter_from_url(EVENTWS_MAPPING.get(url, url),
                                           evt_query_args,
                                           start, end,
                                           timeout, show_progress)

    for url_, data in events_iter:
        try:
            yield response2normalizeddf(url_, data, "event")
        except ValueError as exc:
            logger.warning(formatmsg("Discarding response", exc, url_))


def events_iter_from_file(file_path, format_=None):
    """Yields the tuple (filepath, events_data) from a file, which must exist on
    the local computer.

    :param foramt_: string: None will infer the format (isf or txt), otherwise
        it must be isf or txt. Note that support for isf is not fully complete
        (e.g., no comments allowed)
    """
    try:
        if format_ is None:
            format_ = 'isf' if is_isf(file_path) else 'txt'
        with open(file_path, encoding='utf-8') as opn:
            data = opn.read()
            if data and format_ == 'isf':
                data = isfresponse2txt(data, catalog='', contributor='')
            yield tofileuri(file_path), data
    except Exception as exc:
        raise FailedDownload(formatmsg("Unable to open events file", exc,
                                       file_path))


def is_isf(filepath):
    with open(filepath, 'r') as opn:
        return opn.readline().startswith('DATA_TYPE ')


def tofileuri(file_path):
    '''returns a file uri form the given file, basically file:///+basename(file_path)'''
    # https://en.wikipedia.org/wiki/File_URI_scheme#Format
    # return 'file:///' + os.path.abspath(os.path.normpath(file_path))
    return 'file:///' + os.path.basename(file_path)


def islocalfile(url):
    '''Returns whether url denotes a local file path, existing on the computer machine'''
    return url not in EVENTWS_MAPPING and os.path.isfile(url)


def events_iter_from_url(base_url, evt_query_args, start, end, timeout, show_progress=False):
    """
    Yields an iterator of tuples (url, data), where bith are strings denoting the
    url and the corresponding response body. The returned iterator has length > 1
    if the request was too large and had to be splitted
    """
    evt_query_args.setdefault('format', 'isf' if base_url == EVENTWS_MAPPING['isc'] else 'text')
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

    try:
        url = urljoin(base_url, **evt_query_args)
        result = _urlread(url, timeout, is_isf)
        if result is not None:
            yield url, result  # then result is the tuple (url, raw_data)
        else:
            logger.info("Request seems to be too large, splitting into "
                        "sub-requests")

            # the tricky part below is actually the progressbar part. It must:
            # 1 not be linear, thus advance "more" at lower magnitudes (where events are more
            #   dense)
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
                        result = _urlread(url, timeout, is_isf)
                        if result is not None:
                            # update pbar only if the end of the request equals
                            # the global end_iso (when recursion is done on time, it
                            # updates only on the first time chunk):
                            if evt_q_arg['endtime'] == end_iso:
                                steps = _get_freq_mag_distrib(evt_q_arg)[2].sum()
                                pbar.update(steps)
                            yield url, result  # then result is the tuple (url, raw_data)
                        else:
                            downloads.insert(i, evt_q_arg)
    except Exception as exc:
        raise FailedDownload(formatmsg("Unable to fetch events", exc, url))


def _urlread(url, timeout=None, is_isf=False):
    '''Wrapper around urlread but returns None if the url should be splitted
    becasue of a too long request'''
    try:
        raw_data, code, msg = urlread(url, decode='utf8', timeout=timeout,
                                      raise_http_err=True, wrap_exceptions=False)

        if raw_data is None:  # for safety (None has a special meaning)
            raw_data = ''

        return isfresponse2txt(raw_data) if raw_data and is_isf else raw_data

    except Exception as exc:  # pylint: disable=broad-except
        # raise only if we do NOT have timeout or http err in (413, 504)
        if isinstance(exc, socket.timeout) or \
                (isinstance(exc, HTTPError)
                 and exc.code in (413, 504)):  # pylint: disable=no-member
            return None
        raise


def _split_request(evt_query_args):
    '''Splits the event query issued with the given `event_query_args` (dict)
    and returns a two-element list:
    (event_query_args1, event_query_args2)
    of event query parameters (dicts) resulting from splitting `evt_query_args`
    '''
    minmag, deltamag, evtfreq_freq_mag_dist = _get_freq_mag_distrib(evt_query_args)
    if len(evtfreq_freq_mag_dist) < 2:  # max recusrion on magnitudes, split by time:
        start = strptime(evt_query_args['starttime'])
        end = strptime(evt_query_args['endtime'])
        days_diff = int((end-start).days / 2.0)
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
        while evtfreq_freq_mag_dist[:idx+1].sum() < half:
            idx += 1
        mag_half = minmag + idx * deltamag
        evt_query_args1 = dict(evt_query_args)
        evt_query_args2 = dict(evt_query_args)
        evt_query_args1['maxmagnitude'] = str(round(mag_half, 1))
        evt_query_args2['minmagnitude'] = str(round(mag_half, 1))

    return evt_query_args1, evt_query_args2


def _get_freq_mag_distrib(evt_query_args):
    '''Returns the tuple minmag, step, distrib, where minmag is a float
    representing `func` first point (magnitude), step is the magnitude
    distance two adjacent points of `distrib`, and `distrib` is a a numpy array
    (dtype=int) representing the theoretical events distribution from a given
    magnitude `mag`:
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
    '''Converts an isf formatted string into an FDSN text formatted string'''
    sio = StringIO(nonempty_text)
    sio.seek(0)
    try:
        return '\n'.join('|'.join(_) for _ in isf2text_iter(sio, catalog, contributor))
    except Exception as exc:  # pylint: disable=broad-except
        raise ValueError('Error reading .isf data: %s' % str(exc))


def isf2text_iter(isf_filep, catalog='', contributor=''):
    '''Yields lists of strings representing an event. The yielded list L
    can be passed to a DataFrame: pd.DataFrame[L]) and then converted with
    response2normalizeddf('file:///' + filepath, data, "event")
    For info see:
    http://www.isc.ac.uk/standards/isf/#ET

    Note that comments are not supported: events with comments will be discarded

    :param isf_filep: a file-like object which returns string (unicode) data
    '''

    # To have an idea of the text format parsed  See e.g.:
    # http://www.isc.ac.uk/fdsnws/event/1/query?starttime=2011-01-08T00:00:00&endtime=2011-01-08T01:00:00&format=isf

    buf = []
    origin_subblock_header = ("Date       Time        Err   RMS Latitude Longitude  "
                              "Smaj  Smin  Az Depth   Err Ndef Nsta Gap  mdist  Mdist "
                              "Qual   Author      OrigID")
    mag_subblock_header = "Magnitude  Err Nsta Author      OrigID"

    expects = 0
    eof = False
    while True:
        line = isf_filep.readline()
        eof = not line or line in ('STOP', 'STOP\n')
        if not eof and not line.strip():
            continue
        try:
            if eof or line.startswith('Event '):
                if buf:  # remaining unparsed event
                    yield buf
                if eof:
                    break
                buf = [''] * 13
                buf[0] = line[6:16].strip()  # event id
                buf[12] = line[16:].strip()  # event location name
                buf[6] = catalog  # catalog
                buf[7] = contributor  # contributor
                buf[8] = buf[0]  # contributor id
                expects = 1
            elif expects == 1 and buf:
                # use line.strip to ignore trailing newlines:
                if line.strip() == origin_subblock_header:
                    expects += 1
                else:
                    buf = []
            elif expects == 2 and buf:
                # elements = reg2.split(line)
                dat, tme = line[:10].strip(), line[11:22].strip()
                if '/' in dat:
                    dat = dat.replace('/', '-')
                dtime = ''
                try:
                    dtime = strptime(dat + 'T' + tme).strftime('%Y-%m-%dT%H:%M:%S')
                except (TypeError, ValueError):
                    pass
                buf[1] = dtime  # time
                buf[2] = line[36:44].strip()  # latitude
                buf[3] = line[45: 54].strip()  # longitude
                buf[4] = line[71:76].strip()  # depth
                buf[5] = line[118:127].strip()  # author
                expects += 1
            elif expects == 3 and buf:
                # use line.strip to ignore trailing newlines:
                if line.strip() == mag_subblock_header:
                    expects += 1
                else:
                    buf = []
            elif expects == 4 and buf:
                buf[9] = line[:5].strip()  # magnitude type
                buf[10] = line[6:10].strip()  # magnitude
                buf[11] = line[20:29].strip()  # mag author
                expects += 1
        except IndexError:
            buf = []
