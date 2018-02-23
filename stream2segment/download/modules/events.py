'''
Download module forevents download

:date: Dec 3, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import map, next, zip, range, object

import logging
from datetime import timedelta, datetime
import dateutil

import pandas as pd

from stream2segment.download.utils import dbsyncdf, QuitDownload, response2normalizeddf
from stream2segment.io.db.models import WebService, Event
from stream2segment.utils.msgs import MSG
from stream2segment.utils.url import urlread, URLException
from stream2segment.utils import urljoin

# logger: do not use logging.getLogger(__name__) but point to stream2segment.download.logger:
# this way we preserve the logging namespace hierarchy
# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial) when calling logging
# functions of stream2segment.download.utils:
from stream2segment.download import logger  # @IgnorePep8


def get_events_df(session, eventws_url, db_bufsize, **args):
    """
        Returns the events from an event ws query. Splits the results into smaller chunks
        (according to 'start' and 'end' parameters, if they are not supplied in **args they will
        default to `datetime(1970, 1, 1)` and `datetime.utcnow()`, respectively)
        In case of errors just raise, the caller is responsible of displaying messages to the
        logger, which is used in this function only for all those messages which should not stop
        the program
    """
    eventws_id = session.query(WebService.id).filter(WebService.url == eventws_url).scalar()
    if eventws_id is None:  # write url to table
        data = [("event", eventws_url)]
        df = pd.DataFrame(data, columns=[WebService.type.key, WebService.url.key])
        df = dbsyncdf(df, session, [WebService.url], WebService.id, buf_size=db_bufsize)
        eventws_id = df.iloc[0][WebService.id.key]

    url = urljoin(eventws_url, format='text', **args)
    ret = []
    try:
        datalist = get_events_list(eventws_url, **args)
    except ValueError as exc:
        raise QuitDownload(exc)

    if len(datalist) > 1:
        logger.info(MSG("Request was split into sub-queries, aggregating the results",
                        "Original request entity too large", url))

    for data, msg, url in datalist:
        if not data and msg:
            logger.warning(MSG("Discarding request", msg, url))
        elif data:
            try:
                events_df = response2normalizeddf(url, data, "event")
                ret.append(events_df)
            except ValueError as exc:
                logger.warning(MSG("Discarding response", exc, url))

    if not ret:  # pd.concat below raise ValueError if ret is empty:
        raise QuitDownload(Exception(MSG("",
                                         "No events found. Check input config. or log for details",
                                         url)))

    events_df = pd.concat(ret, axis=0, ignore_index=True, copy=False)
    events_df[Event.webservice_id.key] = eventws_id
    events_df = dbsyncdf(events_df, session,
                         [Event.event_id, Event.webservice_id], Event.id, buf_size=db_bufsize,
                         cols_to_print_on_err=[Event.event_id.key])

    # try to release memory for unused columns (FIXME: NEEDS TO BE TESTED)
    return events_df[[Event.id.key, Event.magnitude.key, Event.latitude.key, Event.longitude.key,
                     Event.depth_km.key, Event.time.key]].copy()


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

