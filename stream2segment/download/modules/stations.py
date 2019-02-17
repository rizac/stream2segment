'''
Download module for stations download

:date: Dec 3, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import map, next, zip, range, object

import logging
from datetime import datetime

import pandas as pd

from stream2segment.io.db.models import DataCenter, Station, Segment
from stream2segment.download.utils import read_async, DbExcLogger, formatmsg, url2str,\
    err2str
from stream2segment.utils import get_progressbar
from stream2segment.io.db.pdsql import DbManager
from stream2segment.io.utils import dumps_inv

from stream2segment.utils.url import Request  # this handles py2and3 compatibility

# logger: do not use logging.getLogger(__name__) but point to stream2segment.download.logger:
# this way we preserve the logging namespace hierarchy
# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial) when calling logging
# functions of stream2segment.download.utils:
from stream2segment.download import logger  # @IgnorePep8
from stream2segment.utils.url import get_host


def _get_sta_request(datacenter_url, network, station, start_time, end_time):
    """
    returns a Request object from the given station arguments to download the inventory xml"""
    # we need a endtime (ingv does not accept * as last param)
    # note :pd.isnull(None) is true, as well as pd.isnull(float('nan')) and so on
    et = datetime.utcnow().isoformat() if pd.isnull(end_time) else end_time.isoformat()
    post_data = " ".join("*" if not x else x for x in[network, station, "*", "*",
                                                      start_time.isoformat(), et])
    return Request(url=datacenter_url, data="level=response\n{}".format(post_data).encode('utf8'))


def save_inventories(session, stations_df, max_thread_workers, timeout,
                     download_blocksize, db_bufsize, show_progress=False):
    """Save inventories. Stations_df must not be empty (this is not checked for)"""

    inv_logger = InventoryLogger()

    downloaded, errors, empty = 0, 0, 0

    db_exc_logger = DbExcLogger([Station.id.key, Station.network.key,
                                 Station.station.key, Station.start_time.key])

    dbmanager = DbManager(session, Station.id,
                          update=[Station.inventory_xml.key],
                          buf_size=db_bufsize,
                          oninsert_err_callback=db_exc_logger.failed_insert,
                          onupdate_err_callback=db_exc_logger.failed_update)

    with get_progressbar(show_progress, length=len(stations_df)) as pbar:
        iterable = zip(stations_df[Station.id.key],
                       stations_df[DataCenter.station_url.key],
                       stations_df[Station.network.key],
                       stations_df[Station.station.key],
                       stations_df[Station.start_time.key],
                       stations_df[Station.end_time.key])
        for obj, result, exc, request in read_async(iterable,
                                                    urlkey=lambda obj: _get_sta_request(*obj[1:]),
                                                    max_workers=max_thread_workers,
                                                    blocksize=download_blocksize, timeout=timeout,
                                                    raise_http_err=True):
            pbar.update(1)
            sta_id = obj[0]
            if exc:
                inv_logger.warn(request, exc)
                errors += 1
            else:
                data, code, msg = result  # @UnusedVariable
                if not data:
                    inv_logger.warn(request, "empty response")
                    empty += 1
                else:
                    downloaded += 1
                    dbmanager.add(pd.DataFrame({Station.id.key: [sta_id],
                                                Station.inventory_xml.key: [dumps_inv(data)]}))

    dbmanager.close()

    return downloaded, empty, errors


def query4inventorydownload(session, force_update):
    '''Returns an sql-alchemy Query yielding the stations for downloading their inventory xml
    The query is a list of tuples (station_id, network, station, station_url)

    :param session: the sql-alchemy session
    :param force_update: boolean, if True an element E of the returned list is a tuple representing
        any station which has at least one segment with data. If False, each E represents
        a station which has at least one segment with data, AND does not have an inventory saved
        yet
    :return: a query yielding the tuples:
    ```(Station.id, Station.network, Station.station, DataCenter.station_url,
        Station.start_time, Station.end_time)```
    '''
    qry = session.query(Station.id, Station.network, Station.station, DataCenter.station_url,
                        Station.start_time, Station.end_time).join(Station.datacenter)

    if force_update:
        qry = qry.filter(Station.segments.any(Segment.has_data))  # @UndefinedVariable
    else:
        qry = qry.filter((~Station.has_inventory) &  # pylint:disable=invalid-unary-operand-type
                         (Station.segments.any(Segment.has_data)))  # @UndefinedVariable

    return qry


class InventoryLogger(set):
    '''A class handling inventory errors and logging only once per error type
    and datacenter to avoid polluting the log file/stream with hundreds of megabytes'''

    def warn(self, request, exc):
        '''issues a logger.warn if the given error is not already reported

        :param request: the Request object
        :pram exc: the reported Exception or string message
        '''
        url = get_host(request)
        item = (url, err2str(exc))  # use err2str to uniquely identify exc
        if item not in self:
            if not self:
                logger.warning('Detailed inventory download errors '
                               '(showing only first of each type per data center):')
            self.add(item)
            request_str = url2str(request)
            logger.warning(formatmsg("Inventory download error", exc, request_str))
