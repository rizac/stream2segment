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

from stream2segment.io.db.models import DataCenter, Station
from stream2segment.download.utils import read_async, handledbexc
from stream2segment.utils.msgs import MSG
from stream2segment.utils import get_progressbar
from stream2segment.io.db.pdsql import DbManager
from stream2segment.io.utils import dumps_inv

# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#aliased-imports):
from future import standard_library
standard_library.install_aliases()
from urllib.parse import urlparse  # @IgnorePep8
from urllib.request import Request  # @IgnorePep8


# logger: do not use logging.getLogger(__name__) but point to stream2segment.download.logger:
# this way we preserve the logging namespace hierarchy
# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial) when calling logging
# functions of stream2segment.download.utils:
from stream2segment.download import logger  # @IgnorePep8


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

    _msg = "Unable to save inventory (station id=%d)"

    downloaded, errors, empty = 0, 0, 0
    cols_to_log_on_err = [Station.id.key, Station.network.key, Station.station.key,
                          Station.start_time.key]
    dbmanager = DbManager(session, Station.id,
                          update=[Station.inventory_xml.key],
                          buf_size=db_bufsize,
                          oninsert_err_callback=handledbexc(cols_to_log_on_err, update=False),
                          onupdate_err_callback=handledbexc(cols_to_log_on_err, update=True))

    with get_progressbar(show_progress, length=len(stations_df)) as bar:
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
            bar.update(1)
            sta_id = obj[0]
            if exc:
                logger.warning(MSG(_msg, exc, request), sta_id)
                errors += 1
            else:
                data, code, msg = result  # @UnusedVariable
                if not data:
                    empty += 1
                    logger.warning(MSG(_msg, "empty response", request), sta_id)
                else:
                    downloaded += 1
                    dbmanager.add(pd.DataFrame({Station.id.key: [sta_id],
                                                Station.inventory_xml.key: [dumps_inv(data)]}))

    dbmanager.close()
    return downloaded, empty, errors
