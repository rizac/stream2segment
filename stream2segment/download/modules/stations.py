"""
Stations (inventory) download functions

:date: Dec 3, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
from datetime import datetime
import logging
from datetime import timedelta
from urllib.request import Request

import pandas as pd

from stream2segment.io.cli import get_progressbar
from stream2segment.io.db.pdsql import DbManager, dbquery2df
from stream2segment.download.db.models import DataCenter, Station, Segment
from stream2segment.download.url import read_async
from stream2segment.download.modules.utils import DbExcLogger, OneTimeLogger, compress

# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial):
logger = logging.getLogger(__name__)


def _get_sta_request(datacenter_url, network, station, start_time, end_time):
    """Return a Request object from the given station arguments to download the
    StationXML
    """
    # fix bug of ncedc and scedc whereby dates exactly on the start are not returned.
    # Adding 1s to the start time is heavily hacky but it fixes the problem easily:
    one_sec = timedelta(seconds=1)
    stime_iso = (start_time + one_sec).isoformat()

    # we need a endtime (ingv does not accept * as last param)
    if pd.isnull(end_time):  # pd.isnull is more general (e.g. NAT nan return true)
        end_time = datetime.utcnow().replace(microsecond=0)
    else:
        # etiem is given, thus it might fall exactly on a "new" inventory that we do not
        # want. Decrease etime to be sure:
        end_time -= one_sec

    etime_iso = end_time.isoformat()

    post_data = " ".join("*" if not x else x for x in
                         [network, station, "*", "*", stime_iso, etime_iso])
    return Request(url=datacenter_url, data="level=response\n{}".format(post_data).
                   encode('utf8'))


def get_station_df_for_inventory_download(session, update_metadata):
    """
    Return a Pandas DataFrame with all station information required to download
    the necessary StationXML from the data stored in the DB mapped by the
    given session object. See `save_inventories`

    :param session: an SQLAlchemy session object
    :param update_metadata: boolean, if True an element E of the returned list is
        a tuple representing any station which has at least one segment with
        data. If False, each E represents a station which has at least one
        segment with data, AND does not have an inventory saved yet
    :return:  a DataFrame that can be used to download inventories needed by the DB
        underlying the given session, with columns:
         (Station.id, Station.network, Station.station, DataCenter.station_url,
        Station.start_time, Station.end_time)
    """
    sta_df = dbquery2df(_query4inventorydownload(session, update_metadata))
    # set 'station_url' as categorical (might save some space):
    sta_url_key = DataCenter.station_url.key
    sta_df[sta_url_key] = sta_df[sta_url_key].astype('category')
    # sort values in order to 1. download first most recent events and 2: shuffle
    # datacenters and try to diversify the requests to different URLs:
    sta_df.sort_values(by=Station.start_time.key, ascending=False, inplace=True)
    return sta_df


def _query4inventorydownload(session, force_update):
    """Return an sql-alchemy Query yielding the stations for downloading their
    inventory xml. Each station is returned as tuple (denoting the station requested
    values).
    See `get_station_df_for_inventory_download` for details
    """
    qry = session.query(Station.id, Station.network, Station.station,
                        DataCenter.station_url, Station.start_time,
                        Station.end_time).join(Station.datacenter)

    if force_update:
        qry = qry.filter(Station.segments.any(Segment.has_data))  # noqa
    else:
        qry = qry.filter((~Station.has_inventory) &  # noqa
                         (Station.segments.any(Segment.has_data)))  # @noqa

    return qry


def save_stationxml(session, stations_df, max_thread_workers, timeout,
                    download_blocksize, db_bufsize, show_progress=False):
    """Save StationXML data. stations_df must not be empty (not checked here)"""

    logger_header = "StationXML"
    inv_logger = OneTimeLogger(logger_header)

    downloaded, errors, empty = 0, 0, 0

    db_exc_logger = DbExcLogger([Station.id.key, Station.network.key,
                                 Station.station.key, Station.start_time.key])

    dbmanager = DbManager(session, Station.id,
                          update=[Station.stationxml.key],
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

        reader = read_async(iterable,
                            urlkey=lambda obj: _get_sta_request(*obj[1:]),
                            max_workers=max_thread_workers,
                            blocksize=download_blocksize, timeout=timeout)

        for obj, request, data, exc, status_code in reader:
            pbar.update(1)
            sta_id = obj[0]
            if exc:
                inv_logger.warn(request, exc)
                errors += 1
            else:
                if not data:
                    inv_logger.warn(request, "empty response")
                    empty += 1
                else:
                    downloaded += 1
                    dfr = pd.DataFrame({Station.id.key: [sta_id],
                                        Station.stationxml.key: [compress(data)]})
                    dbmanager.add(dfr)

    dbmanager.close()

    return downloaded, empty, errors
