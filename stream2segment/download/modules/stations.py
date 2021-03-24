"""
Stations (inventory) download functions

:date: Dec 3, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import zip

from datetime import datetime
from io import BytesIO
import gzip
import zipfile
import zlib
import bz2

import pandas as pd

from stream2segment.download.db import DataCenter, Station, Segment
from stream2segment.download.utils import read_async, DbExcLogger, formatmsg, url2str,\
    err2str
from stream2segment.io.cli import get_progressbar
from stream2segment.io.db.pdsql import DbManager
from stream2segment.download.url import Request  # handles py2,3 compatibility

# logger: do not use logging.getLogger(__name__) but point to
# stream2segment.download.logger: this way we preserve the logging namespace
# hierarchy
# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial) when
# calling logging functions of stream2segment.download.utils:
from stream2segment.download import logger  # @IgnorePep8
from stream2segment.download.url import get_host


def _get_sta_request(datacenter_url, network, station, start_time, end_time):
    """Return a Request object from the given station arguments to download the
    inventory xml
    """
    stime_iso = start_time.isoformat()

    # we need a endtime (ingv does not accept * as last param)
    if pd.isnull(end_time):
        # note :pd.isnull(None) is true, as well as pd.isnull(float('nan'))
        etime_iso = datetime.utcnow().isoformat()
    else:
        etime_iso = end_time.isoformat()

    post_data = " ".join("*" if not x else x for x in
                         [network, station, "*", "*", stime_iso, etime_iso])
    return Request(url=datacenter_url, data="level=response\n{}".format(post_data).
                   encode('utf8'))


def save_inventories(session, stations_df, max_thread_workers, timeout,
                     download_blocksize, db_bufsize, show_progress=False):
    """Save inventories. stations_df must not be empty (not checked here)"""

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

        reader = read_async(iterable,
                            urlkey=lambda obj: _get_sta_request(*obj[1:]),
                            max_workers=max_thread_workers,
                            blocksize=download_blocksize, timeout=timeout,
                            raise_http_err=True)

        for obj, result, exc, request in reader:
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
                    dfr = pd.DataFrame({Station.id.key: [sta_id],
                                        Station.inventory_xml.key: [compress(data)]})
                    dbmanager.add(dfr)

    dbmanager.close()

    return downloaded, empty, errors


def compress(bytestr, compression='gzip', compresslevel=9):
    """Compress `bytestr` returning a new compressed byte sequence

    :param bytestr: (string) a sequence of bytes to be compressed
    :param compression: String, either ['bz2', 'zlib', 'gzip', 'zip'. Default: 'gzip']
        The compression library to use (after serializing `obj` with the given format)
        on the serialized data. If None or empty string, no compression is applied, and
        `bytestr` is returned as it is
    :param compresslevel: integer (9 by default). Ignored if `compression` is None,
        empty or 'zip' (the latter does not accept this argument), this parameter
        controls the level of compression; 1 is fastest and produces the least
        compression, and 9 is slowest and produces the most compression
    """
    if compression == 'bz2':
        return bz2.compress(bytestr, compresslevel=compresslevel)
    elif compression == 'zlib':
        return zlib.compress(bytestr, compresslevel)
    elif compression:
        sio = BytesIO()
        if compression == 'gzip':
            with gzip.GzipFile(mode='wb', fileobj=sio,
                               compresslevel=compresslevel) as gzip_obj:
                gzip_obj.write(bytestr)
                # Note: DO NOT return sio.getvalue() WITHIN the with statement,
                # the gzip file obj needs to be closed first. FIXME: ref?
        elif compression == 'zip':
            # In this case, use the compress argument to ZipFile to compress the data,
            # since writestr() does not take compress as an argument. See:
            # https://pymotw.com/2/zipfile/#writing-data-from-sources-other-than-files
            with zipfile.ZipFile(sio, 'w', compression=zipfile.ZIP_DEFLATED) as zip_obj:
                zip_obj.writestr("x", bytestr)  # first arg must be a nonempty str
        else:
            raise ValueError("compression '%s' not in ('gzip', 'zlib', 'bz2', 'zip')" %
                             str(compression))

        return sio.getvalue()

    return bytestr


def query4inventorydownload(session, force_update):
    """Return an sql-alchemy Query yielding the stations for downloading their
    inventory xml. The query is a list of tuples
    (station_id, network, station, station_url)

    :param session: the sql-alchemy session
    :param force_update: boolean, if True an element E of the returned list is
        a tuple representing any station which has at least one segment with
        data. If False, each E represents a station which has at least one
        segment with data, AND does not have an inventory saved yet
    :return: a query yielding the tuples:
        ```
        (Station.id, Station.network, Station.station, DataCenter.station_url,
        Station.start_time, Station.end_time)
        ```
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


class InventoryLogger(set):
    """Class handling inventory errors and logging only once per error type
    and datacenter to avoid polluting the log file/stream with hundreds of
    megabytes"""

    def warn(self, request, exc):
        """Issue a logger.warn if the given error is not already reported

        :param request: the Request object
        :pram exc: the reported Exception or string message
        """
        url = get_host(request)
        item = (url, err2str(exc))  # use err2str to uniquely identify exc
        if item not in self:
            if not self:
                logger.warning('Detailed inventory download errors (showing '
                               'only first of each type per data center):')
            self.add(item)
            request_str = url2str(request)
            logger.warning(formatmsg("Inventory download error", exc, request_str))
