"""
StationsXML download
"""
import logging
from typing import Optional

from sqlalchemy import select, Engine

from stream2segment.io.cli import get_progressbar
from stream2segment.io.db.pdsql import (
    create_update_statement, create_insert_statement, execute_sql, get_col_max
)
from stream2segment.io.db.models import (
    WebService, Segment, Channel, StationXML, Event, QuakeML
)
from stream2segment.download.url import read_async, get_host, responses
from stream2segment.download.modules.utils import (IdOnceLogFilter, fdsn_url_qs)

# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial):
logger = logging.getLogger(__name__)


def save_stationxml(
    engine: Engine,
    max_thread_workers,
    timeout,
    download_blocksize,
    show_progress=False
):
    """Save StationXML data. stations_df must not be empty (not checked here)"""

    stmt = (
        select(
            Channel.network_code,
            Channel.station_code,
            Channel.webservice_id,
            WebService.url
        )
        .join(WebService, Channel.webservice_id == WebService.id)
        .join(Segment, Segment.channel_id == Channel.id)
        .where(Channel.stationxml_id.is_(None))
        .distinct()
    )

    log_once_filter: Optional[IdOnceLogFilter] = None  # lazily created if needed
    downloaded, saved, errors = 0, 0, 0

    with engine.connect() as conn:
        rows = conn.execute(stmt).fetchall()

    insert_stmt = [create_insert_statement(StationXML)]
    stationxml_id = get_col_max(engine, StationXML.id)
    cache: dict[str, tuple[str, str, int]] = {}

    if len(rows) > 0:

        downloaded = len(rows)

        def url_builder(net, sta, ws_id, ws_url):
            """build url (str) from each item yielded by the previous iterable"""
            url = fdsn_url_qs(ws_url, net=net, sta=sta, level='response')
            cache.setdefault(url, (net, sta, ws_id))
            return url

        with get_progressbar(len(rows) if show_progress else 0) as pbar:

            reader = read_async(
                (url_builder(*row) for row in rows),
                timeout=timeout,
                max_workers=max_thread_workers,
                blocksize=download_blocksize
            )

            for response in reader:
                pbar.update(1)
                url = response.request
                if not response.is_ok or response.status == 204:
                    if log_once_filter is None:  # create lazily
                        log_once_filter = IdOnceLogFilter()
                        logger.addFilter(log_once_filter)
                        logger.warning(
                            "StationXML download errors\n"
                            "(shown once per (URL domain, error type) combination)"
                        )
                    msg = responses.get(response.status, "Unknown error")
                    errors += 1
                    logger.warning(
                        url,msg, extra={'ID': (get_host(url), msg)}
                    )
                else:
                    try:
                        (net, sta, ws_id) = cache.pop(url)
                        update_stmt = create_update_statement(
                            Channel,
                            (
                                (Channel.network_code==net) &
                                (Channel.station_code==sta) &
                                (Channel.webservice_id==ws_id)
                            ),
                            Channel.stationxml_id
                        )
                        stationxml_id += 1
                        execute_sql(
                            engine,
                            [insert_stmt, update_stmt],
                            {
                                'data': response.data,
                                'id': stationxml_id,
                                'stationxml_id': stationxml_id
                            }
                        )
                        saved += 1
                    except Exception:
                        pass

        if log_once_filter is not None:
            logger.removeFilter(log_once_filter)

    return downloaded, saved, errors


def save_quakeml(engine: Engine, max_thread_workers, timeout,
                 download_blocksize, show_progress=False):
    """Save QuakeML data. stations_df must not be empty (not checked here)"""

    stmt = (
        select(
            Event.id,
            Event.eventid,
            WebService.url
        )
        .join(WebService, Event.webservice_id == WebService.id)
        .join(Segment, Segment.event_id == Event.id)
        #.where(Channel.stationxml_id.is_(None))
        .distinct()
    )

    log_once_filter: Optional[IdOnceLogFilter] = None  # lazily created if needed
    downloaded, saved, errors = 0, 0, 0

    with engine.connect() as conn:
        rows = conn.execute(stmt).fetchall()

    insert_stmt = create_insert_statement(QuakeML)
    cache: dict[str, int] = {}

    if len(rows) > 0:

        downloaded = len(rows)

        def url_builder(db_ev_id, cat_ev_id, ws_url):
            """build url (str) from each item yielded by the previous iterable"""
            url = fdsn_url_qs(ws_url, eventid=cat_ev_id, format='xml')
            cache.setdefault(url, db_ev_id)
            return url

        with get_progressbar(len(rows) if show_progress else 0) as pbar:

            reader = read_async(
                (url_builder(*row) for row in rows),
                timeout=timeout,
                max_workers=max_thread_workers,
                blocksize=download_blocksize
            )

            for response in reader:
                pbar.update(1)
                url = response.request
                if not response.is_ok or response.status == 204:
                    if log_once_filter is None:  # create lazily
                        log_once_filter = IdOnceLogFilter()
                        logger.addFilter(log_once_filter)
                        logger.warning(
                            "QuakeML download errors\n"
                            "(shown once per (URL domain, error type) combination)"
                        )
                    msg = responses.get(response.status, "Unknown error")
                    errors += 1
                    logger.warning(
                        url,msg, extra={'ID': (get_host(url), msg)}
                    )
                else:
                    try:
                        db_ev_id = cache.pop(url)
                        execute_sql(
                            engine,
                            [insert_stmt],
                            {
                                'id': db_ev_id,
                                'data': response.data
                            }
                        )
                        saved += 1
                    except Exception:
                        pass

        if log_once_filter is not None:
            logger.removeFilter(log_once_filter)

    return downloaded, saved, errors
