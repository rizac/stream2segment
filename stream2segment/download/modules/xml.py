"""
StationsXML download
"""
import logging
from typing import Optional

from stream2segment.io.utils import get_progressbar
from stream2segment.io.db.pdsql import (
    Engine, execute_sql, get_col_max, insert, update, select
)
from stream2segment.io.db.models import (
    WebService, Segment, Channel, StationXML, Event, QuakeML
)
from stream2segment.download.url import read_urls, get_host, responses
from stream2segment.download.modules.utils import (
    IdOnceLogFilter, fdsn_url_qs, fdsn_url
)


# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial):
logger = logging.getLogger(__name__)


def save_stationxml(
    *,
    engine: Engine,
    max_download_concurrency: int | None,
    download_timeout,
    download_blocksize,
    show_progress=False
):
    """Save StationXML data. stations_df must not be empty (not checked here)"""
    if max_download_concurrency is None:
        max_download_concurrency = 4

    stmt = (
        select(
            Channel.network_code,
            Channel.station_code,
            Segment.webservice_id,
            WebService.url
        )
        .join(WebService, Segment.webservice_id == WebService.id)
        .join(Segment, Segment.channel_id == Channel.id)
        .where(Segment.stationxml_id.is_(None))
        .distinct()
    )

    log_once_filter: Optional[IdOnceLogFilter] = None  # lazily created if needed
    downloaded, saved, errors = 0, 0, 0

    with engine.connect() as conn:
        rows = conn.execute(stmt).fetchall()

    insert_stmt = insert(StationXML)
    stationxml_id = get_col_max(engine, StationXML.id)
    cache: dict[str, tuple[str, str, int]] = {}

    if len(rows) > 0:

        downloaded = len(rows)

        def url_builder(net, sta, ws_id, ws_url):
            """build url (str) from each item yielded by the previous iterable"""
            url = fdsn_url_qs(
                fdsn_url(ws_url, new_service='station'),
                net=net, sta=sta, level='response'
            )
            cache.setdefault(url, (net, sta, ws_id))
            return url

        with (get_progressbar(len(rows) if show_progress else 0) as pbar):

            reader = read_urls(
                (url_builder(*row) for row in rows),
                max_concurrency=max_download_concurrency,
                timeout=download_timeout,
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
                        stationxml_id += 1
                        update_stmt = update(
                            Channel
                        ).where(
                            (Channel.network_code==net) &
                            (Channel.station_code==sta) &
                            (Channel.webservice_id==ws_id)
                        ).values({
                            Channel.stationxml_id: stationxml_id
                        })
                        saved += sum(
                            1 for _ in execute_sql(
                                engine,
                                [insert_stmt, update_stmt],
                                [{
                                    'data': response.data,
                                    'id': stationxml_id
                                }]
                            )
                        )
                    except Exception:
                        pass

        if log_once_filter is not None:
            logger.removeFilter(log_once_filter)

    return downloaded, saved, errors


def save_quakeml(
    *,
    engine: Engine,
    max_download_concurrency: int | None,
    download_timeout,
    download_blocksize,
    show_progress=False
):
    """Save QuakeML data. stations_df must not be empty (not checked here)"""
    if max_download_concurrency is None:
        max_download_concurrency = 4

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

    insert_stmt = insert(QuakeML)
    cache: dict[str, int] = {}

    downloaded = len(rows)

    def url_builder(db_ev_id, cat_ev_id, ws_url):
        """build url (str) from each item yielded by the previous iterable"""
        url = fdsn_url_qs(ws_url, eventid=cat_ev_id, format='xml')
        cache.setdefault(url, db_ev_id)
        return url

    with get_progressbar(len(rows) if show_progress else 0) as pbar:

        reader = read_urls(
            (url_builder(*row) for row in rows),
            max_concurrency=max_download_concurrency,
            timeout=download_timeout,
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
                    saved += sum(
                        1 for _ in execute_sql(
                            engine,
                            [insert_stmt],
                            [{
                                'id': db_ev_id,
                                'data': response.data
                            }]
                        )
                    )
                except Exception:  # noqa
                    pass

    if log_once_filter is not None:
        logger.removeFilter(log_once_filter)

    return downloaded, saved, errors
