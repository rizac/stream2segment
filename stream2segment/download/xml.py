"""
StationsXML download
"""
import logging
from typing import Optional
import pandas as pd
from sqlalchemy.exc import IntegrityError

from stream2segment.download.channels import url_col, net_col, sta_col
from stream2segment.io.utils import get_progressbar
from stream2segment.io.db.pdsql import (
    Engine, executemany, get_col_max, insert, update, select, fetch_df, get_row_count
)
from stream2segment.io.db.models import (
    WebService, Segment, Channel, StationXML, Event, QuakeML
)
from stream2segment.download.url import read_urls, get_host, responses
from stream2segment.download.utils import (
    IdOnceLogFilter, fdsn_url_qs, fdsn_url
)
from sqlalchemy import func, exists

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

    staxml_id_col = Channel.stationxml_id.key
    stmt = (
        select(
            Channel.network_code,
            Channel.station_code,
            Channel.data_webservice_id,
            func.max(Channel.stationxml_id).label(staxml_id_col),
            WebService.url,
        )
        .join(WebService, WebService.id == Channel.data_webservice_id)
        .join(Segment, Segment.channel_id == Channel.id)  # <- inner join (*)
        .group_by(
            Channel.network_code,
            Channel.station_code,
            Channel.data_webservice_id,
        )
        .having(Channel.stationxml_id.is_(None))
    )
    # (*) only channels with at least one matching Segment row are included

    downloaded, saved, errors = 0, 0, 0

    with engine.connect() as conn:
        total = conn.execute(
            select(func.count()).select_from(stmt.subquery())
        ).scalar_one()
        if total < 1:
            return downloaded, saved, errors

    log_once_filter: Optional[IdOnceLogFilter] = None  # lazily created if needed

    # with engine.connect() as conn:
    #     rows = conn.execute(stmt).fetchall()

    db_stationxml_id = get_col_max(engine, StationXML.id)
    cache: dict[str, tuple[str, str, int, int | None]] = {}
    ws_id_col = Channel.data_webservice_id.key

    def url_builder(net, sta, ws_id, ws_url, sta_id):
        """build url (str) from each item yielded by the previous iterable"""
        url = fdsn_url_qs(
            fdsn_url(ws_url, new_service='station'),
            net=net, sta=sta, level='response'
        )
        cache.setdefault(url, (net, sta, ws_id, None if pd.isna(sta_id) else sta_id))
        return url

    with (
        get_progressbar(total if show_progress else 0) as pbar, engine.begin() as conn  # noqa
    ):

        for dfr in fetch_df(engine, stmt):
            dfr[url_col] = dfr[url_col].astype('category')
            dfr[staxml_id_col] = dfr[staxml_id_col].astype('Int64')  # int with Nulls

            rows = zip(
                dfr[net_col],
                dfr[sta_col],
                dfr[ws_id_col],
                dfr[url_col],
                dfr[staxml_id_col]
            )

            for response in read_urls(
                (url_builder(*row) for row in rows),
                max_concurrency=max_download_concurrency,
                timeout=download_timeout,
                blocksize=download_blocksize
            ):
                pbar.update(1)
                url = response.request
                if not response.is_ok or response.status_code == 204:
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
                    if url not in cache:
                        # log wanr?
                        continue
                    (net, sta, ws_id, sta_id) = cache.pop(url)
                    try:
                        with conn.begin_nested() as conn_nested:
                            if sta_id is not None:
                                conn.execute(
                                    update(StationXML).where(
                                        (StationXML.id==sta_id)
                                    ).values({
                                        StationXML.data: response.data
                                    })
                                )
                            else:
                                db_stationxml_id += 1
                                sta_id = db_stationxml_id
                                conn.execute(insert(StationXML), {
                                    'id': sta_id, 'data': response.data
                                })

                            conn.execute(
                                update(  # update_channel_fk
                                    Channel
                                ).where(
                                    Channel.network_code == net,
                                    Channel.station_code == sta,
                                    Channel.data_webservice_id == ws_id
                                ).values({
                                    Channel.stationxml_id: sta_id
                                })
                            )
                            saved += 1
                    except IntegrityError as e:
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
            if not response.is_ok or response.status_code == 204:
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
                        1 for _ in executemany(
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
