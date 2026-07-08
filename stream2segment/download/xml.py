"""
StationsXML download
"""
from __future__ import annotations
import logging
from collections.abc import Iterable, Callable
from datetime import datetime, UTC

from sqlalchemy.exc import IntegrityError

from stream2segment.download.segments import DownloadStats
from stream2segment.io.utils import get_progressbar
from stream2segment.io.db.pdsql import (
    Engine, get_col_max, insert, update, select, fetch_df
)
from stream2segment.io.db.models import (
    WebService, Segment, Channel, StationXML, Event, QuakeML
)
from stream2segment.download.url import build_and_read_urls, get_host, Response
from stream2segment.download.utils import fdsn_url_qs, fdsn_url
from sqlalchemy import func, exists, Select, case

# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial):
logger = logging.getLogger(__name__)


def save_stationxml(
    *,
    engine: Engine,
    max_download_concurrency_per_domain: int | None,
    download_timeout,
    download_blocksize,
    show_progress=False
) -> DownloadStats:
    """Save StationXML data. stations_df must not be empty (not checked here)"""

    stmt = (
        select(
            Channel.network_code,
            Channel.station_code,
            Channel.data_webservice_id,
            Channel.station_id
            # func.max(Channel.stationxml_id).label(staxml_id_col),
        )
        .distinct()
        .select_from(Segment)
        .join(Event, Segment.event_id == Event.id)  # (*) inner join
        .join(Channel, Segment.channel_id == Channel.id)  # (*) inner join
        .join(StationXML, Segment.station_id == StationXML.id)  # (*) inner join
        .where(
            StationXML.last_updated.is_(None) |
            StationXML.data.is_(None) |
            (Event.time > StationXML.last_updated)
        )
    )

    # query all webservice ids and urls in one shot (also avoids to query it inside
    # url_iterator, which is run in a worker thread and might cause problems):
    ws_stmt = select(WebService.id, WebService.url).where(
        WebService.url.contains("/dataselect/") | WebService.url.contains("/station/")
    )
    with engine.connect() as conn:
        ws_urls: dict[int, str] = dict(conn.execute(ws_stmt).all())  # noqa

    def url_builder(df_row: tuple) -> str:
        """
        build url (str) from a row of the dataframe fetched from DB

        :param df_row: a row of the DB table, as normal tuple. This object is accessible
            in `response.meta`
        """
        net, sta, ws_id, _ = df_row  # must match select statement above

        return fdsn_url_qs(
            fdsn_url(ws_urls[ws_id], new_service='station'),
            net=net, sta=sta, level='response'
        )

    if max_download_concurrency_per_domain is None:
        max_download_concurrency_per_domain = 4

    stats = DownloadStats(get_host(w) for w in ws_urls.values())
    saved = 0
    now = datetime.now(UTC).replace(microsecond=0, tzinfo=None)
    with engine.begin() as conn:  # noqa
        for response in download_xml(
            engine = engine,
            stmt = stmt,
            url_builder = url_builder,
            err_log_caption="StationXML download errors",
            max_download_concurrency_per_domain = max_download_concurrency_per_domain,
            download_timeout = download_timeout,
            download_blocksize = download_blocksize,
            show_progress=show_progress
        ):
            stats.increment(get_host(response.request), response.status_code)
            net, sta, ws_id, sta_id = response.meta

            try:
                with conn.begin_nested():
                    conn.execute(
                        update(StationXML).where(
                            (StationXML.id==sta_id)
                        ).values({
                            "data": response.data,
                            "last_updated": now
                        })
                    )
                    saved += 1
            except IntegrityError as e:
                pass

    downloads_completed = stats.downloads_completed
    logger.info(f'{downloads_completed:,} download attempt(s) performed')
    logger.info(f'{saved:,} new station(s) saved to DB')
    return stats


def save_quakeml(
    *,
    engine: Engine,
    max_download_concurrency_per_domain: int | None,
    download_timeout,
    download_blocksize,
    show_progress=False
) -> DownloadStats:
    """Save QuakeML data. stations_df must not be empty (not checked here)"""
    stmt = (
        select(
            Event.id, Event.eventid, Event.webservice_id
        )
        .join(Segment, Segment.event_id == Event.id)  # (*) inner join
        .outerjoin(QuakeML, QuakeML.id == Event.id)
        .where(QuakeML.id.is_(None))
        .distinct()
    )
    # (*) only channels with at least one matching Segment row are included

    # query all webservice ids and urls in one shot (also avoids to query it inside
    # url_iterator, which is run in a worker thread and might cause problems):
    ws_stmt = select(WebService.id, WebService.url).where(
        ~WebService.url.contains("/dataselect/"),
        ~WebService.url.contains("/station/"),
    )
    with engine.connect() as conn:
        ws_urls: dict[int, str] = dict(conn.execute(ws_stmt).all())  # noqa

    def url_builder(df_row: tuple) -> str:
        """
        build url (str) from a row of the dataframe fetched from DB

        :param df_row: a row of the DB table, as normal tuple. This object is accessible
            in `response.meta`
        """
        _, catalog_ev_id, ws_id = df_row  # must match select statement above
        return fdsn_url_qs(ws_urls[ws_id], eventid=catalog_ev_id, format='xml')

    if max_download_concurrency_per_domain is None:
        max_download_concurrency_per_domain = 4  # same domain downloads

    stats = DownloadStats(get_host(w) for w in ws_urls.values())
    saved = 0
    with engine.begin() as conn:  # noqa
        for response in download_xml(
            engine=engine,
            stmt=stmt,
            url_builder=url_builder,
            err_log_caption="QuakeML download errors",
            max_download_concurrency_per_domain=max_download_concurrency_per_domain,
            download_timeout=download_timeout,
            download_blocksize=download_blocksize,
            show_progress=show_progress
        ):
            stats.increment(get_host(response.request), response.status_code)
            db_ev_id = response.meta[0]

            try:
                with conn.begin_nested():
                    conn.execute(insert(QuakeML), {
                        'id': db_ev_id,
                        'data': response.data
                    })
                saved += 1
            except IntegrityError as e:
                pass

    downloads_completed = stats.downloads_completed
    logger.info(f'{downloads_completed:,} download attempt(s) performed')
    logger.info(f'{saved:,} new event(s) saved to DB')
    return stats


def download_xml(
    *,
    engine: Engine,
    stmt: Select,
    url_builder: Callable[[tuple], str],
    err_log_caption: str,
    max_download_concurrency_per_domain: int,
    download_timeout,
    download_blocksize,
    show_progress=False
) -> Iterable[Response | None]:
    """Download XML data"""

    with engine.connect() as conn:
        total = conn.execute(
            select(func.count()).select_from(stmt.subquery())
        ).scalar_one()
        if total < 1:
            return

    # report seg. errors only once per error type and data center:
    already_logged_ids: set[tuple[str, int]] = set()

    with get_progressbar(total if show_progress else 0) as pbar:

        while True:

            total, downloaded = 0, 0

            for dfr in fetch_df(engine, stmt):

                total += len(dfr)

                reader = build_and_read_urls(
                    url_builder,
                    dfr.itertuples(index=False, name=None),
                    max_concurrency=max_download_concurrency_per_domain,
                    timeout=download_timeout,
                    blocksize=download_blocksize
                )

                for response in reader:
                    pbar.update(1)
                    downloaded += 1
                    url = response.request

                    if response.is_ok:
                        yield response
                        continue

                    if not already_logged_ids:
                        logger.warning(
                            f"{err_log_caption}\n"
                            "(shown once per (URL domain, error type) combination)"
                        )

                    log_id = (get_host(url), response.status_code)
                    if log_id not in already_logged_ids:
                        already_logged_ids.add(log_id)
                        logger.warning(str(response))

            if downloaded == total or max_download_concurrency_per_domain // 2 < 1:
                break

            max_download_concurrency_per_domain //= 2
