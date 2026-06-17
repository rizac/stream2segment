"""
StationsXML download
"""
from __future__ import annotations
import dataclasses
import logging
from collections.abc import Iterable, Callable
from dataclasses import asdict
from typing import Optional
import pandas as pd
from sqlalchemy.exc import IntegrityError

from stream2segment.download.channels import url_col, net_col, sta_col
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
) -> XMLStats:
    """Save StationXML data. stations_df must not be empty (not checked here)"""
    staxml_id_col = Channel.stationxml_id.key
    stmt = (
        select(
            Channel.network_code,
            Channel.station_code,
            Channel.data_webservice_id,
            func.max(Channel.stationxml_id).label(staxml_id_col),
            # WebService.url
        ).
        where(
            exists(
                # select(1) is hust a convention, e.g. *, Event also work
                select(1).where(Segment.channel_id == Channel.id)
            )
        ).
        group_by(
            Channel.network_code,
            Channel.station_code,
            Channel.data_webservice_id,
        ).having(
            func.sum(case((Channel.stationxml_id.is_(None), 1), else_=0)) > 0
        )
    )

    total, downloaded, saved = 0, 0, 0
    db_stationxml_id = get_col_max(engine, StationXML.id)
    ws_id_col = Channel.data_webservice_id.key
    # query all webservice ids and urls in one shot (also avoids to query it inside
    # url_iterator, which is run in a worker thread and might cause problems):
    ws_stmt = select(WebService.id, WebService.url).where(
        WebService.url.contains("/dataselect/") | WebService.url.contains("/station/")
    )
    with engine.connect() as conn:
        ws_urls = dict(conn.execute(ws_stmt).all())  # noqa

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
            total += 1
            if response is None:
                continue

            downloaded += 1
            net, sta, ws_id, sta_id = response.meta

            try:
                with conn.begin_nested():
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

    return XMLStats(total, downloaded, saved)


def save_quakeml(
    *,
    engine: Engine,
    max_download_concurrency_per_domain: int | None,
    download_timeout,
    download_blocksize,
    show_progress=False
) -> XMLStats:
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

    total, downloaded, saved = 0, 0, 0
    ws_id_col = Event.webservice_id.key
    # query all webservice ids and urls in one shot (also avoids to query it inside
    # url_iterator, which is run in a worker thread and might cause problems):
    ws_stmt = select(WebService.id, WebService.url).where(
        ~WebService.url.contains("/dataselect/"),
        ~WebService.url.contains("/station/"),
    )
    with engine.connect() as conn:
        ws_urls = dict(conn.execute(ws_stmt).all())  # noqa


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
            total += 1
            if response is None:
                continue

            downloaded += 1
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

    return XMLStats(total, downloaded, saved)


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

            oks = 0

            for dfr in fetch_df(engine, stmt):

                reader = build_and_read_urls(
                    url_builder,
                    dfr.itertuples(index=False, name=None),
                    max_concurrency=max_download_concurrency_per_domain,
                    timeout=download_timeout,
                    blocksize=download_blocksize
                )

                for response in reader:
                    pbar.update(1)
                    url = response.request
                    oks += (200 <= response.status_code < 300)

                    if not response.is_ok:

                        if not already_logged_ids:
                            logger.warning(
                                f"{err_log_caption}\n"
                                "(shown once per (URL domain, error type) combination)"
                            )

                        log_id = (get_host(url), response.status_code)
                        if log_id not in already_logged_ids:
                            already_logged_ids.add(log_id)
                            logger.warning(str(response))

                        yield None

                    else:
                        yield response

            if oks / total >= 0.75 or max_download_concurrency_per_domain // 2 < 1:
                break

            max_download_concurrency_per_domain //= 2


@dataclasses.dataclass(frozen=True, slots=True)
class XMLStats(dict):

    total: int
    downloaded: int
    saved: int

    def __str__(self):
        total = self.total
        downloaded = self.downloaded
        saved = self.saved

        not_downloaded = total - downloaded
        nds = ''
        if not_downloaded > 0:
            nds = f"{not_downloaded:,} download error"
            if not_downloaded > 1:
                nds += 's'
            nds = f'({nds})'

        not_saved = downloaded - saved
        nss = ''
        if not_saved > 0:
            nss = f"{not_saved:,} db error"
            if not_saved > 1:
                nss += 's'
            nss = f'({nss})'

        df = pd.DataFrame(
            columns = ['c1', 'c2'],  # any name is ok (we will not show these)
            index = ['Total', 'Downloaded', 'Saved'],
            data = [
                [total, ""],
                [downloaded, nds],
                [saved, nss]
            ]
        )
        df['c1'] = df['c1'].astype(int)
        df['c2'] = df['c2'].astype(str)
        return df.to_string(
            index=True,
            header=False,
            na_rep="",
            formatters={'c1': "{:,}".format}
        )

    def to_dict(self) -> dict:
        return asdict(self)
