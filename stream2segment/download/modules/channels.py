"""
Stations/Channels download functions

:date: Dec 3, 2017
"""
import re
import logging
from collections.abc import Iterable, Sequence
import json
from datetime import datetime
from multiprocessing.pool import ThreadPool
from urllib.parse import urlunparse, urlparse
from urllib.request import urlopen

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_categorical_dtype
from sqlalchemy import select, Engine

from stream2segment.io.cli import get_progressbar
from stream2segment.io.db.pdsql import df2db, get_row_count, apply_table_dtypes
from stream2segment.io.db.models import Channel, WebService, Segment
from stream2segment.download.exc import FailedDownload
from stream2segment.download.url import urlread, get_host
from stream2segment.download.modules.utils import (
    formatmsg, fdsn_url, fdsn_url_qs, fdsn_response_text_to_df
)

# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial):
logger = logging.getLogger(__name__)

def get_channels(
    engine : Engine,
    datacenter_urls,
    network: list[str],
    station: list[str],
    location: list[str],
    channel: list[str],
    starttime,
    endtime,
    min_sample_rate,
    update_metadata,
    eida_rs_urls,
    restricted_download: bool,
    # max_thread_workers,
    # timeout,
    # blocksize,
    # db_bufsize,
    show_progress=False
):
    cha_urls = get_channel_urls(
        datacenter_urls,
        eida_rs_urls,
        [n for n in network if not n.startswith('!')],
        [s for s in station if not s.startswith('!')],
        [l for l in location if not l.startswith('!')],
        [c for c in channel if not c.startswith('!')],
        starttime,
        endtime
    )

    cha_df = download_channels(
        # session.get_bind(),
        cha_urls,
        show_progress=show_progress
        # eida_rs_urls,
        # max_thread_workers,
        #advanced_settings['s_timeout'],
        #download_blocksize,
        #dbbufsize,
        # isterminal
    )
    if cha_df.empty:
        raise FailedDownload('No channel downloaded')
    num_downloaded_channels = len(cha_df)

    # post filter (negation "!", sample rate) which raises FailedDownload if no rows:
    cha_df = filter_out_channels_df(
        cha_df,
        [n for n in network if n.startswith('!')],
        [s for s in station if s.startswith('!')],
        [l for l in location if l.startswith('!')],
        [c for c in channel if c.startswith('!')],
        # network, station, location, channel, starttime, endtime,
        min_sample_rate
    )
    if cha_df.empty:
        raise FailedDownload('No channel to work with after filtering out')

    # set ranking based on the order of urls
    cha_df = drop_conflicts(engine, cha_df, eida_rs_urls)
    if cha_df.empty:
        raise FailedDownload('No channel to work with after conflicts dropping')

    cha_df = cha_df.drop(columns='__.rank.__')
    cha_df = sync_webservice_ids_with_db(cha_df, engine)
    if cha_df.empty:
        raise FailedDownload(
            'No channel to work with after attempting to save channel webservices'
        )
    cha_df = save_channels(engine, cha_df, update_metadata)
    if cha_df.empty:
        raise FailedDownload(
            'No channel to work with after attempting to save channels'
        )

    # move (rename) current station ids and urls:
    ws_url_col = WebService.url.key
    cha_df = cha_df.rename(columns={
        ws_url_col: f'channel_{ws_url_col}',
        Channel.webservice_id.key: f'channel_{Channel.webservice_id.key}',
    })
    # get dataselect urls:
    dataselect_urls = {}
    for sta_url in pd.unique(cha_df[f'channel_{ws_url_col}']):
        dataselect_urls[sta_url] = fdsn_url(
            sta_url,
            new_service='dataselect',
            new_method='queryuauth' if restricted_download else None
        )

    # set new "url" column with dataselect urls:
    cha_df[ws_url_col] = cha_df[f'channel_{ws_url_col}'].map(
        dataselect_urls
    ).astype('category')
    # sync dataselect urls:
    cha_df = sync_webservice_ids_with_db(cha_df, engine)
    if cha_df.empty:
        raise FailedDownload(
            'No channel to work with after attempting to save channel webservices'
        )

    logger.info(
        f'Working with {len(cha_df):,} station channels '
        f'(downloaded {num_downloaded_channels:,}, '
        f'discarded: {num_downloaded_channels - len(cha_df):,})'
    )

    # convert to categorical type (for safety):
    for c in (
        Channel.network_code.key,
        Channel.station_code.key,
        Channel.location_code.key,
        Channel.channel_code.key,
        ws_url_col
    ):
        if not is_categorical_dtype(cha_df[c]):
            cha_df[c] = cha_df[c].astype('str').astype('category')

    cha_df.rename(columns={Channel.id.key: Segment.channel_id.key}, inplace=True)
    # return a copy of relevant columns only:
    return cha_df[[
        Segment.channel_id.key,
        Channel.network_code.key,
        Channel.station_code.key,
        Channel.location_code.key,
        Channel.channel_code.key,
        Channel.latitude.key,
        Channel.longitude.key,
        Channel.start_time.key,
        Channel.end_time.key,
        # ws_id_col,
        ws_url_col,
        Channel.webservice_id.key
    ]]


def get_channel_urls(
    webservice_url, routing_service_url,
    network: list[str] | None = None,
    station: list[str] | None = None,
    location: list[str] | None = None,
    channel: list[str] | None= None,
    starttime: datetime | None = None,
    endtime: datetime | None = None
) -> Iterable[str]:
    """
    Return an iterator of FDSN station urls from the given arguments
    """
    if isinstance(webservice_url, str):
        webservice_url = [webservice_url]
    params = {
        'net': ','.join(n for n in network or []) or '*',
        'sta': ','.join(s for s in station or []) or '*',
        'loc': ','.join(l for l in location or []) or '*',
        'cha': ','.join(c for c in channel or []) or '*',
        'start': starttime,
        'end': endtime
    }

    urls_done = set()
    parsed_urls = []
    for service_url in webservice_url:
        service_url = service_url.lower().strip()

        if service_url in urls_done:
            continue
        urls_done.add(service_url)

        if service_url == 'eida':
            priority_skipped = 0
            # if eidars_response_text is None:  # FIXME REMOVE
            #     eidars_response_text = get_eidars_response_text(
            #         routing_service_url, **params
            #     )
            eida_rs: dict = get_eida_rs_response(routing_service_url, **params)
            # sort (for easier grouping here below):
            for eida_dc_dict in eida_rs:
                eida_dc_dict['params'] = sorted(
                    eida_dc_dict['params'],
                    key=lambda p: (p['net'], p['sta'], p['loc'], p['cha'])
                )
            for line in eida_rs:
                # overwrite start end:
                url = line['url']
                for _params in line['params']:
                    if _params.get('priority', 2) > 1:
                        priority_skipped += 1
                        continue
                    _params.pop('priority')
                    _params['start'] = starttime
                    _params['end'] = endtime
                    merged_into_previous = False
                    if len(parsed_urls) and parsed_urls[-1][0] == url:
                        _last_params = parsed_urls[-1][1]
                        for prm in ['net', 'sta', 'loc', 'cha']:
                            if all(
                                _params[p] in _last_params[p].split(",")
                                for p in ['net', 'sta', 'loc', 'cha'] if p != prm
                            ):
                                if _params[prm] not in _last_params[prm].split(','):
                                    _last_params[prm] += f',{_params[prm]}'
                                merged_into_previous = True
                                break
                    if not merged_into_previous:
                        parsed_urls.append((url, _params))

        else:
            if service_url == 'iris':
                service_url = 'https://service.iris.edu/fdsnws/station/1/query'

            # restrict the search if too big:
            if params['net'] == '*' or params['sta'] == '*':
                for start_, end_ in split_times(params['start'], params['end'], 5):
                    _params = dict(params)
                    _params['start'] = start_
                    _params['end'] = end_
                    parsed_urls.append((service_url, _params))


    for url, params in parsed_urls:
        try:
            for url, params in check_and_yield_fdsn_urls(url, params):
                yield fdsn_url_qs(url, **params, level='channel', format='text')
        except ValueError as v_err:
            logger.warning(formatmsg(str(v_err), '', url))  # FIXME CHECK


def get_eida_rs_response(
    routing_service_url: list[str],
    net: str | None = None,
    sta: str | None = None,
    loc: str | None = None,
    cha: str | None= None,
    start: datetime | None = None,
    end: datetime | None = None,
    service='station'
) -> dict:
    """Return the EIDA Routing Service response text (str)"""
    for eida_rs_url in routing_service_url:
        url = fdsn_url_qs(
            eida_rs_url, net=net, sta=sta, loc=loc,
            cha=cha, start=start, end=end,
            service=service, format='json'
        )
        response = urlread(url, decode='utf8')
        if response.is_ok:
            return json.loads(response.data)
    raise FailedDownload("None of the EIDA routing services returned valid data. "
                         "Check internet connection or configure the URLs in advanced "
                         "settings")


def split_times(start: datetime, end: datetime, interval_years=5):
    cur = start
    while cur < end:
        nxt = min(cur.replace(year=cur.year + interval_years), end)
        yield cur, nxt
        cur = nxt


def check_and_yield_fdsn_urls(url, params):
    try:
        fdsn_station_url = fdsn_url(url, new_service='station')
    except ValueError as e:
        raise ValueError("Invalid FDSN URL")

    if params['net'] != '*':
        yield fdsn_station_url, params
        return
    # no network specified, query might take long (even for short time bounds).
    # get all networks and perform n subsets queries
    rows = []
    try:
        with urlopen(
            fdsn_url_qs(fdsn_station_url, **params, level='network', format='text')
        ) as r:
            header = r.readline().decode().strip().split('|')
            for line in r:
                split_line = line.decode().strip().split('|')
                rows.append((split_line[0].strip(), split_line[-1].strip()))
        df = pd.DataFrame(rows, columns=['net', 'count'])
        # convert count to numeric (should be int, but we set NaN as #station mean):
        df['count'] = pd.to_numeric(df['count'], errors='coerce')
        df.loc[pd.isna(df['count']), 'count'] = df['count'].mean()
        df['count'] = df['count'].astype(int)
        n = 5 # number of split requests
        # cumulative sum of counts (running total)
        c = df['count'].cumsum()
        # total sum of counts
        t = df['count'].sum()
        # map cumulative proportion to chunk index [0, n-1]
        df['chunk'] = (c / t * n).astype(int).clip(upper=n - 1)
        for _, df_ in df.groupby('chunk'):
            _params = dict(params)
            _params['net'] = ",".join(sorted(set(df_['net'])))
            yield fdsn_station_url, _params
    except Exception as e:
        yield fdsn_station_url, params


def download_channels(fdsn_station_urls, show_progress=False):
    """Return a Dataframe representing a query to the station service of each
    URL in :func:`stream2segment.download.modules.datacenters_df` with the
    given arguments.

    :param datacenters_df: (DataFrame) the first item resulting from
        `get_datacenters_df`
    :param min_sample_rate: minimum sampling rate, set to negative value
        for no-filtering (all channels)
    """
    ws_url_col = WebService.url.key

    t_pool = ThreadPool(4)
    def _urlread(_):
        return _[0], urlread(_[1], timeout=120, blocksize=-1)

    urls = list(fdsn_station_urls)

    channels_dfs = []
    station_urls = set()
    with get_progressbar(len(urls) if show_progress else 0) as pbar:
        for idx, response in t_pool.imap_unordered(_urlread, enumerate(urls)):
            pbar.update(1)
            if not response.is_ok:
                logger.warning(formatmsg("Unable to fetch stations",
                                         response.data,
                                         response.request))
                continue

            try:
                dframe = fdsn_channel_response_text_to_df(response.data.decode('utf8'))
                dframe['__.rank.__'] = idx
                discarded = dframe.attrs.pop('discarded', 0)
                if discarded > 0:
                    logger.warning(formatmsg(f"{discarded} row(s) discarded",
                                             "malformed text data",
                                             response.request))
            except ValueError as verr:
                logger.warning(formatmsg("Discarding response data", verr,
                                         response.request))
                continue

            if dframe.empty:
                continue
            station_url = urlunparse(
                urlparse(response.request)._replace(query='', fragment='')
            )
            dframe['url'] = station_url
            station_urls.add(station_url)
            channels_dfs.append(dframe)

    # build two dataframes which we will concatenate afterwards
    cha_df = pd.DataFrame()
    if channels_dfs:  # pd.concat complains about empty list
        # save urls and set them as categorical
        cha_df = pd.concat(channels_dfs, axis=0, ignore_index=True, copy=False)
        cha_df[ws_url_col] = cha_df[ws_url_col].astype('category')

    return cha_df


def fdsn_channel_response_text_to_df(response: str):
    """
    Convert a response content obtained from a FDSN station webservice with
    level=channel and  format=text into a pandas DataFrame
    with proper dtypes associated to the SQL mapped class
    """
    dframe = fdsn_response_text_to_df(response)
    # Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|
    # Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|
    # StartTime|EndTime`
    columns = {
        dframe.columns[0]: Channel.network_code.key,
        dframe.columns[1]: Channel.station_code.key,
        dframe.columns[2]: Channel.location_code.key,
        dframe.columns[3]: "channel_code",
        dframe.columns[4]: Channel.latitude.key,
        dframe.columns[5]: Channel.longitude.key,
        dframe.columns[6]: Channel.elevation.key,
        dframe.columns[7]: Channel.depth.key,
        dframe.columns[8]: Channel.azimuth.key,
        dframe.columns[9]: Channel.dip.key,
        # skip sensor_description (Rarely used, memory-intensive text field)
        dframe.columns[11]: Channel.scale.key,
        dframe.columns[12]: Channel.scale_freq.key,
        dframe.columns[13]: Channel.scale_units.key,
        dframe.columns[14]: Channel.sample_rate.key,
        dframe.columns[15]: Channel.start_time.key,
        dframe.columns[16]: Channel.end_time.key
    }

    if not dframe.empty:
        # rename and set order:
        dframe = dframe.rename(columns=columns)[list(columns.values())]
        dframe = apply_table_dtypes(Channel, dframe, drop_non_nullable=True)

    if dframe.empty:
        raise ValueError("Malformed data (e.g., no data, type mismatch, NaN)")
    return dframe


def filter_out_channels_df(
    channels: pd.DataFrame,
    net: list[str],
    sta: list[str],
    loc: list[str],
    cha: list[str],
    min_sample_rate
):
    """Filter out `channels_df` according to the given parameters. Raise
    `FailedDownload` if the returned filtered data frame woul be empty

    Note that `net, sta, loc, cha` filters will be considered only if negations
    (i.e., with leading exclamation mark: "!A*") because the 'positive' filters
    are FDSN stantard and are supposed to be already used in producing
    `channels_df`. Example:
    ```
        filter_channels_df(d, [], ['ABC'], [''], ['!A*', 'HH?', 'HN?'])
    ```
    basically takes the dataframe `d`, finds the column related to the
    `channels` key and removes all rows whose channel starts with 'A',
    returning the new filtered data frame.

    Arguments are usually the output of
    :func:`stream2segment.download.utils.nslc_lists`

    :param net: an iterable of strings denoting networks.
    :param sta: an iterable of strings denoting stations.
    :param loc: an iterable of strings denoting locations.
    :param cha: an iterable of strings denoting channels.
    :param min_sample_rate: numeric, minimum sample rate. If negative or zero,
        this parameter is ignored
    """
    # create a dict of regexps for pandas dataframe. FDSNWS do not support NOT
    # operators . Thus concatenate expression with OR
    df_filter = None
    sa_cols = (
        Channel.network_code,
        Channel.station_code,
        Channel.location_code,
        Channel.channel_code
    )

    for lst, sa_col in zip((net, sta, loc, cha), sa_cols):
        if not lst:
            continue
        # FIXME REMOVE:
        # lst = [_ for _ in lst if _[0:1] == '!']  # take only negation expr.
        #if not lst:
        #    continue
        # condition = ("^%s$" if len(lst) == 1 else "^(?:%s)$") % \
        #     "|".join(strconvert.wild2re(x[1:]) for x in lst)
        condition = "|".join(f"^(?:{wild2regex(x[1:])})$" for x in lst)
        flt = channels[sa_col.key].str.match(re.compile(condition))
        if df_filter is None:
            df_filter = flt
        else:
            df_filter |= flt

    if min_sample_rate is not None and min_sample_rate > 0:
        # None should evaluate to False, thus negate the predicate below:
        flt = channels[Channel.sample_rate.key] < min_sample_rate
        if df_filter is None:
            df_filter = flt
        else:
            df_filter |= flt

    ret = channels
    if df_filter is not None:
        ret = channels[~df_filter].copy()

    discarded_sr = len(channels) - len(ret)
    if discarded_sr:
        logger.warning(f"{discarded_sr:,} channel(s) discarded according to "
                       f"current configuration filters (network, channel, sample rate, "
                       "...)")

    return ret


def drop_conflicts(
    engine: Engine, channels: pd.DataFrame, eida_rs_urls: list[str] | None = None
):
    """
    Drop from channels_df conflict between, i.e., network.station codes
    returned by several URLs. Duplicated rows will be resolved against the
    database or, if keep_first is True, by taking the first row

    :param channels: pandas DataFrame
    :param eida_rs_urls: an optional list of source station FDSN urls that
        where used to build the passed dataframe. Order matters as conflicts
        will be resolved by taking
        the first matching url. If None, conflicts will cause all channels
        involved to be dropped
    :return: a new dataframe with duplicated rows removed
    """
    channels.reset_index(drop=True, inplace=True)
    # conflict between case is when station webservice is not unique, e.g.:
    #   net sta webservice_id
    #   N   S   1
    #   N   S   2
    net_col = Channel.network_code.key
    sta_col = Channel.station_code.key
    loc_col = Channel.location_code.key
    cha_col = Channel.channel_code.key
    # band_col = Channel.band_code.key
    # inst_col = Channel.band_code.key
    # orient_col = Channel.orientation_code.key
    start_col = Channel.start_time.key
    end_col = Channel.end_time.key
    webs_url_col = WebService.url.key

    channels.reset_index(drop=True, inplace=True)

    # all orientations of the same channel must have the same URL:
    grp_cols = [net_col, sta_col, loc_col, cha_col, start_col]

    # drop duplicates (against DB unique constraints):
    channels = channels.drop_duplicates(
        subset=grp_cols, keep='first'
    )

    geoloc_cols = [
        Channel.latitude.key,
        Channel.longitude.key,
        # Channel.elevation.key,
        # Channel.depth.key,
        # Channel.azimuth.key,
        # Channel.dip.key,
    ]
    inst_cols = [
        Channel.scale.key,
        Channel.scale_freq.key,
        Channel.scale_units.key,
        Channel.sample_rate.key
    ]

    def allclose(col: pd.Series, **kwargs):
        """np.allclose robust to non-numeric dtypes"""
        if pd.api.types.is_numeric_dtype(col):
            return np.allclose(col.iloc[0], col.iloc[1:], **kwargs)
        return len(pd.unique(col)) == 1

    band_inst_col: pd.Series = channels[cha_col].str[:2]
    # 1) CHANNELS WITH SAME CODE AND START TIME MUST COME FROM A SINGLE URL:
    grp_cols = [net_col, sta_col, loc_col, band_inst_col, start_col]

    def webservice_urls(_df):
        return pd.unique(_df[webs_url_col])

    table_empty = get_row_count(engine, Channel) < 1

    conflicting_indices = set()

    for (net, sta, loc, band_inst, start), cha_df in (
        channels.groupby(grp_cols)
    ):
        urls = webservice_urls(cha_df)
        if len(urls) > 1:

            logger.warning(
                'Duplicated urls for channel (attempting to resolve conflict): ' +
                "\n".join(
                    fdsn_url_qs(u, net=net, sta=sta, loc=loc, cha=band_inst + "?")
                    for u in urls
                )
            )

            if not table_empty:
                keep = _check_conflict_between_via_db(
                    engine,
                    cha_df,
                    net,
                    sta,
                    loc,
                    *band_inst
                )
                conflicting_indices.update(cha_df[~keep].index)
                cha_df = cha_df[keep]
                urls = webservice_urls(cha_df)

            if eida_rs_urls is not None and len(urls) > 1:
                eida_rs_json = get_eida_rs_response(eida_rs_urls, net=net, sta=sta)
                keep = _check_conflict_between_via_eida_rs(
                    cha_df, eida_rs_json
                )
                conflicting_indices.update(cha_df[~keep].index)
                cha_df = cha_df[keep]
                urls = webservice_urls(cha_df)

            if len(urls) > 1:
                # conflict found, unresolvable through already saved data. Get first
                # if instructed to do so. FIXME add param or do it automatically likle here?
                real_ws_url = cha_df[
                    cha_df['__.rank.__'] == cha_df['__.rank.__'].min()
                ].iloc[0][webs_url_col]
                keep = cha_df[webs_url_col] == real_ws_url
                conflicting_indices.update(cha_df[~keep].index)
                cha_df = cha_df[keep]
                urls = webservice_urls(cha_df)

            if len(urls) > 1 or cha_df.empty:
                conflicting_indices.update(cha_df.index)
                continue

        url = urls[0]

        # No double webservice url.
        # First check that geo locations match (otherwise we might have
        # inconsistencies in arrival times for same channel) by relaxing a bit
        # equality (use allclose):
        if not all(allclose(cha_df[c]) for c in geoloc_cols):
            logger.warning(
                'Non-unique (lat, lon) for channel (conflict not handled, all channels kept): ' +
                fdsn_url_qs(url, net=net, sta=sta, loc=loc, cha=band_inst + "?")
            )

    # 2) CHANNELS WITH SAME CODE AND URL MUST HAVE DIFFERENT TIME RANGES:
    grp_cols = [net_col, sta_col, loc_col, band_inst_col, webs_url_col]

    for (net, sta, loc, band_inst, url), cha_df in (
        channels[~channels.index.isin(conflicting_indices)].groupby(grp_cols)
    ):
        if len(pd.unique(cha_df[start_col])) > 1:

            logger.warning(
                'Duplicated start_time for channel (attempting to resolve conflict): ' +
                fdsn_url_qs(url, net=net, sta=sta, loc=loc, cha=band_inst+"?")
            )
            for o_code in pd.unique(cha_df[cha_col].str[2]):
                cha_df_o = cha_df[cha_df[cha_col].str[2] == o_code]
                if all(allclose(cha_df_o[c]) for c in inst_cols):
                    # only time ranges differ, merge all columns into first:
                    idx = cha_df_o.index[0]
                    # merge time ranges:
                    channels.at[idx, start_col] = cha_df_o[start_col].min()
                    channels.at[idx, end_col] = cha_df_o[end_col].max()
                    # discard other columns:
                    conflicting_indices.update(cha_df_o.index[1:])
                else:
                    # Only instrument values differ, then it is likely a problem in time
                    # ranges. Because they all overlap, set end times to not overlap next
                    # start time (keep all columns):
                    for i in range(len(cha_df_o) -1):
                        start_time = cha_df_o.at[cha_df_o.index[i + 1], start_col]
                        channels.at[cha_df_o.index[i], end_col] = start_time

    if conflicting_indices:
        drop = channels.index.isin(conflicting_indices)
        # log_df = channels[drop]
        channels = channels[~drop]
        # logger.warning('Station channels dropped after applying conflict resolver:')
        # logger.warning(
        #     # groupby url and station code, join all channels together and sort by url:
        #     df2str(
        #         log_df.groupby(
        #             [webs_url_col, net_col, sta_col, loc_col]
        #         )[Channel.channel_code.key].agg(lambda x: ",".join({_ for _ in x.astype(str)})).
        #         reset_index().sort_values(by=webs_url_col)
        #     )
        # )

    return channels


def _check_conflict_between_via_db(
    engine, cha_df, net, sta, loc, band, inst
) -> pd.Series:

    keep = np.zeros(len(cha_df), dtype=bool)
    webs_url_col = WebService.url.key

    for _, cha_df in cha_df.groupby([webs_url_col], sort=False):
        stmt = (
            select(WebService.url)
            .join(Channel, Channel.webservice_id == WebService.id)
            .where(
                Channel.network_code == net,
                Channel.station_code == sta,
            )
        )

        with engine.connect() as conn:
            real_ws_urls = conn.execute(stmt).scalars().all()

        if len(real_ws_urls) == 1:
            keep |= (cha_df[webs_url_col] == next(iter(real_ws_urls))).values

    return keep


def _check_conflict_between_via_eida_rs(cha_df, eida_rs_json) -> pd.Series:
    keep = np.zeros(len(cha_df), dtype=bool)

    urls = []

    net_col = Channel.network_code.key
    sta_col = Channel.station_code.key
    loc_col = Channel.location_code.key
    cha_col = Channel.channel_code.key
    webs_url_col = WebService.url.key

    for item in eida_rs_json:
        urls.append(item['url'])
        flt = cha_df[webs_url_col] == item['url']
        for params in item['params']:
            if params['priority'] != 1:
                continue
            for df_col, eida_col in {
                net_col: 'net',
                sta_col: 'sta',
                loc_col: 'loc',
                cha_col: 'cha'
            }.items():
                flt &= cha_df[df_col].str.match(f"^{wild2regex(params[eida_col])}$")
            start = datetime.fromisoformat(params['start'])
            flt &= cha_df[Channel.start_time.key] >= start
            if params['end']:
                end = datetime.fromisoformat(params['end'])
                flt &= (
                    (cha_df[Channel.end_time.key] <= end) |
                    cha_df[Channel.end_time.key].isna()
                )
            keep |= flt.values

    # channels not in any eida url have to be taken because we could not infer:
    keep |= (~cha_df[webs_url_col].isin(urls)).values
    return keep


def wild2regex(text: str):
    return (
        text
        .replace('.', r'\.')
        .replace('?', '.')
        .replace('*', '.*')
    )


def save_channels(engine: Engine, channels: pd.DataFrame, update: bool):
    """Saves to db channels (and their stations) and returns a dataframe with
    only channels saved. The returned Dataframe will have the column 'id'
    (`Station.id`) renamed to 'station_id' (`Channel.station_id`) and a new
    'id' column referring to the Channel id (`Channel.id`)

    :param channels: pandas DataFrame
    """
    # if update is True, don't update inventories HERE (handled later)
    update_cols = []
    if update:
        update_cols = [
            Channel.latitude.key,
            Channel.longitude.key,
            Channel.elevation.key,
            Channel.depth.key,
            Channel.azimuth.key,
            Channel.dip.key,
            Channel.scale.key,
            Channel.scale_freq.key,
            Channel.scale_units.key,
            Channel.sample_rate.key,
        ]

    channels[[
        Channel.band_code.key,
        Channel.instrument_code.key,
        Channel.orientation_code.key
    ]] = channels[Channel.channel_code.key].str.extract(r'(.)(.)(.)').astype('category')

    # Add channels to db. First set columns defining channel identity (db
    # unique constraint):
    # cols = [Channel.station_id, Channel.location, Channel.channel]
    uc_cols = [
        Channel.network_code.key,
        Channel.station_code.key,
        Channel.location_code.key,
        Channel.band_code.key,
        Channel.instrument_code.key,
        Channel.orientation_code.key,
        Channel.start_time.key,
    ]
    channels, i_err, u_failed = df2db(
        channels, Channel, engine, Channel.id.key, uc_cols, update_cols
    )
    logger.info(f'{len(channels):,} of {(len(channels) + len(i_err)):,} '
                f'seismic channel(s) saved')
    if len(i_err):
        logger.warning(f"Unable to save {len(i_err)} seismic channel(s):")
        logger.warning(
            i_err.to_string(max_rows=30, index=False, na_rep='', show_dimensions=True)
        )

    # log_unsaved_channels(conflict_between, conflict_within)
    channels = channels.drop(columns=[
        Channel.band_code.key,
        Channel.instrument_code.key,
        Channel.orientation_code.key
    ])
    return channels


def sync_webservice_ids_with_db(cha_df, engine, urls_col=WebService.url.key):

    ws_df, i_err, _ = df2db(
        pd.DataFrame([{urls_col: u} for u in cha_df[urls_col].cat.categories]),
        WebService,
        engine,
        'id',
        [urls_col]
    )
    # now assign:
    cha_df = cha_df.merge(
        ws_df.rename(columns={"id": Channel.webservice_id.key}),
        on=urls_col,
        how="left"
    )
    wsid_na = pd.isna(cha_df[Channel.webservice_id.key])
    wsurl_na = pd.unique(cha_df[wsid_na][urls_col])
    if wsid_na.any():
        logger.warning(f"Unable to store {wsurl_na:,} url(s) "
                       f"for a total of {wsid_na.sum()} channel(s) "
                       f"discarded:")
        logger.warning(
            cha_df[wsid_na].to_string(
                max_rows=30, index=False, na_rep='', show_dimensions=True
            )
        )
        cha_df = cha_df[~wsid_na].copy()
    return cha_df
