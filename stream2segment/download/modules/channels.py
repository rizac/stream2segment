"""
Stations/Channels download functions

:date: Dec 3, 2017
"""
import re
import logging
from collections.abc import Iterable
import json
from datetime import datetime
from multiprocessing.pool import ThreadPool
from urllib.parse import urlunparse, urlparse
from urllib.request import urlopen

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_categorical_dtype
from sqlalchemy import select, Engine

# from sqlalchemy import or_, and_

from stream2segment.io.cli import get_progressbar
from stream2segment.io.db.pdsql import shared_colnames, \
    Inserter, sync_pkey, df2db, get_row_count  # dbquery2df, , mergeupdate
from stream2segment.io.db.models import Channel, WebService, Segment
from stream2segment.download.exc import FailedDownload
from stream2segment.download.url import urlread, get_host
from stream2segment.download.modules.utils import (fdsn_channel_response_text_to_df,
                                                   formatmsg, fdsn_url,
                                                   fdsn_url_qs, df2str)

# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial):
logger = logging.getLogger(__name__)

def get_channels(
    session,
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
    # max_thread_workers,
    # timeout,
    # blocksize,
    # db_bufsize,
    show_progress=False
):
    cha_urls = list(get_channel_urls(
        datacenter_urls,
        eida_rs_urls,
        [n for n in network if not n.startswith('!')],
        [s for s in station if not s.startswith('!')],
        [l for l in location if not l.startswith('!')],
        [c for c in channel if not c.startswith('!')],
        starttime,
        endtime
    ))

    cha_df = download_channels(
        # session.get_bind(),
        cha_urls,
        # eida_rs_urls,
        # max_thread_workers,
        #advanced_settings['s_timeout'],
        #download_blocksize,
        #dbbufsize,
        # isterminal
    )

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

    # first drop duplicates (all columns the same):
    # this method does very few things as there might be rounding errors that
    # prevent equal columns to be equal. Anyway, we perform here more sound checks
    # cha_df = cha_df.drop_duplicates(keep='first')

    engine = session.get_bind()
    orig_cha_df = cha_df
    # set ranking based on the order of urls
    cha_df = drop_conflict_between(engine, cha_df, eida_rs_urls)
    cha_df = cha_df.drop(columns='__.rank.__')

    cha_df = drop_conflict_within(cha_df)
    cha_df = sync_webservice_ids_with_db(cha_df, engine)
    cha_df = save_channels(engine, cha_df, update_metadata)

    # move (rename) current station ids and urls:
    ws_url_col = WebService.url.key
    cha_df = cha_df.rename(columns={
        ws_url_col: f'channel_{ws_url_col}',
        Channel.webservice_id.key: f'channel_{Channel.webservice_id.key}',
    })
    # get dataselect urls:
    dataselect_urls = {
        u: fdsn_url(u, new_service='dataselect') for u in
        pd.unique(cha_df[f'channel_{ws_url_col}'])
    }
    # set new "url" column with dataselect urls:
    cha_df[ws_url_col] = cha_df[f'channel_{ws_url_col}'].map(dataselect_urls)
    # sync dataselect urls:
    cha_df = sync_webservice_ids_with_db(cha_df, engine)
    cha_df[ws_url_col] = cha_df[ws_url_col].astype('category')

    logger.info(
        f'{len(cha_df):,} of {(len(orig_cha_df)):,} station url(s) saved'
    )
    if len(orig_cha_df) > len(cha_df):
        logger.warning(f"Unable to save {len(orig_cha_df) - len(cha_df)} "
                       f"channel(s) (e.g., dropped due to conflicts, "
                       f"not written due to db errors)")

    # if len(failed_dframe_rows) > 0:
    #     # get_channels_df_from_db(session, sta_ws_id, net, sta, loc, cha,
    #     #                                     starttime, endtime)
    #     # logger.info(formatmsg(f"Fetching stations data from database for "
    #     #                       f"{len(failed_dframe_rows)} failed http request(s)",
    #     #                       "download errors occurred"))
    #     logger.info(formatmsg(f"{len(failed_dframe_rows)} failed download(s), "
    #                           f"unable to fetch all available stations",
    #                           "download errors occurred"))
    if cha_df.empty:
        # ok, now let's see if we have remaining datacenters to be
        # fetched from the db
        raise FailedDownload(formatmsg("No station found",
                                       "Unable to fetch stations from all "
                                       "data-centers, no data to fetch from "
                                       "the database. Check config and log "
                                       "for details"))

    # post process cha_df and return only relevant data

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

    # return a copy of relevant columns only:
    return cha_df[[
        Channel.id.key,
        Channel.latitude.key,
        Channel.longitude.key,
        Channel.network_code.key,
        Channel.station_code.key,
        Channel.location_code.key,
        Channel.channel_code.key,
        # ws_id_col,
        ws_url_col
    ]].copy()


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
    service='dataselect'
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


def download_channels(
    # session,
    fdsn_station_urls,
    # no_net: list[str],
    # no_sta: list[str],
    # no_loc: list[str],
    # no_cha: list[str],
    # starttime, endtime,
    # min_sample_rate,
    # update,
    # eida_rs_urls,
    # max_thread_workers,
    # timeout,
    # blocksize,
    # db_bufsize,
    show_progress=False
):
    """Return a Dataframe representing a query to the station service of each
    URL in :func:`stream2segment.download.modules.datacenters_df` with the
    given arguments.

    :param datacenters_df: (DataFrame) the first item resulting from
        `get_datacenters_df`
    :param min_sample_rate: minimum sampling rate, set to negative value
        for no-filtering (all channels)
    """
    ws_url_col = WebService.url.key
    # ws_id_col = Channel.webservice_id.key
    #
    # iterable = zip(
    #     datacenters_df[ws_url_col],
    #     datacenters_df['net'],
    #     datacenters_df['sta'],
    #     datacenters_df['loc'],
    #     datacenters_df['cha'],
    #     datacenters_df['start'],
    #     datacenters_df['end'],
    #     datacenters_df[ws_id_col]
    # )
    #
    # def url_builder(row):
    #     """build url (str) from each item yielded by the previous iterable"""
    #     return fdsn_url_qs(row[0], net=row[1], sta=row[2], loc=row[3], cha=row[4],
    #                        start=pd.isna(row[5]) or None, end=pd.isna(row[6]) or None,
    #                        level='channel', format='text')

    # dict db id -> (station url, dataselect url)

    # def url_iter():
    #
    #     dfr = datacenters_df
    #     all_cols = [ws_url_col, 'net', 'sta', 'loc', 'cha', 'start', 'end']
    #     for col in ['cha', 'loc', 'sta', 'net']:
    #         ret = []
    #         cols = all_cols.copy()
    #         cols.remove(col)
    #         for _, sub_dfr in dfr.groupby(cols, sort=False, dropna=False):
    #             if len(sub_dfr) > 1:
    #                 tmp_df = sub_dfr.iloc[:1]
    #                 tmp_df[col] = [sub_dfr[col].sort_values().str.cat(sep=",")]
    #                 sub_dfr = tmp_df
    #             ret.append(sub_dfr)
    #         dfr = pd.concat(ret, axis=0, ignore_index=True, copy=False)
    #
    #     for url, net, sta, loc, cha, start, end in dfr[all_cols].itertuples(index=False):
    #         kw_args = {
    #             'level': 'channel',
    #             'format': 'text',
    #             'net': net,
    #             'sta': sta,
    #             'loc': loc,
    #             'cha': cha,
    #         }
    #         if pd.notna(start):
    #             kw_args['start'] = start
    #         if pd.notna(end):
    #             kw_args['end'] = end
    #         yield fdsn_url_qs(url, **kw_args)

    t_pool = ThreadPool(2)
    def _urlread(_):
        return _[0], urlread(_[1], timeout=120, blocksize=-1)

    urls = list(fdsn_station_urls)

    channels_dfs = []
    failed_dframe_rows = []
    station_urls = set()
    with get_progressbar(len(urls) if show_progress else 0) as pbar:
        for idx, response in t_pool.imap_unordered(_urlread, enumerate(urls)):
            pbar.update(1)
            # FIXME REMOVE
            # sta_ws_url = obj[0]
            # sta_ws_id = obj[7]
            if not response.is_ok:
                # failed_dframe_rows.append(obj)
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
            # for col, val in (
            #     (ws_id_col, sta_ws_id),
            #     (ws_url_col, sta_ws_url)
            # ):
            #     dframe[col] = val
            #     dtype = datacenters_df[col].dtype
            #     dframe[col] = dframe[col].astype(dtype)
            channels_dfs.append(dframe)

    # build two dataframes which we will concatenate afterwards
    cha_df = pd.DataFrame()
    if channels_dfs:  # pd.concat complains about empty list
        # save urls and set them as categorical
        cha_df = pd.concat(channels_dfs, axis=0, ignore_index=True, copy=False)
        cha_df[ws_url_col] = cha_df[ws_url_col].astype('category')

    return cha_df


def filter_out_channels_df(
    channels_df, net: list[str], sta: list[str], loc, cha: list[str], min_sample_rate
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
        Channel.network_code, Channel.station_code, Channel.location_code, Channel.channel_code
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
        flt = channels_df[sa_col.key].str.match(re.compile(condition))
        if df_filter is None:
            df_filter = flt
        else:
            df_filter |= flt

    if min_sample_rate is not None and min_sample_rate > 0:
        # None should evaluate to False, thus negate the predicate below:
        flt = channels_df[Channel.sample_rate.key] < min_sample_rate
        if df_filter is None:
            df_filter = flt
        else:
            df_filter |= flt

    ret = channels_df
    if df_filter is not None:
        ret = channels_df[~df_filter].copy()

    if ret.empty:
        raise FailedDownload("No channel matches user defined filters "
                             "(network, channel, sample rate, ...)")

    discarded_sr = len(channels_df) - len(ret)
    if discarded_sr:
        logger.warning(f"{discarded_sr:,} channel(s) discarded according to "
                       f"current configuration filters (network, channel, sample rate, "
                       "...)")

    return ret


def drop_conflict_between(
    engine: Engine, channels_df, eida_rs_urls: list[str] | None = None
):
    """
    Drop from channels_df conflict between, i.e., network.station codes
    returned by several URLs. Duplicated rows will be resolved against the
    database or, if keep_first is True, by taking the first row

    :param channels_df: pandas DataFrame
    :param urls: an optional list of source station FDSN urls that where used to build
        the passed dataframe. Order matters as conflicts will be resolved by taking
        the first matching url. If None, conflicts will cause all channels
        involved to be dropped
    :return: a new dataframe with duplicated rows removed
    """
    channels_df.reset_index(drop=True, inplace=True)
    # conflict between case is when station webservice is not unique, e.g.:
    #   net sta webservice_id
    #   N   S   1
    #   N   S   2
    net_col = Channel.network_code.key
    sta_col = Channel.station_code.key
    loc_col = Channel.location_code.key
    cha_col = Channel.channel_code.key
    grp_cols = [net_col, sta_col, loc_col, cha_col]
    webs_url_col = WebService.url.key

    conflict_between = (
        channels_df.groupby(grp_cols)[webs_url_col].transform("nunique") > 1
    )
    # (channels_df[conflict_between].sort_values(grp1_cols, ascending=True).
    # to_csv(
    #     path_or_buf=os.expanduser('~/work/code/stream2segment/conflict_between.csv'),
    #     index=False
    # ))

    table_empty = get_row_count(engine, Channel) < 1

    # log messages:
    if conflict_between.any():
        log_df = channels_df[conflict_between].groupby(
            [net_col, sta_col], as_index=False
        ).agg({ WebService.url.key: lambda x: ", ".join(sorted(set(x))) })
        logger.warning(
            f'Conflict: different URLs returning the same station. Conflicts summary:'
        )
        columns2show = [net_col, sta_col] + ['URLs (should be 1)']
        log_df = log_df.rename(columns={WebService.url.key: columns2show[-1]})
        logger.warning(
            log_df[columns2show].sort_values(by=columns2show).to_string(
                na_rep='', index=False
            )
        )

        conflict_between_indices = []
        # although we check conflicts by net.sta.loc.cha, we are interested in fixing
        # the station problem here

        for (net, sta, loc, cha_prefix), cha_df in (
            channels_df.loc[conflict_between].groupby([
                net_col, sta_col, loc_col, channels_df[cha_col].str[:2]
            ], sort=False)
        ):
            if not table_empty:
                keep_indices = _check_conflict_between_via_db(
                    engine,
                    cha_df,
                    net,
                    sta,
                    loc,
                    cha_prefix[0],
                    cha_prefix[1]
                )
                cha_df = cha_df[~cha_df.index.isin(keep_indices)]

            if eida_rs_urls is not None and not cha_df.empty:
                eida_rs_json = get_eida_rs_response(eida_rs_urls, net=net, sta=sta)
                _keep_indices = _check_conflict_between_via_eida_rs(
                    cha_df, eida_rs_json
                )
                cha_df = cha_df[~cha_df.index.isin(_keep_indices)]

            if not cha_df.empty:
                # conflict found, unresolvable through already saved data. Get first
                # if instructed to do so. FIXME add param or do it automatically likle here?
                real_ws_url = cha_df[
                    cha_df['__.rank.__'] == cha_df['__.rank.__'].min()
                ].iloc[0][webs_url_col]
                keep_indices = cha_df[cha_df[webs_url_col] == real_ws_url].index
                conflict_between_indices.extend(~cha_df.index.isin(keep_indices))

        channels_df = channels_df[
            ~channels_df.index.isin(conflict_between_indices)
        ].copy()

    return channels_df


def _check_conflict_between_via_db(engine, cha_df, net, sta, loc, band, inst):

    webs_url_col = WebService.url.key
    keep_indices = []

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
            keep_indices.extend(
                cha_df[cha_df[webs_url_col] == next(iter(real_ws_urls))].index
            )
        return keep_indices


def _check_conflict_between_via_eida_rs(cha_df, eida_rs_json):
    keep_indices = []
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
        keep_indices.extend(cha_df[flt].index)

    # channels not in any eida url have to be taken because we could not infer:
    keep_indices.extend(cha_df[~cha_df[webs_url_col].isin(urls)].index)
    return keep_indices


def wild2regex(text: str):
    return (
        text
        .replace('.', r'\.')
        .replace('?', '.')
        .replace('*', '.*')
    )


def drop_conflict_within(channels_df):
    """
    Drop from channels_df conflict within, i.e., same
    network.station.location.channel.start_time  returned by the same URLs.
    Duplicated rows will be resolved by taking the item which spans the
    biggest time range (which is the least bad option)

    :return: a new dataframe with duplicated rows removed
    """
    channels_df.reset_index(drop=True, inplace=True)
    webs_url_col = WebService.url.key
    start_col = Channel.start_time.key
    end_col = Channel.end_time.key
    geoloc_cols = [
        Channel.latitude.key,
        Channel.longitude.key,
        Channel.elevation.key,
        Channel.depth.key,
        Channel.azimuth.key,
        Channel.dip.key,
    ]
    inst_cols = [
        Channel.scale.key,
        Channel.scale_freq.key,
        Channel.scale_units.key,
        Channel.sample_rate.key
    ]
    grp_cols = [
        Channel.network_code.key,
        Channel.station_code.key,
        Channel.location_code.key,
        Channel.channel_code.key,
        # Channel.start_time.key,
        webs_url_col,
    ]

    def allclose(col: pd.Series, **kwargs):
        """np.allclose robust to non-numeric dtypes"""
        if pd.api.types.is_numeric_dtype(col):
            return np.allclose(col.iloc[0], col.iloc[1:], **kwargs)
        return len(pd.unique(col)) == 1

    # Just for ref, these rows are not detected by duplicated (apparently, scale differs):
    #        network_code station_code location_code channel_code   latitude  longitude  elevation  depth  azimuth  dip          scale  scale_freq scale_units  sample_rate          start_time   end_time                                              url  webservice_id
    # 97463            GS        KAN12            01          HNE  37.297383    -97.998      425.5    0.0     90.0  0.0  184349.032785         1.0      m/s**2        200.0 2014-05-08 17:11:06 2015-04-14  https://service.iris.edu/fdsnws/station/1/query             11
    # 131397           GS        KAN12            01          HNE  37.297383    -97.998      425.5    0.0     90.0  0.0  184349.032785         1.0      m/s**2        200.0 2014-05-08 17:11:06 2015-04-14  https://service.iris.edu/fdsnws/station/1/query             11
    #

    conflict_within = (
        channels_df.groupby(grp_cols)[geoloc_cols + inst_cols].transform("nunique") > 1
    ).any(axis=1)
    # (channels_df[conflict_within].sort_values(grp2_cols, ascending=True).
    # to_csv(
    #     path_or_buf=os.expanduser('~/work/code/stream2segment/conflict_within.csv'),
    #     index=False
    # ))
    conflict_within_indices = []
    if conflict_within.any():

        log_df = channels_df[conflict_within].groupby(
            [WebService.url.key] + grp_cols[:-1], as_index=False
        ).size()
        logger.warning(
            f'Conflict: the same URL returning a channel multiple times. '
            f'Conflicts summary:  '
        )
        columns2show = [WebService.url.key] + grp_cols[:4] + ["instances (should be 1)"]
        log_df = log_df.rename(columns={"size": columns2show[-1]})
        logger.warning(
            log_df[columns2show].sort_values(
                by=columns2show[-1:], ascending=False
            ).to_string(na_rep='', index=False)
        )

        for _, cha_df in channels_df[conflict_within].groupby(
            grp_cols, sort=False
        ):
            if len(cha_df) <= 1:  # for safety
                continue

            cha_df = cha_df.copy()
            cha_df.loc[pd.isna(cha_df[end_col]), end_col] = pd.Timestamp.now()
            cha_df = cha_df.sort_values(by=[start_col, end_col], ascending=True)
            overlap_with_next = (
                np.append(
                    [cha_df[end_col].values[:-1] > cha_df[start_col].values[1:]],
                    False
                ).astype(bool)
            )

            if not overlap_with_next.any():  # no time range overlaps -> ok
                continue

            if not overlap_with_next[:-1].all():
                # not all time ranges overlap ->  discard
                conflict_within_indices.extend(cha_df.index)
                continue

            # First check that geo locations match (otherwise we might have
            # inconsistencies in arrival times for same channel) by relaxing a bit
            # equality (use allclose):
            if not all(allclose(cha_df[c]) for c in geoloc_cols):
                # geo position mismatch: discard all
                conflict_within_indices.extend(cha_df.index)
                continue

            if all(allclose(cha_df[c]) for c in inst_cols):
                # only time ranges differ, merge all columns into first:
                idx = cha_df.index[0]
                # merge time ranges:
                channels_df.at[idx, start_col] = cha_df[start_col].min()
                channels_df.at[idx, end_col] = cha_df[end_col].max()
                # discard other columns:
                conflict_within_indices.extend(cha_df.index[1:])
            else:
                # Only instrument values differ, then it is likely a problem in time
                # ranges. Because they all overlap, set end times to not overlap next
                # start time (keep all columns):
                for i in range(len(cha_df) -1):
                    start_time = cha_df.at[cha_df.index[i + 1], start_col]
                    channels_df.at[cha_df.index[i], end_col] = start_time

            # # overlapping times. Check if other columns are qual:
            # col_equal = {
            #     c: len(cha_df[c].value_counts(dropna=False)) == 1
            #     for c in grp_other_cols if c not in {start_col, end_col}
            # }
            # if not all(col_equal.values()):
            #     for key, all_equal in col_equal.items():
            #         if not all_equal:
            #             if not pd.api.types.is_numeric_dtype(cha_df[key]):
            #                 break
            #             kwargs = {}
            #             if key in [Channel.latitude.key, Channel.longitude.key]:
            #                 kwargs = {'atol': 0.005, 'rtol': 0}  # atol 0.005 ~= 555 mt
            #             all_equal = col_equal[key] = np.allclose(
            #                 cha_df[key].iloc[0], cha_df[key].iloc[1:], **kwargs
            #             )
            #         if not all_equal:
            #             break
            # if not all(col_equal.values()):
            #     drop_indices = cha_df.index
            # else:
            #     drop_indices = cha_df.index[1:]
            #     idx = cha_df.index[0]
            #     channels_df.at[idx, start_col] = s_time.min()
            #     channels_df.at[idx, end_col] = e_time.max()

            # conflict_within_indices.extend(drop_indices)

            # # overlapping times, all other columns equal. Take min and max time, and
            # # discard others:
            # discard_indices = cha_df
            # cha_df.loc
            #
            #
            # # get how many end times match existing start times
            # other_idx = ~channels_df.index.isin(cha_df.index)
            # num_matches = np.array([
            #     (channels_df[other_idx][end_col] == cha_df.iloc[i][start_col]).sum()
            #     for i in range(len(cha_df))],
            #     dtype=int
            # )
            # # if some match and some don't remove those who don't:
            # if (num_matches > 0).any() and (num_matches ==0).any():
            #     cha_df = cha_df[num_matches > 0]
            #     conflict_within_indices.extend(cha_df[num_matches == 0].index)
            #
            # if len(cha_df) <= 1:
            #     continue
            #
            # # No conflict resolution. So take the row that has maximum time span.
            # # Though arbitrary and potentially
            # # overlapping with other rows, this way we might download once more which
            # # is preferable to skip some:
            # # get end times, replacing None (no end) to a random max time
            # end_t = cha_df[end_col].copy()
            # end_t[pd.isna(end_t)] = end_t.max().replace(end_t.max().year + 1)
            # time_ranges = end_t - cha_df[start_col]
            # drop_indices = cha_df.index.copy().delete(np.argmax(time_ranges))
            # conflict_within_indices.extend(drop_indices)

        channels_df = channels_df[
            ~channels_df.index.isin(conflict_within_indices)
        ].copy()

    return channels_df


def save_channels(engine: Engine, channels_df: pd.DataFrame, update: bool):
    """Saves to db channels (and their stations) and returns a dataframe with
    only channels saved. The returned Dataframe will have the column 'id'
    (`Station.id`) renamed to 'station_id' (`Channel.station_id`) and a new
    'id' column referring to the Channel id (`Channel.id`)

    :param channels_df: pandas DataFrame
    """
    if channels_df.empty:
        raise FailedDownload('No channel left after cleanup '
                             '(e.g., drop duplicates)')

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

    # # Add stations to db (Note: no need to check for `empty(channels_df)`,
    # # `dbsyncdf` raises a `FailedDownload` in case). First set columns
    # # defining channel identity (db unique constraint):
    # cols = [Station.network, Station.station, Station.webservice_id]
    # colnames = [c.key for c in cols]
    # # convert numeric values from channel level to station level using
    # # mean, min or max depending on column:
    # sta_df = []
    # for _, df_ in channels_df.groupby(colnames, sort=False, observed=False):
    #     if len(df_) > 1:
    #         # modify df_ first row and then take that 1st row only (df_ slice):
    #         i0 = df_.index[0]
    #         df_ = df_.copy()
    #         for c in [Station.latitude.key, Station.longitude.key,
    #                   Station.elevation.key]:
    #             df_.at[i0, c] = df_[c].mean()
    #         df_ = _adjust_times(df_)
    #     sta_df.append(df_)

    # # Then sync with db:
    # sta_df = dbsyncdf(pd.concat(sta_df, axis=0),
    #                   session, cols, Station.id, _update_stations,
    #                   buf_size=db_bufsize, keep_duplicates=False,
    #                   cols_to_print_on_err=colnames)
    # # `sta_df` will have the STA_ID columns, `channels_df` not: set it from the
    # # former to the latter:
    # channels_df = mergeupdate(channels_df, sta_df, colnames, [Station.id.key])
    # # rename now 'id' to 'station_id' before writing the channels to db:
    # channels_df.rename(columns={Channel.id.key: Channel.station_id.key}, inplace=True)

    # check channels with empty station id (should never happen, let's be
    # picky):
    # null_sta_id = channels_df[Channel.station_id.key].isnull()
    # conflict_null_sta_id = pd.DataFrame()
    # if null_sta_id.any():
    #     conflict_null_sta_id = channels_df[null_sta_id]
    #     channels_df = channels_df[~null_sta_id]

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
    channels_df[[
        Channel.band_code.key,
        Channel.instrument_code.key,
        Channel.orientation_code.key
    ]] = channels_df['channel_code'].str.extract(r'(.)(.)(.)').astype('category')
    channels_df, i_err, u_failed = df2db(
        channels_df, Channel, engine, Channel.id.key, uc_cols, update_cols
    )
    logger.info(f'{len(channels_df):,} of {(len(channels_df) + len(i_err)):,} '
                f'seismic channel(s) saved')
    if len(i_err):
        logger.warning(f"Unable to save {len(i_err)} seismic channel(s):")
        logger.warning(df2str(i_err))

    # # Then add (sync actually, already existing channels are not inserted):
    # channels_df = dbsyncdf(
    #     channels_df,
    #     session,
    #     cols,
    #     Channel.id,
    #     update,
    #     buf_size=db_bufsize,
    #     keep_duplicates=False,
    #     cols_to_print_on_err=[c. key for c in cols]
    # )

    # log_unsaved_channels(conflict_between, conflict_within)

    return channels_df


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
        logger.warning(df2str(cha_df[wsid_na]))
        cha_df = cha_df[~wsid_na].copy()
    return cha_df


# def setup_dataselect_urls(session, channels_df, authorizer: Authorizer = None):
#     """Prepares `cgannels_df` and `authorizer` for dataselct download, adding
#     urls and db id of the URLs to the former, and - if the latter is not None -
#     setting users and passwords (required for downloading) in it"""
#     ws_url_col = WebService.url.key
#     station_urls = channels_df[ws_url_col].cat.categories
#
#     url_mapping = {}  # station url -> dataselect_url
#     errors = set()
#
#     for url in station_urls:
#         method = 'query'
#         if authorizer is not None:
#             try:
#                 authorizer.add_url(url)
#                 method = 'queryauth'
#             except Exception as exc:
#                 logger.warning(formatmsg("Downloading open data only, "
#                                          "Unable to acquire credentials for "
#                                          "restricted data",
#                                          str(exc), url))
#                 errors.add(url)
#
#         url_mapping[url] = fdsn_url(url, new_service='dataselect', new_method=method)
#
#     if errors:
#         logger.info(formatmsg('Downloading open data only from: %s'
#                               % ", ".join(errors),
#                               'Unable to acquire credentials for '
#                               'restricted data'))
#
#     # replace station urls with new dataselect urls (query or queryauth methods):
#     channels_df[ws_url_col] = channels_df[ws_url_col].cat.rename_categories(url_mapping)
#
#     # remove webservice_id (which refers to FDSN station). FIXME: useless
#     # channels_df.drop(columns=[Channel.webservice_id.key], inplace=True)
#     # now set webservice id with the FDSN dataselect ids.
#
#     # Step1: get ids of the new dataselect urls (synch with db):
#     ws_df = pd.DataFrame([{'url': u} for u in url_mapping.values()])
#     ws_df = dbsyncdf(
#         ws_df, session, [WebService.url], WebService.id, buf_size=len(url_mapping),
#         keep_duplicates=False
#     )
#     ws_ids = dict(zip(ws_df['url'], ws_df['id']))
#
#     # Step 2: Extract the codes (an integer array of length N = channels_df rows).
#     # Each row gets an int (int8 / int16 / int32 depending on K = number of categories).
#     # Size: N integers (efficient, much smaller than N strings).
#     codes = channels_df[ws_url_col].cat.codes
#
#     # Step 3: Extract the categories (Index of all unique labels).
#     # This is tiny: only K elements.
#     categories = channels_df[ws_url_col].cat.categories
#
#     # Step 4: Build an array that maps category index -> ws_id.
#     # categories.map(ws_ids) creates a Series of length K (one id per category).
#     # .to_numpy() converts it to a NumPy array of length K.
#     codes_to_ids = categories.map(ws_ids).to_numpy()
#
#     # Step 5: Use the codes (length N) to index into codes_to_ids (length K).
#     # This produces a new integer array of length N, one ws_id per row.
#     channels_df[Segment.webservice_id.key] = codes_to_ids[codes]
#
#     return channels_df.copy()


# # FIXME REMOVE
# def _adjust_times(dfr: pd.DataFrame):
#     """Adjust start_time and ent_time in df_, returning a new single row dataframe
#     with min start_time, and max end_time (or NaT if any end time is NaT).
#
#     :param dfr: a DataFrame with ALL rows equal except start_time and end_time
#     """
#     i0 = dfr.index[0]
#     ret = dfr.loc[[i0], :].copy()  # [i0] => 1 row dataframe (i0 => p.Series)
#     ret.at[i0, Channel.start_time.key] = dfr[Channel.start_time.key].min()
#     ret.at[i0, Channel.end_time.key] = pd.NaT
#     if pd.notna(dfr[Channel.end_time.key]).all():
#         ret.at[i0, Channel.end_time.key] = dfr[Channel.end_time.key].max()
#     return ret


# def log_unsaved_channels(conflict_between, conflict_within):
#     """log the results of channels and station saving.
#
#     :param conflict_between: Dataframe of channels conflicts between
#         datacenters (duplicated stations returned by more than one datacenter)
#     :param conflict_within: Dataframe of channels conflicts within the same
#         datacenter (violating channels unique constraints)
#     """
#     max_row_count = 50
#     cols2show = [Channel.network_code.key, Channel.station_code.key]
#     if not conflict_between.empty:
#         # conflict_between happen at a station level (avoid unnecessary channel
#         # details):
#         _ = conflict_between.drop_duplicates(subset=cols2show,
#                                              keep='first')
#         msg = formatmsg('%d station(s) and %d channel(s) not saved to db' %
#                         (len(_), len(conflict_between)),
#                         'wrong datacenter detected using either Routing '
#                         'services or already saved stations')
#         logwarn_dataframe(_, msg, cols2show, max_row_count)
#
#     cols2show = [
#         Channel.network_code.key,
#         Channel.station_code.key,
#         Channel.location_code.key,
#         Channel.channel_code.key
#     ]
#     if not conflict_within.empty:
#         # Do not count stations here, as some of those stations might have been
#         # saved as part of other correct channels
#         msg = formatmsg('%d channel(s) not saved to db' % len(conflict_within),
#                         'conflicting data, e.g. unique constraint failed')
#         logwarn_dataframe(conflict_within, msg, cols2show, max_row_count)
#
#     # if not conflict_null_sta_id.empty:
#     #     # Do not count stations here, as some of those stations might have been saved as
#     #     # part of other correct channels
#     #     msg = formatmsg('%d channel(s) not saved to db' %
#     #                     len(conflict_null_sta_id),
#     #                     'station id not found, unknown cause')
#     #     logwarn_dataframe(conflict_null_sta_id, msg, cols2show, max_row_count)



# FIXME REMOVE
# def _get_channels_df_from_db(session, station_ws_db_id, net, sta, loc, cha,
#                             starttime, endtime):
#     """Return a Dataframe of the database channels according to the
#     arguments"""
#     # Select only relevant datacenters:
#     dc_be = Station.webservice_id == station_ws_db_id
#     # Select by starttime and endtime (below). Note that it must hold
#     # station.endtime > starttime AND station.starttime< endtime
#     stime_be = True
#     if starttime:
#         stime_be = ((Station.end_time == None) |
#                     (Station.end_time > starttime))
#     # endtime: Limit to metadata epochs ending on or before the specified end
#     # time. Note that station's ent_time can be None
#     etime_be = (Station.start_time < endtime) if endtime else True  # noqa
#     sa_cols = [Channel.id, Channel.station_id, Station.latitude,
#                Station.longitude, Station.start_time, Station.end_time,
#                Station.webservice_id, Station.network, Station.station,
#                Channel.sample_rate,
#                Channel.location, Channel.channel]
#     # filter on net, sta, loc, cha, as specified in config and converted to
#     # SQL-Alchemy binary expression:
#     nslc_be = get_sqla_binexp(net, sta, loc, cha)
#     # note below: binary expressions (all variables ending with "_be") might be
#     # the boolean True. SQL-Alchemy seems to understand them as long as they
#     # are preceded by a "normal" binary expression. Thus this works:
#     # `q.filter(binary_expr & True)` and is equal to `q.filter(binary_expr)`,
#     # whereas `q.filter(True & True)` (we hoped it could be a no-op filter)
#     # is not working as a no-op filter, it simply does not work at all.
#     # Here we should be safe cause `dc_be` is a non-True sql alchemy expression
#     # (see above):
#     qry = session.query(*sa_cols).join(Channel.station).filter(and_(dc_be,
#                                                                     nslc_be,
#                                                                     stime_be,
#                                                                     etime_be))
#     return dbquery2df(qry)
#
#
# def get_sqla_binexp(net, sta, loc, cha):
#     """Return the sql-alchemy binary expression to be used as argument for
#     database queries (e.g., `session.query(...)`) which translates to SQL the
#     given net(works), sta(tions), loc(ations) and cha(nnels), all iterable of
#     strings. Example:
#     ```
#     >>> get_sqla_binexp([], ['ABC'], [''], ['!A*', 'HH?', 'HN?'])
#     'sta=ABC&loc=&cha=HH?,HN?'
#     ```
#     Note negations (!A*) mean 'NOT' in this program's syntax (this feature is
#     not standard in an FDSN query).
#
#     Arguments are usually the output of
#     :func:`stream2segment.download.utils.nslc_lists`.
#
#     :param net: an iterable of strings denoting networks.
#     :param sta: an iterable of strings denoting stations.
#     :param loc: an iterable of strings denoting locations.
#     :param cha: an iterable of strings denoting channels.
#     """
#     # build a sql alchemy filter condition
#     sa_cols = (Station.network, Station.station, Channel.location,
#                Channel.channel)
#
#     sa_bin_exprs = []
#
#     wild2sql = strconvert.wild2sql  # conversion function
#
#     for column, lst in zip(sa_cols, (net, sta, loc, cha)):
#         matches = []
#         for string in lst:
#             negate = False
#             if string[0:1] == '!':
#                 negate = True
#                 string = string[1:]
#
#             if '?' in string or '*' in string:
#                 condition = column.like(wild2sql(string))
#             else:
#                 condition = (column == string)
#
#             if negate:
#                 condition = ~condition
#
#             matches.append(condition)
#
#         if matches:
#             sa_bin_exprs.append(or_(*matches))
#
#     return True if not sa_bin_exprs else and_(*sa_bin_exprs)

