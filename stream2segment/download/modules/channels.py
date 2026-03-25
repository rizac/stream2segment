"""
Stations/Channels download functions

:date: Dec 3, 2017
"""
import re
import logging
from itertools import combinations
from multiprocessing.pool import ThreadPool
from urllib.parse import urlunparse, urlparse

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_categorical_dtype
from sqlalchemy.dialects.mssql.information_schema import columns

# from sqlalchemy import or_, and_

from stream2segment.io.cli import get_progressbar
from stream2segment.io.db.pdsql import shared_colnames, \
    Inserter, sync_pkey, df2db  # dbquery2df, , mergeupdate
from stream2segment.io.db.models import Channel, WebService, Segment
from stream2segment.download.exc import FailedDownload
from stream2segment.download.url import urlread, get_host
from stream2segment.download.modules.utils import (fdsn_channel_response_text_to_df,
                                                   formatmsg, fdsn_url,
                                                   logwarn_dataframe, strconvert,
                                                   Authorizer, fdsn_url_qs, df2str)

# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial):
logger = logging.getLogger(__name__)


def get_channels_df(session, fdsn_station_urls, net, sta, loc, cha,
                    starttime, endtime, min_sample_rate, update,
                    max_thread_workers, timeout, blocksize, db_bufsize,
                    show_progress=False):
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
    def _urlread(url):
        return urlread(url, timeout=timeout, blocksize=blocksize)

    urls = list(fdsn_station_urls)

    channels_dfs = []
    failed_dframe_rows = []
    station_urls = set()
    with get_progressbar(len(urls) if show_progress else 0) as pbar:
        for response in t_pool.imap_unordered(_urlread, urls):
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
        # assign the webservice ids. First create / get those ids:

        ws_df, i_err, _ = df2db(
            pd.DataFrame([{ws_url_col: u} for u in cha_df[ws_url_col].cat.categories]),
            WebService,
            session.get_bind(),
            'id',
            [ws_url_col]
        )
        logger.info(f'{len(ws_df):,} of {(len(ws_df) + len(i_err)):,} station url(s) saved')
        if len(i_err):
            logger.warning(f"Unable to save {len(i_err)} station url(s):")
            logger.warning(df2str(i_err))
        # now assign:
        cha_df = cha_df.merge(
            ws_df.rename(columns={"id": Channel.webservice_id.key}),
            on=ws_url_col,
            how="left"
        )
        wsid_na = pd.isna(cha_df[Channel.webservice_id.key])
        if wsid_na.any():
            logger.warning(f"Unable to get station urls for {wsid_na.sum()} channel(s) "
                           f"(discarding):")
            logger.warning(df2str(cha_df[wsid_na]))
            cha_df = cha_df[~wsid_na].copy()

        # post filter (negation "!", sample rate) which raises FailedDownload if no rows:
        cha_df = filter_out_channels_df(
            cha_df, net, sta, loc, cha, min_sample_rate
        )

        # first drop duplicates (all columns the same):
        # this method does very few things as there might be rounding errors that
        # prevent equal columns to be equal. Anyway, we perform here more sound checks
        cha_df = cha_df.drop_duplicates(keep='first').reset_index(drop=True)

        # set ranking based on the order of urls
        cha_df = drop_conflict_between(session, cha_df, urls)

        cha_df = drop_conflict_within(cha_df)

        cha_df = save_channels(session, cha_df, update, db_bufsize)

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


def filter_out_channels_df(channels_df, net, sta, loc, cha, min_sample_rate):
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
        lst = [_ for _ in lst if _[0:1] == '!']  # take only negation expr.
        if not lst:
            continue
        # condition = ("^%s$" if len(lst) == 1 else "^(?:%s)$") % \
        #     "|".join(strconvert.wild2re(x[1:]) for x in lst)
        condition = "|".join(f"^(?:{strconvert.wild2re(x[1:])})$" for x in lst)
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


def drop_conflict_between(session, channels_df, urls:list[str] = None):
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
    # conflict between case is when station webservice is not unique, e.g.:
    #   net sta webservice_id
    #   N   S   1
    #   N   S   2

    grp_cols = [
        Channel.network_code.key,
        Channel.station_code.key,
        Channel.location_code.key,
        Channel.channel_code.key,
    ]
    webs_id_col = Channel.webservice_id.key

    conflict_between = (
        channels_df.groupby(grp_cols)[webs_id_col].transform("nunique") > 1
    )
    # (channels_df[conflict_between].sort_values(grp1_cols, ascending=True).
    # to_csv(
    #     path_or_buf=os.expanduser('~/work/code/stream2segment/conflict_between.csv'),
    #     index=False
    # ))

    # log messages:
    if conflict_between.any():
        log_df = channels_df[conflict_between].groupby(
            grp_cols[:2], as_index=False
        ).agg({ WebService.url.key: lambda x: ", ".join(sorted(set(x))) })
        logger.warning(
            f'Conflict: different URLs returning the same station. Conflicts summary:'
        )
        columns2show = grp_cols[:2] + ['URLs (should be 1)']
        log_df = log_df.rename(columns={WebService.url.key: columns2show[-1]})
        logger.warning(
            log_df[columns2show].sort_values(by=columns2show).to_string(
                na_rep='', index=False
            )
        )

        conflict_between_indices = []
        # although we check conflicts by net.sta.loc.cha, we are interested in fixing
        # the station problem here
        rank_col_name = None
        if urls is not None:
            rank_col_name = '_rank'
            while rank_col_name in channels_df.columns:
                rank_col_name += '_'
            urls_rank = np.full(len(channels_df), np.iinfo(int).max, dtype=int)
            _urls_done = set()
            for url in urls:
                url = url[:url.find("?")] if "?" in url else url  # no query string
                if url in _urls_done:
                    continue
                _urls_done.add(url)
                url_rank = len(_urls_done)
                urls_rank[channels_df[WebService.url.key].str.startswith(url)] = url_rank
            channels_df[rank_col_name] = urls_rank

        net_sta_df = channels_df.loc[conflict_between, grp_cols[:2]]
        net_sta_df = net_sta_df.drop_duplicates(keep='first')
        for (net, sta) in net_sta_df.itertuples(index=False, name=None):
            real_dc_ids = set(
                _[0] for _ in session.query(Channel.webservice_id).filter(
                    (Channel.network_code == net) & (Channel.station_code == sta)
                ).all()
            )
            conflicting = channels_df.loc[
                conflict_between &
                (channels_df[Channel.network_code.key] == net) &
                (channels_df[Channel.station_code.key] == sta),
                :
            ]

            real_ws_id = None
            if len(real_dc_ids) == 1:
                real_ws_id = next(iter(real_dc_ids))
            elif rank_col_name is not None:
                # conflict found, unresolvable through already saved data. Get first
                # if instructed to do so
                real_ws_id = conflicting[
                    conflicting[rank_col_name] == conflicting[rank_col_name].min()
                ].iloc[0][webs_id_col]

            if real_ws_id is not None:
                # Conflict found, resolved through already saved data. The real webservice
                # id is one => discard all (net, sta, stime) with different webservices id:
                conflicting = conflicting[conflicting[webs_id_col] != real_ws_id]

            if not conflicting.empty:
                conflict_between_indices.extend(conflicting.index)
                # channels_df.loc[conflicting.index, 'conflict_between'] = True

        if rank_col_name is not None:
            channels_df = channels_df.drop(columns=rank_col_name)

        channels_df = channels_df[
            ~channels_df.index.isin(conflict_between_indices)
        ].copy()

    return channels_df


def drop_conflict_within(channels_df):
    """
    Drop from channels_df conflict within, i.e., same
    network.station.location.channel.start_time  returned by the same URLs.
    Duplicated rows will be resolved by taking the item which spans the
    biggest time range (which is the least bad option)

    :return: a new dataframe with duplicated rows removed
    """
    # conflict within
    webs_id_col = Channel.webservice_id.key
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
        webs_id_col,
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

            if cha_df.station_code.iloc[0] ==  'KAN12':
                asd = 9
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

def save_channels(session, channels_df, update, db_bufsize):
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
        Channel.channel_code.key,
        Channel.start_time.key,
    ]
    channels_df, i_err, u_failed = df2db(
        channels_df, Channel, session.get_bind(), Channel.id.key, uc_cols, update_cols
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


def setup_dataselect_urls(session, channels_df, authorizer: Authorizer = None):
    """Prepares `cgannels_df` and `authorizer` for dataselct download, adding
    urls and db id of the URLs to the former, and - if the latter is not None -
    setting users and passwords (required for downloading) in it"""
    ws_url_col = WebService.url.key
    station_urls = channels_df[ws_url_col].cat.categories

    url_mapping = {}  # station url -> dataselect_url
    errors = set()

    for url in station_urls:
        method = 'query'
        if authorizer is not None:
            try:
                authorizer.add_url(url)
                method = 'queryauth'
            except Exception as exc:
                logger.warning(formatmsg("Downloading open data only, "
                                         "Unable to acquire credentials for "
                                         "restricted data",
                                         str(exc), url))
                errors.add(url)

        url_mapping[url] = fdsn_url(url, new_service='dataselect', new_method=method)

    if errors:
        logger.info(formatmsg('Downloading open data only from: %s'
                              % ", ".join(errors),
                              'Unable to acquire credentials for '
                              'restricted data'))

    # replace station urls with new dataselect urls (query or queryauth methods):
    channels_df[ws_url_col] = channels_df[ws_url_col].cat.rename_categories(url_mapping)

    # remove webservice_id (which refers to FDSN station). FIXME: useless
    # channels_df.drop(columns=[Channel.webservice_id.key], inplace=True)
    # now set webservice id with the FDSN dataselect ids.

    # Step1: get ids of the new dataselect urls (synch with db):
    ws_df = pd.DataFrame([{'url': u} for u in url_mapping.values()])
    ws_df = dbsyncdf(
        ws_df, session, [WebService.url], WebService.id, buf_size=len(url_mapping),
        keep_duplicates=False
    )
    ws_ids = dict(zip(ws_df['url'], ws_df['id']))

    # Step 2: Extract the codes (an integer array of length N = channels_df rows).
    # Each row gets an int (int8 / int16 / int32 depending on K = number of categories).
    # Size: N integers (efficient, much smaller than N strings).
    codes = channels_df[ws_url_col].cat.codes

    # Step 3: Extract the categories (Index of all unique labels).
    # This is tiny: only K elements.
    categories = channels_df[ws_url_col].cat.categories

    # Step 4: Build an array that maps category index -> ws_id.
    # categories.map(ws_ids) creates a Series of length K (one id per category).
    # .to_numpy() converts it to a NumPy array of length K.
    codes_to_ids = categories.map(ws_ids).to_numpy()

    # Step 5: Use the codes (length N) to index into codes_to_ids (length K).
    # This produces a new integer array of length N, one ws_id per row.
    channels_df[Segment.webservice_id.key] = codes_to_ids[codes]

    return channels_df.copy()


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

