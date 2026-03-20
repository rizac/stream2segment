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
# from sqlalchemy import or_, and_

from stream2segment.io.cli import get_progressbar
from stream2segment.io.db.pdsql import shared_colnames  # dbquery2df, , mergeupdate
from stream2segment.io.db.models import Channel, WebService, Segment
from stream2segment.download.exc import FailedDownload
from stream2segment.download.url import urlread, get_host
from stream2segment.download.modules.utils import (fdsn_channel_response_text_to_df,
                                                   dbsyncdf, formatmsg, fdsn_url,
                                                   logwarn_dataframe, strconvert,
                                                   Authorizer, fdsn_url_qs)

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
        for response in t_pool.imap_unordered(urlread, urls):
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
        ws_df = dbsyncdf(
            pd.DataFrame([{ws_url_col: u} for u in cha_df[ws_url_col].cat.categories]),
            session,
            [WebService.url],
            WebService.id,
            buf_size=db_bufsize or len(urls),
            keep_duplicates=False
        )
        # now assign:
        cha_df = cha_df.merge(
            ws_df.rename(columns={"id": "webservice_id"}), on=ws_url_col, how="left"
        )
        # post filter (negation "!", sample rate) which raises FailedDownload if no rows:
        cha_df = filter_out_channels_df(
            cha_df, net, sta, loc, cha, min_sample_rate
        )
        cha_df = drop_duplicates(session, cha_df)
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
    for c in (Channel.network_code.key, Channel.channel_code.key,
              Channel.location_code.key,
              Channel.station_code.key, ws_url_col):
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
    _update_stations = update
    if _update_stations:
        _update_stations = list(shared_colnames(Channel, channels_df, pkey=False))

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
    cols = [Channel.station_id, Channel.location, Channel.channel]
    colnames = [Channel.network_code.key, Channel.station_code.key,
                Channel.location_code.key, Channel.channel_code.key]
    # Then add (sync actually, already existing channels are not inserted):
    channels_df = dbsyncdf(
        channels_df,
        session,
        cols,
        Channel.id,
        update,
        buf_size=db_bufsize,
        keep_duplicates=False,
        cols_to_print_on_err=colnames
    )

    log_unsaved_channels(conflict_between, conflict_within)

    return channels_df


def drop_duplicates(session, channels_df):
    """Drop from channels_df duplicates (same station between or within data
    centers). For duplicates between data centers, uses `eidavalidator` or the
    database accessible via the session object, if `eidavalidator` is None.

    :return: the tuple of Dataframes:
        `(oks, conflict_between_dc, conflict_within_dc)`, where `oks` is a
        subset of `channels_df` with valid channels (one row per channel), and
        the other two contain channels discarded: `conflict between_dc`
        contains channels whose station is associated to several dc_id
        (datacenter id) and `conflict_within_dc` contains channels of the same
        dc_id violating a database unique constraint, e.g. two different
        channels with the same (network, station, location, channel,
        start_time, dc_id).
    """
    # conflict between case is when station webservice is not unique, e.g.:
    #   net sta webservice_id
    #   N   S   1
    #   N   S   2

    # conflict_between_dc = []  # add here unresolvable conflicts

    # Conflict within is when the same channel has same net sta loc cha webservice_id start_time.
    # In this case, choose the one that has a date matching with another one, if exist,
    # or the most recent one if not


    # oks = []
    # station_datacenters_from_db = None  # dataframe lazy loaded (see below)

    # first drop duplicates (all columns the same):
    # this method does very few things as there might be rounding errors that
    # prevent equal columns to be equalk. Anyway, we perform here more sound checks
    channels_df = channels_df.drop_duplicates(keep='first').reset_index(drop=True)

    # Just for ref, this rows are not detected by duplicated (apparently, scale differs):
    #        network_code station_code location_code channel_code   latitude  longitude  elevation  depth  azimuth  dip          scale  scale_freq scale_units  sample_rate          start_time   end_time                                              url  webservice_id
    # 97463            GS        KAN12            01          HNE  37.297383    -97.998      425.5    0.0     90.0  0.0  184349.032785         1.0      m/s**2        200.0 2014-05-08 17:11:06 2015-04-14  https://service.iris.edu/fdsnws/station/1/query             11
    # 131397           GS        KAN12            01          HNE  37.297383    -97.998      425.5    0.0     90.0  0.0  184349.032785         1.0      m/s**2        200.0 2014-05-08 17:11:06 2015-04-14  https://service.iris.edu/fdsnws/station/1/query             11
    #

    grp1_cols = [
        Channel.network_code.key,
        Channel.station_code.key,
        Channel.location_code.key,
        Channel.channel_code.key,
    ]
    webs_id_col = Channel.webservice_id.key
    grp2_cols = [
        Channel.network_code.key,
        Channel.station_code.key,
        Channel.location_code.key,
        Channel.channel_code.key,
        webs_id_col,
        Channel.start_time.key,
    ]
    grp2_other_cols = list(channels_df.columns.difference(grp2_cols))

    conflict_between = (
        channels_df.groupby(grp1_cols)[webs_id_col].transform("nunique") > 1
    )
    (channels_df[conflict_between].sort_values(grp1_cols, ascending=True).
    to_csv(
        path_or_buf='/Users/rizac/work/code/stream2segment/conflict_between.csv',
        index=False
    ))

    conflict_within = (
        channels_df.groupby(grp2_cols)[grp2_other_cols].transform("nunique") > 1
    ).any(axis=1)
    (channels_df[conflict_within].sort_values(grp2_cols, ascending=True).
    to_csv(
        path_or_buf='/Users/rizac/work/code/stream2segment/conflict_within.csv',
        index=False
    ))

    keep_first = True

    conflict_between_indices = []
    if conflict_between.any():
        for (net, sta, loc, cha), cha_df in channels_df[conflict_between].groupby(
            grp1_cols, sort=False
        ):
            real_dc_ids = set(
                _[0] for _ in session.query(Channel.webservice_id).filter(
                    (Channel.network_code == net) & (Channel.station_code == sta)
                ).all()
            )

            if len(real_dc_ids) != 1:
                # conflict found, unresolvable through already saved data.
                # The real datacenter ids are more than one => we can
                # not save the station: empty (=> discard) the dataframe
                if keep_first:
                    conflicting = cha_df[1:]
                else:
                    conflicting = cha_df
            else:
                # Conflict found, resolved through already saved data. The real webservice
                # id is one => discard all (net, sta, stime) with different webservices id:
                conflicting = cha_df[cha_df[webs_id_col] != next(iter(real_dc_ids))]

            if not conflicting.empty:
                conflict_between_indices.extend(conflicting.index)
                # channels_df.loc[conflicting.index, 'conflict_between'] = True

    # conflict within
    conflict_within_indices = []
    if conflict_between.any():
        for _, cha_df in channels_df[conflict_within].groupby(
            grp2_cols, sort=False
        ):
            if len(cha_df) <= 1:
                continue

            # take the row that has maximum time span. Though arbitrary and potentially
            # overlapping with other rows, this way we might download once more which
            # is preferable to skip some:
            start_t = cha_df[Channel.start_time.key].copy()
            # get end times, replacing None (no end) to a random max time
            end_t = cha_df[Channel.end_time.key].copy()
            end_t[pd.isna(end_t)] = end_t.max().replace(end_t.max().year + 1)
            idx = np.argmax(end_t - start_t)
            cha_df = cha_df.drop(index=cha_df.index[idx])
            conflict_within_indices.extend(cha_df.index)
            # channels_df.loc[conflicting.index, 'conflict_between'] = True

    #
    # start_col = Channel.start_time.key
    # end_col = Channel.end_time.key
    # all_non_time_cols = [c for c in channels_df.columns if c not in {start_col, end_col}]

    # check conflicts
    # for (net, sta), cha_df in channels_df.groupby(grp1_cols, sort=False):
    #
    #     # check conflict between:
    #     if len(pd.unique(cha_df[webs_id_col])) > 1:
    #         # We have more than one data center mapped to the tuple
    #         # (net, sta): get all ids from the db:
    #         real_dc_ids = set(
    #             _[0] for _ in session.query(Channel.webservice_id).filter(
    #                 (Channel.network_code == net) & (Channel.station_code == sta)
    #             ).all()
    #         )
    #
    #         # stmt = select(distinct(Channel.webservice_id)).where(
    #         #     (Channel.network_code == net) & (Channel.station_code == sta)
    #         # )
    #         #
    #         # # Execute and fetch unique IDs
    #         # ids = [row[0] for row in conn.execute(stmt)]
    #
    #         if len(real_dc_ids) != 1:
    #             # conflict found, unresolvable through already saved data.
    #             # The real datacenter ids are more than one => we can
    #             # not save the station: empty (=> discard) the dataframe
    #             conflicting = cha_df
    #         else:
    #             # Conflict found, resolved through already saved data. The real webservice
    #             # id is one => discard all (net, sta, stime) with different webservices id:
    #             conflicting = cha_df[cha_df[webs_id_col] != next(iter(real_dc_ids))]
    #
    #         if not conflicting.empty:
    #             channels_df.loc[conflicting.index, 'conflict_between'] = True
    #
    #         # conflict_between_dc.append(cha_df[conflicting])
    #         # if conflicting.all():  # noqa
    #         #     continue
    #         # cha_df = cha_df[~conflicting]
    #
    #     # Check conflicts within:
    #     for _, cha_df_tmp in cha_df.groupby(grp2_cols, sort=False):
    #         if len(cha_df_tmp) > 1:
    #             # df_sorted = cha_df_tmp.sort_values(Channel.start_time.key)
    #             channels_df.loc[cha_df_tmp.index, 'conflict_within'] = True
    #
    #
    #
    #     # dupes = df_.duplicated(subset=nslc_cols, keep=False)
    #     # if dupes.any():
    #     #     tmp_ = []
    #     #     for _, df__ in df_.groupby(nslc_cols, sort=False, observed=False):
    #     #         if len(df__) > 1:
    #     #             if not all(len(pd.unique(df__[c])) == 1 for c in all_non_time_cols):
    #     #                 conflict_within_dc.append(df__)
    #     #                 continue
    #     #             # else:
    #     #             #     # we still need to provide a single row dataframe with times
    #     #             #     # adjusted, otherwise channels might not be saved due to db
    #     #             #     # constraints (the same time adjustment will be performed to
    #     #             #     # assess the station times from its channels):
    #     #             #     df__ = _adjust_times(df__)
    #     #         tmp_.append(df__)
    #     #     if not tmp_:
    #     #         continue
    #     #     df_ = pd.concat(tmp_, axis=0)
    #
    #     # oks.append(cha_df)

    conflict_between = channels_df.index.isin(conflict_between_indices)
    conflict_within = channels_df.index.isin(conflict_within_indices)
    return (
        channels_df[~(conflict_between | conflict_within)],
        channels_df.loc[conflict_between],
        channels_df.loc[conflict_within],
    )
    oks = pd.DataFrame() if not oks else \
        pd.concat(oks, axis=0, sort=False, ignore_index=True, copy=True)
    conflict_between_dc = pd.DataFrame() if not conflict_between_dc else \
        pd.concat(conflict_between_dc, axis=0, sort=False)
    conflict_within_dc = pd.DataFrame() if not conflict_within_dc else \
        pd.concat(conflict_within_dc, axis=0, sort=False)

    return oks, conflict_between_dc, conflict_within_dc


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


def log_unsaved_channels(conflict_between, conflict_within):
    """log the results of channels and station saving.

    :param conflict_between: Dataframe of channels conflicts between
        datacenters (duplicated stations returned by more than one datacenter)
    :param conflict_within: Dataframe of channels conflicts within the same
        datacenter (violating channels unique constraints)
    """
    max_row_count = 50
    cols2show = [Channel.network_code.key, Channel.station_code.key]
    if not conflict_between.empty:
        # conflict_between happen at a station level (avoid unnecessary channel
        # details):
        _ = conflict_between.drop_duplicates(subset=cols2show,
                                             keep='first')
        msg = formatmsg('%d station(s) and %d channel(s) not saved to db' %
                        (len(_), len(conflict_between)),
                        'wrong datacenter detected using either Routing '
                        'services or already saved stations')
        logwarn_dataframe(_, msg, cols2show, max_row_count)

    cols2show = [
        Channel.network_code.key,
        Channel.station_code.key,
        Channel.location_code.key,
        Channel.channel_code.key
    ]
    if not conflict_within.empty:
        # Do not count stations here, as some of those stations might have been
        # saved as part of other correct channels
        msg = formatmsg('%d channel(s) not saved to db' % len(conflict_within),
                        'conflicting data, e.g. unique constraint failed')
        logwarn_dataframe(conflict_within, msg, cols2show, max_row_count)

    # if not conflict_null_sta_id.empty:
    #     # Do not count stations here, as some of those stations might have been saved as
    #     # part of other correct channels
    #     msg = formatmsg('%d channel(s) not saved to db' %
    #                     len(conflict_null_sta_id),
    #                     'station id not found, unknown cause')
    #     logwarn_dataframe(conflict_null_sta_id, msg, cols2show, max_row_count)


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
