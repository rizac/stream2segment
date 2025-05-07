"""
Stations/Channels download functions

:date: Dec 3, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
import re
import logging

import pandas as pd
from sqlalchemy import or_, and_

from stream2segment.io.cli import get_progressbar
from stream2segment.io.db.pdsql import dbquery2df, shared_colnames, mergeupdate
from stream2segment.download.db.models import Station, Channel
from stream2segment.download.exc import FailedDownload
from stream2segment.download.url import read_async_new
from stream2segment.download.modules.utils import (get_dataframe_from_fdsn,
                                                   dbsyncdf, formatmsg,
                                                   logwarn_dataframe, strconvert)

# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial):
logger = logging.getLogger(__name__)


def get_channels_df(session, datacenters_df, net, sta, loc, cha,
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

    urls = [
        f"{r.station_url}?net={r.net}&sta={r.sta}"
        f"&loc={r.loc}&cha={r.cha}&start={r.start}&end={r.end}"
        for r in datacenters_df.itertuples(index=False)
    ]

    ret = []
    url_failed_dc_ids = []
    from_db = 0
    with get_progressbar(len(urls) if show_progress else 0) as pbar:
        for idx, url, result, exc, status_code in \
                read_async_new(urls, blocksize=blocksize,
                               max_workers=max_thread_workers, decode='utf8',
                               timeout=timeout):
            pbar.update(1)
            if exc:
                logger.warning(formatmsg("Unable to fetch stations", exc, url))
                dframe = get_channels_df_from_db(session,
                                                 datacenters_df['station_ws_id'][idx],
                                                 net, sta, loc, cha, starttime, endtime,
                                                 min_sample_rate)
                logger.warning(formatmsg("Fetching stations from database: "
                                         f"{len(dframe)} stations found"))
                from_db += len(dframe)
            else:
                try:
                    dframe = get_dataframe_from_fdsn(result, "channel", url)
                except ValueError as verr:
                    logger.warning(formatmsg("Discarding response data", verr,
                                             url))
                    continue

            if not dframe.empty:
                for c in ['dataselect_ws_id', 'dataselect_url',
                          'station_ws_id', 'station_url']:
                    dframe[c] = pd.Series(
                        [datacenters_df[c][idx]] * len(dframe),
                        dtype=datacenters_df[c].dtype
                    )
                ret.append(dframe)

    if from_db > 0:
        logger.info(formatmsg(f"Fetching stations data from database for {from_db} "
                              "station(s)", "download errors occurred"))

    # build two dataframes which we will concatenate afterwards
    cha_df = pd.DataFrame()
    if ret:  # pd.concat complains for empty list
        cha_df = filter_out_channels_df(pd.concat(ret, axis=0,
                                                  ignore_index=True,
                                                  copy=False),
                                        net, sta, loc, cha,
                                        min_sample_rate)

        # this raises FailedDownload if we cannot save any element:
        cha_df = save_stations_and_channels(session, cha_df, update, db_bufsize)

    if cha_df.empty:
        # ok, now let's see if we have remaining datacenters to be
        # fetched from the db
        raise FailedDownload(formatmsg("No station found",
                                       "Unable to fetch stations from all "
                                       "data-centers, no data to fetch from "
                                       "the database. Check config and log "
                                       "for details"))

    # the columns for the channels dataframe that will be returned
    ret = cha_df[[c.key for c in (Channel.id, Channel.station_id,
                                  Station.latitude, Station.longitude,
                                  Station.webservice_id, Station.start_time,
                                  Station.end_time, Station.network,
                                  Station.station, Channel.location,
                                  Channel.channel, 'dataselect_ws_url')]]
    # convert to categorical type:
    for c in (Station.network, Station.station, Channel.location, Channel.channel,
              'dataselect_ws_url'):
        ret[c.key] = ret[c.key].astype('category')
    # return a copy:
    return ret.copy()


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
    dffilter = None
    sa_cols = (Station.network, Station.station, Channel.location,
               Channel.channel)

    for lst, sa_col in zip((net, sta, loc, cha), sa_cols):
        if not lst:
            continue
        lst = [_ for _ in lst if _[0:1] == '!']  # take only negation expr.
        if not lst:
            continue
        condition = ("^%s$" if len(lst) == 1 else "^(?:%s)$") % \
            "|".join(strconvert.wild2re(x[1:]) for x in lst)
        flt = channels_df[sa_col.key].str.match(re.compile(condition))
        if dffilter is None:
            dffilter = flt
        else:
            dffilter &= flt

    if min_sample_rate > 0:
        # account for Nones, thus negate the predicate below:
        flt = ~(channels_df[Channel.sample_rate.key] >= min_sample_rate)
        if dffilter is None:
            dffilter = flt
        else:
            dffilter &= flt

    ret = channels_df if dffilter is None else \
        channels_df[~dffilter].copy()  # noqa

    if ret.empty:
        raise FailedDownload("No channel matches user defined filters "
                             "(network, channel, sample rate, ...)")

    discarded_sr = len(channels_df) - len(ret)
    if discarded_sr:
        logger.warning("%d channel(s) discarded according to current "
                       "configuration filters (network, channel, sample rate, "
                       "...)", discarded_sr)

    return ret


def get_channels_df_from_db(session, station_ws_db_id, net, sta, loc, cha,
                            starttime, endtime, min_sample_rate):
    """Return a Dataframe of the database channels according to the
    arguments"""
    # Build SQL-Alchemy binary expressions (suffix '_be'), i.e. an object
    # reflecting a SQL clause)
    srate_be = True
    if min_sample_rate > 0:
        srate_be = Channel.sample_rate >= min_sample_rate
    # Select only relevant datacenters:
    dc_be = Station.webservice_id == station_ws_db_id
    # Select by starttime and endtime (below). Note that it must hold
    # station.endtime > starttime AND station.starttime< endtime
    stime_be = True
    if starttime:
        stime_be = ((Station.end_time == None) |
                    (Station.end_time > starttime))
    # endtime: Limit to metadata epochs ending on or before the specified end
    # time. Note that station's ent_time can be None
    etime_be = (Station.start_time < endtime) if endtime else True  # noqa
    sa_cols = [Channel.id, Channel.station_id, Station.latitude,
               Station.longitude, Station.start_time, Station.end_time,
               Station.webservice_id, Station.network, Station.station,
               Channel.location, Channel.channel]
    # filter on net, sta, loc, cha, as specified in config and converted to
    # SQL-Alchemy binary expression:
    nslc_be = get_sqla_binexp(net, sta, loc, cha)
    # note below: binary expressions (all variables ending with "_be") might be
    # the boolean True. SQL-Alchemy seems to understand them as long as they
    # are preceded by a "normal" binary expression. Thus this works:
    # `q.filter(binary_expr & True)` and is equal to `q.filter(binary_expr)`,
    # whereas `q.filter(True & True)` (we hoped it could be a no-op filter)
    # is not working as a no-op filter, it simply does not work at all.
    # Here we should be safe cause `dc_be` is a non-True sql alchemy expression
    # (see above):
    qry = session.query(*sa_cols).join(Channel.station).filter(and_(dc_be,
                                                                    srate_be,
                                                                    nslc_be,
                                                                    stime_be,
                                                                    etime_be))
    return dbquery2df(qry)


def get_sqla_binexp(net, sta, loc, cha):
    """Return the sql-alchemy binary expression to be used as argument for
    database queries (e.g., `session.query(...)`) which translates to SQL the
    given net(works), sta(tions), loc(ations) and cha(nnels), all iterable of
    strings. Example:
    ```
    >>> get_sqla_binexp([], ['ABC'], [''], ['!A*', 'HH?', 'HN?'])
    'sta=ABC&loc=&cha=HH?,HN?'
    ```
    Note negations (!A*) mean 'NOT' in this program's syntax (this feature is
    not standard in an FDSN query).

    Arguments are usually the output of
    :func:`stream2segment.download.utils.nslc_lists`.

    :param net: an iterable of strings denoting networks.
    :param sta: an iterable of strings denoting stations.
    :param loc: an iterable of strings denoting locations.
    :param cha: an iterable of strings denoting channels.
    """
    # build a sql alchemy filter condition
    sa_cols = (Station.network, Station.station, Channel.location,
               Channel.channel)

    sa_bin_exprs = []

    wild2sql = strconvert.wild2sql  # conversion function

    for column, lst in zip(sa_cols, (net, sta, loc, cha)):
        matches = []
        for string in lst:
            negate = False
            if string[0:1] == '!':
                negate = True
                string = string[1:]

            if '?' in string or '*' in string:
                condition = column.like(wild2sql(string))
            else:
                condition = (column == string)

            if negate:
                condition = ~condition

            matches.append(condition)

        if matches:
            sa_bin_exprs.append(or_(*matches))

    return True if not sa_bin_exprs else and_(*sa_bin_exprs)


class ST:  # noqa
    """Simple enum-like container of strings defining the station's related
    database/dataframe columns needed in this module
    """
    ID = Station.id.key  # pylint: disable=invalid-name
    NET = Station.network.key  # pylint: disable=invalid-name
    STA = Station.station.key  # pylint: disable=invalid-name
    STIME = Station.start_time.key  # pylint: disable=invalid-name
    ETIME = Station.end_time.key  # pylint: disable=invalid-name
    WSID = Station.webservice_id.key  # pylint: disable=invalid-name
    # set columns to show in the log on error ("no row written"):
    ERRCOLS = [NET, STA, STIME, WSID]  # pylint: disable=invalid-name


class CH:  # noqa
    """Simple enum-like container of strings defining the channel's related
    database/dataframe columns needed in this module
    """
    ID = Channel.id.key  # pylint: disable=invalid-name
    STAID = Channel.station_id.key  # pylint: disable=invalid-name
    LOC = Channel.location.key  # pylint: disable=invalid-name
    CHA = Channel.channel.key  # pylint: disable=invalid-name
    # set columns to show in the log on error ("no row written"):
    ERRCOLS = [ST.NET, ST.STA, LOC, CHA, ST.STIME, ST.WSID]  # noqa


def save_stations_and_channels(session, channels_df, update, db_bufsize):
    """Saves to db channels (and their stations) and returns a dataframe with
    only channels saved. The returned Dataframe will have the column 'id'
    (`Station.id`) renamed to 'station_id' (`Channel.station_id`) and a new
    'id' column referring to the Channel id (`Channel.id`)

    :param channels_df: pandas DataFrame
    """
    channels_df, conflict_between, conflict_within = \
        drop_duplicates(session, channels_df)

    if channels_df.empty:
        raise FailedDownload('No channel left after cleanup '
                             '(e.g., drop duplicates)')

    # if update is True, don't update inventories HERE (handled later)
    _update_stations = update
    if _update_stations:
        _update_stations = [_ for _ in shared_colnames(Station, channels_df,
                                                       pkey=False)
                            if _ != Station.stationxml.key]

    # Add stations to db (Note: no need to check for `empty(channels_df)`,
    # `dbsyncdf` raises a `FailedDownload` in case). First set columns
    # defining channel identity (db unique constraint):
    cols = [Station.network, Station.station, Station.start_time]
    # Then add (sync actually, already existing stations are not inserted):
    sta_df = dbsyncdf(channels_df.drop_duplicates(subset=[ST.NET, ST.STA, ST.WSID]),
                      session, cols, Station.id, _update_stations,
                      buf_size=db_bufsize, keep_duplicates=False,
                      cols_to_print_on_err=ST.ERRCOLS)
    # `sta_df` will have the STA_ID columns, `channels_df` not: set it from the
    # former to the latter:
    channels_df = mergeupdate(channels_df, sta_df, [ST.NET, ST.STA, ST.WSID], [ST.ID])
    # rename now 'id' to 'station_id' before writing the channels to db:
    channels_df.rename(columns={ST.ID: CH.STAID}, inplace=True)

    # check channels with empty station id (should never happen, let's be
    # picky):
    null_sta_id = channels_df[CH.STAID].isnull()
    conflict_null_sta_id = pd.DataFrame()
    if null_sta_id.any():
        conflict_null_sta_id = channels_df[null_sta_id]
        channels_df = channels_df[~null_sta_id]

    # Add channels to db. First seet columns defining channel identity (db
    # unique constraint):
    cols = [Channel.station_id, Channel.location, Channel.channel]
    # Then add (sync actually, already existing channels are not inserted):
    channels_df = dbsyncdf(channels_df, session, cols, Channel.id, update,
                           buf_size=db_bufsize, keep_duplicates=False,
                           cols_to_print_on_err=CH.ERRCOLS)

    log_unsaved_channels(conflict_between, conflict_within,
                         conflict_null_sta_id)

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
    # Add to the list below the stations discarded because not uniquely mapped
    # to a single data center (dc_id). E.g.:
    #   net sta loc cha start_time dc_id
    #   N   S   L   C   2010-01-01 1
    #   N   S   L   C   2010-01-01 2
    # (Depending on the eidavalidator, only one or both rows will be added)
    conflict_between_dc = []
    # Add to the list below the stations discarded because of some non-unique
    # database constraint. E.g.:
    #   net sta loc cha start_time end_time   dc_id
    #   N   S   L   C   2010-01-01 None       1
    #   N   S   L   C   2010-01-01 2011-01-01 1
    # (both rows will be added)
    conflict_within_dc = []
    # list of ok channels (not falling in any category above):
    oks = []
    # station_datacenters_from_db = None  # dataframe lazy loaded (see below)

    # first drop duplicates (all columns the same):
    channels_df = channels_df.drop_duplicates()

    # group by (net, sta), which is a unique constraints, and analyse what we get:
    for (net, sta), df_ in channels_df.groupby([ST.NET, ST.STA], sort=False):
        # if we have only ONE dc_id, skip "if" below.. Otherwise:
        if len(pd.unique(df_[ST.WSID])) > 1:
            # We have more than one data center mapped to the tuple
            # (net, sta, stime): get all ids from the db:
            real_dc_ids = set(
                _[0] for _ in session.query(Station.webservice_id).filter(
                                 (Station.network == net) &
                                 (Station.station == sta)).all()
            )

            if len(real_dc_ids) != 1:
                # The real datacenter ids are more than one => we can not save
                # the station: empty (=> discard) the dataframe
                conflict_between_dc.append(df_)
                continue

            # The real datacenter id is only one
            # => discard all (net, sta, stime) with different datacenter id
            dcid = list(real_dc_ids)[0]
            conflicting = df_[ST.WSID] != dcid
            conflict_between_dc.append(df_[conflicting])
            if conflicting.all():  # noqa
                continue
            df_ = df_[~conflicting]

        # df_ now HAS SURELY ONE AND ONLY ONE (dc_id, net, sta).
        # Check now duplicated (net, sta, loc, cha):
        dupes = df_.duplicated(subset=[CH.LOC, CH.CHA], keep=False)
        if dupes.any():
            conflict_within_dc.append(df_[dupes])
            if dupes.all():
                continue

            # take min of stimes, and max of etimes:
            tmp = df_.groupby([CH.LOC, CH.CHA], as_index=False, sort=False).agg({
                ST.STIME: 'min',
                ST.ETIME: 'max'
            })
            # restore dtypes (might be lost during agg function):
            for c in df_.columns:
                tmp[c] = tmp[c].astype(df_[c].dtype)
            df_ = tmp

        oks.append(df_)

    oks = pd.DataFrame() if not oks else \
        pd.concat(oks, axis=0, sort=False, ignore_index=True, copy=True)
    conflict_between_dc = pd.DataFrame() if not conflict_between_dc else \
        pd.concat(conflict_between_dc, axis=0, sort=False)
    conflict_within_dc = pd.DataFrame() if not conflict_within_dc else \
        pd.concat(conflict_within_dc, axis=0, sort=False)

    return oks, conflict_between_dc, conflict_within_dc


def log_unsaved_channels(conflict_between, conflict_within,
                         conflict_null_sta_id):
    """log the results of channels and station saving.

    :param conflict_between: Dataframe of channels conflicts between
        datacenters (duplicated stations returned by more than one datacenter)
    :param conflict_within: Dataframe of channels conflicts within the same
        datacenter (violating channels unique constraints)
    :param conflict_null_sta_id: Dataframe of channels that did not have a
        matching station after station saving and before channel saving (this
        should never happen, but we check it for safety otherwise some db error
        occurs)
    """
    max_row_count = 50
    cols2show = [ST.NET, ST.STA, ST.STIME, ST.ETIME, ST.WSID]
    if not conflict_between.empty:
        # conflict_between happen at a station level (avoid unnecessary channel
        # details):
        _ = conflict_between.drop_duplicates(subset=[ST.NET, ST.STA, ST.STIME],
                                             keep='first')
        msg = formatmsg('%d station(s) and %d channel(s) not saved to db' %
                        (len(_), len(conflict_between)),
                        'wrong datacenter detected using either Routing '
                        'services or already saved stations')
        logwarn_dataframe(_, msg, cols2show, max_row_count)

    cols2show = [ST.NET, ST.STA, CH.LOC, CH.CHA, ST.STIME, ST.ETIME, ST.WSID]
    if not conflict_within.empty:
        # Do not count stations here, as some of those stations might have been
        # saved as part of other correct channels
        msg = formatmsg('%d channel(s) not saved to db' % len(conflict_within),
                        'conflicting data, e.g. unique constraint failed')
        logwarn_dataframe(conflict_within, msg, cols2show, max_row_count)

    if not conflict_null_sta_id.empty:
        # Do not count stations here, as some of those stations might have been saved as
        # part of other correct channels
        msg = formatmsg('%d channel(s) not saved to db' %
                        len(conflict_null_sta_id),
                        'station id not found, unknown cause')
        logwarn_dataframe(conflict_null_sta_id, msg, cols2show, max_row_count)
