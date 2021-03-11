"""
Stations/Channels download functions

:date: Dec 3, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
# make the following(s) behave like python3 counterparts if running from
# python2.7.x (http://python-future.org/imports.html#explicit-imports):
from builtins import zip, object

import re
from itertools import cycle

import numpy as np
import pandas as pd
from sqlalchemy import or_, and_

from stream2segment.io.db.models import DataCenter, Station, Channel
from stream2segment.download.utils import read_async, response2normalizeddf, FailedDownload,\
    DbExcLogger, dbsyncdf, to_fdsn_arg, formatmsg, logwarn_dataframe
from stream2segment.utils import get_progressbar, strconvert
from stream2segment.io.db.pdsql import dbquery2df, shared_colnames, mergeupdate

from stream2segment.utils.url import Request  # this handles py2and3 compatibility


# logger: do not use logging.getLogger(__name__) but point to stream2segment.download.logger:
# this way we preserve the logging namespace hierarchy
# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial) when calling logging
# functions of stream2segment.download.utils:
from stream2segment.download import logger  # @IgnorePep8


def get_channels_df(session, datacenters_df, eidavalidator, net, sta, loc, cha,
                    starttime, endtime, min_sample_rate, update,
                    max_thread_workers, timeout, blocksize, db_bufsize,
                    show_progress=False):
    """Return a Dataframe representing a query to the station service of each
    URL in :func:`stream2segment.download.modules.datacenters_df` with the
    given arguments.

    :param datacenters_df: (DataFrame) the first item resulting from
        `get_datacenters_df`
    :param eidavalidator: None, or an instance of
        :class:`stream2segment.download.modules.datacenter.EidaValidator`
    :param min_sample_rate: minimum sampling rate, set to negative value
        for no-filtering (all channels)
    """
    postdata = get_post_data(net, sta, loc, cha, starttime, endtime)
    dc_url_key, dc_id_key = DataCenter.station_url.key, DataCenter.id.key
    iterable = ((id_, Request(url, data=('format=text\nlevel=channel\n'+
                                         post_data_str).encode('utf8')))
                for url, id_, post_data_str in zip(datacenters_df[dc_url_key],
                                                   datacenters_df[dc_id_key],
                                                   cycle([postdata])))

    ret = []
    url_failed_dc_ids = []
    with get_progressbar(show_progress, length=len(datacenters_df)) as pbar:
        for obj, result, exc, url in read_async(iterable,
                                                urlkey=lambda obj: obj[-1],
                                                blocksize=blocksize,
                                                max_workers=max_thread_workers,
                                                decode='utf8', timeout=timeout):
            pbar.update(1)
            dcen_id = obj[0]
            if exc:
                url_failed_dc_ids.append(dcen_id)
                logger.warning(formatmsg("Unable to fetch stations", exc, url))
            else:
                try:
                    dframe = response2normalizeddf(url, result[0], "channel")
                    if not dframe.empty:
                        dframe[Station.datacenter_id.key] = dcen_id
                        ret.append(dframe)
                except ValueError as verr:
                    logger.warning(formatmsg("Discarding response data", verr,
                                             url))

    db_cha_df = pd.DataFrame()
    if url_failed_dc_ids:
        # if some datacenter does not return station, warn with INFO
        dc_df_fromdb = datacenters_df.loc[datacenters_df[DataCenter.id.key].\
            isin(url_failed_dc_ids)]
        logger.info(formatmsg("Fetching stations from database for %d (of %d) "
                              "data-center(s)", "download errors occurred"),
                    len(dc_df_fromdb), len(datacenters_df))
        logger.info(dc_df_fromdb[DataCenter.dataselect_url.key].
                    to_string(index=False))
        db_cha_df = get_channels_df_from_db(session, dc_df_fromdb, net, sta,
                                            loc, cha, starttime, endtime,
                                            min_sample_rate)

    # build two dataframes which we will concatenate afterwards
    web_cha_df = pd.DataFrame()
    if ret:  # pd.concat complains for empty list
        try:
            web_cha_df = filter_channels_df(pd.concat(ret, axis=0,
                                                      ignore_index=True,
                                                      copy=False),
                                            net, sta, loc, cha,
                                            min_sample_rate)

            # this raises FailedDownload if we cannot save any element:
            web_cha_df = save_stations_and_channels(session, web_cha_df,
                                                    eidavalidator, update,
                                                    db_bufsize)
        except FailedDownload as qexc:
            if db_cha_df.empty:
                raise
            else:
                logger.warning(qexc)

    if db_cha_df.empty and web_cha_df.empty:
        # ok, now let's see if we have remaining datacenters to be
        # fetched from the db
        raise FailedDownload(formatmsg("No station found",
                                       "Unable to fetch stations from all "
                                       "data-centers, no data to fetch from "
                                       "the database. Check config and log "
                                       "for details"))
    ret = None
    if db_cha_df.empty:
        ret = web_cha_df
    elif web_cha_df.empty:
        ret = db_cha_df
    else:
        ret = pd.concat((web_cha_df, db_cha_df), axis=0, ignore_index=True,
                        sort=False)
    # the columns for the channels dataframe that will be returned
    return ret[[c.key for c in (Channel.id, Channel.station_id,
                                Station.latitude, Station.longitude,
                                Station.datacenter_id, Station.start_time,
                                Station.end_time, Station.network,
                                Station.station, Channel.location,
                                Channel.channel)]].copy()


def get_post_data(net, sta, loc, cha, starttime=None, endtime=None):
    """Return the string for a FDSN POST request according to the given
        net(works), sta(tions), loc(ations) and cha(nnels), all iterable of
        strings returned by :func:`stream2segment.download.utils.nslc_lists`

    Example:
    ```
    >>> get_post_data([], ['ABC'], [''], ['!A*', 'HH?', 'HN?'], None, None)
    '* ABC -- HH?,HN? * *'
    ```
    Note negations (!A*) not included: strings starting with "!" mean 'NOT' in
    this program's syntax: as this feature is not supported in an FDSN query it
    cannot be forwarded to any web service. The feature is used here in other
    module functions *after* downloading data

    Arguments are usually the output of
    :func:`stream2segment.download.utils.nslc_lists`:

    :param net: an iterable of strings denoting networks.
    :param sta: an iterable of strings denoting stations.
    :param loc: an iterable of strings denoting locations.
    :param cha: an iterable of strings denoting channels.
    """
    args = []
    for i, lst in enumerate([net, sta, loc, cha]):
        parsearg = '*'
        if lst:
            parsearg = to_fdsn_arg(lst)
            if i == 3 and not parsearg:  # loc: empty has to be input as '--'
                parsearg = '--'
        args.append(parsearg)

    args.append("*" if not starttime else starttime.isoformat())
    args.append("*" if not endtime else endtime.isoformat())

    return "{} {} {} {} {} {}".format(*args)


def filter_channels_df(channels_df, net, sta, loc, cha, min_sample_rate):
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


def get_channels_df_from_db(session, datacenters_df, net, sta, loc, cha,
                            starttime, endtime, min_sample_rate):
    """Return a Dataframe of the database channels according to the
    arguments"""
    # Build SQL-Alchemy binary expressions (suffix '_be'), i.e. an object
    # reflecting a SQL clause)
    srate_be = True
    if min_sample_rate > 0:
        srate_be = Channel.sample_rate >= min_sample_rate
    # Select only relevant datacenters. Convert numnpy array `tolist()` because
    # database clauses work best with native Python objects:
    dc_be = Station.datacenter_id.in_(datacenters_df[DataCenter.id.key].
                                      tolist())
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
               Station.datacenter_id, Station.network, Station.station,
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


class ST(object):  # pylint: disable=too-few-public-methods, useless-object-inheritance
    """Simple enum-like container of strings defining the station's related
    database/dataframe columns needed in this module
    """
    ID = Station.id.key  # pylint: disable=invalid-name
    NET = Station.network.key  # pylint: disable=invalid-name
    STA = Station.station.key  # pylint: disable=invalid-name
    STIME = Station.start_time.key  # pylint: disable=invalid-name
    ETIME = Station.end_time.key  # pylint: disable=invalid-name
    DCID = Station.datacenter_id.key  # pylint: disable=invalid-name
    # set columns to show in the log on error ("no row written"):
    ERRCOLS = [NET, STA, STIME, DCID]  # pylint: disable=invalid-name


class CH(object):  # pylint: disable=too-few-public-methods, useless-object-inheritance
    """Simple enum-like container of strings defining the channel's related
    database/dataframe columns needed in this module
    """
    ID = Channel.id.key  # pylint: disable=invalid-name
    STAID = Channel.station_id.key  # pylint: disable=invalid-name
    LOC = Channel.location.key  # pylint: disable=invalid-name
    CHA = Channel.channel.key  # pylint: disable=invalid-name
    # set columns to show in the log on error ("no row written"):
    ERRCOLS = [ST.NET, ST.STA, LOC, CHA, ST.STIME, ST.DCID]  # noqa


def save_stations_and_channels(session, channels_df, eidavalidator, update,
                               db_bufsize):
    """Saves to db channels (and their stations) and returns a dataframe with
    only channels saved. The returned Dataframe will have the column 'id'
    (`Station.id`) renamed to 'station_id' (`Channel.station_id`) and a new
    'id' column referring to the Channel id (`Channel.id`)

    :param channels_df: pandas DataFrame
    """
    channels_df, conflict_between, conflict_within = \
        drop_duplicates(session, channels_df, eidavalidator)

    if channels_df.empty:
        raise FailedDownload('No channel left after cleanup '
                             '(e.g., drop duplicates)')

    # if update is True, don't update inventories HERE (handled later)
    _update_stations = update
    if _update_stations:
        _update_stations = [_ for _ in shared_colnames(Station, channels_df,
                                                       pkey=False)
                            if _ != Station.inventory_xml.key]

    # Add stations to db (Note: no need to check for `empty(channels_df)`,
    # `dbsyncdf` raises a `FailedDownload` in case). First set columns
    # defining channel identity (db unique constraint):
    cols = [Station.network, Station.station, Station.start_time]
    # Then add (sync actually, already existing stations are not inserted):
    sta_df = dbsyncdf(channels_df.drop_duplicates(subset=[ST.NET, ST.STA,
                                                          ST.STIME, ST.DCID]),
                      session, cols, Station.id, _update_stations,
                      buf_size=db_bufsize, keep_duplicates=False,
                      cols_to_print_on_err=ST.ERRCOLS)
    # `sta_df` will have the STA_ID columns, `channels_df` not: set it from the
    # former to the latter:
    channels_df = mergeupdate(channels_df, sta_df,
                              [ST.NET, ST.STA, ST.STIME, ST.DCID],
                              [ST.ID])
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


def drop_duplicates(session, channels_df, eidavalidator):
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

    # From now on do not use anymore duplicated or drop_duplicates: the removal
    # of duplicates now is more tricky.
    # Let's  group by (net, sta, starttime), which is a unique constraints of
    # the station table, and analyse what we get:
    for (net, sta, stime), df_ in channels_df.\
            groupby([ST.NET, ST.STA, ST.STIME], sort=False):
        # if we have only ONE dc_id, skip "if" below.. Otherwise:
        if len(pd.unique(df_[ST.DCID])) > 1:
            # We have more than one data center mapped to the tuple
            # (net, sta, stime): get all ids from the eidavalidator (=object
            # representing the eida routing service) and put them in the set
            # below:
            real_dc_ids = set()
            # group stations by tuples (net, sta, stime, etime), because etime
            # is needed by the eidavalidator.get_dc_id. (pd.to_datetime fixes
            # https://github.com/pandas-dev/pandas/issues/35449)
            for etime in pd.to_datetime(pd.unique(df_[ST.ETIME])):
                # eidavalidator and session.query below expect Nones, not pd.NaT:
                stime = None if pd.isnull(stime) else stime  # should never be the case
                etime = None if pd.isnull(etime) else etime

                dcids = []
                if eidavalidator is not None:
                    # get the datacenter id(s) at a station level
                    # (loc, cha = None):
                    dcids = eidavalidator.get_dc_ids(net, sta, loc=None, cha=None,
                                                     stime=stime, etime=etime)

                if not dcids:
                    # eidavalidator null, or no dc_id found: query dc_id(s) from db:
                    dcids = [_[0] for _ in
                             session.query(Station.datacenter_id).filter(
                                 (Station.network == net) &
                                 (Station.station == sta) &
                                 (Station.start_time == stime) &
                                 (Station.end_time == etime)).all()
                             ]

                real_dc_ids.update(dcids)

            # Reminder: we are here if we have more than one datacenter mapped
            # to the same tuple (net, sta, stime). Now, real_dc_ids (the
            # real/reliable datacenter ids) might be one or more than one:
            if len(real_dc_ids) != 1:
                # The real datacenter ids are more than one => we can not save
                # the station: empty (=> discard) the dataframe
                conflict_between_dc.append(df_)
                df_ = df_[0:0]  # simply empty dataframe, with same columns
            else:
                # The real datacenter id is only one
                # => discard all (net, sta, stime) with different datacenter id
                dcid = list(real_dc_ids)[0]
                conflicting = df_[ST.DCID] != dcid
                conflict_between_dc.append(df_[conflicting])
                df_ = df_[~conflicting]

        # df_ now HAS SURELY ONE AND ONLY ONE dc_id, and same (net, sta, stime)
        if not df_.empty:
            # Last check: df_ will be written as ONE station (one row of the
            # "stations" table) and then, with the station_id, each df_ row
            # will be written as a different channel. The "channels" table has
            # as unique constraint the tuple (station_id, location, channel),
            # thus we need to drop NOW duplicated values of (location,
            # channel). PS: in case you wonder, yes, this has already happened
            # (e.g. two channels with all fields equal except `endtime`)
            dupes = df_.duplicated(subset=[CH.LOC, CH.CHA], keep=False)
            if dupes.any():
                conflict_within_dc.append(df_[dupes])
                df_ = df_[~dupes]

        if not df_.empty:
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
    cols2show = [ST.NET, ST.STA, ST.STIME, ST.ETIME, ST.DCID]
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

    cols2show = [ST.NET, ST.STA, CH.LOC, CH.CHA, ST.STIME, ST.ETIME, ST.DCID]
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


def chaid2mseedid_dict(channels_df, drop_mseedid_columns=True):
    """Return a dict of the form {channel_id: mseed_id} from `channels_df`,
    where mseed_id is a string of the form
    "[network].[station].[location].[channel]"

    :param channels_df: pandas Dataframe (one channel per row)
    :param drop_mseedid_columns: boolean (default: True), remove all columns
        related to the mseed id from `channels_df`. This might save up a lor
        of memory: pandas strings are stored as Python objects
        (https://www.dataquest.io/blog/pandas-big-data/)
    """
    net = channels_df[ST.NET].str.cat
    sta = channels_df[ST.STA].str.cat
    loc = channels_df[CH.LOC].str.cat
    cha = channels_df[CH.CHA]
    _mseedids = net(sta(loc(cha, sep='.', na_rep=''), sep='.', na_rep=''),
                    sep='.', na_rep='')

    if drop_mseedid_columns:
        # remove string columns, we do not need it anymore and
        # will save a lot of memory for subsequent operations
        channels_df.drop([ST.NET, ST.STA, CH.LOC, CH.CHA], axis=1,
                         inplace=True)
    # we could return
    # pd.DataFrame(index=channels_df[CHA_ID], {'mseed_id': _mseedids})
    # but the latter does NOT consume less memory (strings are python string in
    # pandas) and the search for an mseed_id given a loc[channel_id] is slower
    # than python dicts. As the returned element is intended for searching,
    # then return a dict:
    return {chaid: mseedid
            for chaid, mseedid in zip(channels_df[CH.ID], _mseedids)}
