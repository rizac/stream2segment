'''
Download module for stations (level=channel) download

:date: Dec 3, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import map, next, zip, range, object

import re
import logging
from itertools import cycle

import numpy as np
import pandas as pd
from sqlalchemy import or_, and_

from stream2segment.io.db.models import DataCenter, Station, Channel
from stream2segment.download.utils import read_async, response2normalizeddf, FailedDownload,\
    DbExcLogger, dbsyncdf, to_fdsn_arg, formatmsg
from stream2segment.utils import get_progressbar, strconvert
from stream2segment.io.db.pdsql import dbquery2df, shared_colnames, mergeupdate

from stream2segment.utils.url import Request  # this handles py2and3 compatibility


# logger: do not use logging.getLogger(__name__) but point to stream2segment.download.logger:
# this way we preserve the logging namespace hierarchy
# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial) when calling logging
# functions of stream2segment.download.utils:
from stream2segment.download import logger  # @IgnorePep8


def get_channels_df(session, datacenters_df, eidavalidator,  # <- can be none
                    net, sta, loc, cha, starttime, endtime,
                    min_sample_rate, update,
                    max_thread_workers, timeout, blocksize, db_bufsize,
                    show_progress=False):
    """Returns a dataframe representing a query to the eida services (or the internal db
    if `post_data` is None) with the given argument.  The
    dataframe will have as columns the `key` attribute of any of the following db columns:
    ```
    [Channel.id, Station.latitude, Station.longitude, Station.datacenter_id]
    ```
    :param datacenters_df: the first item resulting from `get_datacenters_df` (pandas DataFrame)
    :param post_data: the second item resulting from `get_datacenters_df` (string)
    :param channels: a list of string denoting the channels, or None for no filtering
    (all channels). Each string follows FDSN specifications (e.g. 'BHZ', 'H??'). This argument
    is not used if `post_data` is given (not None)
    :param min_sample_rate: minimum sampling rate, set to negative value for no-filtering
    (all channels)
    """
    postdata = get_post_data(net, sta, loc, cha, starttime, endtime)

    ret = []
    url_failed_dc_ids = []
    iterable = ((id_, Request(url,
                              data=('format=text\nlevel=channel\n'+post_data_str).encode('utf8')))
                for url, id_, post_data_str in zip(datacenters_df[DataCenter.station_url.key],
                                                   datacenters_df[DataCenter.id.key],
                                                   cycle([postdata])))

    with get_progressbar(show_progress, length=len(datacenters_df)) as pbar:
        for obj, result, exc, url in read_async(iterable, urlkey=lambda obj: obj[-1],
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
                    logger.warning(formatmsg("Discarding response data", verr, url))

    db_cha_df = pd.DataFrame()
    if url_failed_dc_ids:  # if some datacenter does not return station, warn with INFO
        dc_df_fromdb = \
            datacenters_df.loc[datacenters_df[DataCenter.id.key].isin(url_failed_dc_ids)]
        logger.info(formatmsg("Fetching stations from database for %d (of %d) data-center(s)",
                              "download errors occurred"), len(dc_df_fromdb), len(datacenters_df))
        logger.info(dc_df_fromdb[DataCenter.dataselect_url.key].to_string(index=False))
        db_cha_df = get_channels_df_from_db(session, dc_df_fromdb, net, sta, loc, cha,
                                            starttime, endtime, min_sample_rate)

    # build two dataframes which we will concatenate afterwards
    web_cha_df = pd.DataFrame()
    if ret:  # pd.concat complains for empty list
        try:
            web_cha_df = filter_channels_df(pd.concat(ret, axis=0, ignore_index=True, copy=False),
                                            net, sta, loc, cha, min_sample_rate)

            # this raises FailedDownload if we cannot save any element:
            web_cha_df = save_stations_and_channels(session, web_cha_df, eidavalidator, update,
                                                    db_bufsize)
        except FailedDownload as qexc:
            if db_cha_df.empty:
                raise
            else:
                logger.warning(qexc)

    if db_cha_df.empty and web_cha_df.empty:
        # ok, now let's see if we have remaining datacenters to be fetched from the db
        raise FailedDownload(formatmsg("No station found",
                                       ("Unable to fetch stations from all data-centers, "
                                        "no data to fetch from the database. "
                                        "Check config and log for details")))
    ret = None
    if db_cha_df.empty:
        ret = web_cha_df
    elif web_cha_df.empty:
        ret = db_cha_df
    else:
        ret = pd.concat((web_cha_df, db_cha_df), axis=0, ignore_index=True)
    # the columns for the channels dataframe that will be returned
    return ret[[c.key for c in (Channel.id, Channel.station_id, Station.latitude,
                                Station.longitude, Station.datacenter_id, Station.start_time,
                                Station.end_time, Station.network, Station.station,
                                Channel.location, Channel.channel)]].copy()


def get_post_data(net, sta, loc, cha, starttime=None, endtime=None):
    '''Returns the string for a FDSN POST request according to the given
        net(works), sta(tions), loc(ations) and cha(nnels), all iterable of strings
        returned by :func:`stream2segment.download.utils.nslc_lists`

    Example:
        asget([], ['ABC'], [''], ['!A*', 'HH?', 'HN?'], None, None) = '* ABC -- HH?,HN? * *'

    Note negations (!A*) not included: strings starting with "!" mean 'NOT' in this
    program's syntax: as this feature is not supported in an FDSN query it
    cannot be forwarded to any web service. The feature is used here in other module functions
    *after* downloading data

    Arguments are usually the output of :func:`stream2segment.download.utils.nslc_lists`:

    :param net: an iterable of strings denoting networks.
    :param sta: an iterable of strings denoting stations.
    :param loc: an iterable of strings denoting locations.
    :param cha: an iterable of strings denoting channels.
    '''
    args = []
    for i, lst in enumerate([net, sta, loc, cha]):
        parsearg = '*'
        if lst:
            parsearg = to_fdsn_arg(lst)
            if i == 3 and not parsearg:  # location case, empty has to be input as '--'
                parsearg = '--'
        args.append(parsearg)

    args.append("*" if not starttime else starttime.isoformat())
    args.append("*" if not endtime else endtime.isoformat())

    return "{} {} {} {} {} {}".format(*args)


def filter_channels_df(channels_df, net, sta, loc, cha, min_sample_rate):
    '''Filters out `channels_df` according to the given parameters. Raises
    `FailedDownload` if the returned filtered data frame woul be empty

    Note that `net, sta, loc, cha` filters will be considered only if negations (i.e.,
    with leading exclamation mark: "!A*") because the 'positive' filters are FDSN stantard and
    are supposed to be already used in producing channels_df

    Example:
        filter_channels_df(d, [], ['ABC'], [''], ['!A*', 'HH?', 'HN?'])

        basically takes the dataframe `d`, finds the column related to the `channels` key and
        removes all rowv whose channel starts with 'A', returning the new filtered data frame

    Arguments are usually the output of :func:`stream2segment.download.utils.nslc_lists`

    :param net: an iterable of strings denoting networks.
    :param sta: an iterable of strings denoting stations.
    :param loc: an iterable of strings denoting locations.
    :param cha: an iterable of strings denoting channels.
    :param min_sample_rate: numeric, minimum sample rate. If negative or zero, this parameter
        is ignored
    '''
    # create a dict of regexps for pandas dataframe. FDSNWS do not support NOT
    # operators . Thus concatenate expression with OR
    dffilter = None
    sa_cols = (Station.network, Station.station, Channel.location, Channel.channel)

    for lst, sa_col in zip((net, sta, loc, cha), sa_cols):
        if not lst:
            continue
        lst = [_ for _ in lst if _[0:1] == '!']  # take only negation expression
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
        channels_df[~dffilter].copy()  # pylint: disable=invalid-unary-operand-type

    if ret.empty:
        raise FailedDownload("No channel matches user defined filters "
                             "(network, channel, sample rate, ...)")

    discarded_sr = len(channels_df) - len(ret)
    if discarded_sr:
        logger.warning(("%d channel(s) discarded according to current config. filters "
                        "(network, channel, sample rate, ...)"), discarded_sr)

    return ret


def get_channels_df_from_db(session, datacenters_df, net, sta, loc, cha, starttime, endtime,
                            min_sample_rate):
    # Build sql-alchemy binary expressions
    # _be means "binary expression" (sql alchemy object reflecting a sql clause)
    srate_be = Channel.sample_rate >= min_sample_rate if min_sample_rate > 0 else True
    # select only relevant datacenters. Convert tolist() cause python3 complains of numpy ints
    # (python2 doesn't but tolist() is safe for both):
    dc_be = Station.datacenter_id.in_(datacenters_df[DataCenter.id.key].tolist())
    # Starttime and endtime below: it must NOT hold:
    # station.endtime <= starttime OR station.starttime >= endtime
    # i.e. it MUST hold the negation:
    # station.endtime > starttime AND station.starttime< endtime
    stime_be = True
    if starttime:
        stime_be = ((Station.end_time == None) | (Station.end_time > starttime))
    # endtime: Limit to metadata epochs ending on or before the specified end time.
    # Note that station's ent_time can be None
    etime_be = (Station.start_time < endtime) if endtime else True  # @IgnorePep8
    sa_cols = [Channel.id, Channel.station_id, Station.latitude, Station.longitude,
               Station.start_time, Station.end_time, Station.datacenter_id, Station.network,
               Station.station, Channel.location, Channel.channel]
    # filter on net, sta, loc, cha, as specified in config and converted to sql-alchemy be:
    nslc_be = get_sqla_binexp(net, sta, loc, cha)
    # note below: binary expressions (all variables ending with "_be") might be the boolean True.
    # SqlAlchemy seems to understand them as long as they are preceded by a "normal" binary
    # expression. Thus q.filter(binary_expr & True) works and it's equal to q.filter(binary_expr),
    # BUT .filter(True & True) is not working as a no-op filter, it simply does not work
    # Here we should be safe cause dc_be is a non-True sql alchemy expression (see above)
    qry = session.query(*sa_cols).join(Channel.station).filter(and_(dc_be, srate_be, nslc_be,
                                                                    stime_be, etime_be))
    return dbquery2df(qry)


def get_sqla_binexp(net, sta, loc, cha):
    '''Returns the sql-alchemy binary expression to be used as argument
    for db queries (e.g., `session.query(...)`) which translates to SQL the given
        net(works), sta(tions), loc(ations) and cha(nnels), all iterable of strings.

    Example:
        asbinexp([], ['ABC'], [''], ['!A*', 'HH?', 'HN?']) = 'sta=ABC&loc=&cha=HH?,HN?'

    Note negations (!A*) mean 'NOT' in this
    program's syntax (this feature is not standard in an FDSN query)

    Arguments are usually the output of :func:`stream2segment.download.utils.nslc_lists`

    :param net: an iterable of strings denoting networks.
    :param sta: an iterable of strings denoting stations.
    :param loc: an iterable of strings denoting locations.
    :param cha: an iterable of strings denoting channels.
    '''
    # build a sql alchemy filter condition
    sa_cols = (Station.network, Station.station, Channel.location, Channel.channel)

    sa_bin_exprs = []

    wild2sql = strconvert.wild2sql  # conversion function

    for column, lst in zip(sa_cols, (net, sta, loc, cha)):
        matches = []
        for string in lst:
            negate = False
            if string[0:1] == '!':
                negate = True
                string = string[1:]

            condition = column.like(wild2sql(string)) if ('?' in string or '*' in string) \
                else (column == string)

            if negate:
                condition = ~condition

            matches.append(condition)

        if matches:
            sa_bin_exprs.append(or_(*matches))

    return True if not sa_bin_exprs else and_(*sa_bin_exprs)


def save_stations_and_channels(session, channels_df, eidavalidator, update, db_bufsize):
    """
        Saves to db channels (and their stations) and returns a dataframe with only channels saved
        The returned data frame will have the column 'id' (`Station.id`) renamed to
        'station_id' (`Channel.station_id`) and a new 'id' column referring to the Channel id
        (`Channel.id`)
        :param channels_df: pandas DataFrame resulting from `get_channels_df`
    """
    # define columns (sql-alchemy model attrs) and their string names (pandas col names) once:
    STA_NET = Station.network.key  # pylint: disable=invalid-name
    STA_STA = Station.station.key  # pylint: disable=invalid-name
    STA_STIME = Station.start_time.key  # pylint: disable=invalid-name
    STA_DCID = Station.datacenter_id.key  # pylint: disable=invalid-name
    STA_ID = Station.id.key  # pylint: disable=invalid-name
    CHA_STAID = Channel.station_id.key  # pylint: disable=invalid-name
    CHA_LOC = Channel.location.key  # pylint: disable=invalid-name
    CHA_CHA = Channel.channel.key  # pylint: disable=invalid-name
    # set columns to show in the log on error (no row written):
    STA_ERRCOLS = [STA_NET, STA_STA, STA_STIME, STA_DCID]
    CHA_ERRCOLS = [STA_NET, STA_STA, CHA_LOC, CHA_CHA, STA_STIME, STA_DCID]

    # first drop channels of same station:
    sta_df = channels_df.drop_duplicates(subset=[STA_NET, STA_STA, STA_STIME, STA_DCID]).copy()
    sta_df = drop_station_dupliactes(session, sta_df, eidavalidator)

    # remember: dbsyncdf raises a FailedDownload, so no need to check for empty(dataframe). Also,
    # if update is True, for stations only it must NOT update inventories HERE (handled later)
    _update_stations = update
    if _update_stations:
        _update_stations = [_ for _ in shared_colnames(Station, sta_df, pkey=False)
                            if _ != Station.inventory_xml.key]
    sta_df = dbsyncdf(sta_df, session, [Station.network, Station.station, Station.start_time],
                      Station.id, _update_stations, buf_size=db_bufsize, keep_duplicates=True,
                      cols_to_print_on_err=STA_ERRCOLS)
    # sta_df will have the STA_ID columns, channels_df not: set it from the former to the latter:
    channels_df = mergeupdate(channels_df, sta_df, [STA_NET, STA_STA, STA_STIME, STA_DCID],
                              [STA_ID])
    # rename now 'id' to 'station_id' before writing the channels to db:
    channels_df.rename(columns={STA_ID: CHA_STAID}, inplace=True)
    # check dupes and warn:
    channels_df_dupes = channels_df[channels_df[CHA_STAID].isnull()]
    if not channels_df_dupes.empty:
        exc_msg = ("Found %d duplicated channel(s) to be discarded "
                   "as a result of duplicated stations") % len(channels_df_dupes)
        logger.info(exc_msg)
        # If you want to print the duplicated channels, see `drop_station_dupliactes`
        # (don't do it as it's redundant info), and type e.g.:
        # DbExcLogger(columns_to_print).failed_insert(channels_df_dupes, Exception(exc_msg))
        channels_df.dropna(axis=0, subset=[CHA_STAID], inplace=True)

    # add channels to db:
    channels_df = dbsyncdf(channels_df, session,
                           [Channel.station_id, Channel.location, Channel.channel],
                           Channel.id, update, buf_size=db_bufsize, keep_duplicates=True,
                           cols_to_print_on_err=CHA_ERRCOLS)
    return channels_df


def drop_station_dupliactes(session, sta_df, eidavalidator):
    '''Drops station duplicates from the Station Data frame `sta_df`
    using eidavalidator or the database accessible via
    the session object, if eidavalidator is None.
    If no duplicates are found, returns `sta_df`
    '''

    STA_NET = Station.network.key  # pylint: disable=invalid-name
    STA_STA = Station.station.key  # pylint: disable=invalid-name
    STA_STIME = Station.start_time.key  # pylint: disable=invalid-name
    STA_ETIME = Station.end_time.key  # pylint: disable=invalid-name
    STA_DCID = Station.datacenter_id.key  # pylint: disable=invalid-name
    STA_DCID2 = 'invalid_' + STA_DCID  # pylint: disable=invalid-name
    CHA_LOC = Channel.location.key  # pylint: disable=invalid-name
    CHA_CHA = Channel.channel.key  # pylint: disable=invalid-name

    # then check dupes. Same network, station, starttime but different datacenter:
    duplicated = sta_df.duplicated(subset=[STA_NET, STA_STA, STA_STIME],
                                   keep=False)  # keep=False => Mark all duplicates as True
    if duplicated.any():
        sta_df_dupes = sta_df[duplicated].copy()
        sta_df_dupes.rename(columns={STA_DCID: STA_DCID2}, inplace=True)
        sta_df_dupes[STA_DCID] = np.nan

        if eidavalidator is not None:
            for i, net, sta, loc, cha, stime, etime in \
                zip(sta_df_dupes.index, sta_df_dupes[STA_NET], sta_df_dupes[STA_STA],
                    sta_df_dupes[CHA_LOC], sta_df_dupes[CHA_CHA],
                    sta_df_dupes[STA_STIME], sta_df_dupes[STA_ETIME]):
                sta_df_dupes.at[i, STA_DCID] = \
                    eidavalidator.get_dc_id(net, sta, loc, cha,
                                            None if pd.isnull(stime) else stime,
                                            None if pd.isnull(etime) else etime)
        else:
            sta_db = dbquery2df(session.query(Station.network, Station.station, Station.start_time,
                                              Station.datacenter_id))
            mergeupdate(sta_df_dupes, sta_db, [STA_NET, STA_STA, STA_STIME], [STA_DCID])

        sta_df_dupes = sta_df_dupes[sta_df_dupes[STA_DCID] != sta_df_dupes[STA_DCID2]]

        if not sta_df_dupes.empty:
            exc_msg = "Found %d duplicated station(s) to be discarded (checked against %s)" % \
                (len(sta_df_dupes),
                 ("already saved stations" if eidavalidator is None else "eida routing service"))
            logger.info(exc_msg)
            # print the removed dataframe to log.warning (showing
            # [STA_NET, STA_STA, STA_STIME, STA_DCID2] columns only):
            db_exc_logger = DbExcLogger([STA_NET, STA_STA, STA_STIME, STA_DCID2])
            db_exc_logger.failed_insert(sta_df_dupes.sort_values(by=[STA_NET, STA_STA, STA_STIME]),
                                        '', )
            # https://stackoverflow.com/questions/28901683/pandas-get-rows-which-are-not-in-other-dataframe:
            sta_df = sta_df.loc[~sta_df.index.isin(sta_df_dupes.index)]

    return sta_df


def chaid2mseedid_dict(channels_df, drop_mseedid_columns=True):
    '''returns a dict of the form {channel_id: mseed_id} from channels_df, where mseed_id is
    a string of the form

    ```[network].[station].[location].[channel]```

    :param channels_df: the result of `get_channels_df`

    :param drop_mseedid_columns: boolean (default: True), removes all columns related to the mseed
        id from `channels_df`. This might save up a lor of memory when cimputing the
        segments resulting from each event -> stations binding (according to the search radius)
        Remember that pandas strings are not optimized for memory as they are python objects
        (https://www.dataquest.io/blog/pandas-big-data/)
    '''
    # For convenience and readability, define once the mapped column names representing the
    # dataframe columns that we need:
    CHA_ID = Channel.id.key  # pylint: disable=invalid-name
    STA_NET = Station.network.key  # pylint: disable=invalid-name
    STA_STA = Station.station.key  # pylint: disable=invalid-name
    CHA_LOC = Channel.location.key  # pylint: disable=invalid-name
    CHA_CHA = Channel.channel.key  # pylint: disable=invalid-name

    net = channels_df[STA_NET].str.cat
    sta = channels_df[STA_STA].str.cat
    loc = channels_df[CHA_LOC].str.cat
    cha = channels_df[CHA_CHA]
    _mseedids = net(sta(loc(cha, sep='.', na_rep=''), sep='.', na_rep=''), sep='.', na_rep='')

    if drop_mseedid_columns:
        # remove string columns, we do not need it anymore and
        # will save a lot of memory for subsequent operations
        channels_df.drop([STA_NET, STA_STA, CHA_LOC, CHA_CHA], axis=1, inplace=True)
    # we could return
    # pd.DataFrame(index=channels_df[CHA_ID], {'mseed_id': _mseedids})
    # but the latter does NOT consume less memory (strings are python string in pandas)
    # and the search for an mseed_id given a loc[channel_id] is slower than python dicts.
    # As the returned element is intended for searching, then return a dict:
    return {chaid: mseedid for chaid, mseedid in zip(channels_df[CHA_ID], _mseedids)}
