'''
Download module for segments download

:date: Dec 3, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import map, next, zip, range, object

from datetime import timedelta
import sys
from collections import OrderedDict
# import logging

import numpy as np
import pandas as pd

from stream2segment.io.db.models import DataCenter, Station, Channel, Segment
from stream2segment.download.utils import read_async, QuitDownload,\
    handledbexc, custom_download_codes, logwarn_dataframe, DownloadStats
from stream2segment.download.modules.mseedlite import MSeedError, unpack as mseedunpack
from stream2segment.utils.msgs import MSG
from stream2segment.utils import get_progressbar
from stream2segment.io.db.pdsql import dbquery2df, mergeupdate, DbManager

# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#aliased-imports):
from future import standard_library
standard_library.install_aliases()
from urllib.parse import urlparse  # @IgnorePep8
from urllib.request import Request  # @IgnorePep8


# logger: do not use logging.getLogger(__name__) but point to stream2segment.download.logger:
# this way we preserve the logging namespace hierarchy
# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial) when calling logging
# functions of stream2segment.download.utils:
from stream2segment.download import logger  # @IgnorePep8


def chaid2mseedid_dict(channels_df, drop_mseedid_columns=True):
    '''returns a dict of the form {channel_id: mseed_id} from channels_df, where mseed_id is
    a string of the form ```[network].[station].[location].[channel]```
    :param channels_df: the result of `get_channels_df`
    :param drop_mseedid_columns: boolean (default: True), removes all columns related to the mseed
    id from `channels_df`. This might save up a lor of memory when cimputing the
    segments resulting from each event -> stations binding (according to the search radius)
    Remember that pandas strings are not optimized for memory as they are python objects
    (https://www.dataquest.io/blog/pandas-big-data/)
    '''
    # For convenience and readability, define once the mapped column names representing the
    # dataframe columns that we need:
    CHA_ID = Channel.id.key
    STA_NET = Station.network.key
    STA_STA = Station.station.key
    CHA_LOC = Channel.location.key
    CHA_CHA = Channel.channel.key

    n = channels_df[STA_NET].str.cat
    s = channels_df[STA_STA].str.cat
    l = channels_df[CHA_LOC].str.cat
    c = channels_df[CHA_CHA]
    _mseedids = n(s(l(c, sep='.', na_rep=''), sep='.', na_rep=''), sep='.', na_rep='')
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


def prepare_for_download(session, segments_df, timespan, retry_seg_not_found, retry_url_err,
                         retry_mseed_err, retry_client_err, retry_server_err, retry_timespan_err,
                         retry_timespan_warn=False):
    """
        Drops the segments which are already present on the database and updates the primary
        keys for those not present (adding them to the db).
        Adds three new columns to the returned Data frame:
        `Segment.id` and `Segment.download_status_code`

        :param session: the sql-alchemy session bound to an existing database
        :param segments_df: pandas DataFrame resulting from `get_arrivaltimes`
    """
    # For convenience and readability, define once the mapped column names representing the
    # dataframe columns that we need:
    SEG_EVID = Segment.event_id.key
    SEG_ATIME = Segment.arrival_time.key
    SEG_START = Segment.request_start.key
    SEG_END = Segment.request_end.key
    SEG_CHID = Segment.channel_id.key
    SEG_ID = Segment.id.key
    SEG_DSC = Segment.download_code.key
    SEG_RETRY = "__do.download__"

    URLERR_CODE, MSEEDERR_CODE, OUTTIME_ERR, OUTTIME_WARN = custom_download_codes()
    # we might use dbsync('sync', ...) which sets pkeys and updates non-existing, but then we
    # would issue a second db query to check which segments should be re-downloaded (retry).
    # As the segments table might be big (hundred of thousands of records) we want to optimize
    # db queries, thus we first "manually" set the existing pkeys with a SINGLE db query which
    # gets ALSO the status codes (whereby we know what to re-download), and AFTER we call we
    # call dbsync('syncpkeys',..) which sets the null pkeys.
    # This function is basically what dbsync('sync', ...) does with the addition that we set whcch
    # segments have to be re-downloaded, if any

    # query relevant data into data frame:
    db_seg_df = dbquery2df(session.query(Segment.id, Segment.channel_id, Segment.request_start,
                                         Segment.request_end, Segment.download_code,
                                         Segment.event_id))

    # set the boolean array telling whether we need to retry db_seg_df elements (those already
    # downloaded)
    mask = False
    if retry_seg_not_found:
        mask |= pd.isnull(db_seg_df[SEG_DSC])
    if retry_url_err:
        mask |= db_seg_df[SEG_DSC] == URLERR_CODE
    if retry_mseed_err:
        mask |= db_seg_df[SEG_DSC] == MSEEDERR_CODE
    if retry_client_err:
        mask |= db_seg_df[SEG_DSC].between(400, 499.9999, inclusive=True)
    if retry_server_err:
        mask |= db_seg_df[SEG_DSC].between(500, 599.9999, inclusive=True)
    if retry_timespan_err:
        mask |= db_seg_df[SEG_DSC] == OUTTIME_ERR
    if retry_timespan_warn:
        mask |= db_seg_df[SEG_DSC] == OUTTIME_WARN

    db_seg_df[SEG_RETRY] = mask

    # update existing dataframe. If db_seg_df we might NOT set the columns of db_seg_df not
    # in segments_df. So for safetey set them now:
    segments_df[SEG_ID] = np.nan  # coerce to valid type (should be int, however allow nans)
    segments_df[SEG_RETRY] = True  # coerce to valid type
    segments_df[SEG_START] = pd.NaT  # coerce to valid type
    segments_df[SEG_END] = pd.NaT  # coerce to valid type
    segments_df = mergeupdate(segments_df, db_seg_df, [SEG_CHID, SEG_EVID],
                              [SEG_ID, SEG_RETRY, SEG_START, SEG_END])

    # Now check time bounds: segments_df[SEG_START] and segments_df[SEG_END] are the OLD time
    # bounds, cause we just set them on segments_df from db_seg_df. Some of them might be NaT,
    # those not NaT mean the segment has already been downloaded (same (channelid, eventid))
    # Now, for those non-NaT segments, set retry=True if the OLD time bounds are different
    # than the new ones (tstart, tend).
    td0, td1 = timedelta(minutes=timespan[0]), timedelta(minutes=timespan[1])
    tstart, tend = (segments_df[SEG_ATIME] - td0).dt.round('s'), \
        (segments_df[SEG_ATIME] + td1).dt.round('s')
    retry_requests_timebounds = pd.notnull(segments_df[SEG_START]) & \
        ((segments_df[SEG_START] != tstart) | (segments_df[SEG_END] != tend))
    request_timebounds_need_update = retry_requests_timebounds.any()
    if request_timebounds_need_update:
        segments_df[SEG_RETRY] |= retry_requests_timebounds
    # retry column updated: clear old time bounds and set new ones just calculated:
    segments_df[SEG_START] = tstart
    segments_df[SEG_END] = tend

    oldlen = len(segments_df)
    # do a copy to avoid SettingWithCopyWarning. Moreover, copy should re-allocate contiguous
    # arrays which might be faster (and less memory consuming after unused memory is released)
    segments_df = segments_df[segments_df[SEG_RETRY]].copy()
    if oldlen != len(segments_df):
        reason = "already downloaded, no retry"
        logger.info(MSG("%d segments discarded", reason), oldlen-len(segments_df))

    if segments_df.empty:
        raise QuitDownload("Nothing to download: all segments already downloaded according to "
                           "the current configuration")

    # warn the user if we have duplicated segments, i.e. segments of the same
    # (channel_id, request_start, request_end). This can happen when we have to very close
    # events. Note that the time bounds are given by the combinations of
    # [event.lat, event.lon, event.depth_km, segment.event_distance_deg] so the condition
    # 'duplicated segments' might actually happen
    seg_dupes_mask = segments_df.duplicated(subset=[SEG_CHID, SEG_START, SEG_END], keep=False)
    if seg_dupes_mask.any():
        seg_dupes = segments_df[seg_dupes_mask]
        logger.info(MSG("%d suspiciously duplicated segments found:\n"
                        "this is due to different events arriving at the same station's channel\n"
                        "at the same exact date and time (rounded to the nearest second).\n"
                        "Probably, the same event has been returned with different id(s)\n"
                        "by the event web service, but this is not checked for: \n"
                        "all suspiciously duplicated segments will be written to the database."),
                    len(seg_dupes))
        logwarn_dataframe(seg_dupes.sort_values(by=[SEG_CHID, SEG_START, SEG_END]),
                          "Suspicious duplicated segments",
                          [SEG_CHID, SEG_START, SEG_END, SEG_EVID],
                          max_row_count=100)

    segments_df.drop([SEG_RETRY], axis=1, inplace=True)
    # return python bool, not numpy bool: use .item():
    return segments_df, request_timebounds_need_update.item()


def get_seg_request(segments_df, datacenter_url, chaid2mseedid):
    """returns a Request object from the given segments_df

    :param chaid2mseedid: dict of channel ids (int) mapped to mseed ids
        (strings in "Network.station.location.channel" format)
    """
    # For convenience and readability, define once the mapped column names representing the
    # dataframe columns that we need:
    SEG_START = Segment.request_start.key
    SEG_END = Segment.request_end.key
    CHA_ID = Segment.channel_id.key

    stime = segments_df[SEG_START].iloc[0].isoformat()
    etime = segments_df[SEG_END].iloc[0].isoformat()

    post_data = "\n".join("{} {} {}".format(*(chaid2mseedid[chaid].replace("..", ".--.").
                                              replace(".", " "), stime, etime))
                          for chaid in segments_df[CHA_ID] if chaid in chaid2mseedid)
    return Request(url=datacenter_url, data=post_data.encode('utf8'))


def download_save_segments(session, segments_df, datacenters_df, chaid2mseedid, download_id,
                           update_request_timebounds, max_thread_workers, timeout,
                           download_blocksize, db_bufsize, show_progress=False):

    """Downloads and saves the segments. segments_df MUST not be empty (this is not checked for)

        :param segments_df: the dataframe resulting from `prepare_for_download`
        :param chaid2mseedid: dict of channel ids (int) mapped to mseed ids
        (strings in "Network.station.location.channel" format)

    """
    # For convenience and readability, define once the mapped column names representing the
    # dataframe columns that we need:
    SEG_CHAID = Segment.channel_id.key
    SEG_DCID = Segment.datacenter_id.key
    DC_ID = DataCenter.id.key
    DC_DSURL = DataCenter.dataselect_url.key
    SEG_ID = Segment.id.key
    SEG_START = Segment.request_start.key
    SEG_END = Segment.request_end.key
    SEG_STIME = Segment.start_time.key
    SEG_ETIME = Segment.end_time.key
    SEG_DATA = Segment.data.key
    SEG_DSCODE = Segment.download_code.key
    SEG_DATAID = Segment.data_seed_id.key
    SEG_MGAP = Segment.maxgap_numsamples.key
    SEG_SRATE = Segment.sample_rate.key
    SEG_DOWNLID = Segment.download_id.key
    SEG_ATIME = Segment.arrival_time.key

    # set once the dict of column names mapped to their default values.
    # Set nan to let pandas understand it's numeric. None I don't know how it is converted
    # (should be checked) but it's for string types
    # for numpy types, see
    # https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html#specifying-and-constructing-data-types
    # Use OrderedDict to preserve order (see comments below)
    segvals = OrderedDict([(SEG_DATA, None), (SEG_SRATE, np.nan), (SEG_MGAP, np.nan),
                           (SEG_DATAID, None), (SEG_DSCODE, np.nan), (SEG_STIME, pd.NaT),
                           (SEG_ETIME, pd.NaT)])
    # Define separate keys cause we will use it elsewhere:
    # Note that the order of these keys must match `mseed_unpack` returned data
    # (this is why we used OrderedDict above)
    SEG_COLNAMES = list(segvals.keys())
    # define default error codes:
    URLERR_CODE, MSEEDERR_CODE, OUTTIME_ERR, OUTTIME_WARN = custom_download_codes()
    SEG_NOT_FOUND = None

    stats = DownloadStats()

    datcen_id2url = datacenters_df.set_index([DC_ID])[DC_DSURL].to_dict()

    colnames2update = [SEG_DOWNLID, SEG_DATA, SEG_SRATE, SEG_MGAP, SEG_DATAID, SEG_DSCODE,
                       SEG_STIME, SEG_ETIME]
    if update_request_timebounds:
        colnames2update += [SEG_START, SEG_ATIME, SEG_END]

    cols_to_log_on_err = [SEG_ID, SEG_CHAID, SEG_START, SEG_END, SEG_DCID]
    segmanager = DbManager(session, Segment.id, colnames2update,
                           db_bufsize, return_df=False,
                           oninsert_err_callback=handledbexc(cols_to_log_on_err, update=False),
                           onupdate_err_callback=handledbexc(cols_to_log_on_err, update=True))

    # define the groupsby columns
    # remember that segments_df has columns:
    # ['channel_id', 'datacenter_id', 'event_distance_deg', 'event_id', 'arrival_time',
    #  'request_start', 'request_end', 'id']
    # first try to download per-datacenter and time bounds. On 413, load each
    # segment separately (thus use SEG_DCID_NAME, SEG_SART_NAME, SEG_END_NAME, SEG_CHAID_NAME
    # (and SEG_EVTID_NAME for safety?)

    # we should group by (net, sta, loc, stime, etime), meaning that two rows with those values
    # equal will be given in the same sub-dataframe, and if 413 is found, take 413s erros creating a
    # new dataframe, and then group segment by segment, i.e.
    # (net, sta, loc, cha, stime, etime).
    # Unfortunately, for perf reasons we do not have
    # the first 4 columns, but we do have channel_id which basically comprises (net, sta, loc, cha)
    # NOTE: SEG_START and SEG_END MUST BE ALWAYS PRESENT IN THE SECOND AND THORD POSITION!!!!!
    requeststart_index = 1
    requestend_index = 2
    groupsby = [
                [SEG_DCID, SEG_START, SEG_END],
                [SEG_DCID, SEG_START, SEG_END, SEG_CHAID],
                ]

    if sys.version_info[0] < 3:
        def get_host(r):
            return r.get_host()
    else:
        def get_host(r):
            return r.host

    # we assume it's the terminal, thus allocate the current process to track
    # memory overflows
    with get_progressbar(show_progress, length=len(segments_df)) as bar:

        skipped_dataframes = []  # store dataframes with a 413 error and retry later
        for group_ in groupsby:

            if segments_df.empty:  # for safety (if this is the second loop or greater)
                break

            islast = group_ == groupsby[-1]
            seg_groups = segments_df.groupby(group_, sort=False)
            # seg group is an iterable of 2 element tuples. The first element is the tuple
            # of keys[:idx] values, and the second element is the dataframe
            itr = read_async(seg_groups,
                             urlkey=lambda obj: get_seg_request(obj[1], datcen_id2url[obj[0][0]],
                                                                chaid2mseedid),
                             raise_http_err=False,
                             max_workers=max_thread_workers,
                             timeout=timeout, blocksize=download_blocksize)

            for df, result, exc, request in itr:
                groupkeys_tuple = df[0]
                df = df[1]  # copy data so that we do not have refs to the old dataframe
                # and hopefully the gc works better
                url = get_host(request)
                data, code, msg = result if not exc else (None, None, None)
                if code == 413 and len(df) > 1 and not islast:
                    skipped_dataframes.append(df)
                    continue
                # Seems that copy(), although allocates a new small memory chunk,
                # helps gc better managing total memory (which might be an issue):
                df = df.copy()
                # init columns with default values:
                for col in SEG_COLNAMES:
                    df[col] = segvals[col]
                    # Note that we could use
                    # df.insert(len(df.columns), col, segvals[col])
                    # to preserve order, if needed. A starting discussion on adding new column:
                    # https://stackoverflow.com/questions/12555323/adding-new-column-to-existing-dataframe-in-python-pandas
                # init download id column with our download_id:
                df[SEG_DOWNLID] = download_id
                if exc:
                    code = URLERR_CODE
                elif code >= 400:
                    exc = "%d: %s" % (code, msg)
                elif not data:
                    # if we have empty data set only specific columns:
                    # (avoid mseed_id as is useless string data on the db, and we can retrieve it
                    # via station and channel joins in case)
                    df.loc[:, SEG_DATA] = b''
                    df.loc[:, SEG_DSCODE] = code
                    stats[url][code] += len(df)
                else:
                    try:
                        starttime = groupkeys_tuple[requeststart_index]
                        endtime = groupkeys_tuple[requestend_index]
                        resdict = mseedunpack(data, starttime, endtime)
                        oks = 0
                        errors = 0
                        outtime_warns = 0
                        outtime_errs = 0
                        # iterate over df rows and assign the relative data
                        # Note that we could use iloc which is SLIGHTLY faster than
                        # loc for setting the data, but this would mean using column
                        # indexes and we have column labels. A conversion is possible but
                        # would make the code  hard to understand (even more ;))
                        for idxval, chaid in zip(df.index.values, df[SEG_CHAID]):
                            mseedid = chaid2mseedid.get(chaid, None)
                            if mseedid is None:
                                continue
                            # get result:
                            res = resdict.get(mseedid, None)
                            if res is None:
                                continue
                            err, data, s_rate, max_gap_ratio, stime, etime, outoftime = res
                            if err is not None:
                                # set only the code field.
                                # Use set_value as it's faster for single elements
                                df.set_value(idxval, SEG_DSCODE, MSEEDERR_CODE)
                                errors += 1
                            else:
                                _code = code
                                if outoftime is True:
                                    if data:
                                        _code = OUTTIME_WARN
                                        outtime_warns += 1
                                    else:
                                        _code = OUTTIME_ERR
                                        outtime_errs += 1
                                else:
                                    oks += 1
                                # This raises a UnicodeDecodeError:
                                # df.loc[idxval, SEG_COLNAMES] = (data, s_rate,
                                #                                 max_gap_ratio,
                                #                                 mseedid, code)
                                # The problem (bug?) is in pandas.core.indexing.py
                                # on line 517: np.array((data, s_rate, max_gap_ratio,
                                #                                  mseedid, code))
                                # (numpy coerces to unicode if one of the values is unicode,
                                #  and thus fails for the `data` field?)
                                # Anyway, we set first an empty string (which can be
                                # decoded) and then use set_value only for the `data` field
                                # set_value should be relatively fast
                                df.loc[idxval, SEG_COLNAMES] = (b'', s_rate, max_gap_ratio,
                                                                mseedid, _code, stime, etime)
                                df.set_value(idxval, SEG_DATA, data)

                        if oks:
                            stats[url][code] += oks
                        if errors:
                            stats[url][MSEEDERR_CODE] += errors
                        if outtime_errs:
                            stats[url][OUTTIME_ERR] += outtime_errs
                        if outtime_warns:
                            stats[url][OUTTIME_WARN] += outtime_warns

                        unknowns = len(df) - oks - errors - outtime_errs - outtime_warns
                        if unknowns > 0:
                            stats[url][SEG_NOT_FOUND] += unknowns
                    except MSeedError as mseedexc:
                        code = MSEEDERR_CODE
                        exc = mseedexc

                if exc is not None:
                    df.loc[:, SEG_DSCODE] = code
                    stats[url][code] += len(df)
                    logger.warning(MSG("Segment download error, code %s" % str(code),
                                       exc, request))

                segmanager.add(df)
                bar.update(len(df))

            segmanager.flush()  # flush remaining stuff to insert / update, if any

            if skipped_dataframes:
                segments_df = pd.concat(skipped_dataframes, axis=0, ignore_index=True, copy=True,
                                        verify_integrity=False)
                skipped_dataframes = []
            else:
                # break the next loop, if any
                segments_df = pd.DataFrame()

    segmanager.close()  # flush remaining stuff to insert / update, if any, and prints info

    stats.normalizecodes()  # this makes potential string code merge into int codes
    return stats


