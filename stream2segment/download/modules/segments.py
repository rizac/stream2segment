'''
Download module for segments download

:date: Dec 3, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import map, next, zip, range, object
import os
from datetime import timedelta
import sys
from collections import OrderedDict
# import logging

import numpy as np
import pandas as pd

from stream2segment.io.db.models import DataCenter, Station, Channel, Segment, Fdsnws
from stream2segment.download.utils import read_async, NothingToDownload,\
    DbExcLogger, logwarn_dataframe, DownloadStats, formatmsg, s2scodes, url2str
from stream2segment.download.modules.mseedlite import MSeedError, unpack as mseedunpack
from stream2segment.utils import get_progressbar
from stream2segment.io.db.pdsql import dbquery2df, mergeupdate, DbManager

from stream2segment.utils.url import Request, get_host  # handle py2+3 compatibility

# logger: do not use logging.getLogger(__name__) but point to stream2segment.download.logger:
# this way we preserve the logging namespace hierarchy
# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial) when calling logging
# functions of stream2segment.download.utils:
from stream2segment.download import logger  # @IgnorePep8
from stream2segment.utils.url import get_opener


def prepare_for_download(session, segments_df, dc_dataselect_manager, timespan,
                         retry_seg_not_found, retry_url_err, retry_mseed_err, retry_client_err,
                         retry_server_err, retry_timespan_err, retry_timespan_warn=False):
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
    SEG_EVID = Segment.event_id.key  # pylint: disable=invalid-name
    SEG_ATIME = Segment.arrival_time.key  # pylint: disable=invalid-name
    SEG_START = Segment.request_start.key  # pylint: disable=invalid-name
    SEG_END = Segment.request_end.key  # pylint: disable=invalid-name
    SEG_CHID = Segment.channel_id.key  # pylint: disable=invalid-name
    SEG_ID = Segment.id.key  # pylint: disable=invalid-name
    SEG_DSC = Segment.download_code.key  # pylint: disable=invalid-name
    SEG_RETRY = "__do.download__"  # pylint: disable=invalid-name

    opendataonly = dc_dataselect_manager.opendataonly
    codes = s2scodes
    # set the list of columns to query
    columns2query = [Segment.id, Segment.channel_id, Segment.request_start, Segment.request_end,
                     Segment.download_code, Segment.event_id]
    # if downloading with authorization, add a boolean last column representing when retry has to
    # be forced. This happens when all the following two conditions are met:
    # 1. segment was downloaded with no credentials
    # 2. segment download code suggests unauthorized access
    #    (codes.restricted_data = 404, 204, 401, 403)
    # (segments already downloaded with credentials and with code 404, 401, 403 will be retried
    # if the flag 'retry_client_err' is True, as usual)
    if not opendataonly:
        columns2query += [(Segment.download_code.isnot(None) &
                           Segment.download_code.in_(codes.restricted_data) &
                           Segment.queryauth.isnot(True)).label(SEG_RETRY)]
    # Note above: we need isnot(None) because in_(codes.restricted_data) might return None
    # for segment with NULL download status code (we want either True or False, not None)

    # query relevant data into data frame (speeds up calculations:
    chids, evids = \
        pd.unique(segments_df[SEG_CHID]).tolist(), pd.unique(segments_df[SEG_EVID]).tolist()
    db_seg_df = dbquery2df(session.query(*columns2query).
                           filter(Segment.channel_id.in_(chids) &  # @UndefinedVariable
                                  Segment.event_id.in_(evids)  # @UndefinedVariable
                                  )
                           )

    # set the boolean array telling whether we need to retry db_seg_df elements (those already
    # downloaded)
    mask = False
    if retry_seg_not_found:
        mask |= pd.isnull(db_seg_df[SEG_DSC])
    if retry_url_err:
        mask |= db_seg_df[SEG_DSC] == codes.url_err
    if retry_mseed_err:
        mask |= db_seg_df[SEG_DSC] == codes.mseed_err
    if retry_client_err:
        mask |= db_seg_df[SEG_DSC].between(400, 499.9999, inclusive=True)
    if retry_server_err:
        mask |= db_seg_df[SEG_DSC].between(500, 599.9999, inclusive=True)
    if retry_timespan_err:
        mask |= db_seg_df[SEG_DSC] == codes.timespan_err
    if retry_timespan_warn:
        mask |= db_seg_df[SEG_DSC] == codes.timespan_warn

    force_retry_ids = None
    if opendataonly:
        # SEG_RETRY is not in db_seg_df, assing:
        db_seg_df[SEG_RETRY] = mask
    else:
        # store segments to retry ALWAYS now, before merging with other 'retry':
        # these are the segments which are suspiciously restricted and must be always re-downloaded
        # How? by setting their download_code
        force_retry_ids = db_seg_df[SEG_ID][db_seg_df[SEG_RETRY]]
        if force_retry_ids.empty:  # force to None if there are no ids to retry:
            force_retry_ids = None
        if mask is not False:  # just to avoid useless operations
            # SEG_RETRY is in db_seg_df, merge:
            db_seg_df[SEG_RETRY] |= mask

    # update existing dataframe:
    # 1) set columns and defaults (for int types, set np.nan):
    # Note that if we have something to retry (db_seg_df[SEG_RETRY].any()), we add also
    # a None download code as default. Merged with db_seg_df later, we will know when we can skip
    # the download (doiwnload code did not change):
    cols2set = OrderedDict([(SEG_ID, np.nan), (SEG_RETRY, True), (SEG_START, pd.NaT),
                            (SEG_END, pd.NaT)] +
                           ([(SEG_DSC, np.nan)] if db_seg_df[SEG_RETRY].any() else []))
    for colname, default_ in cols2set.items():
        segments_df[colname] = default_
    # 2) assign/override values of cols2set from db_seg_df to segments_df,
    # matching rows via the [SEG_CHID, SEG_EVID] cols:
    segments_df = mergeupdate(segments_df, db_seg_df, [SEG_CHID, SEG_EVID],
                              list(cols2set.keys()))

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
        logger.info(formatmsg("%d segments discarded", reason), oldlen-len(segments_df))

    if segments_df.empty:
        raise NothingToDownload("Nothing to download: all segments already downloaded "
                                "according to the current configuration")

    # warn the user if we have duplicated segments, i.e. segments of the same
    # (channel_id, request_start, request_end). This can happen when we have to very close
    # events. Note that the time bounds are given by the combinations of
    # [event.lat, event.lon, event.depth_km, segment.event_distance_deg] so the condition
    # 'duplicated segments' might actually happen
    seg_dupes_mask = segments_df.duplicated(subset=[SEG_CHID, SEG_START, SEG_END], keep=False)
    if seg_dupes_mask.any():
        seg_dupes = segments_df[seg_dupes_mask]
        logger.info(formatmsg("%d suspiciously duplicated segments found: this is most likely\n"
                              "due to events fetched from the event catalog with different ids\n"
                              "but same latitude, longitude and time."),
                    len(seg_dupes))
        logwarn_dataframe(seg_dupes.sort_values(by=[SEG_CHID, SEG_START, SEG_END]),
                          "Suspicious duplicated segments",
                          [SEG_CHID, SEG_START, SEG_END, SEG_EVID],
                          max_row_count=100)

    # Now set the download code for 204 and 404 with none download code to force rewrite
    # A None download code will force rewrite (SQL update) because the response code will
    # never be equal to None. Do this if there is something to retry
    # (SEG_DSC in segments_df.columns) and we have ids suspiciously downloaded as open:
    if force_retry_ids is not None and SEG_DSC in segments_df.columns:
        segments_df.loc[segments_df[SEG_ID].isin(force_retry_ids), SEG_DSC] = np.nan

    segments_df.drop([SEG_RETRY], axis=1, inplace=True)

    # return python bool, not numpy bool: use .item():
    return segments_df, request_timebounds_need_update.item()


# For convenience and readability, define once the mapped column names representing the
# dataframe columns that we need:
class SEG(object):
    CHAID = Segment.channel_id.key  # pylint: disable=invalid-name
    DCID = Segment.datacenter_id.key  # pylint: disable=invalid-name
    ID = Segment.id.key  # pylint: disable=invalid-name
    START = Segment.request_start.key  # pylint: disable=invalid-name
    END = Segment.request_end.key  # pylint: disable=invalid-name
    STIME = Segment.start_time.key  # pylint: disable=invalid-name
    ETIME = Segment.end_time.key  # pylint: disable=invalid-name
    DATA = Segment.data.key  # pylint: disable=invalid-name
    DSCODE = Segment.download_code.key  # pylint: disable=invalid-name
    DATAID = Segment.data_seed_id.key  # pylint: disable=invalid-name
    MGAP = Segment.maxgap_numsamples.key  # pylint: disable=invalid-name
    SRATE = Segment.sample_rate.key  # pylint: disable=invalid-name
    DOWNLID = Segment.download_id.key  # pylint: disable=invalid-name
    ATIME = Segment.arrival_time.key  # pylint: disable=invalid-name
    QAUTH = Segment.queryauth.key  # pylint: disable=invalid-name

    # these are the column names to be set on a dataframe from a received response,
    # mapped to their default value
    # Set nan to let pandas understand it's numeric. None I don't know how it is converted
    # (should be checked) but it's for string types
    # for numpy types, see
    # https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html#specifying-and-constructing-data-types
    # Use OrderedDict to preserve order (see comments below)
    DEFVALS = OrderedDict([
        (DATA, None),
        (SRATE, np.nan),
        (MGAP, np.nan),
        (DATAID, None),
        (DSCODE, np.nan),
        (STIME, pd.NaT),
        (ETIME, pd.NaT)
    ])

    # these is a list of column names to be set on a dataframe from a received response
    DEFCOLS = list(DEFVALS.keys())


def download_save_segments(session, segments_df, dc_dataselect_manager, chaid2mseedid,
                           download_id, update_request_timebounds, max_thread_workers, timeout,
                           download_blocksize, db_bufsize, show_progress=False):

    """Downloads and saves the segments. segments_df MUST not be empty (this is not checked for)

        :param segments_df: the dataframe resulting from `prepare_for_download`. The Dataframe
            might or might not have the column 'download_code'. If it has, it will skip
            writing to db segments whose code did not change: in this case, nans stored under
            'download_code' in segments_df indicate new segments, or segments for which the update
            has to be forced, whatever code is obtained (e.g., queryauth when previously a simple
            query was used)
        :param chaid2mseedid: dict of channel ids (int) mapped to mseed ids
        (strings in "Network.station.location.channel" format)

    """
    # set queryauth column here, outside the loop:
    restricted_enable_dcids = dc_dataselect_manager.restricted_enabled_ids
    if restricted_enable_dcids:
        segments_df[SEG.QAUTH] = \
            segments_df[SEG.DCID].isin(dc_dataselect_manager.restricted_enabled_ids)
    else:
        segments_df[SEG.QAUTH] = False

    stats = DownloadStats()
    colnames2update = [
        SEG.DOWNLID,
        SEG.DATA,
        SEG.SRATE,
        SEG.MGAP,
        SEG.DATAID,
        SEG.DSCODE,
        SEG.STIME,
        SEG.ETIME,
        SEG.QAUTH
    ]
    if update_request_timebounds:
        colnames2update += [SEG.START, SEG.ATIME, SEG.END]

    db_exc_logger = DbExcLogger([SEG.ID, SEG.CHAID, SEG.START, SEG.END, SEG.DCID])

    segmanager = DbManager(session, Segment.id, colnames2update,
                           db_bufsize, return_df=False,
                           oninsert_err_callback=db_exc_logger.failed_insert,
                           onupdate_err_callback=db_exc_logger.failed_update)

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
    groupsby = [
        [SEG.DCID, SEG.START, SEG.END],
        [SEG.DCID, SEG.START, SEG.END, SEG.CHAID]
    ]

    # some variables to be used in the loop below:
    toupdate = SEG.DSCODE in segments_df.columns
    skipped_same_code = 0
    seg_logger = SegmentLogger()  # report seg. errors only once per error type and data center

    with get_progressbar(show_progress, length=len(segments_df)) as pbar:

        skipped_dataframes = []  # store dataframes with a 413 error and retry later
        for group_ in groupsby:

            if segments_df.empty:  # for safety (if this is the second loop or greater)
                break

            islast = group_ == groupsby[-1]
            seg_groups = segments_df.groupby(group_, sort=False)
            for seg_response, dframe, request in \
                    seg_responses(seg_groups, dc_dataselect_manager, chaid2mseedid,
                                  max_thread_workers, timeout, download_blocksize):

                if seg_response.code == 413 and not islast and len(dframe) > 1:
                    skipped_dataframes.append(dframe)
                    continue

                # update bar now:
                pbar.update(len(dframe))

                if toupdate:
                    dframe, _skippedcount = _purge_not2update(seg_response, dframe, stats)
                    if _skippedcount:
                        skipped_same_code += _skippedcount
                        if dframe.empty:
                            continue

                dframe = _setup_defaults_on_dataframe(dframe, download_id)
                _set_responsedata_on_dataframe(seg_response, dframe, stats, chaid2mseedid)

                if seg_response.exc is not None:
                    # log segment errors only once per error type and data center,
                    # otherwise the log is hundreds of Mb and it's unreadable:
                    seg_logger.warn(request, seg_response.url, seg_response.code, seg_response.exc)

                segmanager.add(dframe)

            segmanager.flush()  # flush remaining stuff to insert / update, if any

            if skipped_dataframes:
                segments_df = pd.concat(skipped_dataframes, axis=0, ignore_index=True, copy=True,
                                        verify_integrity=False)
                skipped_dataframes = []
            else:
                # break the next loop, if any
                segments_df = pd.DataFrame()

    segmanager.close()  # flush remaining stuff to insert / update

    if skipped_same_code:
        logger.warning(formatmsg(("%d already saved segment(s) with no waveform data skipped "
                                  "with no messages, only their count is reported "
                                  "in statistics") % skipped_same_code,
                                 "Still receiving the same download code"))
    return stats


def seg_responses(seg_groups, dc_dataselect_manager, chaid2mseedid,
                  max_thread_workers, timeout, download_blocksize):
    '''Downloads segments and yields results

        :param seg groups: is an iterable of 2 element tuples. The first element is the tuple
             of the 'groupby' values, and the second element is the dataframe
    '''
    def req(group_element):
        '''calls get_seg_request from an item of Pandas groupby. Used in read_async below

        :param group_element: an element yielded by iterating over pandas groupby
            https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#iterating-through-groups
        '''
        dframe = group_element[1]
        # group_element[0][0] is the datacenter id:
        dc_url = dc_dataselect_manager.baseurl(group_element[0][0])
        return get_seg_request(dframe, dc_url, chaid2mseedid)

    def openerfunc(group_element):
        '''Returns a urllib opener from the given obj

        :param group_element: an element yielded by iterating over pandas groupby
            https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#iterating-through-groups
        '''
        # group_element[0][0] is the datacenter id:
        return dc_dataselect_manager.opener(group_element[0][0])

    seg_response = SegResponse()
    code_url_err = s2scodes.url_err
    for (group_keys, dframe), result, exc, request in \
            read_async(seg_groups, urlkey=req, raise_http_err=False,
                       max_workers=max_thread_workers, timeout=timeout,
                       blocksize=download_blocksize, openers=openerfunc):
        # group_keys is the tuple (datacenter_id, request_start_time, request_end_time)
        seg_response.starttime = group_keys[1]  # pylint: disable=attribute-defined-outside-init
        seg_response.endtime = group_keys[2]  # pylint: disable=attribute-defined-outside-init
        seg_response.url = get_host(request)  # pylint: disable=attribute-defined-outside-init
        # set exception (might be None)
        seg_response.exc = exc  # pylint: disable=attribute-defined-outside-init
        # result is the tuple (data, http code, http message), or None if exc is not None
        # Note that exc can be only issued from an URLError, as HTTPErrors are returned in
        # the tuple (data, http code, http message) (code>=400). So:
        if exc:
            seg_response.data = None  # pylint: disable=attribute-defined-outside-init
            seg_response.code = code_url_err  # pylint: disable=attribute-defined-outside-init
            seg_response.msg = None  # pylint: disable=attribute-defined-outside-init
        else:
            seg_response.data = result[0]  # pylint: disable=attribute-defined-outside-init
            seg_response.code = result[1]  # pylint: disable=attribute-defined-outside-init
            seg_response.msg = result[2]  # pylint: disable=attribute-defined-outside-init

        yield seg_response, dframe, request


class SegResponse(object):
    '''Memory efficient container for all data related to a received segment response'''
    # Use slots (https://stackoverflow.com/a/28059785):
    __slots__ = [
        'url',  # the url DOMAIN of the request
        'starttime',  # request start time
        'endtime',  # request end time
        'exc',  # the response exception (URLError, non HTTPError), might be None
        'data',  # the waveform data (might be None)
        'code',  # the stream2segment download code (superset of HTTP status code)
        'msg'  # the server response message
    ]


def get_seg_request(segments_df, datacenter_url, chaid2mseedid):
    """returns a Request object from the given segments_df

    :param chaid2mseedid: dict of channel ids (int) mapped to mseed ids
        (strings in "Network.station.location.channel" format)
    """
    # For convenience and readability, define once the mapped column names representing the
    # dataframe columns that we need:
    SEG_START = Segment.request_start.key  # pylint: disable=invalid-name
    SEG_END = Segment.request_end.key  # pylint: disable=invalid-name
    CHA_ID = Segment.channel_id.key  # pylint: disable=invalid-name

    stime = segments_df[SEG_START].iloc[0].isoformat()
    etime = segments_df[SEG_END].iloc[0].isoformat()

    post_data = "\n".join("{} {} {}".format(*(chaid2mseedid[chaid].replace("..", ".--.").
                                              replace(".", " "), stime, etime))
                          for chaid in segments_df[CHA_ID] if chaid in chaid2mseedid)
    return Request(url=datacenter_url, data=post_data.encode('utf8'))


def _purge_not2update(seg_response, dframe, stats):
    '''Purges from dataframe the segments (rows) which should not be written
    to the database during an update download, because their download code did not
    change and the server response does not have data

    :return: the tuple dframe, skipped, where dframe is the dframe with only rows
        to be written, and skipped >=0 the removed rows
    '''
    # if there are rows to update and response has no data, then
    # discard those for which the code is the same. If we requested a different
    # time window, we should update the time windows but there is no point as the
    # db segment does not have data stored. The last condition (`code is not None`)
    # should never happen but for safety, otherwise we risk to skip segments
    # with download_code NA (which means exactly the opposite: force insert/update
    # them to the db. See comment on line 182):
    _skippedcount = 0
    if not seg_response.data and seg_response.code is not None:
        _skipped = dframe[SEG.DSCODE] == seg_response.code
        _skippedcount = _skipped.sum()
        if _skippedcount:
            dframe = dframe[~_skipped]
            stats[seg_response.url][seg_response.code] += _skippedcount
    return dframe, _skippedcount


def _setup_defaults_on_dataframe(dframe, download_id):
    '''Sets up the default values on a given dataframe representing the segments to
    download (one row per segment)
    '''
    # Seems that copy(), although allocates a new small memory chunk,
    # helps gc better managing total memory (which might be an issue).
    # Moreover, let's avoid pandas SettingsWithCopy warning:
    dframe = dframe.copy()
    # init columns with default values:
    for col, defval in SEG.DEFVALS.items():
        dframe[col] = defval
        # Note that we could use
        # dframe.insert(len(dframe.columns), col, segvals[col])
        # to preserve order, if needed. A starting discussion on adding new column:
        # https://stackoverflow.com/questions/12555323/adding-new-column-to-existing-dataframe-in-python-pandas
    # init download id column with our download_id:
    dframe[SEG.DOWNLID] = download_id
    return dframe


def _set_responsedata_on_dataframe(seg_response, dframe, stats, chaid2mseedid):
    '''Writes to dframe all necessary values according to `seg_response`

    :param seg_response: a SegResponse object
    :param dframe: the dataframe of the segments (one segment per row)
        whose waveform data was requested to the server. `seg_response` is the server response
        of that request
    '''
    # Here it is safe to `set seg_response` attributes, because we will use some
    # modified values in the caller (see above).
    # In _process_seg_response_with_waveform_data, when looping through
    # dframe rows, we can not set attributes otherwise they might affect subsequent looped rows
    if seg_response.exc is None:
        if seg_response.code >= 400:
            seg_response.exc = "%d: %s" % (seg_response.code, seg_response.msg)
        elif not seg_response.data:
            # if we have empty data set only specific columns:
            # (avoid mseed_id as is useless string data on the db, and we can
            # retrieve it via station and channel joins in case)
            dframe.loc[:, SEG.DATA] = b''
            dframe.loc[:, SEG.DSCODE] = seg_response.code
            stats[seg_response.url][seg_response.code] += len(dframe)
        else:
            try:
                _process_seg_response_with_waveform_data(seg_response, dframe, stats, chaid2mseedid)
            except MSeedError as mseedexc:
                seg_response.code = s2scodes.mseed_err
                seg_response.exc = mseedexc

    if seg_response.exc is not None:
        dframe.loc[:, SEG.DSCODE] = seg_response.code
        stats[seg_response.url][seg_response.code] += len(dframe)


def _process_seg_response_with_waveform_data(seg_response, dframe, stats, chaid2mseedid):
    '''Processes a response from the server in the case where waveform data has been received,
    writing to dframe all necessary values according to `seg_response`

    :param seg_response: a SegResponse object. `seg_response.data` (bytes) is surely not None
    :param dframe: the dataframe of the segments (one segment per row)
        whose waveform data was requested to the server. `seg_response` is the server response
        of that request
    '''
    unknowns = len(dframe)  # count of download status code = None (we will substract from it)
    codes = s2scodes
    resdict = mseedunpack(seg_response.data, seg_response.starttime, seg_response.endtime)
    # iterate over dframe rows and assign the relative data
    # Note that we could use iloc which is SLIGHTLY faster than
    # loc for setting the data, but this would mean using column
    # indexes and we have column labels. A conversion is possible but
    # would make the code  hard to understand (even more ;))
    for idxval, chaid in zip(dframe.index.values, dframe[SEG.CHAID]):
        mseedid = chaid2mseedid.get(chaid, None)
        if mseedid is None:
            continue
        # get result:
        res = resdict.get(mseedid, None)
        if res is None:
            continue
        unknowns -= 1
        err, data, s_rate, max_gap_ratio, stime, etime, outoftime = res
        if err is not None:
            # set only the code field.
            dframe.at[idxval, SEG.DSCODE] = codes.mseed_err
            stats[seg_response.url][codes.mseed_err] += 1
        else:
            # DO NOT MODIFY `seg_response` attributes in loop! Otherwise
            # next segments might have invalid value(s)! Therefore, set _code:
            _code = seg_response.code
            if outoftime is True:
                _code = codes.timespan_warn if data else codes.timespan_err
            stats[seg_response.url][_code] += 1
            # This raised a UnicodeDecodeError:
            # dframe.loc[idxval, SEG_COLNAMES] = (data, s_rate,
            #                                 max_gap_ratio,
            #                                 mseedid, code)
            # The problem (bug?) is in pandas.core.indexing.py
            # on line 517: np.array((data, s_rate, max_gap_ratio,
            #                                  mseedid, code))
            # (numpy coerces to unicode if one of the values is unicode,
            #  and thus fails for the `data` field?)
            # Anyway, we set first an empty string (which can be
            # decoded) and then use set_value only for the `data` field
            # set_value should be relatively fast. Update 2018: set_value deprecated
            # we use at
            dframe.loc[idxval, SEG.DEFCOLS] = (b'', s_rate, max_gap_ratio,
                                               mseedid, _code, stime, etime)
            dframe.at[idxval, SEG.DATA] = data

    if unknowns:
        stats[seg_response.url][codes.seg_not_found] += unknowns


class DcDataselectManager(object):
    '''Class building the ground requirements for a dataselect download: it merges
    datacenters and authorization information in order to build
    urls and openers for downloading waveform data'''

    def baseurl(self, dc_id):
        '''Returns the base url from a given datacenter id'''
        return self._data[dc_id][1]

    def opener(self, dc_id):
        '''Returns an Opener to be user with urllib module, or None (if no token/user+password
        has been provided for the given datacenter `dc_id`'''
        return self._data[dc_id][2]

    def __init__(self, datacenters_df, authorizer, show_progress=False):
        '''initializes a new DcDataselectManager'''
        DC_ID = DataCenter.id.key  # pylint: disable=invalid-name
        DC_DSURL = DataCenter.dataselect_url.key  # pylint: disable=invalid-name

        # there is a handy function datacenters_df.set_index(keys_col)[values_col].to_dict,
        # but we want iterrows cause we convert any dc url to its fdsnws object
        dcid2fdsn = {int(row[DC_ID]): Fdsnws(row[DC_DSURL]) for _, row in
                     datacenters_df.iterrows()}
        # Note: Fdsnws might raise, but at this point datacenters_df is assumed to be well
        # formed
        errors = {}  # urls mapped to their exception
        if authorizer.token:
            token = authorizer.token
            self._data, errors = self._get_data_from_token(dcid2fdsn, token, show_progress)
            self._restricted_id = [did for did in self._data if did not in errors]
        elif authorizer.userpass:
            user, password = authorizer.userpass
            self._data, errors = self._get_data_from_userpass(dcid2fdsn, user, password)
            self._restricted_id = list(dcid2fdsn.keys())
        else:  # no authorization required
            self._data, errors = self._get_data_open(dcid2fdsn)
            self._restricted_id = []

        if errors:
            # map urls site to error, not dcids:
            errors = {dcid2fdsn[dcid].site: err for dcid, err in errors.items()}
            logger.info(formatmsg('Downloading open data only from: %s' % ", ".join(errors),
                                  'Unable to acquire credentials for restricted data'))
            for url, exc in errors.items():
                logger.warning(formatmsg("Downloading open data only, "
                                         "Unable to acquire credentials for restricted data",
                                         str(exc), url))

    @property
    def restricted_enabled_ids(self):
        '''returns a set of integers denoting the datacenter id for which restricted data
        download is enabled'''
        return set(self._restricted_id)

    @property
    def opendataonly(self):
        '''Returns true if **all** datacenters will download open data only.
        Open data only will be downloaded when no token is provided, or a wrong one'''
        return False if self._restricted_id else True

    @staticmethod
    def _get_data_open(dcid2fdsn):
        errors = {}  # dummy var to return no error
        return {id_: [fdsn, fdsn.url(service=Fdsnws.DATASEL, method=Fdsnws.QUERY), None]
                for id_, fdsn in dcid2fdsn.items()}, errors

    @staticmethod
    def _get_data_from_userpass(dcid2fdsn, user, password):
        errors = {}  # dummy var to return no error
        return {id_: [fdsn, fdsn.url(service=Fdsnws.DATASEL, method=Fdsnws.QUERYAUTH),
                      get_opener(fdsn.site, user, password)]
                for id_, fdsn in dcid2fdsn.items()}, errors

    @staticmethod
    def _get_data_from_token(dcid2fdsn, token, show_progress=False):

        def req(dcid):
            '''returns a request from a datacenter id'''
            url = dcid2fdsn[dcid].url(service=Fdsnws.DATASEL, method=Fdsnws.AUTH)
            if url.lower().startswith('http:'):
                url = "https:" + url[5:]
            elif not url.lower().startswith('https:'):
                url = 'https:' + ('//' if url[:2] != '//' else '') + url
            return Request(url, data=token)

        data, errors = {}, {}
        with get_progressbar(show_progress, length=len(dcid2fdsn)) as pbar:
            for dcid, result, exc, _ in read_async(dcid2fdsn.keys(), urlkey=req,
                                                   decode='utf8', raise_http_err=True):

                pbar.update(1)
                fdsn = dcid2fdsn[dcid]
                if exc is None:
                    if ':' not in result[0]:
                        exc = ValueError('Invalid user and password returned. '
                                         'This could be a data-center bug')
                    else:
                        user, pswd = result[0].split(':')
                        data[dcid] = [fdsn,
                                      fdsn.url(service=Fdsnws.DATASEL, method=Fdsnws.QUERYAUTH),
                                      get_opener(fdsn.site, user, pswd)]
                if exc is not None:
                    data[dcid] = [fdsn,
                                  fdsn.url(service=Fdsnws.DATASEL, method=Fdsnws.QUERY),
                                  None]  # No opener, download open data only
                    errors[dcid] = exc
        return data, errors


class SegmentLogger(set):
    '''A class handling segment errors and logging only once per error type
    and datacenter to avoid polluting the log file/stream with hundreds of megabytes'''

    def warn(self, request, url, code, exc):
        '''issues a logger.warn if the given error is not already reported

        :param request: the Request object
        :param url: string, usually the request's url host, to identify same data centers
        :param code: the error code
        :pram exc: the reported Exception
        '''
        item = (url, code, str(exc.__class__.__name__))
        if item not in self:
            if not self:
                logger.warning('Detailed segment download errors '
                               '(showing only first of each type per data center):')
            self.add(item)
            request_str = url2str(request)
            logger.warning(formatmsg("Segment download error, code %s" % str(code),
                           exc, request_str))
