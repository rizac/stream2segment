"""
Segments download functions

:date: Dec 3, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
from datetime import timedelta
from collections import OrderedDict
import logging
from urllib.request import Request

import numpy as np
import pandas as pd

from stream2segment.io import Fdsnws
from stream2segment.io.cli import get_progressbar
from stream2segment.io.db.pdsql import dbquery2df, mergeupdate, DbManager
from stream2segment.download.db.models import WebService, Segment, Station, Channel,
from stream2segment.download.modules.utils import (DbExcLogger, logwarn_dataframe,
                                                   DownloadStats, formatmsg,
                                                   s2scodes, url2str, fdsn_url)
from stream2segment.download.exc import NothingToDownload
from stream2segment.download.modules.mseedlite import MSeedError, unpack as mseedunpack
from stream2segment.download.url import get_opener, get_host, read_async, \
    adjust_max_concurrent_downloads

# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial):
logger = logging.getLogger(__name__)


def prepare_for_download(session, segments_df, dc_dataselect_manager, timespan,
                         retry_seg_not_found, retry_url_err, retry_mseed_err,
                         retry_client_err, retry_server_err,
                         retry_timespan_err, retry_timespan_warn=False):
    """Drop the segments which are already present on the database and updates
    the primary keys for those not present (adding them to the db). Add new
    columns to the returned Data frame

    :param session: the sql-alchemy session bound to an existing database
    :param segments_df: pandas DataFrame resulting from `get_arrivaltimes`
    """
    opendataonly = dc_dataselect_manager.opendataonly
    # Fetch already downloaded segments and return the corresponding dataframe.
    # which will have also the boolean column SEG.RETRY, which is True for
    # suspiciously restricted (SR) segments, i.e. segments whose download code
    # MIGHT denote that they are restricted (see `s2scodes.restricted_data`):
    db_seg_df = fetch_already_downloaded_segments_df(session, segments_df,
                                                     opendataonly)
    # store now the ids of the SR segments, we will use them later.
    # If open data, `db_seg_df` does not have the column SEG.RETRY so set the
    # ids to a (empty) DataFrame for consistency:
    force_retry_ids = pd.DataFrame() if opendataonly else \
        db_seg_df[SEG.ID][db_seg_df[SEG.RETRY]]
    # Now update the SEG.RETRY col. (or create it) according to the flags set:
    set_segments_to_retry(db_seg_df, opendataonly, retry_seg_not_found,
                          retry_url_err, retry_mseed_err, retry_client_err,
                          retry_server_err, retry_timespan_err,
                          retry_timespan_warn)

    # Now merge/update existing dataframe (`segments_df`) with the db values
    # (`db_seg_df`). Do it in two steps: 1) set columns and defaults (for int
    # types, sets np.nan). Note that if we have something to retry
    # (db_seg_df[SEG_RETRY].any()), we add also a column SEG.DWLCODE with
    # None/nan as default: checking if that column exists will be the way later
    # to know if we need to update rows or only insert new rows.
    cols2set = OrderedDict([(SEG.ID, np.nan), (SEG.RETRY, True),
                            (SEG.REQSTART, pd.NaT), (SEG.REQEND, pd.NaT)] +
                           ([(SEG.DWLCODE, np.nan)]
                            if db_seg_df[SEG.RETRY].any() else []))
    for colname, default_ in cols2set.items():
        segments_df[colname] = default_
    # 2) assign/override values of cols2set from db_seg_df to segments_df,
    # matching rows via the [SEG_CHID, SEG_EVID] cols:
    segments_df = mergeupdate(segments_df, db_seg_df, [SEG.CHAID, SEG.EVID],
                              list(cols2set.keys()))

    request_timebounds_need_update = set_requested_timebounds(segments_df,
                                                              timespan)

    oldlen = len(segments_df)
    # do a copy to avoid SettingWithCopyWarning. Moreover, copy should
    # re-allocate contiguous arrays which might be faster (and less memory
    # consuming after unused memory is released)
    segments_df = segments_df[segments_df[SEG.RETRY]].copy()
    if oldlen != len(segments_df):
        reason = "already downloaded, no retry"
        logger.info(formatmsg("%d segments discarded", reason),
                    oldlen-len(segments_df))

    if segments_df.empty:
        raise NothingToDownload("Nothing to download: all segments already "
                                "downloaded according to the current "
                                "configuration")

    check_suspiciously_duplicated_segment(segments_df)

    # Last step: the policy later will be to UPDATE (=overwrite existing
    # segments on the database) only segments whose download code changed (see
    # comment on line 354)  because yes, it might save a lot of time. E.g.,
    # suppose retry_server_error=true and a segment on the db with download
    # code=500 => update it only if the server returns some code != 500.
    # However, if we are downloading with credentials, we need to force
    # updating SR segments which were downloaded with no credentials, by
    # definition of SR (suspiciously restricted). Thus, if we have those
    # segments (`not force_retry_ids.empty`) and we are performing a download
    # on an already existing database (`SEG.DWLCODE in segments_df.columns`),
    # for those SR segments we will set the value of the column `SEG.DWLCODE` to
    # None/nan: as we will never get any response code = None from the server,
    # those SR segments will always be updated
    if not force_retry_ids.empty and SEG.DWLCODE in segments_df.columns:
        segments_df.loc[segments_df[SEG.ID].isin(force_retry_ids),
                        SEG.DWLCODE] = np.nan

    segments_df.drop([SEG.RETRY], axis=1, inplace=True)
    # sort values in order to 1. download first most recent events and 2: shuffle
    # datacenters to try diversify the requests to different URLs
    segments_df.sort_values(by=SEG.REQSTART, ascending=False, inplace=True)

    return segments_df, request_timebounds_need_update


def fetch_already_downloaded_segments_df(session, segments_df,
                                         is_opendataonly):
    """Return a Dataframe with potentially already downloaded segments, using
    the existing `segments_df` dataframe of currently to-download segments.
    If `is_opendataonly` is False, the returned dataframe will also have a
    column named SEG.RETRY with boolean denoting segments that should be
    re-downloaded regardless of the user-defined classes (e.g. 204, 404)
    """
    codes = s2scodes
    # set the list of columns to query
    columns2query = [Segment.id, Segment.channel_id, Segment.request_start,
                     Segment.request_end, Segment.download_code,
                     Segment.event_id]
    # if downloading with authorization, add a boolean last column representing
    # when retry has to be forced. This happens when all the following two
    # conditions are met:
    # 1. segment was downloaded with no credentials
    # 2. segment download code suggests unauthorized access
    #    (codes.restricted_data = 404, 204, 401, 403)
    # (segments already downloaded with credentials and with code 404, 401,
    # 403 will be retried if the flag 'retry_client_err' is True, as usual)
    if not is_opendataonly:
        columns2query += [(Segment.download_code.isnot(None) &
                           Segment.download_code.in_(codes.restricted_data) &
                           Segment.queryauth.isnot(True)).label(SEG.RETRY)]
    # Note above: we need isnot(None) because in_(codes.restricted_data) might
    # return None for segment with NULL download status code (we want either
    # True or False, not None)

    # query relevant data into data frame (speeds up calculations:
    chids = pd.unique(segments_df[SEG.CHAID]).tolist()
    evids = pd.unique(segments_df[SEG.EVID]).tolist()
    return dbquery2df(session.query(*columns2query).
                      filter(Segment.channel_id.in_(chids) &  # noqa
                             Segment.event_id.in_(evids)  # noqa
                             )
                      )


def set_segments_to_retry(db_seg_df, is_opendataonly, retry_seg_not_found,
                          retry_url_err, retry_mseed_err, retry_client_err,
                          retry_server_err, retry_timespan_err,
                          retry_timespan_warn):
    """Set the segments to retry by appending a boolean column SEG.RETRY. Such
    a column might already exist if we are downloading restricted data.
    `db_seg_df` is modified in place
    """
    codes = s2scodes
    # set the boolean array telling whether we need to retry db_seg_df elements
    # (those already downloaded)
    mask = False
    if retry_seg_not_found:
        mask |= pd.isnull(db_seg_df[SEG.DWLCODE])
    if retry_url_err:
        mask |= db_seg_df[SEG.DWLCODE] == codes.url_err
    if retry_mseed_err:
        mask |= db_seg_df[SEG.DWLCODE] == codes.mseed_err
    if retry_client_err:
        mask |= (db_seg_df[SEG.DWLCODE] >= 400) & (db_seg_df[SEG.DWLCODE] < 500)
    if retry_server_err:
        mask |= (db_seg_df[SEG.DWLCODE] >= 500) & (db_seg_df[SEG.DWLCODE] < 600)
    if retry_timespan_err:
        mask |= db_seg_df[SEG.DWLCODE] == codes.timespan_err
    if retry_timespan_warn:
        mask |= db_seg_df[SEG.DWLCODE] == codes.timespan_warn

    if is_opendataonly:
        # SEG_RETRY is not in db_seg_df, assing:
        db_seg_df[SEG.RETRY] = mask
    elif mask is not False:  # just to avoid useless operations
        # SEG_RETRY is in db_seg_df, merge:
        db_seg_df[SEG.RETRY] |= mask


def set_requested_timebounds(segments_df, timespan):
    """For each row of `segments_df`: 1. compares the request start and end
    with the new requested time bounds (calculated from `timespan` and each
    segments arrival time). 2. checks for changed time bounds, setting the
    RETRY column to True. 3. eventually, sets the new requested time bounds.
    This function modifies `segments_df` in place.

    :return: boolean indicating if any (at least one) segment must be
        re-downloaded because the request timebounds changed
    """
    # Now check time bounds: segments_df[SEG_START] and segments_df[SEG_END]
    # are the OLD time bounds, cause we just set them on segments_df from
    # db_seg_df. Some of them might be NaT, those not NaT mean the segment has
    # already been downloaded (same (channelid, eventid)). Now, for those
    # non-NaT segments, set retry=True if the OLD time bounds are different
    # than the new ones (tstart, tend).
    td0, td1 = timedelta(minutes=timespan[0]), timedelta(minutes=timespan[1])
    tstart, tend = (segments_df[SEG.ATIME] + td0).dt.round('s'), \
        (segments_df[SEG.ATIME] + td1).dt.round('s')
    retry_requests_timebounds = pd.notnull(segments_df[SEG.REQSTART]) & \
        ((segments_df[SEG.REQSTART] != tstart) |
         (segments_df[SEG.REQEND] != tend))
    request_timebounds_need_update = retry_requests_timebounds.any()
    if request_timebounds_need_update:
        segments_df[SEG.RETRY] |= retry_requests_timebounds
    # SEG.RETRY column updated: clear old time bounds and set new ones just
    # calculated:
    segments_df[SEG.REQSTART] = tstart
    segments_df[SEG.REQEND] = tend
    return request_timebounds_need_update.item()  # return Python boolean


def check_suspiciously_duplicated_segment(segments_df):
    """Check for suspiciously duplicated segments, i.e. different ids
    but same (channel_id, request_start, request_end). These segments stem from distinct
    events with very close spatio-temporal coordinates.
    This function simply logs a message if any such duplicated segment is found,
    it does NOT modify segments_df
    """
    seg_dupes_mask = segments_df.duplicated(subset=[SEG.CHAID, SEG.REQSTART,
                                                    SEG.REQEND],
                                            keep=False)
    if seg_dupes_mask.any():
        seg_dupes = segments_df[seg_dupes_mask]
        msg = ("%d suspiciously duplicated segments found: this is most likely\n"
               "due to events with different ids\n"
               "but same (or very close) latitude, longitude, depth and time.")
        logger.info(msg, len(seg_dupes))
        seg_dupes_sorted = seg_dupes.sort_values(by=[SEG.CHAID, SEG.REQSTART,
                                                     SEG.REQEND])
        logwarn_dataframe(seg_dupes_sorted, "Suspicious duplicated segments",
                          [SEG.CHAID, SEG.REQSTART, SEG.REQEND, SEG.EVID],
                          max_row_count=100)


class SEG:  # noqa
    """Simple enum-like container of strings defining the segment's
    related database/dataframe columns needed in this module
    """
    CHAID = Segment.channel_id.key  # noqa
    EVID = Segment.event_id.key  # noqa
    ATIME = Segment.arrival_time.key  # noqa
    REQSTART = Segment.request_start.key  # noqa
    REQEND = Segment.request_end.key  # noqa
    DCID = Segment.webservice_id.key  # noqa
    ID = Segment.id.key  # noqa
    START = Segment.start_time.key  # noqa
    END = Segment.end_time.key  # noqa
    DATA = Segment.data.key  # noqa
    DWLCODE = Segment.download_code.key  # noqa
    DATAID = Segment.data_seed_id.key  # noqa
    MGAP = Segment.maxgap_numsamples.key  # noqa
    SRATE = Segment.sample_rate.key  # noqa
    DWLID = Segment.download_id.key  # noqa
    QAUTH = Segment.queryauth.key  # noqa
    # non-db column temporary set to get what segment has to be re-downloaded:
    RETRY = "__do.download__"  # noqa
    NET = Station.network.key
    STA = Station.station.key
    LOC = Channel.location.key
    CHA = Channel.channel.key


_RETRY_CODES = {
    # HTTP status codes that, if received, allow to retry the download (but only
    # if the number of concurrent downloads is greater than the mapped int)
    429: 1,
    503: 2
}


def download_save_segments(session, segments_df, dc_dataselect_manager,
                           download_id, update_datacenters,
                           update_request_timebounds, max_thread_workers,
                           timeout, download_blocksize, db_bufsize,
                           show_progress=False):
    """Download and saves the segments. segments_df MUST not be empty (this is
    not checked for)

    :param segments_df: the dataframe resulting from `prepare_for_download`.
        The Dataframe might or might not have the column 'download_code'. If it
        has, it will skip writing to db segments whose code did not change: in
        this case, nans stored under 'download_code' in segments_df indicate
        new segments, or segments for which the update has to be forced,
        whatever code is obtained (e.g., queryauth when previously a simple
        query was used)
    """
    # set queryauth column here, outside the loop:
    restricted_enable_dcids = dc_dataselect_manager.restricted_enabled_ids
    if restricted_enable_dcids:
        segments_df[SEG.QAUTH] = segments_df[SEG.DCID].\
            isin(restricted_enable_dcids)
    else:
        segments_df[SEG.QAUTH] = False

    segmanager = get_dbmanager(session, update_datacenters,
                               update_request_timebounds, db_bufsize)
    stats = DownloadStats()

    # these are the column names to be set on a dataframe from a received
    # response, mapped to their default value. Set nan to let pandas understand
    # it's numeric. None I don't know how it is converted (should be checked)
    # but it's for string types for numpy types, see
    # https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html#specifying-and-constructing-data-types
    defaultvalues = {
        SEG.DATA: None, SEG.SRATE: np.nan, SEG.MGAP: np.nan,
        SEG.DATAID: None, SEG.DWLCODE: np.nan, SEG.START: pd.NaT,
        SEG.END: pd.NaT, SEG.DWLID: download_id
    }
    defaultvalues_nodata = dict(defaultvalues)  # copy
    toupdate = SEG.DWLCODE in segments_df.columns
    code_not_found = s2scodes.seg_not_found
    skipped_same_code = 0

    if max_thread_workers is None:
        # set max thread workers here cause we might want to retry the download
        max_thread_workers = adjust_max_concurrent_downloads()

    # report seg. errors only once per error type and data center:
    seg_logger = SegmentLogger()
    with get_progressbar(len(segments_df) if show_progress else 0) as pbar:
        # store dataframes with a 413 error and retry later:
        skipped_dataframes = []
        while not segments_df.empty:

            dataframes = get_download_iterator(segments_df)
            for data, exc, code, request, dframe in \
                    get_responses(dataframes, dc_dataselect_manager,
                                  chaid2mseedid, max_thread_workers, timeout,
                                  download_blocksize):

                num_segments = len(dframe)
                url = get_host(request)
                url_stats = stats[url]

                if exc is None and data != b'':
                    # set default values on the dataframe (assign returns a
                    # copy):
                    dframe = dframe.assign(**defaultvalues)
                    populate_dataframe(data, code, dframe, chaid2mseedid)
                    # group by download code, count them, and add the counts to
                    # stats:
                    for kode, kount in get_counts(dframe, SEG.DWLCODE,
                                                  code_not_found):
                        url_stats[kode] += kount
                elif max_thread_workers > 1 and \
                        code in _RETRY_CODES and _RETRY_CODES[code] < max_thread_workers:
                    skipped_dataframes.append(dframe)
                    continue
                else:
                    # here we are if: exc is not None OR data = b''
                    url_stats[code] += num_segments
                    if toupdate and code is not None and \
                            (dframe[SEG.DWLCODE] == code).sum():  # noqa
                        # if there are rows to update, then discard those for
                        # which the code is the same in the database. If we
                        # requested a different time window, we should update
                        # the time windows but there is no point in this
                        # overhead. The condition `code is not None` should
                        # never happen but for safety we put it, because we
                        # have set the download code column of `dframe` to
                        # None/nan to mark segments to update nevertheless, on
                        # the assumption that we never get response code = None
                        # (see comment L.94). Thus, if for some weird reason
                        # the response code is None, then update the segment
                        # anyway (as we wanted to)
                        dframe = dframe[dframe[SEG.DWLCODE] != code]
                        skipped_same_code += num_segments - len(dframe)
                        if dframe.empty:  # nothing to update on the db
                            continue
                    # update dict of default values, and set it to the
                    # dataframe:
                    defaultvalues_nodata.update({SEG.DWLCODE: code,
                                                 SEG.DATA: data})
                    # Remember: `assign` returns a copy:
                    dframe = dframe.assign(**defaultvalues_nodata)

                    if exc is not None:
                        # log segment errors only once per error type and data
                        # center, otherwise the log is hundreds of Mb and it's
                        # unreadable:
                        seg_logger.warn(request, url, code, exc)

                segmanager.add(dframe)
                pbar.update(num_segments)

            segmanager.flush()  # flush remaining stuff to insert / update, if any

            if skipped_dataframes:
                segments_df = pd.concat(skipped_dataframes, axis=0,
                                        ignore_index=True, copy=True,
                                        verify_integrity=False)
                max_thread_workers = 2 if max_thread_workers > 2 else 1
                skipped_dataframes = []
            else:
                # break the next loop, if any
                segments_df = pd.DataFrame()

    segmanager.close()  # flush remaining stuff to insert / update

    if skipped_same_code:
        logger.warning(formatmsg(("%d already saved segment(s) with no "
                                  "waveform data skipped with no messages, "
                                  "only their count is reported "
                                  "in statistics") % skipped_same_code,
                                 "Still receiving the same download code"))
    return stats


def get_download_iterator(segments_df):
    """Yield dataframes to be downloaded, one request per dataframe, one waveform per
     dataframe row. The iterator is optimized to alternate datacenters when possible
     and group each dataframe by datacenter and time span
    """
    # Note: remember that the dataframe has been sorted descending by time (see
    # end of prepare_for_download for details):
    for _, dc_df in segments_df.groupby([SEG.REQSTART, SEG.REQEND, SEG.DCID],
                                        sort=False, observed=True):
        yield dc_df


def get_dbmanager(session, update_datacenter, update_request_timebounds, db_bufsize):
    """Return a DbManager for downloading waveform data"""
    colnames2update = [
        SEG.DWLID,
        SEG.DATA,
        SEG.SRATE,
        SEG.MGAP,
        SEG.DATAID,
        SEG.DWLCODE,
        SEG.START,
        SEG.END,
        SEG.QAUTH
    ]
    if update_request_timebounds:
        colnames2update += [SEG.REQSTART, SEG.ATIME, SEG.REQEND]
    if update_datacenter:
        colnames2update += [SEG.DCID]

    db_exc_logger = DbExcLogger([SEG.ID, SEG.CHAID, SEG.REQSTART, SEG.REQEND,
                                 SEG.DCID])

    return DbManager(session, Segment.id, colnames2update,
                     db_bufsize, return_df=False,
                     oninsert_err_callback=db_exc_logger.failed_insert,
                     onupdate_err_callback=db_exc_logger.failed_update)


def get_responses(dataframes, dc_dataselect_manager,
                  max_thread_workers, timeout, download_blocksize):
    """Download segments and yields results

    :param dataframes: iterable of dataframes, one dataframe per request, one row
        per requested waveform. Moreover, each dataframe row is assumed to refer to the
        same data center (base URL) and have the same time span (start end)
    """
    def openerfunc(dframe):
        """Return a Opener (or None) from the given dataframe. An Opener is the
        object needed to download restricted data"""
        return dc_dataselect_manager.opener(dframe[SEG.DCID].iloc[0])

    code_url_err, code_mseed_err = s2scodes.url_err, s2scodes.mseed_err
    for dframe, request, data, exc, code in read_async(
            dataframes,
            urlkey=get_seg_request,
            max_workers=max_thread_workers,
            timeout=timeout,
            blocksize=download_blocksize,
            openers=openerfunc
    ):
        if exc:
            data = None  # for safety
            if code is None:
                code = code_url_err
        else:
            exc = None  # for safety
            if not data:
                data = b''
            else:
                try:
                    start_time = dframe[SEG.REQSTART].iloc[0]
                    end_time = dframe[SEG.REQEND].iloc[0]
                    data = mseedunpack(data, start_time, end_time)
                except MSeedError as mseedexc:
                    code = code_mseed_err
                    exc = mseedexc
                    data = None  # for safety

        yield data, exc, code, request, dframe


def get_seg_request(segments_df):
    """Return a Request object from the given segments_df in form of URL string

    :paramsegments_df:  The dataframe denoting the segments to download. Each dataframe
        row is  assumed to refer to the same data center (base URL) and have the same
        time span (start end)
    """
    dc_url = segments_df['webservice_url'].iloc[0]
    params = {
        'start': segments_df[SEG.REQSTART].iloc[0],
        'end': segments_df[SEG.REQEND].iloc[0],
        'net': ','.join(sorted(segments_df[SEG.NET].unique())) or None,
        'sta': ','.join(sorted(segments_df[SEG.STA].unique())) or None,
        'loc': ','.join(sorted(segments_df[SEG.LOC].unique())) or None,
        'cha': ','.join(sorted(segments_df[SEG.CHA].unique())) or None,
    }

    return fdsn_url(dc_url, **params)

    # FIXME REMOVE
    # stime = segments_df[SEG.REQSTART].iloc[0]
    # etime = segments_df[SEG.REQEND].iloc[0]
    #
    #
    # post_data = "\n".join("{} {} {}".format(*(chaid2mseedid[chaid].
    #                                           replace("..", ".--.").
    #                                           replace(".", " "), stime, etime))
    #                       for chaid in segments_df[SEG.CHAID]
    #                       if chaid in chaid2mseedid)
    # return Request(url=datacenter_url, data=post_data.encode('utf8'))


def populate_dataframe(resdict, code, dframe, chaid2mseedid):
    """Write to dframe all necessary values according to `resdict`.

    :param resdict: a dict mapping miniseed_id (string) to the tuple
        err, data, s_rate, max_gap_ratio, stime, etime, outoftime.
        Return value of `mseedliste.mseedunpack` function
    :param dframe: the dataframe of the segments (one segment per row)
        whose waveform data was requested to the server. `resdict` is the
        result of `mseedliste.mseedunpack` on that server data
    """
    codes = s2scodes
    col_dscode = SEG.DWLCODE
    col_data = SEG.DATA
    # the order of these columns matters! see below
    columns2set = (
        col_data,
        SEG.SRATE,
        SEG.MGAP,
        SEG.DATAID,
        col_dscode,
        SEG.START,
        SEG.END
    )

    # iterate over dframe rows and assign the relative data
    # Note that we could use iloc which is SLIGHTLY faster than
    # loc for setting the data, but this would mean using column
    # indexes and we have column labels. A conversion is possible but
    # would make the code  hard to understand
    for idxval, chaid in zip(dframe.index.values, dframe[SEG.CHAID]):
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
            dframe.at[idxval, col_dscode] = codes.mseed_err
        else:
            # DO NOT MODIFY code attributes in loop! Otherwise
            # next segments might have invalid value(s)! Therefore, set _code:
            _code = code
            if outoftime is True:
                _code = codes.timespan_warn if data else codes.timespan_err
            # On old pandas versions (<=0.20?), this raised a
            # UnicodeDecodeError:
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
            # set_value should be relatively fast. Update 2018: set_value
            # deprecated. We use `at`
            dframe.loc[idxval, columns2set] = (b'', s_rate, max_gap_ratio,
                                               mseedid, _code, stime, etime)
            dframe.at[idxval, col_data] = data


def get_counts(dframe, dframe_column, na_key):
    """Return an iterable yielding the distinct values of
    `dframe[dframe_column]`. Each yielded element is (val, count), where val is
    one of the distinct values of `dframe[dframe_column]`. na_key is the value
    of `val` above to be yielded for na/none/nans'
    """
    # first count NA: if the dataframe has all NA, groupby raises
    # (group by skips NA)
    dframe_column = dframe[dframe_column]
    na_count = dframe_column.isna().sum()
    if na_count < len(dframe):
        # NOTE: groupby DOES NOT COUNT NA/Nones/NaNs
        for result in dframe.groupby(dframe_column).size().items():
            yield result  # result is (code, count)
    if na_count:
        yield na_key, na_count


class DcDataselectManager:
    """Class building the ground requirements for a dataselect download: it
    merges datacenters and authorization information in order to build URLs
    and openers for downloading waveform data"""

    def baseurl(self, dc_id):
        """Return the base url from a given datacenter id"""
        return self._data[dc_id][1]

    def opener(self, dc_id):
        """Return an Opener to be user with urllib module, or None (if no
        token/user+password has been provided for the given datacenter
        `dc_id`"""
        return self._data[dc_id][2]

    def __init__(self, datacenters_df, authorizer, show_progress=False):
        """Initialize a new DcDataselectManager"""
        DC_ID = WebService.id.key  # noqa
        DC_DSURL = WebService.url.key  # noqa

        # there is a handy function:
        # datacenters_df.set_index(keys_col)[values_col].to_dict,
        # but we want iterrows cause we convert any dc url to its FDSNws object
        dcid2fdsn = {int(row[DC_ID]): Fdsnws(row[DC_DSURL]) for _, row in
                     datacenters_df.iterrows()}
        # Note: Fdsnws might raise, but at this point datacenters_df is assumed
        # to be well formed
        errors = {}  # urls mapped to their exception
        if authorizer.token:
            token = authorizer.token
            self._data, errors = self._get_data_from_token(dcid2fdsn, token,
                                                           show_progress)
            self._restricted_id = [did for did in self._data
                                   if did not in errors]
        elif authorizer.userpass:
            user, password = authorizer.userpass
            self._data, errors = self._get_data_from_userpass(dcid2fdsn, user,
                                                              password)
            self._restricted_id = list(dcid2fdsn.keys())
        else:  # no authorization required
            self._data, errors = self._get_data_open(dcid2fdsn)
            self._restricted_id = []

        if errors:
            # map urls site to error, not dcids:
            errors = {dcid2fdsn[did].site: err for did, err in errors.items()}
            logger.info(formatmsg('Downloading open data only from: %s'
                                  % ", ".join(errors),
                                  'Unable to acquire credentials for '
                                  'restricted data'))
            for url, exc in errors.items():
                logger.warning(formatmsg("Downloading open data only, "
                                         "Unable to acquire credentials for "
                                         "restricted data",
                                         str(exc), url))

    @property
    def restricted_enabled_ids(self):
        """Return a set of integers denoting the datacenter id for which
        restricted data download is enabled
        """
        return set(self._restricted_id)

    @property
    def opendataonly(self):
        """Return true if **all** datacenters will download open data only.
        Open data only will be downloaded when no token is provided, or a wrong
        one
        """
        return False if self._restricted_id else True

    @staticmethod
    def _get_data_open(dcid2fdsn):
        errors = {}  # dummy var to return no error
        ret = {}
        for id_, fdsn in dcid2fdsn.items():
            url = fdsn.url(service=Fdsnws.DATASEL, method=Fdsnws.QUERY)
            opener = None
            ret[id_] = [fdsn, url, opener]
        return ret, errors


    @staticmethod
    def _get_data_from_userpass(dcid2fdsn, user, password):
        errors = {}  # dummy var to return no error
        ret = {}
        for id_, fdsn in dcid2fdsn.items():
            url = fdsn.url(service=Fdsnws.DATASEL, method=Fdsnws.QUERYAUTH)
            opener = get_opener(fdsn.site, user, password)
            ret[id_] = [fdsn, url, opener]
        return ret, errors

    @staticmethod
    def _get_data_from_token(dcid2fdsn, token, show_progress=False):

        def req(dcid):
            """Return a request from a datacenter id"""
            url = dcid2fdsn[dcid].url(service=Fdsnws.DATASEL,
                                      method=Fdsnws.AUTH)
            if url.lower().startswith('http:'):
                url = "https:" + url[5:]
            elif not url.lower().startswith('https:'):
                url = 'https:' + ('//' if url[:2] != '//' else '') + url
            return Request(url, data=token)

        data, errors = {}, {}
        with get_progressbar(len(dcid2fdsn) if show_progress else 0) as pbar:
            for dcid, url_, data_, exc, status_code in \
                    read_async(dcid2fdsn.keys(), urlkey=req, decode='utf8'):

                pbar.update(1)
                fdsn = dcid2fdsn[dcid]
                if exc is None:
                    if ':' not in data_:
                        exc = ValueError('Invalid user and password returned. '
                                         'This could be a data-center bug')
                    else:
                        user, pswd = data_.split(':')
                        data[dcid] = [fdsn,
                                      fdsn.url(service=Fdsnws.DATASEL,
                                               method=Fdsnws.QUERYAUTH),
                                      get_opener(fdsn.site, user, pswd)]
                if exc is not None:
                    data[dcid] = [fdsn,
                                  fdsn.url(service=Fdsnws.DATASEL,
                                           method=Fdsnws.QUERY),
                                  None]  # No opener, download open data only
                    errors[dcid] = exc
        return data, errors


class SegmentLogger(set):
    """A class handling segment errors and logging only once per error type and
    datacenter to avoid polluting the log file/stream with hundreds of Mbytes
    of redundant information"""

    def warn(self, request, url, code, exc):
        """Issue a logger.warn if the given error is not already reported

        :param request: the Request object
        :param url: string, usually the request's url host, to identify same
            data centers
        :param code: the error code
        :pram exc: the reported Exception
        """
        item = (url, code, str(exc.__class__.__name__))
        if item not in self:
            if not self:
                logger.warning('Detailed segment download errors '
                               '(showing only first of each type per data '
                               'center):')
            self.add(item)
            request_str = url2str(request)
            logger.warning(formatmsg("Segment download error, code %s" %
                                     str(code), exc, request_str))
