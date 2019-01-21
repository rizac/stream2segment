# -*- coding: utf-8 -*-
"""
Core functions and classes for the download routine

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""

# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import map, next, zip, range, object

import os
import logging

import psutil
import yaml

from stream2segment.io.db.pdsql import dbquery2df
from stream2segment.utils.resources import version
from stream2segment.download.utils import NothingToDownload, FailedDownload
from stream2segment.download.modules.events import get_events_df
from stream2segment.download.modules.datacenters import get_datacenters_df
from stream2segment.download.modules.channels import get_channels_df, chaid2mseedid_dict
from stream2segment.download.modules.stationsearch import merge_events_stations
from stream2segment.download.modules.segments import prepare_for_download,\
    download_save_segments, DcDataselectManager
from stream2segment.download.modules.stations import save_inventories, query4inventorydownload
from stream2segment.utils import tounicode
from stream2segment.io.db.models import Download


logger = logging.getLogger(__name__)


def run(session, download_id, eventws, start, end, dataws, event_query_params,
        networks, stations, locations, channels, min_sample_rate,
        search_radius, update_metadata, inventory, timespan,
        retry_seg_not_found, retry_url_err, retry_mseed_err, retry_client_err, retry_server_err,
        retry_timespan_err, tt_table, advanced_settings, authorizer, isterminal=False):
    """
        Downloads waveforms related to events to a specific path.
        This function is not intended to be called directly (PENDING: update doc?)

        :raise: :class:`FailedDownload` exceptions
    """

    # RAMAINDER: **Any function here EXPECTS THEIR DATAFRAME INPUT TO BE NON-EMPTY.**
    # Thus, any function returning a dataframe is responsible to return well formed (non empty)
    # data frames: if it would not be the case, the function should raise either:
    # 1) a NothingToDownload to stop the routine silently (log to info) and proceed to the
    # inventories (if set),
    # 2) a Faileddownload to stop the download immediately and raise the exception

    # set blocksize if zero:
    if advanced_settings['download_blocksize'] <= 0:
        advanced_settings['download_blocksize'] = -1
    if advanced_settings['max_thread_workers'] <= 0:
        advanced_settings['max_thread_workers'] = None
    dbbufsize = min(advanced_settings['db_buf_size'], 1)

    process = psutil.Process(os.getpid()) if isterminal else None
    # calculate steps (note that bool math works, e.g: 8 - True == 7):
    __steps = 1 if inventory == 'only' else \
        (6 + inventory + (True if authorizer.token else False))
    stepiter = iter(range(1, __steps+1))

    # custom function for logging.info different steps:
    def stepinfo(text, *args, **kwargs):
        step = next(stepiter)
        logger.info("\nSTEP %d of %d: {}".format(text), step, __steps, *args, **kwargs)
        if process is not None:
            percent = process.memory_percent()
            logger.warning("(%.1f%% memory used)", percent)

    try:
        if inventory != 'only':
            
            stepinfo("Requesting events")
            events_df = get_events_df(session, eventws, event_query_params, start, end,
                                      dbbufsize, advanced_settings['e_timeout'],
                                      advanced_settings['e_max_requests'], isterminal)

            # Get datacenters, store them in the db, returns the dc instances (db rows) correctly
            # added
            stepinfo("Requesting data-centers")
            # get dacatanters (might raise FailedDownload):
            datacenters_df, eidavalidator = \
                get_datacenters_df(session, dataws, advanced_settings['routing_service_url'],
                                   networks, stations, locations, channels, start, end, dbbufsize)

            stepinfo("Requesting stations and channels from %d %s", len(datacenters_df),
                     "data-center" if len(datacenters_df) == 1 else "data-centers")
            # get dacatanters (might raise FailedDownload):
            channels_df = get_channels_df(session, datacenters_df, eidavalidator,
                                          networks, stations, locations, channels, start, end,
                                          min_sample_rate, update_metadata,
                                          advanced_settings['max_thread_workers'],
                                          advanced_settings['s_timeout'],
                                          advanced_settings['download_blocksize'], dbbufsize,
                                          isterminal)
            # get channel id to mseed id dict and purge channels_df
            # the dict will be used to download the segments later, but we use it now to drop
            # unnecessary columns and save space (and time)
            chaid2mseedid = chaid2mseedid_dict(channels_df, drop_mseedid_columns=True)

            stepinfo("Selecting stations within search area from %d events", len(events_df))
            # merge vents and stations (might raise FailedDownload):
            segments_df = merge_events_stations(events_df, channels_df, search_radius['minmag'],
                                                search_radius['maxmag'],
                                                search_radius['minmag_radius'],
                                                search_radius['maxmag_radius'], tt_table,
                                                isterminal)
            # help gc by deleting the (only) refs to unused dataframes
            del events_df
            del channels_df

            if authorizer.token:
                stepinfo("Acquiring credentials from token in order to download restricted data")
            dc_dataselect_manager = DcDataselectManager(datacenters_df, authorizer, isterminal)

            stepinfo("%d segments found. Checking already downloaded segments", len(segments_df))
            # raises NothingToDownload
            segments_df, request_timebounds_need_update = \
                prepare_for_download(session, segments_df, dc_dataselect_manager,
                                     timespan, retry_seg_not_found,
                                     retry_url_err, retry_mseed_err, retry_client_err,
                                     retry_server_err, retry_timespan_err,
                                     retry_timespan_warn=False)

            # prepare_for_download raises a NothingToDownload if there is no data, so if we are
            # here segments_df is not empty
            stepinfo("Downloading %d segments %sand saving to db", len(segments_df),
                     '(open data only) ' if dc_dataselect_manager.opendataonly else '')
            # frees memory. Although maybe unecessary, let's do our best to free stuff cause the
            # next one is memory consuming:
            # https://stackoverflow.com/questions/30021923/how-to-delete-a-sqlalchemy-mapped-object-from-memory
            session.expunge_all()
            session.close()

            d_stats = download_save_segments(session, segments_df, dc_dataselect_manager,
                                             chaid2mseedid, download_id,
                                             request_timebounds_need_update,
                                             advanced_settings['max_thread_workers'],
                                             advanced_settings['w_timeout'],
                                             advanced_settings['download_blocksize'],
                                             dbbufsize,
                                             isterminal)
            del segments_df  # help gc?
            session.close()  # frees memory?
            logger.info("")
            logger.info(("** Segments download summary **\n"
                         "Number of segments per data-center (rows) and response "
                         "status (columns):\n%s") %
                        str(d_stats) or "Nothing to show")

    except NothingToDownload as ntdexc:
        # we are here if some function raised a NothingToDownload (e.g., in prepare_for_download
        # there is nothing according to current config). Print message as info, not that
        # inventory might be downloaded (see finally below)
        logger.info(str(ntdexc))
        # comment out: DO NOT RAISE:
        # raise
    except FailedDownload as dexc:
        # We are here if we raised a FailedDownload. Same behaviour as NothingToDownload,
        # except we log an error message, and we prevent downloading inventories by forcing
        # the flag to be false
        inventory = False
        logger.error(dexc)
        raise
    except:  # @IgnorePep8
        inventory = False
        raise
    finally:
        if inventory:
            # frees memory. Although maybe unecessary, let's do our best to free stuff cause the
            # next one might be memory consuming:
            # https://stackoverflow.com/questions/30021923/how-to-delete-a-sqlalchemy-mapped-object-from-memory
            session.expunge_all()
            session.close()

            # query station id, network station, datacenter_url
            # for those stations with empty inventory_xml
            # AND at least one segment non empty/null
            # Download inventories for those stations only
            sta_df = dbquery2df(query4inventorydownload(session, update_metadata))
            # stations = session.query(Station).filter(~withdata(Station.inventory_xml)).all()
            if sta_df.empty:
                stepinfo("Skipping: No station inventory to download")
            else:
                stepinfo("Downloading %d station inventories", len(sta_df))
                n_downloaded, n_empty, n_errors = \
                    save_inventories(session, sta_df,
                                     advanced_settings['max_thread_workers'],
                                     advanced_settings['i_timeout'],
                                     advanced_settings['download_blocksize'], dbbufsize,
                                     isterminal)
                logger.info(("** Station inventories download summary **\n"
                             "- downloaded     %7d \n"
                             "- discarded      %7d (empty response)\n"
                             "- not downloaded %7d (client/server errors)"),
                            n_downloaded, n_empty, n_errors)


def new_db_download(session, params=None):
    if params is None:
        params = {}
    # print local vars: use safe_dump to avoid python types. See:
    # http://stackoverflow.com/questions/1950306/pyyaml-dumping-without-tags
    download_inst = Download(config=tounicode(yaml.safe_dump(params,
                                                             default_flow_style=False)),
                             # log by default shows error. If everything works fine, we replace
                             # the content later
                             log=('N/A: either logger not configured, or '
                                  'an unexpected error interrupted the process'),
                             program_version=version())

    session.add(download_inst)
    session.commit()
    download_id = download_inst.id
    session.close()  # frees memory?
    return download_id
