'''
Created on Jul 20, 2016

@author: riccardo
'''
from StringIO import StringIO
import concurrent.futures
from click import progressbar
import logging
from obspy.core.inventory.inventory import read_inventory
from obspy.core.stream import read
from obspy.core.utcdatetime import UTCDateTime

from stream2segment.s2sio.db.pd_sql_utils import flush, commit
from stream2segment.s2sio.dataseries import dumps, dumps_inv, loads_inv
from stream2segment.async import url_read, read_async
from stream2segment.analysis.mseeds import remove_response, get_gaps, amp_ratio, bandpass, cumsum,\
    cumtimes, fft, maxabs, simulate_wa, get_multievent, snr  # ,dfreq
from stream2segment.s2sio.db import models
from stream2segment.download.utils import get_query
from sqlalchemy.exc import SQLAlchemyError

import urllib2
import httplib
import socket
from collections import defaultdict as defdict
from itertools import cycle

logger = logging.getLogger(__name__)
# from stream2segment.analysis import snr, mseeds
# from sqlalchemy.exc import IntegrityError
# from sqlalchemy import func
# from sqlalchemy.engine import create_engine
# from stream2segment.s2sio.db.models import Base
# from sqlalchemy.orm.session import sessionmaker

# from obspy.core.trace import Trace

# def process_single(session, segments_model_instance, run_id,
#                    if_exists='update',
#                    logger=None,
#                    station_inventories={},
#                    amp_ratio_threshold=0.8, a_time_delay=0,
#                    bandpass_freq_max=20, bandpass_max_nyquist_ratio=0.9, bandpass_corners=2,
#                    remove_response_output='ACC', remove_response_water_level=60,
#                    taper_max_percentage=0.05, savewindow_delta_in_sec=30,
#                    snr_fixedwindow_in_sec=60, multievent_threshold1_percent=0.85,
#                    multievent_threshold1_duration_sec=10,
#                    multievent_threshold2_percent=0.05, **kwargs):


def process_all(session, segments_model_instances, run_id,
                **processing_args):
    """
        Processes all segments_model_instances. FIXME: write detailed doc
    """
    logger.info("Processing %d segments", len(segments_model_instances))
    ret = []

    # db operations complain if the same object instance is not only created but even 
    # its properties are accessed within different threads
    # (regardless of scoped_session or session). Thus we had to implement all db transaction
    # on the child processes. BUT:
    # sqlite has problems with concurrency (see http://www.sqlite.org/whentouse.html,
    # http://docs.sqlalchemy.org/en/rel_0_9/dialects/sqlite.html#database-locking-behavior-concurrency
    # http://stackoverflow.com/questions/13895176/sqlalchemy-and-sqlite-database-is-locked
    # Thus we need to perform calculations in separate
    # processes and write *ALL* at the end in the main process thread

    def _process_(segment, station_inventory, **kwargs):
        pro = models.Processing(run_id=run_id)
        pro.segment = session.query(models.Segment).filter(models.Segment.id == seg_id).first()
        pro = process(pro, station_inventory, **kwargs)
        if pro is None:
            return False
        session.add(pro)
        try:
            session.commit()
        except SQLAlchemyError as exc:
            session.rollback()
            raise
        return True

    sta2segs = defdict(lambda: [])
    for seg in segments_model_instances:
        sta2segs[seg.channel.station_id].append(seg)

    with progressbar(length=len(segments_model_instances)) as bar:
        seg.toattrdict()
        for sta_id, segments in sta2segs.iteritems():
            inventory = None
            try:
                inventory = get_inventory(segments[0], timeout=30)
            except SQLAlchemyError as exc:
                logger.warning("Error while saving inventory (station id=%s), "
                               "%d segment will not be processed: %s",
                               str(sta_id), len(segments), str(exc))
            except (urllib2.HTTPError, urllib2.URLError, httplib.HTTPException, socket.error) as _:
                logger.warning("Error while downloading inventory (station id=%s), "
                               "%d segment will not be processed: %s URL: %s",
                               str(sta_id), len(segments), str(_), get_inventory_query(segments[0]))
            except Exception as exc:  # pylint:disable=broad-except
                logger.warning("Error while creating inventory (station id=%s), "
                               "%d segment will not be processed: %s",
                               str(sta_id), len(segments), str(exc))

            if inventory is None:
                bar.update(len(segments))
                continue
                # pass

            sta_attdic = segments[0].channel.station.toattrdict()  # instantiate once
            # We can use a with statement to ensure threads are cleaned up promptly
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(process, models.Processing().toattrdict(),
                                           seg.toattrdict(),
                                           seg.channel.toattrdict(),
                                           sta_attdic,
                                           seg.event.toattrdict(),
                                           seg.datacenter.toattrdict(),
                                           inventory,
                                           **processing_args): seg.id for seg in segments}
                for future in concurrent.futures.as_completed(futures):
                    seg_id = futures[future]
                    try:
                        bar.update(1)
                        pro_ = future.result()
                        pro_.segment_id = seg_id
                        ret.append(pro_)
                    except Exception as exc:  # pylint:disable=broad-except
                        logger.warning("Unable to process segment (id=%s): %s", seg_id, str(exc))
#             for seg in segments:
#                 bar.update(1)
#                 pro = models.Processing(run_id=run_id)
#                 pro.segment = seg
#                 session.flush()
#                 try:
#                     pro = process(pro, station_inventory=inventory, **processing_args)
#                     if pro and commit(session, on_exc=lambda exc: warn(seg, exc)):
#                         ret.append(pro)
#                 except Exception as exc:
#                     logger.warning("Unable to process segment (id=%s): %s", seg.id, str(exc))

    logger.info("")
    logger.info("Writing %d processed segments to database", len(ret))
    success_num = 0
    for pro_ in ret:
        propro = models.Processing(**pro_)
        session.add(propro)
        if commit(session, on_exc=lambda exc: logger.warning("Unable to write to db (seg_id:%s) %s",
                                                             str(pro_.segment_id), str(exc))):
            success_num += 1
    logger.info("%d segments successfully processed", success_num)
    return ret


def get_inventory_query(segment):
    station = segment.channel.station
    datacenter = segment.datacenter
    return get_query(datacenter.station_query_url, station=station.station, network=station.network,
                     level='response')


def get_inventory(segment, session=None, **kwargs):
    """raises tons of exceptions (see process_all). FIXME: write doc"""
    data = segment.channel.station.inventory_xml
    if not data:
        query_url = get_inventory_query(segment)
        data = url_read(query_url, **kwargs)
        if session and data:
            segment.channel.station.inventory_xml = dumps_inv(data)
            session.commit()
        elif not data:
            raise ValueError("No data from server")
    return loads_inv(data)


def warn(segment, exception_or_msg):
    """ convenience function for logging warnings during processing"""
    logger.warning("while processing segment.id='%s': %s", str(segment.id), str(exception_or_msg))


def dtime(utcdatetime):
    """converts UtcDateTime to datetime, returns None if arg is None"""
    return None if utcdatetime is None else utcdatetime.datetime


# dict: { 'arrival_time_delay': 0}

# def process(session, segments_model_instance, run_id, overwrite_all=False,
#             station_inventory=None,
#             amp_ratio_threshold=0.8, arrival_time_delay=0,
#             bandpass_freq_max=20, bandpass_max_nyquist_ratio=0.9, bandpass_corners=2,
#             remove_response_output='ACC', remove_response_water_level=60,
#             taper_max_percentage=0.05, savewindow_delta=30,
#             snr_window_length=60, multievent_threshold1=0.85,
#             multievent_threshold1_duration=10,
#             multievent_threshold2=0.05, **kwargs):

def process(pro,
            seg,
            cha,
            sta,
            evt,
            dcen,
            station_inventory,
            amp_ratio_threshold,
            arrival_time_delay,
            savewindow_delta,
            taper_max_percentage,
            snr_window_length,
            remove_response_output,
            remove_response_water_level,
            bandpass_corners,
            bandpass_freq_max,
            bandpass_max_nyquist_ratio,
            multi_event_threshold1,
            multi_event_threshold1_duration,
            multi_event_threshold2,
            coda_window_length,
            coda_subwindow_length,
            coda_subwindow_overlap,
            coda_subwindow_amplitude_threshold,
            **kwargs):
    """
        Processes a single segment.
        This function is supposed to perform calculation and set the attributes of the `pro`
        object (it does not need to return it). These attributes are set in the `models` module
        and the value types should match (meaning an attribute reflecting an integer database
        column should be set with integer values only).
        Exceptions are handled externally and should be consulted by looking at the log messages
        stored in the output database (whose address is given in the `config.yaml` file)
        :param pro: a dict-like object (whose keys can be accessed also as attributes, so
        `pro['whatever]=6 is the same as `pro.whatever=4`) which has to be populated with values
        resulting from processing the given segment.
        :param seg: the segment (i.e., time series data) originating the processing. Its actual
        data can be accessed via `loads(seg.data)` which returns a Stream object. Additional
        fields are accessible via attributes and their names can be inspected via `seg.keys()`
        FIXME: write detailed doc!
        parameters and arguments must not conflict with imported functions
    """
    # rename args (see config.yaml 'processing' section):

#     multievent_threshold1 = multi_event['threshold1']
#     multievent_threshold2 = multi_event['threshold2']
#     multievent_threshold1_duration = multi_event['threshold1_duration']

#     coda_subw_length = coda['subwindow_length']
#     coda_subw_ovlap = coda['subwindow_overlap']
#     coda_win_length = coda['window_length']
#     coda_subw_amp_thr = coda['subwindow_amplitude_threshold']

#    snr_window_length = snr['window_length']

#     remove_response_output = remove_response['output']
#     remove_response_water_level = remove_response['water_level']

#     bandpass_corners = bandpass['corners']
#     bandpass_freq_max = bandpass['freq_max']
#     bandpass_max_nyquist_ratio = bandpass['max_nyquist_ratio']

#    seg = pro.segment

#     if overwrite_all:
#         for pro in seg.processings:
#             session.delete(pro)
#         if not flush(session, on_exc=lambda exc: warn(seg, exc)):
#             return None
#     elif seg.processings:
#         return seg.processings[0]

#     pro = models.Processing(segment_id=seg.id, run_id=run_id)

    # convert to UTCDateTime for operations later:
    a_time = UTCDateTime(seg.arrival_time) + arrival_time_delay

    mseed = read(StringIO(seg.data))

    if get_gaps(mseed):
        pro.has_gaps = True
    else:
        if len(mseed) != 1:
            raise ValueError("Mseed has more than one Trace")

        pro.has_gaps = False
        # work on the trace now. All functions will return Traces or scalars, which is better
        # so we can write them to database more easily
        mseed = mseed[0]

        ampratio = amp_ratio(mseed)
        pro.amplitude_ratio = ampratio
        if ampratio >= amp_ratio_threshold:
            pro.is_saturated = True
        else:
            pro.is_saturated = False
            mseed = bandpass(mseed, evt.magnitude, freq_max=bandpass_freq_max,
                             max_nyquist_ratio=bandpass_max_nyquist_ratio, corners=bandpass_corners)

            inv_obj = station_inventory

            mseed_acc = remove_response(mseed, inv_obj, output='ACC',
                                        water_level=remove_response_water_level)
            mseed_vel = remove_response(mseed, inv_obj, output='VEL',
                                        water_level=remove_response_water_level)
            mseed_disp = remove_response(mseed, inv_obj, output='DISP',
                                         water_level=remove_response_water_level)
            mseed_wa = simulate_wa(mseed_disp)

            mseed_rem_resp = mseed_disp if remove_response_output == 'DISP' else \
                (mseed_vel if remove_response_output == 'VEL' else mseed_acc)

            mseed_cum = cumsum(mseed_rem_resp)

            cum_times = cumtimes(mseed_cum, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)

            t05, t10, t90, t95 = cum_times[0], cum_times[1], cum_times[-2], \
                cum_times[-1]

#                     mseed_acc_atime_95 = mseed_acc.slice(a_time, t95)
#                     mseed_vel_atime_t95 = mseed_vel.slice(a_time, t95)
#                     mseed_wa_atime_t95 = mseed_wa.slice(a_time, t95)

            t_PGA, PGA = maxabs(mseed_acc, a_time, t95)
            t_PGV, PGV = maxabs(mseed_vel, a_time, t95)
            t_PWA, PWA = maxabs(mseed_wa, a_time, t95)

            # instantiate the trace below cause it's used also later ...
            mseed_rem_resp_t05_t95 = mseed_rem_resp.slice(t05, t95)

            fft_rem_resp_s = fft(mseed_rem_resp_t05_t95, taper_max_percentage=taper_max_percentage)
            fft_rem_resp_n = fft(mseed_rem_resp, fixed_time=a_time,
                                 window_in_sec=t05-t95,  # negative float (in seconds)
                                 taper_max_percentage=taper_max_percentage)
            # calculate the *real* start time
            snr_rem_resp_t05_t95 = snr(fft_rem_resp_s, fft_rem_resp_n, signals_form='fft',
                                       in_db=False)

            fft_rem_resp_s2 = fft(mseed_rem_resp, t10, t90-t10,
                                  taper_max_percentage=taper_max_percentage)
            fft_rem_resp_n2 = fft(mseed_rem_resp, a_time, t10-t90,
                                  taper_max_percentage=taper_max_percentage)
            snr_rem_resp_t10_t90 = snr(fft_rem_resp_s2, fft_rem_resp_n2, signals_form='fft',
                                       in_db=False)

            fft_rem_resp_s3 = fft(mseed_rem_resp, a_time, snr_window_length,
                                  taper_max_percentage=taper_max_percentage)
            fft_rem_resp_n3 = fft(mseed_rem_resp, a_time, -snr_window_length,
                                  taper_max_percentage=taper_max_percentage)
            snr_rem_resp_fixed_window = snr(fft_rem_resp_s3, fft_rem_resp_n3,
                                            signals_form='fft', in_db=False)

            gme = get_multievent  # rename func just to avoid line below is not too wide
            double_evt = \
                gme(mseed_cum, t05, t95,
                    threshold_inside_tmin_tmax_percent=multi_event_threshold1,
                    threshold_inside_tmin_tmax_sec=multi_event_threshold1_duration,
                    threshold_after_tmax_percent=multi_event_threshold2)

            mseed_rem_resp_savewindow = mseed_rem_resp.slice(a_time-savewindow_delta,
                                                             t95+savewindow_delta).\
                taper(max_percentage=taper_max_percentage)

            wa_savewindow = mseed_wa.slice(a_time-savewindow_delta,
                                           t95+savewindow_delta).\
                taper(max_percentage=taper_max_percentage)

            # deltafreq = dfreq(mseed_rem_resp_t05_t95)

            # write stuff now to instance:
            pro.mseed_rem_resp_savewindow = dumps(mseed_rem_resp_savewindow)
            pro.fft_rem_resp_t05_t95 = dumps(fft_rem_resp_s)
            pro.fft_rem_resp_until_atime = dumps(fft_rem_resp_n)
            pro.wood_anderson_savewindow = dumps(wa_savewindow)
            pro.cum_rem_resp = dumps(mseed_cum)
            pro.pga_atime_t95 = PGA
            pro.pgv_atime_t95 = PGV
            pro.pwa_atime_t95 = PWA
            pro.t_pga_atime_t95 = dtime(t_PGA)
            pro.t_pgv_atime_t95 = dtime(t_PGV)
            pro.t_pwa_atime_t95 = dtime(t_PWA)
            pro.cum_t05 = dtime(t05)
            pro.cum_t10 = dtime(t10)
            pro.cum_t25 = dtime(cum_times[2])
            pro.cum_t50 = dtime(cum_times[3])
            pro.cum_t75 = dtime(cum_times[4])
            pro.cum_t90 = dtime(t90)
            pro.cum_t95 = dtime(t95)
            pro.snr_rem_resp_fixedwindow = snr_rem_resp_fixed_window
            pro.snr_rem_resp_t05_t95 = snr_rem_resp_t05_t95
            pro.snr_rem_resp_t10_t90 = snr_rem_resp_t10_t90
            # pro.amplitude_ratio = Column(Float)
            # pro.is_saturated = Column(Boolean)
            # pro.has_gaps = Column(Boolean)
            pro.double_event_result = double_evt[0]
            pro.secondary_event_time = dtime(double_evt[1])

            # WITH JESSIE IMPLEMENT CODA ANALYSIS:
            # pro.coda_tmax = Column(DateTime)
            # pro.coda_length_sec = Column(Float)
    return pro


# def get_inventory(segment_instance, session):
#     sta = segment_instance.channel.station
#     inv_xml = sta.inventory_xml
#     if not inv_xml:
#         dcen = segment_instance.datacenter
#         if dcen:
#             inventory_url = query(dcen.station_query_url, station=sta.station, network=sta.network,
#                                   level='response')
# #             inventory_url = dcen.station_query_url + ("?station=%s&network=%s&level=response" %
# #                                                       (sta.station, sta.network))
#         try:
#             xmlbytes = url_read(inventory_url)
#         except (IOError, ValueError, TypeError) as exc:
#             warn(segment_instance, "while reading inventory for station.id='%s': %s"
#                  % (str(sta.id), str(exc)))
#             return None
# 
# #        sta.inventory_xml = xmlbytes
# 
# ##         session.query(models.Station).filter(models.Station.id == sta.id).\
# ##             update({"inventory_xml": xmlbytes})
# 
# #         if not flush(session,
# #                      on_exc=lambda exc: warn(logger, segment_instance,
# #                                              "while reading inventory for station.id='%s': %s"
# #                                              % (str(sta.id), str(exc)))):
# #             return None
# 
#     s = StringIO(xmlbytes)
#     s.seek(0)
#     inv = read_inventory(s, format="STATIONXML")
#     s.close()
#     return inv
