'''
Created on Jul 20, 2016

@author: riccardo
'''
from stream2segment.utils import url_read
from obspy.core.inventory.inventory import read_inventory
from stream2segment.s2sio.db.pd_sql_utils import flush
from obspy.core.stream import read
from stream2segment.analysis.mseeds import remove_response, get_gaps, amp_ratio, bandpass, cumsum,\
    cumtimes, fft, maxabs, simulate_wa, get_multievent, dumps, dfreq
from stream2segment.analysis import snr, mseeds
from stream2segment.s2sio.db import models
from obspy.core.utcdatetime import UTCDateTime
from sqlalchemy.exc import IntegrityError
from StringIO import StringIO
from sqlalchemy import func

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

def process_all(session, segments_model_instances, run_id, if_exist='update',
                logger=None, progresslistener=None,
                **processing_args):
#                 amp_ratio_threshold=0.8, a_time_delay=0,
#                 bandpass_freq_max=20, bandpass_max_nyquist_ratio=0.9, bandpass_corners=2,
#                 remove_response_output='ACC', remove_response_water_level=60,
#                 taper_max_percentage=0.05, savewindow_delta_in_sec=30,
#                 snr_fixedwindow_in_sec=60, multievent_threshold1_percent=0.85,
#                 multievent_threshold1_duration_sec=10,
#                 multievent_threshold2_percent=0.05, **kwargs):

    station_inventories = {}  # cache inventories
    for _, seg in enumerate(segments_model_instances):
        pro = process(session, seg, run_id,
                      if_exists=if_exist,
                      logger=logger,
                      station_inventories=station_inventories,
                      **processing_args)
        if pro:
            yield pro


def process(session, segments_model_instance, run_id,
            if_exists='update',
            logger=None,
            station_inventories={},
            amp_ratio_threshold=0.8, a_time_delay=0,
            bandpass_freq_max=20, bandpass_max_nyquist_ratio=0.9, bandpass_corners=2,
            remove_response_output='ACC', remove_response_water_level=60,
            taper_max_percentage=0.05, savewindow_delta_in_sec=30,
            snr_fixedwindow_in_sec=60, multievent_threshold1_percent=0.85,
            multievent_threshold1_duration_sec=10,
            multievent_threshold2_percent=0.05, **kwargs):

    seg = segments_model_instance
    needs_add = False
    if if_exists == 'overwrite':
        pro_obj = session.query(models.Processing).filter(models.Processing.segment_id == seg.id).\
                    delete()
        flush(session)
        needs_add = True
    else:
        pro_objs = session.query(models.Processing).filter(models.Processing.segment_id == seg.id).\
                                                          all()

        if len(pro_obj):
            # take the last. Do an inline calculation although there might be smarter db queries
            # (we are in a hurry!)
            runs = session.query(models.Run).filter(models.Run.id == seg.run_id).all()
            runs.sort(key=lambda row: row.run_time)
            last_run_id = runs[0].id
            for pro_ in pro_objs:
                if pro_.run_id == last_run_id:
                    pro = pro_
                    break
            if if_exists != 'update':
                return pro
        else:
            needs_add = True

    if needs_add:
        pro = models.Processing(segment_id=seg.id, run_id=run_id)

    # convert to UTCDateTime for operations later:
    a_time = UTCDateTime(seg.arrival_time) + a_time_delay

    mseed = read(seg.data)

    if get_gaps(mseed):
        pro.has_gaps = True
    else:
        mseed = read(seg.data)[0]
        if len(mseed) != 1:  # FIXME: better handling
            if logger:
                logger.warning("Mseed has more than one Trace")
            pro = None
        else:
            # work on the trace now. All functions will return Traces or scalars, which is better
            # so we can write them to database more easily
            mseed = mseed[0]

            ampratio = amp_ratio(mseed)
            pro.amplitude_ratio = ampratio
            if ampratio >= amp_ratio_threshold:
                pro.is_saturated = True
            else:
                mseed = bandpass(mseed, seg.event.magnitude, freq_max=bandpass_freq_max,
                                 max_nyquist_ratio=bandpass_max_nyquist_ratio,
                                 corners=bandpass_corners)

                sta = seg.channel.station
                inv_obj = station_inventories.get(sta.id, None)
                if inv_obj is None:
                    if logger:
                        logger.debug("Reading inventory for station (id='%s')" % str(sta.id))
                    inv_obj = get_inventory(seg, session, logger)
                    station_inventories[sta.id] = inv_obj

                if inv_obj is None:
                    pro = None
                    if logger:
                        logger.warning("No inventory found or loaded for segment (id='%s')" %
                                       str(seg.id))
                else:
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

                    t05, t10, t90, t95 = cum_times[0], cum_times[1], cum_times[-2], cum_times[-1]

                    mseed_acc_atime_95 = mseed_acc.slice[a_time, t95]
                    mseed_vel_atime_t95 = mseed_vel.slice[a_time, t95]
                    mseed_wa_atime_t95 = mseed_wa.slice[a_time, t95]

                    t_PGA, PGA = maxabs(mseed_acc_atime_95)
                    t_PGV, PGV = maxabs(mseed_vel_atime_t95)
                    t_PWA, PWA = maxabs(mseed_wa_atime_t95)

                    # instantiate the trace below cause it's used also later ...
                    mseed_rem_resp_t05_t95 = mseed_rem_resp.traces[0].slice[t05, t95]

                    fft_rem_resp_s = fft(mseed_rem_resp_t05_t95,
                                         taper_max_percentage=taper_max_percentage)
                    fft_rem_resp_n = fft(mseed_rem_resp.traces[0], fixed_time=a_time,
                                         window_in_sec=t05-t95,  # negative float (in seconds)
                                         taper_max_percentage=taper_max_percentage)
                    snr_rem_resp_t05_t95 = snr(fft_rem_resp_s, fft_rem_resp_n,
                                               signals_form='fft', in_db=False)

                    fft_rem_resp_s2 = fft(mseed_rem_resp.traces[0], t10, t90-t10,
                                          taper_max_percentage=taper_max_percentage)
                    fft_rem_resp_n2 = fft(mseed_rem_resp.traces[0], a_time, t10-t90,
                                          taper_max_percentage=taper_max_percentage)
                    snr_rem_resp_t10_t90 = snr(fft_rem_resp_s2, fft_rem_resp_n2,
                                               signals_form='fft', in_db=False)

                    fft_rem_resp_s3 = fft(mseed_rem_resp.traces[0], a_time, snr_fixedwindow_in_sec,
                                          taper_max_percentage=taper_max_percentage)
                    fft_rem_resp_n3 = fft(mseed_rem_resp.traces[0], a_time, -snr_fixedwindow_in_sec,
                                          taper_max_percentage=taper_max_percentage)
                    snr_rem_resp_fixed_window = snr(fft_rem_resp_s3, fft_rem_resp_n3,
                                                    signals_form='fft', in_db=False)

                    gme = get_multievent  # rename func just to avoid line below is not too wide
                    double_evt = \
                        gme(mseed_cum, t05, t95,
                            threshold_inside_tmin_tmax_percent=multievent_threshold1_percent,
                            threshold_inside_tmin_tmax_sec=multievent_threshold1_duration_sec,
                            threshold_after_tmax_percent=multievent_threshold2_percent)

                    mseed_rem_resp_savewindow = mseed_rem_resp.slice[a_time-savewindow_delta_in_sec,
                                                                     t95+savewindow_delta_in_sec].\
                        taper(max_percentage=taper_max_percentage)

                    wa_savewindow = mseed_wa.slice[a_time-savewindow_delta_in_sec,
                                                   t95+savewindow_delta_in_sec].\
                        taper(max_percentage=taper_max_percentage)

                    deltafreq = dfreq(mseed_rem_resp_t05_t95[0])

                    # write stuff now to instance:
                    pro.mseed_rem_resp_savewindow = dumps(mseed_rem_resp_savewindow,
                                                          mseeds._IO_FORMAT_STREAM)
                    pro.fft_rem_resp_t05_t95 = dumps(fft_rem_resp_s,
                                                     mseeds._IO_FORMAT_FFT, dx=deltafreq)
                    pro.fft_rem_resp_until_atime = dumps(fft_rem_resp_n,
                                                         mseeds._IO_FORMAT_FFT, dx=deltafreq)
                    pro.wood_anderson_savewindow = dumps(wa_savewindow,
                                                         mseeds._IO_FORMAT_STREAM)
                    pro.cumulative = dumps(mseed_cum,
                                           mseeds._IO_FORMAT_STREAM)
                    pro.pga_atime_t95 = PGA
                    pro.pgv_atime_t95 = PGV
                    pro.pwa_atime_t95 = PWA
                    pro.t_pga_atime_t95 = t_PGA.datetime
                    pro.t_pgv_atime_t95 = t_PGV.datetime
                    pro.t_pwa_atime_t95 = t_PWA.datetime
                    pro.cum_t05 = t05.datetime
                    pro.cum_t10 = t10.datetime
                    pro.cum_t25 = cum_times[2].datetime
                    pro.cum_t50 = cum_times[3].datetime
                    pro.cum_t75 = cum_times[4].datetime
                    pro.cum_t90 = t90.datetime
                    pro.cum_t95 = t95.datetime
                    pro.snr_rem_resp_fixedwindow = snr_rem_resp_fixed_window
                    pro.snr_rem_resp_t05_t95 = snr_rem_resp_t05_t95
                    pro.snr_rem_resp_t10_t90 = snr_rem_resp_t10_t90
                    # pro.amplitude_ratio = Column(Float)
                    # pro.is_saturated = Column(Boolean)
                    # pro.has_gaps = Column(Boolean)
                    pro.double_event_result = double_evt[0]
                    pro.secondary_event_time = double_evt[1]

                    # WITH JESSIE IMPLEMENT CODA ANALYSIS:
                    # pro.coda_tmax = Column(DateTime)
                    # pro.coda_length_sec = Column(Float)

    if pro is None:
        return None
    elif needs_add:
        seg.processings.append(pro)

    flush(session)


def get_inventory(segment_instance, session, logger=None):
    sta = segment_instance.channel.station
    inv_xml = sta.inventory_xml
    if not inv_xml:
        dcen = session.query(models.DataCenter).\
                            filter(models.DataCenter.id == segment_instance.datacenter_id).first()

        if dcen:
            inventory_url = dcen.station_query_url + ("station=%s&network=%s&level=response" %
                                                      (sta.station, sta.network))
        xmlbytes = url_read(inventory_url)
        sta.inventory_xml = xmlbytes
        if not flush(session):
            if logger:
                logger.warning("Unable to read inventory at %s" % inventory_url)
            return None

    s = StringIO(xmlbytes)
    s.seek(0)
    inv = read_inventory(s, format="STATIONXML")
    s.close()
    return inv
