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
from stream2segment.analysis import snr
from stream2segment.s2sio.db import models
from obspy.core.utcdatetime import UTCDateTime
from sqlalchemy.exc import IntegrityError


def process(session, run_instance, segments_model_instances, logger, progresslistener,
            amp_ratio_threshold=0.8, a_time_delay=0,
            bandpass_freq_max=20, bandpass_max_nyquist_ratio=0.9, bandpass_corners=2,
            remove_response_output='ACC', remove_response_water_level=60,
            taper_max_percentage=0.05, savewindow_delta_in_sec=30,
            snr_fixedwindow_in_sec=60, multievent_threshold1_percent=0.85,
            multievent_threshold1_duration_sec=10,
            multievent_threshold2_percent=0.05, **kwargs):

    for i, seg in enumerate(segments_model_instances):
        pro = models.Processing(segment_id=seg.id, run_id=run_instance.id)

        a_time = UTCDateTime(seg.arrival_time) + a_time_delay  # convert to UTCDateTime for operations later
        if progresslistener:
            progresslistener(i+1)

        mseed = read(seg.data)

        if get_gaps(mseed):
            pro.has_gaps = True
            session.add(pro)
            flush(session)
            continue

        if len(mseed) != 1:  # FIXME: better handling
            continue

        ampratio = amp_ratio(mseed)
        pro.amplitude_ratio = ampratio
        if ampratio >= amp_ratio_threshold:
            pro.is_saturated = True
            session.add(pro)
            flush(session)
            continue

        mseed = bandpass(mseed, seg.event.magnitude, freq_max=bandpass_freq_max,
                         max_nyquist_ratio=bandpass_max_nyquist_ratio, corners=bandpass_corners)

        sta = seg.channel.station
        inv_xml = sta.inventory_xml
        if not inv_xml:
            inventory_url = ""  # FIXME::: .. HOW TO GET IT ?
            xmlbytes = url_read(inventory_url)
            sta.inventory_xml = xmlbytes
            if not flush(session):
                if logger:
                    logger.debug("Unable to read inventory at %s" % inventory_url)
                continue
        inventory = read_inventory(inv_xml)

        mseed_acc = remove_response(mseed, inventory, output='ACC',
                                    water_level=remove_response_water_level)
        mseed_vel = remove_response(mseed, inventory, output='VEL',
                                    water_level=remove_response_water_level)
        mseed_disp = remove_response(mseed, inventory, output='DISP',
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

        mseed_rem_resp_t05_t95 = mseed_rem_resp.traces[0].slice[t05, t95]  # used also later ...
        fft_rem_resp_s = fft(mseed_rem_resp_t05_t95, taper_max_percentage=taper_max_percentage)
        fft_rem_resp_n = fft(mseed_rem_resp.traces[0], fixed_time=a_time,
                             window_in_sec=t05-t95,  # negative float (in seconds)
                             taper_max_percentage=taper_max_percentage)
        snr_rem_resp_t05_t95 = snr(fft_rem_resp_s, fft_rem_resp_n, signals_form='fft',
                                   in_db=False)

        fft_rem_resp_s2 = fft(mseed_rem_resp.traces[0], t10, t90-t10, taper_max_percentage=
                              taper_max_percentage)
        fft_rem_resp_n2 = fft(mseed_rem_resp.traces[0], a_time, t10-t90, taper_max_percentage=
                              taper_max_percentage)
        snr_rem_resp_t10_t90 = snr(fft_rem_resp_s2, fft_rem_resp_n2, signals_form='fft',
                                   in_db=False)

        fft_rem_resp_s3 = fft(mseed_rem_resp.traces[0], a_time, snr_fixedwindow_in_sec,
                              taper_max_percentage=taper_max_percentage)
        fft_rem_resp_n3 = fft(mseed_rem_resp.traces[0], a_time, -snr_fixedwindow_in_sec,
                              taper_max_percentage=taper_max_percentage)
        snr_rem_resp_fixed_window = snr(fft_rem_resp_s3, fft_rem_resp_n3, signals_form='fft',
                                        in_db=False)

        double_evt = \
            get_multievent(mseed_cum, t05, t95,
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

        seg.mseed_rem_resp_savewindow = dumps(mseed_rem_resp_savewindow)
        seg.fft_rem_resp_t05_t95 = dumps(fft_rem_resp_s, 'fft', dx=deltafreq)
        seg.fft_rem_resp_until_atime = dumps(fft_rem_resp_n, 'fft', dx=deltafreq)
        seg.wood_anderson_savewindow = dumps(wa_savewindow)
        seg.cumulative = dumps(mseed_cum)
        seg.pga_atime_t95 = PGA
        seg.pgv_atime_t95 = PGV
        seg.pwa_atime_t95 = PWA
        seg.t_pga_atime_t95 = t_PGA.datetime
        seg.t_pgv_atime_t95 = t_PGV.datetime
        seg.t_pwa_atime_t95 = t_PWA.datetime
        seg.cum_t05 = t05.datetime
        seg.cum_t10 = t10.datetime
        seg.cum_t25 = cum_times[2].datetime
        seg.cum_t50 = cum_times[3].datetime
        seg.cum_t75 = cum_times[4].datetime
        seg.cum_t90 = t90.datetime
        seg.cum_t95 = t95.datetime
        seg.snr_rem_resp_fixedwindow = snr_rem_resp_fixed_window
        seg.snr_rem_resp_t05_t95 = snr_rem_resp_t05_t95
        seg.snr_rem_resp_t10_t90 = snr_rem_resp_t10_t90
        # seg.amplitude_ratio = Column(Float)
        # seg.is_saturated = Column(Boolean)
        # seg.has_gaps = Column(Boolean)
        seg.double_event_result = double_evt[0]
        seg.secondary_event_time = double_evt[1]

        # WITH JESSIE IMPLEMENT CODA ANALYSIS: 
        # seg.coda_tmax = Column(DateTime)
        # seg.coda_length_sec = Column(Float)

        flush(session)

    try:
        session.commit()
    except IntegrityError as ierr:
        logger.error(str(ierr))
