'''
Created on Jul 20, 2016

@author: riccardo
'''
import sys
import os
import urllib2
import httplib
import socket
from contextlib import contextmanager
from collections import defaultdict as defdict
import warnings
from StringIO import StringIO
import concurrent.futures
import logging
from obspy.core.inventory.inventory import read_inventory
from obspy.core.stream import read
from obspy.core.utcdatetime import UTCDateTime
from stream2segment.io.db.pd_sql_utils import flush, commit
from stream2segment.io.utils import dumps, dumps_inv, loads_inv, dumps_time
from stream2segment.utils.url import url_read
from stream2segment.analysis.mseeds import remove_response, get_gaps, amp_ratio, bandpass, cumsum,\
    cumtimes, fft, maxabs, simulate_wa, get_multievent, snr  # ,dfreq
from stream2segment.io.db import models
from stream2segment.download.utils import get_query, get_inventory_query
from sqlalchemy.exc import SQLAlchemyError
from stream2segment.utils import msgs, get_progressbar

logger = logging.getLogger(__name__)


# TWO UTILITIES REDIRECTING STANDARD ERROR/OUTPUT FROM EXTERNAL PROGRAM
# The first (used here) restores the old stderr/ stdout
# http://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python
# the second is easier to understand but does not restore old std/err stdout
# Added comments from
# http://stackoverflow.com/questions/8804893/redirect-stdout-from-python-for-c-calls

@contextmanager
def redirected(what='err', to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = (sys.stdout if what == 'out' else sys.stderr).fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_(to):
        if what == 'out':
            sys.stdout.close()  # + implicit flush()
        else:
            sys.stderr.close()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        if what == 'out':
            sys.stdout = os.fdopen(fd, 'w')  # Python writes to fd
        else:
            sys.stderr = os.fdopen(fd, 'w')

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file_:
            _redirect_(to=file_)
        try:
            yield  # allow code to be run with the redirected stdout/stderr
        finally:
            # restore stdout/ stderr. buffering and flags such as CLOEXEC may be different
            _redirect_(to=old_stdout)


def redirect_external_out(fileno=2):
    out = sys.stdout if fileno == 1 else \
        sys.stdin if fileno == 0 else sys.stderr
    # print "Redirecting stderr"
    out.flush()  # <--- important when redirecting to files
    # Duplicate stdout/stderr/stdin (file descriptor 1)
    # to a different file descriptor number
    newstdout = os.dup(fileno)
    # /dev/null is used just to discard what is being printed
    devnull = os.open('/dev/null', os.O_WRONLY)
    # Duplicate the file descriptor for /dev/null
    # and overwrite the value for stdout (file descriptor fileno=1),
    # stdin (fileno=0) or stderr (fileno=2)
    os.dup2(devnull, fileno)
    # Close devnull after duplication (no longer needed)
    os.close(devnull)
    # Use the original stdout/stderr/stdin to still be able
    # to print to stdout/stderr/stdin within python
    sys.stderr = os.fdopen(newstdout, 'w')


def main(session, segments_model_instances, run_id, isterminal=False, **processing_args):
    # suppress obspy warnings. Doing process-wise is more feasible FIXME: do it?
    warnings.filterwarnings("default")  # https://docs.python.org/2/library/warnings.html#the-warnings-filter @IgnorePep8

    s = StringIO()
    logger_handler = logging.StreamHandler(s)
    logger_handler.setLevel(logging.WARNING)

    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")

    warnings_logger.addHandler(logger_handler)

    with redirected('err'):
        progressbar = get_progressbar(isterminal)
        with progressbar(length=len(segments_model_instances)) as bar:
            process_all(session, segments_model_instances, run_id, bar.update,
                        **processing_args)

    captured_warnings = s.getvalue()
    if captured_warnings:
        logger.info("(external warnings captured, please see log for details)")
        logger.info("")
        logger.warning("Captured external warnings:")
        logger.warning("%s", captured_warnings)
        logger.warning("(only the first occurrence of an external warning for each location where "
                       "the warning is issued is reported. Displaying the segment id which "
                       "originated these warnings would require too much effort and performance "
                       "issues compared to the benefits: in most cases, the process "
                       "completed successfully, and if you want to check the correctness of the "
                       "data please check the database results)")

    logging.captureWarnings(True)  # form the docs the redirection of warnings to the logging
    # system will stop, and warnings will be redirected to their original destinations
    # (i.e. those in effect before captureWarnings(True) was called).

    s.close()  # maybe not really necessary


def process_all(session, segments_model_instances, run_id,
                notify_progress_func=lambda *a, **v: None, **processing_args):
    """
        Processes all segments_model_instances. FIXME: write detailed doc
    """
    # redirect stndard error to devnull. FIXME if we can capture it segment-wise (that
    # would be great but.. how much effort and how much performances decreasing?)
    # redirect_external_out(2)

    # set after how many processed segments we want to commit. Setting it higher might speed up
    # calculations at expense of loosing max_session_new segment if just one is wrong
    max_session_new = 10
    # commit for safety:
    commit(session, on_exc=lambda exc: logger.error(str(exc)))

    calculated = 0
    saved = 0

    logger.info("Processing %d segments", len(segments_model_instances))
    ret = []

    sta2segs = defdict(lambda: [])
    for seg in segments_model_instances:
        sta2segs[seg.channel.station_id].append(seg)

    # process segments station-like, so that we load only one inventory at a time
    # and hopefully it will garbage collected (inventory object is big)
    for sta_id, segments in sta2segs.iteritems():
        inventory = None
        try:
            inventory = get_inventory(segments[0], session, timeout=30)
        except SQLAlchemyError as exc:
            logger.warning("Error while saving inventory (station id=%s), "
                           "%d segment will not be processed: %s",
                           str(sta_id), len(segments), str(exc))
            session.rollback()
        except (urllib2.HTTPError, urllib2.URLError, httplib.HTTPException, socket.error) as _:
            logger.warning("Error while downloading inventory (station id=%s), "
                           "%d segment will not be processed: %s URL: %s",
                           str(sta_id), len(segments), str(_), get_inventory_query(segments[0]))
        except Exception as exc:  # pylint:disable=broad-except
            logger.warning("Error while creating inventory (station id=%s), "
                           "%d segment will not be processed: %s",
                           str(sta_id), len(segments), str(exc))

        if inventory is None:
            notify_progress_func(len(segments))
            continue
            # pass

        # THIS IS THE METHOD WITHOUT MULTIPROCESS: 28, 24.7 secs on 30 segments
        for seg in segments:
            notify_progress_func(1)
            pro = models.Processing(run_id=run_id)
            # pro.segment = seg
            # session.flush()
            try:
                pro = process(pro, seg, seg.channel, seg.channel.station, seg.event,
                              seg.datacenter, inventory, **processing_args)
                pro.id = None
                pro.segment = seg
                calculated += 1
                ret.append(pro)
                # flush(session, on_exc=lambda exc: logger.error(str(exc)))
                if len(ret) >= max_session_new:
                    added = len(ret)
                    session.add_all(ret)
                    ret = []
                    if commit(session,
                              on_exc=lambda exc: logger.warning(msgs.db.dropped_seg(added,
                                                                                    None,
                                                                                    exc))):
                        saved += added
            except Exception as exc:  # pylint:disable=broad-except
                logger.warning(msgs.calc.dropped_seg(seg, "segments processing", exc))

    added = len(ret)
    if added and commit(session, on_exc=lambda exc: logger.warning(msgs.db.dropped_seg(added,
                                                                                       None,
                                                                                       exc))):
        saved += added
    logger.info("")
    logger.info("%d segments successfully processed, %d succesfully saved", calculated, saved)
    return ret


def get_inventory(segment, session=None, **kwargs):
    """raises tons of exceptions (see main). FIXME: write doc
    :param session: if **not** None but a valid sqlalchemy session object, then
    the inventory, if downloaded because not present, will be saveed to the db (compressed)
    """
    data = segment.channel.station.inventory_xml
    if not data:
        query_url = get_inventory_query(segment.channel.station)
        data = url_read(query_url, **kwargs)
        if session and data:
            segment.channel.station.inventory_xml = dumps_inv(data)
            session.commit()
        elif not data:
            raise ValueError("No data from server")
    return loads_inv(data)


# def warn(segment, exception_or_msg):
#     """ convenience function for logging warnings during processing"""
#     logger.warning("while processing segment.id='%s': %s", str(segment.id), str(exception_or_msg))


# def dtime(utcdatetime):
#     """converts UtcDateTime to datetime, returns None if arg is None"""
#     return None if utcdatetime is None else utcdatetime.datetime


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
            pro.t_pga_atime_t95 = dumps_time(t_PGA)
            pro.t_pgv_atime_t95 = dumps_time(t_PGV)
            pro.t_pwa_atime_t95 = dumps_time(t_PWA)
            pro.cum_t05 = dumps_time(t05)
            pro.cum_t10 = dumps_time(t10)
            pro.cum_t25 = dumps_time(cum_times[2])
            pro.cum_t50 = dumps_time(cum_times[3])
            pro.cum_t75 = dumps_time(cum_times[4])
            pro.cum_t90 = dumps_time(t90)
            pro.cum_t95 = dumps_time(t95)
            pro.snr_rem_resp_fixedwindow = snr_rem_resp_fixed_window
            pro.snr_rem_resp_t05_t95 = snr_rem_resp_t05_t95
            pro.snr_rem_resp_t10_t90 = snr_rem_resp_t10_t90
            # pro.amplitude_ratio = Column(Float)
            # pro.is_saturated = Column(Boolean)
            # pro.has_gaps = Column(Boolean)
            pro.double_event_result = double_evt[0]
            pro.secondary_event_time = dumps_time(double_evt[1])

            # WITH JESSIE IMPLEMENT CODA ANALYSIS:
            # pro.coda_tmax = Column(DateTime)
            # pro.coda_length_sec = Column(Float)
    return pro

