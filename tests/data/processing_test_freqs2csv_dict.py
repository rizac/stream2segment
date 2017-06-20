'''
Created on Feb 2, 2017

@author: riccardo
'''
# import numpy for fatser numeric array processing:
import numpy as np
# strem2segment functions for processing mseeds
# If you need to use them, import them like this:
from stream2segment.analysis.mseeds import remove_response, amp_ratio, bandpass, cumsum,\
    cumtimes, fft, maxabs, simulate_wa, snr, get_tbounds
# when working with times, use obspy UTCDateTime:
from obspy.core.utcdatetime import UTCDateTime
# stream2segment function for processing numpy arrays (such as stream.traces[0])
# If you need to to use them, import them:
from stream2segment.analysis import ampspec, freqs


def main(seg, config):

    if not seg.data:
        raise ValueError('empty data')

    stream = seg.stream()

    if len(stream) != 1:
        raise ValueError('more than one obspy.Trace. Possible cause: gaps')

    # work on the trace now. All functions will return Traces or scalars, which is better
    # so we can write them to database more easily
    trace = stream[0]

    ampratio = amp_ratio(trace)
    if ampratio >= config['amp_ratio_threshold']:
        raise ValueError('possibly saturated (amp. ratio exceeds)')

    # convert to UTCDateTime for operations later:
    a_time = UTCDateTime(seg.arrival_time) + config['arrival_time_delay']

    evt = seg.event
    fmin = mag2freq(evt.magnitude)
    trace = bandpass(trace, fmin, freq_max=config['bandpass_freq_max'],
                     max_nyquist_ratio=config['bandpass_max_nyquist_ratio'],
                     corners=config['bandpass_corners'])

    inventory = seg.inventory()
    trace_rem_resp = remove_response(trace, inventory, output=config['remove_response_output'],
                                     water_level=config['remove_response_water_level'])

    # to calculate cumulative:
    # mseed_cum = cumsum(trace_rem_resp)
    # and then:
    # t005, t010, t025, t050, t075, t90, t95 = cumtimes(mseed_cum, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
    # then, for instance:
    # mseed_rem_resp_t05_t95 = trace_rem_resp.slice(t05, t95)

    # t_PGA, PGA = maxabs(trace_rem_resp, a_time, t95)  # if remove_response_output == 'ACC'
    # t_PGV, PGV = maxabs(trace_rem_resp, a_time, t95)  # if remove_response_output = 'VEL'

    starttime, endtime = get_tbounds(trace_rem_resp, a_time, config['snr_window_length'])
    amp_spec_freqs, amp_spec = ampspec(trace_rem_resp, starttime, endtime,
                                       taper_max_percentage=config['taper_max_percentage'],
                                       return_freqs=True)
    required_freqs = config['freqs_interp']
    ret = np.interp(required_freqs, amp_spec_freqs, amp_spec)

    return {'f%d' % i: r for i, r in enumerate(ret)}



def mag2freq(magnitude):
    """converts magnitude to frequency. Used in our bandpass function to get the min freq.
    parameter"""
    if magnitude <= 4:
        freq_min = 0.5
    elif magnitude <= 5:
        freq_min = 0.3
    elif magnitude <= 6.0:
        freq_min = 0.1
    else:
        freq_min = 0.05
    return freq_min
