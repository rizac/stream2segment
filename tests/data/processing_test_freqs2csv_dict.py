'''
Created on Feb 2, 2017

@author: riccardo
'''
# import numpy for fatser numeric array processing:
import numpy as np
# strem2segment functions for processing mseeds
# If you need to use them, import them like this:
from stream2segment.analysis.mseeds import remove_response, get_gaps, amp_ratio, bandpass, cumsum,\
    cumtimes, fft, maxabs, simulate_wa, get_multievent, snr, dfreq
# when working with times, use obspy UTCDateTime:
from obspy.core.utcdatetime import UTCDateTime
# stream2segment function for processing numpy arrays (such as stream.traces[0])
# If you need to to use them, import them:
from stream2segment.analysis import amp_spec, freqs


def main(seg, config):

    if not seg.data:
        raise ValueError('empty data')

    stream = seg.stream()

    if get_gaps(stream):
        raise ValueError('has gaps')

    if len(stream) != 1:
        raise ValueError('more than one obspy.Trace')

    # work on the trace now. All functions will return Traces or scalars, which is better
    # so we can write them to database more easily
    trace = stream[0]

    ampratio = amp_ratio(trace)
    if ampratio >= config['amp_ratio_threshold']:
        raise ValueError('possibly saturated (amp. ratio exceeds)')

    # convert to UTCDateTime for operations later:
    a_time = UTCDateTime(seg.arrival_time) + config['arrival_time_delay']

    evt = seg.event
    trace = bandpass(trace, evt.magnitude, freq_max=config['bandpass_freq_max'],
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

    fft_rem_resp = fft(trace_rem_resp, a_time, config['snr_window_length'],
                       taper_max_percentage=config['taper_max_percentage'])
    aspec = amp_spec(fft_rem_resp.data, True)

    ret = np.interp(config['freqs_interp'], freqs(aspec, dfreq(fft_rem_resp)), aspec)

    return {'f%d' % i: r for i, r in enumerate(ret)}
