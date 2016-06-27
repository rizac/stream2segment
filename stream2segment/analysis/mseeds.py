'''
Created on Jun 20, 2016

@author: riccardo
'''
from obspy.core import Stream, Trace, UTCDateTime
from stream2segment.analysis import cumsum as _cumsum, dfreq, env as _env, pow_spec, amp_spec


def isstream(trace_or_stream):
    return hasattr(trace_or_stream, 'traces')


def itertrace(trace_or_stream):
    for tr in (trace_or_stream if isstream(trace_or_stream) else [trace_or_stream]):
        yield tr


def bandpass(trace_or_stream, freq_min=0.1, freq_max=20, corners=2):
    """filters a signal trace_or_stream as obtained from obspy.read"""
    trace_or_stream_filtered = trace_or_stream.copy()

    for tr in itertrace(trace_or_stream_filtered):
        # define sampling freq
        sampling_rate = tr.stats.sampling_rate
        # adjust the max_f_max to 0.9 of the nyquist frea (sampling rate /2)
        # slightly less than nyquist (0.9) seems to avoid artifacts
        max_f_max = 0.9 * (sampling_rate / 2.0)
        freq_max = min(freq_max, max_f_max)
        # tr.taper(type='cosine', max_percentage=0.15)
        tr.filter('bandpass', freqmin=freq_min, freqmax=freq_max, corners=corners, zerophase=True)
        # smooth tail artifacts due to transients:
        tr.taper(type='cosine', max_percentage=0.05)

    return trace_or_stream_filtered


def apply(trace_or_stream, func):
    """"
        func is a function taking a Trace as argument and returning a new Trace
    """
    traces = []
    for tr in itertrace(trace_or_stream):
        traces.append(func(tr))

    if isstream(trace_or_stream):
        return Stream(traces)
    else:
        return traces[0]


def cumsum(trace_or_stream):
    def func(trace):
        return Trace(_cumsum(trace.data, trace.stats.delta), header=trace.stats.copy())

    return apply(trace_or_stream, func)


def env(trace_or_stream):
    def func(trace):
        return Trace(_env(trace.data), header=trace.stats.copy())

    return apply(trace_or_stream, func)


def snr(trace_or_stream, fixed_time, window_in_sec):
    if not isinstance(fixed_time, UTCDateTime):
        fixed_time = UTCDateTime(fixed_time)
    traces = []
    for tr in itertrace(trace_or_stream):
        signal1 = tr.slice(fixed_time-window_in_sec, fixed_time)
        traces.append(FTrace(signal1))
        signal2 = tr.slice(fixed_time, fixed_time+window_in_sec)
        traces.append(FTrace(signal2))

    return Stream(traces)


class FTrace(Trace):

    def __init__(self, trace, calc_pow_spec=True):
        ft1 = pow_spec(trace.data) if calc_pow_spec else amp_spec(trace.data)
        Trace.__init__(self, ft1, header=trace.stats.copy())
        self.df = dfreq(trace.data, trace.stats.delta)
        self.stats.df = self.df


# def snr_(trace, fixed_time, window_in_sec):
#     fixed_time = todt(fixed_time)
# 
#     noise_dt = [fixed_time - timedelta(seconds=window_in_sec), fixed_time]
#     normal_dt = [fixed_time, fixed_time + timedelta(seconds=window_in_sec)]
# 
#     start = todt(trace.stats.starttime)
#     end = todt(trace.stats.endtime)
# 
#     if start > noise_dt[0]:
#         noise_dt[0] = start
# 
#     if end < noise_dt[1]:
#         noise_dt[1] = end
# 
#     dt = trace.stats.delta
# 
#     noise_idxs = [
#                   int((noise_dt[0] - start).total_seconds() / dt),
#                   int((noise_dt[1] - start).total_seconds() / dt)
#                   ]
# 
#     normal_idxs = [
#                   int((normal_dt[0] - start).total_seconds() / dt),
#                   int((normal_dt[1] - start).total_seconds() / dt)
#                   ]
# 
#     signal1 = trace.data[noise_idxs[0]: noise_idxs[1]]
#     signal2 = trace[normal_idxs[0]: normal_idxs[1]]
# 
# #     ft1 = np.fft.rfft(signal1)
# #     freq1 = np.fft.rfftfreq(len(signal1), d=dt)
# #     ft2 = np.fft.rfft(signal2)
# #     freq2 = np.fft.rfftfreq(len(signal2), d=dt)
# #
# #     return [
# #             {'freqs': freq1, 'vals': ft1},
# #             {'freqs': freq2, 'vals': ft2},
# #             ]
# #
#     return [fft(signal1, dt), fft(signal2, dt)]
