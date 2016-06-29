'''
Created on Jun 20, 2016

@author: riccardo
'''
import numpy as np
from obspy.core import Stream, Trace, UTCDateTime
from stream2segment.analysis import cumsum as _cumsum, dfreq, env as _env, pow_spec, amp_spec


def isstream(trace_or_stream):
    """Returns true or false if the argument is an obspy trace or an obspy stream (collection
    of traces"""
    return hasattr(trace_or_stream, 'traces')


def itertrace(trace_or_stream):
    """Iterator over the argument. If the latter is a trace, returns it. If it is a stream
    returns all its traces"""
    for tr in (trace_or_stream if isstream(trace_or_stream) else [trace_or_stream]):
        yield tr


def bandpass(trace_or_stream, freq_min=0.1, freq_max=20, corners=2):
    """filters a signal trace_or_stream"""
    trace_or_stream_filtered = trace_or_stream.copy()

    for tr in itertrace(trace_or_stream_filtered):
        # define sampling freq
        sampling_rate = tr.stats.sampling_rate
        # adjust the max_f_max to 0.9 of the nyquist frea (sampling rate /2)
        # slightly less than nyquist (0.9) seems to avoid artifacts
        max_f_max = 0.9 * (sampling_rate / 2.0)
        freq_max = min(freq_max, max_f_max)

        # remove artifacts:
        # offset:
        tr.data = tr.data - np.nanmean(tr.data)

        # tapering
        tr.taper(type='cosine', max_percentage=0.05)

        lgt = len(tr.data)
        tr.data = np.append(tr.data, np.zeros(lgt))
        # tr.taper(type='cosine', max_percentage=0.15)
        tr.filter('bandpass', freqmin=freq_min, freqmax=freq_max, corners=corners, zerophase=True)
        # smooth tail artifacts due to transients:
        tr.data = tr.data[:lgt]

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
    """Returns the cumulative function, normalized between 0 and 1 of the argument
    :param trace_or_stream: either an obspy trace, or an obspy stream (collection of traces). In the
    latter case, the cumulative is applied on all traces
    :return: an obspy trace or stream (depending on the argument)
    """
    def func(trace):
        return Trace(_cumsum(trace.data, trace.stats.delta), header=trace.stats.copy())

    return apply(trace_or_stream, func)


def env(trace_or_stream):
    """
    Returns the envelope (using scipy hilbert transform) of the argument
    :param trace_or_stream: either an obspy trace, or an obspy stream (collection of traces). In the
    latter case, the cumulative is applied on all traces
    :return: an obspy trace or stream (depending on the argument)
    """
    def func(trace):
        return Trace(_env(trace.data), header=trace.stats.copy())

    return apply(trace_or_stream, func)


def snr(trace_or_stream, fixed_time, window_in_sec):
    """
    Returns an obspy stream object where the first trace is the power spectrum P1 of
    the window (fixed_time-window_in_sec), and the second trace is the power spectrum P2 of the
    window (fixed_time+window_in_sec)
    If the argument is a stream of N traces, returns a stream object of N*2 traces:
    [P1(trace1), P2(trace1), P1(trace2), P2(trace2), ...]
    :param trace_or_stream: either an obspy trace, or an obspy stream (collection of traces). In the
    latter case, the cumulative is applied on all traces
    :return: an obspy stream (regardless of whether the argument is a trace or stream object)
    """
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
    """
        Class extending an obspy trace for frequency x scales instead of the default time x scales.
        The trace.data argument holds numeric values referring to e.g. amplitudes, and trace.stats
        object has an additional attribute 'df' denoting the sampling frequency. The latter is also
        accessible as object attribute: Trace.df
        All other trace.stats attributes refer to the generating time-series trace, in case ones
        wants that kind of info accessible
    """
    def __init__(self, trace, calc_pow_spec=True):
        ft1 = pow_spec(trace.data) if calc_pow_spec else amp_spec(trace.data)
        Trace.__init__(self, ft1, header=trace.stats.copy())
        self.df = dfreq(trace.data, trace.stats.delta)
        self.stats.df = self.df
