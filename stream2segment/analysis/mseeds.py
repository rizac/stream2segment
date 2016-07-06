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
    for tra in trace_or_stream if isstream(trace_or_stream) else [trace_or_stream]:
        yield tra


def bandpass(trace_or_stream, freq_min=0.1, freq_max=20, corners=2):
    """filters a signal trace_or_stream"""
    trace_or_stream_filtered = trace_or_stream.copy()

    for tra in itertrace(trace_or_stream_filtered):
        # define sampling freq
        sampling_rate = tra.stats.sampling_rate
        # adjust the max_f_max to 0.9 of the nyquist frea (sampling rate /2)
        # slightly less than nyquist (0.9) seems to avoid artifacts
        max_f_max = 0.9 * (sampling_rate / 2.0)
        freq_max = min(freq_max, max_f_max)

        # remove artifacts:
        # offset:
        tra.data = tra.data - np.nanmean(tra.data)

        # tapering
        tra.taper(type='cosine', max_percentage=0.05)

        lgt = len(tra.data)
        tra.data = np.append(tra.data, np.zeros(lgt))
        # tra.taper(type='cosine', max_percentage=0.15)
        tra.filter('bandpass', freqmin=freq_min, freqmax=freq_max, corners=corners, zerophase=True)
        # smooth tail artifacts due to transients:
        tra.data = tra.data[:lgt]

    return trace_or_stream_filtered


def apply(trace_or_stream, func, *args, **kwargs):
    """"
        func is a function taking a Trace as argument and returning a new Trace
    """
    traces = []
    for tra in itertrace(trace_or_stream):
        traces.append(func(tra, *args, **kwargs))

    if isstream(trace_or_stream):
        return Stream(traces)
    else:
        return traces[0]


def get_gaps(trace_or_stream):
    """
        Returns a list of gaps for the current argument. The list elements have the form:
            [network, station, location, channel, starttime of the gap, end time of the gap,
             duration of the gap, number of missing samples]
        :param trace_or_stream: a Trace, or a Stream. Due to the fact that obspy get_gaps is
        Stream only, if this argument is a trace it will be converted to a Stream internally. This
        does not affect the returned value type
        :return: a list of gaps
        :rtype: list of lists
    """
    if not isstream(trace_or_stream):
        return []
    return trace_or_stream.get_gaps()


def cumsum(trace_or_stream):
    """Returns the cumulative function, normalized between 0 and 1 of the argument
    :param trace_or_stream: either an obspy trace, or an obspy stream (collection of traces). In the
    latter case, the cumulative is applied on all traces
    :return: an obspy trace or stream (depending on the argument)
    """
    def func(trace):
        """the func to apply to a given trace"""
        return Trace(_cumsum(trace.data, trace.stats.delta), header=trace.stats.copy())

    return apply(trace_or_stream, func)


def cumtimes(cum_trace_or_stream, *percentages):
    """Given cum_trace_or_stream (a trace or stream resulting from cumsum, i.e. the
    normalized cumulative of a given trace or stream), calculates the time(s) where the signal
    reaches the given percentage(s) of the toal signal (which is 1)
    Called P = len(percentages), returns a list of length P if the first argument is a Trace, or
    a list of M sub-lists if the argument is a stream of M traces, where each sub-list is
    a list of length P.
    Note that each element of the list is an obspy.UTCTimeStamp have the timestamp attribute which
    returns the relative timestamp, in case
    a numeric value is needed
    :param: trace_or_stream: a trace or a stream (collection of traces)
    :param: percentages: the precentages to be calculated, e.g. 0.05, 0.95 (5% and 95%)
    :return: a list of length P = len(percentages) denoting the the obspy.UTCTimeStamp(s) where
    the given percentages occur. If the argument
    is a stream, returns a list of lists, where each sub-list (of length P) refers to the i-th trace
    """
    istrace = not isstream(cum_trace_or_stream)
    times = []
    for cum_tra in itertrace(cum_trace_or_stream):
        starttime = cum_tra.stats.starttime
        delta = cum_tra.stats.delta
        val = []
        for perc in percentages:
            val.append(starttime +
                       (np.where(cum_tra.data <= perc)[0][-1]) * delta)  # .timestamp * 1000
        if istrace:
            return val

        times.append(val)
    return times


def env(trace_or_stream):
    """
    Returns the envelope (using scipy hilbert transform) of the argument
    :param trace_or_stream: either an obspy trace, or an obspy stream (collection of traces). In the
    latter case, the cumulative is applied on all traces
    :return: an obspy trace or stream (depending on the argument)
    """
    def func(trace):
        """the func to apply to a given trace"""
        return Trace(_env(trace.data), header=trace.stats.copy())

    return apply(trace_or_stream, func)


def freq_stream(trace_or_stream, fixed_time, window_in_sec):
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
    for tra in itertrace(trace_or_stream):
        signal1 = tra.slice(fixed_time-window_in_sec, fixed_time)
        traces.append(FTrace.from_trace(signal1))
        signal2 = tra.slice(fixed_time, fixed_time+window_in_sec)
        traces.append(FTrace.from_trace(signal2))

    return Stream(traces)


def snr(ftrace_signal, ftrace_noise):
    """Returns the signal to noise ratio of two FTraces, the first representing the signal trace
    (no noise), the second the noisy trace. Usually, both arguments are slices of the same
    trace"""
    return 10 * np.log10(np.sum(ftrace_signal.data) / np.sum(ftrace_noise.data))


class FTrace(Trace):
    """
        Class extending an obspy trace for frequency x scales instead of the default time x scales.
        The trace.data argument holds numeric values referring to e.g. amplitudes, and trace.stats
        object has an additional attribute 'df' denoting the sampling frequency. The latter is also
        accessible as object attribute: Trace.df
        All other trace.stats attributes refer to the generating time-series trace, in case ones
        wants that kind of info accessible
    """
    def __init__(self, freq_data, header=None, df=0):
        Trace.__init__(self, freq_data, header=header)
        self.stats.df = df

    @staticmethod
    def from_trace(trace, calc_pow_spec=True):
        data = pow_spec(trace.data) if calc_pow_spec else amp_spec(trace.data)
        return FTrace(data, header=trace.stats.copy(), df=dfreq(trace.data, trace.stats.delta))
#         Trace.__init__(self, ft1, header=trace.stats.copy())
#         self.stats.df = dfreq(trace.data, trace.stats.delta)

    @property
    def df(self):
        return self.stats.df


def amp_ratio(trace_or_stream):
    """Returns a list of numeric values (if the argument is a stream) or a single
    numeric value (if the argument is a single trace) representing the amplitude ratio given by:
        np.nanmax(np.abs(trace.data)) / 2 ** 23
    """
    istrace = not isstream(trace_or_stream)
    ampratios = []
    for tra in itertrace(trace_or_stream):
        amprat = np.true_divide(np.nanmax(np.abs(tra.data)), 2 ** 23)
        if istrace:
            return amprat
        ampratios.append(amprat)

    return ampratios


def xlinspace(trace, num=None):
    """
    Returns the numpy array of evenly spaced values x values of the given trace, in timestamps
    (derived from UTCDateTime.timestamp). If the argument is an instance of FTrace, returns evenly
    spaced frequencies. The returned array is ASSURED to be within the trace bounds (usually
    time units, or frequency units if the argument is an FTrace)
    :param num: the number of points. If None, the trace number of points is used (i.e., return
    the trace x values, usually timestamps). Otherwise, it creates an evenly spaced array with
    bounds given by the trace bounds
    """
    if num is None:
        num = len(trace.data)

    if isinstance(trace, FTrace):
        return np.linspace(0, trace.df*len(trace.data), num, endpoint=False)
    else:
        return np.linspace(trace.stats.starttime.timestamp,
                           trace.stats.endtime.timestamp, num, endpoint=True)


def interpolate(trace_or_stream, npts_or_new_x_array, align_if_stream=True,
                return_x_array=False):
    """Returns a trace or stream interpolated with the given number of points. This method
    differs from obspy.Trace.interpolate in that is limited to the case where visualization must
    occur and no calculation on the data is needed, so a linear interpolation will take place and
    the sampling_rate will be set according to
    the npts specified. In the original obspy interpolation, the sampling_rate must be specified
    and more complex options can be set
    :param trace_or_stream: the Trace or Stream
    :param npts_or_new_x_array: the new number of points (if python integer) or the new x array
        where to interpolate the trace(s)
    :param align_if_stream: if True, data will be "aligned" with the timestamp of the first trace
    :param return_new_x_array: if True (false by default) a tuple (newtimes, new_trace_or_stream) is
    returned. Otherwise, only new_trace_or_stream is returned. The name return_new_x_array is more
    general as, if FTraces are passed, then the units of the array are frequencies (in Hz)
    """
    try:
        len(npts_or_new_x_array)
        newxarray = npts_or_new_x_array
        x_array_given = True
    except TypeError:
        npts = npts_or_new_x_array
        x_array_given = False

    istrace = not isstream(trace_or_stream)
    tra = trace_or_stream if istrace else trace_or_stream[0]
    isftrace = isinstance(tra, FTrace)
    starttime = tra.stats.starttime

    if not x_array_given:
        newxarray = None if align_if_stream is False else xlinspace(tra, npts)

    def func(tra, newxarray, isftrace):
        """interpolates the trace"""
        if newxarray is None:
            newxarray = xlinspace(tra, npts)
        oldxarray = xlinspace(tra)
        data = np.interp(newxarray, oldxarray, tra.data)
        header = tra.stats.copy()
        header.npts = len(data)
        if isftrace:
            header.df = newxarray[1] - newxarray[0]
        else:
            header.delta = newxarray[1] - newxarray[0]
            header.starttime = starttime  # redundant, for first trace in iter...
            # all other fields are updated automatically

        return FTrace(freq_data=data, header=header, df=header.df) if isftrace else \
            Trace(data=data, header=header)

    ret = apply(trace_or_stream, func, newxarray, isftrace)

    return (newxarray, ret) if return_x_array else ret
