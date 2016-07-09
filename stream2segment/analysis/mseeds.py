'''
Created on Jun 20, 2016

@author: riccardo
'''
import numpy as np
from obspy.core import Stream, Trace, UTCDateTime
from stream2segment.analysis import fft as _fft, linspace as xlinspace, cumsum as _cumsum, dfreq, env as _env, pow_spec, amp_spec
from obspy import read, read_inventory


def itertrace(trace_or_stream):
    """Iterator over the argument. If the latter is a Stream, returns it. If it is a trace
    returns the argument wrapped into a list"""
    return trace_or_stream if isinstance(trace_or_stream, Stream) else [trace_or_stream]


def bandpass(trace_or_stream, magnitude, freq_max=20, max_nyquist_ratio=0.9,
             corners=2):
    """filters a signal trace_or_stream.
    FIXME: add comment!!!
    :param magnitude: the magnitude which originated the trace (or stream). It dictates the value
    of the high-pass corner (the minimum frequency, freq_min, in Hz)
    :param freq_max: the value of the low-pass corner (freq_max), in Hz
    :param max_nyquist_ratio: the ratio of freq_max to be computed. The real low-pass corner will
    be set as max_nyquist_ratio * freq_max (default: 0.9, i.e. 90%)
    :param corners: the corners (i.e., the order of the filter)
    """
    trace_or_stream_filtered = trace_or_stream.copy()

    # get freq_min according to magnitude (see e.g. RRSM or ISM)
    # (this might change in the future)
    if magnitude <= 4:
        freq_min = 0.3
    elif magnitude <= 5:
        freq_min = 0.2
    elif magnitude <= 6.5:
        freq_min = 0.1
    else:
        freq_min = 0.05

    for tra in itertrace(trace_or_stream_filtered):

        # define sampling freq
        sampling_rate = tra.stats.sampling_rate
        # adjust the max_f_max to 0.9 of the nyquist frea (sampling rate /2)
        # slightly less than nyquist (0.9) seems to avoid artifacts
        max_f_max = max_nyquist_ratio * (sampling_rate / 2.0)
        freq_max = min(freq_max, max_f_max)

        # Start filtering (several pre-steps)
        # 1) offset removal:
        tra.data = tra.data - np.nanmean(tra.data)

        # 2) tapering
        tra.taper(type='cosine', max_percentage=0.05)

        # 3) pad data with zeros at the END in order to filter transient
        lgt = len(tra.data)
        tra.data = np.append(tra.data, np.zeros(lgt))

        # 4) apply bandpass filter:
        tra.filter('bandpass', freqmin=freq_min, freqmax=freq_max, corners=corners, zerophase=True)

        # 5) remove padded elements:
        tra.data = tra.data[:lgt]

    return trace_or_stream_filtered


def apply(trace_or_stream, func, *args, **kwargs):
    """"
        func is a function taking a Trace as argument and returning a new Trace
    """
    traces = []
    itr = itertrace(trace_or_stream)
    for tra in itr:
        traces.append(func(tra, *args, **kwargs))

    return Stream(traces) if isinstance(itr, Stream) else traces[0]


def remove_response(trace_or_stream, inventory_or_inventory_path, output='ACC', water_level=60):
    """
        :param inventory: either an inventory object, or a (absolute) path to a specified inventory
            objects
    """
    inventory = read_inventory(inventory_or_inventory_path) \
        if isinstance(inventory_or_inventory_path, basestring) else inventory_or_inventory_path

    def func(tra):
        """removes the response on the trace"""
        tra.remove_response(inventory=inventory, output=output, water_level=water_level)
        return tra

    return apply(trace_or_stream, func)


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
    try:
        return trace_or_stream.get_gaps()
    except AttributeError:  # is a Trace
        return []


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
    a list of M lists if the argument is a stream of M traces, where each sub-list is
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
    itr = itertrace(cum_trace_or_stream)
    isstream = isinstance(itr, Stream)
    times = []
    for cum_tra in itr:
        starttime = cum_tra.stats.starttime
        delta = cum_tra.stats.delta
        val = []
        minv = cum_tra[0]
        maxv = cum_tra[-1]
        for perc in percentages:
            idx = np.searchsorted(cum_tra.data, minv + (maxv - minv) * perc)
            val.append(starttime + idx * delta)
#             val.append(starttime +
#                        (np.where(cum_tra.data <= perc)[0][-1]) * delta)
        if not isstream:
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


def fft(trace_or_stream, fixed_time=None, window_in_sec=None, taper_max_percentage=0.05,
        taper_type='hann'):
    """
    Returns a numpy COMPLEX array (or a list of numpy arrays, if the first argument is a Stream)
    resulting from the fft applied on (a sliced version of) the argument
    :param trace_or_stream: either an obspy trace, or an obspy stream (collection of traces). In the
    latter case, the fft is applied on all traces
    :param fixed_time: the fixed time where to set the start (if `window_in_sec` > 0) or end
    (if `window_in_sec` < 0) of the trace slice on which to apply the fft. If None, it defaults
    to the start of each trace
    :type fixed_time: an `obspy.UTCDateTije` object or any objectthat can be passed as argument
    to the latter (e.g., a numeric timestamp)
    :param window_in_sec: the window, in sec, of the trace slice where to apply the fft. If None,
    it defaults to the amount of time from `fixed_time` till the end of each trace
    :type window_in_sec: numeric
    :return: a NUMPY obspy stream (regardless of whether the argument is a trace or stream object)
    """
    if not isinstance(fixed_time, UTCDateTime) and fixed_time is not None:
        fixed_time = UTCDateTime(fixed_time)
    ret_list = []
    itr = itertrace(trace_or_stream)
    for tra in itr:

        tra = tra.copy()
        if fixed_time is None:
            starttime = None
            endtime = None if window_in_sec is None else window_in_sec
        elif window_in_sec is None:
            starttime = fixed_time
            endtime = None
        else:
            t01 = fixed_time
            t02 = fixed_time+window_in_sec
            starttime, endtime = min(t01, t02), max(t01, t02)

        trim_tra = tra.trim(starttime=starttime, endtime=endtime, pad=True, fill_value=0)
        if taper_max_percentage > 0:
            trim_tra.taper(max_percentage=0.05, type=taper_type)
        dft = _fft(trim_tra.data)
        # remember: now we have a numpy array of complex numbers
        ret_list.append(dft)

    return ret_list if isinstance(itr, Stream) else ret_list[0]


def amp_ratio(trace_or_stream):
    """Returns a list of numeric values (if the argument is a stream) or a single
    numeric value (if the argument is a single trace) representing the amplitude ratio given by:
        np.nanmax(np.abs(trace.data)) / 2 ** 23
    """
    itr = itertrace(trace_or_stream)
    isstream = isinstance(itr, Stream)
    ampratios = []
    threshold = 2 ** 23
    for tra in itr:
        amprat = np.true_divide(np.nanmax(np.abs(tra.data)), threshold)
        if not isstream:
            return amprat
        ampratios.append(amprat)

    return ampratios


def timearray(trace_or_stream, npts=None):
    """
        Returns the x values of the given trace_or_stream according to each trace stats
        if npts is not None, returns a linear space of equally sampled x from
        tra.starttime and tra.endtime. Otherwise, npts equals the trace number of points
    """
    ret = []
    itr = itertrace(trace_or_stream)
    for tra in itr:
        num = len(tra.data) if npts is None else npts  # we don't trust tra.stats.npts
        ret.append(np.linspace(tra.stats.starttime.timestamp,
                               tra.stats.endtime.timestamp, num=num, endpoint=True))

    return ret if isinstance(itr, Stream) else ret[0]


def interpolate(trace_or_stream, npts_or_new_x_array, align_if_stream=True,
                return_x_array=False):
    """Returns a trace or stream interpolated with the given number of points. This method
    differs from obspy.Trace.interpolate in that it offers a probably faster but less fine-tuned
    way to (linearly) interpolate a Trace or a Stream (and in this case optionally align its Traces
    on the same time range, if needed). This method is intended to optimize visualizations
    of the data, rather than performing calculation on it
    :param trace_or_stream: the Trace or Stream
    :param npts_or_new_x_array: the new number of points (if python integer) or the new x array
        where to interpolate the trace(s) (as UTCDateTime.timestamp's)
    :param align_if_stream: Ignored if the first argument is a Trace object. Otherwise, if True,
    data will be "aligned" with the timestamp of the first trace:
    all the traces (if more than one) will have the same start and end time, and the same number
    of points. This argument is always True if npts_or_new_x_array is an array of timestamps
    :param return_new_x_array: if True (false by default) a tuple (newtimes, new_trace_or_stream) is
    returned. Otherwise, only new_trace_or_stream is returned. The name return_new_x_array is more
    general as, if FTraces are passed, then the units of the array are frequencies (in Hz)
    """
    newxarray = None
    try:
        len(npts_or_new_x_array)  # is npts_or_new_x_array an array?
        newxarray = npts_or_new_x_array
        align_if_stream = True
    except TypeError:  # npts_or_new_x_array is scalar (not array)
        npts = npts_or_new_x_array

    ret = []
    oldxarrays = {}
    itr = itertrace(trace_or_stream)
    for tra in itr:
        if newxarray is None or not align_if_stream:
            newxarray = timearray(tra, npts)

        # get old x array. If we have an x array of times already calculated for a previous
        # trace with same start, delta and number of points, use that (improving performances):
        key = (tra.stats.starttime.timestamp, tra.stats.delta, len(tra.data))
        oldxarray = oldxarrays.get(key, None)
        if oldxarray is None:
            oldxarray = timearray(tra)
            oldxarrays[key] = oldxarray

        data = np.interp(newxarray, oldxarray, tra.data)
        header = tra.stats.copy()
        header.npts = len(data)
        header.delta = newxarray[1] - newxarray[0]
        header.starttime = UTCDateTime(newxarray[0])

        ret.append(Trace(data=data, header=header))

    ret = Stream(ret) if isinstance(itr, Stream) else ret[0]
    return (newxarray, ret) if return_x_array else ret
