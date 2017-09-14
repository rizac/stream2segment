'''
Utility functions for analyzing and processing miniSeed (`obspy.core.Stream` objects)

Created on Jun 20, 2016

@author: riccardo
'''
import numpy as np

# from scipy.signal import savgol_filter
# try:
#     import cPickle as pickle
# except ImportError:
#     import pickle  # @UnusedImport
from obspy.core import Stream, Trace, UTCDateTime  # , Stats
# from obspy import read_inventory
from stream2segment.analysis import fft as _fft, ampspec as _ampspec, powspec as _powspec,\
    snr as _snr, cumsum as _cumsum, dfreq, freqs


def stream_compliant(func):
    """Returns a function wrapping (and calling internally) `func` which makes the latter
    stream/trace-compliant: in other words, allows `func` to accept either an
    `obspy.Trace` or an `obspy.Stream` objects.
    As a function decorator:
    ```
        \@stream_compliant
        def func(trace,...)
    ```
    Then `func` can be called with either a Trace or a Stream as first argument.
    If Trace, then func behaves as implemented; If Stream, then func is applied to any of its
    Traces, and a Stream wrapping each result is returned (*if any result is not a Trace,
    a list is returned instead of a Stream*).

    :param func: any function with the only constraint of having an `obspy.Trace` as first
    argument. The function can then be called with a Stream instead of that trace

    Rationale: A Trace is the obspy core object representing a time-series. Therefore, in
    principle all processing functions, like those defined here, should work on traces.
    However, obspy provides also Stream objects (basically, collections of Traces)
    *which represent the miniSEED file stored on disk* (writing e.g. a Trace T
    to disk and reading it back returns a Stream object with a single Trace: T).
    Therefore it would be nice to implement here all functions to accept Traces or Streams,
    implementing only the Trace processing because the Stream one is just a loop over its
    Traces.
    """
    def func_wrapper(obj, *args, **kwargs):
        if isinstance(obj, Stream):
            ret = []
            all_traces = True
            for trace in obj:
                ret_val = func(trace, *args, **kwargs)
                ret.append(ret_val)
                if all_traces and not isinstance(ret_val, Trace):
                    all_traces = False
            return Stream(ret) if all_traces else ret
        else:
            return func(obj, *args, **kwargs)
    return func_wrapper


def bandpass(trace, freq_min, freq_max, max_nyquist_ratio=0.9,
             corners=2, copy=True):
    """filters a signal trace. Wrapper around trace.filter in that it does some pre-processing
    before filtering
    :param trace: the input obspy.core.Trace
    :param magnitude: the magnitude which originated the trace (or stream). It dictates the value
    of the high-pass corner (the minimum frequency, freq_min, in Hz)
    :param freq_max: the value of the low-pass corner (freq_max), in Hz
    :param max_nyquist_ratio: the ratio of freq_max to be computed. The real low-pass corner will
    be set as max_nyquist_ratio * freq_max (default: 0.9, i.e. 90%)
    :param corners: the corners (i.e., the order of the filter)
    :return: the tuple (new_trace, fmin), where fmin is the minimum frequency set according to
    the given magnitude
    """
    tra = trace.copy() if copy is True else trace

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
    # according to Convers and Brady (1992)
    t_zpad = (1.5*corners)/freq_min
    endtime_remainder = tra.stats.endtime
    tra.trim(starttime=None, endtime=endtime_remainder+t_zpad, pad=True, fill_value=0)

    # 4) apply bandpass filter:
    tra.filter('bandpass', freqmin=freq_min, freqmax=freq_max, corners=corners, zerophase=True)

    # 5) remove padded elements:
    tra.trim(starttime=None, endtime=endtime_remainder)

    return tra


@stream_compliant
def maxabs(trace, starttime=None, endtime=None):
    """Returns the trace point
    ```(time, value)```
    where `value = max(abs(trace.data))`
    and time (`UTCDateTime`) is the time occurrence of `value`
    :param trace: the input obspy.core.Trace
    :param starttime: an obspy UTCDateTime object (or any value
    `UTCDateTime` accepts, e.g. integer / `datetime` object) denoting
    the start time (None or missing defaults to the trace end): the maximum of the trace `abs`
    will be searched *from* this time. This argument, if provided, does not affect the
    returned `time` which will be always relative to the trace passed as argument
    :param endtime: an obspy UTCDateTime object (or any value
    `UTCDateTime` accepts, e.g. integer / `datetime` object) denoting
    the end time (None or missing defaults to the trace end): the maximum of the trace `abs`
    will be searched *until* this time
    :return: the tuple (time, value) where `value = max(abs(trace.data))`, and time is
    the value occurrence (`UTCDateTime`)
    """
    original_stime = None if starttime is None else trace.stats.starttime
    if starttime is not None or endtime is not None:
        trace = trace.slice(starttime, endtime)
    if trace.stats.npts < 1:
        return np.nan
    idx = np.nanargmax(np.abs(trace.data))
    val = trace.data[idx]
    tdelta = 0 if original_stime is None else trace.stats.starttime - original_stime
    time = timeof(trace, idx) + tdelta
    return (time, val)


@stream_compliant
def cumsum(trace):
    """Returns the cumulative function, normalized between 0 and 1 of the argument
    :param trace: the input obspy.core.Trace
    :return: an obspy trace or stream (depending on the argument)
    """
    return Trace(_cumsum(trace.data, normalize=True), header=trace.stats.copy())


@stream_compliant
def cumtimes(cum_trace, *percentages):
    """Given cum_trace (a monotonically increasing trace, e.g. as resulting from `cumsum`),
    calculates the time(s) where the signal reaches the given percentage(s) of the total signal.
    Called P = len(percentages), returns a list of `len(percentages)` `obspy.UTCTimeStamp`s
    increasing items
    :param cum_trace: the input obspy.core.Trace (cumulative)
    :param percentages: the precentages to be calculated, e.g. 0.05, 0.95 (5% and 95%)
    :return: a list of length P = len(percentages) denoting the the obspy.UTCTimeStamp(s) where
    the given percentages occur
    """
    starttime = cum_trace.stats.starttime
    delta = cum_trace.stats.delta
    val = []
    minv = cum_trace[0]
    maxv = cum_trace[-1]
    for perc in percentages:
        idx = np.searchsorted(cum_trace.data, minv + (maxv - minv) * perc)
        val.append(starttime + idx * delta)
    return val


@stream_compliant
def fft(trace, starttime=None, endtime=None, taper_max_percentage=0.05, taper_type='hann',
        return_freqs=False):
    """Computes the fft of the given trace returning the relative numpy array `fft` as the second
    element the tuple
    ```(df, fft)```
    if `return_freqs=False` (df is the delta-frequency, as float), or
    ```(freqs, fft)```
    if `return_freqs=True` (`freqs` is linear space of frequencies, starting from 0, in Hz).
    This methods optionally trims the given trace, tapers it and then applies the fft
    :param trace: the input obspy.core.Trace
    :param starttime: the start time for trim, or None (=> starttime = trace start time)
    :param endtime: the end time for trim, or None (=> endtime = trace end time)
    :type taper_max_percentage: if non positive, no taper is applied on the (trimmed) trace before
    computing the fft. Otherwise, is a number between 0 and 1 to be passed to `trace.taper`
    :param taper_type: string, defaults to 'hann'. Ignored if no tapering is required
    (`taper_max_percentage<=0`)
    :param return_freqs: if False (the default) the first item of the returned tuple will be
    the delta frequency, otherwise the array of frequencies
    :return: a tuple where the second element if the numpy array of the fft values, and the first
    item is either the delta frequency (`return_freqs=False`) or the numpy array of the
    frequencies of the fft (`return_freqs=True`)
    """
    if starttime is not None or endtime is not None or taper_max_percentage > 0:
        trace = trace.copy()
    starttime, endtime = utcdatetime(starttime), utcdatetime(endtime)

    if starttime is not None or endtime is not None:
        trace.trim(starttime=starttime, endtime=endtime, pad=True, fill_value=0)
    if taper_max_percentage > 0:
        trace.taper(max_percentage=0.05, type=taper_type)
    dft = _fft(trace.data)
    if return_freqs:
        return freqs(trace.data, trace.stats.delta), dft
    else:
        return dfreq(trace.data, trace.stats.delta), dft


@stream_compliant
def ampspec(trace, starttime=None, endtime=None, taper_max_percentage=0.05, taper_type='hann',
            return_freqs=False):
    """Computes the amplitude spectrum of the given trace.
    See `fft`, the only difference is that the second element of the returned tuple is the
    amplitude spectrum"""
    _, dft = fft(trace, starttime, endtime, taper_max_percentage, taper_type, return_freqs)
    return _, _ampspec(dft, signal_is_fft=True)


@stream_compliant
def powspec(trace, starttime=None, endtime=None, taper_max_percentage=0.05, taper_type='hann',
            return_freqs=False):
    """Computes the power spectrum of the given trace.
    See `fft`, the only difference is that the second element of the returned tuple is the
    power spectrum"""
    _, dft = fft(trace, starttime, endtime, taper_max_percentage, taper_type, return_freqs)
    return _, _powspec(dft, signal_is_fft=True)


@stream_compliant
def get_tbounds(trace, fixed_time=None, window_in_sec=None):
    """Returns the bounds (start_time, end_time) of a given trace, fixed time and a window
    starting (if positive) or ending (if negative) at `fixed_time`
    The returned tuple can be used with obspy trace `trim` method
    :param trace: a given obspy trace
    :param fixed_time: an UTCDateTime denoting a fixed time. This will be the start time or
    end time of the resulting window (see `window_in_sec`). If None, the first element of the
    returned tuple is None, which obspy `trace.trim` considers as the trace start time
    :param window_in_sec: a float of seconds denoting the window length: if positive, then
    `fixed_time` will be the start time (first element) of the returned tuple. If negative, then
    `fixed_time` will be the end time of the returned tuple (seconds element). If None, then
    the second element of the returned tuple will be None, which obspy `trace.trim` considers as
    the trace end time
    :return: the tuple start_time, end_time denoting a time window (both arguments `UTCDateTime`)
    """
    if fixed_time is None:
        starttime = None
        endtime = None if window_in_sec is None else trace.stats.starttime + window_in_sec
    elif window_in_sec is None:
        starttime = fixed_time
        endtime = None
    else:
        t01 = fixed_time
        t02 = fixed_time + window_in_sec
        starttime, endtime = min(t01, t02), max(t01, t02)
    return starttime, endtime


@stream_compliant
def snr(trace, noisy_trace, fmin=None, fmax=None, nearest_sample=False, in_db=False):
    """Wrapper around `analysis.snr` for trace or streams
    :param trace: a given `obspy` Trace denoting the trace of the signal
    :param noisy_trace: a given `obspy` Trace denoting the trace of noise
    s"""
    return _snr(trace.data, noisy_trace.data, signals_form='', fmin=fmin, fmax=fmax,
                delta_signal=trace.stats.delta, delta_noise=noisy_trace.stats.delta,
                nearest_sample=nearest_sample, in_db=in_db)


@stream_compliant
def ampratio(trace, threshold=2**23):
    """Returns a list of numeric values (if the argument is a stream) or a single
    numeric value (if the argument is a single trace) representing the amplitude ratio given by:
        np.nanmax(np.abs(trace.data)) / threshold
    The trace has not be in physical units but in counts
    :param trace: a given obspy Trace
    :param threshold: float, defaults to `2 ** 23`: the denominator of the returned ratio
    """
    return np.true_divide(np.nanmax(np.abs(trace.data)), threshold)


def timeof(trace, index):
    """Returns a UTCDateTime object corresponding to the given index of the given trace
    the index does not need to be inside the trace indices, the method will return the time
    corresponding to that index anyway"""
    return trace.stats.starttime + index * trace.stats.delta


def utcdatetime(time, return_if_none=None):
    '''Convenience function to normalize any datetime object into UTCDateTime:
    converts `time` into obspy UTCDateTime, by returning `time` if already UTCDateTime or
    `UTCDateTime(time)` otherwise. If `time` is None, returns None by default (so that the returned
    value can be safely used when slicing/trimming such as, e.g. `trace.trim`), or any
    value supplied to the optional argument `return_if_none`
    :param time: a float, `datetime.datetime` object, or UtcDateTime. None is permitted and will
    return `return_if_none` (see below)
    :param return_if_none: None by default (when missing), indicates the value to return if
    `time` is None
    '''
    if not isinstance(time, UTCDateTime):
        time = return_if_none if time is None else UTCDateTime(time)
    return time
