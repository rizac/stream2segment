'''
Math utilities for `obspy.Trace` objects.

Remember that all functions processing and returning `Trace`s, e.g.:
```
    new_trace = func(trace, ...)
```
can be applied on a `Stream` easily:
```
    new_stream = Stream([func(trace, ...) for trace in stream])`
```

This package wraps many functions of `analysis` defining their counterparts
for `Trace` objects

:date: Jun 20, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
from __future__ import division

import numpy as np

from obspy.core import Stream, Trace, UTCDateTime  # , Stats
# from obspy import read_inventory
from stream2segment.analysis import fft as _fft, ampspec as _ampspec, powspec as _powspec,\
    cumsum as _cumsum, dfreq, freqs


__all__ = ['bandpass']


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
    :return: a Trace with the bandpass algorithm applied
    """
    tra = trace.copy() if copy is True else trace

    # define sampling freq
    sampling_rate = tra.stats.sampling_rate
    # adjust the max_f_max to 0.9 of the nyquist frea (sampling rate /2)
    # slightly less than nyquist (0.9) seems to avoid artifacts
    max_f_max = max_nyquist_ratio * (sampling_rate / 2)
    freq_max = min(freq_max, max_f_max)

    # Start filtering (several pre-steps)
    # 1) offset removal:
    tra.data = tra.data - np.nanmean(tra.data)

    # 2) tapering
    tra.taper(type='cosine', max_percentage=0.05)

    # 3) pad data with zeros at the END in order to filter transient
    # according to Convers and Brady (1992)
    t_zpad = 1.5 * corners / freq_min
    endtime_remainder = tra.stats.endtime
    tra.trim(starttime=None, endtime=endtime_remainder+t_zpad, pad=True, fill_value=0)

    # 4) apply bandpass filter:
    tra.filter('bandpass', freqmin=freq_min, freqmax=freq_max, corners=corners, zerophase=True)

    # 5) remove padded elements:
    tra.trim(starttime=None, endtime=endtime_remainder)

    return tra


def maxabs(trace, starttime=None, endtime=None):
    """Returns the trace point `(time, value)`
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
    :return: the tuple `(time_of_max_abs, max_abs)`
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


def cumsum(trace):
    """Returns the cumulative sum of `trace.data**2`, normalized between 0 and 1
    :param trace: the input obspy.core.Trace
    :return: a new Trace representing the cumulative sum of the square of `trace.data`
    """
    return Trace(_cumsum(trace.data, normalize=True), header=trace.stats.copy())


def cumtimes(cum_trace, *percentages):
    """Given cum_trace (a monotonically increasing trace, e.g. as resulting from `cumsum`),
    calculates the time(s) where the signal reaches the given percentage(s) of the total signal.
    Called N = `len(percentages)`, returns a list of N `obspy.UTCTimeStamp`s objects
    :param cum_trace: the input obspy.core.Trace (cumulative)
    :param percentages: the precentages to be calculated, e.g. 0.05, 0.95 (5% and 95%)
    :return: a list of length P = len(percentages) denoting the the obspy.UTCTimeStamp(s) where
    the given percentages occur
    :return: a list of `UtcDateTime's denoting the occurrence of the given percentages of the total
    signal in `cum_trace`
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


def ampspec(trace, starttime=None, endtime=None, taper_max_percentage=0.05, taper_type='hann',
            return_freqs=False):
    """Computes the amplitude spectrum of the given trace.
    Same as `fft`, but returns the amplitude spectrum as second element instead of the fft"""
    _, dft = fft(trace, starttime, endtime, taper_max_percentage, taper_type, return_freqs)
    return _, _ampspec(dft, signal_is_fft=True)


def powspec(trace, starttime=None, endtime=None, taper_max_percentage=0.05, taper_type='hann',
            return_freqs=False):
    """Computes the power spectrum of the given trace.
    Same as `fft`, but returns the amplitude spectrum as second element instead of the fft"""
    _, dft = fft(trace, starttime, endtime, taper_max_percentage, taper_type, return_freqs)
    return _, _powspec(dft, signal_is_fft=True)


def ampratio(trace, threshold=2**23):
    """Returns the amplitude ratio given by:
        np.nanmax(np.abs(trace.data)) / threshold
    The trace has not to be in physical units but in counts
    :param trace: a given obspy Trace
    :param threshold: float, defaults to `2 ** 23`: the denominator of the returned ratio
    :return: float indicating the amplitude ratio value
    """
    return np.true_divide(np.nanmax(np.abs(trace.data)), threshold)


def timeof(trace, index):
    """Returns an `UTCDateTime` object corresponding to the time occurrence of the
    `index`-th point of `trace`. Note that the index does not need to be inside the trace indices,
    the corresponding time will be computed anyway according to the trace sampling rate"""
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
