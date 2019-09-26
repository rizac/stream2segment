'''
Math utilities for `obspy.Trace` objects.

This package wraps many functions of the :module:`stream2segment.math.ndarrays`
defining their counterparts for `Trace` objects

Remember that all functions processing and returning Traces, e.g.:
```
    func(trace, ...)  # returns a new Trace from trace
```
can be applied on a Stream `stream` easily:
```
    Stream([func(trace, ...) for trace in stream])` # returns a new Stream
```

:date: Jun 20, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>,
                  Graeme Weatherill <gweather@gfz-potsdam.de>
'''
from __future__ import division
import numpy as np

from obspy.core import Trace, UTCDateTime  # , Stats

# from obspy import read_inventory
from stream2segment.process.math.ndarrays import fft as _fft, ampspec as _ampspec,\
    powspec as _powspec, cumsumsq as _cumsum, dfreq, freqs, ResponseSpectrum as _ResponseSpectrum, \
    NewmarkBeta as _NewmarkBeta, NigamJennings as _NigamJennings


def _add_processing_info(trace, func_name, **kwargs):
    """
    This function attaches information about a processing call as a
    string to the Trace.stats.processing list. Copied (and simplified)
    from obspy Trace to give same consistent behaviour to the
    functions implemented in this module
    We do not use a decorator as it modifies the signature and names of this module functions
    and we don't want complex workaround for that.
    Call this function at the end of any function modifying a trace
    """

    # Attach after executing the function to avoid having it attached
    # while the operation failed.
    # Create info:
    info = "{package}.{function}(%s)".format(package=__name__, function=func_name)
    arguments = \
        ["%s=%s" % (k, repr(v)) if not isinstance(v, str) else
         "%s='%s'" % (k, v) for k, v in kwargs.items()]
    info = info % "::".join(arguments)
    # attach (copied from obspy):
    proc = trace.stats.setdefault('processing', [])
    proc.append(info)


def bandpass(trace, freq_min, freq_max, max_nyquist_ratio=0.9,
             corners=2, copy=True):
    """
    Filters a signal trace with a bandpass and other pre-processing.
    The algorithm steps are:
     1. Set the max frequency to 0.9 of the nyquist freauency (sampling rate /2)
        (slightly less than nyquist seems to avoid artifacts)
     2. Offset removal (subtract the mean from the signal)
     3. Tapering
     4. Pad data with zeros at the END in order to accommodate the filter transient
     5. Apply bandpass filter, where the lower frequency is set according to the magnitude
     6. Remove padded elements

    :param trace: the input obspy.core.Trace
    :param magnitude: the magnitude which originated the trace (or stream). It dictates the value
        of the high-pass corner (the minimum frequency, freq_min, in Hz)
    :param freq_max: the value of the low-pass corner (freq_max), in Hz
    :param max_nyquist_ratio: the ratio of freq_max to be computed. The real low-pass corner will
        be set as max_nyquist_ratio * freq_max (default: 0.9, i.e. 90%)
    :param corners: the corners (i.e., the order of the filter)

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
    """
    Returns the maximum of the absolute values of `trace`, and its occurrence time.
    In other words, returns the point `(time, value)` where `value = max(abs(trace.data))`
    and time (`UTCDateTime`) is the time occurrence of `value`

    :param trace: the input obspy.core.Trace
    :param starttime: (`obspy.UTCDateTime`) the start time (None or missing defaults to the trace
        end): the maximum of the trace `abs` will be searched *from* this time. This argument,
        if provided, does not affect the
        returned `time` which will be always relative to the trace passed as argument
    :param endtime: an obspy UTCDateTime object (or any value
        `UTCDateTime` accepts, e.g. integer / `datetime` object) denoting
        the end time (None or missing defaults to the trace end): the maximum of the trace `abs`
        will be searched *until* this time
        :return: the tuple (time, value) where `value = max(abs(trace.data))`, and time is
        the value occurrence (`UTCDateTime`)

    :return: the tuple `(time_of_max_abs, max_abs)`. If the trace has no point
        (possibly after providing `starttime` or `endtime` out of bounds), returns
        the tuple (None, numpy.nan)
    """
    original_stime = None if starttime is None else trace.stats.starttime
    if starttime is not None or endtime is not None:
        # from the docs: "this returns a New Trace object
        # Does not copy data but just passes a reference to it"
        trace = trace.slice(starttime, endtime)
    if trace.stats.npts < 1:
        return (None, np.nan)
    idx = np.nanargmax(np.abs(trace.data))
    val = trace.data[idx]
    tdelta = 0 if original_stime is None else trace.stats.starttime - original_stime
    time = timeof(trace, idx) + tdelta
    return (time, val)


def cumsumsq(trace, normalize=True, copy=True):
    """
    Returns the cumulative sum of the squares of the trace's data, `trace.data**2`

    :param trace: the input :class:`obspy.core.Trace`
    :param normalize: boolean (default: True), whether to normalize the data in [0, 1]
    :return: a Trace representing the cumulative sum of the square of `trace.data`
    """
    data = _cumsum(trace.data, normalize=normalize)
    if copy:
        trace = Trace(data, header=trace.stats.copy())
    else:
        trace.data = data
    # copied from obspy Trace to keep track of the modifications
    _add_processing_info(trace, cumsumsq.__name__, normalize=normalize, copy=copy)
    return trace


def timeswhere(mi_trace, *values):
    """
    Calculates the time(s) where `mi_trace` reaches the given value(s)
    **`mi_trace.data` need to be monotonically increasing**, e.g., as resulting from
    :func:`stream2segment.process.math.traces.cumsumsq`.

    :param mi_trace: a **monotonically increasing** trace
    :param values: the values whose time occurrence has to be calculated

    :return: a list of N `UtcDateTime`s (N = len(percentages)) denoting the occurrence of
        the given percentages of the total signal in `mi_trace`
    """
    starttime = mi_trace.stats.starttime
    delta = mi_trace.stats.delta
    tracedata = mi_trace.data
    return [starttime + delta * np.searchsorted(tracedata, v) for v in values]


def fft(trace, starttime=None, endtime=None, taper_max_percentage=0.05, taper_type='hann',
        return_freqs=False):
    """
    Computes the Fast Fourier transform of the given trace.
    If `return_freqs=False` (the default), returns the tuple
    ```df, fft```
    where `df` is the frequency resolution (in Hz). Otherwise, returns
    ```(freqs, fft)```
    where `freqs` is the frequencies vector (in Hz), evenly spaced with `df` as frequency
    resolution.
    This function optionally trims and tapers the given trace before applying the fft

    :param trace: the input obspy.core.Trace
    :param starttime: the start time for trim, or None (=> starttime = trace start time)
    :param endtime: the end time for trim, or None (=> endtime = trace end time)
    :param taper_max_percentage: if non positive, no taper is applied on the (trimmed) trace before
        computing the fft. Otherwise, is a number between 0 and 1 to be passed to `trace.taper`
    :param taper_type: string, defaults to 'hann'. Ignored if no tapering is required
        (`taper_max_percentage<=0`)
    :param return_freqs: if False (the default) the first item of the returned tuple will be
        the frequency resolution 'df', otherwise the array of frequencies

    :return: a tuple where the second element if the numpy array of the fft values, and the first
        item is either the frequency resolution (`return_freqs=False`) or the numpy array of the
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
    """
    Computes the amplitude spectrum of the given trace.
    See :func:`stream2segment.process.math.traces.fft` for info (this function does exactly the
    same, it only returns the amplitude spectrum as second element - i.e., the modulus of the fft)
    """
    _, dft = fft(trace, starttime, endtime, taper_max_percentage, taper_type, return_freqs)
    return _, _ampspec(dft, signal_is_fft=True)


def powspec(trace, starttime=None, endtime=None, taper_max_percentage=0.05, taper_type='hann',
            return_freqs=False):
    """
    Computes the power spectrum of the given trace.
    See :func:`stream2segment.process.math.traces.fft` for info (this function does exactly the
    same, it only returns the power spectrum as second element - i.e., the square of the fft)
    """
    _, dft = fft(trace, starttime, endtime, taper_max_percentage, taper_type, return_freqs)
    return _, _powspec(dft, signal_is_fft=True)


def ampratio(trace, threshold=2**23):
    """
    Returns the amplitude ratio given by:
        ```numpy.nanmax(numpy.abs(trace.data)) / threshold```
    The trace has not to be in physical units but in counts

    :param trace: a given obspy Trace
    :param threshold: float, defaults to `2 ** 23`: the denominator of the returned ratio

    :return: float indicating the amplitude ratio value
    """
    return np.true_divide(np.nanmax(np.abs(trace.data)), threshold)


def timeof(trace, index):
    """
    Returns the time occurrence of the `index`-th point of `trace`.
    Note that the index does not need to be inside the trace indices,
    the corresponding time will be computed anyway according to the trace sampling rate

    :param trace: an obspy Trace
    :param index: a numeric integer

    :return an `UTCDateTime` object corresponding to the time of the `inde`-th point of `trace`
    """
    return trace.stats.starttime + index * trace.stats.delta


def utcdatetime(time, return_if_none=None):
    """
    Normalizes `time` into an `UTCDateTime`. Utility function for working consistently
    with different date-time-like inputs and convert them to the same object type.

    :param time: numeric (int, float), `datetime.datetime` object, `UtcDateTime`. If `UtcDateTime`,
        then `time` is returned with no processing. If None, then None (or `return_if_none`, if
        supplied) is returned. Otherwise, `UTCDateTime(time)` is returned
        (see :class:`obspy.core.utcdatetime.UTCDateTime` for info).
    :param return_if_none: None by default (when missing), indicates the value to return if
    `    time` is None

    :return: an :class:`obspy.core.utcdatetime.UTCDateTime` from the given time argument
    """
    if not isinstance(time, UTCDateTime):
        time = return_if_none if time is None else UTCDateTime(time)
    return time


class ResponseSpectrum(_ResponseSpectrum):
    """
    Base abstract Class to implement a response spectrum calculation for :class:`obspy.Trace`s
    """
    def __init__(self, acc_trace, periods, damping=0.05, units="cm/s/s"):
        '''RemoveResponse base class operating on :class:`obspy.Trace`s. When not documented,
        parameters are the same of :class:`stream2segment.process.math.ndarrays.ResponseSpectrum`

        :param acc_trace: a Trace in acceleration units, obtained via, e.g.:
            ```
                acc_trace = trace.remove_response(..., output="ACC", ...)
            ```
        '''
        super(ResponseSpectrum, self).__init__(acc_trace.data, acc_trace.stats.delta,
                                               periods, damping, units)


class NigamJennings(ResponseSpectrum, _NigamJennings):
    """
    Evaluate the response spectrum using the algorithm of Nigam & Jennings
    (1969) on for :class:`obspy.Trace`s objects.
    In general this is faster than the classical Newmark-Beta method, and
    can provide estimates of the spectra at frequencies higher than that
    of the sampling frequency.
    """
    pass


class NewmarkBeta(ResponseSpectrum, _NewmarkBeta):
    """
    Evaluates the response spectrum using the Newmark-Beta methodology
    for :class:`obspy.Trace`s objects.
    """
    pass

# define a global variable for use with the function below:
# note that isinstance(c, type) returns if v is a class but works for new-style classes
# which as of end 2017 is not anymore a restriction
_rs = {c.lower(): v for c, v in globals().items() if isinstance(v, type) and
       issubclass(v, ResponseSpectrum) and v != ResponseSpectrum}


def respspec(method, acc_trace, periods, damping=0.05):
    """
    Evaluates the response spectrum within a single function

    :param method: a string denoting the method. Currently supported are:
        'NewmarkBeta' and 'NigamJennings' (`method` is case-insensitive so you can input also
        lower-case strings). See relative module classes for details. 'NigamJennings' is in
        general faster than the classical Newmark-Beta method, and can provide estimates of the
        spectra at frequencies higher than that of the sampling frequency.
    :param acc_trace: a Trace in acceleration units, obtained via, e.g.:
        ```
            acc_trace = trace.remove_response(..., output="ACC", ...)
        ```
    :param time_step: the sampling period (delta t) of `acceleration`
    :param periods: (numpy.ndarray) Spectral periods (s) for calculation
    :param damping: float (default=0.05) Fractional coefficient of damping

    :returns:
        Response Spectrum - Dictionary containing all response spectrum
                            data. All units depend to the passed `acc_trace` array.
                            Use `class`:ResponseSpectrum.acc2cms2 to convert to
                            cm per second squared, if needed
            'Time' - Time
            'Acceleration' - Acceleration Response Spectrum
            'Velocity' - Velocity Response Spectrum
            'Displacement' - Displacement Response Spectrum
            'Pseudo-Velocity' - Pseudo-Velocity Response Spectrum
            'Pseudo-Acceleration' - Pseudo-Acceleration Response Spectrum

        Time Series - Dictionary containing all time-series data
            'Time' - Time (s)
            'Acceleration' - Acceleration time series
            'Velocity' - Velocity time series
            'Displacement' - Displacement time series
            'PGA' - Peak ground acceleration
            'PGV' - Peak ground velocity
            'PGD' - Peak ground displacement

        accel - Acceleration response of Single Degree of Freedom Oscillator
        vel - Velocity response of Single Degree of Freedom Oscillator
        disp - Displacement response of Single Degree of Freedom Oscillator
    """
    try:
        rs_class = _rs[method.lower()]
    except KeyError:
        raise TypeError('Please supply a response spectrum method in %s' %
                        list(_rs.keys()))
    return rs_class(acc_trace, periods, damping).evaluate()
