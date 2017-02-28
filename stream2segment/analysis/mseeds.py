'''
Created on Jun 20, 2016

@author: riccardo
'''
import numpy as np

# from scipy.signal import savgol_filter
try:
    import cPickle as pickle
except ImportError:
    import pickle
from obspy.core import Stream, Trace, UTCDateTime  # , Stats
from obspy import read_inventory
from stream2segment.analysis import fft as _fft, maxabs as _maxabs,\
    snr as _snr, cumsum as _cumsum, env as _env, dfreq as _dfreq  # , pow_spec, amp_spec


def stream_compliant(func):
    """
        Function decorator which allows the function decorated to accept either
        obspy.Trace or an obspy.Stream objects, and handling the returned output consistently.
        Rationale: A Trace is the obspy core object representing a timeseries. Therefore, in
        principle all processing functions, like those defined here, should work on traces.
        However, obspy provides also Stream objects (basically, collections of Traces)
        *which represent the miniSEED file stored on disk* (writing e.g. a Trace T
        to disk and reading it back returns a Stream object with a single Trace: T).
        Therefore if would be nice to implement here all functions to accept Traces or Streams,
        implementing only the Trace processing because the Stream one is just a loop over its
        Traces.
        After implementing a function func processing a trace, and decorating it like this:
            \@stream_compliant
            def func(trace,...)
        then this decoraor takes care of the rest: it takes the `func` and wraps it creating a
        wrapper function W: if W argument is a Trace, then W calls and returns `func` with the
        trace as argument. On the other hand, if W argument is a Stream, iterates over its traces
        and calls `func` on all of them. Then, if all the objects returned by `func` are again
        Traces, **returns a Stream wrapping the returned traces**, otherwise **returns a list
        of the returned objects**.
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


def itertrace(trace_or_stream):
    """Iterator over the argument. If the latter is a Stream, returns it. If it is a trace
    returns the argument wrapped into a list"""
    return trace_or_stream if isinstance(trace_or_stream, Stream) else [trace_or_stream]


@stream_compliant
def bandpass(trace, magnitude, freq_max=20, max_nyquist_ratio=0.9,
             corners=2, copy=True):
    """filters a signal trace.
    :param trace: the input obspy.core.Trace
    :param magnitude: the magnitude which originated the trace (or stream). It dictates the value
    of the high-pass corner (the minimum frequency, freq_min, in Hz)
    :param freq_max: the value of the low-pass corner (freq_max), in Hz
    :param max_nyquist_ratio: the ratio of freq_max to be computed. The real low-pass corner will
    be set as max_nyquist_ratio * freq_max (default: 0.9, i.e. 90%)
    :param corners: the corners (i.e., the order of the filter)
    """
    tra = trace.copy() if copy is True else trace

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

    return tra


@stream_compliant
def maxabs(trace, starttime=None, endtime=None):
    """
        Returns the maxima of the absolute values of the trace or stream object passed as first
        argument. The returned value is the tuple:
            (time, val)
        if trace is an obspy.Trace, or the list:
            [(time1, va1), ... (timeN, valN)]
        if trace is an obspy.Stream
        All times are UTCDateTime objects
        :param trace: the input obspy.core.Trace
        :param starttime: an obspy UTCDateTime object (or integer, denoting a timestamp) denoting
        the start time of the trace, or None to default to the trace start
        :param endtime: an obspy UTCDateTime object (or integer, denoting a timestamp) denoting
        the start time of the trace, or None to default to the trace end
    """
    tra = trace.slice(starttime, endtime)
    idx, val = _maxabs(tra.data)
    time = timeof(tra, idx)
    return (time, val)


@stream_compliant
def remove_response(trace, inventory_or_inventory_path, output='ACC', water_level=60):
    """
        Removes the response from the trace (or stream) passed as first argument. Calls
        obspy.Stream.remove_response (or obspy.Trace.remove_response, depending on the first
        argument) without pre-filtering. THEREFORE, IF THE SIGNAL HAS TO BE FILTERED ONE SHOULD
        PERFORM THE FILTER PRIOR TO THIS FUNCTION, OR CALL obspy.Stream.remove_response
        (obspy.Trace.remove_response) which offer more fine parameter tuning
        :param trace: the input obspy.core.Trace
        :param inventory: either an inventory object, or an (absolute) path to a specified
        inventory xml file
    """
    trace = trace.copy()
    inventory = read_inventory(inventory_or_inventory_path) \
        if isinstance(inventory_or_inventory_path, basestring) else inventory_or_inventory_path

    trace.remove_response(inventory=inventory, output=output, water_level=water_level)

    return trace


PAZ_WA = {'sensitivity': 2800, 'zeros': [0j], 'gain': 1,
          'poles': [-6.2832 - 4.7124j, -6.2832 + 4.7124j]}


@stream_compliant
def simulate_wa(trace, inventory_or_inventory_path=None, water_level=60):
    """
        Simulates a syntetic wood anderson recording by returning a new Stream (or Trace) object
        (according to the argument)
        :param trace: the input obspy.core.Trace
        :param inventory: either an inventory object, or an (absolute) path to a specified inventory
        xml file. IMPORTANT: IF NOT SUPPLIED (OR NONE) THE FUNCTION ASSUMES THAT THE RESPONSE HAS
        BEEN ALREADY REMOVED (WITH OUTPUT='DISP') FROM THE INPUT TRACE (OR STREAM) SUPPLIED AS FIRST
        ARGUMENT
        :param water_level: ignored if `inventory_or_inventory_path` is None or missing, is the
        water level argument to be passed to remove_response
    """
    if inventory_or_inventory_path is not None:
        trace = remove_response(trace, inventory_or_inventory_path, 'DISP', water_level=water_level)
    else:
        trace = trace.copy()

    trace.simulate(paz_remove=None, paz_simulate=PAZ_WA)
    return trace


def get_gaps(trace_or_stream):
    """
        Returns a list of gaps for the current argument. The list elements have the form:
            [network, station, location, channel, starttime of the gap, end time of the gap,
             duration of the gap, number of missing samples]
        :param trace_or_stream: a Trace, or a Stream (note: due to the fact that obspy get_gaps is
        Stream only, if this argument is a trace this function returns an empty list)
        :return: a list of gaps
        :rtype: list of lists
    """
    try:
        return trace_or_stream.get_gaps()
    except AttributeError:  # is a Trace
        return []


@stream_compliant
def cumsum(trace):
    """
    Returns the cumulative function, normalized between 0 and 1 of the argument
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
    :param: percentages: the precentages to be calculated, e.g. 0.05, 0.95 (5% and 95%)
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
def env(trace):
    """
    Returns the envelope (using scipy hilbert transform) of the argument
    :param trace: the input obspy.core.Trace
    :return: an obspy trace or stream (depending on the argument)
    """
    return Trace(_env(trace.data), header=trace.stats.copy())


@stream_compliant
def fft(trace, fixed_time=None, window_in_sec=None, taper_max_percentage=0.05, taper_type='hann'):
    """
    Returns a trace T resulting from applying the fft on `trace`. The resulting trace
    has **complex** values and thus can only be saved with `trace.write(..., format='PICKLE')`
    The `T.stats` attribute is copied from `trace`, thus referencing the source
    trace. This way, you can call e.g., `D=dfreq(T)` to return the delta frequency of T for building
    the frequency (x axis) values [0, D, 2*D, ...]
    :param trace: the input obspy.core.Trace
    :param fixed_time: the fixed time where to set the start (if `window_in_sec` > 0) or end
    (if `window_in_sec` < 0) of the trace slice on which to apply the fft. If None, it defaults
    to the start of each trace
    :type fixed_time: an `obspy.UTCDateTime` object or any object that can be passed as argument
    to the latter (e.g., a numeric timestamp)
    :param window_in_sec: the window, in sec, of the trace slice where to apply the fft. If None,
    it defaults to the amount of time from `fixed_time` till the end of each trace
    :type window_in_sec: numeric
    :return: an obspy.Trace with *complex* values in trace.data
    """
    fixed_time = utcdatetime(fixed_time)

    tra = trace.copy()
    if fixed_time is None:
        starttime = None
        endtime = None if window_in_sec is None else tra.stats.starttime + window_in_sec
    elif window_in_sec is None:
        starttime = fixed_time
        endtime = None
    else:
        t01 = fixed_time
        t02 = fixed_time + window_in_sec
        starttime, endtime = min(t01, t02), max(t01, t02)

    trim_tra = tra.trim(starttime=starttime, endtime=endtime, pad=True, fill_value=0)
    if taper_max_percentage > 0:
        trim_tra.taper(max_percentage=0.05, type=taper_type)
    dft = _fft(trim_tra.data)
    # remember: now we have a numpy array of complex numbers
    # build a Trace object with complex values and stats referring to the **original** trace
    # then we can use trace.write(format='pickle') to save the trace and hopefully have a robust
    # way to retrieve it back via obspy.read. Implementing subclasses of Trace is less robust
    # cause we need maintainance if we change to the sub-classes in the future, this way we
    # completely delegate obpsy.
    t = Trace(data=dft, header=trim_tra.stats)  # stats are preserved. The only stats changed is
    # when we set the data attribute on a trace (not via the constructor, like we just did)
    t.stats._format = 'PICKLE'  # pylint:disable=protected-access
    # this gives an hint on the format to be saved to
    # note that tt.stats.processing is a list of ALL processing applied on a given Trace,
    # so we have also the history of the stuff done (only for Trace class methods, but should be
    # enough)

    # t.stats.mseed.encoding = ?  # FXIME: see obspy encodings (float64? float32?)
    # if given, suppress warnings when saving (raised if data has unsupported
    # encoding, e.g. is complex).From obspy doc: The ``reclen``, ``encoding``, ``byteorder`` and
    # ``sequence_count``
    # keyword arguments can be set in the ``stats.mseed`` of
    # each :class:`~obspy.core.trace.Trace` as well as ``kwargs`` of this
    # function. If both are given the ``kwargs`` will be used.
    return t


@stream_compliant
def dfreq(trace):
    """Returns the delta frequencies"""
    return _dfreq(trace.data, trace.stats.delta)


@stream_compliant
def snr(trace, noisy_trace, signals_form='normal', in_db=False):
    """
    Returns the signal to noise ratio of trace over noisy_trace.
    :param trace: a given `obspy` Trace denoting the divisor of the snr
    :param noisy_trace: a given `obspy` Trace denoting the dividend of the snr
    :param signals_form: tells this function what the given traces are. If:
        - 'fft' or 'dft': then the traces where obtained from the `fft` function of this module,
            and their data are actually discrete Fourier transforms
        - any other value: then the traces are time series, their amplitude spectra will be
            computed before returning the snr.
    """
    return _snr(trace.data, noisy_trace.data, signals_form, in_db)


@stream_compliant
def amp_ratio(trace):
    """Returns a list of numeric values (if the argument is a stream) or a single
    numeric value (if the argument is a single trace) representing the amplitude ratio given by:
        np.nanmax(np.abs(trace.data)) / 2 ** 23
    Obviously, the trace has not be in physical units but in counts
    """
    threshold = 2 ** 23
    return np.true_divide(np.nanmax(np.abs(trace.data)), threshold)


@stream_compliant
def timearray(trace, npts=None):
    """
        Returns the x values of the given trace according to each trace stats
        if npts is not None, returns a linear space of equally sampled x from
        tra.starttime and tra.endtime. Otherwise, npts equals the trace number of points
    """
    num = len(trace.data) if npts is None else npts  # we don't trust tra.stats.npts
    return np.linspace(trace.stats.starttime.timestamp, trace.stats.endtime.timestamp, num=num,
                       endpoint=True)


def interpolate(trace_or_stream, npts_or_new_x_array, align_if_stream=True,
                return_x_array=False):
    """Returns a trace or stream interpolated with the given number of points. This method
    differs from obspy.Trace.interpolate in that it offers a probably faster but less fine-tuned
    way to (linearly) interpolate a Trace or a Stream (and in this case optionally align its Traces
    on the same time range, if needed). This method is intended to optimize visualizations
    of the data, rather than performing calculation on it
    :param trace: the input obspy.core.Trace
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


@stream_compliant
def get_multievent(cum_trace, tmin, tmax,
                   threshold_inside_tmin_tmax_percent,
                   threshold_inside_tmin_tmax_sec, threshold_after_tmax_percent):
    """
        Returns the tuple (or a list of tuples, if the first argument is a stream) of the
        values (score, UTCDateTime of arrival)
        where scores is: 0: no double event, 1: double event inside tmin_tmax,
            2: double event after tmax, 3: both double event previously defined are detected
        If score is 2 or 3, the second argument is the UTCDateTime denoting the occurrence of the
        first sample triggering the double event after tmax
        :param trace: the input obspy.core.Trace
    """
    tmin = utcdatetime(tmin)
    tmax = utcdatetime(tmax)

    double_event_after_tmax_time = None
    d_order = 2

    # split traces between tmin and tmax and after tmax
    traces = [cum_trace.slice(tmin, tmax), cum_trace.slice(tmax, None)]

    # calculate second derivative and normalize:
    derivs = []
    max_ = None
    for ttt in traces:
        sec_der = np.diff(ttt.data, n=d_order)
        _, mmm = _maxabs(sec_der)
        max_ = np.nanmax([max_, mmm])  # get max (global) for normalization (see below):
        derivs.append(sec_der)

    # normalize second derivatives:
    for der in derivs:
        der /= max_

    result = 0

    # case A: see if after tmax we exceed a threshold
    indices = np.where(derivs[1] >= threshold_after_tmax_percent)[0]

    if len(indices):
        result = 2
        double_event_after_tmax_time = timeof(traces[1], indices[0])

    # case B: see if inside tmin tmax we exceed a threshold, and in case check the duration
    indices = np.where(derivs[0] >= threshold_inside_tmin_tmax_percent)[0]
    if len(indices) >= 2:
        idx0 = indices[0]
        idx1 = indices[-1]

        deltatime = (idx1 - idx0) * cum_trace.stats.delta

        if deltatime >= threshold_inside_tmin_tmax_sec:
            result += 1

    return (result, double_event_after_tmax_time)


def timeof(trace, index):
    """Returns a UTCDateTime object corresponding to the given index of the given trace
    the index does not need to be inside the trace indices, the method will return the time
    corresponding to that index anyway"""
    return trace.stats.starttime + index * trace.stats.delta


def utcdatetime(time, return_if_none=None):
    if not isinstance(time, UTCDateTime):
        time = return_if_none if time is None else UTCDateTime(time)
    return time
