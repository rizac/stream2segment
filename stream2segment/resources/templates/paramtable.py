'''
This editable template shows how to generate a segment-based parametric table.
When used for visualization, this template implements a pre-processing function that
can be toggled with a checkbox, and several task-specific functions (e.g., cumulative,
synthetic Wood-Anderson) that can to be visualized as custom plots in the GUI

{{ PROCESS_PY_MAIN }}
'''

from __future__ import division

# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports). UNCOMMENT or REMOVE
# if you are working in Python3 (recommended):
from builtins import (ascii, bytes, chr, dict, filter, hex, input,
                      int, map, next, oct, open, pow, range, round,
                      str, super, zip)

# From Python >= 3.6, dicts keys are returned (and thus, written to file) in the order they
# are inserted. Prior to that version, to preserve insertion order you needed to use OrderedDict:
from collections import OrderedDict
from datetime import datetime, timedelta  # always useful
from math import factorial  # for savitzky_golay function

# import numpy for efficient computation:
import numpy as np
# import obspy core classes (when working with times, use obspy UTCDateTime when possible):
from obspy import Trace, Stream, UTCDateTime
from obspy.geodetics import degrees2kilometers as d2km
# decorators needed to setup this module @gui.preprocess @gui.plot:
from stream2segment.process import gui
# strem2segment functions for processing obspy Traces. This is just a list of possible functions
# to show how to import them:
from stream2segment.process.math.traces import ampratio, bandpass, cumsumsq,\
    timeswhere, fft, maxabs, utcdatetime, ampspec, powspec, timeof, sn_split
# stream2segment function for processing numpy arrays:
from stream2segment.process.math.ndarrays import triangsmooth, snr


def main(segment, config):
    """{{ PROCESS_PY_MAINFUNC | indent }}
    """
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    trace = stream[0]  # work with the (surely) one trace now

    # discard saturated signals (according to the threshold set in the config file):
    amp_ratio = ampratio(trace)
    if amp_ratio >= config['amp_ratio_threshold']:
        raise ValueError('possibly saturated (amp. ratio exceeds)')

    # bandpass the trace, according to the event magnitude.
    # WARNING: this modifies the segment.stream() permanently!
    # If you want to preserve the original stream, store trace.copy() beforehand.
    # Also, use a 'try catch': sometimes Inventories are corrupted and obspy raises
    # a TypeError, which would break the WHOLE processing execution.
    # Raising a ValueError will stop the execution of the currently processed
    # segment only (logging the error message):
    try:
        trace = bandpass_remresp(segment, config)
    except TypeError as type_error:
        raise ValueError("Error in 'bandpass_remresp': %s" % str(type_error))

    spectra = signal_noise_spectra(segment, config)
    normal_f0, normal_df, normal_spe = spectra['Signal']
    noise_f0, noise_df, noise_spe = spectra['Noise']
    evt = segment.event
    fcmin = mag2freq(evt.magnitude)
    fcmax = config['preprocess']['bandpass_freq_max']  # used in bandpass_remresp
    snr_ = snr(normal_spe, noise_spe, signals_form=config['sn_spectra']['type'],
               fmin=fcmin, fmax=fcmax, delta_signal=normal_df, delta_noise=noise_df)
    snr1_ = snr(normal_spe, noise_spe, signals_form=config['sn_spectra']['type'],
                fmin=fcmin, fmax=1, delta_signal=normal_df, delta_noise=noise_df)
    snr2_ = snr(normal_spe, noise_spe, signals_form=config['sn_spectra']['type'],
                fmin=1, fmax=10, delta_signal=normal_df, delta_noise=noise_df)
    snr3_ = snr(normal_spe, noise_spe, signals_form=config['sn_spectra']['type'],
                fmin=10, fmax=fcmax, delta_signal=normal_df, delta_noise=noise_df)
    if snr_ < config['snr_threshold']:
        raise ValueError('low snr %f' % snr_)

    # calculate cumulative

    cum_labels = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    cum_trace = cumsumsq(trace, normalize=True, copy=True)  #  copy=True prevent original trace from being modified
    cum_times = timeswhere(cum_trace, *cum_labels)

    # double event
    pstart = 1
    pend = -2
    pref = 3
    (score, t_double, tt1, tt2) = \
        get_multievent_sg(cum_trace, cum_times[pstart], cum_times[pend], cum_times[pref],
                          config['threshold_inside_tmin_tmax_percent'],
                          config['threshold_inside_tmin_tmax_sec'],
                          config['threshold_after_tmax_percent'])
    if score in {1, 3}:
        raise ValueError('Double event detected %d %s %s %s' % (score, t_double, tt1, tt2))

    # calculate PGA and times of occurrence (t_PGA):
    # note: you can also provide tstart tend for slicing
    t_PGA, PGA = maxabs(trace, cum_times[1], cum_times[-2])
    trace_int = trace.copy()
    trace_int.integrate()
    t_PGV, PGV = maxabs(trace_int, cum_times[1], cum_times[-2])
    meanoff = meanslice(trace_int, 100, cum_times[-1], trace_int.stats.endtime)

    # calculates amplitudes at the frequency bins given in the config file:
    required_freqs = config['freqs_interp']
    ampspec_freqs = normal_f0 + np.arange(len(normal_spe)) * normal_df
    required_amplitudes = np.interp(np.log10(required_freqs),
                                    np.log10(ampspec_freqs), normal_spe) / segment.sample_rate

    # compute synthetic WA.
    trace_wa = synth_wood_anderson(segment, config, trace.copy())
    t_WA, maxWA = maxabs(trace_wa)

    # write stuff to csv:
    ret = OrderedDict()

    ret['snr'] = snr_
    ret['snr1'] = snr1_
    ret['snr2'] = snr2_
    ret['snr3'] = snr3_
    for cum_lbl, cum_t in zip(cum_labels[slice(1, 8, 3)], cum_times[slice(1, 8, 3)]):
        ret['cum_t%f' % cum_lbl] = float(cum_t)  # convert cum_times to float for saving

    ret['dist_deg'] = segment.event_distance_deg        # dist
    ret['dist_km'] = d2km(segment.event_distance_deg)  # dist_km
    # t_PGA is a obspy UTCDateTime. This type is not supported in HDF output, thus
    # convert it to Python datetime. Note that in CSV output, the value will be written as
    # str(t_PGA.datetime): another option might be to store it as string
    # with str(t_PGA) (returns the iso-formatted string, supported in all output formats):
    ret['t_PGA'] = t_PGA.datetime  # peak info
    ret['PGA'] = PGA
    # (for t_PGV, see note above for t_PGA)
    ret['t_PGV'] = t_PGV.datetime  # peak info
    ret['PGV'] = PGV
    # (for t_WA, see note above for t_PGA)
    ret['t_WA'] = t_WA.datetime
    ret['maxWA'] = maxWA
    ret['channel'] = segment.channel.channel
    ret['channel_component'] = segment.channel.channel[-1]
    # event metadata:
    ret['ev_id'] = segment.event.id
    ret['ev_lat'] = segment.event.latitude
    ret['ev_lon'] = segment.event.longitude
    ret['ev_dep'] = segment.event.depth_km
    ret['ev_mag'] = segment.event.magnitude
    ret['ev_mty'] = segment.event.mag_type
    # station metadata:
    ret['st_id'] = segment.station.id
    ret['st_name'] = segment.station.station
    ret['st_net'] = segment.station.network
    ret['st_lat'] = segment.station.latitude
    ret['st_lon'] = segment.station.longitude
    ret['st_ele'] = segment.station.elevation
    ret['score'] = score
    ret['d2max'] = float(tt1)
    ret['offset'] = np.abs(meanoff/PGV)
    for freq, amp in zip(required_freqs, required_amplitudes):
        ret['f_%.5f' % freq] = float(amp)

    return ret


def assert1trace(stream):
    '''asserts the stream has only one trace, raising an Exception if it's not the case,
    as this is the pre-condition for all processing functions implemented here.
    Note that, due to the way we download data, a stream with more than one trace his
    most likely due to gaps / overlaps'''
    # stream.get_gaps() is slower as it does more than checking the stream length
    if len(stream) != 1:
        raise ValueError("%d traces (probably gaps/overlaps)" % len(stream))


@gui.preprocess
def bandpass_remresp(segment, config):
    """{{ PROCESS_PY_BANDPASSFUNC | indent }}
    """
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    trace = stream[0]

    inventory = segment.inventory()

    # define some parameters:
    evt = segment.event
    conf = config['preprocess']
    # note: bandpass here below copied the trace! important!
    trace = bandpass(trace, mag2freq(evt.magnitude), freq_max=conf['bandpass_freq_max'],
                     max_nyquist_ratio=conf['bandpass_max_nyquist_ratio'],
                     corners=conf['bandpass_corners'], copy=False)
    trace.remove_response(inventory=inventory, output=conf['remove_response_output'],
                          water_level=conf['remove_response_water_level'])
    return trace


def mag2freq(magnitude):
    '''returns a magnitude dependent frequency (in Hz)'''
    if magnitude <= 4.5:
        freq_min = 0.4
    elif magnitude <= 5.5:
        freq_min = 0.2
    elif magnitude <= 6.5:
        freq_min = 0.1
    else:
        freq_min = 0.05
    return freq_min


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise TypeError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size-1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def get_multievent_sg(cum_trace, tmin, tmax, tstart,
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
    tstart = utcdatetime(tstart)

    # split traces between tmin and tmax and after tmax
    traces = [cum_trace.slice(tmin, tmax), cum_trace.slice(tmax, None)]

    # calculate second derivative and normalize:
    second_derivs = []
    max_ = np.nan
    for ttt in traces:
        ttt.taper(type='cosine', max_percentage=0.05)
        sec_der = savitzky_golay(ttt.data, 31, 2, deriv=2)
        sec_der_abs = np.abs(sec_der)
        idx = np.nanargmax(sec_der_abs)
        # get max (global) for normalization:
        max_ = np.nanmax([max_, sec_der_abs[idx]])
        second_derivs.append(sec_der_abs)

    # normalize second derivatives:
    for der in second_derivs:
        der /= max_

    result = 0

    # case A: see if after tmax we exceed a threshold
    indices = np.where(second_derivs[1] >= threshold_after_tmax_percent)[0]
    if len(indices):
        result = 2
    #   starttime = timeof(traces[0], indices[0])

    # case B: see if inside tmin tmax we exceed a threshold, and in case check the duration
    deltatime = 0
    starttime = tmin
    endtime = None
    indices = np.where(second_derivs[0] >= threshold_inside_tmin_tmax_percent)[0]
    if len(indices) >= 2:
        idx0 = indices[0]
        starttime = timeof(traces[0], idx0)
        idx1 = indices[-1]
        endtime = timeof(traces[0], idx1)
        deltatime = endtime - starttime
        if deltatime >= threshold_inside_tmin_tmax_sec:
            result += 1

    return result, deltatime, starttime, endtime


def synth_wood_anderson(segment, config, trace):
    '''
    Low-level function to calculate the synthetic wood-anderson of `trace`.
    The dict ``config['simulate_wa']`` must be implemented
    and houses the wood-anderson configuration 'sensitivity', 'zeros', 'poles' and 'gain'.
    Modifies the trace in place.

    :param trace_input_type:
        None: trace is unprocessed and trace.remove_response(.. output="DISP"...)
            will be applied on it before applying `trace.simulate`
        'ACC': trace is already processed, e.g..
            it's the output of trace.remove_response(..output='ACC')
        'VEL': trace is already processed,
            it's the output of trace.remove_response(..output='VEL')
        'DISP': trace is already processed,
            it's the output of trace.remove_response(..output='DISP')
    '''
    trace_input_type = config['preprocess']['remove_response_output']

    conf = config['preprocess']
    config_wa = dict(config['paz_wa'])
    # parse complex string to complex numbers:
    zeros_parsed = map(complex, (c.replace(' ', '') for c in config_wa['zeros']))
    config_wa['zeros'] = list(zeros_parsed)
    poles_parsed = map(complex, (c.replace(' ', '') for c in config_wa['poles']))
    config_wa['poles'] = list(poles_parsed)
    # compute synthetic WA response. This modifies the trace in-place!

    if trace_input_type in ('VEL', 'ACC'):
        trace.integrate()
    if trace_input_type == 'ACC':
        trace.integrate()

    if trace_input_type is None:
        pre_filt = (0.005, 0.006, 40.0, 45.0)
        trace.remove_response(inventory=segment.inventory(), output="DISP",
                              pre_filt=pre_filt, water_level=conf['remove_response_water_level'])

    return trace.simulate(paz_remove=None, paz_simulate=config_wa)


def signal_noise_spectra(segment, config):
    """
    Computes the signal and noise spectra, as dict of strings mapped to tuples (x0, dx, y).
    Does not modify the segment's stream or traces in-place

    :return: a dict with two keys, 'Signal' and 'Noise', mapped respectively to the tuples
    (f0, df, frequencies)

    :raise: an Exception if `segment.stream()` is empty or has more than one trace (possible
    gaps/overlaps)
    """
    # get sn windows: PLEASE NOTE!! sn_windows might calculate the cumulative of segment.stream(),
    # thus the latter should have been preprocessed (e.g. remove response, bandpass):
    arrival_time = UTCDateTime(segment.arrival_time) + config['sn_windows']['arrival_time_shift']
    signal_trace, noise_trace = sn_split(segment.stream()[0],  # assumes stream has only one trace
                                         arrival_time, config['sn_windows']['signal_window'])
    x0_sig, df_sig, sig = _spectrum(signal_trace, config)
    x0_noi, df_noi, noi = _spectrum(noise_trace, config)
    return {'Signal': (x0_sig, df_sig, sig), 'Noise': (x0_noi, df_noi, noi)}


def _spectrum(trace, config):
    '''Calculate the spectrum of a trace. Returns the tuple (0, df, values), where
    values depends on the config dict parameters.
    Does not modify the trace in-place
    '''
    taper_max_percentage = config['sn_spectra']['taper']['max_percentage']
    taper_type = config['sn_spectra']['taper']['type']
    if config['sn_spectra']['type'] == 'pow':
        func = powspec  # copies the trace if needed
    elif config['sn_spectra']['type'] == 'amp':
        func = ampspec  # copies the trace if needed
    else:
        # raise TypeError so that if called from within main, the iteration stops
        raise TypeError("config['sn_spectra']['type'] expects either 'pow' or 'amp'")

    df_, spec_ = func(trace, taper_max_percentage=taper_max_percentage, taper_type=taper_type)

    # if you want to implement your own smoothing, change the lines below before 'return'
    # and implement your own config variables, if any
    smoothing_wlen_ratio = config['sn_spectra']['smoothing_wlen_ratio']
    if smoothing_wlen_ratio > 0:
        spec_ = triangsmooth(spec_, winlen_ratio=smoothing_wlen_ratio)

    return (0, df_, spec_)


def meanslice(trace, nptmin=100, starttime=None, endtime=None):
    """
    Returns the numpy nanmean of the trace data, optionally slicing the trace first.
    If the trace number of points is lower than `nptmin`, returns NaN (numpy.nan)
    """
    if starttime is not None or endtime is not None:
        trace = trace.slice(starttime, endtime)
    if trace.stats.npts < nptmin:
        return np.nan
    val = np.nanmean(trace.data)
    return val


######################################
# GUI functions for displaying plots #
######################################


@gui.plot
def cumulative(segment, config):
    '''Computes the cumulative of the squares of the segment's trace in the form of a Plot object.
    Modifies the segment's stream or traces in-place. Normalizes the returned trace values
    in [0,1]

    :return: an obspy.Trace

    :raise: an Exception if `segment.stream()` is empty or has more than one trace (possible
    gaps/overlaps)
    '''
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    return cumsumsq(stream[0], normalize=True, copy=False)


@gui.plot('r', xaxis={'type': 'log'}, yaxis={'type': 'log'})
def sn_spectra(segment, config):
    """
    Computes the signal and noise spectra, as dict of strings mapped to tuples (x0, dx, y).
    Does NOT modify the segment's stream or traces in-place

    :return: a dict with two keys, 'Signal' and 'Noise', mapped respectively to the tuples
    (f0, df, frequencies)

    :raise: an Exception if `segment.stream()` is empty or has more than one trace (possible
    gaps/overlaps)
    """
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    return signal_noise_spectra(segment, config)


@gui.plot
def velocity(segment, config):
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    trace = stream[0]
    trace_int = trace.copy()
    return trace_int.integrate()


@gui.plot
def derivcum2(segment, config):
    """
    compute the second derivative of the cumulative function using savitzy-golay.
    Modifies the segment's stream or traces in-place

    :return: the tuple (starttime, timedelta, values)

    :raise: an Exception if `segment.stream()` is empty or has more than one trace (possible
    gaps/overlaps)
    """
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    cum = cumsumsq(stream[0], normalize=True, copy=False)
    sec_der = savitzky_golay(cum.data, 31, 2, deriv=2)
    sec_der_abs = np.abs(sec_der)
    sec_der_abs /= np.nanmax(sec_der_abs)
    # the stream object has surely only one trace (see 'cumulative')
    return segment.stream()[0].stats.starttime, segment.stream()[0].stats.delta, sec_der_abs


@gui.plot
def synth_wa(segment, config):
    '''compute synthetic WA. See ``synth_wood_anderson``.
    Modifies the segment's stream or traces in-place.

    :return:  an obspy Trace
    '''
    stream = segment.stream()
    assert1trace(stream)  # raise and return if stream has more than one trace
    return synth_wood_anderson(segment, config, stream[0])

