"""
Python file for the GUI. In this file the user can implement the plots shown on the GUI by
means of pure python functions. Please read before proceeding.

The GUI shows data plots and metadata for each downloaded segment. The goal of the GUI is:
 - visual inspection after processing
 - experimental features implementation (e.g., visual validation pre-processing)
 - supervised machine learning annotation

The GUI *always* shows three plots:
    1. the plot of the selected segment data, and
    2. the plot of its s/n spectra (signal and noise spectra): the noise time window and the signal
    time window on which the two spectra are applied are settable by the user in the `config.yaml`
    file and from within the GUI
    3. Any custom plot defined here. If more than one plot is defined, a radio button will
    allow the user to select which custom plot to show

Note that for a given segment a pre-process can be defined which applies to the given segment
and all relative plots: in that case, a checkbox will switch from the "raw" data plots to the
"pre-processed" data plots.

Any function defined here which does *not* start with the underscore will denote a plot shown
on the GUI. The function signature (inputs and outputs) is:
```
    def my_func(segment, stream, inventory, config):
        ... code here ...
```
I.e., it must take four arguments: a segment, representing the underlying database row,
the `obspy.Stream` object representing the waveform data,
an `obspy.core.inventory.inventory.Inventory` representing the station inventory
and a configuration dict (resulting from the $CONFIG file):
(`inventory` might be None if "inventory" is missing or False in the config file, or an
exception was raised during download).

The function name will be the name of the custom plot displayed on the GUI. Two reserved names:
'_pre-process' and '_sn_spectrum' denote functions for the pre-processing and the spectrum to be
applied on the noisy and signal segment windows. The '_pre_process' function must return an
obspy Trace. Any other function must return:
      - An obspy.Stream: the plot will display all traces of the Stream: a data series (line) for
        each trace
      - An obspy Trace: the plot will display the given trace as a data series (line)
      - A numpy array (or numeric list): If the given stream has a single trace,
        the array will be converted to a Trace with
        starting time equal to the trace starting time, and same sampling period (if the stream has
        more than one trace, the plot will be empty. A message will show that the program is unable
        to infer starting time and sampling period)
      - A 3-element tuple: (x0, dx, values), where x0 is a number, UTCDateTime or datetime denoting
        the x of the first point, dx is a number or timedelta denoting the sampling period, and
        values is a numpy array / list of numeric values (y axis)
      - A dict of Traces, numpy arrays or 3-elements tuples: this will display multiple data-series
        in the same plot, and each dict key will be the label of the given data-series. If display
        order must match insertion order, use `OrderedDict`

The documentation of the function will be shown in the GUI, if present
"""
from math import factorial
import numpy as np
from obspy.core import Trace, Stream, UTCDateTime
from stream2segment.analysis.mseeds import bandpass, cumsum, cumtimes, powspec, ampspec
from stream2segment.analysis import triangsmooth


def _pre_process(segment, stream, inventory, config):
    """Applies a pre-process on the given segment waveform by
    filtering the signal and removing the instrumental response
    The filter algorithm has the following steps:
    1. Sets the max frequency to 0.9 of the nyquist freauency (sampling rate /2)
    (slightly less than nyquist seems to avoid artifacts)
    2. Offset removal (substract the mean from the signal)
    3. Tapering
    4. Pad data with zeros at the END in order to accomodate the filter transient
    5. Apply bandpass filter, where the lower frequency is set according to the magnitude
    6. Remove padded elements
    7. Remove the instrumental response"""
    # stream is the `obspy.core.Stream` object returned by reading the segment data attribute.
    # If stream has more than one trace, most likely the segment has gaps. It is up to the user
    # to handle the case: you can call `stream.merge`, perform your own processing,
    # or raise an Exception.
    # Remember that any exception thrown by functions here will be caught by the program which
    # will render in the GUI an empty plot with the error message shown
    if len(stream) != 1:
        raise Exception("%d traces (probably gaps/overlaps)" % len(stream))

    trace = stream[0]
    # define some parameters:
    evt = segment.event
    conf = config['filter_settings']
    # note: bandpass here below copied the trace! important!
    trace = bandpass(trace, _mag2freq(evt.magnitude), freq_max=conf['bandpass_freq_max'],
                     max_nyquist_ratio=conf['bandpass_max_nyquist_ratio'],
                     corners=conf['bandpass_corners'], copy=True)
    trace.remove_response(inventory=inventory, output=conf['remove_response_output'],
                          water_level=conf['remove_response_water_level'])
    return trace


def _mag2freq(magnitude):
    """converts magnitude to frequency. Used in our bandpass function to get the min freq.
    parameter"""
    if magnitude <= 4:
        freq_min = 0.5
    elif magnitude <= 5:
        freq_min = 0.3
    elif magnitude <= 6.0:
        freq_min = 0.1
    else:
        freq_min = 0.05
    return freq_min

def _savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
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
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


# def spectra(trace, segment, inventory, config, warning):
# 
#     # calculate delta-freq first:
#     noisy_wdw, signal_wdw = _get_spectra_windows(config, segment.arrival_time, trace)
# 
#     noise_trace = trace.copy().trim(starttime=noisy_wdw[0], endtime=noisy_wdw[1],
#                                     pad=True, fill_value=0)
#     signal_trace = trace.copy().trim(starttime=signal_wdw[0], endtime=signal_wdw[1],
#                                      pad=True, fill_value=0)
# 
#     df_signal = dfreq(signal_trace.data, signal_trace.stats.delta)
#     df_noise = dfreq(noise_trace.data, noise_trace.stats.delta)
# 
#     taper_max_percentage = config['spectra']['taper_max_percentage']
#     if taper_max_percentage > 0:
#         noise_trace.taper(max_percentage=taper_max_percentage,
#                           type=config['spectra']['taper_type'])
#         signal_trace.taper(max_percentage=taper_max_percentage,
#                            type=config['spectra']['taper_type'])
# 
#     # note below: amp_spec(array, True) (or pow_spec, it's the same) simply returns the
#     # abs(array) which apparently works also if array is an obspy Trace. To avoid problems pass
#     # the data value
#     spec_noise, spec_signal = amp_spec(noise_trace.data), amp_spec(signal_trace.data)
# 
#     smoothing_wlen_ratio = config['smoothing_wlen_ratio']
#     if smoothing_wlen_ratio > 0:
#         spec_noise = triangsmooth(spec_noise, winlen_ratio=smoothing_wlen_ratio)
#         spec_signal = triangsmooth(spec_signal, winlen_ratio=smoothing_wlen_ratio)
# 
#     return {'Noise': (0, df_noise, spec_noise), 'Signal': (0, df_signal, spec_signal)}


# def sn_spectra(segment, stream, inventory, config):
#     # stream is the `obspy.core.Stream` object returned by reading the segment data attribute.
#     # If stream has more than one trace, most likely the segment has gaps. It is up to the user
#     # to handle the case: you can call `stream.merge`, perform your own processing,
#     # or raise an Exception.
#     # Remember that any exception thrown by functions here will be caught by the program which
#     # will render in the GUI an empty plot with the error message shown
#     if len(stream) != 1:
#         raise Exception("%d traces (probably gaps/overlaps)" % len(stream))
# 
#     trace = stream[0]
#     taper_max_percentage = config['sn_spectra']['taper']['max_percentage']
#     taper_type = config['sn_spectra']['taper']['type']
#     if config['sn_spectra']['type'] == 'pow':
#         func = powspec
#     else:
#         func = ampspec
# 
# #     nois_wdw = config.noisy_window
# #     sig_wdw = config.signal_window
# 
#     df_noise, spec_noise = func(trace, n_window[0], n_window[1], taper_max_percentage, taper_type)
#     df_signal, spec_signal = func(trace, s_window[0], s_window[1], taper_max_percentage, taper_type)
# 
#     # if you want to implement your own smoothing, change the lines below before 'return'
#     # and implement your own config variables, if any, under sn_spectra
#     smoothing_wlen_ratio = config['sn_spectra']['smoothing_wlen_ratio']
#     # removing the if branch below
#     if smoothing_wlen_ratio > 0:
#         spec_noise = triangsmooth(spec_noise, winlen_ratio=smoothing_wlen_ratio)
#         spec_signal = triangsmooth(spec_signal, winlen_ratio=smoothing_wlen_ratio)
# 
#     return {'Noise': (0, df_noise, spec_noise), 'Signal': (0, df_signal, spec_signal)}


def _sn_spectrum(segment, stream, inventory, config):
    # stream is the `obspy.core.Stream` object returned by reading the segment data attribute.
    # If stream has more than one trace, most likely the segment has gaps. It is up to the user
    # to handle the case: you can call `stream.merge`, perform your own processing,
    # or raise an Exception.
    # Remember that any exception thrown by functions here will be caught by the program which
    # will render in the GUI an empty plot with the error message shown
    if len(stream) != 1:
        raise Exception("%d traces (probably gaps/overlaps)" % len(stream))

    trace = stream[0]
    taper_max_percentage = config['sn_spectra']['taper']['max_percentage']
    taper_type = config['sn_spectra']['taper']['type']
    if config['sn_spectra']['type'] == 'pow':
        func = powspec
    else:
        func = ampspec

#     nois_wdw = config.noisy_window
#     sig_wdw = config.signal_window

    df_, spec_ = func(trace, taper_max_percentage=taper_max_percentage, taper_type=taper_type)

    # if you want to implement your own smoothing, change the lines below before 'return'
    # and implement your own config variables, if any
    smoothing_wlen_ratio = config['sn_spectra']['smoothing_wlen_ratio']
    # removing the if branch below
    if smoothing_wlen_ratio > 0:
        spec_ = triangsmooth(spec_, winlen_ratio=smoothing_wlen_ratio)

    return (0, df_, spec_)


#         fft_noise = fft(trace, *noisy_wdw)
#         fft_signal = fft(trace, *signal_wdw)
#
#
#
#         df = fft_signal.stats.df
#         f0 = 0
#
#
#         # note below: amp_spec(array, True) (or pow_spec, it's the same) simply returns the
#         # abs(array) which apparently works also if array is an obspy Trace. To avoid problems pass
#         # the data value
#         spec_noise, spec_signal = amp_spec(fft_noise.data), amp_spec(fft_signal.data)
# 
#         if postprocess_func is not None:
#             f0noise, dfnoise, spec_noise = postprocess_func(f0, df, spec_noise)
#             f0signal, dfsignal, spec_signal = postprocess_func(f0, df, spec_signal)
#         else:
#             f0noise, dfnoise, f0signal, dfsignal = f0, df, f0, df
# 
#         return Plot(plot_title(trace, segment, "spectra"), warning=warning).\
#             add(f0noise, dfnoise, spec_noise, "Noise").\
#             add(f0signal, dfsignal, spec_signal, "Signal")

#     amp_spec = amp_spec(trace.data, signal_is_fft=False)
#     df = dfreq(trace.data, trace.stats.delta)
#     f0 = 0
#     return f0, df, triangsmooth(amp_spec, winlen_ratio=0.05)





def cumulative(segment, stream, inventory, config):
    '''function returning the cumulative of a trace in the form of a Plot object'''
        # stream is the `obspy.core.Stream` object returned by reading the segment data attribute.
    # If stream has more than one trace, most likely the segment has gaps. It is up to the user
    # to handle the case: you can call `stream.merge`, perform your own processing,
    # or raise an Exception.
    # Remember that any exception thrown by functions here will be caught by the program which
    # will render in the GUI an empty plot with the error message shown
    if len(stream) != 1:
        raise Exception("%d traces (probably gaps/overlaps)" % len(stream))
    trace = stream[0]

    return cumsum(trace)


# Now we can write a custom function. We will implement the 1st derivative
# A custom function accepts the currently selected obspy Trace T, and MUST return another Trace,
# or a numpy array with the same number of points as T:
def first_deriv(segment, stream, inventory, config):
    """Calculates the first derivative of the current segment's trace"""
    # If stream has more than one trace, most likely the segment has gaps. It is up to the user
    # to handle the case: you can call `stream.merge`, perform your own processing,
    # or raise an Exception.
    # Remember that any exception thrown by functions here will be caught by the program which
    # will render in the GUI an empty plot with the error message shown
    if len(stream) != 1:
        raise Exception("%d traces (probably gaps/overlaps)" % len(stream))
    trace = stream[0]

    deriv = np.diff(trace.data)
    # append last point (deriv.size = trace.data.size-1):
    deriv = np.append(deriv, deriv[-1])
    # and return our array:
    return deriv
    # or, alternatively:
    # return Trace(data=deriv, header=trace.stats.copy())

def derivcum2(segment, stream, inventory, config):
    """
       compute the second derivative of the cumulative function
       using savitzy-golay
    """
    cum=cumulative(segment, stream, inventory, config)
    sec_der = _savitzky_golay(cum.data,31,2,deriv=2)
    sec_der_abs = np.abs(sec_der)
    mmm = np.nanmax(sec_der_abs)
    sec_der /= mmm
    
    return sec_der_abs

def synth_wa(segment, stream, inventory, config):
    trace = stream[0]
    # compute synthetic WA,NOTE: keep as last action, it modifies trace!!
    config_wa = dict(config['paz_wa'])
    # parse complex sring to complex numbers:
    zeros_parsed = map(complex,(c.replace(' ','') for c in config_wa['zeros']))
    config_wa['zeros'] = zeros_parsed
    poles_parsed = map(complex,(c.replace(' ','') for c in config_wa['poles']))
    config_wa['poles'] = poles_parsed
    # compute synthetic WA response
    trace_wa = trace.simulate(paz_remove=None, paz_simulate=config_wa)

    return trace_wa


