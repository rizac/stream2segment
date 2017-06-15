"""
Config file for the GUI. Please follow the instructions for customizing it
"""
import numpy as np
from obspy.core import Trace, Stream, UTCDateTime
from stream2segment.analysis.mseeds import bandpass, cumsum, cumtimes
from stream2segment.analysis import triangsmooth, dfreq, amp_spec


def filter(trace, segment, inventory, config, warning):  # @ReservedAssignment
    """Filters the given trace"""
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


def spectra(trace, segment, inventory, config, warning):

    # calculate delta-freq first:
    noisy_wdw, signal_wdw = _get_spectra_windows(config, segment.arrival_time, trace)

    noise_trace = trace.copy().trim(starttime=noisy_wdw[0], endtime=noisy_wdw[1],
                                    pad=True, fill_value=0)
    signal_trace = trace.copy().trim(starttime=signal_wdw[0], endtime=signal_wdw[1],
                                     pad=True, fill_value=0)

    df_signal = dfreq(signal_trace.data, signal_trace.stats.delta)
    df_noise = dfreq(noise_trace.data, noise_trace.stats.delta)

    taper_max_percentage = config['spectra']['taper_max_percentage']
    if taper_max_percentage > 0:
        noise_trace.taper(max_percentage=taper_max_percentage,
                          type=config['spectra']['taper_type'])
        signal_trace.taper(max_percentage=taper_max_percentage,
                           type=config['spectra']['taper_type'])

    # note below: amp_spec(array, True) (or pow_spec, it's the same) simply returns the
    # abs(array) which apparently works also if array is an obspy Trace. To avoid problems pass
    # the data value
    spec_noise, spec_signal = amp_spec(noise_trace.data), amp_spec(signal_trace.data)

    smoothing_wlen_ratio = config['smoothing_wlen_ratio']
    if smoothing_wlen_ratio > 0:
        spec_noise = triangsmooth(spec_noise, winlen_ratio=smoothing_wlen_ratio)
        spec_signal = triangsmooth(spec_signal, winlen_ratio=smoothing_wlen_ratio)

    return {'Noise': (0, df_noise, spec_noise), 'Signal': (0, df_signal, spec_signal)}

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


def _get_spectra_windows(config, a_time, trace):
    '''Returns the spectra windows from a given arguments. Used by `_spectra`
    :return the tuple (start, end), (start, end) where all arguments are `UTCDateTime`s
    and the first tuple refers to the noisy window, the latter to the signal window
    '''
    try:
        a_time = UTCDateTime(a_time) + config['spectra']['arrival_time_shift']
        # Note above: UTCDateTime +float considers the latter in seconds
        # we need UTcDateTime cause the spectra function we implemented accepts that object type
        try:
            cum0, cum1 = config['spectra']['signal_window']
            t0, t1 = cumtimes(cumsum(trace), cum0, cum1)
            nsy, sig = [a_time - (t1-t0), a_time], [t0, t1]
        except TypeError:
            shift = config['spectra']['signal_window']
            nsy, sig = [a_time, a_time-shift], [a_time, a_time+shift]
        return nsy, sig
    except Exception as err:
        raise ValueError("%s (check config)" % str(err))


def cumulative(trace, segment, inventory, config, warning):
    '''function returning the cumulative of a trace in the form of a Plot object'''
    return cumsum(trace)


# Now we can write a custom function. We will implement the 1st derivative
# A custom function accepts the currently selected obspy Trace T, and MUST return another Trace,
# or a numpy array with the same number of points as T:
def first_deriv(trace, segment, inventory, config, warning):
    """Calculates the first derivative of the current segment's trace"""
    deriv = np.diff(trace.data)
    # append last point (deriv.size = trace.data.size-1):
    deriv = np.append(deriv, deriv[-1])
    # and return our array:
    return deriv
    # or, alternatively:
    # return Trace(data=deriv, header=trace.stats.copy())
