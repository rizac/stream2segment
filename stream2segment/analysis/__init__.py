
import numpy as np
from scipy.signal import hilbert


def fft(signal):
    """Returns the fft of a REAL signal
    :param signal: a signal (numeric array)
    :param dt: the delta t (distance from two points of the equally sampled signal)
    :param return_abs: if true, np.abs is applied to the returned fft, thus converting it to
        power spectrum
    """
    return np.fft.rfft(signal)


def pow_spec(signal, signal_is_fft=False):
    """Returns the power spectrum of a REAL signal
    :param signal: a signal (numeric array)
    :param dt: the delta t (distance from two points of the equally sampled signal)
    :param return_abs: if true, np.abs is applied to the returned fft, thus converting it to
        power spectrum
    """
    return np.square(amp_spec(signal, signal_is_fft))


def amp_spec(signal, signal_is_fft=False):
    """Returns the amplitude spectrum of a REAL signal
    :param signal: a signal (numeric array)
    :param dt: the delta t (distance from two points of the equally sampled signal)
    :param return_abs: if true, np.abs is applied to the returned fft, thus converting it to
        power spectrum
    """
    return np.abs(fft(signal) if not signal_is_fft else signal)


def dfreq(signal, delta_t):
    """return the delta frequency of a given signal with given sampling rate delta_t (in seconds)"""
    return 1.0 / (len(signal) * delta_t)


def snr(signal, noisy_signal, signals_form='normal', in_db=False):
    """
    Returns the signal to noise ratio of signal1 over signal2.
    :param signal1: a numeric array denoting the divisor of the snr
    :param signal1: a numeric array denoting the divident of the snr
    :param signal_format: tells this function what are signal1 and signal2. If:
        - 'fft' or 'dft': then the signals are discrete fourier transofrms, and they will be
            converted to amplitude spectra before computing the snr (modulus of each fft component)
        - 'amp;: then the signals are amplitude spectra. ``snr = 20*log10(signal1 /signal2)``
        - 'pow', then the signals are power spectra. ``snr = 20*log10(signal1 /signal2)``
        - any other value: then the signals are time series, their amplitude spectra will be
            computed before returing the snr.
    """
    if signals_form.lower() == 'amp':
        signal = np.square(signal)
        noisy_signal = np.square(noisy_signal)
    elif signals_form.lower() == 'fft' or signals_form.lower() == 'dft':
        signal = pow_spec(signal, signal_is_fft=True)
        noisy_signal = pow_spec(noisy_signal, signal_is_fft=True)
    elif signals_form.lower() != 'pow':
        signal = pow_spec(signal, signal_is_fft=False)
        noisy_signal = pow_spec(noisy_signal, signal_is_fft=False)

    # normalize by the number of points:
    square1 = np.true_divide(signal, len(signal))
    square2 = np.true_divide(noisy_signal, len(noisy_signal))
    ret = np.true_divide(square1, square2)
    # if no db, then return the sqrt.
    # The sqrt is accounted for in db by multiplying by 10 and not 20
    return 10 * np.log10(ret) if in_db else np.sqrt(ret)


def cumsum(signal, normalize=True):
    """
        Returns the cumulative resulting from the cumulative on the given signal
    """
    ret = np.cumsum(np.square(signal), axis=None, dtype=None, out=None)
    if normalize:
        # normalize between 0 and 1. Note true div cause if signal is made of ints we have a floor
        # division with loss of precision
        ret = np.true_divide(ret, np.max(ret))
    return ret


def env(signal):
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope


def linspace(start, delta, npts):
    """
        Return evenly spaced numbers over a specified interval. Calls:
            numpy.linspace(start, start + delta * npts, npts, endpoint=False)
    """
    return np.linspace(start, start + delta * npts, num=npts, endpoint=False)


# def moving_average(np_array, n=3):
#     """Implements the moving average filter. NOT USED. FIXME: REMOVE!"""
#     ret = np.cumsum(np_array, dtype=float)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n - 1:] / n


def maxabs(signal):
    """
        Returns the maximum of the absolute values of the signal, i.e. the tuple:
            (index_of_max, max)
    """
    idx = np.nanargmax(np.abs(signal))
    return idx, signal[idx]
