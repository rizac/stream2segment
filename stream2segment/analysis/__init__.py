
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


def pow_spec(signal):
    """Returns the power spectrum of a REAL signal
    :param signal: a signal (numeric array)
    :param dt: the delta t (distance from two points of the equally sampled signal)
    :param return_abs: if true, np.abs is applied to the returned fft, thus converting it to
        power spectrum
    """
    return np.square(amp_spec(signal))


def amp_spec(signal):
    """Returns the amplitude spectrum of a REAL signal
    :param signal: a signal (numeric array)
    :param dt: the delta t (distance from two points of the equally sampled signal)
    :param return_abs: if true, np.abs is applied to the returned fft, thus converting it to
        power spectrum
    """
    return np.abs(fft(signal))


def dfreq(signal, delta_t):
    """return the delta frequency of a given signal with given sampling rate delta_t (in seconds)"""
    return 1.0 / (len(signal) * delta_t)


def snr(signal1, signal2, signal_form='normal'):
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
            computed before returing the fft.
    """
    if signal_form.lower() == 'amp':
        factor = 20
    elif signal_form.lower() == 'pow':
        factor = 10
    elif signal_form.lower() == 'fft' or signal_form.lower() == 'dft':
        signal1 = np.abs(signal1)
        signal2 = np.abs(signal2)
        factor = 20
    else:
        signal1 = amp_spec(signal1)
        signal2 = amp_spec(signal2)
        factor = 20

    # normalize by the number of points:
    sum1 = np.true_divide(np.sum(signal1), len(signal1))
    sum2 = np.true_divide(np.sum(signal2), len(signal2))
    return factor * np.log10(np.true_divide(sum1, sum2))


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


def moving_average(np_array, n=3):
    """Implements the moving average filter. NOT USED. FIXME: REMOVE!"""
    ret = np.cumsum(np_array, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
