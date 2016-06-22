from datetime import timedelta
import numpy as np
from scipy.signal import hilbert


# def todt(obj):
#     """
#         Returns a python datetime object from obj, which is supposed to be one of the 
#         several types of datetime (pandas, numpy etcetera)
#     """
#     try:
#         return obj.to_datetime()
#     except AttributeError:
#         try:
#             return obj.datetime
#         except AttributeError:
#             return obj.datetime
#
#     return obj


def fft(signal):
    """Returns the fft of signal
    :param signal: a signal (numeric array)
    :param dt: the delta t (distance from two points of the equally sampled signal)
    :param return_abs: if true, np.abs is applied to the returned fft, thus converting it to
        power spectrum
    """
    return np.fft.rfft(signal)


def pow_spec(signal):
    """Returns the power spectrum of signal
    :param signal: a signal (numeric array)
    :param dt: the delta t (distance from two points of the equally sampled signal)
    :param return_abs: if true, np.abs is applied to the returned fft, thus converting it to
        power spectrum
    """
    return np.square(amp_spec(signal))


def amp_spec(signal):
    """Returns the amplitude spectrum of signal
    :param signal: a signal (numeric array)
    :param dt: the delta t (distance from two points of the equally sampled signal)
    :param return_abs: if true, np.abs is applied to the returned fft, thus converting it to
        power spectrum
    """
    return np.abs(fft(signal))


def dfreq(signal, dt):
    return 1.0 / (len(signal) * dt)


def cumsum(signal, normalize=True):
    """
        Returns the tuple times, cumulative resulting from the cumulative on the given signal
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
