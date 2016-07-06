from datetime import timedelta
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


def interp(newnumpoints, startx, deltax, yarray, as_json_serializable=True):
    """Calls numpy.interp, with the difference that we know only
    the startx and the deltax of the evenly spaced sequence yarray, and that we want to return
    newnumpoints from yarray
    :param as_json_serializable: converts the returned array to a python list, so that is
    json serializable
    :return: the tuple (oldxarray, newxarray, newyarray)
    """
    newxarray = np.linspace(startx, startx+deltax*newnumpoints, num=newnumpoints, endpoint=False)
    if newnumpoints == len(yarray):
        oldxarray = newxarray
        newyarray = yarray
    else:
        oldxarray = np.linspace(startx, startx+deltax*len(yarray), num=len(yarray), endpoint=False)
        newyarray = np.interp(newxarray, oldxarray, yarray)

    if as_json_serializable:
        newxarray = newxarray.tolist()
        oldxarray = oldxarray.tolist()
    return oldxarray, newxarray, newyarray


def moving_average(np_array, n=3):
    ret = np.cumsum(np_array, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
