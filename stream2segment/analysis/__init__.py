"""
Math utilities for python scalars or numpy arrays

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""

from __future__ import division

from math import floor, ceil, isnan

import numpy as np
from scipy.signal import hilbert


def powspec(signal, signal_is_fft=False):
    """Returns the power spectrum of a REAL signal. If `signal_is_fft=False`,
    for computing the frequency delta, see `dfreq` (providing the signal delta_t, in seconds)
    :param signal: a signal (numeric array)
    :param signal_is_fft: if true, the signal is already an fft of some signal. Otherwise
    (the default), fft(signal) will be applied first
    """
    return np.square(ampspec(signal, signal_is_fft))


def ampspec(signal, signal_is_fft=False):
    """Returns the amplitude spectrum of a REAL signal. If `signal_is_fft=False`,
    for computing the frequency delta, see `dfreq` (providing the signal delta_t, in seconds)
    :param signal: a signal (numeric array)
    :param signal_is_fft: if true, the signal is already an fft of some signal. Otherwise
    (the default), fft(signal) will be applied first
    """
    return np.abs(fft(signal) if not signal_is_fft else signal)


def fft(signal):
    """Returns the fft of a REAL signal. For computing the frequency delta, see `dfreq` (providing
    the signal delta_t, in seconds)
    :param signal: a signal (numeric array)
    :param dt: the delta t (distance from two points of the equally sampled signal)
    :param return_abs: if true, np.abs is applied to the returned fft, thus converting it to
        power spectrum
    """
    return np.fft.rfft(signal)


def dfreq(time_signal, delta_t):
    """return the delta frequency of a given signal with given sampling rate delta_t (in seconds)
    :param time_signal: numpy array, list, everything with a `__len__` attribute: the time-domain
    signal (time-series)
    :param delta_t: the sample rate of signal (distance in seconds between two points)
    """
    return 1 / (len(time_signal) * delta_t)


def freqs(signal, delta):
    """return the numpy array of the frequencies of a real fft for the given signal:
    ```
        deltaF = dfreq(signal, delta)
        L = floor(1 + len(signal) / 2.0)
        return [0, deltaF, ..., (i-1) * deltaF, ..., (L-1) * deltaF]
    ```
    :param signal: numpy array or numeric list, denoting the time-series
    on which the fft should be (or has been) applied.
    :param delta (float): the signal sampling period (in seconds)
    :return: a new array representing the evenly spaced values of the frequencies. The length
    of the returned array is  `floor(1 + len(signal) / 2.0)`. The first array value will be 0
    """
    try:
        leng = int(floor(1 + len(signal) / 2))
        delta_f = dfreq(signal, delta)
    except TypeError:
        leng = signal
        delta_f = delta
    return np.linspace(0, delta_f * leng, leng, endpoint=False)


def linspace(start, delta, num):
    """Similar to numopy's linspace, but with different argument. Useful for building an
    array of frequencies from f0, df, and the spectrum points. Equivalent to:
    `np.linspace(start, delta * num, num, endpoint=False)`
    :return: An array of evenly spaced `num` numbers starting from `start`, and `delta` as spacing
    value.
    """
    return np.linspace(start, delta * num, num, endpoint=False)


def snr(signal, noise, signals_form='', fmin=None, fmax=None, delta_signal=1.,
        delta_noise=1., nearest_sample=False, in_db=False):
    """Returns the signal to noise ratio (SNR) of `signal` over `noise`. If required, runs `fft`
    before computing the SNR, and/or computes the SNR in a special
    frequency band [`fmin`, `fmax`] only
    :param signal: a numpy array denoting the divisor of the snr
    :param noise: a numpy array denoting the dividend of the snr
    :param signals_form: tells this function what the given signals are. If:
        - 'fft' or 'dft': then the signals are discrete Fourier transforms, and they will be
            converted to power spectra before computing the snr (modulus of each fft component)
        - 'amp;: then the signals are amplitude spectra, they will be converted to power spectra
            before computing the snr
        - 'pow', then the signals are power spectra.
        - any other value: then the signals are time series, their power spectra will be
            computed before returning the snr
    :param fmin: None or float: the start frequency of the interval where to compute the snr.
    None (the default) will set a left-unbounded interval. If `fmin=fmax=None` then no frequency
    interval will be set (compute the snr on all frequencies)
    :param fmax: None or float: the end frequency of the interval where to compute the snr.
    None (the default) will set a right-unbounded interval. If `fmin=fmax=None` then no frequency
    interval will be set (compute the snr on all frequencies)
    :param delta_signal: float (ignored if both `fmin` and `fmax` are None): the sampling interval
    of `signal`:
         - in Herz, if `signal` is a frequency domain array (`signals_form` in
           `['pow', 'dft', 'fft', 'amp']`)
         - in seconds, otherwise
    :param delta_noise: float (ignored if both `fmin` and `fmax` are None): the sampling interval
    of `noise`:
        - in Herz, if `noise` is a frequency domain array (`signals_form` in
          `['pow', 'dft', 'fft', 'amp']`)
        - in seconds, otherwise
    :param nearest_sample: boolean, default False  (ignored if both `fmin` and `fmax` are None):
    whether or not to take the nearest sample when trimming according to `fmin` and `fmax`, or to
    take only the samples strictly included in the interval (the default)
    :param in_db: boolean (False by default): whether to return the SNR in db's or not
    """
    if signals_form.lower() == 'amp':
        signal = np.square(signal)
        noise = np.square(noise)
    elif signals_form.lower() == 'fft' or signals_form.lower() == 'dft':
        signal = powspec(signal, signal_is_fft=True)
        noise = powspec(noise, signal_is_fft=True)
    elif signals_form.lower() != 'pow':
        # convert also deltas to frequencies
        delta_signal = dfreq(signal, delta_signal)
        delta_noise = dfreq(noise, delta_noise)
        # compute power spectra:
        signal = powspec(signal, signal_is_fft=False)
        noise = powspec(noise, signal_is_fft=False)

    # take slices if required:
    signal = trim(signal, delta_signal, fmin, fmax, nearest_sample)
    noise = trim(noise, delta_noise, fmin, fmax, nearest_sample)

    if not len(signal) or not len(noise):  # avoid potential warnings later
        return np.nan

    # normalize by the number of points:
    # use np.true_divide for testing purposes (mock the latter)
    square1 = np.true_divide(np.sum(signal), len(signal))
    square2 = np.true_divide(np.sum(noise), len(noise))

    if square2 == 0:  # avoid potential warnings later
        return np.nan

    ret = square1 / square2

    if in_db:
        # avoid warning from np.log
        return -np.inf if ret == 0 else np.nan if ret < 0 else 10 * np.log10(ret)

    # if no db, then return the sqrt.
    # The sqrt is accounted for in db by multiplying by 10 and not 20
    return np.sqrt(ret)


def trim(signal, deltax, minx=None, maxx=None, nearest_sample=False):
    """Trims the equally-spaced signal. General function that works
    like obspy.Trace.trim for any kind of array
    :param signal: numpy numeric array denoting the values to trim
    :param deltax: the delta between two points of signal on the x axis. The unit must be
    the same as `minx` and `maxx` (e.g., Herz, seconds, etcetera)
    :param minx: float, the minimum x, in `signal`'s unit (the same as `deltax`)
    :param maxx: float, the maximum x, in `signal`s unit (the same as `deltax`)
    :param nearest_sample: boolean, default false.  whether or not to take the nearest sample
    when trimming according to `minx` and `maxx`, or to
    take only the samples strictly included in the interval (the default)
    """
    if minx is None and maxx is None:
        return signal
    idxmin, idxmax = argtrim(signal, deltax, minx, maxx, nearest_sample)
    return signal[idxmin: idxmax]


def argtrim(signal, deltax, minx=None, maxx=None, nearest_sample=False):
    """returns the indices of signal such as `signal[i0:i1]` is the slice of signal
    between (and including) minx and maxx. The returned 2-element tuple might contain `None`s
    (valid python slice argument to indicate: no bounds)
    """
    if minx is None and maxx is None:
        return (None, None)

    idxmin, idxmax = minx, maxx

    if minx is not None:
        idx = int(round(minx / deltax) if nearest_sample else ceil(minx / deltax))
        idxmin = min(max(0, idx), len(signal))

    if maxx is not None:
        idx = int(round(maxx / deltax) if nearest_sample else floor(maxx / deltax)) + 1
        idxmax = min(max(0, idx), len(signal))

    return idxmin, idxmax


def cumsum(signal, normalize=True):
    """Returns the cumulative resulting from the cumulative on the given signal
    """
    ret = np.cumsum(np.square(signal), axis=None, dtype=None, out=None)
    if normalize:  # and (ret != 0).any():
        max_ = np.max(ret)
        if not np.isnan(max_) and (max_ != 0):
            # normalize between 0 and 1. Note that ret /= max_ might lead to cast problems, so:
            ret = ret / max_
    return ret


def triangsmooth(array, winlen_ratio):
    """Smoothes `array` by normalizing each point `array[i]` `with triangular window whose
    length is index-dependent, i.e. it increases with the index. For frequency domain `array`s
    (which is the typical use case), the window length is frequency-dependent, i.e. it increases
    with the frequency
    At boundaries, the window will be shrunk the necessary amount of points not to overflow.
    Thus the point with the largest triangular window will be the one that can accommodate that
    window. From the next on, the window length decreases until it reaches zero for the last
    `array` point
    This function always work on a copy of `array`
    :param array: numpy array of values to be smoothed, usually (but not necessarily)
    frequency-domain values (fft, spectrum, ...)
    :param winlen_ratio: float in [0.1]: the length of a "branch" of the triangular smoothing
    window, as a percentage of the current point index. For each point, the window length will be
    (`2*i*winlen_ratio`). Thus, the higher the index, the higher the window (or, if `array` is a
    frequency domain array, the higher the frequency, the higher the window). If the window
    length overflows `array` indices, it will be set to the maximum possible length
    :return: numpy array of smoothed values"""
    # this function has been converted from matlab code. The code has been vectorized as much as
    # possible, and several redundant math expression related to normalizations (e.g., divide and
    # later re-multiply) have been optimized (especially in the while loop below) for performance
    # reasons
    smoothed_array = np.array(array, copy=True, dtype=float)
    spec_len = len(array)
    # calculate npts, the array of points where npts[i] = length of the window at index i
    npts = np.zeros(spec_len, dtype=int)  # alloc once
    max_idx = int((spec_len - 1) // (winlen_ratio + 1))
    npts[:max_idx+1] = np.round(np.arange(max_idx+1) * winlen_ratio)  # .astype(int)
    if int(np.round(max_idx*winlen_ratio)) < 2:
        # winlen_ratio not big enough, window lengths are at most 1, return array immediately
        # note that max window length == 1 (1 point left, one right) is also trivial as we should
        # multiply by zero the left and right point
        return smoothed_array
    npts[max_idx+1:] = np.arange(spec_len-1-(max_idx+1), -1, -1)

    # compute the windows for each interval and set the array points accordingly:
    # wdw = [-(n-1),..., 0,...(n-1)] is the max window length and
    # tri_wdw = [1,2,3, ... (n-1)-1, ...3,2,1] is the max triangular window. At each loop,
    # wdw and tri_wdw will set the points where the interval length is n (np.argwhere(npts == n))
    # and then they will shrink: wdw = [-n+1,...,0,...,n-1], tri_wdw = [1,2,3, ... n-3, ...3,2,1]
    # for the next loop
    # Note that in wdw and tri_wdw should have length n but the two
    # boundary points tri_wdw[0] == tri_wdw[-1] == 0 are excluded for performance reasons as they
    # would count as zero in the smooth computation
    wlen = npts[max_idx]
    # allocate once window and triangular window. Both have odd num of points
    wdw = np.arange(-wlen+1, wlen, dtype=int)  # allocate once
    tri_wdw = np.arange(1, 2*wlen)  # arange will be converted to triangular array below
    # make second half (right half) decreasing: this seems to be faster than other methods
    tri_wdw[wlen:] = tri_wdw[:wlen-1][::-1]

    # compute smoothing:
    while len(wdw) > 1:  # len(wdw)=1: wdw = [0] => np.sum below is no-op => skip it
        n = wdw[-1] + 1
        idxs = np.argwhere(npts == n)
        smoothed_array[idxs.flatten()] = np.sum(tri_wdw * array[idxs + wdw], axis=1) / (n ** 2)
        wdw = wdw[1:-1]
        tri_wdw[n-2:-2] = tri_wdw[n:]  # shift tri_wdw (this seems to be faster than other methods)
        tri_wdw = tri_wdw[:-2]

    return smoothed_array
