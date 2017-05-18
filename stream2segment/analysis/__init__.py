
import numpy as np
from math import floor, ceil, isnan
from scipy.signal import hilbert


def fft(signal):
    """Returns the fft of a REAL signal. For computing the frequency delta, see `dfreq` (providing
    the signal delta_t, in seconds)
    :param signal: a signal (numeric array)
    :param dt: the delta t (distance from two points of the equally sampled signal)
    :param return_abs: if true, np.abs is applied to the returned fft, thus converting it to
        power spectrum
    """
    return np.fft.rfft(signal)


def pow_spec(signal, signal_is_fft=False):
    """Returns the power spectrum of a REAL signal. If `signal_is_fft=False`,
    for computing the frequency delta, see `dfreq` (providing the signal delta_t, in seconds)
    :param signal: a signal (numeric array)
    :param signal_is_fft: if true, the signal is already an fft of some signal. Otherwise
    (the default), fft(signal) will be applied first
    """
    return np.square(amp_spec(signal, signal_is_fft))


def amp_spec(signal, signal_is_fft=False):
    """Returns the amplitude spectrum of a REAL signal. If `signal_is_fft=False`,
    for computing the frequency delta, see `dfreq` (providing the signal delta_t, in seconds)
    :param signal: a signal (numeric array)
    :param signal_is_fft: if true, the signal is already an fft of some signal. Otherwise
    (the default), fft(signal) will be applied first
    """
    return np.abs(fft(signal) if not signal_is_fft else signal)


def dfreq(time_signal, delta_t):
    """return the delta frequency of a given signal with given sampling rate delta_t (in seconds)
    :param time_signal: numpy array, list, everything with a `__len__` attribute: the time-domain
    signal (time-series)
    :param delta_t: the sample rate of signal (distance in seconds between two points)
    """
    return 1.0 / (len(time_signal) * delta_t)


def freqs(signal, delta, signal_is_timeseries=False, f0=0):
    """return the numpy array of the frequencies of a real fft for the given signal.
    if `signal_is_timeseries=False` (the default), simply returns:
    ```
        [f0, f0 + delta, ..., f0 + (i-1) * delta, ..., f0 + (len(signal)-1) * delta]
    ```
    otherwise,
    ```
        [f0, f0 + deltaF, ..., f0 + (i-1) * deltaF, ..., f0 + (N-1) * deltaF]
    ```
    where:
    ```
        deltaF = dfreq(signal, delta)
        N = floor(1 + len(signal) / 2.0)
    ```
    :param signal: numpy array or numeric list, denoting either the time-series
    (`signal_is_timeseries=True`) or a frequency domain signal (fft, power spectrum, etcetera)
    :param delta (float): the signal sample rate (in seconds) if `signal_is_timeseries=True`, or the
    delta frequency (in Hz)
    :param f0: the starting frequency (number, defaults to 0): the value of the first point in the
    returned array
    :param signal_is_timeseries: boolean (default False). If False, `signal` is already a frequency
    domain array, and thus function basically calls `np.linspace`. Otherwise, allocates an
    array of `floor(1 + len(signal) / 2.0)` points (as required from a real fft applied on
    `signal`), converts `delta` as sample rate to the relative `delta` on the frequency domain
    and calls `np.linspace` on that array with the computed new delta
    :return: a new array representing the linearly spaced values of the frequencies. The length
    of the returned array is `len(signal)` if `signal_is_timeseries=False` (the default), else
    `floor(1 + len(signal) / 2.0)`. The first array value will be in any case `f0`
    """
    leng = floor(1 + len(signal) / 2.0) if signal_is_timeseries else len(signal)
    delta_f = dfreq(signal, delta) if signal_is_timeseries else delta
    return np.linspace(f0, f0 + (delta_f * leng), leng, endpoint=False)


def snr(signal, noise, signals_form='', fmin=None, fmax=None, delta_signal=1.,
        delta_noise=1., nearest_sample=False, in_db=False):
    """Returns the signal to noise ratio (SNR) of `signal` over `noise`. If required, runs `fft`
    before computing the SNR, and/or computes the SNR in a special
    frequency band [`fmin`, `fmax`] only
    :param signal: a numpy array denoting the divisor of the snr
    :param noise: a numpy array denoting the dividend of the snr
    :param signals_form: tells this function what the given signals are. If:
        - 'fft' or 'dft': then the signals are discrete Fourier transforms, and they will be
            converted to amplitude spectra before computing the snr (modulus of each fft component)
        - 'amp;: then the signals are amplitude spectra.
        - 'pow', then the signals are power spectra.
        - any other value: then the signals are time series, their amplitude spectra will be
            computed before returning the snr
    :param fmin: None or float: the minimum frequency to account for when computing the SNR.
    None (the default) will consider all frequencies
    :param fmax: None or float: the maximum frequency to account for when computing the SNR.
    None (the default) will consider all frequencies
    :param delta_signal: float (ignored if both `fmin` and `fmax` are None):
    the delta frequency of `signal` (in Herz), if `signal` is a frequency domain array
    (`signals_form` in `['pow', 'dft', 'fft', 'amp']`), or the sample rate (in seconds)
    otherwise
    :param delta_noise: float (ignored if both `fmin` and `fmax` are None):
    the delta frequency of `noise` (in Herz), if `noise` is a frequency domain array
    (`signals_form` in `['pow', 'dft', 'fft', 'amp']`), or the sample rate (in seconds)
    otherwise
    :param nearest_sample: boolean, default False  (ignored if both `fmin` and `fmax` are None):
    whether or not to take the nearest sample when trimming according to `fmin` and `fmax`, or to
    take only the samples strictly included in the interval (the default)
    :param in_db: boolean (False by default): whether to return the SNR in db's or not
    """
    if signals_form.lower() == 'amp':
        signal = np.square(signal)
        noise = np.square(noise)
    elif signals_form.lower() == 'fft' or signals_form.lower() == 'dft':
        signal = pow_spec(signal, signal_is_fft=True)
        noise = pow_spec(noise, signal_is_fft=True)
    elif signals_form.lower() != 'pow':
        # convert also deltas to frequencies
        delta_signal = dfreq(signal, delta_signal)
        delta_noise = dfreq(noise, delta_noise)
        # compute power spectra:
        signal = pow_spec(signal, signal_is_fft=False)
        noise = pow_spec(noise, signal_is_fft=False)

    # take slices if required:
    signal = trim(signal, delta_signal, fmin, fmax, nearest_sample)
    noise = trim(noise, delta_noise, fmin, fmax, nearest_sample)

    if not len(signal) or not len(noise):  # avoid warning from np.true_divide
        # Return numpy number for consistency
        return np.nan

    # normalize by the number of points:
    square1 = np.true_divide(np.sum(signal), len(signal))
    square2 = np.true_divide(np.sum(noise), len(noise))

    if square2 == 0:  # avoid warning from np.true_divide (see above)
        return np.nan

    ret = np.true_divide(square1, square2)

    if in_db:
        # avoid warning from np.log
        return -np.inf if ret == 0 else np.nan if ret < 0 else 10 * np.log10(ret)

    # if no db, then return the sqrt.
    # The sqrt is accounted for in db by multiplying by 10 and not 20
    return np.sqrt(ret)


def trim(signal, deltax, minx=None, maxx=None, nearest_sample=False):
    if minx is None and maxx is None:
        return signal
    idxmin, idxmax = argtrim(signal, deltax, minx, maxx, nearest_sample)
    return signal[idxmin: idxmax]


def argtrim(signal, deltax, minx=None, maxx=None, nearest_sample=False):
    """returns the indices of signal such as `signal[i0:i1]` is the slice of signal
    between (nd including) minx and maxx
    """
    if minx is None and maxx is None:
        return (None, None)

    idxmin, idxmax = minx, maxx
    deltax = float(deltax)

    if minx is not None:
        idx = int(round(minx/deltax) if nearest_sample else ceil(minx/deltax))
        idxmin = min(max(0, idx), len(signal))

    if maxx is not None:
        idx = int(round(maxx/deltax) if nearest_sample else floor(maxx/deltax)) + 1
        idxmax = min(max(0, idx), len(signal))

    return idxmin, idxmax


def cumsum(signal, normalize=True):
    """Returns the cumulative resulting from the cumulative on the given signal
    """
    ret = np.cumsum(np.square(signal), axis=None, dtype=None, out=None)
    if normalize:  # and (ret != 0).any():
        max_ = np.max(ret)
        if not np.isnan(max_) and (max_ != 0):
            # normalize between 0 and 1. Note true div cause if signal is made of ints we have a
            # floor division with loss of precision
            ret = np.true_divide(ret, max_)
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
    # later re-multiply) have been optimized (especially in the while loop below), for performance
    # reasons
    smoothed_array = np.array(array, copy=True, dtype=float)
    spec_len = len(array)
    # calculate npts, the array of points where npts[i] = length of the window at index i
    npts = np.zeros(spec_len, dtype=int)  # alloc once
    max_idx = int((spec_len - 1) // (winlen_ratio + 1))
    npts[:max_idx+1] = np.round(np.arange(max_idx+1) * winlen_ratio)  # .astype(int)
    if int(np.round(max_idx*winlen_ratio)) < 2:
        # winlen_ratio not big enough, window lengths are at most 1, return array immediately
        # note that max window length == 1 (1 point left, one right) is also trivial as we sould
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
        smoothed_array[idxs] = np.sum(tri_wdw * array[idxs + wdw], axis=1) / float(n ** 2)
        wdw = wdw[1:-1]
        tri_wdw[n-2:-2] = tri_wdw[n:]  # shift tri_wdw (this seems to be faster than other methods)
        tri_wdw = tri_wdw[:-2]

    return smoothed_array


# def env(signal):
#     analytic_signal = hilbert(signal)
#     amplitude_envelope = np.abs(analytic_signal)
#     return amplitude_envelope


# def linspace(start, delta, npts):
#     """
#         Return evenly spaced numbers over a specified interval. Calls:
#             numpy.linspace(start, start + delta * npts, npts, endpoint=False)
#     """
#     return np.linspace(start, start + delta * npts, num=npts, endpoint=False)


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


# def interp(npts_or_new_x_array, oldxarray, signal):
#     try:
#         len(npts_or_new_x_array)  # is npts_or_new_x_array an array?
#         newxarray = npts_or_new_x_array
#     except TypeError:  # npts_or_new_x_array is scalar (not array)
#         newxarray = np.linspace(oldxarray[0], oldxarray[-1], npts_or_new_x_array, endpoint=True)
# 
#     newsignal = np.interp(newxarray, oldxarray, signal)
# 
#     return newxarray, newsignal
