"""
Math utilities operating on `numpy` arrays

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>,
                  Graeme Weatherill <gweather@gfz-potsdam.de>
"""

from __future__ import division

from math import floor, ceil, isnan, sqrt

import numpy as np
from scipy.signal import hilbert
from scipy.integrate import cumtrapz


def powspec(signal, signal_is_fft=False):
    """
    Returns the power spectrum of a REAL signal.
    For computing the frequency resolution or the relative frequencies array,
    see :func:`stream2segment.process.math.ndarrays.dfreq` and
    :func:`stream2segment.process.math.ndarrays.freqs`, respectively

    :param signal: the time-series input signal (numeric array)
    :param signal_is_fft: boolean (default:False). If True, the signal is already a Fft of some
        signal. Otherwise, :func:`stream2segment.process.math.ndarrays.fft`(signal)
        will be computed first

    :return: numpy array representing the signal power spectrum
    """
    return np.square(ampspec(signal, signal_is_fft))


def ampspec(signal, signal_is_fft=False):
    """
    Returns the amplitude spectrum of a REAL signal.
    For computing the frequency resolution or the relative frequencies array,
    see :func:`stream2segment.process.math.ndarrays.dfreq` and
    :func:`stream2segment.process.math.ndarrays.freqs`, respectively

    :param signal: the time-series input signal (numeric array)
    :param signal_is_fft: boolean (default:False). If True, the signal is already a Fft of some
        signal. Otherwise, :func:`stream2segment.process.math.ndarrays.fft`(signal)
        will be computed first

    :return: numpy array representing the signal amplitude spectrum
    """
    return np.abs(fft(signal) if not signal_is_fft else signal)


def fft(signal):
    """
    Returns the discrete fft (fast Fourier transform) of a REAL signal.
    see :func:`stream2segment.process.math.ndarrays.dfreq` and
    :func:`stream2segment.process.math.ndarrays.freqs`, respectively

    :param signal: the time-series input signal (numeric array)
    :param signal_is_fft: boolean (default:False). If True, the signal is already a Fft of some
        signal. Otherwise, :func:`stream2segment.process.math.ndarrays.fft`(signal)
        will be computed first

    :return: numpy array representing the signal Fast-Fourier transform
    """
    return np.fft.rfft(signal)


def dfreq(time_signal, delta_t):
    """
    Returns the frequency resolution (in Hertz) of a real fft applied on `time_signal`

    :param time_signal: numpy array or numeric list: the time-domain signal (time-series)
    :param delta_t: `time_signal` sampling period, in seconds

    :return: the frequency resolution df (in Hertz) of a real fft applied on `time_signal`
    """
    return 1 / (len(time_signal) * delta_t)


def freqs(time_signal, delta_t):
    """
    Returns the numpy array of the frequencies of a real fft applied on `time_signal`:
    ```
        deltaF = dfreq(time_signal, delta_t)
        L = floor(1 + len(time_signal) / 2.0)
        return [0, deltaF, ..., (i-1) * deltaF, ..., (L-1) * deltaF]
    ```

    :param time_signal: numpy array or numeric list: the time-domain signal (time-series)
    :param delta_t (float): `time_signal` sampling period (in seconds)

    :return: the numpy array of the frequencies of a real fft applied on `time_signal`
    """
    try:
        leng = int(floor(1 + len(time_signal) / 2))
        delta_f = dfreq(time_signal, delta_t)
    except TypeError:
        leng = time_signal
        delta_f = delta_t
    return linspace(0, delta_f, leng)


def linspace(start, delta, num):
    """
    Returns an evenly spaced array of values, convenient for building e.g. arrays of
    frequencies, given the fft's frequency resolution `delta`. Equivalent to:
    `numpy.linspace(start, start + delta * num, num, endpoint=False)`

    :param start: numeric, the first element of the returned array
    :param delta: numeric, the distance between points of the returned array
    :param num: integer, the length of the returned array

    :return: A numpy array of evenly spaced `num` numbers starting from `start`,
        with resolution=`delta` (distance between two consecutive points).
    """
    return np.linspace(start, start + delta * num, num, endpoint=False)


def snr(signal, noise, signals_form='', fmin=None, fmax=None, delta_signal=1.,
        delta_noise=1., nearest_sample=False, in_db=False):
    """
    Returns the signal to noise ratio (SNR) of `signal` over `noise`. If required, runs `fft`
    before computing the SNR, and/or computes the SNR in a special
    frequency band [`fmin`, `fmax`] only

    :param signal: a numpy array denoting the divisor of the snr
    :param noise: a numpy array denoting the dividend of the snr
    :param signals_form: tells this function what the given signals are. If:
        - 'fft' or 'dft': then the signals are discrete Fourier transforms, and they will be
          converted to power spectra before computing the snr (modulus of each fft component)
        - 'amp': then the signals are amplitude spectra, they will be converted to power spectra
          before computing the snr
        - 'pow', then the signals are power spectra.
        - any other value: then the signals are time series, their power spectra will be
          computed before returning the snr
    :param fmin: None or float: the start frequency of the interval where to compute the snr.
        None (the default) will set a left-unbounded interval. If `fmin=fmax=None` then no
        frequency interval will be set (compute the snr on all frequencies)
    :param fmax: None or float: the end frequency of the interval where to compute the snr.
        None (the default) will set a right-unbounded interval. If `fmin=fmax=None` then no
        frequency interval will be set (compute the snr on all frequencies)
    :param delta_signal: float (ignored if `fmin=fmax=None`): the frequency resolution of `signal`,
        in Herz, if `signals_form` is in ['pow', 'dft', 'fft', 'amp']. Otherwise, the sampling
        period of `signal`, in seconds
    :param delta_noise: float (ignored if `fmin=fmax=None`): the frequency resolution of `noise`,
        in Herz, if `signals_form` is in ['pow', 'dft', 'fft', 'amp']. Otherwise, the sampling
        period of `noise`, in seconds
    :param nearest_sample: boolean, default False  (ignored if `fmin=fmax=None`):
        whether or not to take the nearest sample when trimming according to `fmin` and `fmax`,
        or to take only the samples strictly included in the interval (the default)
    :param in_db: boolean (False by default): whether to return the SNR in db's or not

    :return: the signal-to-noise ratio of two signals
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
    """
    Trims the evenly spaced sampled signal `signal`.

    :param signal: numpy numeric array
    :param deltax: the distance between two points on `signal`'s domain, in
        whatever unit the domain is (e.g., Herz, seconds)
    :param minx: float, the minimum of `signal`'s domain (the same as `deltax`)
    :param maxx: float, the maximum of `signal`'s domain (the same as `deltax`)
    :param nearest_sample: boolean, default false.  whether or not to take the nearest sample
        when trimming according to `minx` and `maxx`, or to
        take only the samples strictly included in the interval (the default)

    :return: a numeric numpy array representing a slice of `signal` in the given x bounds
    """
    if minx is None and maxx is None:
        return signal
    idxmin, idxmax = argtrim(signal, deltax, minx, maxx, nearest_sample)
    return signal[idxmin: idxmax]


def argtrim(signal, deltax, minx=None, maxx=None, nearest_sample=False):
    """
    Returns the indices (i0, i1) such as `signal[i0:i1]` is the slice of signal
    between (and including) the `signal`'s domain bounds `minx` and `maxx`.
    The returned 2-element tuple might contain `None`s (valid python slice argument to indicate:
    no bounds)

    :param signal: numpy numeric array
    :param deltax: the distance between two points on `signal`'s domain, in
        whatever unit the domain is (e.g., Herz, seconds)
    :param minx: float, the minimum of `signal`'s domain (the same as `deltax`)
    :param maxx: float, the maximum of `signal`'s domain (the same as `deltax`)
    :param nearest_sample: boolean, default false.  whether or not to take the nearest sample
        when trimming according to `minx` and `maxx`, or to
        take only the samples strictly included in the interval (the default)

    :return: a tuple of two numeric values (or None) representing the indices of `signal` where
        its domain values are `minx` and `maxx`
    """
    idxmin, idxmax = None, None
    if minx is not None:
        idx = int(round(minx / deltax) if nearest_sample else ceil(minx / deltax))
        idxmin = min(max(0, idx), len(signal))
    if maxx is not None:
        idx = int(round(maxx / deltax) if nearest_sample else floor(maxx / deltax)) + 1
        idxmax = min(max(0, idx), len(signal))
    return idxmin, idxmax


def cumsumsq(signal, normalize=True):
    """
    Return the cumulative sum of `signal**2`

    :param signal: 1d array or numeric list
    :param normalize: if True (the default), normalizes the cumulative in [0,1]

    :return: the cumulative sum of the square of `signal` (numpy array)
    """
    ret = np.cumsum(np.square(signal), axis=None, dtype=None, out=None)
    if normalize:
        # first check if any not nan, as np.nanmax issues warnings if computed on all-nan array:
        if not np.isnan(ret[0]):  # in a cumulative, 1st element nan => all nan
            min_ = ret[0]
            max_ = ret[-1] if not np.isnan(ret[-1]) else np.nanmax(ret)
            if max_ != min_:
                # normalize between 0 and 1. Note that ret /= max_ might lead to cast problems, so:
                ret = (ret - min_) / (max_ - min_)
    return ret


def triangsmooth(array, winlen_ratio):
    """
    Smoothes `array` by normalizing each point `array[i]` `with triangular window whose
    length is index-dependent, i.e. it increases with the index. For frequency domain `array`s
    (which is the typical use case), the window is therefore frequency-dependent.
    If the window overflows the array length, it will be shrunk the necessary amount of points.
    Thus, depending on `wlen_ratio`, the window length increases with the index until a maximum
    is reached. After that, the window length will decrease (set to the maximum available not to
    overflow). This function always work on a copy of `array`

    :param array: numpy array of values to be smoothed, usually (but not necessarily)
        frequency-domain values (fft, spectrum, ...)
    :param winlen_ratio: float in [0.1]: the length of a "branch" of the triangular smoothing
        window, as a percentage of the current point index. For each point, the window length will
        be (`2*i*winlen_ratio`). Thus, the higher the index, the higher the window (or, if `array`
         is a frequency domain array, the higher the frequency, the higher the window). If the
        window length overflows `array` indices, it will be set to the maximum possible length

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


class ResponseSpectrum(object):
    """
    Base abstract Class to implement a response spectrum calculation
    """
    def __init__(self, acceleration, time_step, periods, damping=0.05,
                 units="cm/s/s"):
        """
        Setup the response spectrum calculator
        :param acceleration: (numpy.ndarray) the acceleration
        :param time_step: the sampling period (delta t) of `acceleration`
        :param periods: (numpy.ndarray) Spectral periods (s) for calculation
        :param damping: float (default=0.05) Fractional coefficient of damping
        :param str units: Units of the `acceleration` {"g", "m/s", "cm/s/s"}, If not
        "cm/s/s", it will be converted to that unit internally before calculations
        """
        self.periods = periods
        self.num_per = len(periods)
        self.acceleration = ResponseSpectrum.acc2cms2(acceleration, units)
        self.damping = damping
        self.d_t = time_step
        self.velocity, self.displacement = \
            ResponseSpectrum.get_velocity_displacement(self.d_t, self.acceleration)
        self.num_steps = len(self.acceleration)
        self.omega = (2. * np.pi) / self.periods
        self.response_spectrum = None

    def evaluate(self):
        """
        Evaluates the response spectrum
        :returns:
            Response Spectrum - Dictionary containing all response spectrum
                                data
                'Time' - Time (s)
                'Acceleration' - Acceleration Response Spectrum (cm/s/s)
                'Velocity' - Velocity Response Spectrum (cm/s)
                'Displacement' - Displacement Response Spectrum (cm)
                'Pseudo-Velocity' - Pseudo-Velocity Response Spectrum (cm/s)
                'Pseudo-Acceleration' - Pseudo-Acceleration Response Spectrum
                                       (cm/s/s)

            Time Series - Dictionary containing all time-series data
                'Time' - Time (s)
                'Acceleration' - Acceleration time series (cm/s/s)
                'Velocity' - Velocity time series (cm/s)
                'Displacement' - Displacement time series (cm)
                'PGA' - Peak ground acceleration (cm/s/s)
                'PGV' - Peak ground velocity (cm/s)
                'PGD' - Peak ground displacement (cm)

            accel - Acceleration response of Single Degree of Freedom Oscillator
            vel - Velocity response of Single Degree of Freedom Oscillator
            disp - Displacement response of Single Degree of Freedom Oscillator
        """
        raise NotImplementedError("This is an abstract class, you should call sub-classes "
                                  "implementing this method")

    @staticmethod
    def acc2cms2(acceleration, units):
        """
        Converts acceleration to different units, returning `acceleration` in 'cm/s^2'
        :param acceleration: numpy array denoting the acceleration
        :param units: string, denoting the units of `acceleration`. It is
            either "m/s/s", "m/s**2", "m/s^2", "g", "cm/s/s", "cm/s**2", "cm/s^2".
            In the three latter cases, `acceleration` is returned as-it-is.
        """
        if units == "g":
            return 981. * acceleration
        elif (units == "m/s/s") or (units == "m/s**2") or (units == "m/s^2"):
            return 100. * acceleration
        elif (units == "cm/s/s") or (units == "cm/s**2") or (units == "cm/s^2"):
            return acceleration
        else:
            raise ValueError("Unrecognised time history units. "
                             "Should take either ''g'', ''m/s/s'' or ''cm/s/s''")

    @staticmethod
    def get_velocity_displacement(time_step, acceleration, units="cm/s/s",
                                  velocity=None, displacement=None):
        """
        Returns the velocity and displacement time series using simple integration.
        By providing `velocity` or `displacement` as argument(s), you can speed up
        this function by skipping either or both calculations.

        :param time_step: float: Time-series time-step (s)
        :param acceleration: numpy.ndarray: the acceleration
        :param units: the acceleration units, either "m/s/s", "m/s**2", "m/s^2", "g", "cm/s/s",
        "cm/s**2", "cm/s^2". The acceleration is supposed to
        be in centimeters over seconds square: if 'units' is not one of the last three strings,
        it will be conveted to cm/s^2 before calculation
        :param velocity: numpt.ndarray or None: if None, the velocity will be computed. Otherwise it
        is the already-computed vector of velocities and it will be returned by this function
        :param displacement: numpt.ndarray or None: if None, the displacement will be computed.
        Otherwise it is the already-computed vector of displacements and it will be returned by this
        function

        :returns:
            velocity - Velocity Time series (cm/s)
            displacement - Displacement Time series (cm)
        """
        acceleration = ResponseSpectrum.acc2cms2(acceleration, units)
        if velocity is None:
            velocity = time_step * cumtrapz(acceleration, initial=0.)
        if displacement is None:
            displacement = time_step * cumtrapz(velocity, initial=0.)
        return velocity, displacement


class NewmarkBeta(ResponseSpectrum):
    """
    Evaluates the response spectrum using the Newmark-Beta methodology
    """

    def evaluate(self):
        """
        Evaluates the response spectrum
        :returns:
            Response Spectrum - Dictionary containing all response spectrum
                                data
                'Time' - Time (s)
                'Acceleration' - Acceleration Response Spectrum (cm/s/s)
                'Velocity' - Velocity Response Spectrum (cm/s)
                'Displacement' - Displacement Response Spectrum (cm)
                'Pseudo-Velocity' - Pseudo-Velocity Response Spectrum (cm/s)
                'Pseudo-Acceleration' - Pseudo-Acceleration Response Spectrum
                                       (cm/s/s)

            Time Series - Dictionary containing all time-series data
                'Time' - Time (s)
                'Acceleration' - Acceleration time series (cm/s/s)
                'Velocity' - Velocity time series (cm/s)
                'Displacement' - Displacement time series (cm)
                'PGA' - Peak ground acceleration (cm/s/s)
                'PGV' - Peak ground velocity (cm/s)
                'PGD' - Peak ground displacement (cm)

            accel - Acceleration response of Single Degree of Freedom Oscillator
            vel - Velocity response of Single Degree of Freedom Oscillator
            disp - Displacement response of Single Degree of Freedom Oscillator
        """
        omega = self.omega  # (2. * np.pi) / self.periods
        cval = self.damping * 2. * omega
        kval = ((2. * np.pi) / self.periods) ** 2.
        # Perform Newmark - Beta integration
        accel, vel, disp, a_t = self._newmark_beta(omega, cval, kval)
        self.response_spectrum = {
            'Period': self.periods,
            'Acceleration': np.max(np.fabs(a_t), axis=0),
            'Velocity': np.max(np.fabs(vel), axis=0),
            'Displacement': np.max(np.fabs(disp), axis=0)}
        self.response_spectrum['Pseudo-Velocity'] = omega * \
            self.response_spectrum['Displacement']
        self.response_spectrum['Pseudo-Acceleration'] = (omega ** 2.) * \
            self.response_spectrum['Displacement']
        time_series = {
            'Time-Step': self.d_t,
            'Acceleration': self.acceleration,
            'Velocity': self.velocity,
            'Displacement': self.displacement,
            'PGA': np.max(np.fabs(self.acceleration)),
            'PGV': np.max(np.fabs(self.velocity)),
            'PGD': np.max(np.fabs(self.displacement))}
        return self.response_spectrum, time_series, accel, vel, disp

    def _newmark_beta(self, omega, cval, kval):
        """
        Newmark-beta integral
        :param numpy.ndarray omega:
            Angular period - (2 * pi) / T
        :param numpy.ndarray cval:
            Damping * 2 * omega
        :param numpy.ndarray kval:
            ((2. * pi) / T) ** 2.

        :returns:
            accel - Acceleration time series
            vel - Velocity response of a SDOF oscillator
            disp - Displacement response of a SDOF oscillator
            a_t - Acceleration response of a SDOF oscillator
        """
        # Pre-allocate arrays
        accel = np.zeros([self.num_steps, self.num_per], dtype=float)
        vel = np.zeros([self.num_steps, self.num_per], dtype=float)
        disp = np.zeros([self.num_steps, self.num_per], dtype=float)
        a_t = np.zeros([self.num_steps, self.num_per], dtype=float)
        # Initial line
        accel[0, :] = (-self.acceleration[0] - (cval * vel[0, :])) - \
                      (kval * disp[0, :])
        a_t[0, :] = accel[0, :] + accel[0, :]
        for j in range(1, self.num_steps):
            disp[j, :] = disp[j-1, :] + (self.d_t * vel[j-1, :]) + \
                (((self.d_t ** 2.) / 2.) * accel[j-1, :])

            accel[j, :] = (1. / (1. + self.d_t * 0.5 * cval)) * \
                (-self.acceleration[j] - kval * disp[j, :] - cval *
                 (vel[j-1, :] + (self.d_t * 0.5) * accel[j-1, :]))
            vel[j, :] = vel[j - 1, :] + self.d_t * (0.5 * accel[j - 1, :] +
                                                    0.5 * accel[j, :])
            a_t[j, :] = self.acceleration[j] + accel[j, :]
        return accel, vel, disp, a_t


class NigamJennings(ResponseSpectrum):
    """
    Evaluate the response spectrum using the algorithm of Nigam & Jennings
    (1969)
    In general this is faster than the classical Newmark-Beta method, and
    can provide estimates of the spectra at frequencies higher than that
    of the sampling frequency.
    """

    def evaluate(self):
        """
        Define the response spectrum
        """
        omega = (2. * np.pi) / self.periods
        omega2 = omega ** 2.
        omega3 = omega ** 3.
        omega_d = omega * sqrt(1.0 - (self.damping ** 2.))
        const = {'f1': (2.0 * self.damping) / (omega3 * self.d_t),
                 'f2': 1.0 / omega2,
                 'f3': self.damping * omega,
                 'f4': 1.0 / omega_d}
        const['f5'] = const['f3'] * const['f4']
        const['f6'] = 2.0 * const['f3']
        const['e'] = np.exp(-const['f3'] * self.d_t)
        const['s'] = np.sin(omega_d * self.d_t)
        const['c'] = np.cos(omega_d * self.d_t)
        const['g1'] = const['e'] * const['s']
        const['g2'] = const['e'] * const['c']
        const['h1'] = (omega_d * const['g2']) - (const['f3'] * const['g1'])
        const['h2'] = (omega_d * const['g1']) + (const['f3'] * const['g2'])
        x_a, x_v, x_d = self._get_time_series(const, omega2)

        self.response_spectrum = {
            'Period': self.periods,
            'Acceleration': np.max(np.fabs(x_a), axis=0),
            'Velocity': np.max(np.fabs(x_v), axis=0),
            'Displacement': np.max(np.fabs(x_d), axis=0)}
        self.response_spectrum['Pseudo-Velocity'] = omega * \
            self.response_spectrum['Displacement']
        self.response_spectrum['Pseudo-Acceleration'] = (omega ** 2.) * \
            self.response_spectrum['Displacement']
        time_series = {
            'Time-Step': self.d_t,
            'Acceleration': self.acceleration,
            'Velocity': self.velocity,
            'Displacement': self.displacement,
            'PGA': np.max(np.fabs(self.acceleration)),
            'PGV': np.max(np.fabs(self.velocity)),
            'PGD': np.max(np.fabs(self.displacement))}

        return self.response_spectrum, time_series, x_a, x_v, x_d

    def _get_time_series(self, const, omega2):
        """
        Calculates the acceleration, velocity and displacement time series for
        the SDOF oscillator
        :param dict const:
            Constants of the algorithm
        :param np.ndarray omega2:
            Square of the oscillator period
        :returns:
            x_a = Acceleration time series
            x_v = Velocity time series
            x_d = Displacement time series
        """
        x_d = np.zeros([self.num_steps - 1, self.num_per], dtype=float)
        x_v = np.zeros_like(x_d)
        x_a = np.zeros_like(x_d)

        for k in range(0, self.num_steps - 1):
            yval = k - 1
            dug = self.acceleration[k + 1] - self.acceleration[k]
            z_1 = const['f2'] * dug
            z_2 = const['f2'] * self.acceleration[k]
            z_3 = const['f1'] * dug
            z_4 = z_1 / self.d_t
            if k == 0:
                b_val = z_2 - z_3
                a_val = (const['f5'] * b_val) + (const['f4'] * z_4)
            else:
                b_val = x_d[k - 1, :] + z_2 - z_3
                a_val = (const['f4'] * x_v[k - 1, :]) +\
                    (const['f5'] * b_val) + (const['f4'] * z_4)

            x_d[k, :] = (a_val * const['g1']) + (b_val * const['g2']) +\
                z_3 - z_2 - z_1
            x_v[k, :] = (a_val * const['h1']) - (b_val * const['h2']) - z_4
            x_a[k, :] = (-const['f6'] * x_v[k, :]) - (omega2 * x_d[k, :])
        return x_a, x_v, x_d


# define a global variable for use with the function below:
# note that isinstance(c, type) returns if v is a class but works for new-style classes
# which as of end 2017 is not anymore a restriction
_rs = {c.lower(): v for c, v in globals().items() if isinstance(v, type) and
       issubclass(v, ResponseSpectrum) and v != ResponseSpectrum}


def respspec(method, acceleration, time_step, periods, damping=0.05):
    """
    Evaluates the response spectrum within a single function

    :param method: a string denoting the method. Currently supported are:
        'NewmarkBeta' and 'NigamJennings' (`method` is case-insensitive so you can input also
        lower-case strings). See relative module classes for details. 'NigamJennings' is in
        general faster than the classical Newmark-Beta method, and can provide estimates of the
        spectra at frequencies higher than that of the sampling frequency.
    :param acceleration: (numpy.ndarray) the acceleration
    :param time_step: the sampling period (delta t) of `acceleration`
    :param periods: (numpy.ndarray) Spectral periods (s) for calculation
    :param damping: float (default=0.05) Fractional coefficient of damping

    :returns:
        Response Spectrum - Dictionary containing all response spectrum
                            data. All units are derived from the units of `acceleration` and
                            `time_step`. Use `class`:ResponseSpectrum.acc2cms2 to convert to
                            cm per second squared, if needed
            'Time' - Time
            'Acceleration' - Acceleration Response Spectrum
            'Velocity' - Velocity Response Spectrum
            'Displacement' - Displacement Response Spectrum
            'Pseudo-Velocity' - Pseudo-Velocity Response Spectrum
            'Pseudo-Acceleration' - Pseudo-Acceleration Response Spectrum

        Time Series - Dictionary containing all time-series data
            'Time' - Time
            'Acceleration' - Acceleration time series
            'Velocity' - Velocity time series
            'Displacement' - Displacement time series
            'PGA' - Peak ground acceleration
            'PGV' - Peak ground velocity
            'PGD' - Peak ground displacement

        accel - Acceleration response of Single Degree of Freedom Oscillator
        vel - Velocity response of Single Degree of Freedom Oscillator
        disp - Displacement response of Single Degree of Freedom Oscillator
    """
    try:
        rs_class = _rs[method.lower()]
    except KeyError:
        raise TypeError('Please supply a response spectrum method in %s' %
                        list(_rs.keys()))
    return rs_class(acceleration, time_step, periods, damping).evaluate()
