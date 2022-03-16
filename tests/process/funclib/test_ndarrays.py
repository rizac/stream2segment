"""
Created on Feb 23, 2016

@author: riccardo
"""
from __future__ import division

import os
import sys
from itertools import product

import mock
from mock import patch
from past.utils import old_div
import pytest
import numpy as np
from numpy import true_divide as np_true_divide


from stream2segment.process.funclib.ndarrays import cumsumsq, argtrim, snr, dfreq, freqs, powspec, \
    triangsmooth


@pytest.mark.parametrize('y',
                          [([-11, 1, 3.3, 4]),
                           (np.array([-11, 1, 3.3, 4])),
                           ])
def test_argtrim(y):
    #y = [1,2,3,4]
    delta = 0.01
    # signal_x is just used for checking:
    signal_x = np.linspace(0, delta*len(y), num=len(y), endpoint=False)

    # assert boundary conditions are satisfied:
    assert np.array_equal(y[slice(*argtrim(y, delta, None, None, nearest_sample=False))], y)
    assert np.array_equal(y[slice(*argtrim(y, delta, None, None, nearest_sample=True))], y)
    assert np.array_equal(y[slice(*argtrim(y, delta, 0, None, nearest_sample=False))], y)
    assert np.array_equal(y[slice(*argtrim(y, delta, 0, None, nearest_sample=True))], y)
    assert np.array_equal(y[slice(*argtrim(y, delta, None, 5, nearest_sample=False))], y)
    assert np.array_equal(y[slice(*argtrim(y, delta, None, 5, nearest_sample=True))], y)
    
    # test nearest sample. xmin is in between x[0 and x[1], but closer to x[0]
    xmin = signal_x[0] + old_div((signal_x[0]+signal_x[1]), 3.0)
    assert np.array_equal(y[slice(*argtrim(y, delta, xmin, None, nearest_sample=False))], y[1:])
    assert np.array_equal(y[slice(*argtrim(y, delta, xmin, None, nearest_sample=True))], y)
    
    # test nearest sample. xmax is in between x[0 and x[1], but closer to x[1]
    xmax = signal_x[0] + old_div((signal_x[0]+signal_x[1]), 1.5)
    assert np.array_equal(y[slice(*argtrim(y, delta, None, xmax, nearest_sample=False))], y[:1])
    assert np.array_equal(y[slice(*argtrim(y, delta, None, xmax, nearest_sample=True))], y[:2])
    
    # test out of bound x
    xmin = signal_x[0] - 1
    assert np.array_equal(y[slice(*argtrim(y, delta, xmin, None, nearest_sample=False))], y)
    assert np.array_equal(y[slice(*argtrim(y, delta, xmin, None, nearest_sample=True))], y)
    xmin = signal_x[-1] + 1
    assert np.array_equal(y[slice(*argtrim(y, delta, xmin, None, nearest_sample=False))], [])
    assert np.array_equal(y[slice(*argtrim(y, delta, xmin, None, nearest_sample=True))], [])
    
    
    # test out of bound x
    xmax = signal_x[0] - 1
    assert np.array_equal(y[slice(*argtrim(y, delta, None, xmax, nearest_sample=False))], [])
    assert np.array_equal(y[slice(*argtrim(y, delta, None, xmax, nearest_sample=True))], [])
    xmax = signal_x[-1] + 1
    assert np.array_equal(y[slice(*argtrim(y, delta, None, xmax, nearest_sample=False))], y)
    assert np.array_equal(y[slice(*argtrim(y, delta, None, xmax, nearest_sample=True))], y)
    
    
    # test out of bound x
    xmin = signal_x[0] - 1
    xmax = signal_x[-1] + 1
    assert np.array_equal(y[slice(*argtrim(y, delta, xmin, xmax, nearest_sample=False))], y)
    assert np.array_equal(y[slice(*argtrim(y, delta, xmin, xmax, nearest_sample=True))], y)
    
    # out of bounds inversed:
    xmax = signal_x[0] - 1
    xmin = signal_x[-1] + 1
    assert np.array_equal(y[slice(*argtrim(y, delta, xmin, xmax, nearest_sample=False))], [])
    assert np.array_equal(y[slice(*argtrim(y, delta, xmin, xmax, nearest_sample=True))], [])
    

@pytest.mark.parametrize('arr, normalize, expected_result, ',
                          [([-1, 1], True, [0, 1]),
                           ([-1, 1], False, [1, 2]),
                           ([-2, 3], True, [0, 1]),
                           ([-2, 3], False, [4, 4+9]),
                           ([], False, []),
                           ([], True, []),
                           ])
@mock.patch('stream2segment.process.funclib.ndarrays.np')
def test_cumsum(mock_np, arr, normalize, expected_result):
    mock_np.cumsum = mock.Mock(side_effect = lambda *a, **k: np.cumsum(*a, **k))
    mock_np.square =  mock.Mock(side_effect = lambda *a, **k: np.square(*a, **k))
    mock_np.nanmax =  mock.Mock(side_effect = lambda *a, **k: np.nanmax(*a, **k))
    mock_np.nanmin =  mock.Mock(side_effect = lambda *a, **k: np.nanmin(*a, **k))
    # mock_np.true_divide =  mock.Mock(side_effect = lambda *a, **k: np.true_divide(*a, **k))
    mock_np.isnan = mock.Mock(side_effect = lambda *a, **k: np.isnan(*a, **k))
    r = cumsumsq(arr, normalize=normalize)
    assert len(r) == len(arr)
    assert (r == np.array(expected_result)).all()
    assert mock_np.cumsum.called
    assert mock_np.square.called

    if arr:
        assert mock_np.isnan.called == normalize
    else:
        assert not mock_np.isnan.called
    assert mock_np.nanmax.called == (normalize and np.isnan(arr).any())
    assert mock_np.nanmin.called == False
#     if normalize:
#         assert mock_np.isnan.called
#         assert mock_np.max.called
#         assert mock_np.true_divide.called
#     else:
#         assert not mock_np.max.called
#         assert not mock_np.true_divide.called


@mock.patch('stream2segment.process.funclib.ndarrays.np')
def test_cumsum_errs(mock_np):
    mock_np.cumsum = mock.Mock(side_effect = lambda *a, **k: np.cumsum(*a, **k))
    mock_np.square =  mock.Mock(side_effect = lambda *a, **k: np.square(*a, **k))
    mock_np.nanmax =  mock.Mock(side_effect = lambda *a, **k: np.nanmax(*a, **k))
    mock_np.nanmin =  mock.Mock(side_effect = lambda *a, **k: np.nanmin(*a, **k))
    # mock_np.true_divide =  mock.Mock(side_effect = lambda *a, **k: np.true_divide(*a, **k))
    mock_np.isnan = mock.Mock(side_effect = lambda *a, **k: np.isnan(*a, **k))
    
    def compute_cumsum(*args, **kwargs):
        '''reset mocks, computes cumsumsq, asserts that the specified function have been called
        and returns cumsumsq'''
        mock_np.cumsum.reset_mock()
        mock_np.square.reset_mock()
        mock_np.nanmax.reset_mock()
        mock_np.nanmin.reset_mock()
        mock_np.isnan.reset_mock()
        r = cumsumsq(*args, **kwargs)
        assert mock_np.cumsum.called
        assert mock_np.square.called
        return r
    
    arr = [0, 0]
    r = compute_cumsum(arr, normalize=True)
    assert len(r) == len(arr)
    assert (r == np.array(arr)).all()
    assert mock_np.isnan.called
    assert not mock_np.nanmax.called
    assert not mock_np.nanmin.called
    # assert not mock_np.true_divide.called
    
    arr = [1, 3, float('nan'), 5, 6]
    r = compute_cumsum(arr, normalize=False)
    assert len(r) == len(arr)
    assert (r[0] == 1) and np.isnan(r[2:]).all()
    assert not mock_np.isnan.called
    assert not mock_np.nanmax.called
    assert not mock_np.nanmin.called
    
    arr = [1, 3, float('nan'), 5, 6]
    r = compute_cumsum(arr, normalize=True)
    assert len(r) == len(arr)
    assert (r[0] == 0) and np.isnan(r[2:]).all()
    assert mock_np.isnan.called
    assert mock_np.nanmax.called
    assert not mock_np.nanmin.called
    
    arr = [float('nan'), 1,3,4]
    r = compute_cumsum(arr, normalize=True)
    assert len(r) == len(arr)
    assert np.isnan(r).all()
    assert mock_np.isnan.called
    assert not mock_np.nanmax.called
    assert not mock_np.nanmin.called
    
    r = compute_cumsum(arr, normalize=False)
    assert len(r) == len(arr)
    assert np.isnan(r).all()
    assert not mock_np.isnan.called
    assert not mock_np.nanmax.called
    assert not mock_np.nanmin.called


# IMPORTANT READ:
# we mock np.true_divide ASSUMING IT's ONY CALLED WITHIN snr!!
# IF we change in the future (we shouldn't), then be aware that the call check might differ!
# impoortant: we must import np.true_divide at module level to avoid recursion problems:
@patch("stream2segment.process.funclib.ndarrays.np.true_divide",
       side_effect=np_true_divide)  # lambda *a, **v: np_true_divide(*a, **v))
def test_snr(mock_np_true_divide):

    signal = np.array([0, 1, 2, 3, 4, 5, 6])
    noise = np.array([0, 1, 2, 3, 4, 5, 6])

    for sf in ('fft', 'dft', 'amp', 'pow', ''):
        assert snr(signal, noise, signals_form=sf, fmin=None, fmax=None, delta_signal=1,
                   delta_noise=1, in_db=False) == 1

    noise[0] += 1
    for sf in ('fft', 'dft', 'amp', 'pow', ''):
        assert snr(signal, noise, signals_form=sf, fmin=None, fmax=None, delta_signal=1,
                   delta_noise=1, in_db=False) < 1

    # assert the snr is one if we take particular frequencies:
    delta_t = 0.01
    delta_f = dfreq(signal, delta_t)
    fmin = delta_f+old_div(delta_f, 100.0)  # just enough to remove first sample
    for sf in ('fft', 'dft', 'amp', 'pow'):
        assert snr(signal, noise, signals_form=sf, fmin=fmin, fmax=None, delta_signal=delta_f,
                   delta_noise=delta_f, in_db=False) == 1
    # now same case as above, but with signals given as time series:
    res = snr(signal, noise, signals_form='', fmin=fmin, fmax=None, delta_signal=delta_t,
              delta_noise=delta_t, in_db=False)
    sspec = powspec(signal, False)[1:]
    nspec = powspec(noise, False)[1:]
    assert (np.sum(sspec) > np.sum(nspec) and res > 1) or \
        (np.sum(sspec) < np.sum(nspec) and res < 1) or \
        (np.sum(sspec) == np.sum(nspec) and res == 1)

    signal[0] += 5
    for sf in ('fft', 'dft', 'amp', 'pow', ''):
        assert snr(signal, noise, signals_form=sf, fmin=None, fmax=None, delta_signal=1,
                   delta_noise=1, in_db=False) > 1

    # test fmin set:
    signal = np.array([0, 1, 2, 3, 4, 5, 6])
    noise = np.array([0, 1, 2, 3, 4, 5, 6])
    delta_t = 0.01
    delta_f = dfreq(signal, delta_t)
    for sf in ('', 'fft', 'dft', 'amp', 'pow'):
        delta = delta_t if not sf else delta_f
        expected_leng_s = len(signal if sf else freqs(signal, delta_t))
        expected_leng_n = len(noise if sf else freqs(noise, delta_t))
        mock_np_true_divide.reset_mock()
        assert snr(signal, noise, signals_form=sf, fmin=delta_f, fmax=None, delta_signal=delta,
                   delta_noise=delta, in_db=False) == 1
        # we called np.true_divide 2 times
        # 1 for normalizing noise
        # 1 for normalizing signal
        # thus
        assert len(mock_np_true_divide.call_args_list) == 2
        # assert that when normalizing the second arg (number of points) is the expected fft
        # number of points minus 1 cause we set fmin=delta_f (ignore first freq bin)
        assert mock_np_true_divide.call_args_list[-2][0][1] == expected_leng_s - 1  # signal
        assert mock_np_true_divide.call_args_list[-1][0][1] == expected_leng_n - 1  # noise

    # test fmin set but negative (same as missing)
    delta_t = 0.01
    delta_f = dfreq(signal, delta_t)
    for sf in ('', 'fft', 'dft', 'amp', 'pow'):
        delta = delta_t if not sf else delta_f
        expected_leng_s = len(signal if sf else freqs(signal, delta_t))
        expected_leng_n = len(noise if sf else freqs(noise, delta_t))
        mock_np_true_divide.reset_mock()
        assert snr(signal, noise, signals_form=sf, fmin=-delta_f, fmax=None, delta_signal=delta,
                   delta_noise=delta, in_db=False) == 1
        # we called np.true_divide 2 times:
        # 1 for normalizing noise
        # 1 for normalizing signal
        # thus
        assert len(mock_np_true_divide.call_args_list) == 2
        # assert that when normalizing the second arg (number of points) is the expected fft
        # number of points minus 1 cause we set fmin=delta_f (ignore first freq bin)
        assert mock_np_true_divide.call_args_list[-2][0][1] == expected_leng_s  # signal
        assert mock_np_true_divide.call_args_list[-1][0][1] == expected_leng_n  # noise

    # test fmax set:
    signal = np.array([0, 1, 2, 3, 4, 5, 6])
    noise = np.array([0, 1, 2, 3, 4, 5, 6])
    delta_t = 0.01
    delta_f = dfreq(signal, delta_t)
    for sf in ('', 'fft', 'dft', 'amp', 'pow'):
        delta = delta_t if not sf else delta_f
        expected_leng_s = len(signal if sf else freqs(signal, delta_t))
        expected_leng_n = len(noise if sf else freqs(noise, delta_t))
        mock_np_true_divide.reset_mock()
        # we need to change expected val. If signal is time series, we run the fft and thus we have
        # a first non-zero point. Otherwise the first point (the only one we take according to fmax)
        # is zero thus we should have nan
        if not sf:
            assert snr(signal, noise, signals_form=sf, fmin=None, fmax=delta_f, delta_signal=delta,
                       delta_noise=delta, in_db=False) == 1
        else:
            np.isnan(snr(signal, noise, signals_form=sf, fmin=None, fmax=delta_f,
                         delta_signal=delta, delta_noise=delta, in_db=False)).all()
        # assert when normalizing we called a slice of signal and noise with the first element
        # removed due to the choice of delta_f and delta
        signal_call = mock_np_true_divide.call_args_list[-2][0]
        noise_call = mock_np_true_divide.call_args_list[-1][0]
        assert signal_call[1] == 2  # fmax removes all BUT first 2 frequencies
        assert noise_call[1] == 2  # fmax removes all BUT first 2 frequencies

    # test fmax set but negative (same as missing)
    delta_t = 0.01
    delta_f = dfreq(signal, delta_t)
    for sf in ('', 'fft', 'dft', 'amp', 'pow'):
        delta = delta_t if not sf else delta_f
        expected_leng_s = len(signal if sf else freqs(signal, delta_t))
        expected_leng_n = len(noise if sf else freqs(noise, delta_t))
        mock_np_true_divide.reset_mock()
        assert np.isnan(snr(signal, noise, signals_form=sf, fmin=None, fmax=-delta_f,
                            delta_signal=delta, delta_noise=delta, in_db=False)).all()
        # assert we did not call true_divide as many times as before
        # as empty arrays (because fmax=-delta_f)
        # are skipped:
        assert len(mock_np_true_divide.call_args_list) == 0


def test_triangsmooth():
    data = [3.7352e+06,
            1.104e+06,
            1.088e+06,
            1.0695e+06,
            7.1923e+05,
            1.2757e+06,
            1.2596e+05,
            9.4364e+05,
            5.8868e+05,
            4.4942e+05,
            6.768e+05,
            4.0295e+05,
            5.1843e+05,
            6.2502e+05,
            4.6077e+05,
            1.4937e+05,
            5.366e+05,
            1.4942e+05,
            2.4361e+05,
            3.5926e+05]
    # test a smooth function. take a parabola
    win_ratio = 0.04
    smooth = triangsmooth(np.array(data), winlen_ratio=win_ratio)
    assert all([smooth[i] <= max(data[i-1:i+2]) and smooth[i] >= min(data[i-1:i+2])
                for i in range(1, len(data)-1)])
    assert np.allclose(smooth, triangsmooth0(np.array(data), win_ratio), rtol=1e-05, atol=1e-08,
                       equal_nan=True)

    data = [x**2 for x in range(115)]
    smooth = triangsmooth(np.array(data), winlen_ratio=win_ratio)
    assert np.allclose(smooth, data, rtol=1e-03, atol=1e-08, equal_nan=True)
    assert np.allclose(smooth, triangsmooth0(np.array(data), win_ratio), rtol=1e-05, atol=1e-08,
                       equal_nan=True)


def triangsmooth0(spectrum, alpha):
    """First working implementation of triangsmooth (or bartlettsmooth).
    Used to check that current version produces the same results as this one"""
    spectrum_ = np.array(spectrum, dtype=float)

    leng = len(spectrum)
    # get the number of points (left branch, center if leng odd, right branch = left reversed)
    nptsl = np.arange(leng // 2)
    nptsr = nptsl[::-1]
    nptsc = np.array([]) if leng % 2 == 0 else np.array([1 + leng // 2])
    # get the array with the interval number of points for each i
    # use np.concatenate((nptsl, nptsc, nptsr)) as array of maxima (to avoid overflow at boundary)
    npts = np.around(np.minimum(np.concatenate((nptsl, nptsc, nptsr)),
                                np.arange(leng, dtype=float) * alpha)).astype(int)
    del nptsl, nptsc, nptsr  # frees up memory?
    npts_u = np.unique(npts)

    startindex = 0
    try:
        startindex = np.argwhere(npts_u <= 1)[-1][0] + 1
    except IndexError:
        pass

    for n in npts_u[startindex:]:
        # n_2 = np.true_divide(2*n-1, 2)
        tri = (1 - np.abs(np.true_divide(np.arange(2*n + 1) - n, n)))
        idxs = np.argwhere(npts == n)
        spec_slices = spectrum[idxs-n + np.arange(2*n+1)]
        spectrum_[idxs.flatten()] = old_div(np.sum(tri * spec_slices, axis=1), np.sum(tri))

    return spectrum_


@pytest.mark.parametrize('start, delta, num',
                         list(product([0, -3.4], [1, 0.004], [0, 45]))
                         )
def test_arange_equals_linspace(start, delta, num):
    a1, a2 = start + np.arange(num) * delta, linspace(start, delta, num)
    if num == 0:
        assert not len(a1) and not len(a2)
    else:
        assert (a1 == a2).all()


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

