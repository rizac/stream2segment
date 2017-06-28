'''
Created on May 12, 2017

@author: riccardo
'''
import unittest
import numpy as np
from numpy.fft import rfft
from numpy import abs
from numpy import true_divide as np_true_divide
from obspy.core.stream import read as o_read
from io import BytesIO
import os
from stream2segment.analysis.mseeds import fft
from stream2segment.analysis import ampspec, triangsmooth, snr, dfreq, freqs, powspec
from itertools import izip
import pytest
from mock.mock import patch, Mock
from datetime import datetime
from obspy.core.utcdatetime import UTCDateTime

class Test(unittest.TestCase):


    def setUp(self):
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)),"data", "trace_GE.APE.mseed"), 'rb') as opn:
            self.mseed = o_read(BytesIO(opn.read()))
            self.fft = fft(self.mseed)
        pass


    def tearDown(self):
        pass


    def tstName(self):
        pass

# IMPORTANT READ:
# we mock np.true_divide ASSUMING IT's ONY CALLED WITHIN snr!!
# IF we change in the future (we shouldn't), then be aware that the call check might differ!
# impoortant: we must import np.true_divide at module level to avoid recursion problems:
@patch("stream2segment.analysis.np.true_divide", side_effect=lambda *a, **v: np_true_divide(*a, **v))
def test_snr(mock_np_true_divide):

    signal = np.array([0,1,2,3,4,5,6])
    noise = np.array([0,1,2,3,4,5,6])

    for sf in ('fft', 'dft', 'amp', 'pow', ''):
        assert snr(signal, noise, signals_form=sf, fmin=None, fmax=None, delta_signal=1, delta_noise=1, in_db=False) == 1

    noise[0]+=1
    for sf in ('fft', 'dft', 'amp', 'pow', ''):
        assert snr(signal, noise, signals_form=sf, fmin=None, fmax=None, delta_signal=1, delta_noise=1, in_db=False) < 1

    # assert the snr is one if we take particular frequencies:
    delta_t = 0.01
    delta_f = dfreq(signal, delta_t)
    fmin = delta_f+delta_f/100.0  # just enough to remove first sample
    for sf in ('fft', 'dft', 'amp', 'pow'):
        assert snr(signal, noise, signals_form=sf, fmin=fmin, fmax=None, delta_signal=delta_f, delta_noise=delta_f, in_db=False) == 1
    # now same case as above, but with signals given as time series:
    res = snr(signal, noise, signals_form='', fmin=fmin, fmax=None, delta_signal=delta_t, delta_noise=delta_t, in_db=False)
    sspec = powspec(signal,False)[1:]
    nspec = powspec(noise,False)[1:]
    assert (np.sum(sspec) > np.sum(nspec) and res > 1) or (np.sum(sspec) < np.sum(nspec) and res < 1) or \
        (np.sum(sspec) == np.sum(nspec) and res ==1)
    
    
    signal[0] += 5
    for sf in ('fft', 'dft', 'amp', 'pow', ''):
        assert snr(signal, noise, signals_form=sf, fmin=None, fmax=None, delta_signal=1, delta_noise=1, in_db=False) > 1
    
    # test fmin set:
    signal = np.array([0,1,2,3,4,5,6])
    noise = np.array([0,1,2,3,4,5,6])
    delta_t = 0.01
    delta_f = dfreq(signal, delta_t)
    for sf in ('', 'fft', 'dft', 'amp', 'pow'):
        delta = delta_t if not sf else delta_f
        expected_leng_s = len(signal if sf else freqs(signal, delta_t))
        expected_leng_n = len(noise if sf else freqs(noise, delta_t))
        mock_np_true_divide.reset_mock()
        assert snr(signal, noise, signals_form=sf, fmin=delta_f, fmax=None, delta_signal=delta, delta_noise=delta, in_db=False) == 1
        # assert when normalizing we worked on a slice of signal and noise with the first element removed due
        # to the choice of delta_f and delta
        signal_call = mock_np_true_divide.call_args_list[0][0]
        noise_call = mock_np_true_divide.call_args_list[1][0]

        assert noise_call[1] == expected_leng_n - 1  # fmin removes first frequency

    # test fmin set but negative (same as missing)
    delta_t = 0.01
    delta_f = dfreq(signal, delta_t)
    for sf in ('', 'fft', 'dft', 'amp', 'pow'):
        delta = delta_t if not sf else delta_f
        expected_leng_s = len(signal if sf else freqs(signal, delta_t))
        expected_leng_n = len(noise if sf else freqs(noise, delta_t))
        mock_np_true_divide.reset_mock()
        assert snr(signal, noise, signals_form=sf, fmin=-delta_f, fmax=None, delta_signal=delta, delta_noise=delta, in_db=False) == 1
        # assert when normalizing we called a slice of signal and noise with the first element removed due
        # to the choice of delta_f and delta
        signal_call = mock_np_true_divide.call_args_list[0][0]
        noise_call = mock_np_true_divide.call_args_list[1][0]
        assert signal_call[1] == expected_leng_s  # fmin does not remove first frequency
        assert noise_call[1] == expected_leng_n  # fmin does not remove first frequency

    # test fmax set:
    signal = np.array([0,1,2,3,4,5,6])
    noise = np.array([0,1,2,3,4,5,6])
    delta_t = 0.01
    delta_f = dfreq(signal, delta_t)
    for sf in ('', 'fft', 'dft', 'amp', 'pow'):
        delta = delta_t if not sf else delta_f
        expected_leng_s = len(signal if sf else freqs(signal, delta_t))
        expected_leng_n = len(noise if sf else freqs(noise, delta_t))
        mock_np_true_divide.reset_mock()
        # we need to change expected val. If signal is time series, we run the fft and thus we have a
        # first non-zero point. Otherwise the first point (the only one we take according to fmax)
        # is zero thus we should have nan
        if not sf:
            assert snr(signal, noise, signals_form=sf, fmin=None, fmax=delta_f, delta_signal=delta, delta_noise=delta, in_db=False) == 1
        else:
            np.isnan(snr(signal, noise, signals_form=sf, fmin=None, fmax=delta_f, delta_signal=delta, delta_noise=delta, in_db=False)).all()
        # assert when normalizing we called a slice of signal and noise with the first element removed due
        # to the choice of delta_f and delta
        signal_call = mock_np_true_divide.call_args_list[0][0]
        noise_call = mock_np_true_divide.call_args_list[1][0]
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
        assert np.isnan(snr(signal, noise, signals_form=sf, fmin=None, fmax=-delta_f, delta_signal=delta, delta_noise=delta, in_db=False)).all()
        # assert we did not call true_divide as empty arrays are skipped:
        # (and we have empty arrays due to the choice of fmax<0)
        signal_call = mock_np_true_divide.call_args_list
        noise_call = mock_np_true_divide.call_args_list
        assert signal_call == []
        assert noise_call == []

# @pytest.mark.parametrize('matlab_data',
#                       [
#                        ([[3.7352e+06,3.7352e+06,1.9811e+06,1.9811e+06],
#                          [1.104e+06,1.104e+06,1.5286e+06,1.5345e+06],
#                          [1.088e+06,1.088e+06,1.3399e+06,1.2977e+06],
#                          [1.0695e+06,1.0695e+06,1.1382e+06,1.0249e+06],
#                          [7.1923e+05,7.1923e+05,1.016e+06,8.8141e+05],
#                          [1.2757e+06,1.2757e+06,9.2363e+05,7.8554e+05],
#                          [1.2596e+05,1.2596e+05,8.6884e+05,7.3936e+05],
#                          [9.4364e+05,9.4364e+05,7.9957e+05,6.668e+05],
#                          [5.8868e+05,5.8868e+05,7.4655e+05,6.4268e+05],
#                          [4.4942e+05,4.4942e+05,6.9982e+05,5.9822e+05],
#                          [6.768e+05,6.768e+05,6.9982e+05,5.5776e+05],
#                          [4.0295e+05,4.0295e+05,6.9982e+05,5.1575e+05],
#                          [5.1843e+05,5.1843e+05,6.9982e+05,4.8452e+05],
#                          [6.2502e+05,6.2502e+05,6.9982e+05,4.5061e+05],
#                          [4.6077e+05,4.6077e+05,6.9982e+05,4.5061e+05],
#                          [1.4937e+05,1.4937e+05,6.9982e+05,4.5196e+05],
#                          [5.366e+05,5.366e+05,6.9982e+05,4.5196e+05],
#                          [1.4942e+05,1.4942e+05,6.9982e+05,4.2709e+05],
#                          [2.4361e+05,2.4361e+05,6.9982e+05,4.2709e+05],
#                          [3.5926e+05,3.5926e+05,6.9982e+05,4.0363e+05]])
#                        ],
#                     )
# def tst_triangsmooth(matlab_data):
#     # data columns are: input_data,smooth0.01,smooth0.99,smooth0.50
#     data = [0]
#     win_ratios = [0.01, 0.99, 0.5]
#     dict = {wr:[] for wr in win_ratios}
#     for d in matlab_data:
#         data.append(d[0])
#         for i, wr in enumerate(win_ratios):
#             dict[wr].append(d[i+1])
#     
#     for alpha in win_ratios:
#         smooth = triangsmooth(np.array(data), winlen_ratio=alpha)
#         try:
#             dfreq=0.1
#             _ = smooth_M2_old((np.arange(len(smooth)+1)*dfreq)[1:], np.array(data), 2*len(smooth), dfreq, alpha)
#         except:
#             pass
#         s1 = smooth
#         s2 = dict[alpha]
#         s3 = _
#         h = 9
#         assert np.allclose(smooth[1:], dict[alpha], rtol=1e-05, atol=1e-08, equal_nan=True)


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
    assert all([smooth[i]<=max(data[i-1:i+2]) and smooth[i]>=min(data[i-1:i+2]) for i in xrange(1, len(data)-1)])
    assert np.allclose(smooth, triangsmooth0(np.array(data), win_ratio), rtol=1e-05, atol=1e-08, equal_nan=True)


    data = [x**2 for x in xrange(115)]
    smooth = triangsmooth(np.array(data), winlen_ratio=win_ratio)
    assert np.allclose(smooth, data, rtol=1e-03, atol=1e-08, equal_nan=True)
    assert np.allclose(smooth, triangsmooth0(np.array(data), win_ratio), rtol=1e-05, atol=1e-08, equal_nan=True)


def triangsmooth0(spectrum, alpha):
    """First implementation of triangsmooth (or bartlettsmooth). Used to check that current
    version is equal to the first implemented"""
    spectrum_ = np.array(spectrum, dtype=float)
#     if copy:
#         spectrum = spectrum.copy()

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
        spectrum_[idxs] = np.sum(tri * spec_slices, axis=1)/np.sum(tri)

    return spectrum_







if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()