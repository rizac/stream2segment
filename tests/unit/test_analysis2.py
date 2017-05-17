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
from stream2segment.analysis import amp_spec, triangsmooth, snr, dfreq, freqs, pow_spec
from itertools import izip
import pytest
from mock.mock import patch, Mock

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
        assert snr(signal, noise, signals_form=sf, fmin=None, fmax=None, delta_s=1, delta_n=1, in_db=False) == 1

    noise[0]+=1
    for sf in ('fft', 'dft', 'amp', 'pow', ''):
        assert snr(signal, noise, signals_form=sf, fmin=None, fmax=None, delta_s=1, delta_n=1, in_db=False) < 1

    # assert the snr is one if we take particular frequencies:
    delta_t = 0.01
    delta_f = dfreq(signal, delta_t)
    fmin = delta_f+delta_f/100.0  # just enough to remove first sample
    for sf in ('fft', 'dft', 'amp', 'pow'):
        assert snr(signal, noise, signals_form=sf, fmin=fmin, fmax=None, delta_s=delta_f, delta_n=delta_f, in_db=False) == 1
    # now same case as above, but with signals given as time series:
    res = snr(signal, noise, signals_form='', fmin=fmin, fmax=None, delta_s=delta_t, delta_n=delta_t, in_db=False)
    sspec = pow_spec(signal,False)[1:]
    nspec = pow_spec(noise,False)[1:]
    assert (np.sum(sspec) > np.sum(nspec) and res > 1) or (np.sum(sspec) < np.sum(nspec) and res < 1) or \
        (np.sum(sspec) == np.sum(nspec) and res ==1)
    
    
    signal[0] += 5
    for sf in ('fft', 'dft', 'amp', 'pow', ''):
        assert snr(signal, noise, signals_form=sf, fmin=None, fmax=None, delta_s=1, delta_n=1, in_db=False) > 1
    
    # test fmin set:
    signal = np.array([0,1,2,3,4,5,6])
    noise = np.array([0,1,2,3,4,5,6])
    delta_t = 0.01
    delta_f = dfreq(signal, delta_t)
    for sf in ('', 'fft', 'dft', 'amp', 'pow'):
        delta = delta_t if not sf else delta_f
        expected_leng_s = len(freqs(signal, delta_t, signal_is_timeseries=not sf))
        expected_leng_n = len(freqs(noise, delta_t, signal_is_timeseries=not sf))
        mock_np_true_divide.reset_mock()
        assert snr(signal, noise, signals_form=sf, fmin=delta_f, fmax=None, delta_s=delta, delta_n=delta, in_db=False) == 1
        # assert when normalizing we called a slice of signal and noise with the first element removed due
        # to the choice of delta_f and delta
        signal_call = mock_np_true_divide.call_args_list[0][0]
        noise_call = mock_np_true_divide.call_args_list[1][0]
        assert signal_call[1] ==expected_leng_s-1  # fmin removes first frequency
        assert noise_call[1] == expected_leng_n-1  # fmin removes first frequency
    
    # test fmin set but negative (same as missing)
    delta_t = 0.01
    delta_f = dfreq(signal, delta_t)
    for sf in ('', 'fft', 'dft', 'amp', 'pow'):
        delta = delta_t if not sf else delta_f
        expected_leng_s = len(freqs(signal, delta_t, signal_is_timeseries=not sf))
        expected_leng_n = len(freqs(noise, delta_t, signal_is_timeseries=not sf))
        mock_np_true_divide.reset_mock()
        assert snr(signal, noise, signals_form=sf, fmin=-delta_f, fmax=None, delta_s=delta, delta_n=delta, in_db=False) == 1
        # assert when normalizing we called a slice of signal and noise with the first element removed due
        # to the choice of delta_f and delta
        signal_call = mock_np_true_divide.call_args_list[0][0]
        noise_call = mock_np_true_divide.call_args_list[1][0]
        assert signal_call[1] ==expected_leng_s  # fmin does not remove first frequency
        assert noise_call[1] == expected_leng_n  # fmin does not remove first frequency
        
    # test fmax set:
    signal = np.array([0,1,2,3,4,5,6])
    noise = np.array([0,1,2,3,4,5,6])
    delta_t = 0.01
    delta_f = dfreq(signal, delta_t)
    for sf in ('', 'fft', 'dft', 'amp', 'pow'):
        delta = delta_t if not sf else delta_f
        expected_leng_s = len(freqs(signal, delta_t, signal_is_timeseries=not sf))
        expected_leng_n = len(freqs(noise, delta_t, signal_is_timeseries=not sf))
        mock_np_true_divide.reset_mock()
        # we need to change expected val. If signal is time series, we run the fft and thus we have a
        # first non-zero point. Otherwise the first point (the only one we take according to fmax)
        # is zero thus we should have nan
        if not sf:
            assert snr(signal, noise, signals_form=sf, fmin=None, fmax=delta_f, delta_s=delta, delta_n=delta, in_db=False) == 1
        else:
            np.isnan(snr(signal, noise, signals_form=sf, fmin=None, fmax=delta_f, delta_s=delta, delta_n=delta, in_db=False)).all()
        # assert when normalizing we called a slice of signal and noise with the first element removed due
        # to the choice of delta_f and delta
        signal_call = mock_np_true_divide.call_args_list[0][0]
        noise_call = mock_np_true_divide.call_args_list[1][0]
        assert signal_call[1] == 1  # fmax removes all BUT first frequency
        assert noise_call[1] == 1  # fmax removes all BUT first frequency

    # test fmax set but negative (same as missing)
    delta_t = 0.01
    delta_f = dfreq(signal, delta_t)
    for sf in ('', 'fft', 'dft', 'amp', 'pow'):
        delta = delta_t if not sf else delta_f
        expected_leng_s = len(freqs(signal, delta_t, signal_is_timeseries=not sf))
        expected_leng_n = len(freqs(noise, delta_t, signal_is_timeseries=not sf))
        mock_np_true_divide.reset_mock()
        assert np.isnan(snr(signal, noise, signals_form=sf, fmin=None, fmax=-delta_f, delta_s=delta, delta_n=delta, in_db=False)).all()
        # assert we did not call true_divide as empty arrays are skipped:
        # (and we have empty arrays due to the choice of fmax<0)
        signal_call = mock_np_true_divide.call_args_list
        noise_call = mock_np_true_divide.call_args_list
        assert signal_call == []
        assert noise_call == []

@pytest.mark.parametrize('matlab_data',
                      [
                       ([[3.7352e+06,3.7352e+06,1.9811e+06,1.9811e+06],
                         [1.104e+06,1.104e+06,1.5286e+06,1.5345e+06],
                         [1.088e+06,1.088e+06,1.3399e+06,1.2977e+06],
                         [1.0695e+06,1.0695e+06,1.1382e+06,1.0249e+06],
                         [7.1923e+05,7.1923e+05,1.016e+06,8.8141e+05],
                         [1.2757e+06,1.2757e+06,9.2363e+05,7.8554e+05],
                         [1.2596e+05,1.2596e+05,8.6884e+05,7.3936e+05],
                         [9.4364e+05,9.4364e+05,7.9957e+05,6.668e+05],
                         [5.8868e+05,5.8868e+05,7.4655e+05,6.4268e+05],
                         [4.4942e+05,4.4942e+05,6.9982e+05,5.9822e+05],
                         [6.768e+05,6.768e+05,6.9982e+05,5.5776e+05],
                         [4.0295e+05,4.0295e+05,6.9982e+05,5.1575e+05],
                         [5.1843e+05,5.1843e+05,6.9982e+05,4.8452e+05],
                         [6.2502e+05,6.2502e+05,6.9982e+05,4.5061e+05],
                         [4.6077e+05,4.6077e+05,6.9982e+05,4.5061e+05],
                         [1.4937e+05,1.4937e+05,6.9982e+05,4.5196e+05],
                         [5.366e+05,5.366e+05,6.9982e+05,4.5196e+05],
                         [1.4942e+05,1.4942e+05,6.9982e+05,4.2709e+05],
                         [2.4361e+05,2.4361e+05,6.9982e+05,4.2709e+05],
                         [3.5926e+05,3.5926e+05,6.9982e+05,4.0363e+05]])
                       ],
                    )
def tst_triangsmooth(matlab_data):
    # data columns are: input_data,smooth0.01,smooth0.99,smooth0.50
    data = []
    win_ratios = [0.01, 0.99, 0.5]
    dict = {wr:[] for wr in win_ratios}
    for d in matlab_data:
        data.append(d[0])
        for i, wr in enumerate(win_ratios):
            dict[wr].append(d[i+1])
    
    for alpha in win_ratios:
        smooth = triangsmooth(np.array(data), winlen_ratio=alpha)
        try:
            dfreq=0.1
            _ = smooth_M2_old((np.arange(len(smooth)+1)*dfreq)[1:], np.array(data), 2*len(smooth), dfreq, alpha)
        except:
            pass
        s1 = smooth
        s2 = dict[alpha]
        s3 = _
        h = 9
        assert np.allclose(smooth, dict[alpha], rtol=1e-05, atol=1e-08, equal_nan=True)
        

def smooth_M2_old(frq,acps,N,fnimin,vlisc):
#vlisc controlla la larghezza della finestra, per esempio 0.1
#occhio a N, deve essere il doppio della lunghezza della finestra
#frq sono le frequenze e acps lo spettro
#fmin e' delta_f
# io generalmente non ho la frequenza 0 nel vettore delle freq e nello spettro
# quindi delta_f=freq(1)
#EG:
#[spesmT] = smooth_M2(freq,spesel,2*length(spesel),freq(1),0.1);
#
# smoothing of the Spectra with triangular window
    NN=int(N/2)
    fh=np.zeros(NN)
    vxxe=np.zeros(NN)
    
    for io2 in range(0,int(NN)):
        io2=int(io2)
        fr=frq[io2]
        ifr25=int(round((fr*vlisc)/fnimin))
        ifp = io2+ifr25
        ifm = io2-ifr25
        if ifm != ifp: 
            if  ifp > N/2:  
                ifp = NN
               
            if  ifm < 0: 
                ifm = 0
             
            vnk = ifp-ifm+1
            ivnk = int(round(vnk/2.))
            il=0
            for ij in range(-ivnk,ivnk+1):
                ij=int(ij)
                fh[il]=(1.-(np.abs(ij)/vnk))
                il=il+1
           
            # nfh=il-1
            if ifm == ifp:
                fh[0]=1.
              
            l=0
            uvxx=0.
            nor=0
            ij=0
            for ii in range(ifm,ifp+1):
                ii=int(ii)
                uvxx=uvxx+acps[ii]*fh[ij]
                nor=nor+fh[ij]
                ij=ij+1
                l=l+1
           
            vxxe[io2]=uvxx/nor
        
        else: 
            vxxe[io2]=acps[io2]
    return vxxe




if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()