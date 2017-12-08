#@PydevCodeAnalysisIgnore
'''
Created on Feb 23, 2016

@author: riccardo
'''
from __future__ import division

from past.utils import old_div
import mock, os, sys
import pytest
import re
import argparse
import numpy as np
import pandas as pd
from stream2segment.process.math.ndarrays import cumsum, argtrim
from scipy.signal import hilbert


from obspy.core.inventory import read_inventory
from obspy.core import read as obspy_read
from obspy.core import Trace, Stream
# from StringIO import StringIO
from obspy.io.stationxml.core import _read_stationxml
from obspy.core.trace import Trace
from itertools import count


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
                          [([-1, 1], True, [0.5, 1]),
                           ([-1, 1], False, [1, 2]),
                           ([-2, 3], True, [old_div(4.0,(4+9)), old_div((4+9.0),(4+9))]),
                           ([-2, 3], False, [4, 4+9]),
                           ])
@mock.patch('stream2segment.process.math.ndarrays.np')
def test_cumsum(mock_np, arr, normalize, expected_result):
    mock_np.cumsum = mock.Mock(side_effect = lambda *a, **k: np.cumsum(*a, **k))
    mock_np.square =  mock.Mock(side_effect = lambda *a, **k: np.square(*a, **k))
    mock_np.max =  mock.Mock(side_effect = lambda *a, **k: np.max(*a, **k))
    # mock_np.true_divide =  mock.Mock(side_effect = lambda *a, **k: np.true_divide(*a, **k))
    mock_np.isnan = mock.Mock(side_effect = lambda *a, **k: np.isnan(*a, **k))
    r = cumsum(arr, normalize=normalize)
    assert len(r) == len(arr)
    assert (r == np.array(expected_result)).all()
    assert mock_np.cumsum.called
    assert mock_np.square.called
    
    assert mock_np.isnan.called == normalize
    assert mock_np.max.called == normalize
#     if normalize:
#         assert mock_np.isnan.called
#         assert mock_np.max.called
#         assert mock_np.true_divide.called
#     else:
#         assert not mock_np.max.called
#         assert not mock_np.true_divide.called


@mock.patch('stream2segment.process.math.ndarrays.np')
def test_cumsum_errs(mock_np):
    mock_np.cumsum = mock.Mock(side_effect = lambda *a, **k: np.cumsum(*a, **k))
    mock_np.square =  mock.Mock(side_effect = lambda *a, **k: np.square(*a, **k))
    mock_np.max =  mock.Mock(side_effect = lambda *a, **k: np.max(*a, **k))
    # mock_np.true_divide =  mock.Mock(side_effect = lambda *a, **k: np.true_divide(*a, **k))
    mock_np.isnan = mock.Mock(side_effect = lambda *a, **k: np.isnan(*a, **k))
    
    arr = [0, 0]
    r = cumsum(arr, normalize=True)
    assert len(r) == len(arr)
    assert (r == np.array(arr)).all()
    assert mock_np.cumsum.called
    assert mock_np.square.called
    assert mock_np.isnan.called
    assert mock_np.max.called
    # assert not mock_np.true_divide.called
    
    arr = [1, float('nan')]
    r = cumsum(arr, normalize=True)
    assert len(r) == len(arr)
    assert (r[0] == 1) and np.isnan(r[1])
    assert mock_np.cumsum.called
    assert mock_np.square.called
    assert mock_np.isnan.called
    assert mock_np.max.called
    # assert not mock_np.true_divide.called
