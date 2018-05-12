#@PydevCodeAnalysisIgnore
'''
Created on Feb 23, 2016

@author: riccardo
'''

from future import standard_library

standard_library.install_aliases()
import mock, os, sys
import pytest
import re
import argparse
from io import BytesIO
import numpy as np
from stream2segment.process.math.ndarrays import fft as orig_fft, linspace, \
    snr as orig_snr, powspec as orig_powspec
from stream2segment.process.math.traces import fft , bandpass, dfreq, maxabs, timeof

from obspy.core.inventory import read_inventory
from obspy.core import read as obspy_read
from obspy.core import Trace, Stream
from io import StringIO
from obspy.io.stationxml.core import _read_stationxml
from obspy.core.trace import Trace
from itertools import count

@pytest.mark.parametrize('arr, arr_len_after_trim, fft_npts',
                        [([1, 2, 3, 4, 5, 6], 6, 4),
                         ([1, 2, 3, 4, 5], 5, 3),
                         ([1, 2, 3, 4], 4, 3),
                         ([1, 2, 3], 3, 2),
                         ])
@mock.patch('stream2segment.process.math.traces._fft', side_effect=lambda *a, **k: orig_fft(*a, **k))
def test_fft(mock_mseed_fft, arr, arr_len_after_trim, fft_npts):
    t = Trace(np.array(arr))
    df, f = fft(t)
    assert len(mock_mseed_fft.call_args[0][0]) == arr_len_after_trim
    assert len(f) == fft_npts
    assert df == dfreq(t.data, t.stats.delta)
    freqs0 = np.linspace(0, len(f) * df, len(f), endpoint=False)
    
    freqs, f = fft(t, return_freqs=True)
    assert (freqs == freqs0).all()  # also assures they have same length
    assert np.allclose(freqs[1] - freqs[0], df)


@pytest.mark.parametrize('start, delta, num',
                        [(0.1, 12, 11),
                         (0, 0.01, 100),
                         (1,1,1),
                         (1.1, 0, 55),
                         (1, 1, 0)
                         ])
def test_linspace(start, delta, num):
    space = linspace(start, delta, num)
    if num == 0:
        assert len(space) == 0
    else:
        expected = np.linspace(start, space[-1], num, endpoint=True)
        assert (space==expected).all()
    

@pytest.fixture(scope="module")
def _data(data):
    """returns a dict with fields 'mseed', 'mseed_ACC', 'mseed_VEL', 'mseed_DISP' (all Streams.
    The last three after removing the response) and  'inventory' (the stream inventory object
    used to remove the response)"""
    inv_name = 'inventory_GE.APE.xml'
    inv_obj = data.read_inv(inv_name)
    ret = {'inventory': data.read_inv(inv_name)}
    for inv_output in [None, 'ACC', 'VEL', 'DISP']:
        key = 'mseed' + ('' if not inv_output else "_" + inv_output)
        ret[key] = data.read_stream('trace_GE.APE.mseed', inv_name if inv_output else None,
                                    inv_output)
    return ret

def test_bandpass(_data):
    trace = _data['mseed'][0]
    res = bandpass(trace, 2, 3)
    assert not np.array_equal(trace.data, res.data)
    assert trace.stats.starttime == res.stats.starttime
    assert trace.stats.endtime == res.stats.endtime
    assert trace.stats.npts == res.stats.npts
    assert len(trace.data) == len(res.data)


def testmaxabs(_data):
    mseed = _data['mseed']
    trace = mseed[0]
    
    t, g = maxabs(trace)

    assert np.max(np.abs(trace.data)) == g
    idx =  np.argmax(np.abs(trace.data))
    
    assert timeof(trace, idx) == t
    
    # assert by slicing times of max are different:
    td = 2 * trace.stats.delta
    assert maxabs(trace, None, t-td)[0] < t < maxabs(trace, t+td, None)[0]
    
    assert np.isnan(maxabs(trace, None, trace.stats.starttime-td))
