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
from stream2segment.mathutils.arrays import fft as orig_fft, linspace, \
    snr as orig_snr, powspec as orig_powspec


from stream2segment.mathutils.mseeds import fft , bandpass, dfreq, maxabs,\
    timeof
# from stream2segment.io.utils import loads, dumps

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
@mock.patch('stream2segment.mathutils.mseeds._fft', side_effect=lambda *a, **k: orig_fft(*a, **k))
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
    
    
def test_bandpass():
    trace = get_data()['mseed'][0]
    res = bandpass(trace, 2, 3)
    assert not np.array_equal(trace.data, res.data)
    assert trace.stats.starttime == res.stats.starttime
    assert trace.stats.endtime == res.stats.endtime
    assert trace.stats.npts == res.stats.npts
    assert len(trace.data) == len(res.data)
    
    
    h = 9


__dd = None

def get_data():
    global __dd
    if __dd is None:
        __dd = _data()
    return __dd


@pytest.fixture(scope="session")
def _data():
    """returns a dict with fields 'mseed', 'mseed_ACC', 'mseed_VEL', 'mseed_DISP' (all Streams.
    The latter three after removing the response)
    'inventory' (an inventory object) and two strings: 'mseed_path' and 'inventory_path'"""
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    mseed_path = os.path.join(folder, 'trace_GE.APE.mseed')
    mseed = obspy_read(mseed_path)
    inv_path = os.path.join(folder, 'inventory_GE.APE.xml')
    s = BytesIO()
    with open(inv_path, 'rb') as _opn:
        s.write(_opn.read())
    s.seek(0)
    inv_obj = read_inventory(s)
    ret = {'mseed': mseed, 'inventory': inv_obj, 'mseed_path': mseed_path,
           'data_path': folder, 
           'inventory_path': inv_path}
    for inv_output in ['ACC', 'VEL', 'DISP']:
        # mseed_c = mseed.copy()
        # mseed2 = remove_response(mseed, inv_obj, output=inv_output)
        ret['mseed_'+inv_output] = mseed.copy().remove_response(inv_obj, output=inv_output)
    return ret


# @pytest.mark.parametrize('inv_output',
#                           ['ACC', 'VEL', 'DISP'])
# def test_remove_response_with_inv_path(_data, inv_output):
#     mseed = get_data()['mseed']
#     mseed2 = get_data()['mseed_'+inv_output]
#     assert isinstance(mseed, Stream) == isinstance(mseed2, Stream)
#     assert len(mseed.traces) == len(mseed2.traces)
#     assert (mseed[0].data != mseed2[0].data).any()
#     assert max(mseed[0].data) > max(mseed2[0].data)
# 
# 
# def test_remove_response_with_inv_object(_data):
#     mseed = get_data()['mseed']
# #     inv_obj = _data['inventory']
#     for inv_output in ['ACC', 'VEL', 'DISP']:
#         mseed2 = get_data()['mseed_' + inv_output]
#         assert isinstance(mseed, Stream) == isinstance(mseed2, Stream)
#         assert len(mseed.traces) == len(mseed2.traces)
#         assert (mseed[0].data != mseed2[0].data).any()
#         assert max(mseed[0].data) > max(mseed2[0].data)


def get_stream_with_gaps(_data):
    mseed_dir = get_data()['data_path']
    return obspy_read(os.path.join(mseed_dir, "IA.BAKI..BHZ.D.2016.004.head"))


def testmaxabs():
    mseed = get_data()['mseed']
    
    t, g = maxabs(mseed[0])

    assert np.max(np.abs(mseed[0].data)) == g
    idx =  np.argmax(np.abs(mseed[0].data))
    
    assert timeof(mseed[0], idx) == t
    
    # assert by slicing times of max are different:
    td = 2*mseed[0].stats.delta
    assert maxabs(mseed[0], None, t-td)[0] < t < maxabs(mseed[0], t+td, None)[0]
    
    assert np.isnan(maxabs(mseed[0], None, mseed[0].stats.starttime-td))
    
    
