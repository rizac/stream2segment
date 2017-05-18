#@PydevCodeAnalysisIgnore
'''
Created on Feb 23, 2016

@author: riccardo
'''

import mock, os, sys
import pytest
import re
import argparse
import numpy as np
from stream2segment.analysis import fft as orig_fft
from stream2segment.analysis import snr as orig_snr
from stream2segment.analysis import pow_spec as orig_powspec


from stream2segment.analysis.mseeds import remove_response, fft, snr , bandpass
# from stream2segment.io.utils import loads, dumps
# from stream2segment.analysis.mseeds import _IO_FORMAT_FFT, _IO_FORMAT_STREAM, _IO_FORMAT_TIME,\
#     _IO_FORMAT_TRACE
from obspy.core.inventory import read_inventory
from obspy.core import read as obspy_read
from obspy.core import Trace, Stream
from StringIO import StringIO
from obspy.io.stationxml.core import _read_stationxml
from obspy.core.trace import Trace
from itertools import count

@pytest.mark.parametrize('arr, arr_len_after_trim, fft_npts',
                        [([1, 2, 3, 4, 5, 6], 6, 4),
                         ([1, 2, 3, 4, 5], 5, 3),
                         ([1, 2, 3, 4], 4, 3),
                         ([1, 2, 3], 3, 2),
                         ])
@mock.patch('stream2segment.analysis.mseeds._fft', side_effect=lambda *a, **k: orig_fft(*a, **k))
def test_fft(mock_mseed_fft, arr, arr_len_after_trim, fft_npts):
    t = Trace(np.array(arr))
    f = fft(t)
    assert len(mock_mseed_fft.call_args[0][0]) == arr_len_after_trim
    assert len(f) == fft_npts
    # assure we preserved the original stats
    # Doing f.stats == t.stats does not work cause in the stats obspy put also other stuff
    # (e.g., the history of the processing done). Use s.stats.default which has the 
    # relevant properties (in particular delta and npts, for calculating the delta freq
    assert f.stats.defaults == t.stats.defaults
    g = 9


@mock.patch('stream2segment.analysis.mseeds._snr', side_effect=lambda *a, **k: orig_snr(*a, **k))
@mock.patch('stream2segment.analysis.pow_spec', side_effect=lambda *a, **k: orig_powspec(*a, **k))
def test_snr(mock_powspec, mock_analysis_snr):
    trace = get_data()['mseed'][0]
    res = snr(trace, trace, fmin=10, fmax=100.34)
    assert res == 1
    assert mock_powspec.call_count == 2  # one for each trace
    assert len(mock_analysis_snr.call_args_list) ==1
    assert mock_analysis_snr.call_args_list[0][1]['fmin'] == 10
    assert mock_analysis_snr.call_args_list[0][1]['fmax'] == 100.34
    assert mock_analysis_snr.call_args_list[0][1]['delta_signal'] == trace.stats.delta
    assert mock_analysis_snr.call_args_list[0][1]['delta_noise'] == trace.stats.delta
    
    
    h = 9
    
    
def test_bp():
    trace = get_data()['mseed'][0]
    res = bandpass(trace, 2, 3)
    h = 9

# @pytest.mark.parametrize('inv_output',
#                           ['ACC', 'VEL', 'DISP'])
# def test_read_dumps(_data, inv_output):
# 
# 
#     # do NOT provide the format, it should complain:
#     with pytest.raises(ValueError):
#         d = dumps(data)
# 
#     # Now not anymore:
#     for f in [_IO_FORMAT_FFT, _IO_FORMAT_STREAM, _IO_FORMAT_TIME, _IO_FORMAT_TRACE]:
#         dmp = dumps(data, f)
#         ret_obj = loads(dmp)
#         _data = ret_obj.data if hasattr(ret_obj, "data") else ret_obj.traces[0].data
#         assert all(_data == data)
#         h = 9

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
    s = StringIO()
    with open(inv_path) as _opn:
        s.write(_opn.read())
    s.seek(0)
    inv_obj = read_inventory(s)
    ret = {'mseed': mseed, 'inventory': inv_obj, 'mseed_path': mseed_path,
           'data_path': folder, 
           'inventory_path': inv_path}
    for inv_output in ['ACC', 'VEL', 'DISP']:
        mseed2 = remove_response(mseed, inv_obj, output=inv_output)
        ret['mseed_'+inv_output] = mseed2
    return ret


@pytest.mark.parametrize('inv_output',
                          ['ACC', 'VEL', 'DISP'])
def test_remove_response_with_inv_path(_data, inv_output):
    mseed = get_data()['mseed']
    mseed2 = get_data()['mseed_'+inv_output]
    assert isinstance(mseed, Stream) == isinstance(mseed2, Stream)
    assert len(mseed.traces) == len(mseed2.traces)
    assert (mseed[0].data != mseed2[0].data).any()
    assert max(mseed[0].data) > max(mseed2[0].data)


def test_remove_response_with_inv_object(_data):
    mseed = get_data()['mseed']
#     inv_obj = _data['inventory']
    for inv_output in ['ACC', 'VEL', 'DISP']:
        mseed2 = get_data()['mseed_' + inv_output]
        assert isinstance(mseed, Stream) == isinstance(mseed2, Stream)
        assert len(mseed.traces) == len(mseed2.traces)
        assert (mseed[0].data != mseed2[0].data).any()
        assert max(mseed[0].data) > max(mseed2[0].data)


def get_stream_with_gaps(_data):
    mseed_dir = get_data()['data_path']
    return obspy_read(os.path.join(mseed_dir, "IA.BAKI..BHZ.D.2016.004.head"))


def test_get_trace_with_gaps(_data):  # WTF am I testing here? obspy?? FIXME: remove
    stream = get_stream_with_gaps(_data)
    arr = stream.get_gaps()
    assert len(arr) > 0

