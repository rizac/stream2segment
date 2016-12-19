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
from stream2segment.analysis.mseeds import remove_response, fft #, loads as s2s_loads, dumps
from stream2segment.io.dataseries import loads, dumps
from stream2segment.analysis.mseeds import _IO_FORMAT_FFT, _IO_FORMAT_STREAM, _IO_FORMAT_TIME,\
    _IO_FORMAT_TRACE

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
    mseed = _data['mseed']
    mseed2 = _data['mseed_'+inv_output]
    assert isinstance(mseed, Stream) == isinstance(mseed2, Stream)
    assert len(mseed.traces) == len(mseed2.traces)
    assert (mseed[0].data != mseed2[0].data).any()
    assert max(mseed[0].data) > max(mseed2[0].data)


def test_remove_response_with_inv_object(_data):
    mseed = _data['mseed']
#     inv_obj = _data['inventory']
    for inv_output in ['ACC', 'VEL', 'DISP']:
        mseed2 = _data['mseed_' + inv_output]
        assert isinstance(mseed, Stream) == isinstance(mseed2, Stream)
        assert len(mseed.traces) == len(mseed2.traces)
        assert (mseed[0].data != mseed2[0].data).any()
        assert max(mseed[0].data) > max(mseed2[0].data)


@pytest.mark.parametrize('inv_output',
                          ['ACC', 'VEL', 'DISP'])
def test_mseed_float32(_data, inv_output):
    mseed = _data['mseed']
#     inv_obj = _data['inventory']
    mseed2 = _data['mseed_'+inv_output]
    sio = StringIO()

    # so, apparently mseed2 has still the eoncoding of the original mseed,
    # which is "STEIM2" for the current mseed (mapped to np.int32, see 
    #  obspy.io.mseed.headers.ENCODINGS
    # if we set a custom encoding, IT MUST MATCH the dtype of the trace data
    
    # THIS RAISES WARNING (commented out):
    # mseed2.write(sio, format="MSEED")
    
    # THIS IS FINE:
    mseed2.write(sio, format="MSEED", encoding=5)
    
    # NOW LET"'s CONVERT:
    mseed2_32 = Stream([Trace(trace.data.astype(np.float32), trace.stats)
                        for trace in mseed2])
    sio32 = StringIO()
    mseed2_32.write(sio32, format='MSEED', encoding=4) 
    
    # sio32 HAS APPROX HALF THE LENGTH OF sio64 (uncomment below and debug breakpoint or print):
    # size32 = len(sio32.getvalue())
    # size64 = len(sio64.getvalue())
    
    # let's see how this affects data. What is the relative error?
    mseed64 = obspy_read(sio)
    mseed32 = obspy_read(sio32)
    
    print ""
    print inv_output
    for t64, t32, i in zip(mseed64, mseed32, count()):
        min = np.nanmin(t64.data)
        max = np.nanmax(t64.data)
        range = np.abs(max-min)
        diff = np.abs(t64.data - t32.data) / range
        meanerr = np.nanmean(diff)
        maxerr = np.nanmax(diff)
        minerr = np.nanmin(diff)
        print "Trace#%d" % (i+1)
        print "min %.2E" % min
        print "max %.2E" % max
        print "min err ([0,1]): %.2E" % minerr
        print "mean err ([0,1]): %.2E" % meanerr
        print "max err ([0,1]): %.2E" % maxerr
        g = 9
    
    print ""
    h = 9
    # assert mseed.__class__.__name__ == mseed2.__class__.__name__


def get_stream_with_gaps(_data):
    mseed_dir = _data['data_path']
    return obspy_read(os.path.join(mseed_dir, "IA.BAKI..BHZ.D.2016.004.head"))


def test_get_trace_with_gaps(_data):
    stream = get_stream_with_gaps(_data)
    arr = stream.get_gaps()
    assert len(arr) > 0
    
    
    # UNCOMMENT JUST TO SEE THE PLOT
    # WARNING: REMEMBER TO COMMENT IT LATER IN CASE!!!!!!!
#     tr = mseed2.traces[0]
#     tr.stats.channel = 'REMOVED_R'
#     news = Stream([mseed.traces[0], tr])
#     news.plot()
#     g = 9


# def test_new_df():
#     dnormal = pd.DataFrame(columns=['Col1\$', 'col2'], data=[[1,2], [3, 'f']])
#     with pytest.raises(KeyError):
#         dnormal['col1']
#     
#     dnew = DataFrame(columns=['Col1\$', 'col2'], data=[[1,2], [3, 'f']])
#     d1 = dnormal['Col1\$']
#     d2 = dnew['col1\$']
#     assert d1.equals(d2)
#     assert isinstance(d1, pd.Series)
#     assert isinstance(d2, pd.Series)
#     d1 = dnormal[['Col1\$', 'col2']]
#     d2 = dnew[['col1\$', 'cOL2']]
#     assert d1.equals(d2)
#     assert isinstance(d1, pd.DataFrame)
#     assert isinstance(d2, DataFrame)
# 
#     # non string slicing returns DataFrame or pd.DataFrame according to the object:
#     d1 = dnormal[1:2]
#     d2 = dnew[1:2]
#     assert d1.equals(d2)
#     assert isinstance(d1, pd.DataFrame)
#     assert isinstance(d2, DataFrame)
# 
#     d1 = dnormal[[]]
#     d2 = dnew[[]]
#     assert d1.equals(d2)
#     assert isinstance(d1, pd.DataFrame)
#     assert isinstance(d2, DataFrame)
#     
#     with pytest.raises(KeyError):
#         dnew['kol1']
#     with pytest.raises(KeyError):
#         dnew[['kol1', 'col2']]
#     with pytest.raises(KeyError):
#         dnormal[4]
#     with pytest.raises(KeyError):
#         dnew[4]
    
