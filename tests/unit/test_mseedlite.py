'''
Created on Apr 9, 2017

@author: riccardo
'''
import os
from obspy.core.stream import read, Stream
from cStringIO import StringIO
from math import log
from collections import defaultdict
from stream2segment.utils.mseedlite3 import unpack, _FIXHEAD_LEN, MSeedError
import pytest
import numpy as np
from obspy.core.trace import Trace
from obspy.io.mseed.core import _read_mseed
from _io import BytesIO




_mrb = {}
def mock_response_inbytes(with_gaps=False):
    global _mrb
    name =  "IA.BAKI..BHZ.D.2016.004.head" if with_gaps else "GE.FLT1..HH?.mseed" 
    filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", name)
    if _mrb.get(name, None) is None:
        with open(filename, 'rb') as opn:
            _mrb[name] = opn.read()
    return _mrb[name]



def get_stream(bytez):
    # global _sd
    #if _sd is None:
    return read(StringIO(bytez))
#     _sd = {x.get_id(): x for x in stream}
#     return _sd


def get_s2s_stream(bytes_dic):
    traces = []
    for v in bytes_dic.itervalues():
        traces.extend(get_stream(v).traces)
    return Stream(traces)
    # return {id: read(StringIO(x))[0] for id, x in bytes_dic.iteritems()}


def keys_with_gaps(obspy_dic):
    keys = set()
    for key, trace in obspy_dic.iteritems():
        if Stream(trace).get_gaps():
            keys.add(key)
    return keys


def streamequal(stream1, stream2, deep=True):
    if len(stream1) != len(stream2):
        return False
    if set([t.get_id() for t in stream1]) != set([t.get_id() for t in stream2]):
        return False

    if deep:
        for t1 in stream1:
            id1 = (t1.get_id(), t1.stats.starttime, t1.stats.endtime)
            t2 = None
            for t_ in stream2:
                id2 = (t_.get_id(), t_.stats.starttime, t_.stats.endtime)
                if id1 == id2:
                    t2 = t_
                    break
            if t2 is None:
                return False
            if not np.array_equal(t1.data, t2.data):
                return False

    return True


def test_standard():
    bytez = mock_response_inbytes()
    g= _read_mseed(BytesIO(bytez))
    # get our dicts of trace_id: trace_bytes
    bytes_dic, gaps, errs = unpack(bytez)
    assert not errs
    assert all(g < 0.0001 for g in gaps.itervalues())
    # get the same dict by calling obspy.read:
    obspy_stream = get_stream(bytez)
    s2s_stream = get_s2s_stream(bytes_dic)
    
    assert streamequal(obspy_stream, s2s_stream, deep=True)
#     assert all(np.array_equal(x.data, obspy_dic[id_].data) for id_, x in mseed_dic.iteritems())
#     assert sorted(mseed_dic.keys()) == sorted(obspy_dic.keys())
#     assert gaps == keys_with_gaps(obspy_dic)

def test_with_gaps():
    # Let's change some header. But keeping the ref to the trace id for the record with errors:
    bytez = mock_response_inbytes(True)

    # get our dicts of trace_id: trace_bytes
    bytes_dic, gaps, errs = unpack(bytez)
    assert not errs
    assert any(g > 1 for g in gaps.itervalues())
    # get the same dict by calling obspy.read:
    obspy_stream = get_stream(bytez)
    s2s_stream = get_s2s_stream(bytes_dic)
    assert streamequal(obspy_stream, s2s_stream, deep=True)
#     assert not all(np.array_equal(x.data, obspy_dic[id_].data) for id_, x in mseed_dic.iteritems())
#     assert sorted(mseed_dic.keys()) == sorted(obspy_dic.keys())
#     assert gaps == keys_with_gaps(obspy_dic)


def test_add_bytes():
    # now change a byte:
    bytez = mock_response_inbytes()
    # this raises 'unexpected end of header':
    bytez2 = bytez[:100] + 'abc' + bytez[101:]
    with pytest.raises(MSeedError):
        mseed_dic, gaps, errs = unpack(bytez2)


def test_change_last_byte():
    # Let's change one byte at the end (a byte of the data part)
    bytez = mock_response_inbytes()
    # get our dicts of trace_id: trace_bytes
    bytes_dic, gaps, errs = unpack( bytez[:-1] + 'a')
    assert not errs
    assert all(g < 0.0001 for g in gaps.itervalues())

    obspy_stream = get_stream(bytez)
    s2s_stream = get_s2s_stream(bytes_dic)
    # assert same num of channels and traces and time ranges:
    assert streamequal(obspy_stream, s2s_stream, deep=False)
    # BUT NOT same data:
    assert not streamequal(obspy_stream, s2s_stream, deep=True)

def test_change_header_change_id():
    
    # Let's change one byte at the end (a byte of the data part), this will
    bytez = mock_response_inbytes()
    # get our dicts of trace_id: trace_bytes
    bytes_dic, gaps, errs = unpack('a' * _FIXHEAD_LEN + bytez[_FIXHEAD_LEN:])
    # erros is not empty but has the trace id 'aa.aaaaa.aa.aaa'. What is that?
    # is the id we created by modyfing the bytes above
    assert errs
    assert all(g < 0.0001 for g in gaps.itervalues())

    obspy_stream = get_stream(bytez)
    s2s_stream = get_s2s_stream(bytes_dic)
    # assert not same num of channels and traces and time ranges:
    assert streamequal(obspy_stream, s2s_stream, deep=False)


def test_change_header_keep_id():
        # Let's change one byte at the end (a byte of the data part), this will
    bytez = mock_response_inbytes()
    # get our dicts of trace_id: trace_bytes
    bytes_dic, gaps, errs = unpack(bytez[:_FIXHEAD_LEN-8] + (b'a' * 8) + bytez[_FIXHEAD_LEN:])
    # erros is not empty but has the trace id 'aa.aaaaa.aa.aaa'. What is that?
    # is the id we created by modyfing the bytes above
    assert errs
    assert all(g < 0.0001 for g in gaps.itervalues())

    obspy_stream = get_stream(bytez)
    s2s_stream = get_s2s_stream(bytes_dic)
    # assert not same num of channels and traces and time ranges:
    assert not streamequal(obspy_stream, s2s_stream, deep=False)

#     # Let's change some header. But keeping the ref to the trace id for the record with errors:
#     bytez = mock_response_inbytes()
#     bytez = bytez[:_FIXHEAD_LEN-8] + (b'a' * 8) + bytez[_FIXHEAD_LEN:]
#     # get our dicts of trace_id: trace_bytes
#     mseed_dic, errs = unpack(bytez)
#     assert errs
#     # get the same dict by calling obspy.read:
#     obspy_dic = getobspydic(bytez)
#     # we skip a record, ignoring all subsequent read for that trace. Thus:
#     assert len(mseed_dic) != len(obspy_dic)
#     # However, for the trace saved, the traces are the same:
#     assert all(np.array_equal(read(StringIO(x))[0].data,
#                               obspy_dic[id].data) for id, x in mseed_dic.iteritems())
    
    


