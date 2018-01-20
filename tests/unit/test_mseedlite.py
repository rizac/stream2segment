'''
Created on Apr 9, 2017

@author: riccardo
'''
import os
from datetime import datetime, timedelta
from obspy.core.stream import read, Stream
# from cStringIO import StringIO
from math import log
from collections import defaultdict
from stream2segment.download.modules.mseedlite import unpack, _FIXHEAD_LEN, MSeedError, Input
import pytest
import numpy as np
from obspy.core.trace import Trace
from obspy.io.mseed.core import _read_mseed
from io import BytesIO


_mrb = {}


def mock_response_inbytes(with_gaps=False):
    global _mrb
    name = "IA.BAKI..BHZ.D.2016.004.head" if with_gaps else "GE.FLT1..HH?.mseed" 
    filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", name)
    if _mrb.get(name, None) is None:
        with open(filename, 'rb') as opn:
            _mrb[name] = opn.read()
    return _mrb[name]


def get_stream(bytez):
    return read(BytesIO(bytez))


def get_s2s_stream(dicread):
    traces = []
    for v in dicread.values():
        if v[0] is None:
            traces.extend(get_stream(v[1]).traces)
    return Stream(traces)
    # return {id: read(StringIO(x))[0] for id, x in bytes_dic.iteritems()}


def keys_with_gaps(obspy_dic):
    keys = set()
    for key, trace in obspy_dic.items():
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


def haserr(dataread):
    for v in dataread.values():
        if v[0] is not None:
            return True
    return False


def test_standard():
    bytez = mock_response_inbytes()
    # g= get_stream(bytez)  # _read_mseed(BytesIO(bytez))
    # get our dicts of trace_id: trace_bytes
    dic = unpack(bytez)
    assert not haserr(dic)
    # assert all max gap ratios are below a certain threshold
    # (we should get 0, some rounding errors might occur):
    assert all(abs(v[3]) < 0.00011 for v in dic.values())
    # get the same dict by calling obspy.read:
    obspy_stream = get_stream(bytez)
    s2s_stream = get_s2s_stream(dic)
    assert streamequal(obspy_stream, s2s_stream, deep=True)

    # assert time read and time from obspy routine coincide. Probably due to rounding errors
    # end times are not strictly equal. However, they are really close (within 1 microsecond):
    tdelta = timedelta(microseconds=1)
    # compare traces, but note that we must match them by id for comparison:
    for t1 in obspy_stream:
        dic_values = dic[t1.get_id()]
        mseedlite_starttime, mseedlite_endtime =  dic_values[4],  dic_values[5]
        assert abs(t1.stats.starttime.datetime - mseedlite_starttime) <= tdelta
        assert abs(t1.stats.endtime.datetime - mseedlite_endtime) <= tdelta
    # assert also same number of traces:
    assert len(obspy_stream) == len(dic)
#     for t1, v in zip(obspy_stream, dic.values()): 
#         assert abs(t1.stats.starttime.datetime - v[4]) <= tdelta
#         assert abs(t1.stats.endtime.datetime - v[5]) <= tdelta
#     assert all(np.array_equal(x.data, obspy_dic[id_].data) for id_, x in mseed_dic.iteritems())
#     assert sorted(mseed_dic.keys()) == sorted(obspy_dic.keys())
#     assert gaps == keys_with_gaps(obspy_dic)


def test_standard_timebounds():
    bytez = mock_response_inbytes()
    # g= get_stream(bytez)  # _read_mseed(BytesIO(bytez))
    # get our dicts of trace_id: trace_bytes
    dic = unpack(bytez, None, None)
    assert not haserr(dic)
    # assert we have data and the error field is empty:
    assert all(v[1] and v[0] is None for v in dic.values())
    s2s_stream = get_s2s_stream(dic)

    start = s2s_stream[0].stats.starttime.datetime
    end = s2s_stream[0].stats.endtime.datetime

    # asssert that the last flag (discarded chunks) is False (we did not provide time bounds)
    assert all(v[6] is False for v in dic.values())

    # assert START times are different: this depends on our miniseedlite3
    # AND obspy routine. Form THE CURENT TRY, the difference is 0.000001 seconds (1 microsecond?)
    # which might be due to floating point rounding errors
    timediffs = list(max(abs(v[4] - s2s_stream[i].stats.starttime.datetime),
                         abs(v[5]-s2s_stream[i].stats.endtime.datetime)) for i, v in enumerate(dic.values()))
    assert all(t <= timedelta(seconds=0.000001) for t in timediffs)

    tdelta = (end-start)/2

    for times in [(start, start+tdelta), (start+tdelta, end)]:
        dic = unpack(bytez, *times)
        # check that everything is read (v[1] is truthy)
        # AND no errors are found (v[0] is None)
        # AND that the last element v[6] (chunks out of bounds) is True
        assert all(v[1] and v[0] is None and v[6] is True for v in dic.values())
        # check the time differences between these unpacked data and the original miniseed without
        # time bounds: the new time diffs should be greater than the old time diffs:
        timediffs2 = list(max(abs(v[4] - s2s_stream[i].stats.starttime.datetime),
                              abs(v[5]-s2s_stream[i].stats.endtime.datetime)) for i, v in enumerate(dic.values()))
        assert all(t2 > t1 for t2, t1 in zip(timediffs2, timediffs))

    # now test for complete out of bounds:
    d1 = datetime.utcnow()
    start = d1 + timedelta(days=365)
    end = d1 + timedelta(days=366)
    dic = unpack(bytez, start, end)
    # check that nothing is read (b'')
    # AND no errors are found (v[0] is None)
    # AND that the last element v[6] (chunks out of bounds) is True
    assert all(v[1] == b'' and v[0] is None and v[6] is True for v in dic.values())
    

def test_fsamp_mismatchs():
    bytez = mock_response_inbytes()
    dic = unpack(bytez, None, None)
    # assert all sample rates are 100:
    assert (v[2] == 100 for v in dic.values())
    # build a fake bytez string
    ret_dic = dict()
    key = None
    records = []
    for rec, is_exc in Input(bytez):
        if is_exc:
            continue
        if key is None:
            key = rec._record_id
        elif rec._record_id != key:
            continue
        records.append(rec)
        if len(records) > 1:
            # change fsamp
            rec.sr_factor *= 2
            rec.sr_mult = -rec.sr_mult
            break
    bytesio = BytesIO()
    for record in records:
        record.write(bytesio, int(log(record.size) / log(2)))
    bytez = bytesio.getvalue()
    bytesio.close()
    # g= get_stream(bytez)  # _read_mseed(BytesIO(bytez))
    # get our dicts of trace_id: trace_bytes
    dic = unpack(bytez, None, None)
    assert haserr(dic)
    assert len(dic) == 1
    # assert we have only one item, whose first element is not none (the exception)
    # and whose second element (the data) is None
    values = list(dic.values())[0]
    assert str(values[0]) == "records sample rate mismatch" and values[1] is None
   

def test_with_gaps_overlaps():
    bytez = mock_response_inbytes(True)

    # get our dicts of trace_id: trace_bytes
    dic = unpack(bytez)
    assert not haserr(dic)
    assert len(dic) == 1
    values = list(dic.values())[0]
    
    # get the same dict by calling obspy.read:
    obspy_stream = get_stream(bytez)
    obspygaps = obspy_stream.get_gaps()
    max_obspy_gap_ratio = max((_[-1] for _ in obspygaps))
    assert values[2] == max_obspy_gap_ratio  # gaps

    s2s_stream = get_s2s_stream(dic)
    assert streamequal(obspy_stream, s2s_stream, deep=True)
#     assert not all(np.array_equal(x.data, obspy_dic[id_].data) for id_, x in mseed_dic.iteritems())
#     assert sorted(mseed_dic.keys()) == sorted(obspy_dic.keys())
#     assert gaps == keys_with_gaps(obspy_dic)


def test_unexpected_end_of_header():
    '''test unexpected end of header, i.e. when unpack raises'''
    bytez = mock_response_inbytes()
    # this raises 'unexpected end of header':
    bytez2 = bytez[:100] + b'abc' + bytez[101:]
    with pytest.raises(MSeedError):
        _ = unpack(bytez2)


def test_change_last_byte():
    '''test when the data is corrupted, i.e. as headers are ok, unpack returns normally'''
    bytez = mock_response_inbytes()
    # get our dicts of trace_id: trace_bytes
    dic = unpack(bytez[:-1] + b'a')
    # this should not have errors as we changed the data, which is not read
    assert not haserr(dic)
    assert len(dic) == 3
    # assert all max gap ratios are below a certain threshold
    # (we should get 0, some rounding errors might occur):
    assert all(abs(v[3]) < 0.00011 for v in dic.values())

    obspy_stream = get_stream(bytez)
    s2s_stream = get_s2s_stream(dic)
    # assert same num of channels and traces and time ranges:
    assert streamequal(obspy_stream, s2s_stream, deep=False)
    # BUT NOT same data:
    assert not streamequal(obspy_stream, s2s_stream, deep=True)


def test_change_header_change_id():
    bytez = mock_response_inbytes()
    # get our dicts of trace_id: trace_bytes
    dic = unpack(b'a' * _FIXHEAD_LEN + bytez[_FIXHEAD_LEN:])
    # erros is not empty but has the trace id 'aa.aaaaa.aa.aaa'. What is that?
    # is the id we created by modyfing the bytes above
    assert haserr(dic)
    # curiously, the returned "traces" are 4 and not 3. The first one being the "error" trace
    assert len(dic) == 4
    # assert first one is erroneous (actually, different python versions might not store it in
    # the first item, so use 'any'):
    assert any(str(list(dic.values())[i][0]) == 'non-data record' for i in range(len(dic)))
    # assert all max gap ratios are below a certain threshold
    # (we should get 0, some rounding errors might occur)
    assert all(abs(v[3]) < 0.00011 for v in dic.values() if v[3] is not None)
    obspy_stream = get_stream(bytez)
    s2s_stream = get_s2s_stream(dic)
    # assert not same num of channels and traces and time ranges:
    assert streamequal(obspy_stream, s2s_stream, deep=False)


def test_invalid_pointers():
    '''test invalid pointers error'''
    bytez = mock_response_inbytes()
    # get our dicts of trace_id: trace_bytes
    dic = unpack(bytez[:_FIXHEAD_LEN-8] + (b'a' * 8) + bytez[_FIXHEAD_LEN:])
    # erros is not empty but has the trace id 'aa.aaaaa.aa.aaa'. What is that?
    # is the id we created by modyfing the bytes above
    assert haserr(dic)
    assert len(dic) == 3
    # assert first one is erroneous (actually, different python versions might not store it in
    # the first item, so use 'any'):
    assert any(str(list(dic.values())[i][0]) == 'invalid pointers' for i in range(len(dic)))
    # assert all max gap ratios are below a certain threshold
    # (we should get 0, some rounding errors might occur)
    assert all(abs(v[3]) < 0.00011 for v in dic.values() if v[3] is not None)

    obspy_stream = get_stream(bytez)
    s2s_stream = get_s2s_stream(dic)
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
    
    


