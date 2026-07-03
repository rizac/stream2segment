"""
Created on Apr 9, 2017

@author: riccardo
"""
from datetime import datetime, timedelta
from unittest.mock import patch
from struct import unpack as original_unpack
import numpy as np
import pytest
from obspy.core.stream import read, Stream

from stream2segment.download.mseedlite import _FIXHEAD_LEN, MSeedError, Input
from stream2segment.download.segments import unpack_miniseed, unpacked_miniseed


def unpack(data) -> dict[str, unpacked_miniseed]:
    return {_.seed_id: _ for _ in unpack_miniseed(data)}

mseed_gaps = "IA.BAKI..BHZ.D.2016.004.head"
mseed_nogaps = "trace_GE.APE.mseed"


def get_s2s_stream(unpacked_mseed: dict[str, unpacked_miniseed]):
    s = Stream()
    for v in unpacked_mseed.values():
        if v.data is not None:
            s += read(v.data, format='MSEED')
    return s
    # return {id: read(StringIO(x))[0] for id, x in bytes_dic.iteritems()}


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


def has_err(dataread):
    for v in dataread.values():
        if v.data is None:
            return True
    return False


def test_standard(test_data_dir):
    f_path = test_data_dir / mseed_nogaps
    obspy_stream = read(f_path, format='MSEED')
    # g= get_stream(bytez)  # _read_mseed(BytesIO(bytez))
    # get our dicts of trace_id: trace_bytes
    dic = unpack(f_path.read_bytes())
    assert not has_err(dic)
    # assert all max gap ratios are below a certain threshold
    # (we should get 0, some rounding errors might occur):
    assert all(abs(v.maxgap) < 0.00011 for v in dic.values())
    s2s_stream = get_s2s_stream(dic)
    assert streamequal(obspy_stream, s2s_stream, deep=True)

    # assert time read and time from obspy routine coincide. Probably due to rounding errors
    # end times are not strictly equal. However, they are really close (within 1 microsecond):
    tdelta = timedelta(microseconds=1)
    # compare traces, but note that we must match them by id for comparison:
    for t1 in obspy_stream:
        dic_values = dic[t1.get_id()]
        mseedlite_starttime =  dic_values.start
        mseedlite_endtime = dic_values.end
        assert abs(t1.stats.starttime.datetime - mseedlite_starttime) <= tdelta
        assert abs(t1.stats.endtime.datetime - mseedlite_endtime) <= tdelta
    # assert also same number of traces:
    assert len(obspy_stream) == len(dic)


def test_with_gaps_overlaps(test_data_dir):
    f_path = test_data_dir / mseed_gaps
    obspy_stream = read(f_path, format='MSEED')

    # get our dicts of trace_id: trace_bytes
    dic = unpack(f_path.read_bytes())
    assert not has_err(dic)
    assert len(dic) == 1
    values = list(dic.values())[0]

    obspygaps = obspy_stream.get_gaps()
    max_obspy_gap_ratio = max((_[-1] for _ in obspygaps))
    assert values.maxgap == max_obspy_gap_ratio  # gaps

    s2s_stream = get_s2s_stream(dic)
    assert streamequal(obspy_stream, s2s_stream, deep=True)


def test_empty_data():
    """test empty data"""
    bytez = b''
    assert not unpack(bytez)


def test_unexpected_end_of_header(test_data_dir):
    """test unexpected end of header, i.e. when unpack raises"""
    bytez = (test_data_dir / mseed_nogaps).read_bytes()
    # this raises 'unexpected end of header':
    bytez2 = bytez[:100] + b'abc' + bytez[101:]
    with pytest.raises(MSeedError):
        _ = unpack(bytez2)


def test_change_last_byte(test_data_dir):
    """test when the data is corrupted, i.e. as headers are ok, unpack returns normally"""
    f_path = test_data_dir / mseed_nogaps

    # get our dicts of trace_id: trace_bytes
    dic = unpack(f_path.read_bytes()[:-1] + b'a')
    # this should not have errors as we changed the data, which is not read
    assert not has_err(dic)

    # assert all max gap ratios are below a certain threshold
    # (we should get 0, some rounding errors might occur):
    assert all(abs(v.maxgap) < 0.00011 for v in dic.values())

    obspy_stream = read(f_path, format='MSEED')
    s2s_stream = get_s2s_stream(dic)
    # assert same num of channels and traces and time ranges:
    assert streamequal(obspy_stream, s2s_stream, deep=False)
    # BUT NOT same data:
    assert not streamequal(obspy_stream, s2s_stream, deep=True)


def test_change_header_change_id(test_data_dir):
    f_path = test_data_dir / mseed_nogaps
    obspy_stream = read(f_path, format='MSEED')
    # get our dicts of trace_id: trace_bytes
    dic = unpack(b'a' * _FIXHEAD_LEN + f_path.read_bytes()[_FIXHEAD_LEN:])
    # erros is not empty but has the trace id 'aa.aaaaa.aa.aaa'. What is that?
    # is the id we created by modyfing the bytes above
    # assert haserr(dic)
    # curiously, the returned "traces" are 4 and not 3. The first one being the "error" trace
    # assert len(dic) == 4
    # assert first one is erroneous (actually, different python versions might not store it in
    # the first item, so use 'any'):
    # assert any(str(list(dic.values())[i][0]) == 'non-data record' for i in range(len(dic)))
    # assert all max gap ratios are below a certain threshold
    # (we should get 0, some rounding errors might occur)
    assert all(abs(v.maxgap) < 0.00011 for v in dic.values() if v[3] is not None)
    s2s_stream = get_s2s_stream(dic)
    # assert not same num of channels and traces and time ranges:
    assert streamequal(obspy_stream, s2s_stream, deep=False)


def test_invalid_pointers(test_data_dir):
    """test invalid pointers error"""
    bytez = (test_data_dir / mseed_nogaps).read_bytes()
    # get our dicts of trace_id: trace_bytes
    with pytest.raises(MSeedError):
        dic = unpack(bytez[:_FIXHEAD_LEN-8] + (b'a' * 8) + bytez[_FIXHEAD_LEN:])


@pytest.mark.parametrize("struct_unpack_arg, should_raise", [
    # these are all the possible arguments passed to struct.unpack
    # in mseedlite (1st argument), and whether they're supposed to raise or not:
    (">6scx5s2s3s2s2H3Bx2H2h4Bl2H", True),
    # ("<xxBBLHHBBBBdLLBBHL", False),  # uncomment if you supply mseed v3
    (">2H", True),
    (">3Bx", True),
    (">BbxB", False),
    (">ll", False),
    (">L", False)
])
@patch('stream2segment.download.mseedlite.struct.unpack')
def test_struct_unpack_error(mock_struct_unpack, struct_unpack_arg, should_raise, test_data_dir):
    """test invalid pointers error"""

    def sunpack(what, bytez):
        if what == struct_unpack_arg:
            bytez = bytez[:-1]
        return original_unpack(what, bytez)

    mock_struct_unpack.side_effect = sunpack
    bytez = (test_data_dir / mseed_nogaps).read_bytes()

    if should_raise:
        with pytest.raises(MSeedError):
            unpack(bytez)
        return