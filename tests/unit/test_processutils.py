'''
Created on Oct 7, 2017

@author: riccardo
'''
import os
import sys
from io import BytesIO
import time
from datetime import datetime
import pandas as pd
from tempfile import NamedTemporaryFile

from mock import patch
import pytest
import numpy as np
from obspy.core.stream import read

from stream2segment.process import SkipSegment
from stream2segment.process.db import get_stream
from stream2segment.process.main import get_slices
from stream2segment.process.writers import CsvWriter, HDFWriter

class MockSegment(object):
    def __init__(self, data):
        self.data = data


def test_get_stream(data):  # <- data is a pytest fixture
    '''test our get_stream calling obspy._read, and obspy.read:
    Rationale: process.db.get_stream reads a stream from a sequence of
    bytes (fetched our database). obspy read supports filelike object such as
    BytesIO, great right? no, because on error it tries to write to file and
    retry. This is absolutely insane. To avoid this, process.db.get_stream
    calls obspy._read instead.
    In this test, we want to assure that obpsy has still this weird
    implementation and that our get_stream is correct (_read is private it
    might be moved in the future)
    '''
    # PLEASE NOTE: we want to mock NamedTemporaryFile as used in obspy,
    # to check that it's called. Problem is, from obpsy version 1.2 whatever
    # they refactored and moved the packages. Thus, try-catch:
    try:
        from obspy.core.stream import NamedTemporaryFile
        patch_str = 'obspy.core.stream.NamedTemporaryFile'
    except ImportError:
        from obspy.core.util.base import NamedTemporaryFile
        patch_str = 'obspy.core.util.base.NamedTemporaryFile'

    with patch(patch_str, return_value=NamedTemporaryFile()) as mock_ntf:
        mseeddata = data.read('trace_GE.APE.mseed')

        segment = MockSegment(mseeddata)
        tobspy = time.time()
        stream_obspy = read(BytesIO(mseeddata))
        tobspy = time.time() - tobspy
        tme = time.time()
        stream_me = get_stream(segment)
        tme = time.time() - tme
        # assert we are faster (actually that calling read with format='MSEED' is faster than
        # calling with format=None)
        assert tme < tobspy
        assert (stream_obspy[0].data == stream_me[0].data).all()
        assert not mock_ntf.called

        with pytest.raises(TypeError):
            stream_obspy = read(BytesIO(mseeddata[:5]))
        assert mock_ntf.called

        mock_ntf.reset_mock()
        segment = MockSegment(mseeddata[:5])
        with pytest.raises(SkipSegment):
            stream_me = get_stream(segment)
        assert not mock_ntf.called


@pytest.mark.parametrize('input, expected_result, ',
                          [
                           ((340, 113), [(0, 113), (113, 226), (226, 340)]),
                           ((338, 113), [(0, 112), (112, 225), (225, 338)]),
                           ((339, 113), [(0, 113), (113, 226), (226, 339)])
                           ],
                        )
def test_get_slices(input, expected_result):
    expected_list = list(range(input[0]))
    assert len(expected_list) == input[0]
    real_list = []
    slices = list(get_slices(*input))
    assert len(slices) == len(expected_result)
    for (s, e), expected in zip(slices, expected_result):
        assert (s, e) == expected
        real_list += list(range(s, e))
    assert real_list == expected_list

    # test with arrays as first argument. Use numpy arrays of dimension two to provide a more
    # general case:
    expected_list = np.array([[i, 2] for i in expected_list])
    slices2 = list(get_slices(expected_list, input[1]))
    assert len(slices2) == len(slices)
    for nparray, (s, e) in zip(slices2, slices):
        assert np.array_equal(nparray, expected_list[s:e])
    # test for safety that we get until the last element:
    assert np.array_equal(nparray[-1], expected_list[-1])


def test_writer_hdf(
                    # fixtures:
                    pytestdir):
    file = pytestdir.newfile('.hd')
    writer = HDFWriter(file, True)
    writer.chunksize = 1

    df1 = pd.DataFrame([{
        'str': 'a',
        'dtime': datetime.utcnow(),
        'float': 1.1,
        'int': 1,
        'bool': True
    }])

    df2 = pd.DataFrame([{
        'str': 'abc',
        'dtime': datetime.utcnow(),
        'float': float('nan'),
        'int': 1,
        'bool': True
    }])

    with pytest.raises(Exception):
        with writer:
            writer.write(1, df1)
            writer.write(2, df2)

    writer = HDFWriter(file, False, {'min_itemsize': {'str': 10}})
    with writer:
        writer.write(1, df1)
        writer.write(2, df2)
    aps = writer.already_processed_segments()
    assert list(aps) == [1, 2]

    writer = HDFWriter(file, True, {'min_itemsize': {'str': 10}})
    with writer:
        writer.write(3, df2.loc[0, :])  # series
        writer.write(4, df2.loc[0, :].to_dict())
    aps = writer.already_processed_segments()
    assert list(aps) == [1, 2, 3, 4]
