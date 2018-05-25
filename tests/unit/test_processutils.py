'''
Created on Oct 7, 2017

@author: riccardo
'''
import unittest
import os
import numpy as np
from obspy.core.stream import read
from stream2segment.process.utils import get_stream, get_slices
from mock import patch
from io import BytesIO
import pytest
import time
from tempfile import NamedTemporaryFile

class MockSegment(object):
     def __init__(self, data):
         self.data = data


@patch('obspy.core.stream.NamedTemporaryFile', return_value=NamedTemporaryFile())
def test_get_stream(mock_ntf, data):
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
    with pytest.raises(ValueError):
        stream_me = get_stream(segment)
    assert not mock_ntf.called


@pytest.mark.parametrize('input, expected_result, ',
                          [
                           ((340, 113), [(0, 114), (114, 227), (227, 340)]),
                           ((338, 113), [(0, 113), (113, 226), (226, 338)]),
                           ((339, 113), [(0, 113), (113, 226), (226, 339)])
                           ],
                        )
def test_getindices(input, expected_result):
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

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()