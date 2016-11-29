'''
Created on Feb 23, 2016

@author: riccardo
'''
import os
import mock
import pytest
import numpy as np
from stream2segment.analysis.mseeds import Stream, Trace, fft
from stream2segment.analysis import fft as orig_fft
from obspy.core.stream import read
from obspy.core import Stream as ObspyStream, Trace as ObspyTrace


def get_stream_with_gaps():
    parentdir = os.path.dirname(os.path.dirname(__file__))
    return read(os.path.join(parentdir, "data", "IA.BAKI..BHZ.D.2016.004.head"))


def test_get_trace_with_gaps():
    stream = get_stream_with_gaps()
    arr = stream.get_gaps()
    assert len(arr) > 0

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
    g = 9
# @pytest.mark.parametrize('arr, normalize, expected_result, ',
#                           [([-1, 1], True, [0.5, 1]),
#                            ([-1, 1], False, [1, 2]),
#                            ([-2, 3], True, [4.0/(4+9), (4+9.0)/(4+9)]),
#                            ([-2, 3], False, [4, 4+9]),
#                            ])
# @mock.patch('stream2segment.mseed.get_gaps')
# def test_get_gaps()  # (mock_np, arr, normalize, expected_result):
#
#     trace = get_trace(0.1, [1.2.3])
#     trace = get_trace(0.1, [1.2.3])
#
#     mock_np.cumsum = mock.Mock(side_effect = lambda *a, **k: np.cumsum(*a, **k))
#     mock_np.square =  mock.Mock(side_effect = lambda *a, **k: np.square(*a, **k))
#     mock_np.max =  mock.Mock(side_effect = lambda *a, **k: np.max(*a, **k))
#     mock_np.true_divide =  mock.Mock(side_effect = lambda *a, **k: np.true_divide(*a, **k))
#
#     r = cumsum(arr, normalize=normalize)
#     assert len(r) == len(arr)
#     assert (r == np.array(expected_result)).all()
#     assert mock_np.cumsum.called
#     assert mock_np.square.called
#     if normalize:
#         assert mock_np.max.called
#         assert mock_np.true_divide.called
#     else:
#         assert not mock_np.max.called
#         assert not mock_np.true_divide.called


# @pytest.mark.parametrize('arr, expected_result, ',
#                           [
#                            ([-1, 1], [1, 1]),
#                            ([-2, 3], [2, 3]),
#                            ([-1, 10, -12, 3], [3.64005494, 11.41271221, 12.5, 6.26498204])
#                            ],
#                         )
# @mock.patch('stream2segment.analysis.hilbert', side_effect=lambda *a,**k: hilbert(*a, **k))
# @mock.patch('stream2segment.analysis.np')
# def test_env(mock_np, mock_hilbert, arr, expected_result):
#     mock_np.abs = mock.Mock(side_effect = lambda *a, **k: np.abs(*a, **k))
#     r = env(arr)
#     assert len(r) == len(arr)
#     mock_hilbert.assert_called_once_with(arr)
#     assert mock_np.abs.called
#     g = 9    