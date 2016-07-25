'''
Created on Feb 23, 2016

@author: riccardo
'''

import mock, os, sys
import pytest
import re
import argparse
import numpy as np
import pandas as pd
from stream2segment.analysis import env, cumsum, fft
from scipy.signal import hilbert
from stream2segment.utils import DataFrame
from stream2segment.analysis.mseeds import remove_response, read as s2s_read, dumps
from stream2segment.analysis.mseeds import _IO_FORMAT_FFT, _IO_FORMAT_STREAM, _IO_FORMAT_TIME,\
    _IO_FORMAT_TRACE

from obspy.core.inventory import read_inventory
from obspy.core import read as obspy_read
from obspy.core import Trace, Stream
from StringIO import StringIO
from obspy.io.stationxml.core import _read_stationxml


def test_read_dumps():
    data = [1, 1.4, 4 + 7j]

    # do NOT provide the format, it should complain:
    with pytest.raises(ValueError):
        d = dumps(data)

    # Now not anymore:
    for f in [_IO_FORMAT_FFT, _IO_FORMAT_STREAM, _IO_FORMAT_TIME, _IO_FORMAT_TRACE]:
        dmp = dumps(data, f)
        ret_obj = s2s_read(dmp)
        _data = ret_obj.data if hasattr(ret_obj, "data") else ret_obj.traces[0].data
        assert all(_data == data)
        h = 9


@pytest.mark.parametrize('inv_output',
                          ['ACC', 'VEL', 'DISP'])
def test_remove_response_with_inv_path(inv_output):
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    mseed = obspy_read(os.path.join(folder, 'trace_GE.APE.mseed'))
    inv_path = os.path.join(folder, 'inventory_GE.APE.xml')
    mseed2 = remove_response(mseed, inv_path, output=inv_output)
    assert isinstance(mseed, Stream) == isinstance(mseed2, Stream)
    assert len(mseed.traces) == len(mseed2.traces)
    assert (mseed[0].data != mseed2[0].data).any()
    assert max(mseed[0].data) > max(mseed2[0].data)


def test_remove_response_with_inv_object():
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    mseed = obspy_read(os.path.join(folder, 'trace_GE.APE.mseed'))
    inv_path = os.path.join(folder, 'inventory_GE.APE.xml')
    s = StringIO()
    with open(inv_path) as _opn:
        s.write(_opn.read())
    s.seek(0)
    inv_obj = read_inventory(s)
    for inv_output in ['ACC', 'VEL', 'DISP']:
        mseed2 = remove_response(mseed, inv_obj, output=inv_output)
        assert isinstance(mseed, Stream) == isinstance(mseed2, Stream)
        assert len(mseed.traces) == len(mseed2.traces)
        assert (mseed[0].data != mseed2[0].data).any()
        assert max(mseed[0].data) > max(mseed2[0].data)


    # assert mseed.__class__.__name__ == mseed2.__class__.__name__

    # UNCOMMENT JUST TO SEE THE PLOT
    # WARNING: REMEMBER TO COMMENT IT LATER IN CASE!!!!!!!
#     tr = mseed2.traces[0]
#     tr.stats.channel = 'REMOVED_R'
#     news = Stream([mseed.traces[0], tr])
#     news.plot()
#     g = 9


def test_new_df():
    dnormal = pd.DataFrame(columns=['Col1\$', 'col2'], data=[[1,2], [3, 'f']])
    with pytest.raises(KeyError):
        dnormal['col1']
    
    dnew = DataFrame(columns=['Col1\$', 'col2'], data=[[1,2], [3, 'f']])
    d1 = dnormal['Col1\$']
    d2 = dnew['col1\$']
    assert d1.equals(d2)
    assert isinstance(d1, pd.Series)
    assert isinstance(d2, pd.Series)
    d1 = dnormal[['Col1\$', 'col2']]
    d2 = dnew[['col1\$', 'cOL2']]
    assert d1.equals(d2)
    assert isinstance(d1, pd.DataFrame)
    assert isinstance(d2, DataFrame)

    # non string slicing returns DataFrame or pd.DataFrame according to the object:
    d1 = dnormal[1:2]
    d2 = dnew[1:2]
    assert d1.equals(d2)
    assert isinstance(d1, pd.DataFrame)
    assert isinstance(d2, DataFrame)

    d1 = dnormal[[]]
    d2 = dnew[[]]
    assert d1.equals(d2)
    assert isinstance(d1, pd.DataFrame)
    assert isinstance(d2, DataFrame)
    
    with pytest.raises(KeyError):
        dnew['kol1']
    with pytest.raises(KeyError):
        dnew[['kol1', 'col2']]
    with pytest.raises(KeyError):
        dnormal[4]
    with pytest.raises(KeyError):
        dnew[4]
    
@pytest.mark.parametrize('arr, normalize, expected_result, ',
                          [([-1, 1], True, [0.5, 1]),
                           ([-1, 1], False, [1, 2]),
                           ([-2, 3], True, [4.0/(4+9), (4+9.0)/(4+9)]),
                           ([-2, 3], False, [4, 4+9]),
                           ])
@mock.patch('stream2segment.analysis.np')
def test_cumsum(mock_np, arr, normalize, expected_result):
    mock_np.cumsum = mock.Mock(side_effect = lambda *a, **k: np.cumsum(*a, **k))
    mock_np.square =  mock.Mock(side_effect = lambda *a, **k: np.square(*a, **k))
    mock_np.max =  mock.Mock(side_effect = lambda *a, **k: np.max(*a, **k))
    mock_np.true_divide =  mock.Mock(side_effect = lambda *a, **k: np.true_divide(*a, **k))

    r = cumsum(arr, normalize=normalize)
    assert len(r) == len(arr)
    assert (r == np.array(expected_result)).all()
    assert mock_np.cumsum.called
    assert mock_np.square.called
    if normalize:
        assert mock_np.max.called
        assert mock_np.true_divide.called
    else:
        assert not mock_np.max.called
        assert not mock_np.true_divide.called


@pytest.mark.parametrize('arr, expected_result, ',
                          [
                           ([-1, 1], [1, 1]),
                           ([-2, 3], [2, 3]),
                           ([-1, 10, -12, 3], [3.64005494, 11.41271221, 12.5, 6.26498204])
                           ],
                        )
@mock.patch('stream2segment.analysis.hilbert', side_effect=lambda *a,**k: hilbert(*a, **k))
@mock.patch('stream2segment.analysis.np')
def test_env(mock_np, mock_hilbert, arr, expected_result):
    mock_np.abs = mock.Mock(side_effect = lambda *a, **k: np.abs(*a, **k))
    r = env(arr)
    assert len(r) == len(arr)
    mock_hilbert.assert_called_once_with(arr)
    assert mock_np.abs.called
    g = 9    