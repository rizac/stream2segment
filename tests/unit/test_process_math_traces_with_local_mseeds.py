'''
Created on Feb 23, 2016

@author: riccardo
'''
import os
import sys

import pytest
import numpy as np
import mock
from obspy.core import Trace
from obspy.core.utcdatetime import UTCDateTime

from stream2segment.process.lib.ndarrays import fft as orig_fft
from stream2segment.process.lib.traces import fft, bandpass, dfreq, maxabs, timeof


@pytest.mark.parametrize('arr, arr_len_after_trim, fft_npts',
                        [([1, 2, 3, 4, 5, 6], 6, 4),
                         ([1, 2, 3, 4, 5], 5, 3),
                         ([1, 2, 3, 4], 4, 3),
                         ([1, 2, 3], 3, 2),
                         ])
@mock.patch('stream2segment.process.lib.traces._fft',
            side_effect=lambda *a, **k: orig_fft(*a, **k))
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


@pytest.fixture(scope="module")
def _data(data):
    """returns a dict with fields 'mseed', 'mseed_ACC', 'mseed_VEL', 'mseed_DISP' (all Streams.
    The last three after removing the response) and  'inventory' (the stream inventory object
    used to remove the response)"""
    inv_name = 'inventory_GE.APE.xml'
    inv_obj = data.read_inv(inv_name)
    ret = {'inventory': data.read_inv(inv_name)}
    for inv_output in [None, 'ACC', 'VEL', 'DISP']:
        key = 'mseed' + ('' if not inv_output else "_" + inv_output)
        ret[key] = data.read_stream('trace_GE.APE.mseed', inv_name if inv_output else None,
                                    inv_output)
    return ret


def test_bandpass(_data):
    trace = _data['mseed'][0]
    res = bandpass(trace, 2, 3)
    assert not np.array_equal(trace.data, res.data)
    assert trace.stats.starttime == res.stats.starttime
    assert trace.stats.endtime == res.stats.endtime
    assert trace.stats.npts == res.stats.npts
    assert len(trace.data) == len(res.data)


def testmaxabs(_data):
    mseed = _data['mseed']
    trace = mseed[0]

    t, g = maxabs(trace)

    assert np.max(np.abs(trace.data)) == g
    idx = np.argmax(np.abs(trace.data))

    assert timeof(trace, idx) == t

    # assert by slicing times of max are different:
    td = 2 * trace.stats.delta
    assert maxabs(trace, None, t-td)[0] < t < maxabs(trace, t+td, None)[0]

    data = trace.data
    npts = trace.stats.npts
    t, g = maxabs(trace, None, trace.stats.starttime-td)
    assert t == UTCDateTime(0) and np.isnan(g)
    # check that data has not been changed by trace.slice (called by maxabs)
    # this is for safety:
    assert data is trace.data
    assert len(trace.data) == npts  # further safety check
