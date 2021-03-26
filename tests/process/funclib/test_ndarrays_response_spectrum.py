'''
Created on Nov 28, 2017

@author: riccardo
'''
import numpy as np
import pytest
from obspy.core.trace import Trace

from stream2segment.process.lib.ndarrays import ResponseSpectrum as a_ResponseSpectrum
from stream2segment.process.lib.traces import ResponseSpectrum as t_ResponseSpectrum
from stream2segment.process.lib.ndarrays import respspec as a_rs
from stream2segment.process.lib.traces import respspec as t_rs


@pytest.fixture(scope='module')
def shareddata(request):
    accel = np.array([1, 2, 1, 2, 1, 2])
    periods = np.array([1, 2])
    deltat = 0.1
    trace = Trace(data=accel, header={'delta': deltat})
    return accel, periods, deltat, trace


def test_abstract(shareddata):
    accel, periods, deltat, trace = shareddata

    with pytest.raises(NotImplementedError):
        a_ResponseSpectrum(accel, deltat, periods).evaluate()

    with pytest.raises(NotImplementedError):
        t_ResponseSpectrum(trace, periods).evaluate()


def test_arrays_traces_response_spectra(shareddata):
    '''this test just assures everything goes right without errors'''
    # FIXME: implement better tests!!!
    accel, periods, deltat, trace = shareddata

    tuple1a = a_rs('NewmarkBeta', accel, deltat, periods)
    tuple1b = a_rs('NigamJennings', accel, deltat, periods)
    tuple2a = t_rs('NewmarkBeta', trace, periods)
    tuple2b = t_rs('NigamJennings', trace, periods)

    # compare dicts:
    for tup1, tup2 in [[tuple1a, tuple2a], [tuple1b, tuple2b]]:
        for dic1, dic2 in zip(tup1, tup2):
            if hasattr(dic1, 'keys'):
                vals = [[dic1[key], dic2[key]] for key in dic1]
            else:
                vals = [[dic1, dic2]]
            for val1, val2 in vals:
                try:
                    assert val1 == val2
                except ValueError:
                    # arrays, assert allclose:
                    assert np.allclose(val1, val2, atol=0, equal_nan=True)
