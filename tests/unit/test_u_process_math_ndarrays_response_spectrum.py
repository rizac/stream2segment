'''
Created on Nov 28, 2017

@author: riccardo
'''
import unittest

import numpy as np

from stream2segment.process.math.ndarrays import ResponseSpectrum as a_ResponseSpectrum
from stream2segment.process.math.traces import ResponseSpectrum as t_ResponseSpectrum

from stream2segment.process.math.ndarrays import respspec as a_rs
from stream2segment.process.math.traces import respspec as t_rs
import pytest
from obspy.core.trace import Trace


class Test(unittest.TestCase):

    def setUp(self):
        self.accel = np.array([1, 2, 1, 2, 1, 2])
        self.periods = np.array([1, 2])
        self.deltat = 0.1
        self.trace = Trace(data=self.accel, header={'delta': self.deltat})
        pass

    def tearDown(self):
        pass

    def testName(self):
        pass

    def test_abstract(self):
        accel, periods, deltat = self.accel, self.periods, self.deltat

        with pytest.raises(NotImplementedError):
            a_ResponseSpectrum(accel, deltat, periods).evaluate()

        with pytest.raises(NotImplementedError):
            t_ResponseSpectrum(self.trace, periods).evaluate()

    def test_arrays_traces_response_spectra(self):
        '''this test just assures everything goes right without errors'''
        # FIXME: implement better tests!!!
        accel, periods, deltat = self.accel, self.periods, self.deltat

        tuple1a = a_rs('NewmarkBeta', accel, deltat, periods)
        tuple1b = a_rs('NigamJennings', accel, deltat, periods)
        tuple2a = t_rs('NewmarkBeta', self.trace, periods)
        tuple2b = t_rs('NigamJennings', self.trace, periods)

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

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
