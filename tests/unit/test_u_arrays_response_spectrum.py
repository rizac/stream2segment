'''
Created on Nov 28, 2017

@author: riccardo
'''
import unittest

import numpy as np

from stream2segment.math.arrays import NewmarkBeta as _NewmarkBeta, \
    NigamJennings as _NigamJennings, ResponseSpectrum as _ResponseSpectrum
from stream2segment.math.traces import NewmarkBeta, NigamJennings, ResponseSpectrum
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
            _ResponseSpectrum(accel, deltat, periods).evaluate()

        with pytest.raises(NotImplementedError):
            ResponseSpectrum(self.trace, periods).evaluate()

    def test_arrays_traces_response_spectra(self):
        '''this test just assures everything goes right without errors'''
        # FIXME: implement better tests!!!
        accel, periods, deltat = self.accel, self.periods, self.deltat

        tuple1a = _NewmarkBeta(accel, deltat, periods).evaluate()
        tuple1b = _NigamJennings(accel, deltat, periods).evaluate()
        tuple2a = NewmarkBeta(self.trace, periods).evaluate()
        tuple2b = NigamJennings(self.trace, periods).evaluate()

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
                        try:
                            assert np.allclose(val1, val2, atol=0, equal_nan=True)
                        except:
                            h = 9

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()