'''
Created on Jul 25, 2016

@author: riccardo
'''
import unittest
import os
from obspy import read as obspy_read
from stream2segment.analysis import coda as coda_module

class Test(unittest.TestCase):


    def read_data_trace(self, file_name):
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
        return obspy_read(os.path.join(folder, file_name))

    def setUp(self):
        """setup function called **before** a 'test' is executed. a test is any
        class method with the word 'test' in it"""
        pass

    def tearDown(self):
        """setup function called **after** a 'test' is executed. a test is any
        class method with the word 'test' in it"""
        pass


    def test_coda_jessie_mseed(self):
        mseed = self.read_data_trace("20091217_231838.FR.ESCA.00.HHZ.SAC")
        ret = coda_module.analyze_coda(mseed)
        assert len(ret) == 1
        coda_result = ret[0]
        trace = mseed[0]
        coda_start_time = coda_result[0]
        assert coda_start_time > trace.stats.starttime
        coda_slope = coda_result[1]
        assert coda_slope < 0


    def test_coda_low_noise_level(self):
        mseed = self.read_data_trace("trace_GE.APE.mseed")
        ret = coda_module.analyze_coda(mseed)
        assert len(ret) == 1 and ret[0] is None


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()