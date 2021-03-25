'''
Created on Jul 25, 2016

@author: riccardo
'''
import os
from obspy import read as obspy_read
from stream2segment.process.lib import coda as coda_module

def test_coda_jessie_mseed(data):
    mseed = data.read_stream("20091217_231838.FR.ESCA.00.HHZ.SAC")
    trace = mseed[0]
    coda_result = coda_module.analyze_coda(trace)
    coda_start_time = coda_result[0]
    assert coda_start_time > trace.stats.starttime
    coda_slope = coda_result[1]
    assert coda_slope < 0


def test_coda_low_noise_level(data):
    mseed = data.read_stream("trace_GE.APE.mseed")
    ret = coda_module.analyze_coda(mseed[0])
    assert ret is None
