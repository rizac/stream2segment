'''
Created on Apr 2, 2019

@author: riccardo
'''
import numpy as np
from stream2segment.gui.webapp.mainapp.plots import jsplot
from obspy.core.utcdatetime import UTCDateTime


def test_jsplot(data):
    stream = data.read_stream("trace_GE.APE.mseed")
    assert len(stream) == 1
    trace = stream[0]
    tstamp = trace.stats.starttime.timestamp
    plt1 = jsplot.Plot.fromstream(stream)
    plt2 = jsplot.Plot.fromtrace(trace)
    plt3 = jsplot.Plot().add(trace.stats.starttime, trace.stats.delta,
                             trace.data)

    x0_fromstream = plt1.data[0][0]
    x0_fromtrace = plt2.data[0][0]
    x0_custom = plt3.data[0][0]

    assert x0_fromstream == x0_fromtrace == x0_custom
    tstamp2 = UTCDateTime(x0_custom).timestamp
    assert np.abs(tstamp - tstamp2) < 1e-3
    data = plt1.tojson()[1]
    assert len(data) == 1
    [x0, dx, y, label] = data[0]
    assert x0 == UTCDateTime(x0_fromstream).isoformat() + 'Z'
    