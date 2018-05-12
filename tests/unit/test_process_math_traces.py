'''
Created on May 12, 2017

@author: riccardo
'''
from __future__ import division
from builtins import range
from past.utils import old_div
import unittest
import numpy as np
from numpy.fft import rfft
from numpy import true_divide as np_true_divide
from obspy.core.stream import read as o_read
from io import BytesIO
import os
from stream2segment.process.math.traces import fft, cumsumsq, cumtimes

import pytest
from mock.mock import patch, Mock
from datetime import datetime
from obspy.core.utcdatetime import UTCDateTime


class Test(object):

    # execute this fixture always even if not provided as argument:
    # https://docs.pytest.org/en/documentation-restructure/how-to/fixture.html#autouse-fixtures-xunit-setup-on-steroids
    @pytest.fixture(autouse=True)
    def init(self, request, data):
        self.mseed = data.read_stream("trace_GE.APE.mseed")

    def testCum(self):
        t = self.mseed[0]
        # we did not write any processing to the trace:
        assert 'processing' not in t.stats or not t.stats.processing
        c1 = cumsumsq(t)
        assert t is not c1
        assert not np.allclose(t.data, c1.data, equal_nan=True)
        assert max(c1.data) <= 1
        # we wrote processing information in the trace:
        assert c1.stats.processing
        assert cumsumsq.__name__ in c1.stats.processing[0]

        c3 = cumsumsq(t, normalize=False)
        assert t is not c3
        assert not np.allclose(c1.data, c3.data, equal_nan=True)
        assert max(c3.data) > 1
        # we wrote processing information in the trace:
        assert c3.stats.processing
        assert cumsumsq.__name__ in c3.stats.processing[0]

        c2 = cumsumsq(t, copy=False)
        assert t is c2
        assert max(c2.data) <= 1
        assert np.allclose(c1.data, c2.data, equal_nan=True)
        # we wrote processing information in the trace:
        assert t.stats.processing
        assert cumsumsq.__name__ in c3.stats.processing[0]


    def test_cumtimes(self):
        c1 = cumsumsq(self.mseed[0])
        t0, t1 = cumtimes(c1, 0, 1)
        assert t0 == c1.stats.starttime
        assert t1 == c1.stats.endtime

        c2 = cumsumsq(self.mseed[0], normalize=True)
        t0, t1 = cumtimes(c2, 0, 1)
        assert t0 == c2.stats.starttime
        assert t1 == c2.stats.endtime

        for c in [c1, c2]:
            t0, t1 = cumtimes(c, 0.1, .9)
            assert t0 > c.stats.starttime
            assert t1 < c.stats.endtime

        c1.data[-1] = np.nan
        t0, t1 = cumtimes(c1, 0, 1)
        assert t0 == c1.stats.starttime
        assert t1 == c1.stats.endtime - c1.stats.delta

        # test what happens if all are nans
        c1.data = np.array([np.nan] * len(c1.data))
        t0, t1 = cumtimes(c1, 0, 1)
        assert t0 == t1 == c1.stats.starttime
