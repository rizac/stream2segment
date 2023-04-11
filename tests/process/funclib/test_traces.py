"""
Created on May 12, 2017

@author: riccardo
"""
import pytest
import numpy as np

from stream2segment.process.funclib.traces import cumsumsq


class Test:

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


def test_searchsorted():
    """this test is just a check to assure that the new implementation of timeswhere
    works as the original code"""
    arr = [1, 4.5, 6]
    tosearch = [-1, 3, 4.5, 6.0, 8.1]
    assert (np.array([np.searchsorted(arr, v) for v in tosearch]) \
        == np.searchsorted(arr, tosearch)).all()


