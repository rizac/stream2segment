'''
Created on Oct 7, 2017

@author: riccardo
'''
import unittest
import os
from obspy.core.stream import read
from stream2segment.process.utils import get_stream
from mock import patch
from io import BytesIO
import pytest
import time
from tempfile import NamedTemporaryFile as NamedTemporaryFile_

class MockSegment(object):
     def __init__(self, data):
         self.data = data

class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    @patch('obspy.core.stream.NamedTemporaryFile', return_value = NamedTemporaryFile_())
    def test_get_stream(self, mock_ntf):
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
        mseed_path = os.path.join(folder, 'trace_GE.APE.mseed')
        with open(mseed_path, 'rb') as opn:
            mseeddata = opn.read()

        segment = MockSegment(mseeddata)
        tobspy = time.time()
        stream_obspy = read(BytesIO(mseeddata))
        tobspy = time.time() - tobspy
        tme = time.time()
        stream_me = get_stream(segment)
        tme = time.time() - tme
        # assert we are faster (actually that calling read with format='MSEED' is faster than
        # calling with format=None)
        assert tme < tobspy
        assert (stream_obspy[0].data == stream_me[0].data).all()
        assert not mock_ntf.called

        with pytest.raises(TypeError):
            stream_obspy = read(BytesIO(mseeddata[:5]))
        assert mock_ntf.called

        mock_ntf.reset_mock()
        segment = MockSegment(mseeddata[:5])
        with pytest.raises(ValueError):
            stream_me = get_stream(segment)
        assert not mock_ntf.called

        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()