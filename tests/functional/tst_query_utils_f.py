'''
Created on Feb 4, 2016

@author: riccardo
'''
# from event2waveform import getWaveforms
# from utils import date
# assert sys.path[0] == os.path.realpath(myPath + '/../../')
from mock import patch
import pytest
from mock import Mock
from datetime import datetime, timedelta
from stream2segment.query_utils import get_time_range, get_stations, get_events, get_arrival_time,\
get_search_radius, get_events, save_waveforms
from stream2segment.utils import datetime as dtime
from StringIO import StringIO
from obspy.taup.taup import getTravelTimes


#@patch('stream2segment.query_utils.getTravelTimes')
#def test_get_arrival_times(mock_get_tt):
#    pass


def test_to_datetime():
    pass


# class TestUrlopen(object):
#     def __init__(self, ret_val):
#         self.ret_val = ret_val
#         self.ret_switch = 0
# 
#     def read(self, *kwargs):
#         self.ret_switch = 1 - self.ret_switch
#         return self.ret_val if self.ret_switch == 1 else ''
# 
#     def close(self):
#         pass


def test_get_events():
    pass


def test_get_waveforms():
    pass


def test_get_stations():
    pass
    # mock_urlopen.return_value=Urlopen("a\nb\nc")
    # lst = getStations('a', 'b',  datetime.utcnow(), 4, 3 , 5)


def test_get_timerange():
    pass


def test_url_read():
    pass


def test_save_waveforms():
    pass
