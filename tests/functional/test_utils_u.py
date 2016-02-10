'''
Created on Feb 4, 2016

@author: riccardo
'''
# from event2waveform import getWaveforms
# from utils import date
# assert sys.path[0] == os.path.realpath(myPath + '/../../')
from mock import patch
from mock import Mock
from datetime import datetime, timedelta
from seiswavproc.utils import getWaveforms, getTimeRange, getStations, getEvents
from StringIO import StringIO
import seiswavproc

def test_get_timerange():
#     d = datetime.utcnow()
#     d1, d2 = getTimeRange(d, timedelta(days=1))
#     assert d-d1 == d2-d == timedelta(days=1)
# 
#     d = datetime.utcnow()
#     d1, d2 = getTimeRange(d, timedelta(days=1), timedelta(days=2))
#     assert d-d1 == timedelta(days=1)
#     assert d2-d == timedelta(days=2)
    
    d = datetime.utcnow()
    d1, d2 = getTimeRange(d, days=1)
    assert d-d1 == d2-d == timedelta(days=1)

    d = datetime.utcnow()
    d1, d2 = getTimeRange(d)
    assert d == d1 and d == d2

    d1, d2 = getTimeRange(d, days=(1,2))
    assert d-d1 == timedelta(days=1)
    assert d2-d == timedelta(days=2)

    d1, d2 = getTimeRange(d, days=(1, 2), minutes=1)
    assert d-d1 == timedelta(days=1, minutes=1)
    assert d2-d == timedelta(days=2, minutes=1)


class Urlopen(object):
    def __init__(self, ret_val):
        self.ret_val = ret_val
        self.ret_switch=0
    def read(self, *kwargs):
        self.ret_switch = 1 - self.ret_switch
        return self.ret_val if self.ret_switch == 1 else ''
    def close(self):
        pass


@patch('seiswavproc.utils.ul.Request', return_value='Request')
@patch('seiswavproc.utils.ul.urlopen', return_value=Urlopen('a'))
def test_get_waveforms(mock_urlopen, mock_request):
    a, b = getWaveforms('a', 'b', 'c', 'd', '3' , '5')
    assert not a and not b
    assert not mock_request.called
    assert not mock_urlopen.called
    
    a, b = getWaveforms('a', 'b', 'c',  datetime.utcnow(), '3' , '5')
    assert not a and not b
    assert not mock_request.called
    assert not mock_urlopen.called

    a, b = getWaveforms('a', 'b', 'c', 'd', 3, 5)
    assert not a and not b
    assert not mock_request.called
    assert not mock_urlopen.called


@patch('seiswavproc.utils.ul.Request', return_value='Request')
@patch('seiswavproc.utils.ul.urlopen', return_value=Urlopen('a'))
def test_get_stations(mock_urlopen, mock_request):
    lst = getStations('a', 'b', 'c', 'd' , '5' ,'6')
    assert not lst
    assert not mock_request.called
    assert not mock_urlopen.called
    
    try:
        lst = getStations('a', 'b',  datetime.utcnow(), '4', '3' , '5')
        assert False
    except TypeError:
        assert not mock_request.called
        assert not mock_urlopen.called
#     assert not a and not b
#     assert not mock_request.called
#     assert not mock_urlopen.called

    lst = getStations('a', 'b',  datetime.utcnow(), 4, 3 , 5)
    assert not lst
    assert mock_request.called
    assert mock_urlopen.called
    
    # mock_urlopen.return_value=Urlopen("a\nb\nc")
    # lst = getStations('a', 'b',  datetime.utcnow(), 4, 3 , 5)
    
    
# 1995-07-14T00:00:00

# for side effect below, see https://docs.python.org/3/library/unittest.mock-examples.html#partial-mocking
@patch('seiswavproc.utils.datetime', side_effect=lambda *args, **kw: datetime(*args, **kw))
def test_to_datetime(mock_datetime):
    dt = seiswavproc.utils.to_datetime("asd")  # invalid date (string)
    assert dt is None
    assert not mock_datetime.called
    
    dt = seiswavproc.utils.to_datetime("-45")  # invalid date (invalid number)
    assert dt is None
    assert mock_datetime.called_with(45)
    
    dt = seiswavproc.utils.to_datetime("6-46-6")
    assert dt is None
    assert mock_datetime.called_with(6,46,6)
    
    
    dt = seiswavproc.utils.to_datetime("2006-6-6")
    assert dt == datetime(2006,6,6)
    assert mock_datetime.called_with(2006, 6, 6)
    

    