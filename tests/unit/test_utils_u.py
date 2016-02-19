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
from stream2segment.utils import getWaveforms, getTimeRange, getStations, getEvents, to_datetime,\
    getArrivalTime, getSearchRadius, getEvents, url_read, saveWaveforms
from StringIO import StringIO
import stream2segment


@patch('stream2segment.utils.getTravelTimes', return_value = 'a')
def test_get_arrival_times(mock_get_tt):

    with pytest.raises(AttributeError):
        _ = getArrivalTime('d', 'q', 'g')

    from obspy.taup.taup import getTravelTimes as gtt

    mock_get_tt.side_effect = lambda *args, **kw: gtt(*args, **kw)

    with pytest.raises(IOError):
        _ = getArrivalTime('d', 'q', 'g')

    tt = getArrivalTime(dist=52.474, depth=611.0, model='ak135')
    # check for the value (account for round errors):
    assert tt > 497.525385547 and tt < 497.525385548

    def gtt2(*args, **kw):
        ret = gtt(*args, **kw)
        for r in ret:
            if 'phase_name' in r:
                r['phase_name'] = ''
        return ret

    mock_get_tt.side_effect = gtt2
    a = getArrivalTime(dist=52.474, depth=611.0, model='ak135')
    assert a is None


@pytest.mark.parametrize('mag, args, expected_val',
                         [(5, None, 3), (2, None, 1), (-1, None, 1), (7, None, 5), (8, None, 5),
                          (5, [3, 7, 1, 5], 3), (2, [3, 7, 1, 5], 1), (-1, [3, 7, 1, 5], 1),
                          (7, [3, 7, 1, 5], 5), (8, [3, 7, 1, 5], 5)])
def test_getSearchRadius(mag, args, expected_val):
    if args is None:
        assert getSearchRadius(mag) == expected_val
    else:
        args.insert(0, mag)
        assert getSearchRadius(*args) == expected_val


# for side effect below
# see https://docs.python.org/3/library/unittest.mock-examples.html#partial-mocking
@patch('stream2segment.utils.datetime', side_effect=lambda *args, **kw: datetime(*args, **kw))
def test_to_datetime(mock_datetime):
    mock_datetime.reset_mock()
    dt = to_datetime("asd")  # invalid date (string)
    assert dt is None
    assert not mock_datetime.called

    mock_datetime.reset_mock()
    dt = to_datetime("-45")  # invalid date (invalid number)
    assert dt is None
    mock_datetime.assert_called_with(45)

    mock_datetime.reset_mock()
    dt = to_datetime("6-46-6")
    assert dt is None
    mock_datetime.assert_called_with(6, 46, 6)

    mock_datetime.reset_mock()
    dt = to_datetime("2006-6-6")
    assert dt == datetime(2006, 6, 6)
    mock_datetime.assert_called_with(2006, 6, 6)

    mock_datetime.reset_mock()
    dt = to_datetime("1995-07-14T00:00:00")
    assert dt == datetime(1995, 7, 14)
    mock_datetime.assert_called_with(1995, 7, 14, 0, 0, 0)


# @patch('stream2segment.utils.ul.Request', return_value='Request')
# @patch('stream2segment.utils.ul.urlopen', return_value=Urlopen('read'))
@patch('stream2segment.utils.url_read', return_value='url_read')
def test_get_events(mock_url_read):  # , mock_urlopen, mock_request):
    with pytest.raises(KeyError):
        getEvents()

    args = {'eventws': 'eventws', 'minmag': 1.1,
            'start': datetime.now().isoformat(),
            'end': datetime.now().isoformat(),
            'minlon': '90', 'maxlon': '80',
            'minlat': '85', 'maxlat': '57'}

    mock_url_read.reset_mock()
    lst = getEvents(**args)
    assert not lst
    assert mock_url_read.called

    mock_url_read.reset_mock()
    mock_url_read.return_value = 'header\na|b|c'
    lst = getEvents(**args)
    assert not lst
    assert mock_url_read.called

    # value error:
    mock_url_read.reset_mock()
    mock_url_read.return_value = 'header\na|'+datetime.now().isoformat()+'|c'
    lst = getEvents(**args)
    assert not lst
    assert mock_url_read.called

    # index error:
    mock_url_read.reset_mock()
    mock_url_read.return_value = 'header\na|'+datetime.now().isoformat()+'|1.1'
    lst = getEvents(**args)
    assert not lst
    assert mock_url_read.called

    mock_url_read.reset_mock()
    d = datetime.now()
    mock_url_read.return_value = 'header\na|'+d.isoformat()+'|1.1|2|3.0|4.0|a|b|c|d|1.1'
    lst = getEvents(**args)
    assert len(lst) == 1
    assert lst[0] == ['a', d, 1.1, 2.0, 3.0, '4.0', 'a', 'b', 'c', 'd', 1.1]
    assert mock_url_read.called


@patch('stream2segment.utils.url_read', return_value='url_read')
def test_get_waveforms(mock_url_read):
    mock_url_read.reset_mock()
    a, b = getWaveforms('a', 'b', 'c', 'd', '3', '5')
    assert not a and not b
    assert not mock_url_read.called

    mock_url_read.reset_mock()
    a, b = getWaveforms('a', 'b', 'c',  datetime.utcnow(), '3', '5')
    assert not a and not b
    assert not mock_url_read.called

    with patch('stream2segment.utils.getTimeRange') as mock_get_tr:
        mock_url_read.reset_mock()
        d1 = datetime.now()
        d2 = d1 + timedelta(seconds=1)
        mock_get_tr.return_value = d1, d2
        a, b = getWaveforms('a', 'b', 'c', 'd', '3', '5')
        assert a == 'c' and b == mock_url_read.return_value
        assert mock_url_read.called
        mock_get_tr.assert_called_with('d', minutes=('3','5'))

        mock_url_read.reset_mock()
        a, b = getWaveforms('a', 'b', 'c*', 'd', '3', '5')
        assert a == 'c' and b == mock_url_read.return_value
        assert mock_url_read.called
        mock_get_tr.assert_called_with('d', minutes=('3','5'))

        mock_url_read.reset_mock()
        a, b = getWaveforms('a', 'b', [], 'd', '3', '5')
        assert not a and not b
        assert not mock_url_read.called
        mock_get_tr.assert_called_with('d', minutes=('3','5'))

        mock_url_read.reset_mock()
        mock_get_tr.side_effect = lambda *args, **kw: getTimeRange(*args, **kw)
        a, b = getWaveforms('a', 'b', 'c', 'd', '3', '5')
        assert not a and not b
        assert not mock_url_read.called
        mock_get_tr.assert_called_with('d', minutes=('3','5'))


@patch('stream2segment.utils.url_read', return_value='url_read')
def test_get_stations(mock_url_read):
    mock_url_read.reset_mock()
    lst = getStations('a', 'b', 'c', 'd', '5', '6')
    assert not lst
    assert not mock_url_read.called

    mock_url_read.reset_mock()
    with pytest.raises(TypeError):
        lst = getStations('a', 'b',  datetime.utcnow(), '4', '3', '5')
        # assert not mock_url_read.called

    mock_url_read.reset_mock()
    lst = getStations('a', 'b',  datetime.utcnow(), 4, 3, 5)
    assert not lst
    assert mock_url_read.called

    with patch('stream2segment.utils.getTimeRange') as mock_get_timerange:
        mock_url_read.reset_mock()
        mock_get_timerange.return_value = (datetime.now(), datetime.now()+timedelta(seconds=1))
        d = datetime.now()
        mock_url_read.return_value = 'header\na|b|c'
        with pytest.raises(TypeError):
            lst = getStations('dc', ['listCha'], d, 'lat', 'lon', 'dist')
            # mock_get_timerange.assert_called_with(d, 1)
            # assert not mock_url_read.called

        mock_url_read.reset_mock()
        with pytest.raises(IndexError):
            lst = getStations('dc', ['listCha'], d, 3.1, 2.5, 1.1)
            # mock_get_timerange.assert_called_with(d, 1)
            # assert mock_url_read.called

        mock_url_read.reset_mock()
        mock_url_read.return_value = 'header\na|b|c|d|e|f|g|h'
        lst = getStations('dc', ['listCha'], d, 3.1, 2.5, 1.1)
        mock_get_timerange.assert_called_with(d, days=1)
        assert mock_url_read.called
        assert not lst

        mock_url_read.reset_mock()
        mock_url_read.return_value = 'header\na|b|1|1.1|2.0|f|'+d.isoformat()+'|h'
        lst = getStations('dc', ['listCha'], d, 3.1, 2.5, 1.1)
        mock_get_timerange.assert_called_with(d, days=1)
        assert mock_url_read.called
        assert len(lst) == 1
        assert lst[0][6] == d
        assert lst[0][7] == None

        mock_url_read.reset_mock()
        d2 = datetime.now()
        mock_url_read.return_value = 'header\na|b|1|1.1|2.0|f|'+d.isoformat()+'|'+d2.isoformat()
        lst = getStations('dc', ['listCha'], d, 3.1, 2.5, 1.1)
        mock_get_timerange.assert_called_with(d, days=1)
        assert mock_url_read.called
        assert len(lst) == 1
        assert lst[0][6] == d
        assert lst[0][7] == d2


@patch('stream2segment.utils.timedelta', side_effect=lambda *args, **kw: timedelta(*args, **kw))
def test_get_timerange(mock_timedelta):
    mock_timedelta.reset_mock()
    d = datetime.utcnow()
    d1, d2 = getTimeRange(d, days=1)
    assert d-d1 == d2-d == timedelta(days=1)

    mock_timedelta.reset_mock()
    d = datetime.utcnow()
    d1, d2 = getTimeRange(d, days=1, minutes=(1, 2))
    assert d-d1 == timedelta(days=1, minutes=1)
    assert d2-d == timedelta(days=1, minutes=2)

    mock_timedelta.reset_mock()
    d = datetime.utcnow()
    _, _ = getTimeRange(d, days=1)
    assert mock_timedelta.called

    mock_timedelta.reset_mock()
    _, _ = getTimeRange(d)
    assert mock_timedelta.called

    mock_timedelta.reset_mock()
    _, _ = getTimeRange(d, days=(1, 2))
    assert mock_timedelta.called

    mock_timedelta.reset_mock()
    _, _ = getTimeRange(d, days=(1, 2), minutes=1)
    assert mock_timedelta.called

    mock_timedelta.reset_mock()
    with pytest.raises(Exception):
        _, _ = getTimeRange(d, days="abc", minutes=1)
        # assert mock_timedelta.called


# @patch('mod_a.urllib2.urlopen')
# def mytest(mock_urlopen):
#     a = Mock()
#     a.read.side_effect = ['resp1', 'resp2']
#     mock_urlopen.return_value = a
#     res = mod_a.myfunc()
#     print res
#     assert res == 'resp1'
# 
#     res = mod_a.myfunc()
#     print res
#     assert res == 'resp2'

@patch('stream2segment.utils.ul')
def test_url_read(mock_ul):  # mock_ul_urlopen, mock_ul_request, mock_ul):
    blockSize=1024*1024
    
    mock_ul.Request.side_effect = lambda arx: arx
    
    def xyz(argss):
        return StringIO(argss)

    # mock_ul.urlopen = Mock()
    mock_ul.urlopen.side_effect = xyz
    # mock_ul.urlopen.return_value = lambda arg: StringIO(arg)
    
    val = 'url'
    assert url_read(val, "name") == val
    mock_ul.Request.assert_called_with(val)
    mock_ul.urlopen.assert_called_with(val)
    # mock_ul.urlopen.read.assert_called_with(blockSize)

    def ioerr(**kwargs):
        ret = IOError()
        for key, value in kwargs.iteritems():
            setattr(ret, key, value)
        return ret
    
    for kwz in [{'reason':'reason'}, {'code': 'code'}, {}]:
        def xyz2(**kw):
            raise ioerr(**kw)
    
        mock_ul.urlopen.side_effect = lambda arg: xyz2(**kwz)
        assert url_read(val, "name") == ''
        mock_ul.Request.assert_called_with(val)
        mock_ul.urlopen.assert_called_with(val)
        assert not mock_ul.urlopen.read.called
    
    def xyz3():
        raise ValueError()
    mock_ul.urlopen.side_effect = lambda arg: xyz3()
    assert url_read(val, "name") == ''
    mock_ul.Request.assert_called_with(val)
    mock_ul.urlopen.assert_called_with(val)
    assert not mock_ul.urlopen.read.called
    
    def xyz4():
        raise AttributeError()
    mock_ul.urlopen.side_effect = lambda arg: xyz4()
    with pytest.raises(AttributeError):
        _ = url_read(val, "name")
        
    def xyz5(argss):
        class sio(StringIO):
            def read(self, *args, **kw):
                raise IOError('oops')
        return sio(argss)
    mock_ul.urlopen.side_effect = lambda arg: xyz5(arg)
    assert url_read(val, "name") == ''
    mock_ul.Request.assert_called_with(val)
    mock_ul.urlopen.assert_called_with(val)
    # mock_ul.urlopen.read.assert_called_with(blockSize)
    
#     def excp():
#         raise IOError('oops')
#     mock_ul.urlopen.read.side_effect = excp
#     assert url_read(val, "name") == ''
    
# @patch('stream2segment.utils.ul.urlopen')
# def test_url_read(mock_ul_urlopen):  # mock_ul_urlopen, mock_ul_request, mock_ul):
#     a = Mock()
#     a.read.side_effect = ['resp1', 'resp2']
#     mock_ul_urlopen.return_value = a
#     
#     val = 'url'
#     assert url_read(val, "name") == "resp1"
#     
#     assert url_read(val, "name") == "resp2"
#     
#     pass


@patch('stream2segment.utils.locations2degrees', return_value = 'l2d')
@patch('stream2segment.utils.getArrivalTime')
@patch('stream2segment.utils.getEvents')
@patch('stream2segment.utils.getStations')
@patch('stream2segment.utils.getWaveforms')
@patch('stream2segment.utils.os.path.exists', return_value=False)
def test_save_waveforms_nopath(mock_os_path_exists, mock_gw, mock_gs, mock_ge, mock_gat, mock_ltd):
    mock_os_path_exists.side_effect = lambda arg: False
    saveWaveforms('eventws', 'minmag', 'minlat', 'maxlat', 'minlon', 'maxlon', 
                  'distFromEvent', 'datacenters_dict',
                  'channelList', 'start', 'end', 'minBeforeP', 'minAfterP', 'outpath')
    mock_os_path_exists.assert_called_with('outpath')
    assert not mock_ge.called and not mock_gs.called and not mock_gw.called and \
        not mock_gat.called and not mock_ltd.called


@patch('stream2segment.utils.locations2degrees', return_value = 'l2d')
@patch('stream2segment.utils.getArrivalTime')
@patch('stream2segment.utils.getEvents')
@patch('stream2segment.utils.getStations')
@patch('stream2segment.utils.getWaveforms')
@patch('stream2segment.utils.os.path.exists', return_value=True)
def test_save_waveforms_getevents_returns_empty(mock_os_path_exists, mock_gw, mock_gs, mock_ge, mock_gat, mock_ltd):

    mock_ge.side_effect = lambda **args: []
    saveWaveforms('eventws', 'minmag', 'minlat', 'maxlat', 'minlon', 'maxlon', 
                  'distFromEvent', 'datacenters_dict',
                  'channelList', 'start', 'end', 'minBeforeP', 'minAfterP', 'outpath')
    mock_os_path_exists.assert_called_with('outpath')
    mock_ge.assert_called_with(**{"eventws": "eventws",
                                  "minmag": "minmag",
                                  "minlat": "minlat",
                                  "maxlat": "maxlat",
                                  "minlon": "minlon",
                                  "maxlon": "maxlon",
                                  "start": "start",
                                  "end": "end",
                                  "outpath": "outpath"})
    assert not mock_gs.called and not mock_gw.called and not mock_gat.called and not mock_ltd.called


# global vars (FIXME: check if good!)
dcs = {'dc1' : 'www.dc1'}
channels = [['a', 'b' , 'c']]


@patch('stream2segment.utils.locations2degrees', return_value='l2d')
@patch('stream2segment.utils.getArrivalTime')
@patch('stream2segment.utils.getEvents', return_value=[[str(i) for i in xrange(12)]])
@patch('stream2segment.utils.getStations')
@patch('stream2segment.utils.getWaveforms')
@patch('stream2segment.utils.os.path.exists', return_value=True)
def test_save_waveforms_indexerr_on_getevents(mock_os_path_exists, mock_gw, mock_gs, mock_ge,
                                              mock_gat, mock_ltd):
    with pytest.raises(IndexError):
        saveWaveforms('eventws', 'minmag', 'minlat', 'maxlat', 'minlon', 'maxlon',
                      'distFromEvent', dcs,
                      channels, 'start', 'end', 'minBeforeP', 'minAfterP', 'outpath')


@patch('stream2segment.utils.locations2degrees', return_value='l2d')
@patch('stream2segment.utils.getArrivalTime')
@patch('stream2segment.utils.getEvents', return_value=[[str(i) for i in xrange(13)]])
@patch('stream2segment.utils.getStations', return_value=[])
@patch('stream2segment.utils.getWaveforms')
@patch('stream2segment.utils.os.path.exists', return_value=True)
def test_save_waveforms_getstations_returns_empty(mock_os_path_exists, mock_gw, mock_gs, mock_ge,
                                                  mock_gat, mock_ltd):

    saveWaveforms('eventws', 'minmag', 'minlat', 'maxlat', 'minlon', 'maxlon',
                  'distFromEvent', dcs,
                  channels, 'start', 'end', 'minBeforeP', 'minAfterP', 'outpath')
    mock_os_path_exists.assert_called_with('outpath')
    mock_ge.assert_called_with(**{"eventws": "eventws",
                                  "minmag": "minmag",
                                  "minlat": "minlat",
                                  "maxlat": "maxlat",
                                  "minlon": "minlon",
                                  "maxlon": "maxlon",
                                  "start": "start",
                                  "end": "end",
                                  "outpath": "outpath"})

    ev = mock_ge.return_value[0]
    mock_gs.assert_called_with(dcs.values()[0], channels[0], ev[1], ev[2], ev[3],
                               'distFromEvent')
    assert not mock_gw.called


@patch('stream2segment.utils.locations2degrees', return_value='l2d')
@patch('stream2segment.utils.getArrivalTime')
@patch('stream2segment.utils.getEvents', return_value=[[str(i) for i in xrange(13)]])
@patch('stream2segment.utils.getStations', return_value=[[str(i) for i in xrange(3)]])
@patch('stream2segment.utils.getWaveforms')
@patch('stream2segment.utils.os.path.exists', return_value=True)
def test_save_waveforms_indexerr_on_getstations(mock_os_path_exists, mock_gw, mock_gs, mock_ge,
                                                mock_gat, mock_ltd):
    with pytest.raises(IndexError):
        saveWaveforms('eventws', 'minmag', 'minlat', 'maxlat', 'minlon', 'maxlon',
                      'distFromEvent', dcs,
                      channels, 'start', 'end', 'minBeforeP', 'minAfterP', 'outpath')


@patch('stream2segment.utils.locations2degrees', return_value='l2d')
@patch('stream2segment.utils.getArrivalTime', return_value=None)
@patch('stream2segment.utils.getEvents', return_value=[[str(i) for i in xrange(13)]])
@patch('stream2segment.utils.getStations', return_value=[[str(i) for i in xrange(4)]])
@patch('stream2segment.utils.getWaveforms')
@patch('stream2segment.utils.os.path.exists', return_value=True)
def test_save_waveforms_get_arrival_time_none(mock_os_path_exists, mock_gw, mock_gs, mock_ge,
                                              mock_gat, mock_ltd):
    saveWaveforms('eventws', 'minmag', 'minlat', 'maxlat', 'minlon', 'maxlon',
                  'distFromEvent', dcs,
                  channels, 'start', 'end', 'minBeforeP', 'minAfterP', 'outpath')
    mock_os_path_exists.assert_called_with('outpath')
    mock_ge.assert_called_with(**{"eventws": "eventws",
                                  "minmag": "minmag",
                                  "minlat": "minlat",
                                  "maxlat": "maxlat",
                                  "minlon": "minlon",
                                  "maxlon": "maxlon",
                                  "start": "start",
                                  "end": "end",
                                  "outpath": "outpath"})
    ev = mock_ge.return_value[0]
    st = mock_gs.return_value[0]
    mock_gs.assert_called_with(dcs.values()[0], channels[0], ev[1], ev[2], ev[3],
                               'distFromEvent')
    mock_ltd.assert_called_with(ev[2], ev[3], st[2], st[3])
    mock_gat.assert_called_with(mock_ltd.return_value, ev[4])
    assert not mock_gw.called


@patch('__builtin__.open')
@patch('stream2segment.utils.locations2degrees', return_value='l2d')
@patch('stream2segment.utils.getArrivalTime', return_value=5)
@patch('stream2segment.utils.getEvents', return_value=[[str(i) for i in xrange(13)]])
@patch('stream2segment.utils.getStations', return_value=[[str(i) for i in xrange(4)]])
@patch('stream2segment.utils.getWaveforms', return_value=('', ''))
@patch('stream2segment.utils.os.path.exists', return_value=True)
@patch('stream2segment.utils.os.path.join', return_value='joined')
def test_save_waveforms_get_arrival_time_no_wav(mock_os_path_join, mock_os_path_exists, mock_gw,
                                                mock_gs, mock_ge,
                                                mock_gat, mock_ltd, mock_open):
    d = datetime.now()
    evz = mock_ge.return_value
    evz[0][1] = d
    mock_ge.return_value = evz
    saveWaveforms('eventws', 'minmag', 'minlat', 'maxlat', 'minlon', 'maxlon',
                  'distFromEvent', dcs,
                  channels, 'start', 'end', 'minBeforeP', 'minAfterP', 'outpath')
    mock_os_path_exists.assert_called_with('outpath')
    mock_ge.assert_called_with(**{"eventws": "eventws",
                                  "minmag": "minmag",
                                  "minlat": "minlat",
                                  "maxlat": "maxlat",
                                  "minlon": "minlon",
                                  "maxlon": "maxlon",
                                  "start": "start",
                                  "end": "end",
                                  "outpath": "outpath"})
    ev = mock_ge.return_value[0]
    st = mock_gs.return_value[0]
    mock_gs.assert_called_with(dcs.values()[0], channels[0], ev[1], ev[2], ev[3],
                               'distFromEvent')
    mock_ltd.assert_called_with(ev[2], ev[3], st[2], st[3])
    mock_gat.assert_called_with(mock_ltd.return_value, ev[4])
    origTime = ev[1] + timedelta(seconds=float(mock_gat.return_value))
    mock_gw.assert_called_with(dcs.values()[0], st[1], channels[0], origTime, 'minBeforeP',
                               'minAfterP')
    assert not mock_os_path_join.called
    assert not mock_open.called


@patch('__builtin__.open')
@patch('stream2segment.utils.locations2degrees', return_value='l2d')
@patch('stream2segment.utils.getArrivalTime', return_value=5)
@patch('stream2segment.utils.getEvents', return_value=[[str(i) for i in xrange(13)]])
@patch('stream2segment.utils.getStations', return_value=[[str(i) for i in xrange(4)]])
@patch('stream2segment.utils.getWaveforms', return_value=('', 'wav'))
@patch('stream2segment.utils.os.path.exists', return_value=True)
@patch('stream2segment.utils.os.path.join', return_value='joined')
def test_save_waveforms_get_arrival_time(mock_os_path_join, mock_os_path_exists, mock_gw, mock_gs,
                                         mock_ge, mock_gat, mock_ltd, mock_open):
    d = datetime.now()
    evz = mock_ge.return_value
    evz[0][1] = d
    mock_ge.return_value = evz

    saveWaveforms('eventws', 'minmag', 'minlat', 'maxlat', 'minlon', 'maxlon',
                  'distFromEvent', dcs, channels, 'start', 'end', 'minBeforeP', 'minAfterP',
                  'outpath')
    mock_os_path_exists.assert_called_with('outpath')
    mock_ge.assert_called_with(**{"eventws": "eventws",
                                  "minmag": "minmag",
                                  "minlat": "minlat",
                                  "maxlat": "maxlat",
                                  "minlon": "minlon",
                                  "maxlon": "maxlon",
                                  "start": "start",
                                  "end": "end",
                                  "outpath": "outpath"})
    ev = mock_ge.return_value[0]
    st = mock_gs.return_value[0]
    mock_gs.assert_called_with(dcs.values()[0], channels[0], ev[1], ev[2], ev[3], 'distFromEvent')
    mock_ltd.assert_called_with(ev[2], ev[3], st[2], st[3])
    mock_gat.assert_called_with(mock_ltd.return_value, ev[4])
    origTime = ev[1] + timedelta(seconds=float(mock_gat.return_value))
    mock_gw.assert_called_with(dcs.values()[0], st[1], channels[0], origTime, 'minBeforeP',
                               'minAfterP')
    mock_os_path_join.assert_called_with('outpath', 'ev-%s-%s-%s.mseed' % (ev[0], st[1], mock_gw.return_value[0]))
    mock_open.assert_called_with(mock_os_path_join.return_value, 'wb')
