'''
Created on Feb 4, 2016

@author: riccardo
'''
# from event2waveform import get_waveforms
# from utils import date
# assert sys.path[0] == os.path.realpath(myPath + '/../../')
from mock import patch
import pytest
from mock import Mock
from datetime import datetime, timedelta
from stream2segment.query_utils import get_time_range, get_stations, get_events,\
    get_travel_time, get_search_radius, url_read, save_waveforms
from stream2segment.utils import datetime as dtime
from StringIO import StringIO
import stream2segment


@patch('stream2segment.query_utils.TauPyModel')
def test_get_travel_times(mock_taup):
    
    
    a = get_travel_time('d', 'q', 'g')
    assert a is None
    mock_taup.assert_called_once_with('g')
    
    model = 'ak135'
    
    from obspy.taup.tau import TauPyModel
    realtpm = TauPyModel(model)
    abc = Mock()
    abc.get_travel_times.side_effect = lambda *args, **kw: realtpm.get_travel_times(*args, **kw) 
    # gettt.side_effect = abc
    mock_taup.return_value = abc

#     with pytest.raises(IOError):
#         _ = get_arrival_time('d', 'q', 'g')
    mock_taup.reset_mock()
    tt = get_travel_time(distance_in_degree=52.474, source_depth_in_km=611.0, model=model)
    # check for the value (account for round errors):
    assert tt > 497.525385547 and tt < 497.525385548
    mock_taup.assert_called_once_with(model)
    abc.get_travel_times.assert_called_once_with(611.0, 52.474)


    mock_taup.reset_mock()
#     abc.get_travel_times.reset_mock()
    abc.get_travel_times.side_effect = lambda *args, **kw: []
    a = get_travel_time(distance_in_degree=52.2, source_depth_in_km=611.0, model=model)
    mock_taup.assert_called_once_with(model)
    abc.get_travel_times.assert_called_once_with(611.0, 52.2)
    assert a is None


@pytest.mark.parametrize('mag, args, expected_val',
                         [(5, None, 3), (2, None, 1), (-1, None, 1), (7, None, 5), (8, None, 5),
                          (5, [3, 7, 1, 5], 3), (2, [3, 7, 1, 5], 1), (-1, [3, 7, 1, 5], 1),
                          (7, [3, 7, 1, 5], 5), (8, [3, 7, 1, 5], 5)])
def test_get_search_radius(mag, args, expected_val):
    if args is None:
        assert get_search_radius(mag) == expected_val
    else:
        args.insert(0, mag)
        assert get_search_radius(*args) == expected_val

@pytest.mark.parametrize('inargs, expected_dt',
                         [
                           ((56,True,True), 56),
                           ((56,False,True), 56),
                           ((56,True,False), 56),
                           ((56,False,False), 56),
                           (('56',True,True), '56'),
                           (('56',False,True), '56'),
                           (('56',True,False), '56'),
                           (('56',False,False), '56'),
                           (('a sd ',True,True), 'aTsdT'),
                           (('a sd ',False,True), 'aTsdT'),
                           (('a sd ',True,False), 'a sd '),
                           (('a sd ',False,False), 'a sd '),
                           (('a sd Z',True,True), 'aTsdT'),
                           (('a sd Z',False,True), 'aTsdTZ'),
                           (('a sd Z',True,False), 'a sd '),
                           (('a sd Z',False,False), 'a sd Z'),
                           (('2015-01-03 22:22:22Z',True,True), '2015-01-03T22:22:22'),
                           (('2015-01-03 22:22:22Z',False,True), '2015-01-03T22:22:22Z'),
                           (('2015-01-03 22:22:22Z',True,False), '2015-01-03 22:22:22'),
                           (('2015-01-03 22:22:22Z',False,False), '2015-01-03 22:22:22Z'),
                           ]
                         )
def test_prepare_datestr(inargs, expected_dt):
    from stream2segment.utils import prepare_datestr
    assert prepare_datestr(*inargs) == expected_dt

        
@pytest.mark.parametrize('prepare_datestr_return_value, strptime_callcount, expected_dt',
                         [
                          (56, 1, TypeError()),
                          ('abc', 3, ValueError()),
                          ("2006", 3, ValueError()),
                          ("2006-06", 3, ValueError()),
                          ("2006-06-06", 2, datetime(2006, 6, 6)),
                          ("2006-06-06T", 3, ValueError()),
                          ("2006-06-06T03", 3, ValueError()),
                          ("2006-06-06T03:22", 3, ValueError()),
                          ("2006-06-06T03:22:12", 1, datetime(2006,6,6, 3,22,12)),
                          ("2006-06-06T03:22:12.45", 3, datetime(2006,6,6, 3,22,12,450000)),
                          ]
                         )
# for side effect below
# see https://docs.python.org/3/library/unittest.mock-examples.html#partial-mocking
@patch('stream2segment.utils._datetime_strptime', side_effect = lambda *args, **kw: datetime.strptime(*args, **kw))
# @patch('stream2segment.utils.dt.datetime', spec=datetime, side_effect=lambda *args, **kw: datetime(*args, **kw))
@patch('stream2segment.utils.prepare_datestr')
def test_to_datetime_crap(mock_prepare_datestr, mock_strptime, prepare_datestr_return_value,
                          strptime_callcount, expected_dt):

    mock_prepare_datestr.return_value = prepare_datestr_return_value

    inarg = "x"
    if isinstance(expected_dt, BaseException):
        with pytest.raises(expected_dt.__class__):
            dtime(inarg)
        expected_dt = None

    mock_prepare_datestr.reset_mock()
    # mock_datetime.reset_mock()
    mock_strptime.reset_mock()
 
    dt = dtime(inarg, on_err_return_none=True)
    assert dt == expected_dt
    mock_prepare_datestr.assert_called_once_with(inarg, True, True)
    first_args_to_strptime = [c[0][0] for c in mock_strptime.call_args_list]
    assert all(x == prepare_datestr_return_value for x in first_args_to_strptime)
    assert mock_strptime.call_count == strptime_callcount


# @patch('stream2segment.query_utils.ul.Request', return_value='Request')
# @patch('stream2segment.query_utils.ul.urlopen', return_value=Urlopen('read'))
@patch('stream2segment.query_utils.url_read', return_value='url_read')
def test_get_events(mock_url_read):  # , mock_urlopen, mock_request):
    with pytest.raises(KeyError):
        get_events()

    args = {'eventws': 'eventws', 'minmag': 1.1,
            'start': datetime.now().isoformat(),
            'end': datetime.now().isoformat(),
            'minlon': '90', 'maxlon': '80',
            'minlat': '85', 'maxlat': '57'}

    mock_url_read.reset_mock()
    lst = get_events(**args)
    assert lst.empty
    assert mock_url_read.called

    mock_url_read.reset_mock()
    mock_url_read.return_value = 'header\na|b|c'
    lst = get_events(**args)
    assert lst.empty
    assert mock_url_read.called

    # value error:
    mock_url_read.reset_mock()
    mock_url_read.return_value = 'header\na|'+datetime.now().isoformat()+'|c'
    lst = get_events(**args)
    assert lst.empty
    assert mock_url_read.called

    # index error:
    mock_url_read.reset_mock()
    mock_url_read.return_value = 'header\na|'+datetime.now().isoformat()+'|1.1'
    lst = get_events(**args)
    assert lst.empty
    assert mock_url_read.called

    mock_url_read.reset_mock()
    d = datetime.now()
    mock_url_read.return_value = 'header\na|'+d.isoformat()+'|1.1|2|3.0|4.0|a|b|c|d|1.1'
    lst = get_events(**args)
    assert lst.empty
    assert mock_url_read.called

    mock_url_read.reset_mock()
    d = datetime.now()
    mock_url_read.return_value = 'header|a|b|c|d|e|f|g|h|i|j\na|'+d.isoformat()+'|1.1|2|3.0|4.0|a|b|c|d|1.1'
    lst = get_events(**args)
    assert len(lst) == 1
    assert lst.values[0].tolist() == ['a', d, 1.1, 2.0, 3.0, '4.0', 'a', 'b', 'c', 'd', 1.1]
    assert mock_url_read.called

# @patch('stream2segment.query_utils.url_read', return_value='url_read')
# def test_get_waveforms(mock_url_read):
#     mock_url_read.reset_mock()
#     a, b = get_waveforms('a', 'b', 'c', 'd', '3', '5')
#     assert not a and not b
#     assert not mock_url_read.called
# 
#     mock_url_read.reset_mock()
#     a, b = get_waveforms('a', 'b', 'c',  datetime.utcnow(), '3', '5')
#     assert not a and not b
#     assert not mock_url_read.called
# 
#     with patch('stream2segment.query_utils.getTimeRange') as mock_get_tr:
#         mock_url_read.reset_mock()
#         d1 = datetime.now()
#         d2 = d1 + timedelta(seconds=1)
#         mock_get_tr.return_value = d1, d2
#         a, b = get_waveforms('a', 'b', 'c', 'd', '3', '5')
#         assert a == 'c' and b == mock_url_read.return_value
#         assert mock_url_read.called
#         mock_get_tr.assert_called_with('d', minutes=('3','5'))
# 
#         mock_url_read.reset_mock()
#         a, b = get_waveforms('a', 'b', 'c*', 'd', '3', '5')
#         assert a == 'c' and b == mock_url_read.return_value
#         assert mock_url_read.called
#         mock_get_tr.assert_called_with('d', minutes=('3','5'))
# 
#         mock_url_read.reset_mock()
#         a, b = get_waveforms('a', 'b', [], 'd', '3', '5')
#         assert not a and not b
#         assert not mock_url_read.called
#         mock_get_tr.assert_called_with('d', minutes=('3','5'))
# 
#         mock_url_read.reset_mock()
#         mock_get_tr.side_effect = lambda *args, **kw: get_time_range(*args, **kw)
#         a, b = get_waveforms('a', 'b', 'c', 'd', '3', '5')
#         assert not a and not b
#         assert not mock_url_read.called
#         mock_get_tr.assert_called_with('d', minutes=('3','5'))


@patch('stream2segment.query_utils.url_read', return_value='url_read')
def test_get_stations(mock_url_read):
    mock_url_read.reset_mock()
    lst = get_stations('a', 'b', 'c', 'd', '5', '6')
    assert lst.empty
    assert not mock_url_read.called

    mock_url_read.reset_mock()
    with pytest.raises(TypeError):
        lst = get_stations('a', 'b',  datetime.utcnow(), '4', '3', '5')
        # assert not mock_url_read.called

    mock_url_read.reset_mock()
    lst = get_stations('a', 'b',  datetime.utcnow(), 4, 3, 5)
    assert lst.empty
    assert mock_url_read.called

    with patch('stream2segment.query_utils.get_time_range') as mock_get_timerange:
        mock_url_read.reset_mock()
        mock_get_timerange.return_value = (datetime.now(), datetime.now()+timedelta(seconds=1))
        d = datetime.now()
        mock_url_read.return_value = 'header\na|b|c'
        with pytest.raises(TypeError):
            lst = get_stations('dc', ['listCha'], d, 'lat', 'lon', 'dist')
            # mock_get_timerange.assert_called_with(d, 1)
            # assert not mock_url_read.called

        mock_url_read.reset_mock()
        with pytest.raises(IndexError):
            lst = get_stations('dc', ['listCha'], d, 3.1, 2.5, 1.1)
            # mock_get_timerange.assert_called_with(d, 1)
            # assert mock_url_read.called

        mock_url_read.reset_mock()
        mock_url_read.return_value = 'header\na|b|c|d|e|f|g|h'
        lst = get_stations('dc', ['listCha'], d, 3.1, 2.5, 1.1)
        mock_get_timerange.assert_called_with(d, days=1)
        assert mock_url_read.called
        assert lst.empty

        mock_url_read.reset_mock()
        mock_url_read.return_value = 'header\na|b|1|1.1|2.0|f|'+d.isoformat()+'|h'
        lst = get_stations('dc', ['listCha'], d, 3.1, 2.5, 1.1)
        mock_get_timerange.assert_called_with(d, days=1)
        assert mock_url_read.called
        assert len(lst) == 1
        assert lst[0][6] == d
        assert lst[0][7] == None

        mock_url_read.reset_mock()
        d2 = datetime.now()
        mock_url_read.return_value = 'header\na|b|1|1.1|2.0|f|'+d.isoformat()+'|'+d2.isoformat()
        lst = get_stations('dc', ['listCha'], d, 3.1, 2.5, 1.1)
        mock_get_timerange.assert_called_with(d, days=1)
        assert mock_url_read.called
        assert len(lst) == 1
        assert lst[0][6] == d
        assert lst[0][7] == d2


@patch('stream2segment.query_utils.timedelta', side_effect=lambda *args, **kw: timedelta(*args, **kw))
def test_get_timerange(mock_timedelta):
    mock_timedelta.reset_mock()
    d = datetime.utcnow()
    d1, d2 = get_time_range(d, days=1)
    assert d-d1 == d2-d == timedelta(days=1)

    mock_timedelta.reset_mock()
    d = datetime.utcnow()
    d1, d2 = get_time_range(d, days=1, minutes=(1, 2))
    assert d-d1 == timedelta(days=1, minutes=1)
    assert d2-d == timedelta(days=1, minutes=2)

    mock_timedelta.reset_mock()
    d = datetime.utcnow()
    _, _ = get_time_range(d, days=1)
    assert mock_timedelta.called

    mock_timedelta.reset_mock()
    _, _ = get_time_range(d)
    assert mock_timedelta.called

    mock_timedelta.reset_mock()
    _, _ = get_time_range(d, days=(1, 2))
    assert mock_timedelta.called

    mock_timedelta.reset_mock()
    _, _ = get_time_range(d, days=(1, 2), minutes=1)
    assert mock_timedelta.called

    mock_timedelta.reset_mock()
    with pytest.raises(Exception):
        _, _ = get_time_range(d, days="abc", minutes=1)
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

@patch('stream2segment.utils.Request')
@patch('stream2segment.utils.urlopen')
def test_url_read(mock_urlopen, mock_urllib_request):  # mock_ul_urlopen, mock_ul_request, mock_ul):
    blockSize = 1024*1024

    mock_urllib_request.side_effect = lambda arx: arx

    def xyz(argss):
        return StringIO(argss)

    # mock_ul.urlopen = Mock()
    mock_urlopen.side_effect = xyz
    # mock_ul.urlopen.return_value = lambda arg: StringIO(arg)

    val = 'url'
    assert url_read(val, "name") == val
    mock_urllib_request.assert_called_with(val)
    mock_urlopen.assert_called_with(val)
    # mock_ul.urlopen.read.assert_called_with(blockSize)

    def ioerr(**kwargs):
        ret = IOError()
        for key, value in kwargs.iteritems():
            setattr(ret, key, value)
        return ret

    for kwz in [{'reason':'reason'}, {'code': 'code'}, {}]:
        def xyz2(**kw):
            raise ioerr(**kw)

        mock_urlopen.side_effect = lambda arg: xyz2(**kwz)
        assert url_read(val, "name") == ''
        mock_urllib_request.assert_called_with(val)
        mock_urlopen.assert_called_with(val)
        assert not mock_urlopen.read.called

    def xyz3():
        raise ValueError()
    mock_urlopen.side_effect = lambda arg: xyz3()
    assert url_read(val, "name") == ''
    mock_urllib_request.assert_called_with(val)
    mock_urlopen.assert_called_with(val)
    assert not mock_urlopen.read.called

    def xyz4():
        raise AttributeError()
    mock_urlopen.side_effect = lambda arg: xyz4()
    with pytest.raises(AttributeError):
        _ = url_read(val, "name")

    def xyz5(argss):
        class sio(StringIO):
            def read(self, *args, **kw):
                raise IOError('oops')
        return sio(argss)
    mock_urlopen.side_effect = lambda arg: xyz5(arg)
    assert url_read(val, "name") == ''
    mock_urllib_request.assert_called_with(val)
    mock_urlopen.assert_called_with(val)
    # mock_ul.urlopen.read.assert_called_with(blockSize)

#     def excp():
#         raise IOError('oops')
#     mock_ul.urlopen.read.side_effect = excp
#     assert url_read(val, "name") == ''
    
# @patch('stream2segment.query_utils.ul.urlopen')
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


# @patch('stream2segment.query_utils.locations2degrees', return_value = 'l2d')
# @patch('stream2segment.query_utils.get_arrival_time')
# @patch('stream2segment.query_utils.get_events')
# @patch('stream2segment.query_utils.get_stations')
# @patch('stream2segment.query_utils.get_waveforms')
# @patch('stream2segment.query_utils.os.path.exists', return_value=False)
# def test_save_waveforms_nopath(mock_os_path_exists, mock_gw, mock_gs, mock_ge, mock_gat, mock_ltd):
#     mock_os_path_exists.side_effect = lambda arg: False
#     save_waveforms('eventws', 'minmag', 'minlat', 'maxlat', 'minlon', 'maxlon', 
#                   'distFromEvent', 'datacenters_dict',
#                   'channelList', 'start', 'end', ('minBeforeP', 'minAfterP'), 'outpath')
#     mock_os_path_exists.assert_called_with('outpath')
#     assert not mock_ge.called and not mock_gs.called and not mock_gw.called and \
#         not mock_gat.called and not mock_ltd.called
# 
# 
# @patch('stream2segment.query_utils.locations2degrees', return_value = 'l2d')
# @patch('stream2segment.query_utils.get_arrival_time')
# @patch('stream2segment.query_utils.get_events')
# @patch('stream2segment.query_utils.get_stations')
# @patch('stream2segment.query_utils.get_waveforms')
# @patch('stream2segment.query_utils.os.path.exists', return_value=True)
# def test_save_waveforms_get_events_returns_empty(mock_os_path_exists, mock_gw, mock_gs, mock_ge, mock_gat, mock_ltd):
# 
#     mock_ge.side_effect = lambda **args: []
#     save_waveforms('eventws', 'minmag', 'minlat', 'maxlat', 'minlon', 'maxlon', 
#                   'distFromEvent', 'datacenters_dict',
#                   'channelList', 'start', 'end', ('minBeforeP', 'minAfterP'), 'outpath')
#     mock_os_path_exists.assert_called_with('outpath')
#     mock_ge.assert_called_with(**{"eventws": "eventws",
#                                   "minmag": "minmag",
#                                   "minlat": "minlat",
#                                   "maxlat": "maxlat",
#                                   "minlon": "minlon",
#                                   "maxlon": "maxlon",
#                                   "start": "start",
#                                   "end": "end",
#                                   "outpath": "outpath"})
#     assert not mock_gs.called and not mock_gw.called and not mock_gat.called and not mock_ltd.called


# # global vars (FIXME: check if good!)
# dcs = {'dc1' : 'www.dc1'}
# channels = {'chan': ['a', 'b' , 'c']}
# search_radius_args = ['1', None, '4' ,'5']
# 
# @patch('stream2segment.query_utils.locations2degrees', return_value='l2d')
# @patch('stream2segment.query_utils.get_arrival_time')
# @patch('stream2segment.query_utils.get_events', return_value=[[str(i) for i in xrange(12)]])
# @patch('stream2segment.query_utils.get_stations')
# @patch('stream2segment.query_utils.get_waveforms')
# @patch('stream2segment.query_utils.os.path.exists', return_value=True)
# def test_save_waveforms_indexerr_on_get_events(mock_os_path_exists, mock_gw, mock_gs, mock_ge,
#                                               mock_gat, mock_ltd):
#     with pytest.raises(IndexError):
#         save_waveforms('eventws', 'minmag', 'minlat', 'maxlat', 'minlon', 'maxlon',
#                       search_radius_args, dcs,
#                       channels, 'start', 'end', ('minBeforeP', 'minAfterP'), 'outpath')
# 
# 
# @patch('stream2segment.query_utils.locations2degrees', return_value='l2d')
# @patch('stream2segment.query_utils.get_arrival_time')
# @patch('stream2segment.query_utils.get_events', return_value=[[str(i) for i in xrange(13)]])
# @patch('stream2segment.query_utils.get_stations', return_value=[])
# @patch('stream2segment.query_utils.get_waveforms')
# @patch('stream2segment.query_utils.os.path.exists', return_value=True)
# @patch('stream2segment.query_utils.get_search_radius', return_value='gsr')
# def test_save_waveforms_get_stations_returns_empty(mock_gsr, mock_os_path_exists, mock_gw, mock_gs, mock_ge,
#                                                   mock_gat, mock_ltd):
# 
#     save_waveforms('eventws', 'minmag', 'minlat', 'maxlat', 'minlon', 'maxlon',
#                   search_radius_args, dcs,
#                   channels, 'start', 'end', ('minBeforeP', 'minAfterP'), 'outpath')
#     mock_os_path_exists.assert_called_with('outpath')
#     mock_ge.assert_called_with(**{"eventws": "eventws",
#                                   "minmag": "minmag",
#                                   "minlat": "minlat",
#                                   "maxlat": "maxlat",
#                                   "minlon": "minlon",
#                                   "maxlon": "maxlon",
#                                   "start": "start",
#                                   "end": "end",
#                                   "outpath": "outpath"})
# 
#     ev = mock_ge.return_value[0]
#     mock_gsr.assert_called_once_with(ev[10], search_radius_args[0], search_radius_args[1],
#                                      search_radius_args[2], search_radius_args[3])
#     mock_gs.assert_called_with(dcs.values()[0], channels.values()[0], ev[1], ev[2], ev[3],
#                                mock_gsr.return_value)
#     assert not mock_gw.called
# 
# 
# @patch('stream2segment.query_utils.locations2degrees', return_value='l2d')
# @patch('stream2segment.query_utils.get_arrival_time')
# @patch('stream2segment.query_utils.get_events', return_value=[[str(i) for i in xrange(13)]])
# @patch('stream2segment.query_utils.get_stations', return_value=[[str(i) for i in xrange(3)]])
# @patch('stream2segment.query_utils.get_waveforms')
# @patch('stream2segment.query_utils.os.path.exists', return_value=True)
# def test_save_waveforms_indexerr_on_get_stations(mock_os_path_exists, mock_gw, mock_gs, mock_ge,
#                                                 mock_gat, mock_ltd):
#     with pytest.raises(IndexError):
#         save_waveforms('eventws', 'minmag', 'minlat', 'maxlat', 'minlon', 'maxlon',
#                       'distFromEvent', dcs,
#                       channels, 'start', 'end', ('minBeforeP', 'minAfterP'), 'outpath')
# 
# 
# @patch('stream2segment.query_utils.locations2degrees', return_value='l2d')
# @patch('stream2segment.query_utils.get_arrival_time', return_value=None)
# @patch('stream2segment.query_utils.get_events', return_value=[[str(i) for i in xrange(13)]])
# @patch('stream2segment.query_utils.get_stations', return_value=[[str(i) for i in xrange(4)]])
# @patch('stream2segment.query_utils.get_waveforms')
# @patch('stream2segment.query_utils.os.path.exists', return_value=True)
# @patch('stream2segment.query_utils.get_search_radius', return_value='gsr')
# def test_save_waveforms_get_arrival_time_none(mock_gsr, mock_os_path_exists, mock_gw, mock_gs, mock_ge,
#                                               mock_gat, mock_ltd):
#     save_waveforms('eventws', 'minmag', 'minlat', 'maxlat', 'minlon', 'maxlon',
#                   search_radius_args, dcs,
#                   channels, 'start', 'end', ('minBeforeP', 'minAfterP'), 'outpath')
#     mock_os_path_exists.assert_called_with('outpath')
#     mock_ge.assert_called_with(**{"eventws": "eventws",
#                                   "minmag": "minmag",
#                                   "minlat": "minlat",
#                                   "maxlat": "maxlat",
#                                   "minlon": "minlon",
#                                   "maxlon": "maxlon",
#                                   "start": "start",
#                                   "end": "end",
#                                   "outpath": "outpath"})
#     ev = mock_ge.return_value[0]
#     st = mock_gs.return_value[0]
#     mock_gsr.assert_called_once_with(ev[10], search_radius_args[0], search_radius_args[1],
#                                      search_radius_args[2], search_radius_args[3])
#     mock_gs.assert_called_with(dcs.values()[0], channels.values()[0], ev[1], ev[2], ev[3],
#                                mock_gsr.return_value)
#     mock_ltd.assert_called_with(ev[2], ev[3], st[2], st[3])
#     mock_gat.assert_called_with(ev[4], mock_ltd.return_value)
#     assert not mock_gw.called
# 
# 
# @patch('__builtin__.open')
# @patch('stream2segment.query_utils.locations2degrees', return_value='l2d')
# @patch('stream2segment.query_utils.get_arrival_time', return_value=5)
# @patch('stream2segment.query_utils.get_events', return_value=[[str(i) for i in xrange(13)]])
# @patch('stream2segment.query_utils.get_stations', return_value=[[str(i) for i in xrange(4)]])
# @patch('stream2segment.query_utils.get_waveforms', return_value=('', ''))
# @patch('stream2segment.query_utils.os.path.exists', return_value=True)
# @patch('stream2segment.query_utils.get_search_radius', return_value='gsr')
# @patch('stream2segment.query_utils.os.path.join', return_value='joined')
# def test_save_waveforms_get_arrival_time_no_wav(mock_os_path_join, mock_gsr, mock_os_path_exists,
#                                                 mock_gw, mock_gs, mock_ge, mock_gat, mock_ltd,
#                                                 mock_open):
#     d = datetime.now()
#     evz = mock_ge.return_value
#     evz[0][1] = d
#     mock_ge.return_value = evz
# 
#     save_waveforms('eventws', 'minmag', 'minlat', 'maxlat', 'minlon', 'maxlon',
#                   search_radius_args, dcs, channels, 'start', 'end', ('minBeforeP', 'minAfterP'),
#                   'outpath')
#     mock_os_path_exists.assert_called_with('outpath')
#     mock_ge.assert_called_with(**{"eventws": "eventws",
#                                   "minmag": "minmag",
#                                   "minlat": "minlat",
#                                   "maxlat": "maxlat",
#                                   "minlon": "minlon",
#                                   "maxlon": "maxlon",
#                                   "start": "start",
#                                   "end": "end",
#                                   "outpath": "outpath"})
#     ev = mock_ge.return_value[0]
#     st = mock_gs.return_value[0]
#     mock_gsr.assert_called_once_with(ev[10], search_radius_args[0], search_radius_args[1],
#                                      search_radius_args[2], search_radius_args[3])
#     mock_gs.assert_called_with(dcs.values()[0], channels.values()[0], ev[1], ev[2], ev[3],
#                                mock_gsr.return_value)
#     mock_ltd.assert_called_with(ev[2], ev[3], st[2], st[3])
#     mock_gat.assert_called_with(ev[4], mock_ltd.return_value)
#     origTime = ev[1] + timedelta(seconds=float(mock_gat.return_value))
#     mock_gw.assert_called_with(dcs.values()[0], st[1], channels.values()[0], origTime, 'minBeforeP',
#                                'minAfterP')
#     assert not mock_os_path_join.called
#     assert not mock_open.called
# 
# 
# @patch('__builtin__.open')
# @patch('stream2segment.query_utils.locations2degrees', return_value='l2d')
# @patch('stream2segment.query_utils.get_arrival_time', return_value=5)
# @patch('stream2segment.query_utils.get_events', return_value=[[str(i) for i in xrange(13)]])
# @patch('stream2segment.query_utils.get_stations', return_value=[[str(i) for i in xrange(4)]])
# @patch('stream2segment.query_utils.get_waveforms', return_value=('', 'wav'))
# @patch('stream2segment.query_utils.os.path.exists', return_value=True)
# @patch('stream2segment.query_utils.get_search_radius', return_value='gsr')
# @patch('stream2segment.query_utils.os.path.join', return_value='joined')
# def test_save_waveforms_get_arrival_time(mock_os_path_join, mock_gsr, mock_os_path_exists, mock_gw, mock_gs,
#                                          mock_ge, mock_gat, mock_ltd, mock_open):
#     d = datetime.now()
#     evz = mock_ge.return_value
#     evz[0][1] = d
#     mock_ge.return_value = evz
# 
#     save_waveforms('eventws', 'minmag', 'minlat', 'maxlat', 'minlon', 'maxlon',
#                   search_radius_args, dcs, channels, 'start', 'end', ('minBeforeP', 'minAfterP'),
#                   'outpath')
#     mock_os_path_exists.assert_called_with('outpath')
#     mock_ge.assert_called_with(**{"eventws": "eventws",
#                                   "minmag": "minmag",
#                                   "minlat": "minlat",
#                                   "maxlat": "maxlat",
#                                   "minlon": "minlon",
#                                   "maxlon": "maxlon",
#                                   "start": "start",
#                                   "end": "end",
#                                   "outpath": "outpath"})
#     ev = mock_ge.return_value[0]
#     st = mock_gs.return_value[0]
#     mock_gsr.assert_called_once_with(ev[10], search_radius_args[0], search_radius_args[1],
#                                      search_radius_args[2], search_radius_args[3])
#     mock_gs.assert_called_with(dcs.values()[0], channels.values()[0], ev[1], ev[2], ev[3],
#                                mock_gsr.return_value)
#     mock_ltd.assert_called_with(ev[2], ev[3], st[2], st[3])
#     mock_gat.assert_called_with( ev[4], mock_ltd.return_value)
#     origTime = ev[1] + timedelta(seconds=float(mock_gat.return_value))
#     mock_gw.assert_called_with(dcs.values()[0], st[1], channels.values()[0], origTime, 'minBeforeP',
#                                'minAfterP')
#     mock_os_path_join.assert_called_with('outpath', 'ev-%s-%s-%s.mseed' % (ev[0], st[1], mock_gw.return_value[0]))
#     mock_open.assert_called_with(mock_os_path_join.return_value, 'wb')



# mock_dt.py
# import datetime
# import mock
# 
# real_datetime_class = datetime.datetime
# 
# def mock_datetime_now(target, dt):
#     class DatetimeSubclassMeta(type):
#         @classmethod
#         def __instancecheck__(mcs, obj):
#             return isinstance(obj, real_datetime_class)
# 
#     class BaseMockedDatetime(real_datetime_class):
#         @classmethod
#         def now(cls, tz=None):
#             return target.replace(tzinfo=tz)
# 
#         @classmethod
#         def utcnow(cls):
#             return target
# 
#     # Python2 & Python3 compatible metaclass
#     # Note: type('X', (object,), dict(a=1)) is the same as:
#     # class X(object):
#     #    a = 1
#     MockedDatetime = DatetimeSubclassMeta('datetime', (BaseMockedDatetime,), {})
#     # Note
#     
#     
#     return mock.patch.object(dt, 'datetime', MockedDatetime)