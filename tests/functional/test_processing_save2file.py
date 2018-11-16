'''
Created on Feb 14, 2017

@author: riccardo
'''
from __future__ import print_function, division

from builtins import str, object  # pylint: disable=redefined-builtin

import os
from datetime import datetime, timedelta
import mock
from mock import patch, MagicMock
from future.backports.urllib.error import URLError
import pytest
import numpy as np

from obspy.core.stream import read

from stream2segment.cli import cli
from stream2segment.io.db.models import Base, Event, Station, WebService, Segment,\
    Channel, Download, DataCenter
from stream2segment.utils.inputargs import yaml_load as orig_yaml_load
from stream2segment.process.main import run as process_main_run, \
    get_advanced_settings as o_get_advanced_settings, process_segments as o_process_segments,\
    process_segments_mp as o_process_segments_mp, \
    _get_chunksize_defaults as _o_get_chunksize_defaults, query4process
from stream2segment.utils.log import configlog4processing as o_configlog4processing

from future import standard_library
from stream2segment.utils.resources import get_templates_fpaths
import re
from stream2segment.process.writers import BaseWriter
from future.utils import native, integer_types
standard_library.install_aliases()


def yaml_load_side_effect(**overrides):
    """Side effect for the function reading the yaml config which enables the input
    of parameters to be overridden just after reading and before any other operation"""
    if overrides:
        def func(*a, **v):
            ret = orig_yaml_load(*a, **v)
            ret.update(overrides)  # note: this OVERRIDES nested dicts
            # whereas passing coverrides as second argument of orig_yaml_load MERGES their keys
            # with existing one
            return ret
        return func
    return orig_yaml_load

class Test(object):

    # execute this fixture always even if not provided as argument:
    # https://docs.pytest.org/en/documentation-restructure/how-to/fixture.html#autouse-fixtures-xunit-setup-on-steroids
    @pytest.fixture(autouse=True)
    def init(self, request, db, data, pytestdir):
        # re-init a sqlite database (no-op if the db is not sqlite):
        db.create(to_file=True)

        # init db:
        session = db.session

        # setup a run_id:
        r = Download()
        session.add(r)
        session.commit()
        self.run = r

        ws = WebService(id=1, url='eventws')
        session.add(ws)
        session.commit()
        self.ws = ws
        # setup an event:
        e1 = Event(id=1, webservice_id=ws.id, event_id='abc1', latitude=8, longitude=9, magnitude=5,
                   depth_km=4, time=datetime.utcnow())
        e2 = Event(id=2, webservice_id=ws.id, event_id='abc2', latitude=8, longitude=9, magnitude=5,
                   depth_km=4, time=datetime.utcnow())
        e3 = Event(id=3, webservice_id=ws.id, event_id='abc3', latitude=8, longitude=9, magnitude=5,
                   depth_km=4, time=datetime.utcnow())
        e4 = Event(id=4, webservice_id=ws.id, event_id='abc4', latitude=8, longitude=9, magnitude=5,
                   depth_km=4, time=datetime.utcnow())
        e5 = Event(id=5, webservice_id=ws.id, event_id='abc5', latitude=8, longitude=9, magnitude=5,
                   depth_km=4, time=datetime.utcnow())
        session.add_all([e1, e2, e3, e4, e5])
        session.commit()
        self.evt1, self.evt2, self.evt3, self.evt4, self.evt5 = e1, e2, e3, e4, e5

        d = DataCenter(station_url='asd', dataselect_url='sdft')
        session.add(d)
        session.commit()
        self.dc = d

        # s_ok stations have lat and lon > 11, other stations do not
        s_ok = Station(datacenter_id=d.id, latitude=11, longitude=12, network='ok', station='ok',
                       start_time=datetime.utcnow())
        session.add(s_ok)
        session.commit()
        self.sta_ok = s_ok

        s_err = Station(datacenter_id=d.id, latitude=-21, longitude=5, network='err', station='err',
                        start_time=datetime.utcnow())
        session.add(s_err)
        session.commit()
        self.sta_err = s_err

        s_none = Station(datacenter_id=d.id, latitude=-31, longitude=-32, network='none',
                         station='none', start_time=datetime.utcnow())
        session.add(s_none)
        session.commit()
        self.sta_none = s_none

        c_ok = Channel(station_id=s_ok.id, location='ok', channel="ok", sample_rate=56.7)
        session.add(c_ok)
        session.commit()
        self.cha_ok = c_ok

        c_err = Channel(station_id=s_err.id, location='err', channel="err", sample_rate=56.7)
        session.add(c_err)
        session.commit()
        self.cha_err = c_err

        c_none = Channel(station_id=s_none.id, location='none', channel="none", sample_rate=56.7)
        session.add(c_none)
        session.commit()
        self.cha_none = c_none

        atts = data.to_segment_dict('trace_GE.APE.mseed')

        # build three segments with data:
        # "normal" segment
        sg1 = Segment(channel_id=c_ok.id, datacenter_id=d.id, event_id=e1.id, download_id=r.id,
                      event_distance_deg=35, **atts)

        # this segment should have inventory returning an exception (see url_read above)
        sg2 = Segment(channel_id=c_err.id, datacenter_id=d.id, event_id=e2.id, download_id=r.id,
                      event_distance_deg=45, **atts)
        # segment with gaps
        atts = data.to_segment_dict('IA.BAKI..BHZ.D.2016.004.head')
        sg3 = Segment(channel_id=c_ok.id, datacenter_id=d.id, event_id=e3.id, download_id=r.id,
                      event_distance_deg=55, **atts)

        # build two segments without data:
        # empty segment
        atts['data'] = b''
        atts['request_start'] += timedelta(seconds=1)  # avoid unique constraint
        sg4 = Segment(channel_id=c_none.id, datacenter_id=d.id, event_id=e4.id, download_id=r.id,
                      event_distance_deg=45, **atts)

        # null segment
        atts['data'] = None
        atts['request_start'] += timedelta(seconds=2)  # avoid unique constraint
        sg5 = Segment(channel_id=c_none.id, datacenter_id=d.id, event_id=e5.id, download_id=r.id,
                      event_distance_deg=45, **atts)

        session.add_all([sg1, sg2, sg3, sg4, sg5])
        session.commit()
        self.seg1 = sg1
        self.seg2 = sg2
        self.seg_gaps = sg2
        self.seg_empty = sg3
        self.seg_none = sg4


        # mock get inventory:
        def url_read(*a, **v):
            '''mock urlread for inventories. Checks in the url (first arg if there is the 'err',
            'ok' or none' substring and returns appropriated data'''
            if "=err" in a[0]:
                raise URLError('error')
            elif "=none" in a[0]:
                return None, 500, 'Server error'
            else:
                return data.read("inventory_GE.APE.xml"), 200, 'Ok'

        with patch('stream2segment.process.utils.urlread', side_effect=url_read):
            with patch('stream2segment.utils.inputargs.get_session', return_value=session):
                with patch('stream2segment.main.closesession',
                           side_effect=lambda *a, **v: None):
                    with patch('stream2segment.main.configlog4processing') as mock2:

                        def clogd(logger, logfilebasepath, verbose):
                            # config logger as usual, but redirects to a temp file
                            # that will be deleted by pytest, instead of polluting the program
                            # package:
                            ret = o_configlog4processing(logger,
                                                         pytestdir.newfile('.log') \
                                                         if logfilebasepath else None,
                                                         verbose)

                            self._logfilename = ret[0].baseFilename
                            return ret

                        mock2.side_effect = clogd

                        yield

    # ## ======== ACTUAL TESTS: ================================

    # Recall: we have 5 segments:
    # 2 are empty, out of the remaining three:
    # 1 has errors if its inventory is queried to the db. Out of the other two:
    # 1 has gaps
    # 1 has no gaps
    # Thus we have several levels of selection possible
    # as by default withdata is True in segment_select, then we process only the last three
    #
    # Here a simple test for a processing file returning dict. Save inventory and check it's saved
    @pytest.mark.parametrize("advanced_settings, cmdline_opts",
                             [({}, []),
                              ({'segments_chunk': 1}, []),
                              ({'segments_chunk': 1}, ['--multi-process']),
                              ({}, ['--multi-process']),
                              ({'segments_chunk': 1}, ['--multi-process', '--num-processes', '1']),
                              ({}, ['--multi-process', '--num-processes', '1'])])
    @mock.patch('stream2segment.utils.inputargs.yaml_load')
    @mock.patch('stream2segment.main.run_process', side_effect=process_main_run)
    def test_simple_run_no_outfile_provided(self, mock_run, mock_yaml_load, advanced_settings,
                                            cmdline_opts,
                                            # fixtures:
                                            pytestdir, db, clirunner):
        '''test a case where save inventory is True, and that we saved inventories
        db is a fixture implemented in conftest.py and setup here in self.transact fixture
        '''
        # set values which will override the yaml config in templates folder:
        dir_ = pytestdir.makedir()
        config_overrides = {'save_inventory': True,
                            'snr_threshold': 0,
                            'segment_select': {'has_data': 'true'},
                            'root_dir': os.path.abspath(dir_)}
        if advanced_settings:
            config_overrides['advanced_settings'] = advanced_settings

        mock_yaml_load.side_effect = yaml_load_side_effect(**config_overrides)
        # get seiscomp path of OK segment before the session is closed:
        path = os.path.join(dir_, self.seg1.sds_path())
        # query data for testing now as the program will expunge all data from the session
        # and thus we want to avoid DetachedInstanceError(s):
        expected_first_row_seg_id = str(self.seg1.id)
        station_id_whose_inventory_is_saved = self.sta_ok.id

        pyfile, conffile = get_templates_fpaths("save2fs.py", "save2fs.yaml")

        result = clirunner.invoke(cli, ['process', '--dburl', db.dburl,
                                        '-p', pyfile, '-c', conffile] + cmdline_opts)
        assert clirunner.ok(result)

        filez = os.listdir(os.path.dirname(path))
        assert len(filez) == 2
        stream1 = read(os.path.join(os.path.dirname(path), filez[0]), format='MSEED')
        stream2 = read(os.path.join(os.path.dirname(path), filez[1]), format='MSEED')
        assert len(stream1) == len(stream2) == 1
        assert not np.allclose(stream1[0].data, stream2[0].data)

        lst = mock_run.call_args_list
        assert len(lst) == 1
        args, kwargs = lst[0][0], lst[0][1]
        # assert third argument (`ondone` callback) is None 'ondone' or is a BaseWriter (no-op)
        # class:
        assert args[2] is None or type(args[2]) == BaseWriter
        # assert "Output file:  n/a" in result output:
        assert re.search('Output file:\\s+n/a', result.output)

        # Note that apparently CliRunner() (see clirunner fixture) puts stderr and stdout
        # together (https://github.com/pallets/click/pull/868)
        # Reminder: previously, log erros where redirected to stderr
        # This is dangerous as we use a redirect to avoid external libraries to pritn to stderr
        # and logging to stderr might cause 'operation on closed file'.
        for subs in ["Processing function: ", "Config. file: "]:
            idx = result.output.find(subs)
            assert idx > -1

        # these assertion are just copied from the test below and left here cause they
        # should still hold (db behaviour does not change of we provide output file or not):

        # save_downloaded_inventory True, test that we did save any:
        assert len(db.session.query(Station).filter(Station.has_inventory).all()) > 0

        # Or alternatively:
        # test we did save any inventory:
        stas = db.session.query(Station).all()
        assert any(s.inventory_xml for s in stas)
        assert db.session.query(Station).\
            filter(Station.id == station_id_whose_inventory_is_saved).first().inventory_xml


    # Recall: we have 5 segments:
    # 2 are empty, out of the remaining three:
    # 1 has errors if its inventory is queried to the db. Out of the other two:
    # 1 has gaps
    # 1 has no gaps
    # Thus we have several levels of selection possible
    # as by default withdata is True in segment_select, then we process only the last three
    #
    # Here a simple test for a processing file returning dict. Save inventory and check it's saved
    @pytest.mark.parametrize("def_chunksize",
                             [None, 2])
    @pytest.mark.parametrize("advanced_settings, cmdline_opts",
                             [({}, []),
                              ({'segments_chunk': 1}, []),
                              ({'segments_chunk': 1}, ['--multi-process']),
                              ({}, ['--multi-process']),
                              ({'segments_chunk': 1}, ['--multi-process', '--num-processes', '1']),
                              ({}, ['--multi-process', '--num-processes', '1'])])
    @mock.patch('stream2segment.utils.inputargs.yaml_load')
    @mock.patch('stream2segment.process.main.Pool')
    @mock.patch('stream2segment.process.main.get_advanced_settings',
                side_effect=o_get_advanced_settings)
    @mock.patch('stream2segment.process.main.process_segments', side_effect=o_process_segments)
    @mock.patch('stream2segment.process.main.process_segments_mp',
                side_effect=o_process_segments_mp)
    @mock.patch('stream2segment.process.main._get_chunksize_defaults')
    def test_simple_run_no_outfile_provided_good_argslists(self, mock_get_chunksize_defaults,
                                                           mock_process_segments_mp,
                                                           mock_process_segments,
                                                           mock_get_advanced_settings,
                                                           mock_mp_Pool, mock_yaml_load,
                                                           advanced_settings,
                                                           cmdline_opts, def_chunksize,
                                                           # fixtures:
                                                           pytestdir, db, clirunner):
        '''test arguments and calls are ok. Mock Pool imap_unordered as we do not
        want to confuse pytest in case
        '''

        if def_chunksize is None:
            mock_get_chunksize_defaults.side_effect = _o_get_chunksize_defaults
        else:
            mock_get_chunksize_defaults.side_effect = \
                lambda *a, **v: (def_chunksize, _o_get_chunksize_defaults()[1])

        class MockPool(object):
            def __init__(self, *a, **kw):
                pass

            def imap_unordered(self, *a, **kw):
                return map(*a, **kw)

            def close(self, *a, **kw):
                pass

            def join(self, *a, **kw):
                pass

        mock_mp_Pool.return_value = MockPool()

        # set values which will override the yaml config in templates folder:
        dir_ = pytestdir.makedir()
        config_overrides = {'save_inventory': True,
                            'snr_threshold': 0,
                            'segment_select': {},  # take everything
                            'root_dir': os.path.abspath(dir_)}
        if advanced_settings:
            config_overrides['advanced_settings'] = advanced_settings

        mock_yaml_load.side_effect = yaml_load_side_effect(**config_overrides)

        # need to reset this global variable: FIXME: better handling?
        # process.main._inventories = {}

        pyfile, conffile = get_templates_fpaths("save2fs.py", "save2fs.yaml")

        result = clirunner.invoke(cli, ['process', '--dburl', db.dburl,
                                        '-p', pyfile, '-c', conffile] + cmdline_opts)
        assert clirunner.ok(result)

        # test some stuff and get configarg, the the REAL config passed in the processing
        # subroutines:
        assert mock_get_advanced_settings.called
        assert len(mock_get_advanced_settings.call_args_list) == 1
        configarg = mock_get_advanced_settings.call_args_list[0][0][0]  # positional argument

        seg_processed_count = query4process(db.session,
                                            configarg.get('segment_select', {})).count()
        # seg_process_count is 5. advanced_settings is not given or 1.
        # def_chunksize can be None (i,e., 1200) or given (2)
        # See stream2segment.process.core._get_chunksize_defaults to see how we calculated
        # the expected calls to mock_process_segments*:
        expected_callcount = (seg_processed_count if 'segments_chunk' in advanced_settings
                              else seg_processed_count if def_chunksize is None else
                              2)

        # assert we called the functions the specified amount of times
        if '--multi-process' in cmdline_opts and not advanced_settings:
            # remember that when we have advanced_settings it OVERRIDES
            # the original advanced_settings key in config, thus also multi-process flag
            assert mock_process_segments_mp.called
            assert mock_process_segments_mp.call_count == expected_callcount
            # process_segments_mp calls process_segments:
            assert mock_process_segments_mp.call_count == mock_process_segments.call_count
        else:
            assert not mock_process_segments_mp.called
            assert mock_process_segments.called
            assert mock_process_segments.call_count == expected_callcount
        # test that advanced settings where correctly written:
        real_advanced_settings = configarg.get('advanced_settings', {})
        assert ('segments_chunk' in real_advanced_settings) == \
            ('segments_chunk' in advanced_settings)
        # 'advanced_settings', if present HERE, will REPLACE 'advanced_settings' in config
        #  See module function 'yaml_load_side_effect'. THus:
        if advanced_settings:
            assert sorted(real_advanced_settings.keys()) == sorted(advanced_settings.keys())
            for k in advanced_settings.keys():
                assert advanced_settings[k] == real_advanced_settings[k]
        else:
            if 'segments_chunk' in advanced_settings:
                assert real_advanced_settings['segments_chunk'] == \
                    advanced_settings['segments_chunk']
            assert ('multi_process' in real_advanced_settings) == \
                ('--multi-process' in cmdline_opts)
            if '--multi-process' in cmdline_opts:
                assert real_advanced_settings['multi_process'] is True
            assert ('num_processes' in real_advanced_settings) == \
                ('--num-processes' in cmdline_opts)
            if '--num-processes' in cmdline_opts:
                val = cmdline_opts[cmdline_opts.index('--num-processes')+1]
                assert str(real_advanced_settings['num_processes']) == val
                # assert real_advanced_settings['num_processes'] is an int.
                # As we import int from futures in templates, we might end-up having
                # futures.newint. The type check is made by checking we have an integer
                # type as the native type. For info see:
                # http://python-future.org/what_else.html#passing-data-to-from-python-2-libraries
                assert type(native(real_advanced_settings['num_processes'])) in integer_types
