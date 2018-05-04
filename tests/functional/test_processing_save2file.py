'''
Created on Feb 14, 2017

@author: riccardo
'''
from __future__ import print_function, division

from builtins import str, object

from tempfile import NamedTemporaryFile
from past.utils import old_div
import unittest
import os, sys
from datetime import datetime, timedelta
import mock
from mock import patch
import tempfile
import csv
from future.backports.urllib.error import URLError
import pytest
import numpy as np
from click.testing import CliRunner
from sqlalchemy.engine import create_engine
from sqlalchemy.orm.session import sessionmaker
# from urllib.error import URLError
# import multiprocessing
from obspy.core.stream import read

from stream2segment.cli import cli
from stream2segment.io.db.models import Base, Event, Station, WebService, Segment,\
    Channel, Download, DataCenter
from stream2segment.utils.inputargs import yaml_load as load_proc_cfg
from stream2segment import process
from stream2segment.utils.resources import get_templates_fpaths
from stream2segment.process.utils import get_inventory_url, save_inventory as original_saveinv

from stream2segment.process.core import run as process_core_run
from future import standard_library
from stream2segment.process.utils import enhancesegmentclass
import click
from itertools import product
standard_library.install_aliases()


class Test(object):

    # execute this fixture always even if not provided as argument:
    # https://docs.pytest.org/en/documentation-restructure/how-to/fixture.html#autouse-fixtures-xunit-setup-on-steroids
    @pytest.fixture(autouse=True)
    def transact(self, request, db):
        # re-init a sqlite database (no-op if the db is not sqlite):
        db.reinit_to_file()

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

        data = self.read_stream_raw('trace_GE.APE.mseed')

        # build three segments with data:
        # "normal" segment
        sg1 = Segment(channel_id=c_ok.id, datacenter_id=d.id, event_id=e1.id, download_id=r.id,
                      event_distance_deg=35, **data)

        # this segment should have inventory returning an exception (see url_read above)
        sg2 = Segment(channel_id=c_err.id, datacenter_id=d.id, event_id=e2.id, download_id=r.id,
                      event_distance_deg=45, **data)
        # segment with gaps
        data = self.read_stream_raw('IA.BAKI..BHZ.D.2016.004.head')
        sg3 = Segment(channel_id=c_ok.id, datacenter_id=d.id, event_id=e3.id, download_id=r.id,
                      event_distance_deg=55, **data)

        # build two segments without data:
        # empty segment
        data['data'] = b''
        data['request_start'] += timedelta(seconds=1)  # avoid unique constraint
        sg4 = Segment(channel_id=c_none.id, datacenter_id=d.id, event_id=e4.id, download_id=r.id,
                      event_distance_deg=45, **data)

        # null segment
        data['data'] = None
        data['request_start'] += timedelta(seconds=2)  # avoid unique constraint
        sg5 = Segment(channel_id=c_none.id, datacenter_id=d.id, event_id=e5.id, download_id=r.id,
                      event_distance_deg=45, **data)

        session.add_all([sg1, sg2, sg3, sg4, sg5])
        session.commit()
        self.seg1 = sg1
        self.seg2 = sg2
        self.seg_gaps = sg2
        self.seg_empty = sg3
        self.seg_none = sg4


        # values to override the config, if specified:
        self.config_overrides = {}
        self.inventory = True

        # init patchers:
        self.patchers = []
        # mock get inventory:
        self.patchers.append(patch('stream2segment.process.utils.urlread'))
        self.mock_url_read = self.patchers[-1].start()
        self.mock_url_read.side_effect = self.url_read

        self.patchers.append(patch('stream2segment.utils.inputargs.get_session'))
        self.mock_session = self.patchers[-1].start()
        self.mock_session.return_value = session

        self.patchers.append(patch('stream2segment.main.closesession'))
        self.mock_closing = self.patchers[-1].start()
        self.mock_closing.side_effect = lambda *a, **v: None

        yield

        # cleanup patchers:
        for patcher in self.patchers:
            if patcher:
                patcher.stop()
     

    @staticmethod
    def read_stream_raw(file_name):
        '''returns a dict to be passed as argument for creating new Segment(s), by reading
        an existing miniseed'''
        stream = read(Test.get_file(file_name))

        start_time = stream[0].stats.starttime
        end_time = stream[0].stats.endtime

        # set arrival time to one third duration
        return dict(data=Test.read_data_raw(file_name),
                    arrival_time=(start_time + old_div((end_time - start_time), 3)).datetime,
                    request_start=start_time.datetime,
                    request_end=end_time.datetime,
                    sample_rate=stream[0].stats.sampling_rate)

    @staticmethod
    def read_data_raw(file_name):
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
        with open(os.path.join(folder, file_name), 'rb') as opn:
            return opn.read()

    @staticmethod
    def read_and_remove(filepath):
        assert os.path.isfile(filepath)
        with open(filepath) as opn:
            sss = opn.read()
        os.remove(filepath)
        return sss

    @staticmethod
    def get_file(filename):
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
        path = os.path.abspath(os.path.join(folder, filename))
        assert os.path.isfile(path)
        return path

    @staticmethod
    def get_processing_files():
        pyfile, conffile = get_templates_fpaths("save2fs.py", "save2fs.yaml") #pylint: disable=unbalanced-tuple-unpacking
        return pyfile, conffile

    @staticmethod
    def url_read(*a, **v):
        '''mock urlread for inventories. Checks in the url (first arg if there is the 'err', 'ok'
        or none' substring and returns appropriated data'''
        if "=err" in a[0]:
            raise URLError('error')
        elif "=none" in a[0]:
            return None, 500, 'Server error'
        else:
            return Test.read_data_raw("inventory_GE.APE.xml"), 200, 'Ok'

    def load_proc_cfg(self, *a, **kw):
        """called by mocked read config: updates the parsed dict with the custom config"""
        cfg = load_proc_cfg(*a, **kw)
        cfg.update(self.config_overrides)
        return cfg

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
    @pytest.mark.parametrize("segments_chunksize", [1, None])
    @mock.patch('stream2segment.utils.inputargs.yaml_load')
    @mock.patch('stream2segment.process.main.process_core_run', side_effect=process_core_run)
    def test_simple_run_no_outfile_provided(self, mock_run, mock_load_cfg, segments_chunksize, db):
        '''test a case where save inventory is True, and that we saved inventories
        db is a fixture implemented in conftest.py and setup here in self.transact fixture
        '''
        # set values which will override the yaml config in templates folder:
        runner = CliRunner()
        with runner.isolated_filesystem() as dir_:
            self.config_overrides = {'save_inventory': True,
                                     'snr_threshold': 0,
                                     'segment_select': {'has_data': 'true'},
                                     'root_dir': os.path.abspath(dir_)}
            if segments_chunksize:
                self.config_overrides['advanced_setting'] = {'segments_chunk': segments_chunksize}
        
            mock_load_cfg.side_effect = self.load_proc_cfg
    
            # get seiscomp path of OK segment before the session is closed:
            path = os.path.join(dir_, self.seg1.seiscomp_path())
            # query data for testing now as the program will expunge all data from the session
            # and thus we want to avoid DetachedInstanceError(s):
            expected_first_row_seg_id = str(self.seg1.id)
            station_id_whose_inventory_is_saved = self.sta_ok.id
    
            # need to reset this global variable: FIXME: better handling?
            process.main._inventories = {}
            
            
            pyfile, conffile = self.get_processing_files()
            
            result = runner.invoke(cli, ['process', '--dburl', db.dburl,
                                   '-p', pyfile, '-c', conffile])
    
            if result.exception:
                import traceback
                traceback.print_exception(*result.exc_info)
                print(result.output)
                assert False
                return
            
            filez = os.listdir(os.path.dirname(path))
            assert len(filez) == 2
            stream1 = read(os.path.join(os.path.dirname(path), filez[0]), format='MSEED')
            stream2 = read(os.path.join(os.path.dirname(path), filez[1]), format='MSEED')
            assert len(stream1) == len(stream2) == 1
            assert not np.allclose(stream1[0].data, stream2[0].data)
    
            
    
        lst = mock_run.call_args_list
        assert len(lst) == 1
        args, kwargs = lst[0][0], lst[0][1]
        assert args[2] is None  # assert third argument (`ondone` callback) is None 'ondone' 
        assert "Output file:" not in result.output
    
        # Note that apparently CliRunner() puts stderr and stdout together 
        # (https://github.com/pallets/click/pull/868)
        # So we should test that we have these string twice:
        for subs in ["Executing 'main' in ", "Config. file: "]:
            idx = result.output.find(subs)
            assert idx > -1
            assert result.output.find(subs, idx+1) > idx
        
        
        
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


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
