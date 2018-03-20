'''
Created on Feb 14, 2017

@author: riccardo
'''
from __future__ import print_function, division

from builtins import str, object

# from tempfile import NamedTemporaryFile
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
from stream2segment.process.core import query4process

from stream2segment.process.core import run as process_core_run
from future import standard_library
from stream2segment.process.utils import enhancesegmentclass
from stream2segment.download.utils import custom_download_codes
import json
import re
standard_library.install_aliases()


class DB(object):
    def __init__(self):
        self.dburi = os.getenv("DB_URL", "sqlite:///:memory:")
        # an Engine, which the Session will use for connection
        # resources
        # some_engine = create_engine('postgresql://scott:tiger@localhost/')
        self.engine = create_engine(self.dburi)
        # Base.metadata.drop_all(cls.engine)
        Base.metadata.create_all(self.engine)  # @UndefinedVariable
        # create a configured "Session" class

    def create(self):
        Session = sessionmaker(bind=self.engine)
        # create a Session
        self.session = Session()

        # setup a run_id:
        r = Download()
        self.session.add(r)
        self.session.commit()
        self.run = r

        ws = WebService(id=1, url='eventws')
        self.session.add(ws)
        self.session.commit()
        self.ws = ws
        # setup an event:
        e1 = Event(id=1, webservice_id=ws.id, event_id='ev1', latitude=8, longitude=9, magnitude=5,
                   depth_km=4, time=datetime.utcnow())
        e2 = Event(id=2, webservice_id=ws.id, event_id='ev2', latitude=8, longitude=9, magnitude=5,
                   depth_km=4, time=datetime.utcnow())
        e3 = Event(id=3, webservice_id=ws.id, event_id='ev3', latitude=8, longitude=9, magnitude=5,
                   depth_km=4, time=datetime.utcnow())
        e4 = Event(id=4, webservice_id=ws.id, event_id='ev4', latitude=8, longitude=9, magnitude=5,
                   depth_km=4, time=datetime.utcnow())
        e5 = Event(id=5, webservice_id=ws.id, event_id='ev5', latitude=8, longitude=9, magnitude=5,
                   depth_km=4, time=datetime.utcnow())
        self.session.add_all([e1, e2, e3, e4, e5])
        self.session.commit()

        d1 = DataCenter(station_url='asd', dataselect_url='www.dc1/dataselect/query')
        d2 = DataCenter(station_url='asd', dataselect_url='www.dc2/dataselect/query')
        self.session.add_all([d1, d2])
        self.session.commit()

        # d1 has one station
        s_d1 = Station(datacenter_id=d1.id, latitude=11, longitude=11, network='N1', station='S1',
                       start_time=datetime.utcnow())
        s_d2 = Station(datacenter_id=d1.id, latitude=22.1, longitude=22.1, network='N1', station='S2a',
                       start_time=datetime.utcnow())
        s2_d2 = Station(datacenter_id=d1.id, latitude=22.2, longitude=22.2, network='N2', station='S2b',
                       start_time=datetime.utcnow())
        self.session.add_all([s_d1, s_d2, s2_d2])
        self.session.commit()

        url_err, mseed_err, timespan_err, timespan_warn = custom_download_codes()
        # we are about to add 3 stations * 4 channels = 12 channels
        # we add also 1 segment pre channel
        # the segments data is as follows (data, download_code, maxgap)
        seg_data = ([None, url_err, None],
                    [None, mseed_err, None],
                    [None, None, None],
                    [None, timespan_err, None],
                    # station s_d2:
                    [b'x', 200, 0.2],
                    [b'x', timespan_warn, 3.9],
                    [b'x', 200, 0.6],
                    [b'x', 200, 0.3],
                    # station s_d3:
                    [b'x', 200, 0.1],
                    [b'x', timespan_warn, 3.9],
                    [b'x', 400, None],
                    [b'x', 500, None],
                    )

        i = 0
        for s in [s_d1, s_d2, s2_d2]:
            for cha in ['HHZ', 'HHE', 'HHN', 'ABC']:
                c = Channel(station_id=s.id, location='', channel=cha, sample_rate=56.7)
                self.session.add(c)
                self.session.commit()
                
                data, code, gap = seg_data[i]
                i += 1
                seg = Segment(channel_id=c.id, datacenter_id=s.datacenter_id,
                       event_id=e1.id, download_id=r.id,
                       event_distance_deg=35, request_start=datetime.utcnow(),
                       arrival_time=datetime.utcnow(),
                       request_end=datetime.utcnow() + timedelta(seconds=5), data=data,
                       download_code=code, maxgap_numsamples=gap)
                self.session.add(seg)
                self.session.commit()

    def close(self):
        if self.engine:
            if self.session:
                try:
                    self.session.rollback()
                    self.session.close()
                except:
                    pass
            try:
                Base.metadata.drop_all(self.engine)  # @UndefinedVariable
            except:
                pass
#        self.session.close()
#         self.patcher1.stop()
#         self.patcher2.stop()


class Test(unittest.TestCase):

    dburi = ""
    file = None

    @staticmethod
    def cleanup(self):
        db = getattr(self, 'db', None)
        if db:
            db.close()

        for patcher in self.patchers:
            if patcher:
                patcher.stop()

    @property
    def is_sqlite(self):
        return str(self.db.engine.url).startswith("sqlite:///")

    @property
    def is_postgres(self):
        return str(self.db.engine.url).startswith("postgresql://")

    def setUp(self):
        # add cleanup (in case tearDown is not called due to exceptions):
        self.addCleanup(Test.cleanup, self)

        # values to override the config, if specified:
        self.config_overrides = {}
        self.inventory = True

        self.db = DB()
        self.db.create()
        self.session = self.db.session
        self.dburi = self.db.dburi

        self.patchers = []

        self.patchers.append(patch('stream2segment.utils.inputargs.get_session'))
        self.mock_session = self.patchers[-1].start()
        self.mock_session.return_value = self.session

#         self.patchers.append(patch('stream2segment.main.closesession'))
#         self.mock_closing = self.patchers[-1].start()
#         self.mock_closing.side_effect = lambda *a, **v: None

# ## ======== ACTUAL TESTS: ================================

    @patch('stream2segment.main.open_in_browser')
    @patch('stream2segment.main.gettempdir')
    def test_simple_dinfo(self, mock_gettempdir, mock_open_in_browser):
        '''test a case where save inventory is True, and that we saved inventories'''
        
        runner = CliRunner()

        # text output, to file
        with tempfile.NamedTemporaryFile() as file:  # @ReservedAssignment
            
            result = runner.invoke(cli, ['utils', 'dinfo', '--dburl', self.dburi,
                                         file.name])

            if result.exception:
                import traceback
                traceback.print_exception(*result.exc_info)
                print(result.output)
                assert False

            content = open(file.name).read()
            assert """           OK        OK         Time                          Internal  Segment       
           Gaps      Partially  Span   MSeed  Url    Bad      Server    Not           
       OK  Overlaps  Saved      Error  Error  Error  Request  Error     Found    TOTAL
-----  --  --------  ---------  -----  -----  -----  -------  --------  -------  -----
        3         1          2      1      1      1        1         1        1     12
TOTAL   3         1          2      1      1      1        1         1        1     12""" in content
            assert result.output.startswith("""Fetching data, please wait (this might take a while depending on the db size and connection)
download info and statistics written to """)
        assert not mock_open_in_browser.called
        assert not mock_gettempdir.called

        # text output, to stdout
        result = runner.invoke(cli, ['utils', 'dinfo', '--dburl', self.dburi])

        if result.exception:
            import traceback
            traceback.print_exception(*result.exc_info)
            print(result.output)
            assert False

            assert """           OK        OK         Time                          Internal  Segment       
           Gaps      Partially  Span   MSeed  Url    Bad      Server    Not           
       OK  Overlaps  Saved      Error  Error  Error  Request  Error     Found    TOTAL
-----  --  --------  ---------  -----  -----  -----  -------  --------  -------  -----
        3         1          2      1      1      1        1         1        1     12
TOTAL   3         1          2      1      1      1        1         1        1     12""" in result.output

        assert not mock_open_in_browser.called
        assert not mock_gettempdir.called

        # Test html output.
        # First, implement function that parses the sta_data content into dict and check stuff:
        def jsonloads(varname, content):
            start = re.search('%s\\s*:\\s*' % varname, content).end()
            end = start
            while content[end] != '[':
                end += 1
            end += 1
            brakets = 1
            while brakets > 0:
                if content[end] == "[":
                    brakets += 1
                elif content[end] == ']':
                    brakets -= 1
                end += 1
            return json.loads(content[start:end])

        # html output, to file
        with tempfile.NamedTemporaryFile() as file:  # @ReservedAssignment
            result = runner.invoke(cli, ['utils', 'dinfo', '--html',  '--dburl', self.dburi,
                                         file.name])

            if result.exception:
                import traceback
                traceback.print_exception(*result.exc_info)
                print(result.output)
                assert False

            content = open(file.name).read()

            sta_data = jsonloads('sta_data', content)
            networks = jsonloads('networks', content)
            assert len(sta_data) == self.session.query(Station.id).count() * 2
            for i in range(0, len(sta_data), 2):
                staname = sta_data[i]
                data = sta_data[i+1]
                #Each sta is: [sta_name, [staid, stalat, stalon, sta_dcid,
                #                        d_id1, [code1, num_seg1 , ..., codeN, num_seg],
                #                        d_id2, [code1, num_seg1 , ..., codeN, num_seg],
                #                       ]
                if staname == 'S1':
                    assert data[0] == 1
                    assert data[1] == 11  # lat
                    assert data[2] == 11  # lon
                    assert data[3] == 1  # dc id
                    assert data[4] == networks.index('N1')
                    assert data[5] == 1  # download id (only 1)
                    # assert the correct segments categories
                    # (list length is the number of categories,
                    # each element is the num of segments for that category,
                    # and the sum of all elements must be 4)
                    assert sorted(data[6][1::2]) == [1, 1, 1, 1]
                elif staname == 'S2a':
                    assert data[0] == 2
                    assert data[1] == 22.1  # lat
                    assert data[2] == 22.1  # lon
                    assert data[3] == 1  # dc id
                    assert data[4] == networks.index('N1')
                    assert data[5] == 1  # download id (only 1)
                    # assert the correct segments categories
                    # (list length is the number of categories,
                    # each element is the num of segments for that category,
                    # and the sum of all elements must be 4)
                    assert sorted(data[6][1::2]) == [1, 1, 2]
                elif staname == 'S2b':
                    assert data[0] == 3
                    assert data[1] == 22.2  # lat
                    assert data[2] == 22.2  # lon
                    assert data[3] == 1  # dc id
                    assert data[4] == networks.index('N2')
                    assert data[5] == 1  # download id (only 1)
                    # assert the correct segments categories
                    # (list length is the number of categories,
                    # each element is the num of segments for that category,
                    # and the sum of all elements must be 4)
                    assert sorted(data[6][1::2]) == [1, 1, 1, 1]
                else:
                    raise Exception('station should not be there: %s' % staname)

            assert result.output.startswith("""Fetching data, please wait (this might take a while depending on the db size and connection)
download info and statistics written to """)
        assert not mock_open_in_browser.called
        assert not mock_gettempdir.called


        # html output, to file, setting maxgap to 0.2, so that S1a' has all three ok segments with gaps
        with tempfile.NamedTemporaryFile() as file:  # @ReservedAssignment
            result = runner.invoke(cli, ['utils', 'dinfo', '-g', '0.15', '--html',  '--dburl', self.dburi,
                                         file.name])

            if result.exception:
                import traceback
                traceback.print_exception(*result.exc_info)
                print(result.output)
                assert False

            content = open(file.name).read()
            # parse the sta_data content into dict and check stuff:
            sta_data = jsonloads('sta_data', content)
            networks = jsonloads('networks', content)
            assert len(sta_data) == self.session.query(Station.id).count() * 2
            for i in range(0, len(sta_data), 2):
                staname = sta_data[i]
                data = sta_data[i+1]
                #Each sta is: [sta_name, [staid, stalat, stalon, sta_dcid,
                #                        d_id1, [code1, num_seg1 , ..., codeN, num_seg],
                #                        d_id2, [code1, num_seg1 , ..., codeN, num_seg],
                #                       ]
                if staname == 'S1':
                    assert data[0] == 1
                    assert data[1] == 11  # lat
                    assert data[2] == 11  # lon
                    assert data[3] == 1  # dc id
                    assert data[4] == networks.index('N1')
                    assert data[5] == 1  # download id (only 1)
                    # assert the correct segments categories
                    # (list length is the number of categories,
                    # each element is the num of segments for that category,
                    # and the sum of all elements must be 4)
                    assert sorted(data[6][1::2]) == [1, 1, 1, 1]
                elif staname == 'S2a':
                    assert data[0] == 2
                    assert data[1] == 22.1  # lat
                    assert data[2] == 22.1  # lon
                    assert data[3] == 1  # dc id
                    assert data[4] == networks.index('N1')
                    assert data[5] == 1  # download id (only 1)
                    # assert the correct segments categories
                    # (list length is the number of categories,
                    # each element is the num of segments for that category,
                    # and the sum of all elements must be 4)
                    assert sorted(data[6][1::2]) == [1, 3]
                elif staname == 'S2b':
                    assert data[0] == 3
                    assert data[1] == 22.2  # lat
                    assert data[2] == 22.2  # lon
                    assert data[3] == 1  # dc id
                    assert data[4] == networks.index('N2')
                    assert data[5] == 1  # download id (only 1)
                    # assert the correct segments categories
                    # (list length is the number of categories,
                    # each element is the num of segments for that category,
                    # and the sum of all elements must be 4)
                    assert sorted(data[6][1::2]) == [1, 1, 1, 1]
                else:
                    raise Exception('station should not be there: %s' % staname)

            assert result.output.startswith("""Fetching data, please wait (this might take a while depending on the db size and connection)
download info and statistics written to """)
        assert not mock_open_in_browser.called
        assert not mock_gettempdir.called


        # html output, to temp file
        tmpdir = tempfile.gettempdir()
        mock_gettempdir.side_effect = lambda *a, **v: tmpdir
        try:
            result = runner.invoke(cli, ['utils', 'dinfo', '--html', '--dburl',  self.dburi])

            if result.exception:
                import traceback
                traceback.print_exception(*result.exc_info)
                print(result.output)
                assert False

                assert """           OK        OK         Time                          Internal  Segment       
               Gaps      Partially  Span   MSeed  Url    Bad      Server    Not           
           OK  Overlaps  Saved      Error  Error  Error  Request  Error     Found    TOTAL
    -----  --  --------  ---------  -----  -----  -----  -------  --------  -------  -----
            3         1          2      1      1      1        1         1        1     12
    TOTAL   3         1          2      1      1      1        1         1        1     12""" in result.output
            assert mock_open_in_browser.called
            assert mock_gettempdir.called
        finally:
            file = os.path.join(tmpdir, 's2s_dinfo.html')
            assert os.path.isfile(file)
            try:
                os.remove(file)
            except:
                pass


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
