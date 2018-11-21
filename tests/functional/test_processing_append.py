'''
Created on Feb 14, 2017

@author: riccardo
'''
from __future__ import print_function, division

from builtins import str, object

from past.utils import old_div
import re
import os
import sys
from datetime import datetime, timedelta
import mock
from mock import patch
import csv

import pytest
import pandas as pd
from pandas.util.testing import assert_frame_equal
import numpy as np
from sqlalchemy.engine import create_engine
from sqlalchemy.orm.session import sessionmaker
from obspy.core.stream import read

from stream2segment.cli import cli
from stream2segment.io.db.models import Base, Event, Station, WebService, Segment,\
    Channel, Download, DataCenter
from stream2segment.utils.inputargs import yaml_load as orig_yaml_load
from stream2segment.utils.resources import get_templates_fpaths
from stream2segment.process.utils import get_inventory_url, save_inventory as original_saveinv
from stream2segment.process.main import run as process_main_run, query4process
from stream2segment.utils.log import configlog4processing as o_configlog4processing
from stream2segment.process.utils import enhancesegmentclass
from stream2segment.utils.url import URLError
from stream2segment.process.writers import BaseWriter


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


def readcsv(filename, header=True):
    return pd.read_csv(filename, header=None) if not header else pd.read_csv(filename)


class Test(object):

    pyfile, conffile = get_templates_fpaths("paramtable.py", "paramtable.yaml")

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

        with patch('stream2segment.process.utils.urlread', side_effect=url_read) as mock1:
            self.mock_url_read = mock1
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

    @property
    def logfilecontent(self):
        assert os.path.isfile(self._logfilename)
        with open(self._logfilename) as opn:
            return opn.read()

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
    @mock.patch('stream2segment.utils.inputargs.yaml_load')
    @mock.patch('stream2segment.main.run_process', side_effect=process_main_run)
    def test_simple_run_no_outfile_provided(self, mock_run, mock_yaml_load,
                                            # fixtures:
                                            db, clirunner):
        '''test a case where save inventory is True, and that we saved inventories'''
        # set values which will override the yaml config in templates folder:
        config_overrides = {'save_inventory': True,
                            'snr_threshold': 0,
                            'segment_select': {'has_data': 'true'}}
        mock_yaml_load.side_effect = yaml_load_side_effect(**config_overrides)

        # query data for testing now as the program will expunge all data from the session
        # and thus we want to avoid DetachedInstanceError(s):
        expected_first_row_seg_id = str(self.seg1.id)
        station_id_whose_inventory_is_saved = self.sta_ok.id

        pyfile, conffile = self.pyfile, self.conffile
        result = clirunner.invoke(cli, ['process', '--dburl', db.dburl,
                                        '-p', pyfile, '-c', conffile, '-a'])
        assert clirunner.ok(result)

        lst = mock_run.call_args_list
        assert len(lst) == 1
        args, kwargs = lst[0][0], lst[0][1]
        # assert third argument (`ondone` callback) is None 'ondone' or is a BaseWriter (no-op)
        # class:
        assert args[2] is None or type(args[2]) == BaseWriter
        # assert "Output file:  n/a" in result output:
        assert re.search('Output file:\\s+n/a', result.output)
        # assert "Output file:  n/a" in result output:
        assert re.search('Ignoring `append` functionality: output file does not exist '
                         'or not provided', result.output)

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
                             [({}, ['-a']),
                              ])
    @mock.patch('stream2segment.utils.inputargs.yaml_load')
    def test_simple_run_retDict_saveinv_emptyfile(self, mock_yaml_load, advanced_settings,
                                                  cmdline_opts,
                                                  # fixtures:
                                                  pytestdir, db, clirunner):
        '''test a case where we create a temporary file, empty but opened before writing'''
        # set values which will override the yaml config in templates folder:
        config_overrides = {'save_inventory': True,
                            'snr_threshold': 0,
                            'segment_select': {'has_data': 'true'}}
        if advanced_settings:
            config_overrides['advanced_settings'] = advanced_settings
        mock_yaml_load.side_effect = yaml_load_side_effect(**config_overrides)

        # query data for testing now as the program will expunge all data from the session
        # and thus we want to avoid DetachedInstanceError(s):
        expected_first_row_seg_id = str(self.seg1.id)
        station_id_whose_inventory_is_saved = self.sta_ok.id

        filename = pytestdir.newfile('output.csv', create=True)
        pyfile, conffile = self.pyfile, self.conffile
        result = clirunner.invoke(cli, ['process', '--dburl', db.dburl,
                               '-p', pyfile, '-c', conffile, filename] + cmdline_opts)

        assert clirunner.ok(result)

        # check file has been correctly written:
        csv1 = readcsv(filename)
        assert len(csv1) == 1
        assert str(csv1.loc[0, csv1.columns[0]]) == expected_first_row_seg_id
        logtext = self.logfilecontent
        assert len(logtext) > 0
        assert "Appending results to existing file." in logtext

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
    @pytest.mark.parametrize('return_list', [True, False])
    @pytest.mark.parametrize("advanced_settings, cmdline_opts",
                             [({}, ['-a']),
                              ({}, ['-a', '--multi-process']),
                              ])
    @mock.patch('stream2segment.utils.inputargs.yaml_load')
    @mock.patch('stream2segment.cli.click.confirm', return_value=True)
    def test_append(self, mock_click_confirm, mock_yaml_load, advanced_settings, cmdline_opts,
                    return_list,
                    # fixtures:
                    pytestdir, db, clirunner):
        '''test a typical case where we supply the append option'''
        # set values which will override the yaml config in templates folder:
        config_overrides = {'save_inventory': True,
                            'snr_threshold': 0,
                            'segment_select': {'has_data': 'true'}}
        if advanced_settings:
            config_overrides['advanced_settings'] = advanced_settings
        mock_yaml_load.side_effect = yaml_load_side_effect(**config_overrides)

        # query data for testing now as the program will expunge all data from the session
        # and thus we want to avoid DetachedInstanceError(s):
        expected_first_row_seg_id = str(self.seg1.id)
        station_id_whose_inventory_is_saved = self.sta_ok.id

        filename = pytestdir.newfile('.csv')
        pyfile, conffile = self.pyfile, self.conffile

        if return_list:
            # modify python so taht 'main' returns a list by calling the default 'main'
            # and returning its keys:
            with open(pyfile, 'r') as opn:
                content = opn.read()

            pyfile = pytestdir.newfile('.py')
            cont2 = content.replace("def main(segment, config):", """def main(segment, config):
    return list(main2(segment, config).values())
def main2(segment, config):""")
            with open(pyfile, 'wb') as _opn:
                _opn.write(cont2.encode('utf8'))

        mock_click_confirm.reset_mock()
        result = clirunner.invoke(cli, ['process', '--dburl', db.dburl,
                                        '-p', pyfile, '-c', conffile, filename] + cmdline_opts)
        assert clirunner.ok(result)

        # check file has been correctly written:
        csv1 = readcsv(filename, header=not return_list)
        assert len(csv1) == 1
        assert str(csv1.loc[0, csv1.columns[0]]) == expected_first_row_seg_id
        logtext1 = self.logfilecontent
        assert "3 segment(s) found to process" in logtext1
        assert "Skipping 1 already processed segment(s)" not in logtext1
        assert "Ignoring `append` functionality: output file does not exist or not provided" \
            in logtext1
        assert "1 of 3 segment(s) successfully processed" in logtext1
        assert not mock_click_confirm.called

        # now test a second call, the same as before:
        mock_click_confirm.reset_mock()
        result = clirunner.invoke(cli, ['process', '--dburl', db.dburl,
                                        '-p', pyfile, '-c', conffile, filename] + cmdline_opts)
        # check file has been correctly written:
        # check file has been correctly written:
        csv2 = readcsv(filename, header=not return_list)
        assert len(csv2) == 1
        assert str(csv2.loc[0, csv1.columns[0]]) == expected_first_row_seg_id
        logtext2 = self.logfilecontent
        assert "2 segment(s) found to process" in logtext2
        assert "Skipping 1 already processed segment(s)" in logtext2
        assert "Appending results to existing file." in logtext2
        assert "0 of 2 segment(s) successfully processed" in logtext2
        assert not mock_click_confirm.called
        # assert two rows are equal:
        assert_frame_equal(csv1, csv2, check_dtype=True)

        # change the segment id of the written segment
        seg = db.session.query(Segment).filter(Segment.id == expected_first_row_seg_id).\
            first()
        new_seg_id = seg.id * 100
        seg.id = new_seg_id
        db.session.commit()

        # now test a second call, the same as before:
        mock_click_confirm.reset_mock()
        result = clirunner.invoke(cli, ['process', '--dburl', db.dburl, '-p', pyfile,
                                        '-c', conffile, filename] + cmdline_opts)
        # check file has been correctly written:
        csv3 = readcsv(filename, header=not return_list)
        assert len(csv3) == 2
        assert str(csv3.loc[0, csv1.columns[0]]) == expected_first_row_seg_id
        assert csv3.loc[1, csv1.columns[0]] == new_seg_id
        logtext3 = self.logfilecontent
        assert "3 segment(s) found to process" in logtext3
        assert "Skipping 1 already processed segment(s)" in logtext3
        assert "Appending results to existing file." in logtext3
        assert "1 of 3 segment(s) successfully processed" in logtext3
        assert not mock_click_confirm.called
        # assert two rows are equal:
        assert_frame_equal(csv1, csv3[:1], check_dtype=True)

        # last try: no append (also set no-prompt to test that we did not prompt the user)
        mock_click_confirm.reset_mock()
        result = clirunner.invoke(cli, ['process', '--dburl', db.dburl, '-p', pyfile,
                                        '-c', conffile, filename] + cmdline_opts[1:])
        # check file has been correctly written:
        csv4 = readcsv(filename, header=not return_list)
        assert len(csv4) == 1
        assert csv4.loc[0, csv1.columns[0]] == new_seg_id
        logtext4 = self.logfilecontent
        assert "3 segment(s) found to process" in logtext4
        assert "Skipping 1 already processed segment(s)" not in logtext4
        assert "Appending results to existing file." not in logtext4
        assert "1 of 3 segment(s) successfully processed" in logtext4
        assert 'Overwriting existing output file' in logtext4
        assert mock_click_confirm.called

        # last try: prompt return False
        mock_click_confirm.reset_mock()
        mock_click_confirm.return_value = False
        result = clirunner.invoke(cli, ['process',  '--dburl', db.dburl, '-p', pyfile,
                                        '-c', conffile, filename] + cmdline_opts[1:])
        assert result.exception
        assert type(result.exception) == SystemExit
        assert result.exception.code == 1
