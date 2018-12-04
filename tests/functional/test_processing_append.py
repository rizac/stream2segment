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
from stream2segment.process.main import run as process_main_run, query4process
from stream2segment.utils.log import configlog4processing as o_configlog4processing
from stream2segment.process.utils import enhancesegmentclass
from stream2segment.utils.url import URLError
from stream2segment.process.writers import BaseWriter
from stream2segment.io.utils import dumps_inv


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

    @property
    def logfilecontent(self):
        assert os.path.isfile(self._logfilename)
        with open(self._logfilename) as opn:
            return opn.read()

    # The class-level `init` fixture is marked with autouse=true which implies that all test
    # methods in the class will use this fixture without a need to state it in the test
    # function signature or with a class-level usefixtures decorator. For info see:
    # https://docs.pytest.org/en/latest/fixture.html#autouse-fixtures-xunit-setup-on-steroids
    @pytest.fixture(autouse=True)
    def init(self, request, pytestdir, db4process):
        db4process.create(to_file=True)
        session = db4process.session
        # sets up the mocked functions: db session handling (using the already created session)
        # and log file handling:
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

    # Recall: we have 6 segments, issued from all combination of
    # station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
    # use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
    # those segments in case. For info see db4process in conftest.py
    @mock.patch('stream2segment.utils.inputargs.yaml_load')
    @mock.patch('stream2segment.main.run_process', side_effect=process_main_run)
    def test_simple_run_no_outfile_provided(self, mock_run, mock_yaml_load,
                                            # fixtures:
                                            db4process, clirunner):
        '''test a case where save inventory is True, and that we saved inventories'''
        # set values which will override the yaml config in templates folder:
        config_overrides = {'snr_threshold': 0,
                            'segment_select': {'has_data': 'true'}}
        mock_yaml_load.side_effect = yaml_load_side_effect(**config_overrides)

        pyfile, conffile = self.pyfile, self.conffile
        result = clirunner.invoke(cli, ['process', '--dburl', db4process.dburl,
                                        '-p', pyfile, '-c', conffile, '-a'])
        assert clirunner.ok(result)

        lst = mock_run.call_args_list
        assert len(lst) == 1
        args, kwargs = lst[0][0], lst[0][1]
        # assert third argument (`ondone` callback) is None 'ondone' or is a BaseWriter (no-op)
        # class:
        assert args[2] is None or \
            type(args[2]) == BaseWriter  # pylint: disable=unidiomatic-typecheck
        # assert "Output file:  n/a" in result output:
        assert re.search('Output file:\\s+n/a', result.output)
        # assert "Output file:  n/a" in result output:
        assert re.search('Ignoring `append` functionality: output file does not exist '
                         'or not provided', result.output)

    # Recall: we have 6 segments, issued from all combination of
    # station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
    # use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
    # those segments in case. For info see db4process in conftest.py
    @pytest.mark.parametrize("advanced_settings, cmdline_opts",
                             [({}, ['-a']),
                              ])
    @mock.patch('stream2segment.utils.inputargs.yaml_load')
    def test_simple_run_retDict_saveinv_emptyfile(self, mock_yaml_load, advanced_settings,
                                                  cmdline_opts,
                                                  # fixtures:
                                                  pytestdir, db4process, clirunner):
        '''test a case where we create a temporary file, empty but opened before writing'''
        # set values which will override the yaml config in templates folder:
        config_overrides = {'snr_threshold': 0,
                            'segment_select': {'has_data': 'true'}}
        if advanced_settings:
            config_overrides['advanced_settings'] = advanced_settings
        mock_yaml_load.side_effect = yaml_load_side_effect(**config_overrides)

        _seg = db4process.segments(with_inventory=True, with_data=True, with_gap=False).one()
        expected_first_row_seg_id = _seg.id
        station_id_whose_inventory_is_saved = _seg.station.id

        filename = pytestdir.newfile('output.csv', create=True)
        pyfile, conffile = self.pyfile, self.conffile
        result = clirunner.invoke(cli, ['process', '--dburl', db4process.dburl,
                                  '-p', pyfile, '-c', conffile, filename] + cmdline_opts)
        assert clirunner.ok(result)

        # check file has been correctly written:
        csv1 = readcsv(filename)
        assert len(csv1) == 1
        assert csv1.loc[0, csv1.columns[0]] == expected_first_row_seg_id
        logtext = self.logfilecontent
        assert len(logtext) > 0
        assert "Appending results to existing file." in logtext

    # Recall: we have 6 segments, issued from all combination of
    # station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
    # use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
    # those segments in case. For info see db4process in conftest.py
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
                    pytestdir, db4process, clirunner):
        '''test a typical case where we supply the append option'''
        # set values which will override the yaml config in templates folder:
        config_overrides = {'snr_threshold': 0,
                            'segment_select': {'has_data': 'true'}}
        if advanced_settings:
            config_overrides['advanced_settings'] = advanced_settings
        mock_yaml_load.side_effect = yaml_load_side_effect(**config_overrides)

        _seg = db4process.segments(with_inventory=True, with_data=True, with_gap=False).one()
        expected_first_row_seg_id = _seg.id
        station_id_whose_inventory_is_saved = _seg.station.id

        session = db4process.session

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
        result = clirunner.invoke(cli, ['process', '--dburl', db4process.dburl,
                                        '-p', pyfile, '-c', conffile, filename] + cmdline_opts)
        assert clirunner.ok(result)

        # check file has been correctly written:
        csv1 = readcsv(filename, header=not return_list)
        assert len(csv1) == 1
        assert csv1.loc[0, csv1.columns[0]] == expected_first_row_seg_id
        logtext1 = self.logfilecontent
        assert "4 segment(s) found to process" in logtext1
        assert "Skipping 1 already processed segment(s)" not in logtext1
        assert "Ignoring `append` functionality: output file does not exist or not provided" \
            in logtext1
        assert "1 of 4 segment(s) successfully processed" in logtext1
        assert not mock_click_confirm.called

        # now test a second call, the same as before:
        mock_click_confirm.reset_mock()
        result = clirunner.invoke(cli, ['process', '--dburl', db4process.dburl,
                                        '-p', pyfile, '-c', conffile, filename] + cmdline_opts)
        # check file has been correctly written:
        csv2 = readcsv(filename, header=not return_list)
        assert len(csv2) == 1
        assert csv2.loc[0, csv1.columns[0]] == expected_first_row_seg_id
        logtext2 = self.logfilecontent
        assert "3 segment(s) found to process" in logtext2
        assert "Skipping 1 already processed segment(s)" in logtext2
        assert "Appending results to existing file." in logtext2
        assert "0 of 3 segment(s) successfully processed" in logtext2
        assert not mock_click_confirm.called
        # assert two rows are equal:
        assert_frame_equal(csv1, csv2, check_dtype=True)

        # change the segment id of the written segment
        seg = session.query(Segment).filter(Segment.id == expected_first_row_seg_id).\
            first()
        new_seg_id = seg.id * 100
        seg.id = new_seg_id
        session.commit()

        # now test a second call, the same as before:
        mock_click_confirm.reset_mock()
        result = clirunner.invoke(cli, ['process', '--dburl', db4process.dburl, '-p', pyfile,
                                        '-c', conffile, filename] + cmdline_opts)
        # check file has been correctly written:
        csv3 = readcsv(filename, header=not return_list)
        assert len(csv3) == 2
        assert csv3.loc[0, csv1.columns[0]] == expected_first_row_seg_id
        assert csv3.loc[1, csv1.columns[0]] == new_seg_id
        logtext3 = self.logfilecontent
        assert "4 segment(s) found to process" in logtext3
        assert "Skipping 1 already processed segment(s)" in logtext3
        assert "Appending results to existing file." in logtext3
        assert "1 of 4 segment(s) successfully processed" in logtext3
        assert not mock_click_confirm.called
        # assert two rows are equal:
        assert_frame_equal(csv1, csv3[:1], check_dtype=True)

        # last try: no append (also set no-prompt to test that we did not prompt the user)
        mock_click_confirm.reset_mock()
        result = clirunner.invoke(cli, ['process', '--dburl', db4process.dburl, '-p', pyfile,
                                        '-c', conffile, filename] + cmdline_opts[1:])
        # check file has been correctly written:
        csv4 = readcsv(filename, header=not return_list)
        assert len(csv4) == 1
        assert csv4.loc[0, csv1.columns[0]] == new_seg_id
        logtext4 = self.logfilecontent
        assert "4 segment(s) found to process" in logtext4
        assert "Skipping 1 already processed segment(s)" not in logtext4
        assert "Appending results to existing file." not in logtext4
        assert "1 of 4 segment(s) successfully processed" in logtext4
        assert 'Overwriting existing output file' in logtext4
        assert mock_click_confirm.called

        # last try: prompt return False
        mock_click_confirm.reset_mock()
        mock_click_confirm.return_value = False
        result = clirunner.invoke(cli, ['process',  '--dburl', db4process.dburl, '-p', pyfile,
                                        '-c', conffile, filename] + cmdline_opts[1:])
        assert result.exception
        assert type(result.exception) == SystemExit
        assert result.exception.code == 1
