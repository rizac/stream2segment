'''
Created on Feb 14, 2017

@author: riccardo
'''
from __future__ import print_function, division

from builtins import str, object  # pylint: disable=redefined-builtin
import os
import re

import mock
from mock import patch
import pytest
import numpy as np
from obspy.core.stream import read
from future.utils import integer_types

from stream2segment.cli import cli
from stream2segment.main import s2smap
from stream2segment.process import SkipSegment
from stream2segment.process.main import run as process_main_run, \
    get_default_chunksize as o_get_default_chunksize, process_segments as o_process_segments,\
    process_segments_mp as o_process_segments_mp, \
    _get_chunksize_defaults as _o_get_chunksize_defaults, query4process
from stream2segment.utils.log import configlog4processing as o_configlog4processing
from stream2segment.resources import get_templates_fpath
from stream2segment.io import yaml_load
from stream2segment.process.writers import BaseWriter

class Test(object):

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
        with patch('stream2segment.utils.inputvalidation.valid_session', return_value=session):
            with patch('stream2segment.main.closesession',
                       side_effect=lambda *a, **v: None):
                # with patch('stream2segment.main.configlog4processing') as mock2:
                #
                #     def clogd(logger, logfilebasepath, verbose):
                #         # config logger as usual, but redirects to a temp file
                #         # that will be deleted by pytest, instead of polluting the program
                #         # package:
                #         ret = o_configlog4processing(logger,
                #                                      pytestdir.newfile('.log') \
                #                                      if logfilebasepath else None,
                #                                      verbose)
                #         if ret:
                #             self._logfilename = ret[0].baseFilename
                #         return ret
                #
                #     mock2.side_effect = clogd
                #
                #     yield
                yield

    # ## ======== ACTUAL TESTS: ================================

    # Recall: we have 6 segments, issued from all combination of
    # station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
    # use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
    # those segments in case. For info see db4process in conftest.py
    @pytest.mark.parametrize("advanced_settings, cmdline_opts",
                             [({}, []),
                              ({'segments_chunksize': 1}, []),
                              ({'segments_chunksize': 1}, ['--multi-process']),
                              ({}, ['--multi-process']),
                              ({'segments_chunksize': 1}, ['--multi-process', '--num-processes', '1']),
                              ({}, ['--multi-process', '--num-processes', '1'])])
    def test_s2smap(self, advanced_settings,
                    cmdline_opts,
                    # fixtures:
                    pytestdir, db4process, capsys):
        '''test the save2file python module, and also test a case when
        no output file is provided
        '''
        # set values which will override the yaml config in templates folder:
        cfg = {'snr_threshold': 0}
        seg_sel = {'has_data': 'true', 'id': '>3'}
        # if advanced_settings:
        #     config_overrides['advanced_settings'] = advanced_settings

        # ret = {'a', 1}
        # cfg = yaml_load(pytestdir.yamlfile(get_templates_fpath('save2fs.yaml'),
        #                                    **config_overrides))
        # with capsys.disabled():
        def func(segment, config):
            assert config == 'abc'
            return segment.id

        for res, id in s2smap(func, db4process.dburl, seg_sel, 'abc'):
            assert res == id
            assert id > 3
            pass

        def func(segment, config):
            assert cfg is config
            raise SkipSegment('a-6')

        count = 0
        for res, id in s2smap(func, db4process.dburl, seg_sel, cfg):
            # assert res == ret
            # assert id > 3
            count += 1
            pass
        assert count == 0

        def func(segment, config):
            raise ValueError('a-6')

        count = 0
        with pytest.raises(ValueError):
            for res, id in s2smap(func, db4process.dburl, seg_sel, cfg):
                count += 1
                pass
        assert count == 0


        def func(segment, config):
            raise ValueError('a-6')

        count = 0
        for res, id in s2smap(func, db4process.dburl, seg_sel, cfg,
                              skip_exceptions=[ValueError]):
            count += 1
            pass
        assert count == 0

