"""
Created on Feb 14, 2017

@author: riccardo
"""
from __future__ import print_function, division

import os
from builtins import object  # pylint: disable=redefined-builtin

from unittest.mock import patch
import pytest

from stream2segment.io.inputvalidation import BadParam
from stream2segment.process.main import imap
from stream2segment.process import SkipSegment


class patches(object):
    # paths container for class-level patchers used below. Hopefully
    # will mek easier debug when refactoring/move functions
    get_session = 'stream2segment.process.main.get_session'
    close_session = 'stream2segment.process.main.close_session'
    configlog4processing = 'stream2segment.process.main.configlog4processing'

class Test(object):

    # The class-level `init` fixture is marked with autouse=true which implies that all test
    # methods in the class will use this fixture without a need to state it in the test
    # function signature or with a class-level usefixtures decorator. For info see:
    # https://docs.pytest.org/en/latest/fixture.html#autouse-fixtures-xunit-setup-on-steroids
    @pytest.fixture(autouse=True)
    def init(self, request, pytestdir, db4process):
        db4process.create(to_file=True)
        session = db4process.session

        # sets up the mocked functions: db session handling (using the already created
        # session) and log file handling:
        with patch(patches.get_session, return_value=session):
            with patch(patches.close_session, side_effect=lambda *a, **v: None):
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
    def test_imap(self, advanced_settings,
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

        for res in imap(func, db4process.dburl, seg_sel, 'abc'):
            # assert res == id
            assert res > 3
            pass

        def func(segment, config):
            assert cfg is config
            raise SkipSegment('a-6')

        count = 0
        for res in imap(func, db4process.dburl, seg_sel, cfg):
            # assert res == ret
            # assert id > 3
            count += 1
            pass
        assert count == 0

        def func(segment, config):
            raise ValueError('a-6')

        count = 0
        with pytest.raises(ValueError):
            for res in imap(func, db4process.dburl, seg_sel, cfg):
                count += 1
                pass
        assert count == 0


        def func(segment, config):
            raise ValueError('a-6')

        count = 0
        for res in imap(func, db4process.dburl, seg_sel, cfg,
                              skip_exceptions=[ValueError]):
            count += 1
            pass
        assert count == 0


def test_imap_wrong_sqlite(# fixtures:
                           pytestdir):
    '''test a non existing sqlite checking that we have the right error
    '''

    # first test, provide a fake dburl that does not exist:
    with pytest.raises(BadParam) as bparam:
        fname = pytestdir.newfile('.sqlite', create=False)
        assert not os.path.isfile(fname)  # for safety
        dburl = 'sqlite:///' + fname
        assert not os.path.isfile(dburl[10:])
        for res in imap(lambda *a, **v: 0, dburl, {}, 'abc'):
            pass

    assert 'dburl' in str(bparam.value)
