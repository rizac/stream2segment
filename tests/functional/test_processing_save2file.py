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
from stream2segment.process.main import run as process_main_run, \
    get_advanced_settings as o_get_advanced_settings, process_segments as o_process_segments,\
    process_segments_mp as o_process_segments_mp, \
    _get_chunksize_defaults as _o_get_chunksize_defaults, query4process
from stream2segment.utils.log import configlog4processing as o_configlog4processing
from stream2segment.utils.resources import get_templates_fpath
from stream2segment.process.writers import BaseWriter


@pytest.fixture
def yamlfile(pytestdir):
    '''global fixture wrapping pytestdir.yamlfile'''
    def func(**overridden_pars):
        return pytestdir.yamlfile(get_templates_fpath('save2fs.yaml'), **overridden_pars)

    return func


class Test(object):

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
    @pytest.mark.parametrize("advanced_settings, cmdline_opts",
                             [({}, []),
                              ({'segments_chunk': 1}, []),
                              ({'segments_chunk': 1}, ['--multi-process']),
                              ({}, ['--multi-process']),
                              ({'segments_chunk': 1}, ['--multi-process', '--num-processes', '1']),
                              ({}, ['--multi-process', '--num-processes', '1'])])
    @mock.patch('stream2segment.main.run_process', side_effect=process_main_run)
    def test_simple_run_no_outfile_provided(self, mock_run, advanced_settings,
                                            cmdline_opts,
                                            # fixtures:
                                            pytestdir, db4process, clirunner, yamlfile):
        '''test a case where save inventory is True, and that we saved inventories
        db is a fixture implemented in conftest.py and setup here in self.transact fixture
        '''
        # set values which will override the yaml config in templates folder:
        dir_ = pytestdir.makedir()
        config_overrides = {'snr_threshold': 0,
                            'segment_select': {'has_data': 'true'},
                            'root_dir': os.path.abspath(dir_)}
        if advanced_settings:
            config_overrides['advanced_settings'] = advanced_settings

        yaml_file = yamlfile(**config_overrides)
        # query data for testing now as the program will expunge all data from the session
        # and thus we want to avoid DetachedInstanceError(s):
        expected_only_written_segment = \
            db4process.segments(with_inventory=True, with_data=True, with_gap=False).one()
        # get seiscomp path of OK segment before the session is closed:
        path = os.path.join(dir_, expected_only_written_segment.sds_path())

        pyfile = get_templates_fpath("save2fs.py")

        result = clirunner.invoke(cli, ['process', '--dburl', db4process.dburl,
                                        '-p', pyfile, '-c', yaml_file] + cmdline_opts)
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
        assert args[2] is None or \
            type(args[2]) == BaseWriter  # pylint: disable=unidiomatic-typecheck
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

    # Recall: we have 6 segments, issued from all combination of
    # station_inventory in [true, false] and segment.data in [ok, with_gaps, empty]
    # use db4process(with_inventory, with_data, with_gap) to return sqlalchemy query for
    # those segments in case. For info see db4process in conftest.py
    @pytest.mark.parametrize("def_chunksize",
                             [None, 2])
    @pytest.mark.parametrize("advanced_settings, cmdline_opts",
                             [({}, []),
                              ({'segments_chunk': 1}, []),
                              ({'segments_chunk': 1}, ['--multi-process']),
                              ({}, ['--multi-process']),
                              ({'segments_chunk': 1}, ['--multi-process', '--num-processes', '1']),
                              ({}, ['--multi-process', '--num-processes', '1'])])
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
                                                           mock_mp_Pool,
                                                           advanced_settings,
                                                           cmdline_opts, def_chunksize,
                                                           # fixtures:
                                                           pytestdir, db4process, clirunner,
                                                           yamlfile):
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
        config_overrides = {'snr_threshold': 0,
                            'segment_select': {},  # take everything
                            'root_dir': os.path.abspath(dir_)}
        if advanced_settings:
            config_overrides['advanced_settings'] = advanced_settings

        yaml_file = yamlfile(**config_overrides)

        # need to reset this global variable: FIXME: better handling?
        # process.main._inventories = {}

        pyfile = get_templates_fpath("save2fs.py")

        result = clirunner.invoke(cli, ['process', '--dburl', db4process.dburl,
                                        '-p', pyfile, '-c', yaml_file] + cmdline_opts)
        assert clirunner.ok(result)

        # test some stuff and get configarg, the the REAL config passed in the processing
        # subroutines:
        assert mock_get_advanced_settings.called
        # assert there is no "skipped without messages" message, as it should be the case
        # when there is no function processing the output:
        assert "skipped without messages" not in result.output.lower()
        assert len(mock_get_advanced_settings.call_args_list) == 1
        configarg = mock_get_advanced_settings.call_args_list[0][0][0]  # positional argument

        seg_processed_count = query4process(db4process.session,
                                            configarg.get('segment_select', {})).count()
        # seg_process_count is 6. 'segments_chunk' in advanced_settings is not given or 1.
        # def_chunksize can be None (i,e., 1200) or given (2)
        # See stream2segment.process.core._get_chunksize_defaults to see how we calculated
        # the expected calls to mock_process_segments*:
        if 'segments_chunk' in advanced_settings:
            expected_callcount = seg_processed_count
        elif def_chunksize is None:
            expected_callcount = seg_processed_count
        else:
            _1 = seg_processed_count/def_chunksize
            if _1 == int(_1):
                expected_callcount = int(_1)
            else:
                expected_callcount = int(_1) + 1

        # assert we called the functions the specified amount of times
        if '--multi-process' in cmdline_opts and not advanced_settings:
            # remember that when we have advanced_settings it OVERRIDES
            # the original advanced_settings key in config, thus also multi-process flag
            assert mock_process_segments_mp.called
            assert mock_process_segments_mp.call_count == expected_callcount
            # process_segments_mp calls process_segments:
            assert mock_process_segments_mp.call_count == mock_process_segments.call_count
        else:
            assert mock_process_segments_mp.called == ('--multi-process' in cmdline_opts)
            assert mock_process_segments.called
            assert mock_process_segments.call_count == expected_callcount
        # test that advanced settings where correctly written:
        real_advanced_settings = configarg.get('advanced_settings', {})
        assert ('segments_chunk' in real_advanced_settings) == \
            ('segments_chunk' in advanced_settings)

        # 'advanced_settings', if present HERE, will REPLACE 'advanced_settings' in config. Thus:
        if advanced_settings and '--multi-process' not in cmdline_opts:
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
                # assert type(native(real_advanced_settings['num_processes'])) in integer_types
                assert isinstance(real_advanced_settings['num_processes'], integer_types)
