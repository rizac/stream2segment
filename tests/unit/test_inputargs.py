'''
Created on May 23, 2017

@author: riccardo
'''
from contextlib import contextmanager
import os
from datetime import datetime, timedelta
import shutil
from mock.mock import patch
# this can not apparently be fixed with the future package:
# The problem is io.StringIO accepts unicodes in python2 and strings in python3:
try:
    from cStringIO import StringIO  # python2.x pylint: disable=unused-import
except ImportError:
    from io import StringIO  # @UnusedImport

from future.utils import string_types, PY2
from click.testing import CliRunner
from _pytest.capture import capsys
import pytest
import yaml

from stream2segment.cli import cli
from stream2segment.main import configlog4download as o_configlog4download,\
    new_db_download as o_new_db_download, run_download as o_run_download,\
    configlog4processing as o_configlog4processing, run_process as o_run_process, \
    process as o_process, download as o_download
from stream2segment.utils.inputargs import yaml_load as o_yaml_load,\
    get_session as o_get_session, nslc_param_value_aslist
from stream2segment.io.db.models import Download
from stream2segment.utils import get_session, secure_dburl
from stream2segment.utils.resources import get_templates_fpath, yaml_load, get_templates_fpaths
from stream2segment.main import init as orig_init, helpmathiter as main_helpmathiter, download


class Test(object):

    d_yaml_file, p_yaml_file, p_py_file = \
        get_templates_fpaths("download.yaml", "paramtable.yaml", "paramtable.py")

    # execute this fixture always even if not provided as argument:
    # https://docs.pytest.org/en/documentation-restructure/how-to/fixture.html#autouse-fixtures-xunit-setup-on-steroids
    @pytest.fixture(autouse=True)
    def init(self, request, db, data, pytestdir):
        # re-init a sqlite database (no-op if the db is not sqlite):
        db.create(to_file=False)
        # Although we do not test db stuff here other than checking a download id has written,
        # we iterate through all given db's

        # patchers:
        def cfd_side_effect(logger, logfilebasepath, verbose):
            # config logger as usual, but redirects to a temp file
            # that will be deleted by pytest, instead of polluting the program
            # package:
            return o_configlog4download(logger, pytestdir.newfile('.log'), verbose)

        with patch('stream2segment.main.configlog4download',
                   side_effect=cfd_side_effect) as _:
            self.mock_config4download = _

            with patch('stream2segment.main.new_db_download',
                       side_effect=o_new_db_download) as _:
                self.mock_new_db_download = _

                with patch('stream2segment.utils.inputargs.get_session') as _:
                    def csess(dbpath, *a, **v):
                        if dbpath == db.dburl:
                            return db.session
                        return o_get_session(dbpath, *a, **v)
                    _.side_effect = csess

                    with patch('stream2segment.main.closesession') as _:
                        self.mock_close_session = _
                        self._lastrun_lastdownload_id = None

                        def clsess(*a, **v):  #pylint: disable=unused-argument
                            qry = db.session.query(Download)
                            if self._lastrun_lastdownload_id is not None:
                                qry = qry.filter(Download.id > self._lastrun_lastdownload_id)
                            dwnlds = qry.order_by(Download.id).all()
                            # the following members are re-initialized in run_cli_download:
                            self.lastrun_download_count = len(dwnlds)
                            if self.lastrun_download_count:
                                self.lastrun_lastdownload_config = dwnlds[-1].config
                                self.lastrun_lastdownload_log = dwnlds[-1].log
                                self.lastrun_lastdownload_id = \
                                    self._lastrun_lastdownload_id = dwnlds[-1].id
#                             do not close the session:
#                             db.session.close()
                        self.mock_close_session.side_effect = clsess

                        with patch('stream2segment.main.run_download',
                                   side_effect = lambda *a, **v: None) as _:  # no-op
                            self.mock_run_download = _

                            with patch('stream2segment.main.configlog4processing',
                                       side_effect=o_configlog4processing) as _:
                                self.mock_configlog4processing = _

                                with patch('stream2segment.main.run_process',
                                           side_effect=lambda *a, **v: None) as _:
                                    self.mock_run_process = _

                                    with patch('stream2segment.utils.inputargs.yaml_load') as _:
                                        self.mock_yaml_load = _
                                        self.yaml_overrides = {}
                                        def yload(filepath, **updates):
                                            dic = yaml_load(filepath, **updates)
                                            if isinstance(filepath, string_types) and\
                                                os.path.isfile(filepath):
                                                # if we passed a file name then override
                                                # the template dburl with our one
                                                # If it's not the case, leave as it is:
                                                # in download.yaml, it is a fake address
                                                dic['dburl'] = db.dburl
                                            # provide a valid one so that we explicitly inject
                                            # a bad one in self.yaml_overrides,
                                            # if needed
                                            dic.update(self.yaml_overrides or {})
                                            # IMPORTANT: reset NOW yaml overrides.
                                            # In download, yaml_load is called
                                            # twice, the second time reading from the builtin
                                            # download.yaml to check unknown
                                            # parameters. In this case, the yaml must return
                                            # EXACTLY the file ignoring our
                                            # overrides. NOTE THAT THIS IS A HACK AND FAILS
                                            # IF WE READ FROM THE BUILTIN CONFIG FIRST
                                            self.yaml_overrides = {}
                                            return dic
                                        self.mock_yaml_load.side_effect = yload  # no-op

                                        yield

    def run_cli_download(self, *args, **yaml_overrides):
        # reset database stuff:
        self.lastrun_lastdownload_id = None
        self.lastrun_lastdownload_log = None
        self.lastrun_lastdownload_config = None
        self.lastrun_download_count = 0
        # process inputs:
        args = list(args)
        if all(a not in ("-c", "--configfile") for a in args):
            args += ['-c', self.d_yaml_file]
        self.yaml_overrides = yaml_overrides
        runner = CliRunner()
        result = runner.invoke(cli, ['download'] + args)
        return result

    def run_cli_process(self, *args, **kwargs):
        '''kwargs has a single arg: dburl_in_yaml (None by default)'''
        dburl_in_yaml = kwargs.get('dburl_in_yaml', None)
        args = list(args)
        if all(a not in ("-d", "--dburl") for a in args):
            args += ['-d', self.d_yaml_file]
        if all(a not in ("-p", "--pyfile") for a in args):
            args += ['-p', self.p_py_file]
        if all(a not in ("-c", "--config") for a in args):
            args += ['-c', self.p_yaml_file]
        if dburl_in_yaml is not None:
            self.yaml_overrides = {'dburl': dburl_in_yaml}
        runner = CliRunner()
        result = runner.invoke(cli, ['process'] + args)
        return result

    def test_download_eventws_query_args(self, db):
        '''test different scenarios where we provide eventws query args from the command line'''
        # FIRST SCENARIO: PROVIDE A PARAMETER P in the cli ALREADY PRESENT IN THE CONFIG
        # eventws_query_args
        # Test that the name stays the same provided in the coinfig, regardless of the cli name
        def_yaml_dict = yaml_load(self.d_yaml_file)['eventws_query_args']
        param = 'minmag'
        other_param = 'minmagnitude'
        assert other_param not in def_yaml_dict
        oldval = def_yaml_dict[param]  # which also asserts we do have the value
        newval = oldval + 1.1
        result = self.run_cli_download('-%s' % param, newval)  # invalid type
        assert result.exit_code == 0  # WHAT?? because networks needs to be just an iterable
        # assert new yaml (as saved on the db) has the correct value:
        new_yaml_dict = \
            yaml_load(StringIO(self.lastrun_lastdownload_config))['eventws_query_args']
        assert new_yaml_dict[param] == newval
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.lastrun_download_count == 1
        assert other_param not in new_yaml_dict

        newval += 1.1
        result = self.run_cli_download('--%s' % other_param, newval)  # invalid type
        assert result.exit_code == 0
        new_yaml_dict = \
            yaml_load(StringIO(self.lastrun_lastdownload_config))['eventws_query_args']
        assert new_yaml_dict[param] == newval
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.lastrun_download_count == 1
        assert other_param not in new_yaml_dict

        # SECOND SCENARIO: PROVIDE A PARAMETER P in the cli NOT PRESENT IN THE CONFIG
        # eventws_query_args. Test that the name is the one provided from the cli (long name)
        # regardless of the cli name (short or long)
        param = 'lat'
        other_param = 'latitude'
        assert param not in def_yaml_dict  # if it fails, change param/other_param name above
        assert other_param not in def_yaml_dict  # if it fails, change/other_param param name above
        newval = 1.1
        expected_param = other_param  # because it is the default cli param name
        nonexpected_param = param  # see above
        result = self.run_cli_download('-%s' % param, newval)  # invalid type
        assert result.exit_code == 0  # WHAT?? because networks needs to be just an iterable
        # assert new yaml (as saved on the db) has the correct value:
        new_yaml_dict = \
            yaml_load(StringIO(self.lastrun_lastdownload_config))['eventws_query_args']
        assert new_yaml_dict[expected_param] == newval
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.lastrun_download_count == 1
        assert nonexpected_param not in new_yaml_dict

        newval += 1.1
        result = self.run_cli_download('--%s' % other_param, newval)  # invalid type
        assert result.exit_code == 0  # WHAT?? because networks needs to be just an iterable
        # assert new yaml (as saved on the db) has the correct value:
        new_yaml_dict = \
            yaml_load(StringIO(self.lastrun_lastdownload_config))['eventws_query_args']
        assert new_yaml_dict[expected_param] == newval
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.lastrun_download_count == 1
        assert nonexpected_param not in new_yaml_dict

    def test_download_bad_values(self, db):
        '''test different scenarios where the value in the dwonload.yaml are not well formatted'''
        result = self.run_cli_download(networks={'a': 'b'})  # conflict
        assert result.exit_code != 0
        assert 'Error: Conflicting names "networks" / "network"' in result.output
        result = self.run_cli_download(network={'a': 'b'})
        assert result.exit_code == 0  # WHAT?? because networks needs to be just an iterable
        # thus providing dict is actually fine and will iterate over its keys:
        assert self.mock_run_download.call_args_list[0][1]['networks'] == ['a']
        # do some asserts only for this case to test how we print the arguments to string:
        # assert "tt_table: <TTTable object, " in result.output
        assert "start: 2006-01-01 00:00:00" in result.output
        assert "traveltimes_model:" in result.output
        _dburl = db.dburl
        if not db.is_sqlite:
            _dburl = secure_dburl(_dburl)
        # assert dburl is in result.output (sqlite:memory is quotes, postgres not. we do not
        # care to investigate why, jsut assert either string is there:
        assert "dburl: '%s'" % _dburl in result.output or "dburl: %s" % _dburl in result.output

        # check the session:
        assert self.lastrun_lastdownload_log  # assert we have something written
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.lastrun_download_count == 1

        result = self.run_cli_download(networks='!*')  # conflicting names
        assert result.exit_code != 0
        assert 'Error: Conflicting names "networks" / "network"' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.lastrun_download_count == 0

        result = self.run_cli_download(network='!*')  # invalid value
        assert result.exit_code != 0
        assert 'Error: Invalid value for "network": ' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.lastrun_download_count == 0

        result = self.run_cli_download(net='!*')  # conflicting names
        assert result.exit_code != 0
        assert 'Error: Conflicting names "net" / "network"' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.lastrun_download_count == 0

        # test error from the command line. Result is the same as above as the check is made
        # AFTER click
        result = self.run_cli_download('-n', '!*')  # invalid value
        assert result.exit_code != 0
        assert 'Error: Invalid value for "network": ' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.lastrun_download_count == 0

        # no such option:
        result = self.run_cli_download('--zrt', '!*')
        assert result.exit_code != 0
        assert 'Error: no such option: --zrt' in result.output  # why -z and not -zz? whatever...
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.lastrun_download_count == 0

        # no such option from within the yaml:
        result = self.run_cli_download(zz='!*')
        assert result.exit_code != 0
        assert 'Error: No such option "zz"' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.lastrun_download_count == 0

        # what about conflicting arguments?
        result = self.run_cli_download(networks='!*', net='opu')  # invalid value
        assert result.exit_code != 0
        assert 'Conflicting names "net" / "networks"' in result.output or \
            'Conflicting names "networks" / "net"' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.lastrun_download_count == 0

        result = self.run_cli_download(start=[])  # invalid type
        assert result.exit_code != 0
        assert 'Error: Invalid type for "start":' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.lastrun_download_count == 0

        # mock implementing conflicting names in the yaml file:
        result = self.run_cli_download(starttime='wat')  # invalid value
        assert result.exit_code != 0
        assert 'Error: Conflicting names "starttime" / "start"' in result.output or \
            'Error: Conflicting names "start" / "starttime"' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.lastrun_download_count == 0

        # mock implementing bad value in the cli: (cf with the previous test):
        # THE MESSAGE BELOW IS DIFFERENT BECAUSE WE PROVIDE A CLI VALIDATION FUNCTION
        # See the case of travetimes model below where, without a cli validation function,
        # the message is the same when we provide a bad argument in the yaml or from the cli
        result = self.run_cli_download('--starttime', 'wat')  # invalid value
        assert result.exit_code != 0
        assert 'Error: Invalid value for "-s" / "--start" / "--starttime": wat' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.lastrun_download_count == 0

        # This should work:
        result = self.run_cli_download('--starttime', '2006-03-14')  # invalid value
        assert result.exit_code == 0
        run_download_kwargs = self.mock_run_download.call_args_list[-1][1]
        assert run_download_kwargs['start'] == datetime(2006, 3, 14)
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.lastrun_download_count == 1

        # now test the same as above BUT with a cli-only argument (-t0):
        result = self.run_cli_download('-s', 'wat')  # invalid value typed from the command line
        assert result.exit_code != 0
        assert 'Error: Invalid value for "-s" / "--start" / "--starttime":' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.lastrun_download_count == 0

        result = self.run_cli_download(end='wat') # try with end
        assert result.exit_code != 0
        assert 'Error: Invalid value for "end":' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.lastrun_download_count == 0

        # now test the same as above BUT with the wrong value from the command line:
        result = self.run_cli_download('-e', 'wat')  # invalid value typed from the command line
        assert result.exit_code != 0
        assert 'Error: Invalid value for "-e" / "--end" / "--endtime":' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.lastrun_download_count == 0

        result = self.run_cli_download(traveltimes_model=[])  # invalid type
        assert result.exit_code != 0
        assert 'Error: Invalid type for "traveltimes_model":' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.lastrun_download_count == 0

        result = self.run_cli_download(traveltimes_model='wat')  # invalid value
        assert result.exit_code != 0
        assert 'Error: Invalid value for "traveltimes_model":' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.lastrun_download_count == 0

        # same as above but with error from the cli, not from within the config yaml:
        result = self.run_cli_download('--traveltimes-model', 'wat')  # invalid value
        assert result.exit_code != 0
        assert 'Error: Invalid value for "traveltimes_model":' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.lastrun_download_count == 0

        result = self.run_cli_download(dburl=self.d_yaml_file)  # existing file, invalid db url
        assert result.exit_code != 0
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.lastrun_download_count == 0

        result = self.run_cli_download(dburl="sqlite:/whatever")  # invalid db url
        assert result.exit_code != 0
        assert 'Error: Invalid value for "dburl":' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.lastrun_download_count == 0

        result = self.run_cli_download(dburl="sqlite://whatever")  # invalid db url
        assert result.exit_code != 0
        assert 'Error: Invalid value for "dburl":' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.lastrun_download_count == 0

        result = self.run_cli_download(dburl=[])  # invalid type
        assert result.exit_code != 0
        assert 'Error: Invalid type for "dburl":' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.lastrun_download_count == 0

        # Test an invalif configfile. This can be done only via command line
        result = self.run_cli_download('-c', 'frjkwlag5vtyhrbdd_nleu3kvshg w')
        assert result.exit_code != 0
        assert 'Error: Invalid value for "-c" / "--config":' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.lastrun_download_count == 0

    def test_process_bad_types(self):
        '''bad types must be passed directly to download as click does a preliminary check'''

        result = self.run_cli_process('--pyfile', 'nrvnkenrgdvf')  # invalid value from cli
        # Note the resulting message is SIMILAR to the following, not exactly the same
        # as this is issued by click, the other by our functions in inputargs module
        assert result.exit_code != 0
        assert 'Error: Invalid value for "-p" / "--pyfile":' in result.output

        result = self.run_cli_process('--dburl', 'nrvnkenrgdvf')
        assert result.exit_code != 0
        assert 'Error: Invalid value for "dburl":' in result.output

        result = self.run_cli_process(dburl_in_yaml='abcde')
        assert result.exit_code != 0
        assert 'Error: Invalid value for "dburl":' in result.output
        assert "abcde" in result.output

        result = self.run_cli_process('--funcname', 'nrvnkenrgdvf')
        assert result.exit_code != 0
        assert 'Error: Invalid value for "pyfile": function "nrvnkenrgdvf" not found in' \
            in result.output

        result = self.run_cli_process('-c', 'nrvnkenrgdvf')
        assert result.exit_code != 0
        # this is issued by click (see comment above)
        assert 'Invalid value for "-c" / "--config"' in result.output

        result = self.run_cli_process('-c', self.p_py_file)
        assert result.exit_code != 0
        assert 'Error: Invalid value for "config"' in result.output


@patch('stream2segment.utils.inputargs.get_session')
@patch('stream2segment.main.closesession')
@patch('stream2segment.main.configlog4processing')
@patch('stream2segment.main.run_process')
def test_process_verbosity(mock_run_process, mock_configlog, mock_closesess, mock_getsess,
                           # fixtures:
                           capsys, pytestdir):

    # store stuff in this dict when running configure loggers below:
    vars = {'numloggers': 0, 'logfilepath': None}

    def clogd(logger, logfilebasepath, verbose):
        for h in logger.handlers[:]:
            logger.removeHandler(h)
        # config logger as usual, but redirects to a temp file
        # that will be deleted by pytest, instead of polluting the program
        # package:
        ret = o_configlog4processing(logger,
                                     pytestdir.newfile('.log') if logfilebasepath else None,
                                     verbose)

        vars['numloggers'] = len(ret)

        vars['logfilepath'] = None
        try:
            vars['logfilepath'] = ret[0].baseFilename
        except (IndexError, AttributeError):
            pass
        return ret
    mock_configlog.side_effect = clogd

    dburl = 'sqlite:///:memory:'
    # close session should not close session, otherwise with a memory db we loose the data
    mock_closesess.side_effect = lambda *a, **v: None
    # also, mock get_session cause we need the same session object to
    # retrieve what's been written on the db
    sess = get_session(dburl)
    # mock get_session in order to return always the same session objet:
    mock_getsess.side_effect = lambda *a, **kw: sess

    conffile, pyfile = get_templates_fpaths("paramtable.yaml", "paramtable.py")

    # run verbosity = True, with output file. This configures a logger to log file and a logger
    # stdout
    outfile = pytestdir.newfile()
    mock_run_process.side_effect = lambda *a, **v: None
    ret = o_process(dburl, pyfile, funcname=None, config=conffile, outfile=outfile,
                    log2file=True, verbose=True)
    out, err = capsys.readouterr()
    assert len(out)  # assert empty (avoid comparing to strings and potential py2 py3 headache)
    assert vars['numloggers'] == 2

    # run verbosity = False, with output file. This configures a logger to log file
    outfile = pytestdir.newfile()
    mock_run_process.side_effect = lambda *a, **v: None
    ret = o_process(dburl, pyfile, funcname=None, config=conffile, outfile=outfile,
                    log2file=False, verbose=False)
    out, err = capsys.readouterr()
    assert not out  # assert empty (avoid comparing to strings and potential py2 py3 headache)
    assert vars['numloggers'] == 0

    # run verbosity = False, with output file. This configures a logger to log file
    outfile = pytestdir.newfile()
    mock_run_process.side_effect = lambda *a, **v: None
    ret = o_process(dburl, pyfile, funcname=None, config=conffile, outfile=outfile,
                    log2file=True, verbose=False)
    out, err = capsys.readouterr()
    assert not out  # assert empty (avoid comparing to strings and potential py2 py3 headache)
    assert vars['numloggers'] == 1

    # run verbosity = True, with no output file. This configures a logger stderr and a logger stdout
    mock_run_process.side_effect = lambda *a, **v: None
    ret = o_process(dburl, pyfile, funcname=None, config=conffile, outfile=None,
                    log2file=True, verbose=True)
    out, err = capsys.readouterr()
    with open(vars['logfilepath']) as _opn:
        expected_out = _opn.read()
    # out below is uncicode in PY2, whereas expected_string is str. Thus:
    if PY2:
        expected_out = expected_out.decode('utf8')
    assert expected_out == out
    assert vars['numloggers'] == 2

    # run verbosity = False, with no output file. This configures a logger stderr but no logger stdout
    # with no file
    mock_run_process.side_effect = lambda *a, **v: None
    ret = o_process(dburl, pyfile, funcname=None, config=conffile, outfile=None,
                    log2file=False, verbose=False)
    out, err = capsys.readouterr()
    assert not out  # assert empty (avoid comparing to strings and potential py2 py3 headache)
    assert vars['logfilepath'] is None
    assert vars['numloggers'] == 0


@patch('stream2segment.utils.inputargs.get_session')
@patch('stream2segment.main.closesession')
@patch('stream2segment.main.configlog4download')
@patch('stream2segment.main.run_download')
def test_download_verbosity(mock_run_download, mock_configlog, mock_closesess, mock_getsess,
                            # fixtures:
                            capsys, pytestdir):
    # handlers should be removed each run_download call, otherwise we end up
    # appending them
    numloggers = [0]

    def clogd(logger, logfilebasepath, verbose):
        for h in logger.handlers[:]:
            logger.removeHandler(h)
        # config logger as usual, but redirects to a temp file
        # that will be deleted by pytest, instead of polluting the program
        # package:
        ret = o_configlog4download(logger,
                                   pytestdir.newfile('.log') if logfilebasepath else None,
                                   verbose)
        numloggers[0] = len(ret)
        return ret
    mock_configlog.side_effect = clogd

    dburl = 'sqlite:///:memory:'
    # close session should not close session, otherwise with a memory db we loose the data
    mock_closesess.side_effect = lambda *a, **v: None
    # also, mock get_session cause we need the same session object to
    # retrieve what's been written on the db
    sess = get_session(dburl)
    # mock get_session in order to return always the same session objet:
    mock_getsess.side_effect = lambda *a, **kw: sess

    last_known_id = [None]  # stupid hack to assign to out-of-scope var (py2 compatible)
    def dblog_err_warn():
        qry = sess.query(Download.id, Download.log, Download.warnings, Download.errors)
        if last_known_id[0] is not None:
            qry = qry.filter(Download.id > last_known_id[0])
        tup = qry.first()
        last_known_id[0] = tup[0]
        return tup[1], tup[2], tup[3]

    d_yaml_file = get_templates_fpath("download.yaml")

    # run verbosity = 0. As this does not configure loggers, previous loggers will not be removed
    # (see mock above). Thus launch all tests in increasing verbosity order (from 0 on)
    mock_run_download.side_effect = lambda *a, **v: None
    ret = o_download(d_yaml_file, log2file=False, verbose=False, dburl=dburl)
    out, err = capsys.readouterr()
    assert not out  # assert empty (avoid comparing to strings and potential py2 py3 headache)
    log, err, warn = dblog_err_warn()
    assert "N/A: either logger not configured, or " in log
    assert err == 0
    assert warn == 0
    assert numloggers[0] == 0

    # now let's see that if we raise an exception we also
    mock_run_download.side_effect = KeyError('a')
    # verbosity=1 configures loggers, but only the Db logger
    with pytest.raises(KeyError) as kerr:
        ret = o_download(d_yaml_file,  log2file=False, verbose=False, dburl=dburl)
    out, err = capsys.readouterr()
    assert not out
    log, err, warn = dblog_err_warn()
    assert "N/A: either logger not configured, or " in log
    assert err == 0
    assert warn == 0
    assert numloggers[0] == 0

    # verbosity=1 configures loggers, but only the Db logger
    mock_run_download.side_effect = lambda *a, **v: None
    ret = o_download(d_yaml_file,  log2file=True, verbose=False, dburl=dburl)
    out, err = capsys.readouterr()
    # this is also empty cause mock_run_download is no-op
    assert not out  # assert empty
    log, err, warn = dblog_err_warn()
    assert "Completed in " in log
    assert 'No errors' in log  # 0 total errors
    assert 'No warnings' in log  # 0 total warnings
    assert numloggers[0] == 1

    # now let's see that if we raise an exception we also
    mock_run_download.side_effect = KeyError('a')
    with pytest.raises(KeyError) as kerr:
        ret = o_download(d_yaml_file, log2file=True, verbose=False, dburl=dburl)
    out, err = capsys.readouterr()
    assert not out
    log, err, warn = dblog_err_warn()
    assert "Traceback (most recent call last):" in log
    assert err == 0
    assert warn == 0
    assert numloggers[0] == 1

    mock_run_download.side_effect = lambda *a, **v: None
    ret = o_download(d_yaml_file, log2file=True, verbose=True, dburl=dburl)
    out, err = capsys.readouterr()
    assert out  # assert non empty
    log, err, warn = dblog_err_warn()
    assert "Completed in " in log
    assert 'No errors' in log  # 0 total errors
    assert 'No warnings' in log  # 0 total warnings
    assert numloggers[0] == 2

    # now let's see that if we raise an exception we also
    mock_run_download.side_effect = KeyError('a')
    with pytest.raises(KeyError) as kerr:
        ret = o_download(d_yaml_file, log2file=True, verbose=True, dburl=dburl)
    out, err = capsys.readouterr()
    # Now out is not empty cause the logger which prints to stdout infos errors and critical is set:
    assert "Traceback (most recent call last):" in out
    assert "KeyError" in out
    log, err, warn = dblog_err_warn()
    assert "Traceback (most recent call last):" in log
    assert err == 0
    assert warn == 0
    assert numloggers[0] == 2


@pytest.mark.parametrize('val, exp_value',
                         [(['A', 'D', 'C', 'B'], ['A', 'B', 'C', 'D']),
                          ("B,D,C,A", ['A', 'B', 'C', 'D']),
                          ('A*, B??, C*', ['A*', 'B??', 'C*']),
                          ('!A*, B??, C*', ['!A*', 'B??', 'C*']),
                          (' A, B ', ['A', 'B']),
                          ('*', []),
                          ([], []),
                          ('  ', ['']),
                          (' ! ', ['!']),
                          (' !* ', None),  # None means: raises ValueError
                          ("!H*, H*", None),
                          ("A B, CD", None),
                          ])
def test_nslc_param_value_aslist(val, exp_value):

    for i in range(4):
        if exp_value is None:
            with pytest.raises(ValueError):
                nslc_param_value_aslist(val)
        else:
            assert nslc_param_value_aslist(val) == exp_value

@pytest.mark.parametrize('val, exp_value',
                         [(['A', 'D', 'C', '--'], ['--', 'A', 'C', 'D']),
                          ('A , D , C , --', ['--', 'A', 'C', 'D']),
                          ([' --  '], ['--']),
                          (' -- ',  ['--']),
                          ])
def test_nslc_param_value_aslist_locations(val, exp_value):

    for i in range(4):
        if exp_value is None:
            with pytest.raises(ValueError):
                nslc_param_value_aslist(val)
        else:
            assert nslc_param_value_aslist(val) == exp_value
