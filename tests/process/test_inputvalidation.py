'''
Created on May 23, 2017

@author: riccardo
'''
# as we patch os.path.isfile, this seems to be the correct way to store beforehand
# the original functions (also in other packages, e.g. pytestdir in conftest does not break):
from os.path import isfile, join, dirname
from datetime import datetime
from itertools import product

# this can not apparently be fixed with the future package:
# The problem is io.StringIO accepts unicodes in python2 and strings in python3:
from stream2segment.process.inputvalidation import load_config

try:
    from cStringIO import StringIO  # python2.x pylint: disable=unused-import
except ImportError:
    from io import StringIO  # @UnusedImport

from mock.mock import patch
from future.utils import PY2
from click.testing import CliRunner
import pytest

from stream2segment.cli import cli
from stream2segment.process.main import (configlog4processing as o_configlog4processing,
                                         get_session as o_get_session)
from stream2segment.process.main import process as o_process
from stream2segment.io.inputvalidation import BadParam
from stream2segment.resources import get_templates_fpaths, get_templates_fpath
from stream2segment.io import yaml_load


@pytest.fixture
def run_cli_download(pytestdir, db):
    '''returns a function(*arg, removals=None, **kw) where each arg is the
    COMMAND LINE parameter to be overridden, removals is a list of pyaml
    parameters TO BE REMOVED, and **kw the YAML CONFIG PARAMETERS TO BE
    overridden.
    Uses fixture pytestdir defined in conftest
    '''
    def func(*args, removals=None, **yaml_overrides):
        args = list(args)
        nodburl = False
        # override the db path with our currently tested one:
        if '-d' not in args and '--dburl' not in args and'dburl' not in yaml_overrides:
            yaml_overrides['dburl'] = db.dburl
            nodburl = True
        # if -c or configfile is not specified, add it:
        if "-c" not in args and "--configfile" not in args:
            args.extend(['-c', pytestdir.yamlfile(get_templates_fpath("download.yaml"),
                                                  removals=removals,
                                                  **yaml_overrides)])
        elif nodburl:
            args += ['-d', str(db.dburl)]
        # process inputs:
        runner = CliRunner()
        result = runner.invoke(cli, ['download'] + args)
        return result

    return func


def msgin(msg, click_output):
    '''click changes the quote character in messages. Provide a common
    function to test messages regardeless of the quote character
    '''
    if '"' in msg and "'" not in msg:
        return (msg in click_output) or (msg.replace('"', "'") in click_output)
    elif '"' not in msg and "'" in msg:
        return (msg in click_output) or (msg.replace("'", '"') in click_output)
    else:
        return msg in click_output

class patches(object):
    # paths container for class-level patchers used below. Hopefully
    # will mek easier debug when refactoring/move functions
    get_session = 'stream2segment.process.main.get_session'
    close_session = 'stream2segment.process.main.close_session'
    configlog4processing = 'stream2segment.process.main.configlog4processing'
    run_process = 'stream2segment.process.main._run_and_write'


class Test(object):

    yaml_def_params = yaml_load(get_templates_fpath("download.yaml"))

    # execute this fixture always even if not provided as argument:
    # https://docs.pytest.org/en/documentation-restructure/how-to/fixture.html#autouse-fixtures-xunit-setup-on-steroids
    @pytest.fixture(autouse=True)
    def init(self, request, db, data, pytestdir):
        # re-init a sqlite database (no-op if the db is not sqlite):
        db.create(to_file=False)
        # Although we do not test db stuff here other than checking a download id has written,
        # we iterate through all given db's


        with patch(patches.get_session) as _:
            def csess(dbpath, *a, **v):
                if dbpath == db.dburl:
                    return db.session
                return o_get_session(dbpath, *a, **v)
            _.side_effect = csess

            with patch(patches.close_session) as _:
                self.mock_close_session = _

                with patch(patches.configlog4processing,
                           side_effect=o_configlog4processing) as _:
                    self.mock_configlog4processing = _

                    with patch(patches.run_process,
                               side_effect=lambda *a, **v: None) as _:
                        self.mock_run_process = _

                        yield


def test_process_bad_types(pytestdir):
    '''bad types must be passed directly to download as click does a preliminary check'''

    p_yaml_file, p_py_file = \
        get_templates_fpaths("paramtable.yaml", "paramtable.py")

    # Note that our functions in inputvalidation module return SIMILART messages as click
    # not exactly the same one

    result = CliRunner().invoke(cli, ['process', '--pyfile', 'nrvnkenrgdvf'])
    assert result.exit_code != 0
    assert msgin('Error: Invalid value for "-p" / "--pyfile":', result.output)

    result = CliRunner().invoke(cli, ['process', '--dburl', 'nrvnkenrgdvf', '-c', p_yaml_file,
                                      '-p', p_py_file])
    assert result.exit_code != 0
    assert msgin('Error: Invalid value for "dburl":', result.output)

    # if we do not provide click default values, they have invalid values and they take priority
    # (the --dburl arg is skipped):
    result = CliRunner().invoke(cli, ['process', '--dburl', 'nrvnkenrgdvf', '-c', p_yaml_file])
    assert result.exit_code != 0
    assert msgin('Missing option "-p" / "--pyfile"', result.output)

    result = CliRunner().invoke(cli, ['process', '--dburl', 'nrvnkenrgdvf'])
    assert result.exit_code != 0
    assert msgin('Missing option "-c" / "--config"', result.output)

    result = CliRunner().invoke(cli, ['process', '--dburl', 'nrvnkenrgdvf', '-c', p_yaml_file,
                                      '-p', p_py_file])
    assert result.exit_code != 0
    assert msgin('Error: Invalid value for "dburl":', result.output)
    assert "nrvnkenrgdvf" in result.output

    d_yaml_file = get_templates_fpath('download.yaml')
    d_yaml_file = pytestdir.yamlfile(d_yaml_file, dburl='sqlite:///./path/to/my/db/sqlite.sqlite')
    result = CliRunner().invoke(cli, ['process', '--dburl', d_yaml_file, '-c', p_yaml_file,
                                      '-p', p_py_file])
    assert result.exit_code != 0
    assert msgin('Error: Invalid value for "dburl":', result.output)

    d_yaml_file = pytestdir.yamlfile(d_yaml_file, dburl='sqlite:///:memory:')
    result = CliRunner().invoke(cli, ['process', '--funcname', 'nrvnkenrgdvf', '-c', p_yaml_file,
                                      '-p', p_py_file, '-d', d_yaml_file])
    assert result.exit_code != 0
    assert 'Error: Invalid value for "pyfile": function "nrvnkenrgdvf" not found in' \
        in result.output

    result = CliRunner().invoke(cli, ['process', '-c', 'nrvnkenrgdvf', '-p', p_py_file,
                                      '-d', d_yaml_file])
    assert result.exit_code != 0
    # this is issued by click (see comment above)
    assert msgin('Invalid value for "-c" / "--config"', result.output)

    result = CliRunner().invoke(cli, ['process', '-c', p_py_file, '-p', p_py_file,
                                      '-d', d_yaml_file])
    assert result.exit_code != 0
    assert msgin('Error: Invalid value for "config"', result.output)

    # test an old python module without the SkipSegment import
    with open(p_py_file) as _:
        content =  _.read()
    import_statement = 'from stream2segment.process import gui, SkipSegment'
    assert import_statement in content
    content = content.replace(import_statement,
                              "from stream2segment.process import gui")
    assert import_statement not in content
    p_py_file2 = pytestdir.newfile(".py", True)
    with open(p_py_file2, 'wt') as _:
        _.write(content)

    result = CliRunner().invoke(cli, ['process', '-c', p_yaml_file,
                                      '-p', p_py_file2,
                                      '-d', "sqlite://"])
    assert result.exit_code != 0
    assert msgin('Invalid value for "pyfile": the provided Python module looks outdated.',
                 result.output)


# @patch('stream2segment.utils.inputvalidation.valid_session')
# @patch('stream2segment.main.closesession')
# @patch('stream2segment.main.configlog4processing')
# @patch('stream2segment.main.run_process')
@patch(patches.get_session)
@patch(patches.close_session)
@patch(patches.configlog4processing)
@patch(patches.run_process)
def test_process_verbosity(mock_run_process, mock_configlog, mock_closesess, mock_getsess,
                           # fixtures:
                           db, capsys, pytestdir):

    if not db.is_sqlite:
        pytest.skip("Skipping postgres test (only sqlite memory used)")

    db.create(to_file=False)
    dburl = db.dburl
    sess = db.session
    # mock get_session in order to return always the same session objet:
    mock_getsess.side_effect = lambda *a, **kw: sess
    # close session should not close session, otherwise with a memory db we loose the data
    mock_closesess.side_effect = lambda *a, **v: None

    # store stuff in this dict when running configure loggers below:
    vars = {'numloggers': 0, 'logfilepath': None}

    def clogd(logger, logfilebasepath, verbose):
        for h in logger.handlers[:]:
            logger.removeHandler(h)
        # config logger as usual, but redirects to a temp file
        # that will be deleted by pytest, instead of polluting the program
        # package:
        o_configlog4processing(logger,
                               pytestdir.newfile('.log') if logfilebasepath else None,
                               verbose)

        vars['numloggers'] = len(logger.handlers)

        vars['logfilepath'] = None
        try:
            vars['logfilepath'] = logger.handlers[0].baseFilename
        except (IndexError, AttributeError):
            pass

    mock_configlog.side_effect = clogd

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

    # run verbosity = False, with no output file. This configures a logger stderr but no logger
    # stdout with no file
    mock_run_process.side_effect = lambda *a, **v: None
    ret = o_process(dburl, pyfile, funcname=None, config=conffile, outfile=None,
                    log2file=False, verbose=False)
    out, err = capsys.readouterr()
    assert not out  # assert empty (avoid comparing to strings and potential py2 py3 headache)
    assert vars['logfilepath'] is None
    assert vars['numloggers'] == 0

@pytest.mark.parametrize('adv_set, exp_multiprocess_value',
                         [({'multi_process': True, 'num_processes': 4}, 4),
                          ({'multi_process': False, 'num_processes': 4}, False),
                          ({'multi_process': 3, 'num_processes': 4}, 3),
                          ])
def test_processing_advanced_settings_num_processes(adv_set,
                                                    exp_multiprocess_value):
    """Test old and new multi_process ploicy in advanced_settings
    (before: two params, multi_process and num_propcesses, now single
    param multi_process accepting bool or int
    """
    p_yaml_file, p_py_file = \
        get_templates_fpaths("paramtable.yaml", "paramtable.py")

    config, seg_sel, multi_process, chunksize, writer_options =\
        load_config(config=p_yaml_file,
                    advanced_settings=adv_set)
    assert exp_multiprocess_value == multi_process


def test_processing_advanced_settings_bad_params():
    p_yaml_file, p_py_file = \
        get_templates_fpaths("paramtable.yaml", "paramtable.py")
    adv_set = {'multi_process': 'a'}
    # (pytest.raises problems with PyCharm, simply try .. catch the old way):
    try:
        _ = load_config(config=p_yaml_file, advanced_settings=adv_set)
        assert False, "should raise"
    except BadParam as bp:
        assert 'Invalid type for "advanced_settings.multi_process"' in str(bp)

    adv_set = {'multi_process': True, "segments_chunksize": 'a'}
    # (pytest.raises problems with PyCharm, simply try .. catch the old way):
    try:
        _ = load_config(config=p_yaml_file, advanced_settings=adv_set)
        assert False, "should raise"
    except BadParam as bp:
        assert 'Invalid type for "advanced_settings.segments_chunksize"' in str(bp)