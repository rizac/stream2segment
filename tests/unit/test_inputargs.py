'''
Created on May 23, 2017

@author: riccardo
'''
# as we patch os.path.isfile, this seems to be the correct way to store beforehand
# the original functions (also in other packages, e.g. pytestdir in conftest does not break):
from os.path import isfile, basename, join, abspath, dirname, relpath
from datetime import datetime
from itertools import product

# this can not apparently be fixed with the future package:
# The problem is io.StringIO accepts unicodes in python2 and strings in python3:
try:
    from cStringIO import StringIO  # python2.x pylint: disable=unused-import
except ImportError:
    from io import StringIO  # @UnusedImport

from mock.mock import patch
from future.utils import PY2
from click.testing import CliRunner
import pytest

from stream2segment.cli import cli
from stream2segment.main import configlog4download as o_configlog4download,\
    new_db_download as o_new_db_download, configlog4processing as o_configlog4processing, \
    process as o_process, download as o_download
from stream2segment.utils.inputargs import get_session as o_get_session, \
    nslc_param_value_aslist
from stream2segment.io.db.models import Download
from stream2segment.utils import secure_dburl
from stream2segment.utils.resources import get_templates_fpath, yaml_load, get_templates_fpaths


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


class Test(object):

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

                        with patch('stream2segment.main.run_download',
                                   side_effect = lambda *a, **v: None) as _:  # no-op
                            self.mock_run_download = _

                            with patch('stream2segment.main.configlog4processing',
                                       side_effect=o_configlog4processing) as _:
                                self.mock_configlog4processing = _

                                with patch('stream2segment.main.run_process',
                                           side_effect=lambda *a, **v: None) as _:
                                    self.mock_run_process = _

                                    yield

    def test_paper_suppl_config(self,
                                # fixtures:
                                db, data, run_cli_download):
        '''test the yaml providede in the supplemented material for the paper'''
        result = run_cli_download('-c', data.path('download_poligon_article.yaml'))  # conflict
        # assert we did write to the db:
        assert result.exit_code == 0
        assert db.session.query(Download).count() == 1

    def test_download_bad_values(self,
                                 # fixtures:
                                 db, run_cli_download):
        '''test different scenarios where the value in the dwonload.yaml are not well formatted'''

        # INCREMENT THIS VARIABLE EVERY TIME YOU RUN A SUCCESSFUL DOWNLOAD
        dcount = 0

        # test that eventws_params is optional if not provided
        result = run_cli_download(removals=['eventws_params'])  # conflict
        assert result.exit_code == 0
        dcount += 1

        eventswparams = ['minlat', 'minlatitude', 'maxlat', 'maxlatitude',
                         'minlon', 'minlongitude', 'maxlon', 'maxlongitude']
        result = run_cli_download(removals=eventswparams)  # conflict
        assert all(_ not in result.output for _ in eventswparams)
        assert result.exit_code == 0
        dcount += 1

        result = run_cli_download(min_sample_rate='abc')  # conflict
        assert result.exit_code != 0
        assert '"min_sample_rate"' in result.output

        result = run_cli_download(removals=['min_sample_rate'])  # conflict
        assert self.mock_run_download.call_args_list[-1][1]['min_sample_rate'] == 0
        assert result.exit_code == 0
        dcount += 1

        result = run_cli_download(networks={'a': 'b'})  # conflict
        assert result.exit_code != 0
        assert 'Error: Conflicting names "network" / "networks"' in result.output
        result = run_cli_download(network={'a': 'b'})
        assert result.exit_code == 0
        dcount += 1
        # thus providing dict is actually fine and will iterate over its keys:
        assert self.mock_run_download.call_args_list[-1][1]['network'] == ['a']
        # do some asserts only for this case to test how we print the arguments to string:
        # assert "tt_table: <TTTable object, " in result.output
        assert "starttime: 2006-01-01 00:00:00" in result.output
        assert "traveltimes_model:" in result.output
        _dburl = db.dburl
        if not db.is_sqlite:
            _dburl = secure_dburl(_dburl)
        # assert dburl is in result.output (sqlite:memory is quotes, postgres not. we do not
        # care to investigate why, jsut assert either string is there:
        assert "dburl: '%s'" % _dburl in result.output or "dburl: %s" % _dburl in result.output

        # check the session:
        # assert we did write to the db:
        assert db.session.query(Download).count() == dcount

        result = run_cli_download(networks='!*')  # conflicting names
        assert result.exit_code != 0
        assert 'Error: Conflicting names "network" / "networks"' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        result = run_cli_download(network='!*')  # invalid value
        assert result.exit_code != 0
        assert 'Error: Invalid value for "network": ' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        result = run_cli_download(net='!*')  # conflicting names
        assert result.exit_code != 0
        assert 'Error: Conflicting names "network" / "net"' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        # test error from the command line. Result is the same as above as the check is made
        # AFTER click
        result = run_cli_download('-n', '!*')  # invalid value
        assert result.exit_code != 0
        assert 'Error: Invalid value for "network": ' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        # no such option:
        result = run_cli_download('--zrt', '!*')
        assert result.exit_code != 0
        assert 'Error: no such option: --zrt' in result.output  # why -z and not -zz? whatever...
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        # no such option from within the yaml:
        result = run_cli_download(zz='!*')
        assert result.exit_code != 0
        assert 'Error: No such option "zz"' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        # what about conflicting arguments?
        result = run_cli_download(networks='!*', net='opu')  # invalid value
        assert result.exit_code != 0
        assert 'Conflicting names "network" / "net" / "networks"' in result.output or \
            'Conflicting names "network" / "networks" / "net"' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        result = run_cli_download(starttime=[])  # invalid type
        assert result.exit_code != 0
        assert 'Error: Invalid type for "starttime":' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        # mock implementing conflicting names in the yaml file:
        result = run_cli_download(start='wat')  # invalid value
        assert result.exit_code != 0
        assert 'Error: Conflicting names "starttime" / "start"' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        # mock implementing bad value in the cli: (cf with the previous test):
        # THE MESSAGE BELOW IS DIFFERENT BECAUSE WE PROVIDE A CLI VALIDATION FUNCTION
        # See the case of travetimes model below where, without a cli validation function,
        # the message is the same when we provide a bad argument in the yaml or from the cli
        result = run_cli_download('--starttime', 'wat')  # invalid value
        assert result.exit_code != 0
        assert 'Error: Invalid value for "-s" / "--start" / "--starttime": wat' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        # This should work:
        result = run_cli_download('--start', '2006-03-14')  # invalid value
        assert result.exit_code == 0
        dcount += 1
        run_download_kwargs = self.mock_run_download.call_args_list[-1][1]
        assert run_download_kwargs['starttime'] == datetime(2006, 3, 14)
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        # now test the same as above BUT with a cli-only argument (-t0):
        result = run_cli_download('-s', 'wat')  # invalid value typed from the command line
        assert result.exit_code != 0
        assert 'Error: Invalid value for "-s" / "--start" / "--starttime":' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        result = run_cli_download(endtime='wat')  # try with end
        assert result.exit_code != 0
        assert 'Error: Invalid value for "endtime":' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        result = run_cli_download(end='wat')  # try with end
        assert result.exit_code != 0
        assert 'Error: Conflicting names "endtime" / "end"' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        # now test the same as above BUT with the wrong value from the command line:
        result = run_cli_download('-e', 'wat')  # invalid value typed from the command line
        assert result.exit_code != 0
        assert 'Error: Invalid value for "-e" / "--end" / "--endtime":' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        result = run_cli_download(traveltimes_model=[])  # invalid type
        assert result.exit_code != 0
        assert 'Error: Invalid type for "traveltimes_model":' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        result = run_cli_download(traveltimes_model='wat')  # invalid value
        assert result.exit_code != 0
        assert 'Error: Invalid value for "traveltimes_model":' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        # same as above but with error from the cli, not from within the config yaml:
        result = run_cli_download('--traveltimes-model', 'wat')  # invalid value
        assert result.exit_code != 0
        assert 'Error: Invalid value for "traveltimes_model":' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        result = run_cli_download(removals=['inventory'])  # invalid value
        assert result.exit_code != 0
        assert 'Error: Missing value for "inventory"' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        d_yaml_file = get_templates_fpath("download.yaml")

        result = run_cli_download(dburl=d_yaml_file)  # existing file, invalid db url
        assert result.exit_code != 0
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        result = run_cli_download(dburl="sqlite:/whatever")  # invalid db url
        assert result.exit_code != 0
        assert 'Error: Invalid value for "dburl":' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        result = run_cli_download(dburl="sqlite://whatever")  # invalid db url
        assert result.exit_code != 0
        assert 'Error: Invalid value for "dburl":' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        result = run_cli_download(dburl=[])  # invalid type
        assert result.exit_code != 0
        assert 'Error: Invalid type for "dburl":' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        # Test an invalif configfile. This can be done only via command line
        result = run_cli_download('-c', 'frjkwlag5vtyhrbdd_nleu3kvshg w')
        assert result.exit_code != 0
        assert 'Error: Invalid value for "-c" / "--config":' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        result = run_cli_download(removals=['maxmagnitude'])  # remove an opt. param.
        assert result.exit_code == 0
        dcount += 1
        # check maxmagnitude is NOT in the eventws params:
        eventws_params = self.mock_run_download.call_args_list[-1][1]['eventws_params']
        assert 'maxmagnitude' not in eventws_params
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        result = run_cli_download(removals=['advanced_settings'])  # remove an opt. param.
        assert result.exit_code != 0
        assert 'Error: Missing value for "advanced_settings"' in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        result = run_cli_download(advanced_settings={})  # remove an opt. param.
        assert result.exit_code != 0
        assert ('Error: Invalid value for "advanced_settings": '
                'Missing value for "download_blocksize"') in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        # search radius:
        for search_radius in [{'min': 5}, {'min': 5, 'max': 6, 'minmag': 7}]:
            result = run_cli_download(search_radius=search_radius)
            assert result.exit_code != 0
            assert ('Error: Invalid value for "search_radius": '
                    "provide either 'min', 'max' or "
                    "'minmag', 'maxmag', 'minmag_radius', 'maxmag_radius'") in result.output

        result = run_cli_download(search_radius={'min': 5, 'max': '6'})
        assert result.exit_code != 0
        assert ('Error: Invalid value for "search_radius": '
                "numeric values expected") in result.output

        result = run_cli_download(search_radius={'minmag': 15, 'maxmag': 7,
                                                 'minmag_radius': 5,
                                                 'maxmag_radius': 4})
        assert result.exit_code != 0
        assert ('Error: Invalid value for "search_radius": '
                'minmag should not be greater than maxmag') in result.output

        result = run_cli_download(search_radius={'minmag': 7, 'maxmag': 8,
                                                 'minmag_radius': -1,
                                                 'maxmag_radius': 0})
        assert result.exit_code != 0
        assert ('Error: Invalid value for "search_radius": '
                'minmag_radius and maxmag_radius should be greater than 0') in result.output

        result = run_cli_download(search_radius={'minmag': 5, 'maxmag': 5,
                                                 'minmag_radius': 4,
                                                 'maxmag_radius': 4})
        assert result.exit_code != 0
        assert ('Error: Invalid value for "search_radius": '
                'To supply a constant radius, '
                'set "min: 0" and specify the radius with the "max" argument') in result.output

        result = run_cli_download(search_radius={'min': -1, 'max': 5})
        assert result.exit_code != 0
        assert ('Error: Invalid value for "search_radius": '
                'min should not be lower than 0') in result.output

        result = run_cli_download(search_radius={'min': 0, 'max': 0})
        assert result.exit_code != 0
        assert ('Error: Invalid value for "search_radius": '
                'max should be greater than 0') in result.output

        result = run_cli_download(search_radius={'min': 4, 'max': 3})
        assert result.exit_code != 0
        assert ('Error: Invalid value for "search_radius": '
                'min should be lower than max') in result.output


@patch('stream2segment.main.run_download', side_effect=lambda *a, **v: None)
@patch('stream2segment.utils.inputargs.os.path.isfile', side_effect=isfile)
def test_download_eventws_query_args(mock_isfile, mock_run_download,
                                     # fixtures:
                                     run_cli_download):  # pylint: disable=redefined-outer-name
    '''test different scenarios where we provide eventws query args from the command line'''

    d_yaml_file = get_templates_fpath("download.yaml")
    # FIRST SCENARIO: no  eventws_params porovided
    mock_run_download.reset_mock()
    def_yaml_dict = yaml_load(d_yaml_file)['eventws_params']
    assert not def_yaml_dict  # None or empty dict
    result = run_cli_download()  # invalid type
    assert result.exit_code == 0
    # assert the yaml (as passed to the download function) has the correct value:
    real_eventws_params = mock_run_download.call_args_list[0][1]['eventws_params']
    # just assert it has keys merged from the global event-related yaml keys
    assert 'maxmagnitude' not in real_eventws_params
    assert real_eventws_params

    # test by providing an eventsws param which is not optional:
    mock_run_download.reset_mock()
    def_yaml_dict = yaml_load(d_yaml_file)['eventws_params']
    assert not def_yaml_dict  # None or empty dict
    result = run_cli_download('--minmagnitude', '15.5')
    assert result.exit_code == 0
    # assert the yaml (as passed to the download function) has the correct value:
    real_eventws_params = mock_run_download.call_args_list[0][1]['eventws_params']
    # just assert it has keys merged from the global event-related yaml keys
    assert real_eventws_params['minmagnitude'] == 15.5

    # test by providing a eventsws param which is optional:
    mock_run_download.reset_mock()
    def_yaml_dict = yaml_load(d_yaml_file)['eventws_params']
    assert not def_yaml_dict  # None or empty dict
    result = run_cli_download('--minmagnitude', '15.5',
                              eventws_params={'format': 'abc'})
    assert result.exit_code == 0
    # assert the yaml (as passed to the download function) has the correct value:
    real_eventws_params = mock_run_download.call_args_list[0][1]['eventws_params']
    # just assert it has keys merged from the global event-related yaml keys
    assert real_eventws_params['minmagnitude'] == 15.5
    assert real_eventws_params['format'] == 'abc'

    # conflicting args (supplying a global non-optional param in eventws's config):
    for pars in [['--minlatitude', '-minlat'], ['--maxlatitude', '-maxlat'],
                 ['--minlongitude', '-minlon'], ['--maxlongitude', '-maxlon'],
                 ['--minmagnitude', '-minmag'], ['--maxmagnitude', '-maxmag'],
                 ['--mindepth'], ['--maxdepth']]:
        for par1, par2 in product(pars, pars):
            mock_run_download.reset_mock()
            result = run_cli_download(par1, '15.5',
                                      eventws_params={par2.replace('-', ''): 15.5})
            assert result.exit_code != 1
            assert 'conflict' in result.output
            assert 'Invalid value for "eventws_params"' in result.output

    # test a eventws supplied as non existing file and not valid fdsnws:
    mock_isfile.reset_mock()
    assert not mock_isfile.called
    result = run_cli_download('--eventws', 'myfile')
    assert result.exit_code != 0
    assert 'eventws' in result.output
    assert mock_isfile.called


@pytest.mark.parametrize('filepath_is_abs',
                         [True, False])
@pytest.mark.parametrize('yamlarg',
                         ['eventws', 'restricted_data', 'dburl'])
@patch('stream2segment.main.run_download', side_effect=lambda *a, **v: None)
def test_argument_which_accept_files_relative_and_abs_paths(mock_run_download,
                                                            yamlarg, filepath_is_abs,
                                                            # fixtures:
                                                            pytestdir):
    '''test that arguments accepting files are properly processed and the relative paths
    are resolved relative to the yaml config file'''
    # setup files and relative paths depending on whether we passed relative path or absolute
    # int he config
    if filepath_is_abs:
        yamlarg_file = pytestdir.newfile()
        overrides = {yamlarg: ('sqlite:///' if yamlarg == 'dburl' else '') + yamlarg_file}
        # provide a sqlite memory if we are not testing dburl, otherwise run would fail:
        if yamlarg != 'dburl':
            overrides['dburl'] = 'sqlite:///:memory:'
        yamlfile = pytestdir.yamlfile(get_templates_fpath('download.yaml'),
                                      **overrides)
    else:
        overrides = {yamlarg: ('sqlite:///' if yamlarg == 'dburl' else '') + 'abc'}
        # provide a sqlite memory if we are not testing dburl, otherwise run would fail:
        if yamlarg != 'dburl':
            overrides['dburl'] = 'sqlite:///:memory:'
        yamlfile = pytestdir.yamlfile(get_templates_fpath('download.yaml'),
                                      **overrides)
        # and now create the file:
        yamlarg_file = join(dirname(yamlfile), 'abc')

    # create relative path:
    with open(yamlarg_file, 'w') as opn:
        if yamlarg == 'restricted_data':  # avoid errors if we are testing token file
            opn.write('BEGIN PGP MESSAGE ABC')

    # if we are not testing dburl

    runner = CliRunner()
    result = runner.invoke(cli, ['download',
                                 '-c',
                                 yamlfile])
    assert result.exit_code == 0
    run_download_args = mock_run_download.call_args_list[-1][1]

    if yamlarg == 'restricted_data':
        # assert we read the correct file:
        assert run_download_args['authorizer'].token == b'BEGIN PGP MESSAGE ABC'
    elif yamlarg == 'dburl':
        # assert we have the right db url:
        assert str(run_download_args['session'].bind.engine.url) == 'sqlite:///' + yamlarg_file
    else:
        assert run_download_args[yamlarg] == yamlarg_file


def test_process_bad_types(pytestdir):
    '''bad types must be passed directly to download as click does a preliminary check'''

    p_yaml_file, p_py_file = \
        get_templates_fpaths("paramtable.yaml", "paramtable.py")

    # Note that our functions in inputargs module return SIMILART messages as click
    # not exactly the same one

    result = CliRunner().invoke(cli, ['process', '--pyfile', 'nrvnkenrgdvf'])
    assert result.exit_code != 0
    assert 'Error: Invalid value for "-p" / "--pyfile":' in result.output

    result = CliRunner().invoke(cli, ['process', '--dburl', 'nrvnkenrgdvf', '-c', p_yaml_file,
                                      '-p', p_py_file])
    assert result.exit_code != 0
    assert 'Error: Invalid value for "dburl":' in result.output

    # if we do not provide click default values, they have invalid values and they take priority
    # (the --dburl arg is skipped):
    result = CliRunner().invoke(cli, ['process', '--dburl', 'nrvnkenrgdvf', '-c', p_yaml_file])
    assert result.exit_code != 0
    assert 'Missing option "-p" / "--pyfile"' in result.output

    result = CliRunner().invoke(cli, ['process', '--dburl', 'nrvnkenrgdvf'])
    assert result.exit_code != 0
    assert 'Missing option "-c" / "--config"' in result.output

    result = CliRunner().invoke(cli, ['process', '--dburl', 'nrvnkenrgdvf', '-c', p_yaml_file,
                                      '-p', p_py_file])
    assert result.exit_code != 0
    assert 'Error: Invalid value for "dburl":' in result.output
    assert "nrvnkenrgdvf" in result.output

    d_yaml_file = get_templates_fpath('download.yaml')
    d_yaml_file = pytestdir.yamlfile(d_yaml_file, dburl='sqlite:///./path/to/my/db/sqlite.sqlite')
    result = CliRunner().invoke(cli, ['process', '--dburl', d_yaml_file, '-c', p_yaml_file,
                                      '-p', p_py_file])
    assert result.exit_code != 0
    assert 'Error: Invalid value for "dburl":' in result.output

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
    assert 'Invalid value for "-c" / "--config"' in result.output

    result = CliRunner().invoke(cli, ['process', '-c', p_py_file, '-p', p_py_file,
                                      '-d', d_yaml_file])
    assert result.exit_code != 0
    assert 'Error: Invalid value for "config"' in result.output


@patch('stream2segment.utils.inputargs.get_session')
@patch('stream2segment.main.closesession')
@patch('stream2segment.main.configlog4processing')
@patch('stream2segment.main.run_process')
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


@patch('stream2segment.utils.inputargs.get_session')
@patch('stream2segment.main.closesession')
@patch('stream2segment.main.configlog4download')
@patch('stream2segment.main.run_download')
def test_download_verbosity(mock_run_download, mock_configlog, mock_closesess, mock_getsess,
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
