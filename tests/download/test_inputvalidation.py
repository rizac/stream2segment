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
try:
    from cStringIO import StringIO  # python2.x pylint: disable=unused-import
except ImportError:
    from io import StringIO  # @UnusedImport

from mock.mock import patch
from future.utils import PY2
from click.testing import CliRunner
import pytest

from stream2segment.cli import cli
from stream2segment.download.main import (configlog4download as o_configlog4download,
                                          new_db_download as o_new_db_download)
from stream2segment.download.main import download as o_download
from stream2segment.download.inputvalidation import valid_session as o_valid_session
from stream2segment.download.inputvalidation import valid_nslc as nslc_param_value_aslist
from stream2segment.download.db.models import Download
from stream2segment.io.db import secure_dburl
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

        # patchers:
        def cfd_side_effect(logger, logfilebasepath, verbose):
            # config logger as usual, but redirects to a temp file
            # that will be deleted by pytest, instead of polluting the program
            # package:
            return o_configlog4download(logger, pytestdir.newfile('.log'), verbose)

        class patches(object):
            # paths container for class-level patchers used below. Hopefully
            # will mek easier debug when refactoring/move functions
            configlog4download = 'stream2segment.download.main.configlog4download'
            new_db_download = 'stream2segment.download.main.new_db_download'
            valid_session = 'stream2segment.download.inputvalidation.valid_session'
            close_session = 'stream2segment.download.main.close_session'
            run_download = 'stream2segment.download.main._run'

        with patch(patches.configlog4download, side_effect=cfd_side_effect) as _:
            self.mock_config4download = _

            with patch(patches.new_db_download, side_effect=o_new_db_download) as _:
                self.mock_new_db_download = _

                with patch(patches.valid_session) as _:
                    def csess(dbpath, *a, **v):
                        if dbpath == db.dburl:
                            return db.session
                        return o_valid_session(dbpath, *a, **v)
                    _.side_effect = csess

                    with patch(patches.close_session) as _:
                        self.mock_close_session = _

                        with patch(patches.run_download,
                                   side_effect=lambda *a, **v: None) as _:  # no-op
                            self.mock_run_download = _

                            yield

    def test_paper_suppl_config(self,
                                # fixtures:
                                db, data, run_cli_download):
        '''test the yaml providede in the supplemented material for the paper'''
        result = run_cli_download('-c', data.path('download_poligon_article.yaml'))  # conflict
        # assert we did write to the db:
        assert result.exit_code == 0
        assert db.session.query(Download).count() == 1

    def test_download_bad_values_times(self,
                                       # fixtures:
                                       db, run_cli_download):
        # INCREMENT THIS VARIABLE EVERY TIME YOU RUN A SUCCESSFUL DOWNLOAD
        dcount = 0

        result = run_cli_download(starttime=[])  # invalid type
        assert result.exit_code != 0
        assert msgin('Error: Invalid type for "starttime":', result.output)
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        # mock implementing conflicting names in the yaml file:
        result = run_cli_download(start='wat')  # invalid value
        assert result.exit_code != 0
        assert msgin('Error: Conflicting names "starttime" / "start"', result.output)
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        # mock implementing bad value in the cli: (cf with the previous test):
        # Note that the validation is not performed abnymore at click level so the
        # message will be the same as if the same value was provided via the config file:
        for key, args, kwargs in [
            ['starttime', ('--starttime', 'wat'), {}],  # provided via the cli
            ['starttime', ('-s', 'wat'), {}],
            ['starttime', ('--start', 'wat'), {}],
            ['starttime', [] , {'starttime': 'wat'}],  # provided as download config value
            ['endtime', ('--endtime', 'wat'), {}],  # provided via the cli
            ['endtime', ('-e', 'wat'), {}],
            ['endtime', ('--end', 'wat'), {}],
            ['endtime', [], {'endtime': 'wat'}]
        ]:
            result = run_cli_download(*args, **kwargs)
            assert result.exit_code != 0
            assert msgin('Error: Invalid value for "%s": ' % key, result.output)
            assert msgin(' wat', result.output)
            # assert we did not write to the db, cause the error threw before setting up db:
            assert db.session.query(Download).count() == dcount

        # This should work:
        result = run_cli_download('--start', '2006-03-14')
        assert result.exit_code == 0
        dcount += 1
        run_download_kwargs = self.mock_run_download.call_args_list[-1][1]
        assert run_download_kwargs['starttime'] == datetime(2006, 3, 14)
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

    def test_download_bad_values_searchradius(self,
                                              # fixtures:
                                              db, run_cli_download):
        # INCREMENT THIS VARIABLE EVERY TIME YOU RUN A SUCCESSFUL DOWNLOAD
        dcount = 0

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
                'to supply a constant radius, '
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

    def test_download_bad_values_eventparams(self,
                                             # fixtures:
                                             db, run_cli_download):
        # INCREMENT THIS VARIABLE EVERY TIME YOU RUN A SUCCESSFUL DOWNLOAD
        dcount = 0
        # test that eventws_params is optional if not provided
        result = run_cli_download(removals=['eventws_params'])
        assert result.exit_code == 0
        dcount += 1

        # now check parameters supplied in config and in `eventws` subdict
        result = run_cli_download(eventws_params={'maxmagnitude': 5})
        assert result.exit_code == 0
        dcount += 1
        # check maxmagnitude is NOT in the eventws params:
        eventws_params = self.mock_run_download.call_args_list[-1][1]['eventws_params']
        assert 'maxmagnitude' in eventws_params

        # same as above, but let's check a type error? No, it does not raise
        # because the validation function is float (and float('5') works)
        result = run_cli_download(eventws_params={'maxmagnitude': '5'})
        assert result.exit_code == 0
        dcount += 1

        # same as above, but let's check real a type error, because in this case
        # the validation function is stream2segment 'valid_between''
        result = run_cli_download(minlatitude = '5')
        assert result.exit_code != 0
        assert msgin('Error: Invalid type for "minlatitude":', result.output)

        # same as above, but with conflicts:
        # COMMENT OUT: THIS IS IMPLEMENTED IN test_download_eventws_query_args:
        # for param_name in ['minlat', 'maxlatitude']:
        #     result = run_cli_download(eventws_params={param_name: 5})
        #     assert result.exit_code != 0
        #     assert msgin('Conflicting names ', result.output)
        #     assert msgin('"eventws_params"', result.output)

        eventswparams = ['minlat', 'minlatitude', 'maxlat', 'maxlatitude',
                         'minlon', 'minlongitude', 'maxlon', 'maxlongitude']
        result = run_cli_download(removals=eventswparams)  # conflict
        assert all(_ not in result.output for _ in eventswparams)
        assert result.exit_code == 0
        dcount += 1

        result = run_cli_download(removals=['maxmagnitude'])  # remove an opt. param.
        assert result.exit_code == 0
        dcount += 1

        # check maxmagnitude is NOT in the eventws params:
        eventws_params = self.mock_run_download.call_args_list[-1][1]['eventws_params']
        assert 'maxmagnitude' not in eventws_params
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount


    def test_download_bad_values_nslc(self,
                                      # fixtures:
                                      db, run_cli_download):
        # INCREMENT THIS VARIABLE EVERY TIME YOU RUN A SUCCESSFUL DOWNLOAD
        dcount = 0

        result = run_cli_download(networks={'a': 'b'})  # conflict
        assert result.exit_code != 0
        assert msgin('Error: Conflicting names "network" / "networks"', result.output)
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
        assert msgin('Error: Conflicting names "network" / "networks"', result.output)
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount
        # to be sure we have printed the bad parameter message only:
        assert len(result.output.strip().split('\n')) == 1

        result = run_cli_download(network='!*')  # invalid value
        assert result.exit_code != 0
        assert msgin('Error: Invalid value for "network": ', result.output)
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount
        # to be sure we have printed the bad parameter message only:
        assert len(result.output.strip().split('\n')) == 1

        result = run_cli_download(net='!*')  # conflicting names
        assert result.exit_code != 0
        assert msgin('Error: Conflicting names "network" / "net"', result.output)
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount
        # to be sure we have printed the bad parameter message only:
        assert len(result.output.strip().split('\n')) == 1

        # what about conflicting arguments?
        result = run_cli_download(networks='!*', net='opu')  # invalid value
        assert result.exit_code != 0
        assert msgin('Conflicting names "network" / "net" / "networks"',
                     result.output) or \
               msgin('Conflicting names "network" / "networks" / "net"', result.output)
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount


    def test_download_bad_values_dburl(self,
                                          # fixtures:
                                          db, run_cli_download):
        '''test different scenarios with the value of the db url'''
        # INCREMENT THIS VARIABLE EVERY TIME YOU RUN A SUCCESSFUL DOWNLOAD
        dcount = 0

        d_yaml_file = get_templates_fpath("download.yaml")

        result = run_cli_download(dburl=d_yaml_file)  # existing file, invalid db url
        assert result.exit_code != 0
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        result = run_cli_download(dburl="sqlite:/whatever")  # invalid db url
        assert result.exit_code != 0
        assert msgin('Error: Invalid value for "dburl":', result.output)
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        result = run_cli_download(dburl="sqlite://whatever")  # invalid db url
        assert result.exit_code != 0
        assert msgin('Error: Invalid value for "dburl":', result.output)
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        result = run_cli_download(dburl=[])  # invalid type
        assert result.exit_code != 0
        assert msgin('Error: Invalid type for "dburl":', result.output)
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

    def test_download_non_existing_database(self,
                                            # fixtures:
                                            db, run_cli_download, pytestdir):
        # test non existing databases. For sqlite, create a new non existing file,
        if db.is_sqlite:
            dburl = "sqlite:///" + pytestdir.newfile(".sqlite", create=False)
        else:
            # for postgres, just modify the actual dburl
            dburl = str(db.session.get_bind().url)[:-1] + str(datetime.utcnow().microsecond)
        result = run_cli_download(dburl=dburl)  # invalid db url
        if db.is_sqlite:
            assert result.exit_code == 0
        else:
            import re
            assert 'Invalid value for "dburl":' in result.output
            assert re.search(' database ".*" does not exist, it needs to be created first',
                             result.output)
            assert 'needs to be created first' in result.output
            assert result.exit_code != 0


    def test_download_bad_values(self,
                                 # fixtures:
                                 db, run_cli_download):
        '''test different scenarios where the value in the dwonload.yaml are not well formatted'''

        # INCREMENT THIS VARIABLE EVERY TIME YOU RUN A SUCCESSFUL DOWNLOAD
        dcount = 0

        # test dataws provided multiple times in the cli:
        result = run_cli_download('-ds', 'http://a/fdsnws/dataselect/1/query',
                                  '--dataws', 'http://b/fdsnws/dataselect/1/query')
        dataws = self.mock_run_download.call_args_list[-1][1]['dataws']
        assert sorted(dataws) == ['http://a/fdsnws/dataselect/1/query',
                                  'http://b/fdsnws/dataselect/1/query']
        assert result.exit_code == 0
        dcount += 1

        result = run_cli_download(min_sample_rate='abc')  # conflict
        assert result.exit_code != 0
        assert msgin('"min_sample_rate"', result.output)
        # sometimes we move functions, messing around with what should be logged
        # or not. In most of the tests of this function, the exception traceback should
        # not be there. Perform the check in this case:
        assert "traceback" not in result.output.lower()

        result = run_cli_download(removals=['min_sample_rate'])  # conflict
        assert self.mock_run_download.call_args_list[-1][1]['min_sample_rate'] == 0
        assert result.exit_code == 0
        dcount += 1

        # test error from the command line. Result is the same as above as the check is made
        # AFTER click
        result = run_cli_download('-n', '!*')  # invalid value
        assert result.exit_code != 0
        assert msgin('Error: Invalid value for "network": ', result.output)
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
        assert msgin('Error: No such option "zz"', result.output)
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        result = run_cli_download(traveltimes_model=[])  # invalid type
        assert result.exit_code != 0
        assert msgin('Error: Invalid type for "traveltimes_model":', result.output)
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        result = run_cli_download(traveltimes_model='wat')  # invalid value
        assert result.exit_code != 0
        assert msgin('Error: Invalid value for "traveltimes_model":', result.output)
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        # same as above but with error from the cli, not from within the config yaml:
        result = run_cli_download('--traveltimes-model', 'wat')  # invalid value
        assert result.exit_code != 0
        assert msgin('Error: Invalid value for "traveltimes_model":', result.output)
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        result = run_cli_download(removals=['inventory'])  # invalid value
        assert result.exit_code != 0
        assert msgin('Error: Missing value for "inventory"', result.output)
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        # Test an invalif configfile. This can be done only via command line
        result = run_cli_download('-c', 'frjkwlag5vtyhrbdd_nleu3kvshg w')
        assert result.exit_code != 0
        assert msgin('Error: Invalid value for "-c" / "--config":', result.output)
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        result = run_cli_download(removals=['advanced_settings'])  # remove an opt. param.
        assert result.exit_code != 0
        assert msgin('Error: Missing value for "advanced_settings"', result.output)
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        result = run_cli_download(advanced_settings={})  # remove an opt. param.
        assert result.exit_code != 0
        assert 'Error: Missing value for "advanced_settings.download_blocksize"' \
               in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert db.session.query(Download).count() == dcount

        # check advanced_settings renaming of max_thread_workers
        # normal "new" case:
        adv = dict(self.yaml_def_params['advanced_settings'])
        adv['max_concurrent_downloads'] = 2
        result = run_cli_download(advanced_settings=adv)
        # Test max_concurrent_downloads checking the printed config (hacky but quick):
        assert 'max_concurrent_downloads: 2' in \
               result.output[result.output.index('advanced_settings'):]
        # now provide a "old" config and check we converted to the new param:

        # with patch('stream2segment.main.run_download') as mock_r_d:

        adv.pop('max_concurrent_downloads')
        adv['max_thread_workers'] = 55
        result = run_cli_download(advanced_settings=adv)
        # in the printed config, we still have mac_thread_workers
        assert 'max_thread_workers: 55' in \
               result.output[result.output.index('advanced_settings'):]
        # But not in the advanced settings dict passed to the download routine:
        adv_settings = self.mock_run_download.call_args[-1]['advanced_settings']
        # adv_settings = mock_r_d.call_args[-1]['advanced_settings']
        assert 'max_thread_workers' not in adv_settings
        assert adv_settings['max_concurrent_downloads'] == 55


@patch('stream2segment.download.main._run', side_effect=lambda *a, **v: None)
@patch('stream2segment.download.inputvalidation.os.path.isfile', side_effect=isfile)
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
            assert msgin('Conflicting name(s) ', result.output)
            assert msgin('"eventws_params"', result.output)
            # assert 'conflict' in result.output
            # assert msgin('Invalid value for "eventws_params"', result.output)

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
@patch('stream2segment.download.main._run', side_effect=lambda *a, **v: None)
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


@patch('stream2segment.download.inputvalidation.valid_session')
@patch('stream2segment.download.main.close_session')
@patch('stream2segment.download.main.configlog4download')
@patch('stream2segment.download.main._run')
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
        # config logger as usual, but redirects to a temp file that will be deleted
        # by pytest, instead of polluting the program package:
        o_configlog4download(logger,
                             pytestdir.newfile('.log') if logfilebasepath else None,
                             verbose)
        numloggers[0] = len(logger.handlers)

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
    assert '0 errors' in log  # 0 total errors
    assert '0 warnings' in log  # 0 total warnings
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
    assert '0 errors' in log  # 0 total errors
    assert '0 warnings' in log  # 0 total warnings
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
