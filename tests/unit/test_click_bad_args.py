'''
Created on May 23, 2017

@author: riccardo
'''
import unittest
from click.testing import CliRunner
from stream2segment.cli import cli
from mock.mock import patch
from stream2segment.utils.resources import get_templates_fpath, yaml_load
from stream2segment.main import init as orig_init, helpmathiter as main_helpmathiter
from tempfile import NamedTemporaryFile
import yaml
from contextlib import contextmanager
import os
from datetime import datetime, timedelta
import tempfile
import shutil

from stream2segment.main import configlog4download as o_config4download, \
    new_db_download as o_new_db_download, create_session as o_create_session, run_download as o_run_download, \
    configlog4processing as o_configlog4processing, to_csv as o_to_csv, yaml_load as o_yaml_load
from stream2segment.io.db.models import Download


class Test(unittest.TestCase):

    @staticmethod
    def cleanup(me):
        for patcher in me.patchers:
            patcher.stop()
    
    @property
    def is_sqlite(self):
        return str(self.dburl).startswith("sqlite:///")
    
    @property
    def is_postgres(self):
        return str(self.dburl).startswith("postgresql://")

    def setUp(self):
        
        self.dburl = os.getenv("DB_URL", "sqlite:///:memory:")
        self.patchers = []
        
        self.patchers.append(patch('stream2segment.main.configlog4download'))
        self.mock_config4download = self.patchers[-1].start()
        self.mock_config4download.side_effect = o_config4download
        
        self.patchers.append(patch('stream2segment.main.new_db_download'))
        self.mock_new_db_download = self.patchers[-1].start()
        self.mock_new_db_download.side_effect = o_new_db_download
        
        self.patchers.append(patch('stream2segment.main.create_session'))
        self.mock_create_session = self.patchers[-1].start()
        def csess(*a, **v):
            self.session = o_create_session(*a, **v)
            return self.session
        self.mock_create_session.side_effect = csess
        
        self.patchers.append(patch('stream2segment.main.run_download'))
        self.mock_run_download = self.patchers[-1].start()
        self.mock_run_download.side_effect = lambda *a, **v: None  # no-op
        
        self.patchers.append(patch('stream2segment.main.configlog4processing'))
        self.mock_configlog4processing = self.patchers[-1].start()
        self.mock_configlog4processing.side_effect = o_configlog4processing
        
        self.patchers.append(patch('stream2segment.main.to_csv'))
        self.mock_to_csv = self.patchers[-1].start()
        self.mock_to_csv.side_effect = lambda *a, **v: None  # no-op
        
        self.patchers.append(patch('stream2segment.main.yaml_load'))
        self.mock_yaml_load = self.patchers[-1].start()
        self.yaml_overrides = {}
        def yload(*a, **v):
            dic = yaml_load(*a, **v)
            dic['dburl'] = self.dburl  # in download.yaml, it is a fake address
            # provide a valid one so that we explicitly inject a bad one in self.yaml_overrides,
            # if needed
            dic.update(self.yaml_overrides or {})
            return dic
        self.mock_yaml_load.side_effect = yload  # no-op
        
        #add cleanup (in case tearDown is not called due to exceptions):
        self.addCleanup(Test.cleanup, self)

        self.d_yaml_file = get_templates_fpath("download.yaml")
        self.p_yaml_file = get_templates_fpath("processing.yaml")
        self.p_py_file = get_templates_fpath("processing.yaml")
        
    def run_download(self, *args, **yaml_overrides):
        args = list(args)
        if all(a not in ("-c", "--configfile") for a in args):
            args += ['-c', self.d_yaml_file]
        self.yaml_overrides, _tmp = yaml_overrides, self.yaml_overrides
        runner = CliRunner()
        result = runner.invoke(cli, ['download'] + args)
        self.yaml_overrides = _tmp
        return result
    
    def run_process(self, *args, **yaml_overrides):
        args = list(args)
        if all(a not in ("-c", "--configfile") for a in args):
            args += ['-c', self.p_yaml_file]
        if all(a not in ("-p", "--pyfile") for a in args):
            args += ['-p', self.p_py_file]
        self.yaml_overrides, _tmp = yaml_overrides, self.yaml_overrides
        runner = CliRunner()
        result = runner.invoke(cli, ['process'] + args)
        self.yaml_overrides = _tmp
        return result
        
        
    def test_download_bad_values(self):
        
        result = self.run_download(networks={'a': 'b'})  # invalid type
        assert result.exit_code == 0 # RHAT?? because networks needs to be just an iterable
        # thus providing dict is actually fine and will iterate over its keys:
        assert self.mock_run_download.call_args_list[0][1]['networks'] == ['a']
        # do some asserts only for this case to test how we print the arguments to string:
        assert "tt_table: <type" in result.output
        assert "traveltimes_model:" in result.output
        assert 'dburl: \''+ self.dburl + "'" in result.output
        
        dwnl = self.session.query(Download).first()
        assert dwnl.log  # assert we have something written
        
        result = self.run_download(networks='!*')  # invalid value
        assert result.exit_code != 0
        assert "Error: Invalid value for networks:" in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.session.query(Download).count() == 1
        
        result = self.run_download(start=[])  # invalid type
        assert result.exit_code != 0
        assert "Error: Invalid type for start:" in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.session.query(Download).count() == 1

        result = self.run_download(start='wat')  # invalid value
        assert result.exit_code != 0
        assert "Error: Invalid value for start:" in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.session.query(Download).count() == 1
        
        result = self.run_download(end='wat') # try with end
        assert result.exit_code != 0
        assert "Error: Invalid value for end:" in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.session.query(Download).count() == 1
        
        result = self.run_download(traveltimes_model=[])  # invalid type
        assert result.exit_code != 0
        assert "Error: Invalid type for traveltimes_model:" in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.session.query(Download).count() == 1
        
        result = self.run_download(traveltimes_model='wat')  # invalid value
        assert result.exit_code != 0
        assert "Error: Invalid value for traveltimes_model:" in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.session.query(Download).count() == 1
        

        result = self.run_download(dburl=self.d_yaml_file)  # valid file
        assert result.exit_code != 0
        assert "Error: Invalid value for dburl:" in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.session.query(Download).count() == 1
        
        
        result = self.run_download(dburl="sqlite:/whatever")  # invalid db url
        assert result.exit_code != 0
        assert "Error: Invalid value for dburl:" in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.session.query(Download).count() == 1
        
        result = self.run_download(dburl="sqlite://whatever")  # invalid db url
        assert result.exit_code != 0
        assert "Error: Invalid value for dburl:" in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.session.query(Download).count() == 1
        
#         result = self.run_download(dburl="sqlite:///jngqoryuves__")  # valid db url file
#         assert result.exit_code != 0
#         assert "Error: Invalid value for dburl:" in result.output
#         # assert we did not write to the db, cause the error threw before setting up db:
#         assert self.session.query(Download).count() == 1
        
        result = self.run_download(dburl=[])  # invalid type
        result.output


    def tst_download_bad_types(self):
        '''bad types must be passed directly to download as click does a preliminary check'''
        
        result = self.run_download(networks={'a': 'b'})  # invalid type
        result.output
        
        result = self.run_download(networks='!*')  # invalid value
        result.output
        
        result = self.run_download(start=[])  # invalid type
        result.output

        result = self.run_download(start='wat')  # invalid value
        result.output
        
        result = self.run_download(end='wat') # try with end
        result.output
        
        result = self.run_download(traveltimes_model=[])  # invalid type
        result.output
        
        result = self.run_download(traveltimes_model='wat')  # invalid value
        result.output

        result = self.run_download(dburl=self.d_yaml_file)  # valid file
        result.output
        
        result = self.run_download(dburl="sqlite:/whatever")  # invalid db url
        result.output
        
        result = self.run_download(dburl="sqlite://whatever")  # invalid db url
        result.output
        
        result = self.run_download(dburl="sqlite:///jngqoryuves__")  # invalid db url file
        result.output
        
        result = self.run_download(dburl=[])  # invalid type
        result.output
    def testName(self):
        pass


@contextmanager
def download_setup(filename, **params):
    yamldic = yaml_load(get_templates_fpath(filename))
    for k, v in params.items():
        if v is None:
            yamldic.pop(k, None)
        else:
            yamldic[k] = v
    with NamedTemporaryFile('w') as tf:  # supply 'w' as default is 'w+b'
        name = tf.name
        assert os.path.isfile(name)
        yaml.safe_dump(yamldic, tf)
        yield name, yamldic

    assert not os.path.isfile(name)

@patch("stream2segment.main.configlog4download")
@patch("stream2segment.main.new_db_download")
@patch("stream2segment.main.create_session")
@patch("stream2segment.main.run_download", return_value=0)
def test_click_download(mock_download, mock_create_sess, mock_new_db_download,
                        mock_configlog4download):
    runner = CliRunner()
    # test normal case and arguments.
    with download_setup("download.yaml") as (conffile, yamldic):

        result = runner.invoke(cli, ['download', '-c', conffile])
        dic = mock_download.call_args_list[0][1]
        assert dic['start'] == yamldic['start']
        assert dic['end'] == yamldic['end']
        mock_create_sess.assert_called_once_with(yamldic['dburl'])
        #assert dic['dburl'] == yamldic['dburl']
        assert result.exit_code == 0

        # test by supplying an argument it is overridden
        mock_download.reset_mock()
        mock_create_sess.reset_mock()
        newdate = yamldic['start'] + timedelta(seconds=1)
        result = runner.invoke(cli, ['download', '-c', conffile, '--start', newdate])
        dic = mock_download.call_args_list[0][1]
        assert dic['start'] == newdate
        assert dic['end'] == yamldic['end']
        mock_create_sess.assert_called_once_with(yamldic['dburl'])
        # assert dic['dburl'] == yamldic['dburl']
        assert result.exit_code == 0

        # test by supplying the same argument as string instead of datetime (use end instead of start this time)
        mock_download.reset_mock()
        mock_create_sess.reset_mock()
        result = runner.invoke(cli, ['download', '-c', conffile, '--end', newdate.isoformat()])
        dic = mock_download.call_args_list[0][1]
        assert dic['end'] == newdate
        assert dic['start'] == yamldic['start']
        mock_create_sess.assert_called_once_with(yamldic['dburl'])
        # assert dic['dburl'] == yamldic['dburl']
        assert result.exit_code == 0

    # test start and end given as integers
    with download_setup("download.yaml", start=1, end=0) as (conffile, yamldic):
        mock_download.reset_mock()
        mock_create_sess.reset_mock()
        result = runner.invoke(cli, ['download', '-c', conffile])
        dic = mock_download.call_args_list[0][1]
        d = datetime.utcnow()
        startd = datetime(d.year, d.month, d.day) - timedelta(days=1)
        endd = datetime(d.year, d.month, d.day)
        assert dic['start'] == startd
        assert dic['end'] == endd
        mock_create_sess.assert_called_once_with(yamldic['dburl'])
        # assert dic['dburl'] == yamldic['dburl']
        assert result.exit_code == 0

    # test removing an item in the config.yaml this item is not passed to download func
    with download_setup("download.yaml", inventory=None) as (conffile, yamldic):
        mock_download.reset_mock()
        result = runner.invoke(cli, ['download', '-c', conffile])
        dic = mock_download.call_args_list[0][1]
        assert not 'inventory' in dic

    # test with an unknown argument
    with download_setup("download.yaml", inventory=None) as (conffile, yamldic):
        mock_download.reset_mock()
        result = runner.invoke(cli, ['download', '--wtf_is_this_argument_#$%TGDAGRHNBGAWEtqevt3t', 5])
        assert result.exit_code != 0
        assert not mock_download.called


    mock_download.reset_mock()
    result = runner.invoke(cli, ['download', '--help'])
    assert result.exit_code == 0
    assert not mock_download.called


@patch("stream2segment.main.process", return_value=0)
def tst_click_process(mock_process):
    runner = CliRunner()
    d_conffile = get_templates_fpath("download.yaml")
    conffile = get_templates_fpath("processing.yaml")
    pyfile = get_templates_fpath("processing.py")

    # test no dburl supplied
    mock_process.reset_mock()
    result = runner.invoke(cli, ['process', '-c', conffile, '-p', pyfile, 'c'])
    assert "Missing option" in result.output
    assert result.exc_info
    
    # test dburl supplied
    mock_process.reset_mock()
    result = runner.invoke(cli, ['process', '-d', 'd', '-c', conffile, '-p', pyfile, 'c'])
    lst = list(mock_process.call_args_list[0][0])
    assert lst == ['d', pyfile, None, conffile, 'c']
    assert result.exit_code == 0
    
    # test dburl supplied via config
    mock_process.reset_mock()
    result = runner.invoke(cli, ['process', '-d', d_conffile , '-c', conffile, '-p', pyfile, 'c'])
    lst = list(mock_process.call_args_list[0][0])
    assert lst == [yaml_load(d_conffile)['dburl'], pyfile, None, conffile, 'c']
    assert result.exit_code == 0
    
    # test funcname supplied via cli:
    mock_process.reset_mock()
    result = runner.invoke(cli, ['process', '--funcname', 'wat?', '-d', d_conffile , '-c', conffile, '-p', pyfile, 'c'])
    lst = list(mock_process.call_args_list[0][0])
    assert lst == [yaml_load(d_conffile)['dburl'], pyfile, 'wat?', conffile, 'c']
    assert result.exit_code == 0

    # test an error in params: -dburl instead of --dburl:
    mock_process.reset_mock()
    result = runner.invoke(cli, ['process', '-dburl', d_conffile , '-c', conffile, '-p', pyfile, 'c'])
    assert not mock_process.called
    assert result.exit_code != 0

    # assert help works:
    mock_process.reset_mock()
    result = runner.invoke(cli, ['process', '--help'])
    assert not mock_process.called
    assert result.exit_code == 0


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()