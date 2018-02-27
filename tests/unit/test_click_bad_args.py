'''
Created on May 23, 2017

@author: riccardo
'''
import unittest
from click.testing import CliRunner
from stream2segment.cli import cli
from mock.mock import patch
from stream2segment.utils.resources import get_templates_fpath, yaml_load
from stream2segment.main import init as orig_init, helpmathiter as main_helpmathiter, download
from tempfile import NamedTemporaryFile
import yaml
from contextlib import contextmanager
import os
from datetime import datetime, timedelta
import tempfile
import shutil

from stream2segment.main import configlog4download as o_config4download, \
    new_db_download as o_new_db_download, create_session as o_create_session, run_download as o_run_download, \
    configlog4processing as o_configlog4processing, to_csv as o_to_csv, process as o_process, \
    download as o_download
from stream2segment.utils.inputargs import yaml_load as o_yaml_load
from stream2segment.io.db.models import Download
from _pytest.capture import capsys


class Test(unittest.TestCase):

    @staticmethod
    def cleanup(me):
        for patcher in me.patchers:
            patcher.stop()
    
#     @property
#     def is_sqlite(self):
#         return str(self.dburl).startswith("sqlite:///")
#     
#     @property
#     def is_postgres(self):
#         return str(self.dburl).startswith("postgresql://")

    def setUp(self):
        # we do not need to specify a db url other than in memory sqlite, as we do not
        # test db stuff here other than checking a download id has written:
        self.dburl = 'sqlite:///:memory:'  # os.getenv("DB_URL", "sqlite:///:memory:")
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
        
        self.patchers.append(patch('stream2segment.utils.inputargs.yaml_load'))
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
        # self.p_yaml_file = get_templates_fpath("processing.yaml")
        self.p_py_file = get_templates_fpath("processing.yaml")
        
    def run_cli_download(self, *args, **yaml_overrides):
        args = list(args)
        if all(a not in ("-c", "--configfile") for a in args):
            args += ['-c', self.d_yaml_file]
        self.yaml_overrides, _tmp = yaml_overrides, self.yaml_overrides
        runner = CliRunner()
        result = runner.invoke(cli, ['download'] + args)
        self.yaml_overrides = _tmp
        return result
    
    def run_cli_process(self, *args, **dburl_yaml_overrides):
        args = list(args)
        if all(a not in ("-d", "--dburl") for a in args):
            args += ['-c', self.d_yaml_file]
        if all(a not in ("-p", "--pyfile") for a in args):
            args += ['-p', self.p_py_file]
        self.yaml_overrides, _tmp = dburl_yaml_overrides, self.yaml_overrides
        runner = CliRunner()
        result = runner.invoke(cli, ['process'] + args)
        self.yaml_overrides = _tmp
        return result

        
    def test_download_bad_values(self):
        '''test different scenarios where the value in the dwonload.yaml are not well formatted'''
        result = self.run_cli_download(networks={'a': 'b'})  # invalid type
        assert result.exit_code == 0 # WHAT?? because networks needs to be just an iterable
        # thus providing dict is actually fine and will iterate over its keys:
        assert self.mock_run_download.call_args_list[0][1]['networks'] == ['a']
        # do some asserts only for this case to test how we print the arguments to string:
        assert "tt_table: <TTTable object>" in result.output
        assert "traveltimes_model:" in result.output
        assert 'dburl: \''+ self.dburl + "'" in result.output
        
        dwnl = self.session.query(Download).first()
        assert dwnl.log  # assert we have something written
        
        result = self.run_cli_download(networks='!*')  # invalid value
        assert result.exit_code != 0
        assert "Error: Invalid value for \"networks\":" in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.session.query(Download).count() == 1
        
        #what about conflicting arguments?
        result = self.run_cli_download(networks='!*', net='opu')  # invalid value
        assert result.exit_code != 0
        assert "Conflicting names \"net\" / \"networks\"" in result.output or \
            "Conflicting names \"networks\" / \"net\"" in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.session.query(Download).count() == 1
        
        result = self.run_cli_download(start=[])  # invalid type
        assert result.exit_code != 0
        assert "Error: Invalid type for \"start\":" in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.session.query(Download).count() == 1

        result = self.run_cli_download(start='wat')  # invalid value
        assert result.exit_code != 0
        assert "Error: Invalid value for \"start\":" in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.session.query(Download).count() == 1
        
        # now test the same as above BUT with the wrong value from the command line:
        result = self.run_cli_download('-t0', 'wat')  # invalid value typed from the command line
        assert result.exit_code != 0
        assert "Error: Invalid value for \"-t0\" / \"--start\":" in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.session.query(Download).count() == 1
        
        result = self.run_cli_download(end='wat') # try with end
        assert result.exit_code != 0
        assert "Error: Invalid value for \"end\":" in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.session.query(Download).count() == 1
        
        # now test the same as above BUT with the wrong value from the command line:
        result = self.run_cli_download('-t1', 'wat')  # invalid value typed from the command line
        assert result.exit_code != 0
        assert "Error: Invalid value for \"-t1\" / \"--end\":" in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.session.query(Download).count() == 1
        
        result = self.run_cli_download(traveltimes_model=[])  # invalid type
        assert result.exit_code != 0
        assert "Error: Invalid type for \"traveltimes_model\":" in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.session.query(Download).count() == 1
        
        result = self.run_cli_download(traveltimes_model='wat')  # invalid value
        assert result.exit_code != 0
        assert "Error: Invalid value for \"traveltimes_model\":" in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.session.query(Download).count() == 1
        
        result = self.run_cli_download(dburl=self.d_yaml_file)  # valid file
        assert result.exit_code != 0
        assert "Error: Invalid value for \"dburl\":" in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.session.query(Download).count() == 1
        
        result = self.run_cli_download(dburl="sqlite:/whatever")  # invalid db url
        assert result.exit_code != 0
        assert "Error: Invalid value for \"dburl\":" in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.session.query(Download).count() == 1
        
        result = self.run_cli_download(dburl="sqlite://whatever")  # invalid db url
        assert result.exit_code != 0
        assert "Error: Invalid value for \"dburl\":" in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.session.query(Download).count() == 1
                
        result = self.run_cli_download(dburl=[])  # invalid type
        assert result.exit_code != 0
        assert "Error: Invalid type for \"dburl\":" in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.session.query(Download).count() == 1
        result.output
        
        # Test an invalif configfile. This can be done only via command line
        result = self.run_cli_download('-c', 'frjkwlag5vtyhrbdd_nleu3kvshg w') 
        assert result.exit_code != 0
        assert "Error: Invalid value for \"-c\" / \"--configfile\":" in result.output
        # assert we did not write to the db, cause the error threw before setting up db:
        assert self.session.query(Download).count() == 1


    def tst_process_bad_types(self):
        '''bad types must be passed directly to download as click does a preliminary check'''
        
        result = self.run_cli_process('--pyfile', 'nrvnkenrgdvf')  # invalid value from cli
        assert result.exit_code != 0
        assert "Error: Invalid type for \"-p\" / \"--pyfile\":" in result.output
        
        
        result = self.run_cli_process(networks='!*')  # invalid value
        result.output
        
        result = self.run_cli_process(start=[])  # invalid type
        result.output

        result = self.run_cli_process(start='wat')  # invalid value
        result.output
        
        # invalid value
        result = self.run_cli_process('t0', 'wat')  # invalid value typed from the command line
        result.output
        
        result = self.run_cli_process(end='wat') # try with end
        result.output
        
        result = self.run_cli_process(traveltimes_model=[])  # invalid type
        result.output
        
        result = self.run_cli_process(traveltimes_model='wat')  # invalid value
        result.output

        result = self.run_cli_process(dburl=self.d_yaml_file)  # valid file
        result.output
        
        result = self.run_cli_process(dburl="sqlite:/whatever")  # invalid db url
        result.output
        
        result = self.run_cli_process(dburl="sqlite://whatever")  # invalid db url
        result.output
        
        result = self.run_cli_process(dburl="sqlite:///jngqoryuves__")  # invalid db url file
        result.output
        
        result = self.run_cli_process(dburl=[])  # invalid type
        result.output

    def testName(self):
        pass


@patch('stream2segment.main.run_download')
def test_download_verbosity(mock_run_download, capsys):
    dburl = 'sqlite:///:memory:' 
    d_yaml_file = get_templates_fpath("download.yaml")
    
    ret = o_download(d_yaml_file, verbosity=0, dburl=dburl)
    out, err = capsys.readouterr()
    assert not out  # assert empty (avoid comparing to strings and potential py2 py3 headache)
    
    ret = o_download(d_yaml_file, verbosity=1, dburl=dburl)
    out, err = capsys.readouterr()
    
    ret = o_download(d_yaml_file, verbosity=2, dburl=dburl)
    out, err = capsys.readouterr()
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()