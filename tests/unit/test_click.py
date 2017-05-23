'''
Created on May 23, 2017

@author: riccardo
'''
import unittest
from click.testing import CliRunner
from stream2segment.main import main, get_def_timerange
from mock.mock import patch
from stream2segment.utils import yaml_load
from stream2segment.utils.resources import get_templates_fpath
from tempfile import NamedTemporaryFile
import yaml
from contextlib import contextmanager
import os
from datetime import datetime, timedelta


class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testName(self):
        pass


@contextmanager
def _tmp_file(filename, **params):
    yamldic = yaml_load(get_templates_fpath(filename))
    for k, v in params.iteritems():
        if v is None:
            yamldic.pop(k, None)
        else:
            yamldic[k] = v
    with NamedTemporaryFile() as tf:
        name = tf.name
        assert os.path.isfile(name)
        yaml.safe_dump(yamldic, tf)
        yield name, yamldic

    assert not os.path.isfile(name)


@patch("stream2segment.main.download", return_value=0)
def test_click_download(mock_download):
    runner = CliRunner()
    # test normal case and arguments.
    with _tmp_file("download.yaml") as (conffile, yamldic):
        result = runner.invoke(main, ['d', '-c', conffile])
        dic = mock_download.call_args_list[0][1]
        assert dic['start'] == yamldic['start']
        assert dic['end'] == yamldic['end']
        assert dic['dburl'] == yamldic['dburl']
        assert result.exit_code == 0
        
        # test by supplying an argument it is overridden
        mock_download.reset_mock()
        newdate = yamldic['start'] + timedelta(seconds=1)
        result = runner.invoke(main, ['d', '-c', conffile, 'start', newdate])
        dic = mock_download.call_args_list[0][1]
        assert dic['start'] == newdate
        assert dic['end'] == yamldic['end']
        assert dic['dburl'] == yamldic['dburl']
        assert result.exit_code == 0

    # test no start no end then current time range is used
    with _tmp_file("download.yaml", start=None, end=None) as (conffile, yamldic):
        mock_download.reset_mock()
        result = runner.invoke(main, ['d', '-c', conffile])
        dic = mock_download.call_args_list[0][1]
        start, end = get_def_timerange()
        assert dic['start'] == start
        assert dic['end'] == end
        assert dic['dburl'] == yamldic['dburl']
        assert result.exit_code == 0
    
    # test removing an item in the config.yaml this item is not passed to download func
    with _tmp_file("download.yaml", inventory=None) as (conffile, yamldic):
        mock_download.reset_mock()
        result = runner.invoke(main, ['d', '-c', conffile])
        dic = mock_download.call_args_list[0][1]
        start, end = get_def_timerange()
        assert not 'start' in dic


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()