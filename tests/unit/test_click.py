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
def download_setup(filename, **params):
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
    with download_setup("download.yaml") as (conffile, yamldic):
        result = runner.invoke(main, ['d', '-c', conffile])
        dic = mock_download.call_args_list[0][1]
        assert dic['start'] == yamldic['start']
        assert dic['end'] == yamldic['end']
        assert dic['dburl'] == yamldic['dburl']
        assert result.exit_code == 0

        # test by supplying an argument it is overridden
        mock_download.reset_mock()
        newdate = yamldic['start'] + timedelta(seconds=1)
        result = runner.invoke(main, ['d', '-c', conffile, '--start', newdate])
        dic = mock_download.call_args_list[0][1]
        assert dic['start'] == newdate
        assert dic['end'] == yamldic['end']
        assert dic['dburl'] == yamldic['dburl']
        assert result.exit_code == 0

        # test by supplying the same argument as string instead of datetime (use end instead of start this time)
        mock_download.reset_mock()
        result = runner.invoke(main, ['d', '-c', conffile, '--end', newdate.isoformat()])
        dic = mock_download.call_args_list[0][1]
        assert dic['end'] == newdate
        assert dic['start'] == yamldic['start']
        assert dic['dburl'] == yamldic['dburl']
        assert result.exit_code == 0

    # test no start no end then current time range is used
    with download_setup("download.yaml", start=None, end=None) as (conffile, yamldic):
        mock_download.reset_mock()
        result = runner.invoke(main, ['d', '-c', conffile])
        dic = mock_download.call_args_list[0][1]
        start, end = get_def_timerange()
        assert dic['start'] == start
        assert dic['end'] == end
        assert dic['dburl'] == yamldic['dburl']
        assert result.exit_code == 0

    # test removing an item in the config.yaml this item is not passed to download func
    with download_setup("download.yaml", inventory=None) as (conffile, yamldic):
        mock_download.reset_mock()
        result = runner.invoke(main, ['d', '-c', conffile])
        dic = mock_download.call_args_list[0][1]
        assert not 'inventory' in dic

    # test with an unknown argument
    with download_setup("download.yaml", inventory=None) as (conffile, yamldic):
        mock_download.reset_mock()
        result = runner.invoke(main, ['d', '--wtf_is_this_argument_#$%TGDAGRHNBGAWEtqevt3t', 5])
        assert result.exit_code != 0
        assert not mock_download.called


    mock_download.reset_mock()
    result = runner.invoke(main, ['d', '--help'])
    assert result.exit_code == 0
    assert not mock_download.called


@patch("stream2segment.main.process", return_value=0)
def test_click_process(mock_process):
    runner = CliRunner()
    # test normal case and arguments.
    with download_setup("download.yaml") as (conffile, yamldic):
        result = runner.invoke(main, ['p', 'a', 'b', 'c'])
        lst = list(mock_process.call_args_list[0][0])
        assert lst == [None, 'a', 'b', 'c']
        assert result.exit_code == 0

        mock_process.reset_mock()
        result = runner.invoke(main, ['p', 'a', 'b', 'c', '-d' ,'d'])
        lst = list(mock_process.call_args_list[0][0])
        assert lst == ['d', 'a', 'b', 'c']
        assert result.exit_code == 0

        mock_process.reset_mock()
        # test an error in params: -dburl instead of --dburl:
        result = runner.invoke(main, ['p', 'a', 'b', 'c', '-dburl' ,'d'])
        assert not mock_process.called
        assert result.exit_code != 0

        mock_process.reset_mock()
        # test an error in params: -dburl instead of --dburl:
        result = runner.invoke(main, ['p', 'a', 'b', 'c', '--dburl' ,'d'])
        lst = list(mock_process.call_args_list[0][0])
        assert lst == ['d', 'a', 'b', 'c']
        assert result.exit_code == 0


        mock_process.reset_mock()
        result = runner.invoke(main, ['p', 'a', 'b', 'c', '--dburl' , conffile])
        lst = list(mock_process.call_args_list[0][0])
        assert lst == [yamldic['dburl'], 'a', 'b', 'c']
        assert result.exit_code == 0

    # assert help works:
    mock_process.reset_mock()
    result = runner.invoke(main, ['p', '--help'])
    assert not mock_process.called
    assert result.exit_code == 0


@patch("stream2segment.main.data_aval", return_value=0)
def test_click_dataaval(mock_da):
    runner = CliRunner()
    # test normal case and arguments.
    mock_da.reset_mock()
    with download_setup("download.yaml") as (conffile, yamldic):
        result = runner.invoke(main, ['a', '-d', 'dburl', 'outfile'])
        lst = list(mock_da.call_args_list[0][0])
        assert lst == ['dburl', 'outfile', 0.5]
        assert result.exit_code == 0

    # test wrong arg.
    mock_da.reset_mock()
    with download_setup("download.yaml") as (conffile, yamldic):
        result = runner.invoke(main, ['a', '-d', 'dburl', 'outfile', '-mm', 0.5])
        assert not mock_da.called
#         assert lst == ['dburl', 'outfile']
        assert result.exit_code != 0

    # test wrong arg.
    mock_da.reset_mock()
    with download_setup("download.yaml") as (conffile, yamldic):
        result = runner.invoke(main, ['a', '--dburl', 'dburl', 'outfile', '-m', 0.4])
        lst = list(mock_da.call_args_list[0][0])
        assert lst == ['dburl', 'outfile', 0.4]
        assert result.exit_code == 0

    # assert help works:
    mock_da.reset_mock()
    result = runner.invoke(main, ['a', '--help'])
    assert not mock_da.called
    assert result.exit_code == 0


@patch("stream2segment.main.visualize", return_value=0)
def tst_click_visualize(mock_process):
    runner = CliRunner()
    # test normal case and arguments.
    with download_setup("download.yaml") as (conffile, yamldic):
        result = runner.invoke(main, ['p', 'a', 'b', 'c'])
        lst = list(mock_process.call_args_list[0][0])
        assert lst == [None, 'a', 'b', 'c']
        assert result.exit_code == 0

        mock_process.reset_mock()
        result = runner.invoke(main, ['p', 'a', 'b', 'c', '-d', 'd'])
        lst = list(mock_process.call_args_list[0][0])
        assert lst == ['d', 'a', 'b', 'c']
        assert result.exit_code == 0

        mock_process.reset_mock()
        # test an error in params: -dburl instead of --dburl:
        result = runner.invoke(main, ['p', 'a', 'b', 'c', '-dburl', 'd'])
        assert not mock_process.called
        assert result.exit_code != 0

        mock_process.reset_mock()
        # test an error in params: -dburl instead of --dburl:
        result = runner.invoke(main, ['p', 'a', 'b', 'c', '--dburl', 'd'])
        lst = list(mock_process.call_args_list[0][0])
        assert lst == ['d', 'a', 'b', 'c']
        assert result.exit_code == 0

        mock_process.reset_mock()
        result = runner.invoke(main, ['p', 'a', 'b', 'c', '--dburl', conffile])
        lst = list(mock_process.call_args_list[0][0])
        assert lst == [yamldic['dburl'], 'a', 'b', 'c']
        assert result.exit_code == 0

    # assert help works:
    mock_process.reset_mock()
    result = runner.invoke(main, ['p', '--help'])
    assert not mock_process.called
    assert result.exit_code == 0





if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()