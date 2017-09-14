'''
Created on May 23, 2017

@author: riccardo
'''
import unittest
from click.testing import CliRunner
from stream2segment.main import main, get_def_timerange
from mock.mock import patch
from stream2segment.utils.resources import get_templates_fpath, yaml_load
from tempfile import NamedTemporaryFile
import yaml
from contextlib import contextmanager
import os
from datetime import datetime, timedelta
import tempfile
import shutil


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
    d_conffile = get_templates_fpath("download.yaml")
    conffile = get_templates_fpath("processing.yaml")
    pyfile = get_templates_fpath("processing.py")

    # test no dburl supplied
    mock_process.reset_mock()
    result = runner.invoke(main, ['p', '-c', conffile, '-p', pyfile, 'c'])
    assert "Missing option" in result.output
    assert result.exc_info
    
    # test dburl supplied
    mock_process.reset_mock()
    result = runner.invoke(main, ['p', '-d', 'd', '-c', conffile, '-p', pyfile, 'c'])
    lst = list(mock_process.call_args_list[0][0])
    assert lst == ['d', pyfile, conffile, 'c']
    assert result.exit_code == 0
    
    # test dburl supplied via config
    mock_process.reset_mock()
    result = runner.invoke(main, ['p', '-d', d_conffile , '-c', conffile, '-p', pyfile, 'c'])
    lst = list(mock_process.call_args_list[0][0])
    assert lst == [yaml_load(d_conffile)['dburl'], pyfile, conffile, 'c']
    assert result.exit_code == 0

    # test an error in params: -dburl instead of --dburl:
    mock_process.reset_mock()
    result = runner.invoke(main, ['p', '-dburl', d_conffile , '-c', conffile, '-p', pyfile, 'c'])
    assert not mock_process.called
    assert result.exit_code != 0

    # assert help works:
    mock_process.reset_mock()
    result = runner.invoke(main, ['p', '--help'])
    assert not mock_process.called
    assert result.exit_code == 0


@patch("stream2segment.main.visualize", return_value=0)
def test_click_visualize(mock_visualize):
    runner = CliRunner()
    d_conffile = get_templates_fpath("download.yaml")
    conffile = get_templates_fpath("gui.yaml")
    pyfile = get_templates_fpath("gui.py")

    # test no dburl supplied
    mock_visualize.reset_mock()
    result = runner.invoke(main, ['v', '-c', conffile, '-p', pyfile])
    assert "Missing option" in result.output
    assert result.exc_info
    
    # test dburl supplied
    mock_visualize.reset_mock()
    result = runner.invoke(main, ['v', '-d', 'd', '-c', conffile, '-p', pyfile])
    lst = list(mock_visualize.call_args_list[0][0])
    assert lst == ['d', pyfile, conffile]
    assert result.exit_code == 0
    
    # test dburl supplied via config
    mock_visualize.reset_mock()
    result = runner.invoke(main, ['v', '-d', d_conffile , '-c', conffile, '-p', pyfile])
    lst = list(mock_visualize.call_args_list[0][0])
    assert lst == [yaml_load(d_conffile)['dburl'], pyfile, conffile]
    assert result.exit_code == 0

    # test an error in params: -dburl instead of --dburl:
    mock_visualize.reset_mock()
    result = runner.invoke(main, ['v', '-dburl', d_conffile , '-c', conffile, '-p', pyfile])
    assert not mock_visualize.called
    assert result.exit_code != 0

    # assert help works:
    mock_visualize.reset_mock()
    result = runner.invoke(main, ['v', '--help'])
    assert not mock_visualize.called
    assert result.exit_code == 0

from stream2segment.main import create_templates as orig_ct
@patch("stream2segment.main.shutil.copy2")
@patch("stream2segment.main.os.path.isfile")
@patch("stream2segment.main.create_templates", side_effect = lambda outdir, prompt, *files: orig_ct(outdir, False, *files))
def test_click_template(mock_create_templates, mock_isfile, mock_copy2):
    mock_isfile.side_effect = lambda *a, **v: True

    runner = CliRunner()
    with runner.isolated_filesystem():

        # a REALLY STUPID TEST. WE SHOULD ASSERT MORE STUFF.
        # btw: how to check click prompt?? is there a way?
        result = runner.invoke(main, ['t', 'abc'])
        # FIXME: check how to mock os.path.isfile properly. This doesnot work:
        # assert mock_isfile.call_count == 5
        assert result.exit_code == 0

        # assert help works:
        mock_create_templates.reset_mock()
        mock_isfile.reset_mock()
        mock_copy2.reset_mock()
        result = runner.invoke(main, ['t', '--help'])
        assert not mock_create_templates.called
        assert result.exit_code == 0


def test_click_template_realcopy():
    '''test a real example of copying files to a tempdir that will be removed'''
    runner = CliRunner()

    runner = CliRunner()
    with runner.isolated_filesystem() as mydir:
        result = runner.invoke(main, ['t', mydir])
        filez = os.listdir(mydir)
        assert "download.yaml" in filez
        assert "processing.yaml" in filez
        assert "gui.yaml" in filez
        assert "processing.py" in filez
        assert "gui.py" in filez


    # assert help works:
    assert result.exit_code == 0

# THIS HAS TO BE IMPLEMENTED (if we set the datareport html in place)
@patch("stream2segment.main.data_aval", return_value=0)
def tst_click_dataaval(mock_da):
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








if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()