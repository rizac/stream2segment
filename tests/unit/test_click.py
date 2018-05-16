'''
Created on May 23, 2017

@author: riccardo
'''
import unittest
from click.testing import CliRunner
from stream2segment.cli import cli
from mock.mock import patch
from stream2segment.utils.resources import get_templates_fpath, yaml_load, get_templates_dirpath,\
    get_templates_fpaths
from stream2segment.main import init as orig_init, helpmathiter as main_helpmathiter, show as orig_show
from tempfile import NamedTemporaryFile
import yaml
from contextlib import contextmanager
import os
from datetime import datetime, timedelta
import tempfile
import shutil
import time
import mock
from stream2segment.utils import load_source
import pytest


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
@patch("stream2segment.utils.inputargs.create_session")
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

    # test again start and end given as integers (30 days) FROM THE CLI
    with download_setup("download.yaml") as (conffile, yamldic):
        mock_download.reset_mock()
        mock_create_sess.reset_mock()
        result = runner.invoke(cli, ['download', '-c', conffile, '-s', '30', '-e', '0'])
        dic = mock_download.call_args_list[0][1]
        d = datetime.utcnow()
        startd = datetime(d.year, d.month, d.day) - timedelta(days=30)
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
def test_click_process(mock_process):
    runner = CliRunner()

    d_conffile, conffile, pyfile = \
        get_templates_fpaths("download.yaml", "paramtable.yaml", "paramtable.py")

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


@patch("stream2segment.main.show", side_effect=orig_show)
@patch("stream2segment.gui.main.open_in_browser")
@patch("stream2segment.main.create_main_app")  # , return_value=mock.Mock())
def test_click_show(mock_create_main_app, mock_open_in_browser, mock_show):
    runner = CliRunner()
    d_conffile, conffile, pyfile = \
        get_templates_fpaths("download.yaml", "paramtable.yaml", "paramtable.py")

    # when asserting if we called open_in_browser, since tha latter is inside a thread which
    # executes with a delay of 1.5 seconds, we need to make our function here. Quite hacky,
    # but who cares
    def assert_opened_in_browser(url=None):  # if None, assert
        time.sleep(2)  # to be safe
        mock_open_in_browser.assert_called_once
        args = mock_open_in_browser.call_args_list[0][0]
        assert len(args) == 1
        assert args[0].startswith('http://127.0.0.1:')
    # test no dburl supplied
    mock_show.reset_mock()
    mock_open_in_browser.reset_mock()
    result = runner.invoke(cli, ['show', '-c', conffile, '-p', pyfile])
    assert "Missing option" in result.output
    assert result.exc_info
    assert not mock_open_in_browser.called

    # test dburl supplied
    mock_show.reset_mock()
    mock_open_in_browser.reset_mock()
    result = runner.invoke(cli, ['show', '-d', 'd', '-c', conffile, '-p', pyfile])
    lst = list(mock_show.call_args_list[0][0])
    assert lst == ['d', pyfile, conffile]
    assert result.exit_code == 0
    assert_opened_in_browser('d')

    # test dburl supplied via config
    mock_show.reset_mock()
    mock_open_in_browser.reset_mock()
    result = runner.invoke(cli, ['show', '-d', d_conffile , '-c', conffile, '-p', pyfile])
    lst = list(mock_show.call_args_list[0][0])
    dburl = yaml_load(d_conffile)['dburl']
    assert lst == [dburl, pyfile, conffile]
    assert result.exit_code == 0
    assert_opened_in_browser(dburl)

    # test an error in params: -dburl instead of --dburl:
    mock_show.reset_mock()
    mock_open_in_browser.reset_mock()
    result = runner.invoke(cli, ['show', '-dburl', d_conffile , '-c', conffile, '-p', pyfile])
    assert not mock_show.called
    assert result.exit_code != 0
    assert not mock_open_in_browser.called

    # assert help works:
    mock_show.reset_mock()
    mock_open_in_browser.reset_mock()
    result = runner.invoke(cli, ['show', '--help'])
    assert not mock_show.called
    assert result.exit_code == 0
    assert not mock_open_in_browser.called



@patch("stream2segment.cli.click.prompt")
@patch("stream2segment.main.init", side_effect=orig_init)
def test_click_template(mock_main_init, mock_click_prompt):  #, mock_isfile, mock_copy2):
    runner = CliRunner()
    # assert help works:
    result = runner.invoke(cli, ['init', '--help'])
    assert not mock_main_init.called
    assert result.exit_code == 0

    expected_files = ['download.yaml', 'paramtable.py', 'paramtable.yaml',
                      'save2fs.py', 'save2fs.yaml']

    with runner.isolated_filesystem() as dir_:

        # FIXME: how to check click prompt?? is there a way?
        path = os.path.join(dir_, 'abc')

        def max_mod_time():
            return max(os.path.getmtime(os.path.join(path, f)) for f in os.listdir(path))

        result = runner.invoke(cli, ['init', path])
        # FIXME: check how to mock os.path.isfile properly. This doesnot work:
        # assert mock_isfile.call_count == 5
        assert result.exit_code == 0
        assert mock_main_init.called
        files = os.listdir(path)
        assert sorted(files) == expected_files
        assert not mock_click_prompt.called

        # assert we correctly wrote the files
        for fle in files:
            sourcepath = get_templates_fpath(fle)
            destpath = os.path.join(path, fle)
            if os.path.splitext(fle)[1] == '.yaml':
                # check loaded yaml, which also assures our templates are well formed:
                sourceconfig = yaml_load(sourcepath)
                destconfig = yaml_load(destpath)
                assert sorted(sourceconfig.keys()) == sorted(destconfig.keys())
                for key in sourceconfig.keys():
                    assert type(sourceconfig[key]) == type(destconfig[key])
            elif os.path.splitext(fle)[1] == '.py':
                # check loaded python modules, which also assures our templates are well formed:
                sourcepy = load_source(sourcepath)
                destpy = load_source(destpath)
                sourcekeys = [a for a in dir(sourcepy)]
                destkeys = [a for a in dir(destpy)]
                assert sorted(sourcekeys) == sorted(destkeys)
                for key in sourcekeys:
                    assert type(getattr(sourcepy, key)) == type(getattr(destpy, key))
            else:
                raise ValueError('There should be only python files or yaml files in %s' %
                                 get_templates_dirpath())

        # try to write to the same dir (1)
        mock_click_prompt.reset_mock()
        mock_click_prompt.side_effect = lambda arg: '1'  # overwrite all files
        maxmodtime = max_mod_time()
        # we'll test that files are modified, but on mac timestamps are rounded to seconds
        # so wait 1 second to be safe
        time.sleep(1)
        result = runner.invoke(cli, ['init', path])
        assert mock_click_prompt.called
        assert max_mod_time() > maxmodtime
        assert '%d file(s) copied in' % len(expected_files) in result.output

        # try to write to the same dir (2)
        for click_prompt_ret_val in ('', '2'):
            # '' => skip overwrite
            # '2' => overwrite only non existing
            # in thus case, both the above returned values produce the same result
            mock_click_prompt.reset_mock()
            mock_click_prompt.side_effect = lambda arg: click_prompt_ret_val
            maxmodtime = max_mod_time()
            time.sleep(1)  # see comment above
            result = runner.invoke(cli, ['init', path])
            assert mock_click_prompt.called
            assert max_mod_time() == maxmodtime
            assert 'No file copied' in result.output

        os.remove(os.path.join(path, expected_files[0]))
        # try to write to the same dir (2)
        mock_click_prompt.reset_mock()
        mock_click_prompt.side_effect = lambda arg: '2'   # overwrite non-existing (1) file
        maxmodtime = max_mod_time()
        time.sleep(1)  # see comment above
        result = runner.invoke(cli, ['init', path])
        assert mock_click_prompt.called
        assert max_mod_time() > maxmodtime
        assert '1 file(s) copied in' in result.output


def test_click_template_realcopy():
    '''test a real example of copying files to a tempdir that will be removed'''
    runner = CliRunner()
    with runner.isolated_filesystem() as mydir:
        result = runner.invoke(cli, ['init', mydir])
        filez = os.listdir(mydir)
        assert "download.yaml" in filez
        assert "paramtable.yaml" in filez
        # assert "gui.yaml" in filez
        assert "paramtable.py" in filez
        # assert "gui.py" in filez


    # assert help works:
    assert result.exit_code == 0


# THIS HAS TO BE IMPLEMENTED (if we set the datareport html in place)
@patch("stream2segment.main.helpmathiter", side_effect=main_helpmathiter)
def test_click_funchelp(mock_helpmathiter):
    runner = CliRunner()
    command = 'mathinfo'
    # simply assert it does not raise
    result = runner.invoke(cli, ['utils', command, '-t', 'all', '-f', 'cumsum'])
    assert result.exit_code == 0

    result2 = runner.invoke(cli, ['utils', command, '-t', 'obspy'])
    assert result2.exit_code == 0
    assert len(result2.output) > len(result.output)

    result3 = runner.invoke(cli, ['utils', command, '-t', 'numpy'])
    assert result3.exit_code == 0

    result4 = runner.invoke(cli, ['utils', command, '-t', 'all'])
    assert result4.exit_code == 0
    assert len(result4.output) > len(result3.output)
    assert len(result4.output) > len(result2.output)

    result4 = runner.invoke(cli, ['utils', command, '-t', 'all', '-f', 'nfiwruhfnhgvcfwa___qrfwv'])
    assert result4.exit_code == 0
    assert len(result4.output) == 0


# THIS HAS TO BE IMPLEMENTED (if we set the datareport html in place)
@patch("stream2segment.main.dinfo", return_value=0)
def test_click_dataaval(mock_da):
    
    runner = CliRunner()
     # assert help works:
    mock_da.reset_mock()
    result = runner.invoke(cli, ['utils', 'dinfo', '--help'])
    assert not mock_da.called
    assert result.exit_code == 0

    # do a little test with variable length download ids

    result = runner.invoke(cli, ['utils', 'dinfo', '-d', 'dburl', '-did', 1, '-did', 2])
    lst = list(mock_da.call_args_list[-1][0])
    assert lst == ['dburl', (1, 2), 0.5, False, None]
    assert result.exit_code == 0
    
    result = runner.invoke(cli, ['utils', 'dinfo', '-d', 'dburl'])
    lst = list(mock_da.call_args_list[-1][0])
    assert lst == ['dburl', None, 0.5, False, None]
    assert result.exit_code == 0
    
    result = runner.invoke(cli, ['utils', 'dinfo', '-d', 'dburl', '-g', 0.77, '--html', 'abc'])
    lst = list(mock_da.call_args_list[-1][0])
    assert lst == ['dburl', None, 0.77, True, 'abc']
    assert result.exit_code == 0

    mock_da.reset_mock()
    result = runner.invoke(cli, ['utils', 'dinfo', '-d', 'dburl', '-g', 'a'])
    assert not mock_da.called
    assert result.exit_code != 0
    
    mock_da.reset_mock()
    result = runner.invoke(cli, ['utils', 'dinfo', '-d', 'dburl', '-did', 'a'])
    assert not mock_da.called
    assert result.exit_code != 0
