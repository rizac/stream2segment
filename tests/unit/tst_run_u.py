'''
Created on Feb 23, 2016

@author: riccardo
'''

import mock
from run import config_logging
import pytest
import argparse
from itertools import combinations


@pytest.mark.parametrize('verbosity, expected_level',
                         [(-1, 40), (0, 40), (1, 30), (2, 20), (3, 10), (4, 10),])
@mock.patch('run.logging.basicConfig')
def test_config_logging(mock_logging_basicconfig, verbosity, expected_level):
    config_logging(log_verbosity=verbosity)
    assert len(mock_logging_basicconfig.call_args_list) == 1
    basicconfigcall = mock_logging_basicconfig.call_args_list[0]
    assert len(basicconfigcall) == 2 # w=FIXME: check why?!!
    logging_level = basicconfigcall[1]['level']
    assert logging_level == expected_level


@mock.patch('run.dtime')
def test_valid_date(mock_to_datetime):
    from run import valid_date
    mock_to_datetime.return_value = None
    with pytest.raises(argparse.ArgumentTypeError):
        _ = valid_date("anything")
    mock_to_datetime.reset_mock()
    mock_to_datetime.return_value = "x"
    assert valid_date("anything") == "anything"
    


@mock.patch('os.path.exists')
@mock.patch('os.path.isabs')
@mock.patch('os.path.isdir')
@mock.patch('os.makedirs')
@mock.patch('os.path.abspath', return_value= 'abs_path')
def test_existing_directory(mock_abspath, mock_mkdirs, mock_isdir, mock_isabs, mock_exists):
    from run import existing_directory as ed
    for a in [True, False]:
        for b in [True, False]:
            for c in [True, False]:
                mock_exists.reset_mock()
                mock_isabs.reset_mock()
                mock_isdir.reset_mock()
                mock_abspath.reset_mock()
                mock_mkdirs.reset_mock()

                mock_exists.return_value = a
                mock_isabs.return_value = b
                mock_isdir.return_value = c
                
                with pytest.raises(argparse.ArgumentTypeError):
                    _ = ed(None)
                
                ipt = 'x'
                if not c:
                    with pytest.raises(argparse.ArgumentTypeError):
                        _ = ed(ipt)
                    continue
                ed(ipt)
                mock_isabs.assert_called_once_with(ipt)
                assert mock_abspath.called is not mock_isabs.return_value
                new_input = mock_abspath.return_value if mock_abspath.called else ipt
                mock_exists.assert_called_once_with(new_input)
                assert mock_mkdirs.called is not mock_exists.return_value
                if mock_mkdirs.called:
                    mock_mkdirs.assert_called_once_with(new_input)
                mock_isdir.assert_called_once_with(new_input)
                