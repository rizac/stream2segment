#@PydevCodeAnalysisIgnore
'''
Created on Dec 12, 2016

@author: riccardo
'''
from future import standard_library
from itertools import product
standard_library.install_aliases()

from datetime import datetime, timedelta
from builtins import range
from builtins import object
from builtins import str
import unittest
from mock import patch
from io import StringIO
from stream2segment.utils.url import urlread, URLException
import pytest
from urllib.error import URLError, HTTPError
import socket
import mock
from stream2segment.utils import secure_dburl, get_progressbar, Nop, strconvert, strptime
from io import BytesIO
from click.termui import progressbar

DEFAULT_TIMEOUT = socket._GLOBAL_DEFAULT_TIMEOUT

# class Test(unittest.TestCase):
# 
# 
#     def setUp(self):
#         pass
# 
# 
#     def tearDown(self):
#         pass
# 
# 
#     def testName(self):
#         pass


def test_strconvert():
    strings =       ["%", "_", "*", "?", ".*", "."]
    
    # sql 2 wildcard
    expected =  ["*", "?", "*", "?", ".*", "."] 
    for a, exp in zip(strings, expected):
        assert strconvert.sql2wild(a) == exp
        assert strconvert.sql2wild(a+a) == exp+exp
        assert strconvert.sql2wild("a"+a) == "a"+exp
        assert strconvert.sql2wild(a+"a") == exp+"a"
        
    # wildcard 2 sql
    expected =  ["%", "_", "%", "_", ".%", "."] 
    for a, exp in zip(strings, expected):
        assert strconvert.wild2sql(a) == exp
        assert strconvert.wild2sql(a+a) == exp+exp
        assert strconvert.wild2sql("a"+a) == "a"+exp
        assert strconvert.wild2sql(a+"a") == exp+"a"

    # sql 2 regex
    expected =  [".*", ".", "\\*", "\\?", "\\.\\*", "\\."]
    for a, exp in zip(strings, expected):
        assert strconvert.sql2re(a) == exp
        assert strconvert.sql2re(a+a) == exp+exp
        assert strconvert.sql2re("a"+a) == "a"+exp
        assert strconvert.sql2re(a+"a") == exp+"a"
        
    # wild 2 regex    
    expected =  ["\\%", "_", ".*", ".", "\\..*", "\\."]
    for a, exp in zip(strings, expected):
        assert strconvert.wild2re(a) == exp
        assert strconvert.wild2re(a+a) == exp+exp
        assert strconvert.wild2re("a"+a) == "a"+exp
        assert strconvert.wild2re(a+"a") == exp+"a"
    


@patch('stream2segment.utils.url.urllib.request.urlopen')
def test_utils_url_read(mock_urlopen):  # mock_ul_urlopen, mock_ul_request, mock_ul):
    

    def side_effect(argss):
        return StringIO(argss)
    
    mockread = mock.Mock()
    class mybytesio(object):

        def __init__(self, url, **kwargs):
            mockread.reset_mock()
            if isinstance(url, Exception):
                self.a = url
            else:
                self.code = 200
                self.msg = 'Ok'
                self.a = BytesIO(url)

        def read(self, *a, **kw):
            if isinstance(self.a, Exception):
                raise self.a
            mockread(*a, **kw)
            return self.a.read(*a, **kw)

        def close(self, *a, **kw):
            if not isinstance(self.a, Exception):
                self.a.close(*a, **kw)

    mock_urlopen.side_effect = lambda url, **kw: mybytesio(url, **kw)
    with pytest.raises(TypeError):
        urlread('', "name")

    val = b'url'
    blockSize = 1024*1024
    assert urlread(val, blockSize)[0] == val
    mock_urlopen.assert_called_with(val)  # , timeout=DEFAULT_TIMEOUT)
    assert mockread.call_count == 2
    mockread.assert_called_with(blockSize)

    mock_urlopen.side_effect = lambda url, **kw: mybytesio(url, **kw)
    defBlockSize = -1
    assert urlread(val, arg_to_read=56)[0] == val
    mock_urlopen.assert_called_with(val, arg_to_read=56)  #, timeout=DEFAULT_TIMEOUT)
    assert mockread.call_count == 1  # because blocksize is -1

    mock_urlopen.side_effect = lambda url, **kw: mybytesio(URLError('wat?'))
    with pytest.raises(URLError):
        urlread(val, wrap_exceptions=False)  # note urlexc
    with pytest.raises(URLException):
        urlread(val, wrap_exceptions=True)  # note urlexc

    mock_urlopen.side_effect = lambda url, **kw: mybytesio(URLError('wat?'))
    with pytest.raises(URLException):
        urlread(val)  # note urlexc

    mock_urlopen.side_effect = lambda url, **kw: mybytesio(socket.timeout())
    with pytest.raises(URLException):
        urlread(val)  # note urlexc
    
    mock_urlopen.side_effect = lambda url, **kw: mybytesio(HTTPError('url', 500, '?', None, None))
    with pytest.raises(URLException):
        urlread(val)  # note urlexc
        
    mock_urlopen.side_effect = lambda url, **kw: mybytesio(HTTPError('url', 500, '?', None, None))
    assert urlread(val, raise_http_err=False) == (None, 500, '?')  # note urlexc
    

@pytest.mark.parametrize('input, expected_result, ',
                          [
                           ("postgresql://scott:@localhost/mydatabase", "postgresql://scott:***@localhost/mydatabase"),
                           ("postgresql://scott:tiger@localhost/mydatabase", "postgresql://scott:***@localhost/mydatabase"),
                           ('postgresql+psycopg2://scott:tiger@localhost/mydatabase', 'postgresql+psycopg2://scott:***@localhost/mydatabase'),
                           ('postgresql+pg8000://scott:tiger@localhost/mydatabase', 'postgresql+pg8000://scott:***@localhost/mydatabase'),
                           ('mysql://scott:tiger@localhost/foo', 'mysql://scott:***@localhost/foo'),
                           ('mysql+mysqldb://scott:tiger@localhost/foo', 'mysql+mysqldb://scott:***@localhost/foo'),
                           ('sqlite:////absolute/path/to/foo.db', 'sqlite:////absolute/path/to/foo.db')
                           ],
                        )
def test_secure_dburl(input, expected_result):
    assert secure_dburl(input) == expected_result

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
### IMPORTANT=======================================================================================
### THE FOLLOWING TESTS INVOLVING PROGRESSBARS PRINTOUT    
### WILL FAIL IN PYDEV 5.2.0 and PYTHON 3.6.2 (typical bytes vs string error)
### RUN FROM TERMINAL
### IMPORTANT=======================================================================================
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

@pytest.mark.skip(reason="fails if run from within n eclipse because of cryptic bytes vs string propblem")
@patch("stream2segment.utils.Nop", side_effect=lambda *a, **v: Nop(*a, **v))
@patch("stream2segment.utils.click_progressbar", side_effect=lambda *a, **v: progressbar(*a, **v))
def test_progressbar(mock_pbar, mock_nop):
    '''this test has problems with eclipse'''
    N = 5
    with get_progressbar(False) as bar:  # no-op
        for i in range(N):
            bar.update(i)
    assert mock_nop.call_count == 1
    assert mock_pbar.call_count == 0
    
    with get_progressbar(False, length=0) as bar: # no-op
        for i in range(N):
            bar.update(i)
    assert mock_nop.call_count == 2
    assert mock_pbar.call_count == 0
    
    with get_progressbar(False, length=10) as bar: # normal progressbar
        for i in range(N):
            bar.update(i)
    assert mock_nop.call_count == 3
    assert mock_pbar.call_count == 0
    
    with get_progressbar(True, length=0) as bar: # normal progressbar
        for i in range(N):
            bar.update(i)
    assert mock_nop.call_count == 4
    assert mock_pbar.call_count == 0
    
    with get_progressbar(True, length=10) as bar: # normal progressbar
        for i in range(N):
            bar.update(i)
    assert mock_nop.call_count == 4
    assert mock_pbar.call_count == 1


# same as above, but we run for safety the real classes (not mocked)
@pytest.mark.skip(reason="fails if run from within n eclipse because of cryptic bytes vs string propblem")
def test_progressbar_functional():
    '''this test has problems with eclipse'''
    N = 5
    with get_progressbar(False) as bar:  # no-op
        for i in range(N):
            bar.update(i)
    
    with get_progressbar(False, length=0) as bar: # no-op
        for i in range(N):
            bar.update(i)
    
    with get_progressbar(False, length=10) as bar: # normal progressbar
        for i in range(N):
            bar.update(i)
    
    with get_progressbar(True, length=0) as bar: # normal progressbar
        for i in range(N):
            bar.update(i)
    
    with get_progressbar(True, length=10) as bar: # normal progressbar
        for i in range(N):
            bar.update(i)


@pytest.mark.parametrize('str_input, expected_diff, ',
                          [
                           ("2016-01-01", timedelta(minutes=60)),
                           ("2016-01-01T01:11:15", timedelta(minutes=60)),
                           ("2016-01-01 01:11:15", timedelta(minutes=60)),
                           ("2016-01-01T01:11:15.556734", timedelta(minutes=60)),
                           ("2016-01-01 01:11:15.556734", timedelta(minutes=60)),
                           ("2016-07-01", timedelta(minutes=120)),
                           ("2016-07-01T01:11:15", timedelta(minutes=120)),
                           ("2016-07-01 01:11:15", timedelta(minutes=120)),
                           ("2016-07-01T01:11:15.431778", timedelta(minutes=120)),
                           ("2016-07-01 01:11:15.431778", timedelta(minutes=120)),
                           ],
                        )
def test_strptime(str_input, expected_diff):
    
    if ":" in str_input:
        arr = [str_input, str_input + 'UTC', str_input+'Z', str_input+'CET']
    else:
        arr = [str_input]
    for ds1, ds2 in product(arr, arr):
        
        d1 = strptime(ds1)
        d2 = strptime(ds2)

        if ds1[-3:] == 'CET' and not ds2[-3:] == 'CET':
            # ds1 was CET, it means that d1 (which is UTC) is one hour less than d2 (which is UTC)
            assert d1 == d2 - expected_diff
        elif ds2[-3:] == 'CET' and not ds1[-3:] == 'CET':
            # ds2 was CET, it means that d2 (which is UTC) is one hour less than d1 (which is UTC)
            assert d2 == d1 - expected_diff
        else:
            assert d1 == d2
        assert d1.tzinfo is None and d2.tzinfo is None
        assert strptime(d1) == d1
        assert strptime(d2) == d2
        
    
    # test a valueerror:
    if ":" not in str_input:
        for dtimestr in [str_input+'Z', str_input+'CET']:
            with pytest.raises(ValueError):
                strptime(dtimestr)
                
    # test type error:
    with pytest.raises(TypeError):
        strptime(5)
    


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
