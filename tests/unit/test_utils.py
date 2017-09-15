#@PydevCodeAnalysisIgnore
'''
Created on Dec 12, 2016

@author: riccardo
'''
from future import standard_library
standard_library.install_aliases()
from builtins import range
from builtins import object
import unittest
from mock import patch
from io import StringIO
from stream2segment.utils.url import urlread, URLException
import pytest
from urllib.error import URLError, HTTPError
import socket
import mock
from stream2segment.utils import secure_dburl, get_progressbar, Nop
from io import BytesIO
from click.termui import progressbar

DEFAULT_TIMEOUT = socket._GLOBAL_DEFAULT_TIMEOUT

class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testName(self):
        pass

@patch('stream2segment.utils.url.urllib2.urlopen')
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

    val = 'url'
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
    

@patch("stream2segment.utils.Nop", side_effect=lambda *a, **v: Nop(*a, **v))
@patch("stream2segment.utils.click_progressbar", side_effect=lambda *a, **v: progressbar(*a, **v))
def test_progressbar(mock_pbar, mock_nop):
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
def test_progressbar_functional():
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

# @patch('stream2segment.utils.Request')
# @patch('stream2segment.utils.urlopen')
# def test_urlread(mock_urlopen, mock_urllib_request):  # mock_ul_urlopen, mock_ul_request, mock_ul):
#     blockSize = 1024*1024
# 
#     mock_urllib_request.side_effect = lambda arx: arx
# 
#     def xyz(argss):
#         return StringIO(argss)
# 
#     # mock_ul.urlopen = Mock()
#     mock_urlopen.side_effect = xyz
#     # mock_ul.urlopen.return_value = lambda arg: StringIO(arg)
# 
#     val = 'url'
#     assert urlread(val, "name") == val
#     mock_urllib_request.assert_called_with(val)
#     mock_urlopen.assert_called_with(val)
#     # mock_ul.urlopen.read.assert_called_with(blockSize)
# 
#     def ioerr(**kwargs):
#         ret = IOError()
#         for key, value in kwargs.iteritems():
#             setattr(ret, key, value)
#         return ret
# 
#     for kwz in [{'reason':'reason'}, {'code': 'code'}, {}]:
#         def xyz2(**kw):
#             raise ioerr(**kw)
# 
#         mock_urlopen.side_effect = lambda arg: xyz2(**kwz)
#         assert urlread(val, "name") == ''
#         mock_urllib_request.assert_called_with(val)
#         mock_urlopen.assert_called_with(val)
#         assert not mock_urlopen.read.called
# 
#     def xyz3():
#         raise ValueError()
#     mock_urlopen.side_effect = lambda arg: xyz3()
#     assert urlread(val, "name") == ''
#     mock_urllib_request.assert_called_with(val)
#     mock_urlopen.assert_called_with(val)
#     assert not mock_urlopen.read.called
# 
#     def xyz4():
#         raise AttributeError()
#     mock_urlopen.side_effect = lambda arg: xyz4()
#     with pytest.raises(AttributeError):
#         _ = urlread(val, "name")
# 
#     def xyz5(argss):
#         class sio(StringIO):
#             def read(self, *args, **kw):
#                 raise IOError('oops')
#         return sio(argss)
#     mock_urlopen.side_effect = lambda arg: xyz5(arg)
#     assert urlread(val, "name") == ''
#     mock_urllib_request.assert_called_with(val)
#     mock_urlopen.assert_called_with(val)



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()