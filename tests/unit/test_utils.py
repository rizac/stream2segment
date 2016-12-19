#@PydevCodeAnalysisIgnore
'''
Created on Dec 12, 2016

@author: riccardo
'''
import unittest
from mock import patch
from StringIO import StringIO
from stream2segment.utils.url import url_read
import pytest
from urllib2 import URLError
import socket
import mock

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
    class mystringio(object):

        def __init__(self, url, **kwargs):
            mockread.reset_mock()
            if isinstance(url, Exception):
                self.a = url
            else:
                self.a = StringIO(url)

        def read(self, *a, **kw):
            if isinstance(self.a, Exception):
                raise self.a
            mockread(*a, **kw)
            return self.a.read(*a, **kw)

        def close(self, *a, **kw):
            if not isinstance(self.a, Exception):
                self.a.close(*a, **kw)

    mock_urlopen.side_effect = lambda url, **kw: mystringio(url, **kw)
    with pytest.raises(TypeError):
        url_read('', "name")

    val = 'url'
    blockSize = 1024*1024
    assert url_read(val, blockSize) == val
    mock_urlopen.assert_called_with(val)
    assert mockread.call_count == 2
    mockread.assert_called_with(blockSize)

    mock_urlopen.side_effect = lambda url, **kw: mystringio(url, **kw)
    defBlockSize = -1
    assert url_read(val, arg_to_read=56) == val
    mock_urlopen.assert_called_with(val, arg_to_read=56)
    assert mockread.call_count == 1  # because blocksize is -1

    mock_urlopen.side_effect = lambda url, **kw: mystringio(URLError('wat?'))
    with pytest.raises(URLError):
        url_read(val)

    mock_urlopen.side_effect = lambda url, **kw: mystringio(socket.timeout())
    with pytest.raises(socket.error):
        url_read(val)
    
    

# @patch('stream2segment.utils.Request')
# @patch('stream2segment.utils.urlopen')
# def test_url_read(mock_urlopen, mock_urllib_request):  # mock_ul_urlopen, mock_ul_request, mock_ul):
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
#     assert url_read(val, "name") == val
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
#         assert url_read(val, "name") == ''
#         mock_urllib_request.assert_called_with(val)
#         mock_urlopen.assert_called_with(val)
#         assert not mock_urlopen.read.called
# 
#     def xyz3():
#         raise ValueError()
#     mock_urlopen.side_effect = lambda arg: xyz3()
#     assert url_read(val, "name") == ''
#     mock_urllib_request.assert_called_with(val)
#     mock_urlopen.assert_called_with(val)
#     assert not mock_urlopen.read.called
# 
#     def xyz4():
#         raise AttributeError()
#     mock_urlopen.side_effect = lambda arg: xyz4()
#     with pytest.raises(AttributeError):
#         _ = url_read(val, "name")
# 
#     def xyz5(argss):
#         class sio(StringIO):
#             def read(self, *args, **kw):
#                 raise IOError('oops')
#         return sio(argss)
#     mock_urlopen.side_effect = lambda arg: xyz5(arg)
#     assert url_read(val, "name") == ''
#     mock_urllib_request.assert_called_with(val)
#     mock_urlopen.assert_called_with(val)



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()