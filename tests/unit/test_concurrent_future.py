#@PydevCodeAnalysisIgnore
'''
Created on Nov 18, 2016

@author: riccardo
'''
import unittest


import concurrent.futures
import time
import threading
import urllib2
import httplib
import mock
# from stream2segment import async
from stream2segment.utils.url import _ismainthread, read_async
from mock import patch
from itertools import product

class Test(unittest.TestCase):


    def setUp(self):
        self.urls = ["http://sdgfjvkherkdfvsffd",
                     "http://www.google.com", 
#                      "http://www.apple.com",
#                      "http://www.microsoft.com",
#                      "http://www.amazon.com",
#                      "http://www.facebook.com"
                    ]
        self.thread = threading.current_thread()
        
        self.successes = []
        self.errors = []
        self.onsuccess_return_value = None
        self.onerror_return_value = None
        
        self.onsuccess = mock.Mock()
        self.onsuccess.side_effect = self._onsuccess
        self.onerror = mock.Mock()
        self.onerror.side_effect = self._onerror
        
        
    def _onsuccess(self, data, url, index):
        assert _ismainthread()
        self.successes.append(data)
        return self.onsuccess_return_value

    def _onerror(self, error, url, index):
        assert _ismainthread()
        self.errors.append(error)
        return self.onerror_return_value

    def tearDown(self):
        self.successes = []
        self.errors = []
        pass


    @patch('stream2segment.utils.url.urllib2.urlopen')
    def test_mocking_urlread(self, mock_urlopen):
        """Tests onsuccess. WE mock urllib2urlopen.read to return user defined strings"""
        
        data = ['none', '', 'google', '']  # supply an empty string otherwise urllib.read does not stop
        a = mock.Mock()
        a.read.side_effect = data  # returns each item in list
        mock_urlopen.return_value = a

        # self.urls has a valid url (which should execute onsuccess) and an invalid one
        # which should execute onerror)
        read_async(self.urls, self.onsuccess, self.onerror)

        assert len(self.successes) == 2
        assert sorted(data) == ['', ''] + sorted(self.successes)  # sort them as the order might differ
        assert a.read.call_count == len(data)

    @patch('stream2segment.utils.url.urllib2.urlopen')
    def test_urlerrors(self, mock_urlopen):
        """Tests onerror. WE mock urllib2urlopen.read to raise an excpected Exception"""
        
        data = ['none', '', 'google', '']  # supply an empty string otherwise urllib.read does not stop
        a = mock.Mock()
        def _(*a, **v):
            raise urllib2.URLError("")
        a.read.side_effect = urllib2.URLError("")  # raises it
        mock_urlopen.return_value = a

        # self.urls has a valid url (which should execute onsuccess) and an invalid one
        # which should execute onerror)
        read_async(self.urls, self.onsuccess, self.onerror)

        assert len(self.errors) == 2
        assert a.read.call_count == len(self.urls)


    @patch('stream2segment.utils.url.urllib2.urlopen')
    def test_onerror_onsuccess_returning_false(self, mock_urlopen):
        """Tests onerror onsuccess returning False, i.e. onsuccess must be called once"""
        
        data = ['none', '', 'google', '']  # supply an empty string otherwise urllib.read does not stop
        a = mock.Mock()
        a.read.side_effect = data
        mock_urlopen.return_value = a

        self.onsuccess_return_value = self.onerror_return_value = False
        # self.urls has a valid url (which should execute onsuccess) and an invalid one
        # which should execute onerror)
        read_async(self.urls, self.onsuccess, self.onerror)

        assert len(self.successes) == 1  # or alternatively:
        assert self.onsuccess.call_count == 1
        
    @patch('stream2segment.utils.url.urllib2.urlopen')
    def test_onerror_onsuccess_returning_false2(self, mock_urlopen):
        """Tests onerror onsuccess returning False, i.e. the they must be called N times, where
        N < len(self.urls). WE mock urllib2urlopen.read with a time to wait in order to be sure
        that urllib2urlopen.read is also called LESS times than len(self.urls)"""

        # I want to test the above, PLUS that urllib2.urlopen.read is not called the total
        # amount of time. For that, let's the working thread make some work, otherwise we do not
        # see the difference as all wrkers have finished before calling the first 'onsuccess'

        
        # increase the number of url. Doesnt matter their names, out an int:
        urls = [str(i) for i in xrange(10)]

        # we cannot set assert urlib.read.call_count < expected_urllib_read_call_count
        # BECAUSE it might be equal (if all read are executed before first call to onsuccess)
        # On the other hand, testing that urlib.read.call_count <= expected_urllib_read_call_count
        # returns True also if they are equal, which does not assures me the onsuccess function
        # return value + threading works as expected. So we do some heuristic: we set a combination
        # of urls each and urls lengths. We assert that for  *at least one* holds:
        # urlib.read.call_count < expected_urllib_read_call_count
        how_many_one_strictly_lower_than = 0
        for blocksize, reads_per_url in product([1, 1024], [1, 10, 100]):
            # needs this to rest:
            self.setUp()
            
            # set the expected times we weill call urllib2.read:
            expected_urllib_read_call_count = len(urls) * reads_per_url
            # build return values which satisfy: for each url, read must be called reads_per_url
            # times
            data = len(urls) * (['x' * blocksize] * (reads_per_url-1) + [''])  # supply an empty string otherwise urllib.read does not stop
            urllib2_urlopen_read = mock.Mock()
            urllib2_urlopen_read.read.side_effect = data  # _read
            mock_urlopen.return_value = urllib2_urlopen_read
    
            self.onsuccess_return_value = self.onerror_return_value = False
            # self.urls has a valid url (which should execute onsuccess) and an invalid one
            # which should execute onerror)
            read_async(urls, self.onsuccess, self.onerror, blocksize=blocksize)
    
            assert self.onsuccess.call_count == 1
            assert urllib2_urlopen_read.read.call_count <= expected_urllib_read_call_count
            if urllib2_urlopen_read.read.call_count < expected_urllib_read_call_count:
                how_many_one_strictly_lower_than +=1

        assert how_many_one_strictly_lower_than > 0


    @patch('stream2segment.utils.url.urllib2.urlopen')
    def test_string_perf(self, mock_urlopen):
        """tests string concat performances (result: ininfluent compared to time taken for urllib connection)"""
        
        from contextlib import closing
        import array
        import cStringIO
        
        # read the page google.com once. Thus we are less dependent from urllib performances
        # and we can measure the string ones more precisely
        import urllib  # hack: if we import urllib2.urlopen it is the mocked class
        url = "http://www.google.com"
        blocksize= 10  # 1024*1024
        ret = b''
        with closing(urllib.urlopen(url)) as conn:
           if blocksize < 0:  # https://docs.python.org/2.4/lib/bltin-file-objects.html
               ret = conn.read()
           else:
               while True:
                   buf = conn.read(blocksize)
                   if not buf:
                       break
                   ret += buf

        def _(*a, **v):
            ret_ = cStringIO.StringIO(ret)
            ret_.seek(0)
            return ret_

        mock_urlopen.side_effect = _
        
        
        

        
        
        def read_using_plus(url, blocksize=1024*1024, **kwargs):
            ret = b''
            with closing(mock_urlopen(url, **kwargs)) as conn:
               if blocksize < 0:  # https://docs.python.org/2.4/lib/bltin-file-objects.html
                   ret = conn.read()
               else:
                   while True:
                       buf = conn.read(blocksize)
                       if not buf:
                           break
                       ret += buf
            return ret

        def read_using_join(url, blocksize=1024*1024, **kwargs):
            arr = []
            with closing(mock_urlopen(url, **kwargs)) as conn:
               if blocksize < 0:  # https://docs.python.org/2.4/lib/bltin-file-objects.html
                   ret = conn.read()
               else:
                   while True:
                       buf = conn.read(blocksize)
                       if not buf:
                           break
                       arr.append(buf)
            return b"".join(arr)
        
        def read_using_perc(url, blocksize=1024*1024, **kwargs):
            ret = b''
            with closing(mock_urlopen(url, **kwargs)) as conn:
               if blocksize < 0:  # https://docs.python.org/2.4/lib/bltin-file-objects.html
                   ret = conn.read()
               else:
                   while True:
                       buf = conn.read(blocksize)
                       if not buf:
                           break
                       ret = b"%s%s" % (ret, buf)
            return ret

        def read_using_array(url, blocksize=1024*1024, **kwargs):
            arr = array.array('c', [])
            with closing(mock_urlopen(url, **kwargs)) as conn:
               if blocksize < 0:  # https://docs.python.org/2.4/lib/bltin-file-objects.html
                   ret = conn.read()
               else:
                   while True:
                       buf = conn.read(blocksize)
                       if not buf:
                           break
                       arr.fromstring(buf)
#                        for c in buf:
#                            arr.append(c)
            return arr.tostring()

        def read_using_cstringio(url, blocksize=1024*1024, **kwargs):
            arr = cStringIO.StringIO()
            with closing(mock_urlopen(url, **kwargs)) as conn:
               if blocksize < 0:  # https://docs.python.org/2.4/lib/bltin-file-objects.html
                   ret = conn.read()
               else:
                   while True:
                       buf = conn.read(blocksize)
                       if not buf:
                           break
                       arr.write(buf)
            return arr.getvalue()

        
        attemps = 5000
        
        print ""
        
        methods = {"string +": read_using_plus,
                   "list join": read_using_join,
                   "%%": read_using_perc,
                   "array package": read_using_array,
                   "cStringIO": read_using_cstringio}
        first_res = None
        import time
        
        for key, method in methods.iteritems():
            times = []
            total_bytes = len(ret)
            for iter in xrange(100):
                blocksize = int(float(total_bytes)/(iter+1))
                start = time.time()
                tmp = read_using_plus("", blocksize=blocksize)
                end = time.time()
                times.append(end-start)
                assert ret == tmp
            mean = sum(times) / float(len(times))
            print "Urllib2.read Using %s: estimation from %d attempts: %f" % (key, attemps, attemps*mean)
                
        
#     def test_custom_url_read_zero_timeout(self):
# 
#         def my_load_url(url, timeout=60, decode=None):
#             assert not ismainthread()  # to check that is NOT the main thread
#             conn = urllib2.urlopen(url, timeout=timeout)
#             return conn.read().decode(decode) if decode else conn.read()
# 
#         datas = []
#         errors = []
# 
#         def onsuccess(data, url, index):
#             datas.append(data)
# 
#         def onerror(error, url, index):
#             errors.append(error)
# 
#         # self.urls has a valid url (which should execute onsuccess) and an invalid one
#         # which should execute onerror)
#         AsyncUrl(5).run(self.urls, onsuccess, onerror, urlread_func=my_load_url, timeout=0.001)
# 
#         assert len(datas) == 0 and len(errors) == 2
# 
#     def test_custom_url_fake_keyboard_interrupt(self):
# 
#         datas = []
#         errors = []
# 
#         def onsuccess(data, url, index):
#             if index == 1:
#                 raise KeyboardInterrupt()
#             datas.append(data)
# 
#         def onerror(error, url, index):
#             if index == 1:
#                 raise KeyboardInterrupt()
#             errors.append(error)
# 
#         try:
#             # self.urls has a valid url (which should execute onsuccess) and an invalid one
#             # which should execute onerror)
#             AsyncUrl(5).run(self.urls, onsuccess, onerror)
#         except KeyboardInterrupt as exc:
#             g = 9
#         assert len(datas) + len(errors) == 1  # only one is populated
# 
#     def test_custom_url_fake_keyboard_interrupt1(self):
# 
#         datas = []
#         errors = []
# 
#         def onsuccess(data, url, index):
#             raise KeyboardInterrupt()
# 
#         def onerror(error, url, index):
#             errors.append(error)
# 
#         try:
#             # self.urls has a valid url (which should execute onsuccess) and an invalid one
#             # which should execute onerror)
#             AsyncUrl(5).run(self.urls, onsuccess, onerror)
#         except KeyboardInterrupt as exc:
#             pass
#         assert len(datas) + len(errors) <= 1  # only one is populated
#         
#     
#     def test_custom_url_fake_keyboard_interrupt2(self):
# 
#         datas = []
#         errors = []
# 
#         def onsuccess(data, url, index):
#             datas.append(data)
# 
#         def onerror(error, url, index):
#             raise KeyboardInterrupt()
# 
#         try:
#             # self.urls has a valid url (which should execute onsuccess) and an invalid one
#             # which should execute onerror)
#             AsyncUrl(5).run(self.urls, onsuccess, onerror)
#         except KeyboardInterrupt as exc:
#             pass
#         assert len(datas) + len(errors) <= 1  # only one is populated
# 
#     # @mock.patch('test_concurrent.load_url', side_effect="s")
#     def test_custom_url_general_exception(self):
# 
#         def mock_load_url(*a, **v):
#             time.sleep(3)  # this in order to be safer that exception raise really
#             # avoids calling the next url
#             return "a"
# 
#         load_url = mock.Mock(side_effect=mock_load_url)
#         datas = []
#         errors = []
#         
#         def my_load_url(url, timeout=60, decode=None):
#             assert not ismainthread()  # to check that is NOT the main thread
#             conn = urllib2.urlopen(url, timeout=timeout)
#             return conn.read().decode(decode) if decode else conn.read()
# 
#         def onsuccess(data, url, index):
#             datas.append(data)
#             raise TypeError()
#         
#         def onerror(error, url, index):
#             errors.append(error)
#             raise TypeError()
# 
#         # increase the number of urls to query to, so that we can check that not ALL
#         # of them has been called
#         urls = self.urls*1000
#         try:
#             # self.urls has a valid url (which should execute onsuccess) and an invalid one
#             # which should execute onerror)
#             AsyncUrl(5).run(urls, onsuccess, onerror, urlread_func=load_url)
#         except TypeError:
#             g = 9
#         except KeyboardInterrupt as exc:
#             pass
# 
#         assert load_url.call_count < len(urls)
#         assert load_url.call_count >= len(datas)
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()