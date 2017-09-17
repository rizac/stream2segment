#@PydevCodeAnalysisIgnore
'''
Created on Nov 18, 2016

@author: riccardo
'''
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import next
import unittest


import concurrent.futures
import time
import threading
import urllib.request, urllib.error, urllib.parse
import http.client
import mock
# from stream2segment import async
from stream2segment.utils.url import _ismainthread, read_async
from mock import patch
from itertools import product, cycle
import pytest
from urllib.error import URLError
from time import sleep
from subprocess import call




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
        self.cancelled = []
        
        self.patcher = patch('stream2segment.utils.url.urllib.request.urlopen')
        self.mock_urlopen = self.patcher.start()
        #add cleanup (in case tearDown is not 
        self.addCleanup(Test.cleanup, self.patcher)
        self.progress = 0
        
    @staticmethod
    def cleanup(*patchers):
        for patcher in patchers:
            patcher.stop()

    def read_async(self, *a, **v):
        for obj, result, exc, url in read_async(*a, **v):
            assert _ismainthread()
            self.progress += 1
            if exc:
                self.errors.append(exc)
            else:
                self.successes.append(result)
            
    def read_async_raise_exc_in_called_func(self, *a, **v):
        """it is easy to check what happens if an unknown exception is raised from urllib: just mock it
        but what about an exception raised in the caller body, if urlread is ok? Check it here
        """
        for obj, result, exc, url in read_async(*a, **v):
            assert _ismainthread()
            raise KeyboardInterrupt()
            self.progress += 1
            if exc:
                self.errors.append(exc)
            else:
                self.successes.append(result)
                    

    def tearDown(self):
        self.successes = []
        self.errors = []
        pass

    def config_urlopen(self, read_side_effect_as_list, sleep_time=None):
        a = mock.Mock()
        read_side_effect_as_cycle = cycle(read_side_effect_as_list)
        def retfunc(*a, **v):
            if sleep_time:
                call(["sleep", "{:d}".format(sleep_time)])
                # time.sleep(sleep_time)
            val = next(read_side_effect_as_cycle)
            if isinstance(val, Exception):
                raise val
            else:
                return val
        a.read.side_effect = retfunc  # returns each item in list
        self.mock_urlopen.return_value = a

    @property
    def mock_urlread(self):
        return self.mock_urlopen.return_value.read

    def test_mocking_urlread(self):
        """Tests onsuccess. WE mock urllib2urlopen.read to return user defined strings"""
        
        data = [b'none', b'', b'google', b'']  # supply an empty string otherwise urllib.read does not stop
        self.config_urlopen(data)

        # self.urls has a valid url (which should execute onsuccess) and an invalid one
        # which should execute onerror)
        successes = []
        self.read_async(self.urls)

        assert len(self.successes) == 2
        
        for res in data:
            if not res: continue
            assert any(res == x[0] for x in self.successes)
        
        assert self.mock_urlread.call_count == len(data)
        
        assert self.progress == 2

    def test_urlerrors(self):
        """Tests onerror. WE mock urllib2urlopen.read to raise an excpected Exception"""
        
        self.config_urlopen([urllib.error.URLError("")])
        
        # self.urls has a valid url (which should execute onsuccess) and an invalid one
        # which should execute onerror)
        self.read_async(self.urls)

        assert len(self.errors) == 2
        assert self.mock_urlread.call_count == len(self.urls)
        
        assert self.progress == 2
        # check also argument calls:
#         call_args = [a[0] for a in self.ondone.call_args_list]
#         assert all(c[2] for c in call_args)  # all exc argument are truthy
#         assert not any(c[1] for c in call_args) # all result argument are falsy


    
        
    def test_general_exception_from_urlopen(self):
        self.config_urlopen([ValueError("")], sleep_time=None)
        
        # self.urls has a valid url (which should execute onsuccess) and an invalid one
        # which should execute onerror)
        with pytest.raises(ValueError):
            self.read_async(self.urls)
        assert self.progress == 0
        
        
    def test_general_exception_inside_yield(self):
        data = [b'none', b''] * 10000  # supply an empty string otherwise urllib.read does not stop
        self.config_urlopen(data, sleep_time=None)
        
        # self.urls has a valid url (which should execute onsuccess) and an invalid one
        # which should execute onerror)
        with pytest.raises(KeyboardInterrupt):
            self.read_async_raise_exc_in_called_func(self.urls)
        assert self.progress == 0
        
        # same regardless of urllib2 returned value:
        self.config_urlopen([urllib.error.URLError("")], sleep_time=None)
        # self.urls has a valid url (which should execute onsuccess) and an invalid one
        # which should execute onerror)
        with pytest.raises(KeyboardInterrupt):
            self.read_async_raise_exc_in_called_func(self.urls)
        assert self.progress == 0
    
    @patch("stream2segment.utils.url._mem_percent")
    def test_exception_max_mem_cons(self, mock_mem_consumption):
        N= 5000
        data = [b''] * N  # supply an empty string otherwise urllib.read does not stop
        urls = ["http://sdgfjvkherkdfvsffd"] *N 
        self.config_urlopen(data)
        
        mmc = 90
        
        itr = [0]
        def mmc_se(*a, **v):
            if itr[0] > 5:
                return mmc+1
            itr[0] += 1
            return 0
        mock_mem_consumption.side_effect = mmc_se
        
        with pytest.raises(MemoryError) as excinfo:
            self.read_async(urls, max_memoru_consumption=mmc)
        assert self.progress == 6
        assert "Memory overflow: %.2f%% (used) > %.2f%% (threshold)" % (mmc+1, mmc) in str(excinfo.value)
    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()