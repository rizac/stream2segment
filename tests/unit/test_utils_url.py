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
from itertools import product, cycle
import pytest
from urllib2 import URLError

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
        self.ondone_return_value = None
        
        self.ondone = mock.Mock()
        self.ondone.side_effect = self._ondone
        
        self.patcher = patch('stream2segment.utils.url.urllib2.urlopen')
        self.mock_urlopen = self.patcher.start()
        #add cleanup (in case tearDown is not 
        self.addCleanup(Test.cleanup, self.patcher)
        
    @staticmethod
    def cleanup(*patchers):
        for patcher in patchers:
            patcher.stop()
        
    def _ondone(self, obj, res, exc, url):
        assert _ismainthread()
        if exc:
            self.errors.append(exc)
        else:
            self.successes.append(res)
        return self.ondone_return_value

    def tearDown(self):
        self.successes = []
        self.errors = []
        pass

    def config_urlopen(self, read_side_effect_as_list):
        a = mock.Mock()
        a.read.side_effect = cycle(read_side_effect_as_list)  # returns each item in list
        self.mock_urlopen.return_value = a

    @property
    def mock_urlread(self):
        return self.mock_urlopen.return_value.read

    def test_mocking_urlread(self):
        """Tests onsuccess. WE mock urllib2urlopen.read to return user defined strings"""
        
        data = ['none', '', 'google', '']  # supply an empty string otherwise urllib.read does not stop
        self.config_urlopen(data)

        # self.urls has a valid url (which should execute onsuccess) and an invalid one
        # which should execute onerror)
        read_async(self.urls, self.ondone)

        assert len(self.successes) == 2
        
        for res in data:
            if not res: continue
            assert any(res == x[0] for x in self.successes)
        
        assert self.mock_urlread.call_count == len(data)

    def test_urlerrors(self):
        """Tests onerror. WE mock urllib2urlopen.read to raise an excpected Exception"""
        
        self.config_urlopen([urllib2.URLError("")])
        
        # self.urls has a valid url (which should execute onsuccess) and an invalid one
        # which should execute onerror)
        read_async(self.urls, self.ondone)

        assert len(self.errors) == 2
        assert self.mock_urlread.call_count == len(self.urls)
        # check also argument calls:
        call_args = [a[0] for a in self.ondone.call_args_list]
        assert all(c[2] for c in call_args)  # all exc argument are truthy
        assert not any(c[1] for c in call_args) # all result argument are falsy


    
        
    def test_general_exception(self):
        self.config_urlopen([ValueError("")])
        
        # self.urls has a valid url (which should execute onsuccess) and an invalid one
        # which should execute onerror)
        with pytest.raises(ValueError):
            read_async(self.urls, self.ondone)
        
#     def test_cancelled1(self):
#         """Tests cancelled, when url returns without errors"""
#         
#         data = ['none', '', 'google', '']  # supply an empty string otherwise urllib.read does not stop
#         self.config_urlopen(data)
# 
#         self.ondone_return_value = lambda obj: True
#         # self.urls has a valid url (which should execute onsuccess) and an invalid one
#         # which should execute onerror)
#         read_async(self.urls, self.ondone, cancel=True)
# 
#         assert len(self.successes) == 1  # or alternatively:
#         assert self.ondone.call_count == 2
#         # assert there was a call to ondone with cancel argument (the last) as False,
#         # and another with cancel argument as true. We do not care about the order
#         # as with threading is not deterministic
#         call_args = [a[0] for a in self.ondone.call_args_list]
#         cancelled_args = [c[-1] for c in call_args]
#         assert cancelled_args.count(True) == 1  # amazing python feature!!!
#         assert cancelled_args.count(False) == 1  # amazing python feature!!!
#         
#                 
#         assert len(self.cancelled)==1
#     
#     def test_cancelled2(self):
#         """Tests cancelled, when url returns with errors"""
#         
#         data = [URLError(""), 'google', '']  # supply an empty string otherwise urllib.read does not stop
#         self.config_urlopen(data)
# 
#         self.ondone_return_value = lambda obj: True
#         # self.urls has a valid url (which should execute onsuccess) and an invalid one
#         # which should execute onerror)
#         read_async(self.urls, self.ondone, cancel=True)
# 
#         assert self.ondone.call_count == 2
#         call_args = [a[0] for a in self.ondone.call_args_list]
#         cancelled_args = [c[-1] for c in call_args]
#         assert len(self.cancelled) == 1
# 
#         # now the tricky part: which url has been read first?
#         # if the one giving errors, then self.errors has one element
#         # otherwise, not, because the url giving error has been
#         # cancelled! So:
#         if not len(self.errors):
#             assert len(self.successes) == 1
#         else:
#             assert len(self.errors) == 1
#         # assert there was a call to ondone with cancel argument (the last) as False,
#         # and another with cancel argument as true. This should be True
#         # regardless of the urlread function execution order
#         assert cancelled_args.count(True) == 1  # amazing python feature!!!
#         assert cancelled_args.count(False) == 1  # amazing python feature!!!

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()