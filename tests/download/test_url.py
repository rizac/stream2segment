"""
Created on Nov 18, 2016

@author: riccardo
"""
import threading
from itertools import cycle
from subprocess import call

from unittest.mock import Mock, patch, MagicMock
import pytest

from stream2segment.download.url import _ismainthread, read_async
from stream2segment.download.url import URLError


class Test:

    # execute this fixture always even if not provided as argument:
    # https://docs.pytest.org/en/documentation-restructure/how-to/fixture.html#autouse-fixtures-xunit-setup-on-steroids
    @pytest.fixture(autouse=True)
    def init(self, request):
        
        self.urls = ["http://sdgfjvkherkdfvsffd",
                     "http://www.google.com", 
                     # "http://www.apple.com",
                     # "http://www.microsoft.com",
                     # "http://www.amazon.com",
                     # "http://www.facebook.com"
                    ]
        self.thread = threading.current_thread()
        
        self.successes = []
        self.errors = []
        self.cancelled = []
        self.progress = 0
        
        with patch('stream2segment.download.url.urlopen') as mock_urlopen:
            self.mock_urlopen = mock_urlopen
            yield

    def read_async(self, *a, **v):
        for obj, url, result, exc, code in read_async(*a, **v):
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
        for obj, url, result, exc, code in read_async(*a, **v):
            assert _ismainthread()
            raise KeyboardInterrupt()
            # self.progress += 1
            # if exc:
            #     self.errors.append(exc)
            # else:
            #     self.successes.append(result)

    def config_urlopen(self, read_side_effect_as_list, sleep_time=None):
        a = Mock()
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
        ret = MagicMock()
        ret.__enter__.return_value = a
        self.mock_urlopen.return_value = ret

    @property
    def mock_urlread(self):
        return self.mock_urlopen.return_value.__enter__.return_value.read

    def test_mocking_urlread(self):
        """Tests onsuccess. WE mock urllib2urlopen.read to return user defined strings"""
        
        data = [b'none', b'', b'google', b'']  # supply an empty string otherwise urllib.read does not stop
        self.config_urlopen(data)

        # self.urls has a valid url (which should execute onsuccess) and an invalid one
        # which should execute onerror)
        successes = []
        self.read_async(self.urls)

        assert len(self.successes) == 2
        
        data_ = list(self.successes)
        for res in data:
            if not res:  # the empty byte is not returned, it serves only to stop urlread
                continue
            assert res in data_
        
        assert self.mock_urlread.call_count == len(data)
        
        assert self.progress == 2

    def test_urlerrors(self):
        """Tests onerror. WE mock urllib2urlopen.read to raise an excpected Exception"""
        
        self.config_urlopen([URLError("")])
        
        # self.urls has a valid url (which should execute onsuccess) and an invalid one
        # which should execute onerror)
        self.read_async(self.urls)

        assert len(self.errors) == 2
        assert self.mock_urlread.call_count == len(self.urls)
        
        assert self.progress == 2
        
    def test_general_exception_from_urlopen(self):
        self.config_urlopen([ValueError("")], sleep_time=None)

        # self.urls has a valid url (which should execute onsuccess) and an invalid one
        # which should execute onerror)
        with pytest.raises(ValueError):
            self.read_async(self.urls)
        assert self.progress == 0

    def test_general_exception_inside_yield(self):
        data = [b'none', b''] * 10000  # supply an empty string otherwise urllib.read does not stop
        self.config_urlopen(data)  # , sleep_time=1)
        
        # self.urls has a valid url (which should execute onsuccess) and an invalid one
        # which should execute onerror)
        with pytest.raises(KeyboardInterrupt):
            self.read_async_raise_exc_in_called_func(self.urls)
        assert self.progress == 0
        # set the totalcounts of mock_urlread: 2 * len(url):
        totalcounts = 2 * len(self.urls)
        # assert we stopped before reading all url(s). Relax the condition by putting <=, as
        # if self.mock_urlread.call_count == totalcounts does not mean the test failed, it
        # can be due to the fact that we mock io-bound operations in urlread with non-io bound operations
        assert self.mock_urlread.call_count <= totalcounts
        
        # same regardless of urllib2 returned value:
        self.config_urlopen([URLError("")], sleep_time=None)
        # self.urls has a valid url (which should execute onsuccess) and an invalid one
        # which should execute onerror)
        with pytest.raises(KeyboardInterrupt):
            self.read_async_raise_exc_in_called_func(self.urls)
        assert self.progress == 0
