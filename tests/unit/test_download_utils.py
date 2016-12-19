#@PydevCodeAnalysisIgnore
'''
Created on Dec 14, 2016

@author: riccardo
'''
import unittest
from stream2segment.download.utils import UrlStats


class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testUrlStats(self):
        s = UrlStats()
        assert s['x'] == 0
        s['x'] # does nothing
        assert s['x'] == 0
        s['a'] = 1
        assert s['a'] == 1
        s['a'] += 3
        assert s['a'] == 4
        s['v'] += 3
        assert s['v'] == 3
        
        e = Exception("ucalla")
        s[e] += 2
        assert s[e] == 2
        assert s["%s: %s" % (e.__class__.__name__, str(e))] == 2
        
        e = Exception("ucalla2")
        s[e] = -2
        assert s[e] == -2
        assert s["%s: %s" % (e.__class__.__name__, str(e))] == -2
        
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()