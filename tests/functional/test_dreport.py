'''
Created on 15 Mar 2018

@author: riccardo
'''
import unittest
from stream2segment.utils import get_session
from stream2segment.gui.dreport import get_dstats_dicts, get_dstats_str_iter


class Test(unittest.TestCase):


    def setUp(self):
        self.dburl = 'sqlite:////Users/riccardo/work/gfz/data/s2s/download/2018_03_15.sqlite'
#         self.session = \
#             get_session('sqlite:////Users/riccardo/work/gfz/data/s2s/download/2018_03_15.sqlite')
        pass


    def tearDown(self):
        pass


    def test_get_data(self):
        d = get_dstats_dicts(self.dburl)
        
        
        xv = "\n".join(get_dstats_str_iter(self.dburl))
        h = 9

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()