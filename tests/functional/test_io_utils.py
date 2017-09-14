'''
Created on Sep 14, 2017

@author: riccardo
'''
import os
import unittest
from stream2segment.io.utils import compress, decompress
from itertools import product
import pytest


class Test(unittest.TestCase):


    def setUp(self):
        self.data = "\x00"+os.urandom(1024*1024)+"\x00"
        pass


    def tearDown(self):
        pass


    def test_compress_decompress(self):
        for comp, compresslevel in product(['bz2', 'zlib', 'gzip', 'zip'], range(1, 10)):
            compr_ = compress(self.data, comp, compresslevel)
            # assert len(compr_) <= len(self.data)
            dec = decompress(compr_)
            assert dec != compr_
            # now test that a non compressed file is returned as-it-is:
            assert decompress(self.data) == self.data
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_compress_decompress']
    unittest.main()