'''
Created on Sep 14, 2017

@author: riccardo
'''
from builtins import range
import os
import unittest
from stream2segment.io.utils import compress, decompress
from itertools import product
import pytest


def test_compress_decompress():
    bytesdata = b"\x00"+os.urandom(1024*1024)+b"\x00"
    for comp, compresslevel in product(['bz2', 'zlib', 'gzip', 'zip'], list(range(1, 10))):
        compr_ = compress(bytesdata, comp, compresslevel)
        # assert len(compr_) <= len(self.data)
        dec = decompress(compr_)
        assert dec != compr_
        # now test that a non compressed file is returned as-it-is:
        assert decompress(bytesdata) == bytesdata
