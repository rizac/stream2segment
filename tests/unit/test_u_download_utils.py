#@PydevCodeAnalysisIgnore
# -*- coding: utf-8 -*-
'''
Created on Feb 4, 2016

@author: riccardo
'''
from __future__ import print_function


from future import standard_library
from stream2segment.download.main import get_datacenters_df
from stream2segment.io.db.models import DataCenter
import re
standard_library.install_aliases()
from builtins import zip
from mock import patch
import pytest
from mock import Mock
from datetime import datetime, timedelta
from io import StringIO
import stream2segment
from stream2segment.download.modules.stationsearch import locations2degrees as s2sloc2deg, get_search_radius
from stream2segment.download.modules.datacenters import EidaValidator
from stream2segment.download.utils import custom_download_codes, DownloadStats, to_fdsn_arg, intkeysdict
from obspy.geodetics.base import locations2degrees  as obspyloc2deg
import numpy as np
import pandas as pd
import code
from itertools import count, product
import time
from obspy.taup.tau_model import TauModel


@pytest.mark.parametrize('lat1, lon1, lat2, lon2',
                         [
                            (5, 3, 5, 7),
                            ([11, 1.4, 3, -17.11], [-1, -.4, 33, -17.11], [0, 0, 0, 0], [1,2,3,4])
                          ]
                         )
def test_loc2deg(lat1, lon1, lat2, lon2):
    if hasattr(lat1, "__iter__"):
        assert np.array_equal(s2sloc2deg(lat1, lon1, lat2, lon2), np.asarray(list(obspyloc2deg(l1, l2, l3, l4) for l1,l2,l3,l4 in zip(lat1,lon1,lat2,lon2))))
    else:   
        assert np.array_equal(s2sloc2deg(lat1, lon1, lat2, lon2), np.asarray(obspyloc2deg(lat1, lon1, lat2, lon2)))

# this is not run as tests, if you want name it test_.. or move it elsewhere to
# see perf differences between obspy loc2deg and s2s loc2deg
# (use -s with pytest in case)
def dummy_test_perf():
    N = 1000
    lat1 = np.random.randint(0, 90, N).astype(float)
    lon1 = np.random.randint(0, 90, N).astype(float)
    lat2 = np.random.randint(0, 90, N).astype(float)
    lon2 = np.random.randint(0, 90, N).astype(float)
    
    s = time.time()
    s2sloc2deg(lat1, lon1, lat2, lon2)
    end = time.time() - s
    
    s2 = time.time()
    for l1, l2, l3, l4 in zip(lat1, lon1, lat2, lon2):
        s2sloc2deg(l1, l2, l3, l4)
    end2 = time.time() - s2
    
    print("%d loops. Numpy loc2deg: %f, obspy loc2deg: %f" % (N, end, end2))        


@pytest.mark.parametrize('mag, minmag_maxmag_minradius_maxradius, expected_val',
                         [
                            (5, [3,3,5,7], 7), (2, [3,3,5,7], 5), (3, [3,3,5,7], 6), ([5,2,3], [3,3,5,7], [7,5,6]), 
                            (2, [3,3,7,7], 7), (3, [3,3,7,7], 7), (13, [3,3,7,7], 7), ([2,3,13], [3,3,7,7], [7,7,7]),
                            (2, [3,5,7,7], 7), (3, [3,5,7,7], 7), (13, [3,5,7,7], 7), ([2,3,13], [3,5,7,7], [7,7,7]),
                            (np.array(2), [3, 7, 1, 5], 1),
                            (2, [3, 7, 1, 5], 1), (8, [3, 7, 1, 5], 5),
                            (5, [3, 7, 1, 5], 3), (-1, [3, 7, 1, 5], 1),
                            (7, [3, 7, 1, 5], 5),
                            ([2, 8, 5, -1, 7], [3, 7, 1, 5], [1, 5, 3, 1, 5]),
                            (np.array([2, 8, 5, -1, 7]), [3, 7, 1, 5], [1, 5, 3, 1, 5]),
                          ]
                         )
def test_get_search_radius(mag, minmag_maxmag_minradius_maxradius, expected_val):
    minmag_maxmag_minradius_maxradius.insert(0, mag)
    try:
        assert get_search_radius(*minmag_maxmag_minradius_maxradius) == expected_val
    except ValueError:  # we passed an array as magnitude, so check with numpy.all()
        assert (get_search_radius(*minmag_maxmag_minradius_maxradius) == expected_val).all()
            
            

def test_stats_table():
    
    ikd = intkeysdict()
    ikd['a'] += 5
    ikd['b'] = 5
    ikd[1] += 5
    ikd[2] = 5
    
    assert all(_ in ikd for _ in ['a', 'b', 1, 2])
    
    urlerr, mseederr, tbound_err, tbound_warn = custom_download_codes()
    seg_not_found = None
    
    d = DownloadStats()
    assert str(d) == ""
    
    d['geofon'][200] += 5
    
    assert str(d) == """
        OK  TOTAL
------  --  -----
geofon   5      5
TOTAL    5      5

COLUMNS DETAILS:
 - OK: Data saved (download ok, no additional warning)"""[1:]
    
    d['geofon']['200'] += 100
    
    assert str(d) == """
        OK   TOTAL
------  ---  -----
geofon  105    105
TOTAL   105    105

COLUMNS DETAILS:
 - OK: Data saved (download ok, no additional warning)"""[1:]

    
    d['geofon'][413] += 5
    
    assert str(d) == """
             Request       
             Entity        
             Too           
        OK   Large    TOTAL
------  ---  -------  -----
geofon  105        5    110
TOTAL   105        5    110

COLUMNS DETAILS:
 - OK: Data saved (download ok, no additional warning)
 - Request Entity Too Large: No data saved (download failed: Client error, server response code 413)"""[1:]
    
    d['geofon'][urlerr] += 3
    
    assert str(d) == """
                    Request       
                    Entity        
             Url    Too           
        OK   Error  Large    TOTAL
------  ---  -----  -------  -----
geofon  105      3        5    113
TOTAL   105      3        5    113

COLUMNS DETAILS:
 - OK: Data saved (download ok, no additional warning)
 - Url Error: No data saved (download failed, generic url error: timeout, no internet connection, ...)
 - Request Entity Too Large: No data saved (download failed: Client error, server response code 413)"""[1:]
    
    d['geofon'][mseederr] += 11
    
    assert str(d) == """
                           Request       
                           Entity        
             MSeed  Url    Too           
        OK   Error  Error  Large    TOTAL
------  ---  -----  -----  -------  -----
geofon  105     11      3        5    124
TOTAL   105     11      3        5    124

COLUMNS DETAILS:
 - OK: Data saved (download ok, no additional warning)
 - MSeed Error: No data saved (download ok, malformed MiniSeed data)
 - Url Error: No data saved (download failed, generic url error: timeout, no internet connection, ...)
 - Request Entity Too Large: No data saved (download failed: Client error, server response code 413)"""[1:]
   
    d['eida'][tbound_err] += 3
    
    assert str(d) == """
                                  Request       
             Time                 Entity        
             Span   MSeed  Url    Too           
        OK   Error  Error  Error  Large    TOTAL
------  ---  -----  -----  -----  -------  -----
geofon  105      0     11      3        5    124
eida      0      3      0      0        0      3
TOTAL   105      3     11      3        5    127

COLUMNS DETAILS:
 - OK: Data saved (download ok, no additional warning)
 - Time Span Error: No data saved (download ok, data completely outside requested time span)
 - MSeed Error: No data saved (download ok, malformed MiniSeed data)
 - Url Error: No data saved (download failed, generic url error: timeout, no internet connection, ...)
 - Request Entity Too Large: No data saved (download failed: Client error, server response code 413)"""[1:]
   
    d['eida'][tbound_warn] += 0
    
    assert str(d) == """
                                             Request       
             OK         Time                 Entity        
             Partially  Span   MSeed  Url    Too           
        OK   Saved      Error  Error  Error  Large    TOTAL
------  ---  ---------  -----  -----  -----  -------  -----
geofon  105          0      0     11      3        5    124
eida      0          0      3      0      0        0      3
TOTAL   105          0      3     11      3        5    127

COLUMNS DETAILS:
 - OK: Data saved (download ok, no additional warning)
 - OK Partially Saved: Data saved (download ok, some received data chunks were completely outside the requested time span and discarded)
 - Time Span Error: No data saved (download ok, data completely outside requested time span)
 - MSeed Error: No data saved (download ok, malformed MiniSeed data)
 - Url Error: No data saved (download failed, generic url error: timeout, no internet connection, ...)
 - Request Entity Too Large: No data saved (download failed: Client error, server response code 413)"""[1:]
    
    d['eida'][tbound_warn] += 6
    
    assert str(d) == """
                                             Request       
             OK         Time                 Entity        
             Partially  Span   MSeed  Url    Too           
        OK   Saved      Error  Error  Error  Large    TOTAL
------  ---  ---------  -----  -----  -----  -------  -----
geofon  105          0      0     11      3        5    124
eida      0          6      3      0      0        0      9
TOTAL   105          6      3     11      3        5    133

COLUMNS DETAILS:
 - OK: Data saved (download ok, no additional warning)
 - OK Partially Saved: Data saved (download ok, some received data chunks were completely outside the requested time span and discarded)
 - Time Span Error: No data saved (download ok, data completely outside requested time span)
 - MSeed Error: No data saved (download ok, malformed MiniSeed data)
 - Url Error: No data saved (download failed, generic url error: timeout, no internet connection, ...)
 - Request Entity Too Large: No data saved (download failed: Client error, server response code 413)"""[1:]
   
    d['geofon'][500] += 1
    
    assert str(d) == """
                                             Request                 
             OK         Time                 Entity   Internal       
             Partially  Span   MSeed  Url    Too      Server         
        OK   Saved      Error  Error  Error  Large    Error     TOTAL
------  ---  ---------  -----  -----  -----  -------  --------  -----
geofon  105          0      0     11      3        5         1    125
eida      0          6      3      0      0        0         0      9
TOTAL   105          6      3     11      3        5         1    134

COLUMNS DETAILS:
 - OK: Data saved (download ok, no additional warning)
 - OK Partially Saved: Data saved (download ok, some received data chunks were completely outside the requested time span and discarded)
 - Time Span Error: No data saved (download ok, data completely outside requested time span)
 - MSeed Error: No data saved (download ok, malformed MiniSeed data)
 - Url Error: No data saved (download failed, generic url error: timeout, no internet connection, ...)
 - Request Entity Too Large: No data saved (download failed: Client error, server response code 413)
 - Internal Server Error: No data saved (download failed: Server error, server response code 500)"""[1:]
    
    
    d['eida'][300] += 3
    
    assert str(d) == """
                                             Request                           
             OK         Time                 Entity   Internal                 
             Partially  Span   MSeed  Url    Too      Server    Multiple       
        OK   Saved      Error  Error  Error  Large    Error     Choices   TOTAL
------  ---  ---------  -----  -----  -----  -------  --------  --------  -----
geofon  105          0      0     11      3        5         1         0    125
eida      0          6      3      0      0        0         0         3     12
TOTAL   105          6      3     11      3        5         1         3    137

COLUMNS DETAILS:
 - OK: Data saved (download ok, no additional warning)
 - OK Partially Saved: Data saved (download ok, some received data chunks were completely outside the requested time span and discarded)
 - Time Span Error: No data saved (download ok, data completely outside requested time span)
 - MSeed Error: No data saved (download ok, malformed MiniSeed data)
 - Url Error: No data saved (download failed, generic url error: timeout, no internet connection, ...)
 - Request Entity Too Large: No data saved (download failed: Client error, server response code 413)
 - Internal Server Error: No data saved (download failed: Server error, server response code 500)
 - Multiple Choices: Data status unknown (download completed, server response code 300 indicates Redirection)"""[1:]
    
     
    d['geofon'][599] += 3
    
    assert str(d) == """
                                             Request                                 
             OK         Time                 Entity   Internal                       
             Partially  Span   MSeed  Url    Too      Server    Multiple  Code       
        OK   Saved      Error  Error  Error  Large    Error     Choices   599   TOTAL
------  ---  ---------  -----  -----  -----  -------  --------  --------  ----  -----
geofon  105          0      0     11      3        5         1         0     3    128
eida      0          6      3      0      0        0         0         3     0     12
TOTAL   105          6      3     11      3        5         1         3     3    140

COLUMNS DETAILS:
 - OK: Data saved (download ok, no additional warning)
 - OK Partially Saved: Data saved (download ok, some received data chunks were completely outside the requested time span and discarded)
 - Time Span Error: No data saved (download ok, data completely outside requested time span)
 - MSeed Error: No data saved (download ok, malformed MiniSeed data)
 - Url Error: No data saved (download failed, generic url error: timeout, no internet connection, ...)
 - Request Entity Too Large: No data saved (download failed: Client error, server response code 413)
 - Internal Server Error: No data saved (download failed: Server error, server response code 500)
 - Multiple Choices: Data status unknown (download completed, server response code 300 indicates Redirection)
 - Code 599: Data status unknown (download completed, server response code 599 is unknown)"""[1:]
    
    d['eida'][204] += 3
    
    assert str(d) == """
                                                      Request                                 
             OK                  Time                 Entity   Internal                       
             Partially  No       Span   MSeed  Url    Too      Server    Multiple  Code       
        OK   Saved      Content  Error  Error  Error  Large    Error     Choices   599   TOTAL
------  ---  ---------  -------  -----  -----  -----  -------  --------  --------  ----  -----
geofon  105          0        0      0     11      3        5         1         0     3    128
eida      0          6        3      3      0      0        0         0         3     0     15
TOTAL   105          6        3      3     11      3        5         1         3     3    143

COLUMNS DETAILS:
 - OK: Data saved (download ok, no additional warning)
 - OK Partially Saved: Data saved (download ok, some received data chunks were completely outside the requested time span and discarded)
 - No Content: Data saved but empty (download ok, the server did not return any data)
 - Time Span Error: No data saved (download ok, data completely outside requested time span)
 - MSeed Error: No data saved (download ok, malformed MiniSeed data)
 - Url Error: No data saved (download failed, generic url error: timeout, no internet connection, ...)
 - Request Entity Too Large: No data saved (download failed: Client error, server response code 413)
 - Internal Server Error: No data saved (download failed: Server error, server response code 500)
 - Multiple Choices: Data status unknown (download completed, server response code 300 indicates Redirection)
 - Code 599: Data status unknown (download completed, server response code 599 is unknown)"""[1:]
     
    
    d['what'][seg_not_found] += 0
    
    assert str(d) == """
                                                               Request                                 
             OK                  Time                 Segment  Entity   Internal                       
             Partially  No       Span   MSeed  Url    Not      Too      Server    Multiple  Code       
        OK   Saved      Content  Error  Error  Error  Found    Large    Error     Choices   599   TOTAL
------  ---  ---------  -------  -----  -----  -----  -------  -------  --------  --------  ----  -----
geofon  105          0        0      0     11      3        0        5         1         0     3    128
eida      0          6        3      3      0      0        0        0         0         3     0     15
what      0          0        0      0      0      0        0        0         0         0     0      0
TOTAL   105          6        3      3     11      3        0        5         1         3     3    143

COLUMNS DETAILS:
 - OK: Data saved (download ok, no additional warning)
 - OK Partially Saved: Data saved (download ok, some received data chunks were completely outside the requested time span and discarded)
 - No Content: Data saved but empty (download ok, the server did not return any data)
 - Time Span Error: No data saved (download ok, data completely outside requested time span)
 - MSeed Error: No data saved (download ok, malformed MiniSeed data)
 - Url Error: No data saved (download failed, generic url error: timeout, no internet connection, ...)
 - Segment Not Found: No data saved (download ok, segment data not found, e.g., after a multi-segment request)
 - Request Entity Too Large: No data saved (download failed: Client error, server response code 413)
 - Internal Server Error: No data saved (download failed: Server error, server response code 500)
 - Multiple Choices: Data status unknown (download completed, server response code 300 indicates Redirection)
 - Code 599: Data status unknown (download completed, server response code 599 is unknown)"""[1:]
    
    
    d['geofon'][413] += 33030000
    
    assert str(d) == """
                                                               Request                                     
             OK                  Time                 Segment  Entity    Internal                          
             Partially  No       Span   MSeed  Url    Not      Too       Server    Multiple  Code          
        OK   Saved      Content  Error  Error  Error  Found    Large     Error     Choices   599   TOTAL   
------  ---  ---------  -------  -----  -----  -----  -------  --------  --------  --------  ----  --------
geofon  105          0        0      0     11      3        0  33030005         1         0     3  33030128
eida      0          6        3      3      0      0        0         0         0         3     0        15
what      0          0        0      0      0      0        0         0         0         0     0         0
TOTAL   105          6        3      3     11      3        0  33030005         1         3     3  33030143

COLUMNS DETAILS:
 - OK: Data saved (download ok, no additional warning)
 - OK Partially Saved: Data saved (download ok, some received data chunks were completely outside the requested time span and discarded)
 - No Content: Data saved but empty (download ok, the server did not return any data)
 - Time Span Error: No data saved (download ok, data completely outside requested time span)
 - MSeed Error: No data saved (download ok, malformed MiniSeed data)
 - Url Error: No data saved (download failed, generic url error: timeout, no internet connection, ...)
 - Segment Not Found: No data saved (download ok, segment data not found, e.g., after a multi-segment request)
 - Request Entity Too Large: No data saved (download failed: Client error, server response code 413)
 - Internal Server Error: No data saved (download failed: Server error, server response code 500)
 - Multiple Choices: Data status unknown (download completed, server response code 300 indicates Redirection)
 - Code 599: Data status unknown (download completed, server response code 599 is unknown)"""[1:]

    d['what'][100] -= 8  # try a negative one. It should work
    
    assert str(d) == """
                                                               Request                                               
             OK                  Time                 Segment  Entity    Internal                                    
             Partially  No       Span   MSeed  Url    Not      Too       Server              Multiple  Code          
        OK   Saved      Content  Error  Error  Error  Found    Large     Error     Continue  Choices   599   TOTAL   
------  ---  ---------  -------  -----  -----  -----  -------  --------  --------  --------  --------  ----  --------
geofon  105          0        0      0     11      3        0  33030005         1         0         0     3  33030128
eida      0          6        3      3      0      0        0         0         0         0         3     0        15
what      0          0        0      0      0      0        0         0         0        -8         0     0        -8
TOTAL   105          6        3      3     11      3        0  33030005         1        -8         3     3  33030135

COLUMNS DETAILS:
 - OK: Data saved (download ok, no additional warning)
 - OK Partially Saved: Data saved (download ok, some received data chunks were completely outside the requested time span and discarded)
 - No Content: Data saved but empty (download ok, the server did not return any data)
 - Time Span Error: No data saved (download ok, data completely outside requested time span)
 - MSeed Error: No data saved (download ok, malformed MiniSeed data)
 - Url Error: No data saved (download failed, generic url error: timeout, no internet connection, ...)
 - Segment Not Found: No data saved (download ok, segment data not found, e.g., after a multi-segment request)
 - Request Entity Too Large: No data saved (download failed: Client error, server response code 413)
 - Internal Server Error: No data saved (download failed: Server error, server response code 500)
 - Continue: Data status unknown (download completed, server response code 100 indicates Informational response)
 - Multiple Choices: Data status unknown (download completed, server response code 300 indicates Redirection)
 - Code 599: Data status unknown (download completed, server response code 599 is unknown)"""[1:]


    # test supplying an object as http code (e.g. a regular expression),
    # everything should work anyway. As classes string representations might have different across
    # python version, supply a new object
    
    class MYObj():
        
        def __str__(self):
            return 'abc_123'
    d['what'][MYObj()] = 14

    assert str(d) == """
                                                               Request                                                        
             OK                  Time                 Segment  Entity    Internal                                             
             Partially  No       Span   MSeed  Url    Not      Too       Server              Multiple  Code  Code             
        OK   Saved      Content  Error  Error  Error  Found    Large     Error     Continue  Choices   599   abc_123  TOTAL   
------  ---  ---------  -------  -----  -----  -----  -------  --------  --------  --------  --------  ----  -------  --------
geofon  105          0        0      0     11      3        0  33030005         1         0         0     3        0  33030128
eida      0          6        3      3      0      0        0         0         0         0         3     0        0        15
what      0          0        0      0      0      0        0         0         0        -8         0     0       14         6
TOTAL   105          6        3      3     11      3        0  33030005         1        -8         3     3       14  33030149

COLUMNS DETAILS:
 - OK: Data saved (download ok, no additional warning)
 - OK Partially Saved: Data saved (download ok, some received data chunks were completely outside the requested time span and discarded)
 - No Content: Data saved but empty (download ok, the server did not return any data)
 - Time Span Error: No data saved (download ok, data completely outside requested time span)
 - MSeed Error: No data saved (download ok, malformed MiniSeed data)
 - Url Error: No data saved (download failed, generic url error: timeout, no internet connection, ...)
 - Segment Not Found: No data saved (download ok, segment data not found, e.g., after a multi-segment request)
 - Request Entity Too Large: No data saved (download failed: Client error, server response code 413)
 - Internal Server Error: No data saved (download failed: Server error, server response code 500)
 - Continue: Data status unknown (download completed, server response code 100 indicates Informational response)
 - Multiple Choices: Data status unknown (download completed, server response code 300 indicates Redirection)
 - Code 599: Data status unknown (download completed, server response code 599 is unknown)
 - Code abc_123: Data status unknown (download completed, server response code abc_123 is unknown)"""[1:]
    

    d['what']['206'] = 14
    
    assert str(d) == """
                                                                        Request                                                        
             OK                           Time                 Segment  Entity    Internal                                             
             Partially  No       Partial  Span   MSeed  Url    Not      Too       Server              Multiple  Code  Code             
        OK   Saved      Content  Content  Error  Error  Error  Found    Large     Error     Continue  Choices   599   abc_123  TOTAL   
------  ---  ---------  -------  -------  -----  -----  -----  -------  --------  --------  --------  --------  ----  -------  --------
geofon  105          0        0        0      0     11      3        0  33030005         1         0         0     3        0  33030128
eida      0          6        3        0      3      0      0        0         0         0         0         3     0        0        15
what      0          0        0       14      0      0      0        0         0         0        -8         0     0       14        20
TOTAL   105          6        3       14      3     11      3        0  33030005         1        -8         3     3       14  33030163

COLUMNS DETAILS:
 - OK: Data saved (download ok, no additional warning)
 - OK Partially Saved: Data saved (download ok, some received data chunks were completely outside the requested time span and discarded)
 - No Content: Data saved but empty (download ok, the server did not return any data)
 - Partial Content: Data probably saved (download completed, server response code 206 indicates Success)
 - Time Span Error: No data saved (download ok, data completely outside requested time span)
 - MSeed Error: No data saved (download ok, malformed MiniSeed data)
 - Url Error: No data saved (download failed, generic url error: timeout, no internet connection, ...)
 - Segment Not Found: No data saved (download ok, segment data not found, e.g., after a multi-segment request)
 - Request Entity Too Large: No data saved (download failed: Client error, server response code 413)
 - Internal Server Error: No data saved (download failed: Server error, server response code 500)
 - Continue: Data status unknown (download completed, server response code 100 indicates Informational response)
 - Multiple Choices: Data status unknown (download completed, server response code 300 indicates Redirection)
 - Code 599: Data status unknown (download completed, server response code 599 is unknown)
 - Code abc_123: Data status unknown (download completed, server response code abc_123 is unknown)"""[1:]


def eq(str1, str2):
    """too much pain to compare if two dataframes string representations are equal: sometimes
    alignment are different (the number of spaces) and the result is ok BUT == returns False.
    let's implement a custom method which tests what we cares"""
    
    ll1 = str1.split("\n")
    ll2 = str2.split("\n")
    
    if len(ll1) != len(ll2):
        return False
    
    # assert splits on each line returns the same lists and if there is an offset
    # (different num spaces) this offset is maintained
    offset = None
    for i, l1, l2 in zip(count(), ll1, ll2):
        # do NOT check for this:
#         if len(l1) != len(l2):
#             return False

        c1 = l1.split()
        c2 = l2.split()
        
        if c1 != c2:
            return False
        
        if i == 1: # skip header (i==0, lengths might not match)
            offset = len(l1) - len(l2)
        elif i > 1 and offset != len(l1) - len(l2):
            return False

    return True
    

def test_eidavalidator():
    responsetext = """http://ws.resif.fr/fdsnws/station/1/query
Z3 A001A * HL? 2017-09-27T00:00:00 2017-10-01T00:00:00
YF * * H?? 2017-09-27T00:00:00 2017-10-01T00:00:00

http://eida.ethz.ch/fdsnws/station/1/query
Z3 A291A * HH? 2017-09-27T00:00:00 2017-10-01T00:00:00

http:wrong
"""
    dc_df = pd.DataFrame(columns=[DataCenter.id.key, DataCenter.station_url.key,
                                  DataCenter.dataselect_url.key],
                         data=[[1, 'http://ws.resif.fr/fdsnws/station/1/query', 'http://ws.resif.fr/fdsnws/dataselect/1/query' ],
                               [2, 'http://eida.ethz.ch/fdsnws/station/1/query', 'http://eida.ethz.ch/fdsnws/dataselect/1/query' ]])
    eidavalidator = EidaValidator(dc_df, responsetext)
    
    tests = {
        (1, 'Z3', 'A001A', '01', 'HLLL'): False,
        (1, 'Z3', 'A001A', '01', 'HLL'): True,
        (2, '', '', '', ''): False,
        (3, '', '', '', ''): False,
        (1, 'Z3', 'A002a', '01', 'HLL'): False,
        (1, 'Z3', 'A001A', '01', 'HLO'): True,
        (1, 'Z3', 'A001A', '', 'HLL'): True,
        (1, 'Z3', 'A291A', '01', 'HHL'): False,
        (1, 'Z3', 'A291A', '01', 'HH'): False,
        (2, 'Z3', 'A291A', '01', 'HH?'): True,
        (1, 'YF', '*', '01', 'HH?'): True,
        (1, 'YF', '*', '01', 'abc'): False,
        (1, 'YF', '*', '01', 'HLW'): True,
        (1, 'YF', '*fwe', 'bla', 'HL?'): True,
        (1, 'YF', 'aewf*', '', 'HDF'): True,
        (1, 'YFA', 'aewf*', '', 'HHH'): False,
        (1, 'YFA', 'aewf*', '', 'HHH'): False,
        }
    
    for k, expected in tests.items():
        assert eidavalidator.isin(*k) == expected

# (any, ['A','D','C','B'])   ['A', 'B', 'C', 'D']  # note result is sorted
#     (any, 'B,C,D,A')          ['A', 'B', 'C', 'D']  # same as above
#     (any, 'A*, B??, C*')      ['A*', 'B??', 'C*']  # fdsn wildcards accepted
#     (any, '!A*, B??, C*')     ['!A*', 'B??', 'C*']  # we support negations: !A* means "not A*"
#     (any, ' A, B ')           ['A', 'B']  # leading and trailing spaces ignored
#     (any, '*')                []  # if any chunk is '*', then [] (=match all) is returned
#     (any, [])                 []  # same as above
#     (any, '  ')               ['']  # this means: match the empty string
#     (2, "--")                 ['']  # for locations (index=2), "--" means empty (fdsn spec.)
#     (1, "--")                 ["--"]  # for others (index = 0,1,3), "--" is what it is
#     (any, "!")                ['!']  # match any non empty string
#     (any, "!*")               this raises (you cannot specify "discard all")
#     (any, "!H*, H*")          this raises (it's a paradox)

def test_to_fdsn_arg():

    val = ['A' , 'B']
    assert to_fdsn_arg(val) == 'A,B'
    
    val = ['!A' , 'B']
    assert to_fdsn_arg(val) == 'B'
    
    val = ['!A' , 'B  ']
    assert to_fdsn_arg(val) == 'B  '
    

# PIECES OF MUSEUMS BELOW!!! OLD TESTS!! leaving as i would do with ancient ruins ;)    
    
# @pytest.mark.parametrize('inargs, expected_dt',
#                          [
#                            ((56,True,True), 56),
#                            ((56,False,True), 56),
#                            ((56,True,False), 56),
#                            ((56,False,False), 56),
#                            (('56',True,True), '56'),
#                            (('56',False,True), '56'),
#                            (('56',True,False), '56'),
#                            (('56',False,False), '56'),
#                            (('a sd ',True,True), 'aTsdT'),
#                            (('a sd ',False,True), 'aTsdT'),
#                            (('a sd ',True,False), 'a sd '),
#                            (('a sd ',False,False), 'a sd '),
#                            (('a sd Z',True,True), 'aTsdT'),
#                            (('a sd Z',False,True), 'aTsdTZ'),
#                            (('a sd Z',True,False), 'a sd '),
#                            (('a sd Z',False,False), 'a sd Z'),
#                            (('2015-01-03 22:22:22Z',True,True), '2015-01-03T22:22:22'),
#                            (('2015-01-03 22:22:22Z',False,True), '2015-01-03T22:22:22Z'),
#                            (('2015-01-03 22:22:22Z',True,False), '2015-01-03 22:22:22'),
#                            (('2015-01-03 22:22:22Z',False,False), '2015-01-03 22:22:22Z'),
#                            ]
#                          )

        
# @pytest.mark.parametrize('prepare_datestr_return_value, strptime_callcount, expected_dt',
#                          [
#                           (56, 1, TypeError()),
#                           ('abc', 3, ValueError()),
#                           ("2006", 3, ValueError()),
#                           ("2006-06", 3, ValueError()),
#                           ("2006-06-06", 2, datetime(2006, 6, 6)),
#                           ("2006-06-06T", 3, ValueError()),
#                           ("2006-06-06T03", 3, ValueError()),
#                           ("2006-06-06T03:22", 3, ValueError()),
#                           ("2006-06-06T03:22:12", 1, datetime(2006,6,6, 3,22,12)),
#                           ("2006-06-06T03:22:12.45", 3, datetime(2006,6,6, 3,22,12,450000)),
#                           ]
#                          )
# # for side effect below
# # see https://docs.python.org/3/library/unittest.mock-examples.html#partial-mocking
# @patch('stream2segment.utils._datetime_strptime', side_effect = lambda *args, **kw: datetime.strptime(*args, **kw))
# # @patch('stream2segment.utils.dt.datetime', spec=datetime, side_effect=lambda *args, **kw: datetime(*args, **kw))
# @patch('stream2segment.utils.prepare_datestr')
# def test_to_datetime_crap(mock_prepare_datestr, mock_strptime, prepare_datestr_return_value,
#                           strptime_callcount, expected_dt):
# 
#     mock_prepare_datestr.return_value = prepare_datestr_return_value
# 
#     inarg = "x"
#     if isinstance(expected_dt, BaseException):
#         with pytest.raises(expected_dt.__class__):
#             dtime(inarg)
#         expected_dt = None
# 
#     mock_prepare_datestr.reset_mock()
#     # mock_datetime.reset_mock()
#     mock_strptime.reset_mock()
#  
#     dt = dtime(inarg, on_err_return_none=True)
#     assert dt == expected_dt
#     mock_prepare_datestr.assert_called_once_with(inarg, True, True)
#     first_args_to_strptime = [c[0][0] for c in mock_strptime.call_args_list]
#     assert all(x == prepare_datestr_return_value for x in first_args_to_strptime)
#     assert mock_strptime.call_count == strptime_callcount


# @patch('stream2segment.download.utils.url_read', return_value='url_read')
# def test_get_events(mock_url_read):  # , mock_urlopen, mock_request):
#     with pytest.raises(KeyError):
#         get_events()
# 
#     args = {'eventws': 'eventws', 'minmag': 1.1,
#             'start': datetime.now().isoformat(),
#             'end': datetime.now().isoformat(),
#             'minlon': '90', 'maxlon': '80',
#             'minlat': '85', 'maxlat': '57'}
# 
#     mock_url_read.reset_mock()
#     lst = get_events(**args)
#     assert lst.empty
#     assert mock_url_read.called
# 
#     mock_url_read.reset_mock()
#     mock_url_read.return_value = 'header\na|b|c'
#     lst = get_events(**args)
#     assert lst.empty
#     assert mock_url_read.called
# 
#     # value error:
#     mock_url_read.reset_mock()
#     mock_url_read.return_value = 'header\na|'+datetime.now().isoformat()+'|c'
#     lst = get_events(**args)
#     assert lst.empty
#     assert mock_url_read.called
# 
#     # index error:
#     mock_url_read.reset_mock()
#     mock_url_read.return_value = 'header\na|'+datetime.now().isoformat()+'|1.1'
#     lst = get_events(**args)
#     assert lst.empty
#     assert mock_url_read.called
# 
#     mock_url_read.reset_mock()
#     d = datetime.now()
#     mock_url_read.return_value = 'header\na|'+d.isoformat()+'|1.1|2|3.0|4.0|a|b|c|d|1.1'
#     lst = get_events(**args)
#     assert lst.empty
#     assert mock_url_read.called
# 
#     mock_url_read.reset_mock()
# 
#     d = datetime.now()
#     mock_url_read.return_value = 'header|a|b|c|d|e|f|g|h|i|j\na|'+d.isoformat()+'|1.1|2|3.0|4.0|a|b|c|d|1.1'
#     lst = get_events(**args)
#     assert len(lst) == 1
#     assert lst.values[0].tolist() == ['a', d, 1.1, 2.0, 3.0, '4.0', 'a', 'b', 'c', 'd', 1.1]
#     assert mock_url_read.called

# @patch('stream2segment.query_utils.url_read', return_value='url_read')
# def test_get_waveforms(mock_url_read):
#     mock_url_read.reset_mock()
#     a, b = get_waveforms('a', 'b', 'c', 'd', '3', '5')
#     assert not a and not b
#     assert not mock_url_read.called
# 
#     mock_url_read.reset_mock()
#     a, b = get_waveforms('a', 'b', 'c',  datetime.utcnow(), '3', '5')
#     assert not a and not b
#     assert not mock_url_read.called
# 
#     with patch('stream2segment.query_utils.getTimeRange') as mock_get_tr:
#         mock_url_read.reset_mock()
#         d1 = datetime.now()
#         d2 = d1 + timedelta(seconds=1)
#         mock_get_tr.return_value = d1, d2
#         a, b = get_waveforms('a', 'b', 'c', 'd', '3', '5')
#         assert a == 'c' and b == mock_url_read.return_value
#         assert mock_url_read.called
#         mock_get_tr.assert_called_with('d', minutes=('3','5'))
# 
#         mock_url_read.reset_mock()
#         a, b = get_waveforms('a', 'b', 'c*', 'd', '3', '5')
#         assert a == 'c' and b == mock_url_read.return_value
#         assert mock_url_read.called
#         mock_get_tr.assert_called_with('d', minutes=('3','5'))
# 
#         mock_url_read.reset_mock()
#         a, b = get_waveforms('a', 'b', [], 'd', '3', '5')
#         assert not a and not b
#         assert not mock_url_read.called
#         mock_get_tr.assert_called_with('d', minutes=('3','5'))
# 
#         mock_url_read.reset_mock()
#         mock_get_tr.side_effect = lambda *args, **kw: get_time_range(*args, **kw)
#         a, b = get_waveforms('a', 'b', 'c', 'd', '3', '5')
#         assert not a and not b
#         assert not mock_url_read.called
#         mock_get_tr.assert_called_with('d', minutes=('3','5'))


# @patch('stream2segment.query_utils.url_read', return_value='url_read')
# def test_get_stations(mock_url_read):
#     mock_url_read.reset_mock()
#     lst = get_stations('a', 'b', 'c', 'd', '5', '6')
#     assert lst.empty
#     assert not mock_url_read.called
# 
#     mock_url_read.reset_mock()
#     with pytest.raises(TypeError):
#         lst = get_stations('a', 'b',  datetime.utcnow(), '4', '3', '5')
#         # assert not mock_url_read.called
# 
#     mock_url_read.reset_mock()
#     lst = get_stations('a', 'b',  datetime.utcnow(), 4, 3, 5)
#     assert lst.empty
#     assert mock_url_read.called
# 
#     with patch('stream2segment.query_utils.get_time_range') as mock_get_timerange:
#         mock_url_read.reset_mock()
#         mock_get_timerange.return_value = (datetime.now(), datetime.now()+timedelta(seconds=1))
#         d = datetime.now()
#         mock_url_read.return_value = 'header\na|b|c'
#         with pytest.raises(TypeError):
#             lst = get_stations('dc', ['listCha'], d, 'lat', 'lon', 'dist')
#             # mock_get_timerange.assert_called_with(d, 1)
#             # assert not mock_url_read.called
# 
#         mock_url_read.reset_mock()
#         with pytest.raises(IndexError):
#             lst = get_stations('dc', ['listCha'], d, 3.1, 2.5, 1.1)
#             # mock_get_timerange.assert_called_with(d, 1)
#             # assert mock_url_read.called
# 
#         mock_url_read.reset_mock()
#         mock_url_read.return_value = 'header\na|b|c|d|e|f|g|h'
#         lst = get_stations('dc', ['listCha'], d, 3.1, 2.5, 1.1)
#         mock_get_timerange.assert_called_with(d, days=1)
#         assert mock_url_read.called
#         assert lst.empty
# 
#         mock_url_read.reset_mock()
#         mock_url_read.return_value = 'header\na|b|1|1.1|2.0|f|'+d.isoformat()+'|h'
#         lst = get_stations('dc', ['listCha'], d, 3.1, 2.5, 1.1)
#         mock_get_timerange.assert_called_with(d, days=1)
#         assert mock_url_read.called
#         assert len(lst) == 1
#         assert lst[0][6] == d
#         assert lst[0][7] == None
# 
#         mock_url_read.reset_mock()
#         d2 = datetime.now()
#         mock_url_read.return_value = 'header\na|b|1|1.1|2.0|f|'+d.isoformat()+'|'+d2.isoformat()
#         lst = get_stations('dc', ['listCha'], d, 3.1, 2.5, 1.1)
#         mock_get_timerange.assert_called_with(d, days=1)
#         assert mock_url_read.called
#         assert len(lst) == 1
#         assert lst[0][6] == d
#         assert lst[0][7] == d2


# @patch('stream2segment.query_utils.timedelta', side_effect=lambda *args, **kw: timedelta(*args, **kw))
# def test_get_timerange(mock_timedelta):
#     mock_timedelta.reset_mock()
#     d = datetime.utcnow()
#     d1, d2 = get_time_range(d, days=1)
#     assert d-d1 == d2-d == timedelta(days=1)
# 
#     mock_timedelta.reset_mock()
#     d = datetime.utcnow()
#     d1, d2 = get_time_range(d, days=1, minutes=(1, 2))
#     assert d-d1 == timedelta(days=1, minutes=1)
#     assert d2-d == timedelta(days=1, minutes=2)
# 
#     mock_timedelta.reset_mock()
#     d = datetime.utcnow()
#     _, _ = get_time_range(d, days=1)
#     assert mock_timedelta.called
# 
#     mock_timedelta.reset_mock()
#     _, _ = get_time_range(d)
#     assert mock_timedelta.called
# 
#     mock_timedelta.reset_mock()
#     _, _ = get_time_range(d, days=(1, 2))
#     assert mock_timedelta.called
# 
#     mock_timedelta.reset_mock()
#     _, _ = get_time_range(d, days=(1, 2), minutes=1)
#     assert mock_timedelta.called
# 
#     mock_timedelta.reset_mock()
#     with pytest.raises(Exception):
#         _, _ = get_time_range(d, days="abc", minutes=1)
#         # assert mock_timedelta.called


# @patch('mod_a.urllib2.urlopen')
# def mytest(mock_urlopen):
#     a = Mock()
#     a.read.side_effect = ['resp1', 'resp2']
#     mock_urlopen.return_value = a
#     res = mod_a.myfunc()
#     print res
#     assert res == 'resp1'
# 
#     res = mod_a.myfunc()
#     print res
#     assert res == 'resp2'

    # mock_ul.urlopen.read.assert_called_with(blockSize)

#     def excp():
#         raise IOError('oops')
#     mock_ul.urlopen.read.side_effect = excp
#     assert url_read(val, "name") == ''
    
# @patch('stream2segment.query_utils.ul.urlopen')
# def test_url_read(mock_ul_urlopen):  # mock_ul_urlopen, mock_ul_request, mock_ul):
#     a = Mock()
#     a.read.side_effect = ['resp1', 'resp2']
#     mock_ul_urlopen.return_value = a
#     
#     val = 'url'
#     assert url_read(val, "name") == "resp1"
#     
#     assert url_read(val, "name") == "resp2"
#     
#     pass



# @patch('stream2segment.query_utils.locations2degrees', return_value = 'l2d')
# @patch('stream2segment.query_utils.get_arrival_time')
# @patch('stream2segment.query_utils.get_events')
# @patch('stream2segment.query_utils.get_stations')
# @patch('stream2segment.query_utils.get_waveforms')
# @patch('stream2segment.query_utils.os.path.exists', return_value=False)
# def test_save_waveforms_nopath(mock_os_path_exists, mock_gw, mock_gs, mock_ge, mock_gat, mock_ltd):
#     mock_os_path_exists.side_effect = lambda arg: False
#     save_waveforms('eventws', 'minmag', 'minlat', 'maxlat', 'minlon', 'maxlon', 
#                   'distFromEvent', 'datacenters_dict',
#                   'channelList', 'start', 'end', ('minBeforeP', 'minAfterP'), 'outpath')
#     mock_os_path_exists.assert_called_with('outpath')
#     assert not mock_ge.called and not mock_gs.called and not mock_gw.called and \
#         not mock_gat.called and not mock_ltd.called
# 
# 
# @patch('stream2segment.query_utils.locations2degrees', return_value = 'l2d')
# @patch('stream2segment.query_utils.get_arrival_time')
# @patch('stream2segment.query_utils.get_events')
# @patch('stream2segment.query_utils.get_stations')
# @patch('stream2segment.query_utils.get_waveforms')
# @patch('stream2segment.query_utils.os.path.exists', return_value=True)
# def test_save_waveforms_get_events_returns_empty(mock_os_path_exists, mock_gw, mock_gs, mock_ge, mock_gat, mock_ltd):
# 
#     mock_ge.side_effect = lambda **args: []
#     save_waveforms('eventws', 'minmag', 'minlat', 'maxlat', 'minlon', 'maxlon', 
#                   'distFromEvent', 'datacenters_dict',
#                   'channelList', 'start', 'end', ('minBeforeP', 'minAfterP'), 'outpath')
#     mock_os_path_exists.assert_called_with('outpath')
#     mock_ge.assert_called_with(**{"eventws": "eventws",
#                                   "minmag": "minmag",
#                                   "minlat": "minlat",
#                                   "maxlat": "maxlat",
#                                   "minlon": "minlon",
#                                   "maxlon": "maxlon",
#                                   "start": "start",
#                                   "end": "end",
#                                   "outpath": "outpath"})
#     assert not mock_gs.called and not mock_gw.called and not mock_gat.called and not mock_ltd.called


# # global vars (FIXME: check if good!)
# dcs = {'dc1' : 'www.dc1'}
# channels = {'chan': ['a', 'b' , 'c']}
# search_radius_args = ['1', None, '4' ,'5']
# 
# @patch('stream2segment.query_utils.locations2degrees', return_value='l2d')
# @patch('stream2segment.query_utils.get_arrival_time')
# @patch('stream2segment.query_utils.get_events', return_value=[[str(i) for i in xrange(12)]])
# @patch('stream2segment.query_utils.get_stations')
# @patch('stream2segment.query_utils.get_waveforms')
# @patch('stream2segment.query_utils.os.path.exists', return_value=True)
# def test_save_waveforms_indexerr_on_get_events(mock_os_path_exists, mock_gw, mock_gs, mock_ge,
#                                               mock_gat, mock_ltd):
#     with pytest.raises(IndexError):
#         save_waveforms('eventws', 'minmag', 'minlat', 'maxlat', 'minlon', 'maxlon',
#                       search_radius_args, dcs,
#                       channels, 'start', 'end', ('minBeforeP', 'minAfterP'), 'outpath')
# 
# 
# @patch('stream2segment.query_utils.locations2degrees', return_value='l2d')
# @patch('stream2segment.query_utils.get_arrival_time')
# @patch('stream2segment.query_utils.get_events', return_value=[[str(i) for i in xrange(13)]])
# @patch('stream2segment.query_utils.get_stations', return_value=[])
# @patch('stream2segment.query_utils.get_waveforms')
# @patch('stream2segment.query_utils.os.path.exists', return_value=True)
# @patch('stream2segment.query_utils.get_search_radius', return_value='gsr')
# def test_save_waveforms_get_stations_returns_empty(mock_gsr, mock_os_path_exists, mock_gw, mock_gs, mock_ge,
#                                                   mock_gat, mock_ltd):
# 
#     save_waveforms('eventws', 'minmag', 'minlat', 'maxlat', 'minlon', 'maxlon',
#                   search_radius_args, dcs,
#                   channels, 'start', 'end', ('minBeforeP', 'minAfterP'), 'outpath')
#     mock_os_path_exists.assert_called_with('outpath')
#     mock_ge.assert_called_with(**{"eventws": "eventws",
#                                   "minmag": "minmag",
#                                   "minlat": "minlat",
#                                   "maxlat": "maxlat",
#                                   "minlon": "minlon",
#                                   "maxlon": "maxlon",
#                                   "start": "start",
#                                   "end": "end",
#                                   "outpath": "outpath"})
# 
#     ev = mock_ge.return_value[0]
#     mock_gsr.assert_called_once_with(ev[10], search_radius_args[0], search_radius_args[1],
#                                      search_radius_args[2], search_radius_args[3])
#     mock_gs.assert_called_with(dcs.values()[0], channels.values()[0], ev[1], ev[2], ev[3],
#                                mock_gsr.return_value)
#     assert not mock_gw.called
# 
# 
# @patch('stream2segment.query_utils.locations2degrees', return_value='l2d')
# @patch('stream2segment.query_utils.get_arrival_time')
# @patch('stream2segment.query_utils.get_events', return_value=[[str(i) for i in xrange(13)]])
# @patch('stream2segment.query_utils.get_stations', return_value=[[str(i) for i in xrange(3)]])
# @patch('stream2segment.query_utils.get_waveforms')
# @patch('stream2segment.query_utils.os.path.exists', return_value=True)
# def test_save_waveforms_indexerr_on_get_stations(mock_os_path_exists, mock_gw, mock_gs, mock_ge,
#                                                 mock_gat, mock_ltd):
#     with pytest.raises(IndexError):
#         save_waveforms('eventws', 'minmag', 'minlat', 'maxlat', 'minlon', 'maxlon',
#                       'distFromEvent', dcs,
#                       channels, 'start', 'end', ('minBeforeP', 'minAfterP'), 'outpath')
# 
# 
# @patch('stream2segment.query_utils.locations2degrees', return_value='l2d')
# @patch('stream2segment.query_utils.get_arrival_time', return_value=None)
# @patch('stream2segment.query_utils.get_events', return_value=[[str(i) for i in xrange(13)]])
# @patch('stream2segment.query_utils.get_stations', return_value=[[str(i) for i in xrange(4)]])
# @patch('stream2segment.query_utils.get_waveforms')
# @patch('stream2segment.query_utils.os.path.exists', return_value=True)
# @patch('stream2segment.query_utils.get_search_radius', return_value='gsr')
# def test_save_waveforms_get_arrival_time_none(mock_gsr, mock_os_path_exists, mock_gw, mock_gs, mock_ge,
#                                               mock_gat, mock_ltd):
#     save_waveforms('eventws', 'minmag', 'minlat', 'maxlat', 'minlon', 'maxlon',
#                   search_radius_args, dcs,
#                   channels, 'start', 'end', ('minBeforeP', 'minAfterP'), 'outpath')
#     mock_os_path_exists.assert_called_with('outpath')
#     mock_ge.assert_called_with(**{"eventws": "eventws",
#                                   "minmag": "minmag",
#                                   "minlat": "minlat",
#                                   "maxlat": "maxlat",
#                                   "minlon": "minlon",
#                                   "maxlon": "maxlon",
#                                   "start": "start",
#                                   "end": "end",
#                                   "outpath": "outpath"})
#     ev = mock_ge.return_value[0]
#     st = mock_gs.return_value[0]
#     mock_gsr.assert_called_once_with(ev[10], search_radius_args[0], search_radius_args[1],
#                                      search_radius_args[2], search_radius_args[3])
#     mock_gs.assert_called_with(dcs.values()[0], channels.values()[0], ev[1], ev[2], ev[3],
#                                mock_gsr.return_value)
#     mock_ltd.assert_called_with(ev[2], ev[3], st[2], st[3])
#     mock_gat.assert_called_with(ev[4], mock_ltd.return_value)
#     assert not mock_gw.called
# 
# 
# @patch('__builtin__.open')
# @patch('stream2segment.query_utils.locations2degrees', return_value='l2d')
# @patch('stream2segment.query_utils.get_arrival_time', return_value=5)
# @patch('stream2segment.query_utils.get_events', return_value=[[str(i) for i in xrange(13)]])
# @patch('stream2segment.query_utils.get_stations', return_value=[[str(i) for i in xrange(4)]])
# @patch('stream2segment.query_utils.get_waveforms', return_value=('', ''))
# @patch('stream2segment.query_utils.os.path.exists', return_value=True)
# @patch('stream2segment.query_utils.get_search_radius', return_value='gsr')
# @patch('stream2segment.query_utils.os.path.join', return_value='joined')
# def test_save_waveforms_get_arrival_time_no_wav(mock_os_path_join, mock_gsr, mock_os_path_exists,
#                                                 mock_gw, mock_gs, mock_ge, mock_gat, mock_ltd,
#                                                 mock_open):
#     d = datetime.now()
#     evz = mock_ge.return_value
#     evz[0][1] = d
#     mock_ge.return_value = evz
# 
#     save_waveforms('eventws', 'minmag', 'minlat', 'maxlat', 'minlon', 'maxlon',
#                   search_radius_args, dcs, channels, 'start', 'end', ('minBeforeP', 'minAfterP'),
#                   'outpath')
#     mock_os_path_exists.assert_called_with('outpath')
#     mock_ge.assert_called_with(**{"eventws": "eventws",
#                                   "minmag": "minmag",
#                                   "minlat": "minlat",
#                                   "maxlat": "maxlat",
#                                   "minlon": "minlon",
#                                   "maxlon": "maxlon",
#                                   "start": "start",
#                                   "end": "end",
#                                   "outpath": "outpath"})
#     ev = mock_ge.return_value[0]
#     st = mock_gs.return_value[0]
#     mock_gsr.assert_called_once_with(ev[10], search_radius_args[0], search_radius_args[1],
#                                      search_radius_args[2], search_radius_args[3])
#     mock_gs.assert_called_with(dcs.values()[0], channels.values()[0], ev[1], ev[2], ev[3],
#                                mock_gsr.return_value)
#     mock_ltd.assert_called_with(ev[2], ev[3], st[2], st[3])
#     mock_gat.assert_called_with(ev[4], mock_ltd.return_value)
#     origTime = ev[1] + timedelta(seconds=float(mock_gat.return_value))
#     mock_gw.assert_called_with(dcs.values()[0], st[1], channels.values()[0], origTime, 'minBeforeP',
#                                'minAfterP')
#     assert not mock_os_path_join.called
#     assert not mock_open.called
# 
# 
# @patch('__builtin__.open')
# @patch('stream2segment.query_utils.locations2degrees', return_value='l2d')
# @patch('stream2segment.query_utils.get_arrival_time', return_value=5)
# @patch('stream2segment.query_utils.get_events', return_value=[[str(i) for i in xrange(13)]])
# @patch('stream2segment.query_utils.get_stations', return_value=[[str(i) for i in xrange(4)]])
# @patch('stream2segment.query_utils.get_waveforms', return_value=('', 'wav'))
# @patch('stream2segment.query_utils.os.path.exists', return_value=True)
# @patch('stream2segment.query_utils.get_search_radius', return_value='gsr')
# @patch('stream2segment.query_utils.os.path.join', return_value='joined')
# def test_save_waveforms_get_arrival_time(mock_os_path_join, mock_gsr, mock_os_path_exists, mock_gw, mock_gs,
#                                          mock_ge, mock_gat, mock_ltd, mock_open):
#     d = datetime.now()
#     evz = mock_ge.return_value
#     evz[0][1] = d
#     mock_ge.return_value = evz
# 
#     save_waveforms('eventws', 'minmag', 'minlat', 'maxlat', 'minlon', 'maxlon',
#                   search_radius_args, dcs, channels, 'start', 'end', ('minBeforeP', 'minAfterP'),
#                   'outpath')
#     mock_os_path_exists.assert_called_with('outpath')
#     mock_ge.assert_called_with(**{"eventws": "eventws",
#                                   "minmag": "minmag",
#                                   "minlat": "minlat",
#                                   "maxlat": "maxlat",
#                                   "minlon": "minlon",
#                                   "maxlon": "maxlon",
#                                   "start": "start",
#                                   "end": "end",
#                                   "outpath": "outpath"})
#     ev = mock_ge.return_value[0]
#     st = mock_gs.return_value[0]
#     mock_gsr.assert_called_once_with(ev[10], search_radius_args[0], search_radius_args[1],
#                                      search_radius_args[2], search_radius_args[3])
#     mock_gs.assert_called_with(dcs.values()[0], channels.values()[0], ev[1], ev[2], ev[3],
#                                mock_gsr.return_value)
#     mock_ltd.assert_called_with(ev[2], ev[3], st[2], st[3])
#     mock_gat.assert_called_with( ev[4], mock_ltd.return_value)
#     origTime = ev[1] + timedelta(seconds=float(mock_gat.return_value))
#     mock_gw.assert_called_with(dcs.values()[0], st[1], channels.values()[0], origTime, 'minBeforeP',
#                                'minAfterP')
#     mock_os_path_join.assert_called_with('outpath', 'ev-%s-%s-%s.mseed' % (ev[0], st[1], mock_gw.return_value[0]))
#     mock_open.assert_called_with(mock_os_path_join.return_value, 'wb')



# mock_dt.py
# import datetime
# import mock
# 
# real_datetime_class = datetime.datetime
# 
# def mock_datetime_now(target, dt):
#     class DatetimeSubclassMeta(type):
#         @classmethod
#         def __instancecheck__(mcs, obj):
#             return isinstance(obj, real_datetime_class)
# 
#     class BaseMockedDatetime(real_datetime_class):
#         @classmethod
#         def now(cls, tz=None):
#             return target.replace(tzinfo=tz)
# 
#         @classmethod
#         def utcnow(cls):
#             return target
# 
#     # Python2 & Python3 compatible metaclass
#     # Note: type('X', (object,), dict(a=1)) is the same as:
#     # class X(object):
#     #    a = 1
#     MockedDatetime = DatetimeSubclassMeta('datetime', (BaseMockedDatetime,), {})
#     # Note
#     
#     
#     return mock.patch.object(dt, 'datetime', MockedDatetime)

