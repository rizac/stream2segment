# -*- coding: utf-8 -*-
'''
Created on Feb 4, 2016

@author: riccardo
'''
from __future__ import print_function

import re
from builtins import zip
from datetime import datetime, timedelta
from itertools import count, product
import time

import pytest
import numpy as np
import pandas as pd
from obspy.geodetics.base import locations2degrees as obspyloc2deg

from stream2segment.download.db.models import DataCenter
from stream2segment.download.modules.stationsearch import \
    locations2degrees as s2sloc2deg, get_magdep_search_radius
from stream2segment.download.modules.datacenters import EidaValidator
from stream2segment.download.modules.utils import (s2scodes, DownloadStats,
                                                   HTTPCodesCounter, logwarn_dataframe,
                                                   strconvert, strptime)


@pytest.mark.parametrize('str_input, expected_diff, ',
                          [
                           ("2016-01-01", timedelta(minutes=60)),
                           ("2016-01-01T01:11:15", timedelta(minutes=60)),
                           ("2016-01-01 01:11:15", timedelta(minutes=60)),
                           ("2016-01-01T01:11:15.556734", timedelta(minutes=60)),
                           ("2016-01-01 01:11:15.556734", timedelta(minutes=60)),
                           ("2016-07-01", timedelta(minutes=120)),
                           ("2016-07-01T01:11:15", timedelta(minutes=120)),
                           ("2016-07-01 01:11:15", timedelta(minutes=120)),
                           ("2016-07-01T01:11:15.431778", timedelta(minutes=120)),
                           ("2016-07-01 01:11:15.431778", timedelta(minutes=120)),
                           ],
                        )
def test_strptime(str_input, expected_diff):

    if ":" in str_input:
        arr = [str_input, str_input + 'UTC', str_input+'Z', str_input+'CET']
    else:
        arr = [str_input]
    for ds1, ds2 in product(arr, arr):

        d1 = strptime(ds1)
        d2 = strptime(ds2)

        if ds1[-3:] == 'CET' and not ds2[-3:] == 'CET':
            # ds1 was CET, it means that d1 (which is UTC) is one hour less than d2 (which is UTC)
            assert d1 == d2 - expected_diff
        elif ds2[-3:] == 'CET' and not ds1[-3:] == 'CET':
            # ds2 was CET, it means that d2 (which is UTC) is one hour less than d1 (which is UTC)
            assert d2 == d1 - expected_diff
        else:
            assert d1 == d2
        assert d1.tzinfo is None and d2.tzinfo is None
        assert strptime(d1) == d1
        assert strptime(d2) == d2

    # test a valueerror:
    if ":" not in str_input:
        for dtimestr in [str_input+'Z', str_input+'CET']:
            with pytest.raises(ValueError):
                strptime(dtimestr)

    # test type error:
    with pytest.raises(TypeError):
        strptime(5)


def test_strconvert():
    strings = ["%", "_", "*", "?", ".*", "."]

    # sql 2 wildcard
    expected = ["*", "?", "*", "?", ".*", "."]
    for a, exp in zip(strings, expected):
        assert strconvert.sql2wild(a) == exp
        assert strconvert.sql2wild(a+a) == exp+exp
        assert strconvert.sql2wild("a"+a) == "a"+exp
        assert strconvert.sql2wild(a+"a") == exp+"a"

    # wildcard 2 sql
    expected = ["%", "_", "%", "_", ".%", "."]
    for a, exp in zip(strings, expected):
        assert strconvert.wild2sql(a) == exp
        assert strconvert.wild2sql(a+a) == exp+exp
        assert strconvert.wild2sql("a"+a) == "a"+exp
        assert strconvert.wild2sql(a+"a") == exp+"a"

    # sql 2 regex
    expected = [".*", ".", "\\*", "\\?", "\\.\\*", "\\."]
    for a, exp in zip(strings, expected):
        assert strconvert.sql2re(a) == exp
        assert strconvert.sql2re(a+a) == exp+exp
        assert strconvert.sql2re("a"+a) == "a"+exp
        assert strconvert.sql2re(a+"a") == exp+"a"

    # wild 2 regex
    # Note that we escape '%' and '_' becasue different versions
    # of python escape them (=> insert a backslash before) differently
    # See https://docs.python.org/3/library/re.html#re.escape
    expected = [re.escape("%"), re.escape("_"), ".*", ".", "\\..*", "\\."]
    for a, exp in zip(strings, expected):
        assert strconvert.wild2re(a) == exp
        assert strconvert.wild2re(a+a) == exp+exp
        assert strconvert.wild2re("a"+a) == "a"+exp
        assert strconvert.wild2re(a+"a") == exp+"a"


@pytest.mark.parametrize('lat1, lon1, lat2, lon2',
                         [(5, 3, 5, 7),
                          ([11, 1.4, 3, -17.11], [-1, -.4, 33, -17.11], [0, 0, 0, 0],
                           [1, 2, 3, 4])
                          ])
def test_loc2deg(lat1, lon1, lat2, lon2):
    if hasattr(lat1, "__iter__"):
        assert np.array_equal(s2sloc2deg(lat1, lon1, lat2, lon2),
                              np.asarray(list(obspyloc2deg(l1, l2, l3, l4)
                                              for l1, l2, l3, l4 in zip(lat1, lon1, lat2, lon2))))
    else:
        assert np.array_equal(s2sloc2deg(lat1, lon1, lat2, lon2),
                              np.asarray(obspyloc2deg(lat1, lon1, lat2, lon2)))


# this is not run as tests, if you want name it test_.. or move it elsewhere to
# see perf differences between obspy loc2deg and s2s loc2deg
# (use -s with pytest in case)
def dummy_tst_perf():
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
                         [(5, [3, 3, 5, 7], 7),
                          (2, [3, 3, 5, 7], 5),
                          (3, [3, 3, 5, 7], 7),
                          ([5, 2, 3], [3, 3, 5, 7], [7, 5, 7]),
                          (2, [3, 3, 7, 7], 7), (3, [3, 3, 7, 7], 7),
                          (13, [3, 3, 7, 7], 7),
                          ([2, 3, 13], [3, 3, 7, 7], [7, 7, 7]),
                          (2, [3, 5, 7, 7], 7),
                          (3, [3, 5, 7, 7], 7),
                          (13, [3, 5, 7, 7], 7),
                          ([2, 3, 13], [3, 5, 7, 7], [7, 7, 7]),
                          (np.array(2), [3, 7, 1, 5], 1),
                          (2, [3, 7, 1, 5], 1),
                          (8, [3, 7, 1, 5], 5),
                          (5, [3, 7, 1, 5], 3),
                          (-1, [3, 7, 1, 5], 1),
                          (7, [3, 7, 1, 5], 5),
                          ([2, 8, 5, -1, 7], [3, 7, 1, 5], [1, 5, 3, 1, 5]),
                          (np.array([2, 8, 5, -1, 7]), [3, 7, 1, 5], [1, 5, 3, 1, 5]),
                          ]
                         )
def test_get_magdep_search_radius(mag, minmag_maxmag_minradius_maxradius, expected_val):
    minmag_maxmag_minradius_maxradius.insert(0, mag)
    try:
        assert get_magdep_search_radius(*minmag_maxmag_minradius_maxradius) == expected_val
    except ValueError:  # we passed an array as magnitude, so check with numpy.all()
        assert (get_magdep_search_radius(*minmag_maxmag_minradius_maxradius) == expected_val).all()


def test_stats_table():

    ikd = HTTPCodesCounter()
    ikd['a'] += 5
    ikd['b'] = 5
    ikd[1] += 5
    ikd[2] = 5

    assert all(_ in ikd for _ in ['a', 'b', 1, 2])

    urlerr, mseederr, tbound_err, tbound_warn = \
        s2scodes.url_err, s2scodes.mseed_err, s2scodes.timespan_err, s2scodes.timespan_warn
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
    alignment are different (the number of spaces) and the result is ok BUT comparison
    returns False.
    let's implement a custom method which tests what we cares"""

    ll1 = str1.split("\n")
    ll2 = str2.split("\n")

    if len(ll1) != len(ll2):
        return False

    # assert splits on each line returns the same lists and if there is an offset
    # (different num spaces) this offset is maintained
    offset = None
    for i, l1, l2 in zip(count(), ll1, ll2):

        # do NOT check for len(l1) == len(l2), BUT:
        c1 = l1.split()
        c2 = l2.split()

        if c1 != c2:
            return False

        if i == 1:  # skip header (i==0, lengths might not match)
            offset = len(l1) - len(l2)
        elif i > 1 and offset != len(l1) - len(l2):
            return False

    return True


def test_eidavalidator():
    responsetext = """http://ws.resif.fr/fdsnws/station/1/query
Z3 A001A * HL? 2000-01-01T00:00:00 2001-01-01T00:00:00
YF * * H?? * *
ZE ABC * * 2000-01-01T00:00:00 2001-01-01T00:00:00

http://eida.ethz.ch/fdsnws/station/1/query
Z3 A291A * HH? 2000-01-01T00:00:00 2001-01-01T00:00:00
ZE ABC * * 2001-01-01T00:00:00 2002-01-01T00:00:00

http:wrong
"""
    dc_df = pd.DataFrame(columns=[DataCenter.id.key, DataCenter.station_url.key,
                                  DataCenter.dataselect_url.key],
                         data=[[1, 'http://ws.resif.fr/fdsnws/station/1/query',
                                'http://ws.resif.fr/fdsnws/dataselect/1/query'],
                               [2, 'http://eida.ethz.ch/fdsnws/station/1/query',
                                'http://eida.ethz.ch/fdsnws/dataselect/1/query']])
    eidavalidator = EidaValidator(dc_df, responsetext)

    tests = {
        ('Z3', 'A001A', '01', 'HLLL', None, None): None,
        ('Z3', 'A001A', '01', 'HLL', None, None): [1],
        ('', '', '', '', None, None): None,
        ('', '', '', '', None, None): None,
        ('Z3', 'A002a', '01', 'HLL', None, None): None,
        ('Z3', 'A001A', '01', 'HLO', None, None): [1],
        ('Z3', 'A001A', '', 'HLL', None, None): [1],
        ('Z3', 'A291A', '01', 'HHL', None, None): [2],
        ('Z3', 'A291A', '01', 'HH', None, None): None,
        ('Z3', 'A291A', '01', 'HH?', None, None): [2],
        ('YF', '*', '01', 'HH?', None, None): [1],
        ('YF', '*', '01', 'abc', None, None): None,
        ('YF', '*', '01', 'HLW', None, None): [1],
        ('YF', '*fwe', 'bla', 'HL?', None, None): [1],
        ('YF', 'aewf*', '', 'HDF', None, None): [1],
        ('YFA', 'aewf*', '', 'HHH', None, None): None,
        ('YFA', 'aewf*', '', 'HHH', None, None): None,
        # time bounds:
        ('ZE', 'ABC', None, None, None, None): [1, 2],
        ('ZE', 'ABC', None, None,  None, datetime(2000, 4, 1)): [1],
        ('ZE', 'ABC', '01', 'HLL', datetime(2020, 1, 1), None): None,
        ('ZE', 'ABC', '02', 'ABC', datetime(1990, 1, 1), None): [1, 2],
        ('ZE', 'ABC', '02', 'ABC', None, datetime(1990, 1, 1)): None,
        ('ZE', 'ABC', '03', 'XYZ', datetime(2000, 6, 1), None): [1, 2],
        ('ZE', 'ABC', '04', 'ELE', datetime(2001, 4, 1), None): [2],
        }

    for k, expected in tests.items():
        # convert to new values:
        expected = set() if expected is None else set(expected)
        assert eidavalidator.get_dc_ids(*k) == expected

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


# def test_to_fdsn_arg():
#
#     val = ['A', 'B']
#     assert to_fdsn_arg(val) == 'A,B'
#
#     val = ['!A', 'B']
#     assert to_fdsn_arg(val) == 'B'
#
#     val = ['!A', 'B  ']
#     assert to_fdsn_arg(val) == 'B  '



def test_logwarn_dataframe_columns_none():
    """Simple test asserting there are no exceptions. FIXES problems
    with duplicated routing services where we pass None as columns"""
    dfr = pd.DataFrame({'c1': ['a', 'b'], 'c2': [1, np.nan]})
    logwarn_dataframe(dfr, 'a message', columns=['c2'])
    logwarn_dataframe(dfr, 'a message', columns=['c1', 'c2'])
    logwarn_dataframe(dfr, 'a message')
    logwarn_dataframe(dfr, 'a message', max_row_count=45)
    logwarn_dataframe(dfr, 'a message', max_row_count=1)

