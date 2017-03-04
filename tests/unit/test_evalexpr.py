'''
Created on Mar 1, 2017

@author: riccardo
'''
import sys

#from ..mathexpr import interval
from stream2segment.utils.evalexpr import interval, match, where
from datetime import datetime
import numpy as np
import unittest
import pytest
from itertools import product, izip, count
import time


class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testName(self):
        pass



def test_pieceweisefunc():
    # pf = Piecewisefunc({'<7': 'e', '[7, 13.87001]': '11.5+(21-11.5)(x-7)/(13.87001-7)', '>7': 'inf'})
    
    # pf = Piecewisefunc({'<7': 'e', '[7, 13.87001]': '3+(4-3)(x-7)/(7-6)', '>7': 'inf'})
    pass
    

def test_match():
    # compare apples with oranges (different dtypes): return False
    with pytest.raises(TypeError):
        match("!=inf", [float('inf'), 'a', 5])
    
    assert (match("!=inf", [float('inf'), 'a', 5], on_type_mismatch='ignore') == [False, False, False]).all()
    
    #right compare, return correct value:
    assert (match("!=inf", [float('inf'), -23.4, 5]) == [False, True, True]).all()
    
    # check floats / ints:
    assert (match("!=inf", np.array([12, 23, 5], dtype=int)) == [True, True, True]).all()
    
    assert (match(">=5", np.array([12, 23, 5.0], dtype=int)) == [True, True, True]).all()
    
    assert (match("<=23.0", np.array([12.3, 23, 5], dtype=int)) == [True, True, True]).all()
    
    assert (match("<=23", [12.3, 23, 5]) == [True, True, True]).all()
    
    with pytest.raises(Exception): # SyntaxError, needs to quote
        assert match("<2006-01-01T00:00:00", "2006-01-01T00:00:00")
    
    assert (match("<'2006-01-01T00:00:00'", "2006-01-01T00:00:00") == [False]).all()
    
    # test filter (not python filter, filter in test module)
    # it returns always a numpy array

    # test with scalar if we have a suitable filter for arrays
    flt = where("<'2006-01-01T00:00:00'", "2006-01-01T00:00:00")
    assert flt.size == 0
    
    # test with scalar if we have a suitable filter for arrays
    flt = where("<='2006-01-01T00:00:00'", "2006-01-01T00:00:00")
    assert flt.size == 1 and flt[0] == "2006-01-01T00:00:00"
    
    # test with arrays if we have a suitable filter for arrays
    flt = where("<='2006-01-01T00:00:00'", ["2016-01-01T00:00:00", "2006-01-01T00:00:00"])
    assert flt.size == 1 and flt[0] == "2006-01-01T00:00:00"
    
     # test with arrays if we have a suitable filter for arrays
    flt = where("<='2006-01-01T00:00:01'", ["2006-01-01T00:00:00", "2006-01-01T00:00:00"])
    assert flt.size == 2 and np.array_equal(flt, ["2006-01-01T00:00:00", "2006-01-01T00:00:00"])
    
    pass


def test_intervals():
    i_parse = interval.parse

    i0 = i_parse("<=5")
    assert 5 in i0
    assert "5" not in i0
    # not the difference between is True and == True:
    assert i0(5) == True
    assert i0(5) is not True
    # again:
    assert i0("5") == False
    assert i0("5") is not False

    assert i0(5) is not False
    # test list and numpy lists:
    assert [1, 2, -1, -float('inf')] in i0
    #calling the interval return a numpy boolen array
    assert i0([1, 2, -1, -float('inf')]).all()
    
    assert np.array([1, 2, -1, -float('inf')]) in i0

    assert 6 not in i0
    assert "6" not in i0
    assert ["6", "7"] not in i0
    

    i0 = i_parse("<=5")
    i1 = i_parse("[5, 7.25[")
    assert i0 <= i1
    assert not i0 < i1
    assert not i0 >= i1
    assert not i0 == i1
    assert not i0 > i1

    i0 = i_parse("<=5")
    i1 = i_parse("]5, 7[")
    assert i0 < i1
    assert not i0 <= i1

    i0 = i_parse('5')
    i1 = i_parse(']5, 5[')
    assert i1.empty
    # EMPTY INTERVALS EVALUATE TO FALSE EVERY TIME:
    assert not i0 < i1
    assert not i0 > i1
    assert not i0 < i1
    assert not i0 > i1
    assert i0 != i1

    i1 = i_parse("[5,5]")
    assert i0 == i1

    with pytest.raises(Exception):
        i1 = i_parse("]5, 5]")

    with pytest.raises(Exception):
        i1 = i_parse("]5, -5]")

    with pytest.raises(Exception):
        i1 = i_parse("]'z', 'a']")

    with pytest.raises(Exception):
        i1 = i_parse("]1.34, 'a']")
        
    with pytest.raises(Exception):
        i1 = i_parse("]a.b, 'a']")

    with pytest.raises(Exception):
        i1 = i_parse(True, np.array([1,2,3]), 5, False)
        
    with pytest.raises(Exception):
        i1 = i_parse(True, "abc", datetime.utcnow(), False)

    with pytest.raises(Exception):
        i1 = i_parse(True, -1, [1,2, 5], False)

    # test unicodes and strings:
    i0 = interval(']', np.array([u'a'])[0], 'acd', ']')
    i0 = interval(']', np.array([u'a'])[0], u'acd',']')
    with pytest.raises(Exception):
        # lbound not greater than ubound
        i1 = i_parse(True, np.array([u'z'])[0], 'acd', False)

    # test different numbers:
    i0 = interval(']', np.array([1], dtype=int)[0], 23.5, ']')
    i1 = interval(']', np.array([1], dtype=float)[0], 23, ']')
    
    
    i0 = i_parse('<5')
    i1 = i_parse(']-inf, 5[')
    assert i0 != i1
    i1 = i_parse('[-inf, 5.[')
    assert i0 == i1

    i0 = i_parse(">=6.87")
    i1 = i_parse("[6.87, inf[")
    assert i0 != i1
    i1 = i_parse("[6.87, inf]")
    assert i0 == i1

    i0 = i_parse("[-inf, inf]")
    i1 = i_parse("[-inf, inf[")
    assert float('inf') in i0
    assert np.Inf in i0
    assert float('inf') not in i1
    assert np.Inf not in i1

    i0 = i_parse("[1, 1]")
    i1 = i_parse("1")
    assert 1 in i0
    assert np.array(1) in i0
    assert i0 == i1
    
    i0 = i_parse("['a', 's.df'[")
    i1 = i_parse("]'ase', 'rty']")
    
    assert i0 != i1
    assert not i0 < i1
    assert not i0 <= i1
    assert not i0 > i1
    assert not i0 >= i1

    i0 = i_parse("['a', 'sdf']")
    i1 = i_parse(">='sdf'")
    assert i1 >= i0
    assert "sdf" in i1
    assert "sd" not in i1
    
    i0 = interval('[','a', None, '[')
    assert 'a' in i0

    i0 = i_parse("''")
    i1 = i_parse('""')
    assert i0 == i1

    i0 = i_parse("''")
    i1 = i_parse(">=''")
    assert i1 >= i0
    i1 = i_parse(">''")
    assert i1 > i0

    assert not i0.empty  # becasue it's an interval with a single element: the empty string!
    assert i_parse("<''").empty
    assert not i_parse("<=''").empty
    assert not i_parse("<inf").empty
    assert i_parse("<-inf").empty
    assert not i_parse("<=-inf").empty

    i0 = i_parse("'abcd'")
    # as datetime(s) is a kind of string subset, let's see if it recognized that this is a string not datetime
    assert i0._dtype != 'datetime64[us]'
    assert 'a' not in i0
    assert 'abcd' in i0

    # datetime(s):
    with pytest.raises(Exception):
        i1 = interval(']', "2006-01-01", 45, '[')

    #datetimes, test if parses works. In principle, it parses what numpy can parse
    # (at least after some tries)
    i0 = i_parse('["2016-08-17", "2016-09-17T05:04:04"]')
    i0 = i_parse('["2016-08-17T05:34:21", "2016-08-17T05:34:21."]')
    i0 = i_parse('["2016-08-17T05:34:21.1", "2016-08-17T05:34:21.21"]')
    i0 = i_parse('["2016-08-17T05:34:21.100", "2016-08-17T05:34:21.2034"]')
    i0 = i_parse('["2016-08-17T05:34:21.1098", "2016-08-17T05:34:21.45621"]')
    i0 = i_parse('["2016-08-17T05:34:21.098897", "2016-08-17T05:34:21.456217"]')
    # now try with spaces:
    i0 = i_parse('["2016-08-17", "2016-09-17 05:04:04"]')
    i0 = i_parse('["2016-08-17 05:34:21", "2016-08-17 05:34:21."]')
    i0 = i_parse('["2016-08-17 05:34:21.1", "2016-08-17 05:34:21.21"]')
    i0 = i_parse('["2016-08-17 05:34:21.100", "2016-08-17 05:34:21.2034"]')
    i0 = i_parse('["2016-08-17 05:34:21.1098", "2016-08-17 05:34:21.45621"]')
    i0 = i_parse('["2016-08-17 05:34:21.098897", "2016-08-17 05:34:21.456217"]')

    # datetimes. Quoted are interpreted as strings:
    i0 = i_parse('["2016-08-17T05:04:04", "2016-09-17T05:04:04"]')
    assert i0._dtype != 'datetime64[us]'
    assert '2016-08-23T05:04:09' in i0
    assert datetime.utcnow() not in i0

    # check without quotes, it should work
    i1 = i_parse('[2016-08-17T05:04:04, 2016-09-17T05:04:04]')
    assert i1._dtype == 'datetime64[us]'
    assert i0 != i1

    # what about rounding?
    i0 = i_parse('[2016-08-17T05:04:04.000, 2016-09-17T05:04:04.000]')
    assert i0._dtype == 'datetime64[us]'
    assert i0 == i1

    assert np.array_equal(i0(['2016-08-23T05:04:09.000', datetime.utcnow()]), [True, False])
    assert '2006-01abc' not in i0
    assert '2017-08-23T05:14:04' not in i0
    assert '2015-08-23T05:24:01' not in i0

    assert datetime(2016, 8, 17, 5, 4, 4) in i0

    i0 = i_parse(']2016-08-17T05:04:04, 2016-09-17T05:04:04]')
    assert datetime(2016, 8, 17, 5, 4, 4) not in i0

    d1 = datetime.utcnow()
    d2 = datetime.utcnow()
    for lb, ub in product([True, False], [True, False]):
        lbstr = ']' if lb else '['
        ubstr = '[' if ub else ']'
        assert interval(lbstr, d1, d2, ubstr) == i_parse("%s%s, %s%s" % (lbstr, d1.isoformat(),
                                                                   d2.isoformat(),
                                                                   ubstr))

    i0 = i_parse(']"2016-08-17T05:04:04", "2016-09-17T05:04:04"[')
    assert not i0._dtype
    assert '2016-09-17T05:04:04' not in i0  # string, out of bounds
    assert '2016-09-17T05:04:03' in i0  # is a string
    assert datetime(2016,8,25) not in i0  # datetime, not same type
    
    i0 = i_parse(']2016-08-17T05:04:04, 2016-09-17T05:04:04[')
    assert i0._dtype == 'datetime64[us]'
    assert '2016-09-17T05:04:05' not in i0  # is a string not same type
    assert datetime(2016,8,25) in i0  # datetime, in range
    assert datetime(2014,8,25) not in i0  # datetime, not in range
    

    i0 = i_parse('<2016-08-17T05:04:04')
    assert i0._dtype == 'datetime64[us]'
    assert datetime(2016, 8, 17, 5, 4, 4) not in i0
    assert datetime(1968, 8, 17, 5, 4, 4) in i0
    
    i0 = i_parse('<=2016-08-17T05:04:04')
    assert i0._dtype == 'datetime64[us]'
    assert datetime(2016, 8, 17, 5, 4, 4) in i0
    assert datetime(1968, 8, 17, 5, 4, 4) in i0
    
    assert i0 == interval('[', None, datetime(2016, 8, 17, 5, 4, 4), ']')

    i0 = i_parse("[False, True]")
    assert not i0._dtype
    assert "False" not in i0
    assert "True" not in i0

    assert False in i0
    assert True in i0


def test_perf():
    N = 1567000
    a = np.random.rand(N)
    val = np.mean(a)
    start = time.time()
#     nps = [a[a<val],
#     a[a<=val],
#     a[a==val],
#     a[a>=val],
#     a[a>val]
#     ]
    nps = [
        a<val,
        a<=val,
        a==val,
        a>=val,
        a>val
    ]
    print "numpy filter: %s" % (time.time() - start)
    
    start = time.time()
    i1 = interval('[', None, val, ']')
    i2 = interval('[', None, val, '[')
    i3 = interval('[', val, val, ']')
    i4 = interval(']', val, None, ']')
    i5 = interval('[', val, None, ']')
    int1 = [i1(a), i2(a), i3(a), i4(a), i5(a)]
    print "interval with constructor: %s" % (time.time() - start)

    assert all(np.array_equal(n, i) for n, i in izip(nps, int1))

#     for i, n, idx in izip(nps, int1, count()):
#         if not np.array_equal(n, i):
#             dfg = 9

def test_str():
    pass
#     print "Printing some stuff to check str(interval), please ignore"
#     print ""
#     for i in [(True, None, 'a', True),
#               (False, None, 'a', False),
#               (True, 'f', None, True),
#               (False, 'f', None, False),
#               (True, None, 123.5, True),
#               (False, None, 123.5, False),
#               (True, -112, None, True),
#               (False, -112, None, False),
#               (True, -float('inf'), 123.5, True),
#               (False, float('-inf'), 123.5, False),
#               (True, -112, float('inf'), True),
#               (False, -112, float('inf'), False),
#               (True, -112, -112, True),
#               (False, -112, -112, False),
#               (False, None, datetime.utcnow(), True)]:
#         print str(interval(*i))


# import pytest
# @pytest.mark.parametrize("test_input", [
#     ("<=5", "[5,6]", "__"),
#     ("2+4", 6),
#     ("6*9", 42),
# ])
# def test_intervals(test_input, opr, expected):
# def test_intervals():
#     
#     i0 = interval("<=5")
#     assert 5 in i0
#     assert "5" not in i0
#     # not the difference between is True and == True:
#     assert i0(5) == True
#     assert i0(5) is not True
#     # again:
#     assert i0("5") == False
#     assert i0("5") is not False
# 
#     assert i0(5) is not False
#     # test list and numpy lists:
#     assert [1, 2, -1, -float('inf')] in i0
#     assert np.array([1, 2, -1, -float('inf')]) in i0
# 
#     assert 6 not in i0
#     assert "6" not in i0
# 
#     i0 = interval("<=5")
#     i1 = interval("[5, 7.25[")
#     assert i0 <= i1
#     assert not i0 < i1
#     assert not i0 >= i1
#     assert not i0 == i1
#     assert not i0 > i1
# 
#     i0 = interval("<=5")
#     i1 = interval("]5, 7[")
#     assert i0 < i1
#     assert not i0 <= i1
# 
#     i0 = interval('5')
#     i1 = interval(']5, 5[')
#     assert i1.empty
#     # EMPTY INTERVALS EVALUATE TO FALSE EVERY TIME:
#     assert not i0 < i1
#     assert not i0 > i1
#     assert not i0 < i1
#     assert not i0 > i1
#     assert i0 != i1
# 
#     i1 = interval("[5,5]")
#     assert i0 == i1
# 
#     with pytest.raises(ValueError):
#         i1 = interval("]5, 5]")
# 
#     i0 = interval('<5')
#     i1 = interval(']-inf, 5[')
#     assert i0 != i1
#     i1 = interval('[-inf, 5[')
#     assert i0 == i1
# 
#     i0 = interval(">=6.87")
#     i1 = interval("[6.87, inf[")
#     assert i0 != i1
#     i1 = interval("[6.87, inf]")
#     assert i0 == i1
# 
#     i0 = interval("[-inf, inf]")
#     i1 = interval("[-inf, inf[")
#     assert float('inf') in i0
#     assert np.Inf in i0
#     assert float('inf') not in i1
#     assert np.Inf not in i1
# 
#     i0 = interval("[1, 1]")
#     i1 = interval("1")
#     assert 1 in i0
#     assert np.array(1) in i0
#     assert i0 == i1
#     
#     i0 = interval("['a', 'sdf'[")
#     i1 = interval("]'ase', 'rty']")
#     
#     assert i0 != i1
#     assert not i0 < i1
#     assert not i0 <= i1
#     assert not i0 > i1
#     assert not i0 >= i1
# 
#     i0 = interval("['a', 'sdf']")
#     i1 = interval(">='sdf'")
#     assert i1 >= i0
# 
#     i0 = interval("''")
#     i1 = interval('""')
#     assert i0 == i1
# 
#     i0 = interval("''")
#     i1 = interval(">=''")
#     assert i1 >= i0
#     i1 = interval(">''")
#     assert i1 > i0
# 
#     assert not i0.empty  # becasue it's an interval with a single element: the empty string!
#     assert interval("<''").empty
# 
#     i0 = interval("'abcd'")
#     assert 'a' not in i0
#     assert 'abcd' in i0
# 
#     i0 = interval('["2016-08-17T05:04:04", "2016-09-17T05:04:04"]')
#     assert '2016-08-23T05:04:09' in i0
#     assert '2017-08-23T05:14:04' not in i0
#     assert '2015-08-23T05:24:01' not in i0
# 
#     assert '2016-08-17T05:04:04' in i0
#     assert '2016-09-17T05:04:04' in i0
#     i0 = interval(']"2016-08-17T05:04:04", "2016-09-17T05:04:04"]')
#     assert '2016-08-17T05:04:04' not in i0
# 
#     i0 = interval(']"2016-08-17T05:04:04", "2016-09-17T05:04:04"[')
#     assert '2016-09-17T05:04:04' not in i0
# 
#     i0 = interval('["2016-08-17T05:04:04", "2016-09-17T05:04:04"[')
#     assert '2016-09-17T05:04:04' not in i0
# 
#     i0 = interval('<"2016-08-17T05:04:04"')
#     assert '2016-08-17T05:04:04' not in i0
#     assert '2017-08-17T05:04:04' not in i0
#     assert '2015-08-23T05:24:01' in i0
# 
#     i0 = interval('<="2016-08-17T05:04:04"')
#     assert '2016-08-17T05:04:04' in i0
#     assert '2017-08-17T05:04:04' not in i0
#     assert '2015-08-23T05:24:01' in i0
# 
#     i0 = interval("[False, True]")
#     assert "False" not in i0
#     assert "True" not in i0
# 
#     assert False in i0
#     assert True in i0


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()