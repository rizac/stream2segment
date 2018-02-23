'''
Created on 21 Feb 2018

@author: riccardo
'''
import unittest
from itertools import product
from stream2segment.download.utils import nslc_join, aspdfilter, asbinexp, nslc_lists
import pytest
import pandas as pd
import re
from stream2segment.io.db.models import Station, Channel


def aspost(net, sta, loc, cha):
    return " ".join("*" if not lst else nslc_join(lst, "--") for lst in (net, sta, loc, cha))


def asget(net, sta, loc, cha):
    ret = []
    for param, lst in zip(('net', 'sta', 'loc', 'cha'), (net, sta, loc, cha)):
        if lst:
            ret.append("{}={}".format(param, (nslc_join(lst, ""))))
     
    return "&".join(ret)

class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testName(self):
        pass

    @staticmethod
    def todict(n,s,l,c):
        return {'net': n, 'sta': s, 'loc': l, 'cha': c}

    def test_stripspaces(self):
        data = [None, '*', 'CC?,AAA', 'CC?, AAA', 'CC? ,AAA', 'CC? , AAA', ""]
        for n, s, l, c in product(data, data, data, data):
            nl, sl, ll, cl = nslc_lists(self.todict(n, s, l, c))
            if any(_ is not None for _ in (n,s,l,c)):
                # test with leading and trailing spaces:
                for ws1, ws2 in [(' ', ''), ('', ' '), (' ', ' ')]:
                    nl_, sl_, ll_, cl_ = nslc_lists(self.todict(*(n if n is None else ws1 + n + ws2,
                                                                  s if s is None else ws1 + s + ws2,
                                                                  l if l is None else ws1 + l + ws2,
                                                                  c if c is None else ws1 + c + ws2)
                                                                  ))
                    assert asget(nl_, sl_, ll_, cl_) == asget(nl, sl, ll, cl)
                    assert aspost(nl_, sl_, ll_, cl_) == aspost(nl, sl, ll, cl)
    
    def test_strings_and_array_of_strings(self):
        data = [None, '*', 'CC?,*', '', 'CC?, AA, AA, BB',]
        for n, s, l, c in product(data, data, data, data):
            nl, sl, ll, cl = nslc_lists(self.todict(n, s, l, c))
            if any(_ is not None for _ in (n,s,l,c)):
                # test that with arrays is the same:
                nl_, sl_, ll_, cl_ = nslc_lists(self.todict(*(n if n is None else n.split(','),
                                                              s if s is None else s.split(','),
                                                              l if l is None else l.split(','),
                                                              c if c is None else c.split(','))))
                assert asget(nl_, sl_, ll_, cl_) == asget(nl, sl, ll, cl)
                assert aspost(nl_, sl_, ll_, cl_) == aspost(nl, sl, ll, cl)

    def test_nslc_aspostasget_duplicates_are_removed(self):
        '''general test to see if post data get data return the correct values'''
        data = ['*', 'AA?', 'AA?, BBB', "", "--"]

        for n, s, l, c in product(data, data, data, data):
            nl1, sl1, ll1, cl1 = nslc_lists(self.todict(n, s, l, c))
            nl2, sl2, ll2, cl2 = nslc_lists(self.todict(n+","+n, s+","+s, l+","+l, c+","+c))
            
            assert asget(nl1, sl1, ll1, cl1) == asget(nl2, sl2, ll2, cl2)
            assert aspost(nl1, sl1, ll1, cl1) == aspost(nl2, sl2, ll2, cl2)


    def test_nslc_aspostasget(self):
        '''general test to see if post data get data return the correct values'''
        data = [None, '*', 'CC?,*', 'AA?', 'AA?, BBB', # 'AA,BB,AA', 'AA,BB,,', 'AA,BB,--',
                "", "--"]
        
        expected_get = {'AA?': 'AA?',
                        'AA?, BBB': 'AA?,BBB',
                        'AA,BB,AA': 'AA,BB',
                        'AA,BB,,': ',AA,BB',
                        'AA,BB,--': ',AA,BB',
                        "": "",
                        "--": ""}
        expected_post = {'AA?': 'AA?',
                        'AA?, BBB': 'AA?,BBB',
                        'AA,BB,AA': 'AA,BB',
                        'AA,BB,,': '--,AA,BB',
                        'AA,BB,--': '--,AA,BB',
                        "": "--",
                        "--": "--"}
        
        for n, s, l, c in product(data, data, data, data):
            nl, sl, ll, cl = nslc_lists(self.todict(n, s, l, c))
            # now test asget and aspost in details:
            args = {'net': n, 'sta': s, 'loc': l, 'cha': c}
            pos = {'net': 0, 'sta': 1, 'loc': 2, 'cha': 3}
            for k, v in args.items():
                # test as get
                getstr = asget(nl, sl, ll, cl)
                if v is None or re.search('(?<!\\w)\\*(?!\\w)', v):
                    assert ("%s=" % k) not in getstr
                else:
                    assert "%s=%s" % (k, expected_get[v]) in getstr
                # test post:
                poststr = aspost(nl, sl, ll, cl)
                post_array = poststr.split(' ')
                if v is None or re.search('(?<!\\w)\\*(?!\\w)', v):
                    assert post_array[pos[k]] == '*'
                else:
                    assert post_array[pos[k]] == expected_post[v]
                    
    def test_nslc_errors(self):
        '''test un-consistent parameters (those which we can catch)'''
        data = ['HH?, !HH? ', ' !* ']
        
        for _ in range(4):
            args = [None, None, None, None]
            for d in data:
                args[_] = d
                with pytest.raises(Exception) as exc:
                    n, s, l, c =  nslc_lists(self.todict(*args))
        
        # test duplicates keyword given
        d = self.todict([], [], [], [])
        for key1, key2 in [['net', 'network'], ['sta', 'station'], ['loc', 'location'],
                    ['cha', 'channel']]:
            dd = dict(d)
            dd[key2] = d[key1]
            with pytest.raises(Exception) as exc:
                n, s, l, c =  nslc_lists(self.todict(*args))
            dd = dict(d)
            dd[key2+'s'] = d[key1]
            with pytest.raises(Exception) as exc:
                n, s, l, c =  nslc_lists(self.todict(*args))
                    
    def test_nslc_empty(self):
        '''test a value specified but empty does NOT default to 'TAKE ALL' '''
        for key, index in {'net': 0, 'sta': 1, 'loc': 2, 'cha': 3}.items():
            args = [None, None, None, None]
            args[index] = ''
            n,s, l, c = nslc_lists(self.todict(*args))
            assert ("%s=" % key) == asget(n,s, l, c)
            for idx, _ in enumerate(aspost(n,s, l, c).split(' ')):
                if idx == index:
                    assert _ == '--'
                else:
                    assert _ == '*'
                    
    def test_df(self):
        a = ['ABA', 'ABB', 'AAA', '']
        d = pd.DataFrame(list(product(a,a,a,a)),
                         columns=[Station.network.key, Station.station.key, Channel.location.key,
                                  Channel.channel.key])

        # test simple filter:
        for i in range(4):
            data = [None] * 4
            data[i] = 'AA?'
            purgedf = aspdfilter(*nslc_lists(self.todict(*data)))
            newd = purgedf(d, True)
            # we did NOT specify negative matches, so purgedf should return the original dataframe:
            assert len(newd) == len(d)
            
            # now try with a negative pattern:
            data[i] = '!A*'
            # the above removes ALL rows except those with a column with spaces
            purgedf = aspdfilter(*nslc_lists(self.todict(*data)))
            newd = purgedf(d, True)
            assert len(newd) == len(d) / 4
            
            # try with a less strict filter:
            data[i] = '!AB*'
            # the above removes ALL rows except those with a column with spaces
            purgedf = aspdfilter(*nslc_lists(self.todict(*data)))
            newd = purgedf(d, True)
            assert len(newd) == len(d) / 2
            
            # what if we specify '!' (not empty) should work isn't it?
            data[i] = '!'
            # the above removes ALL rows except those with a column with spaces
            purgedf = aspdfilter(*nslc_lists(self.todict(*data)))
            newd = purgedf(d, True)
            assert len(newd) == 3 * len(d) / 4
            
            
            
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()