'''
Created on Nov 23, 2016

@author: riccardo
'''
import pandas as pd
import uuid
import time


def test_pandas_loc(N_rows):
    uuids = [str(uuid.uuid4()) for _ in xrange(N_rows)]
    d1 = pd.DataFrame(index=uuids,
                      columns=['A', 'B'],
                      data=None)

    d2 = pd.DataFrame({'A': uuids, 'B': [None] * len(uuids)})

    start = time.time()
    for x in uuids:
        d1.loc[x, 'B'] = 'a'
    end = time.time()
    t1 = end - start

    start = time.time()
    for x in uuids:
        d2.loc[d2['A'] == x, 'B'] = 'a'
    end = time.time()
    t2 = end - start

    return t1, t2


if __name__ == '__main__':
    for N in (10, 100, 1000, 10000):
        t1, t2 = test_pandas_loc(N)
        print "df len: %d: %s: %f, %s: %f" % (N, "locating on index", t1, "locating on column", t2)