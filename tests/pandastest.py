'''
Created on Apr 22, 2016

@author: riccardo
'''
import time
if __name__ == '__main__':
    # pandas test. When vectorization is not possible (the fastest one)
    # here some performance with
    # 1: iteration over len(rows_)
    # 2 iteration via dataframe.iterrows
    # 3 iteration with apply
    
    # performances: (seconds)
    
    # =============================================================================================
    # N = 50 (L=100)  N=100 (L=100)    N=50, L=1000  N=5, L=10000
    # =============================================================================================
    # 2.6013 seconds  5.1241 seconds    25.76        25.772
    # 0.9552 seconds  1.8251 seconds    9.014        9.12
    # 0.0065 seconds  0.0063 seconds    0.054        0.55
    # =============================================================================================
    
    N = 5
    L = 10000
    cols= range(L)
    import pandas as pd
    import numpy as np
    # test 1: normal Case
    d = pd.DataFrame({'str' : ["abc" for x in cols],
                       'int': [4 for x in cols],
                       'float' :[4.5 for x in cols]})
    
    
    def f(row):
        return row['int']
    
    
    # try1
    t = time.time()
    for j in xrange(N):
        val = np.zeros(L)
        for i in xrange(L):
            val[i] = d.loc[i, 'int']
    t = time.time() - t
    print "loop on length(df): %s seconds" % str(t)
    v1 = val

    # try1
    t = time.time()
    for j in xrange(N):
        val = np.zeros(L)
        for i, row in d.iterrows():
            val[i] = row['int']
    t = time.time() - t
    print "iterrows: %s seconds" % str(t)
    v2 = val

    # try1
    t = time.time()
    val = d.apply(f, axis=1)
    t = time.time() - t
    print "apply: %s seconds" % str(t)
    v3 = val


    assert np.array_equal(v1, v2) and np.array_equal(v1, v3)