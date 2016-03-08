
# coding: utf-8

# In[7]:

from obspy.core import read
get_ipython().magic(u'matplotlib inline')
data=read('ev-20160105_0000089-XOR-HHX.mseed')
data.plot()


# In[9]:

sel=data[1]
print(sel.stats)


# In[10]:

print(data)


# In[ ]:



