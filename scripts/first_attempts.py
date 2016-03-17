
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')


# In[47]:

#import obspy.signal.trigger #import classicSTALTA
import numpy as np
import matplotlib.pyplot as plt
import math as M
from obspy.core import read
from obspy.signal.trigger import classicSTALTA
from obspy.signal.trigger import plotTrigger
from obspy.signal.util import smooth
from obspy.signal.filter import envelope
from obspy.imaging.spectrogram import spectrogram
from pyhht.visualization import plot_imfs
from pyhht.emd import EMD
from matplotlib import mlab
from obspy.core.util import getMatplotlibVersion
get_ipython().magic(u'matplotlib inline')

rep='/home/reakt/EPOSpro/stream2segment/example/mseeds/'
#filen='ev-20160307_0000063-GVD-HHX.mseed'  #ok
#filen='ev-20160307_0000040-SFS-BHX.mseed'  #noise
filen='ev-20160307_0000026-IMMV-HHX.mseed'  #doppio close large
#filen='ev-20160307_0000005-IMMV-HNX.mseed'  #doppio small later
#filen='ev-20160307_0000026-KTHA-HHX.mseed'
#filen='ev-20160308_0000008-UPC-HHX.mseed'

import os

str=os.path.join(rep, filen)
print(str)
data=read(str)
dt=data[1].stats.delta  #check if df already defined
print(dt)
df=1./dt
t1=60*dt

print(dt)
print(len(data[1].data))
data.plot()
#trN=data[1].copy()

trNfil=data.copy()
#trNfil.filter('highpass',freq=0.5,corners=2,zerophase=True)
trNfil.filter('bandpass',freqmin=1,freqmax=20,corners=2,zerophase=True)
trNfil.taper(type='cosine',max_percentage=0.05)
cft =classicSTALTA(trNfil[2].data, int(4 / dt), int(30 / dt))

envel = envelope(trNfil[2].data)
plt.plot(trNfil[2].data)
plt.plot(envel,c="r")
plt.xlim(7500, 8000)
plt.show



# In[48]:

tmp=trNfil[0].copy()
tmp.spectrogram(log=True)



# In[4]:

def _nearestPow2(x):
    """
    Find power of two nearest to x

    >>> _nearestPow2(3)
    2.0
    >>> _nearestPow2(15)
    16.0

    :type x: Float
    :param x: Number
    :rtype: Int
    :return: Nearest power of 2 to x
    """
    a = M.pow(2, M.ceil(np.log2(x)))
    b = M.pow(2, M.floor(np.log2(x)))
    if abs(a - x) < abs(b - x):
        return a
    else:
        return b


# In[49]:

MATPLOTLIB_VERSION = getMatplotlibVersion()
npts=len(tmp)
print(npts)
wlen=1
per_lap=0
nfft = int(_nearestPow2(wlen * df))
if nfft > npts:
   nfft = int(_nearestPow2(npts / 8.0))
print(nfft)
mult=2
mult = int(_nearestPow2(mult))
mult = mult * nfft
nlap = int(nfft * float(per_lap))

if MATPLOTLIB_VERSION >= [0, 99, 0]:
    specgram, freq, time = mlab.specgram(tmp, Fs=df, NFFT=nfft,noverlap=nlap)
    #specgram, freq, time = mlab.specgram(tmp, Fs=df, NFFT=nfft,pad_to=mult, noverlap=nlap)
else:
    specgram, freq, time = mlab.specgram(tmp, Fs=df,NFFT=nfft,noverlap=nlap)
    #specgram, freq, time = mlab.specgram(tmp, Fs=df,NFFT=nfft, noverlap=nlap)
#print(time)
ha = plt.subplot(111)
ha.pcolor(time, freq, specgram)
ha.set_ylim([0,10])
ha.set_xlim([50,150])
#ha.set_yscale('log')
plt.show()



vtime=[i*dt for i in range(len(tmp))]
ax1 = plt.subplot(211)
plt.plot(vtime,tmp)   # 
#plt.xlim([0,15])
ha=plt.subplot(212 )
#ha.set_yscale('log')
Pxx, freqs, bins, im = plt.specgram(tmp,Fs=df,NFFT=nfft,noverlap=nlap)

plt.show()


# In[51]:

print(specgram.dtype)
print(specgram.shape)
print(freq.shape)
print(time.shape)
fmin=2
fmax=10
tmin=60
tmax=150

indxf=[i for i in list(range(len(freq))) if (freq[i]>=fmin)&(freq[i]<=fmax)]
indxt=[i for i in list(range(len(time))) if (time[i]>=tmin)&(time[i]<=tmax)]
print(len(indxt))
#print(indxf)
#print(freq[indxf])
indf1=min(indxf)
indf2=max(indxf)

indt1=min(indxt)
indt2=max(indxt)



ener=[0]*len(indxt)
print(len(ener))
ii=0
for i in indxt:
    #print(time[i])
    dum=specgram[indxf,i]**2
    ener[ii]=dum.sum()
    ii=ii+1
    #print(time[i])
    #print(aa)
    #ene[i]=dum.sum
    #print(ene[i])
#print(ener)
enerdb=10*np.log10(ener)
ener=ener/max(ener)
plt.plot(time[indxt],ener)
plt.show()
plt.plot(vtime,tmp)
plt.xlim(tmin,tmax)
plt.show()


# In[33]:

npt=len(trNfil[0].data)
emd=EMD(trNfil[0].data)
imfs = emd.decompose()
time=(np.linspace(1,npt,npt))*dt
plt.rcParams['figure.figsize'] = (30.0, 50.0)
plot_imfs(trNfil[0].data, time, imfs)


# In[44]:

aa=trNfil[2].copy()
plotTrigger(aa, cft, 2.2, 0.5)
dm=len(cft)
item=[i for i in list(range(dm)) if cft[i]>2.2]
print(min(item))


ene=[0]*dm
for i in xrange(1, dm):
    #ene[i] = ene[i-1] + envel[i] ** 2
    ene[i] = ene[i-1] + trNfil[2].data[i] ** 2
    
    #print(ene[i])
#print(ene/dm)
enen=ene/ene[-1]
enend=np.diff(enen)
item1=[i for i in list(range(len(ene))) if enen[i]>0.1]
istart=min(item1)
print(min(item1))
item2=[i for i in list(range(len(ene))) if enen[i]<0.75]
iend=max(item2)
print(max(item1))

plt.plot(ene/ene[-1])
plt.show()

plt.plot(enend[5000:15000])
plt.show()

pr=smooth(enend,100)
plt.plot(pr[5000:10000])
plt.show()

yhat = savitzky_golay(pr, 801, 3)
plt.plot(yhat[5000:15000])
plt.show()


# In[39]:

maxima_num=0
minima_num=0
max_locations=[]
min_locations=[]
count=0
gradients=np.diff(yhat)
for i in gradients[:-1]:
    count+=1
    if ((cmp(i,0)>0) & (cmp(gradients[count],0)<0) & (i != gradients[count])):
        maxima_num+=1
        max_locations.append(count)     
    if ((cmp(i,0)<0) & (cmp(gradients[count],0)>0) & (i != gradients[count])):
        minima_num+=1
        min_locations.append(count)


turning_points = {'maxima_number':maxima_num,'minima_number':minima_num,'maxima_locations':max_locations,'minima_locations':min_locations}  

#print turning_points

plt.plot(yhat)
plt.plot(max_locations,yhat[max_locations],"o")
plt.plot(min_locations,yhat[min_locations],"o",c="r")

plt.plot(istart,yhat[istart],"x",c="k")
plt.plot(iend,yhat[iend],"x",c="k")
plt.xlim(5000, 15000)
plt.show()


# In[36]:

nmax=[max_locations[i] for i in range(len(max_locations)) if ((max_locations[i] > istart) & (max_locations[i] < iend))]
nmin=[min_locations[i] for i in range(len(min_locations)) if ((min_locations[i] > istart) & (min_locations[i] < iend))]
print(len(nmax))
print(len(nmin))
#print(pippo)
#print(istart)
#print(iend)
#print((max_locations))
#type(istart)
print(nmax)
print(nmin)


# In[130]:

# Now let's plot the raw and filtered data...
t = np.arange(0, trNfil[1].stats.npts / trNfil[1].stats.sampling_rate, trNfil[1].stats.delta)
plt.subplot(211)
plt.plot(t, data[1].data, 'k')
plt.ylabel('Raw Data')
plt.subplot(212)
plt.plot(t, trNfil[1].data, 'k')
plt.ylabel('Highpassed Data')
plt.xlabel('Time [s]')
plt.suptitle(trNfil[1].stats.starttime)
plt.show()


# In[29]:

dt=data[1].stats.delta
print(dt)


# In[133]:

print(data[0])


# In[40]:

stp=trNfil.copy()
stp.taper(type='cosine',max_percentage=0.05)


# In[7]:

#spectrogram.func_globals


# In[5]:

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


# In[22]:

time=(np.linspace(1,10,10))*0.01
time1=time*0.01
print(time)


# In[23]:

print(dt)


# In[103]:




# In[ ]:




# In[ ]:



