'''
Created on Jul 6, 2016

@author: riccardo
'''

from obspy.signal.util import smooth

# CREATE SMOOTHED DERIVATIVE OF THE CUMULATIVE

aa=trNfil[2].copy()
cft =classicSTALTA(aa.data, int(4 / dt), int(30 / dt))
plotTrigger(aa, cft, 2.2, 0.5,show=False)
if (flagSave==1):
    plt.savefig("stalta.png")
plt.show()
dm=len(cft)
item=[i for i in list(range(dm)) if cft[i]>2.2]
print(min(item))
#########################################################################################################################
tmp=trNfil[0].data
l=len(tmp)
ene=[0]*l
for i in xrange(1, l):
    #ene[i] = ene[i-1] + envel[i] ** 2
    ene[i] = ene[i-1] + tmp[i] ** 2
    
    #print(ene[i])
#print(ene/dm)
enen=ene/ene[-1]    #normalized cumulative

enend=np.diff(enen)   #derivative for checking local extremes

soglia1=0.1
soglia2=0.9

item1=[i for i in list(range(len(ene))) if enen[i]>soglia1]
istart=min(item1)
print(min(item1))
item2=[i for i in list(range(len(ene))) if enen[i]<soglia2]
iend=max(item2)
print(max(item1))

ratio=ene/ene[-1]
fig=plt.figure()
ax=fig.gca()
plt.plot(tempo,ratio)
plt.xlabel("time [s]")
plt.ylabel("Cumulative squared amplitude")
plt.plot(tempo[iend],ratio[iend],marker="*",c="r",markersize=20,fillstyle="full")
plt.plot(tempo[istart],ratio[istart],marker="*",c="r",markersize=20,fillstyle="full")
ax.text(tempo[istart]+10,ratio[istart], r'10%', color='red',fontsize=15,bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})
ax.text(tempo[iend]+10,ratio[iend], r'90%', color='red',fontsize=15,bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})

plt.grid()
if (flagSave==1):
    fig.savefig('cumul.png')
plt.show()

fig=plt.figure()
plt.plot(tempo[:-1],enend)
ax=plt.gca()
plt.xlabel("time [s]")
plt.ylabel("first derivative of cumulative")
plt.grid()
if (flagSave==1):
    plt.savefig("derivCumuOrig.png")
plt.show()

pr=smooth(enend,100)
plt.plot(pr[5000:10000])
plt.show()

yhat = savitzky_golay(pr, 801, 3)
plt.plot(yhat[5000:15000])
plt.show()



# CREATE MAXIMA AND MINIMA:


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

plt.plot(tempo[:-1],yhat,c="k",lw=3)
plt.plot(tempo[max_locations],yhat[max_locations],"o",markersize=10)
plt.plot(tempo[min_locations],yhat[min_locations],"o",c="b",markersize=10)
plt.plot(tempo[iend],yhat[iend],marker="*",c="r",markersize=20,fillstyle="full")
plt.plot(tempo[istart],yhat[istart],marker="*",c="r",markersize=20,fillstyle="full")
#plt.plot(istart,yhat[istart],"x",c="k")
#plt.plot(iend,yhat[iend],"x",c="k")
plt.xlim(50, 150)
plt.xlabel("Time [s]")
plt.grid()
plt.savefig("derCumFiltExtreme.png")
plt.show()
