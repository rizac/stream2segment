'''
Created on Jul 25, 2016

@author: jessie mayor
'''

#!/usr/local/EPD7.3-2/bin/python

# selection automatique de "clean coda"

from obspy import read
import numpy as np
from obspy.signal.filter import bandpass
from obspy.signal.trigger import classic_sta_lta
import scipy as sc
from stream2segment.analysis.mseeds import stream_compliant

################# function for smooth the signal ######

#def mysmooth(signal,fm,cycle,dt): #cycle=nombre de periode "moyenne" dans une fenetre; signal est un array, fm=freq moyenne du signal (filtre), dt=pas d'echant
# signal=list(signal)
# window=cycle*1/float(fm) #longueur de la fenetre doit etre plus grande que la periode moyenne (filtree) = 1/fm ou fm=fmax-fmin 
# npts=window/dt #nombre de points dans la fenetre glissante=duree de la fenetre divise par le pas de tps
# signal_smooth=[]
# for i in range(len(signal)):  #data c'est chaque point du signal
#  fin=i+npts
#  signal_smooth.append(np.mean(signal[i:int(fin)]))
# return (signal_smooth)


def mysmooth(signal, time, fm, cycle, dt):  # cycle=nombre de periode "moyenne" dans une fenetre; signal est un array, fm=freq moyenne du signal (filtre), dt=pas d'echant
    """
        FIXME: write README

        :param signal:
        :type signal:
        :param time:
        :type time:
        :param fm:
        :type fm:
        :param cycle:
        :type cycle:
        :param dt:
        :type dt:
    """
    signal = list(signal)
    window = cycle*1/float(fm)  # longueur de la fenetre en temps 
    npts = int(window/dt)  # nombre de points dans la fenetre glissante=duree de la fenetre divise par le pas de tps
    signal_smooth = []
    time_smooth = []
    for i in range(0, len(signal)-npts/2, npts/2):  # data c'est chaque point du signal
        fin = i+npts
        signal_smooth.append(np.mean(signal[i:int(fin)]))
        time_smooth.append(time[i+npts/2])
    return (signal_smooth,np.array(time_smooth))


def group(indices_list):
    """
        FIXME: write doc
        :param indices_list: asdasdadfrb
        :type indices_list:
    """
    first = last = indices_list[0]
    for n in indices_list[1:]:
        if n - 1 == last:  # Part of the group, bump the end
            last = n
        else:  # Not part of the group, yield current group and start a new
            yield first, last
            first = last = n
    yield first, last  # Yield the last group


# def test_coda(tr, dt, window, rec):
#     # tr is the coda trace
#     tr = np.log10(tr)  # on travaille en log avec la coda pour avoir une pente
#     Npts = len(tr)  # nombre de point dans la coda
#     # window=5
#     # rec=2.5
#     wdw_npts = int(window/dt)  # nombre de pts dans la fenetre de 5 seconde
#     wdw_rec = int(rec/dt)  # nombre de point pour la fenetre de recouvrement
#     Nmax = np.floor(Npts/wdw_npts)  # borne maximale a atteindre pour faire des fenetres de 5 seconde  
#     start = 0
#     end = wdw_npts
# 
#     moy = [];
#     xmoy = []
#     k = 0
#     while end < Nmax*wdw_npts:
#         moy.append(abs(np.mean(tr[start:end])))
#         xmoy.append(k)
#         k = k+1
#         start = start+wdw_rec
#         end = end+wdw_rec
#     slope, intercept, R, pvalue, stderr = sc.stats.linregress(xmoy, moy)
#     ok = 1
#     print R
#     if R < 0.9:  # si on a pas une excellent regression lineaire alors on rejette
#         ok=0
# 
#     return ok

########### end function #############################


# fm = 6  # mean frequency, in Hz
# cycle = 10  # nombre de periode moyenne dans la fenetre glissante pour le smooth
# niveau_bruit = 16.
# Lw = 50  # coda window duration, en seconds
# noise_duration = 5  # en seconds beginning noise window
# subwdw_length = 5
# subwdw_length_rec = 2.5

# stream=read('20091217_231838.FR.ESCA.00.HHZ.SAC')  


@stream_compliant
def analyze_coda(trace, fm=6, cycle=10, niveau_bruit=16, Lw=50, noise_duration=5, subwdw_length=5,
                 subwdw_length_rec=2.5):
    """
        FIXME: write doc!


        NOTE: this function accepts also streams objects (see @stream_compliant decorator in
        stream2segments.mseeds)

        :param trace: an obspy.core.Trace object
        :return: a list of tuples of the form:
        (slope_start_time, slope, intercept, R, pvalue, stderr)
        where slope_start_time is an obspy UTCDateTime object. For the other values, see:
        http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.linregress.html
        for details
        :rtype: see return field
    """
    st = trace
    try:
        st.data = bandpass(st.data, freqmin=4, freqmax=8, df=st.stats. sampling_rate, corners=2)
    except ValueError:
        return None
    if (st.stats.npts*st.stats.delta) > 100:
        st.detrend('demean')  # on ramene le signal a 0
        energy = st.data * st.data
        t = st.times()
        st_smooth, t_smooth = mysmooth(energy, t, fm, cycle, st.stats.delta)
        imax = st_smooth.index(max(st_smooth))
        new_dt = round(t_smooth[1]-t_smooth[0], 2)
        sec = int(noise_duration/new_dt)  # on prend 10seconde de debut de signal
        bruit = st_smooth[0:sec]  # on prend 5 seconde pour la moyenne de bruit
        # df=st.stats.sampling_rate
        df = 1/new_dt
        cft = classic_sta_lta(bruit, nsta=2, nlta=5)  # valeur que j'ai prise= 2 et 5 (en echantillon)
        stalta = np.where(cft > 3)[0]  # valeur que j'ai prise =1.5
        if len(stalta) > 0:  # si on detecte effectivement du signal dans la fenetre de bruit: ca va pas
            return None  # on ne peut pas definir une bonne moyenne de bruit
        else:
            databruit = bruit
        #--------------------------------fin definition moyenne du bruit ----------------------------------------
        ###### duree de la coda = du maximum de l'enveloppe ------> ratio signal/bruit<4 ##############
        j = 0
        debut = imax
        fin = debut+int(subwdw_length/new_dt)  # on prend 5s de fenetre glissante
        # rec_window = new_dt/2.  # 50% de recouvrement
        Nrec = int(subwdw_length_rec/new_dt)  # nombre de pts de recouvrement : on choisit 2.5s
        ratio = []
        while j < len(st_smooth[imax:imax+int(Lw/new_dt)]):    
            ratio.append(np.mean(st_smooth[debut:fin])/np.mean(databruit))
            j = j+Nrec
            debut = debut+Nrec
            fin = debut+int(subwdw_length/new_dt)
        indok = np.where(np.array(ratio) > niveau_bruit)[0]  # ou est ce que le signal dans les 80s de fenetre de coda est superieur au niveau de bruit
        ret_vals = None
        if len(indok) > 0:
            doublons = list(group(indok))
            if (len(doublons) == 1) and (doublons[0][-1] == len(ratio)-1) or (doublons[0][0] == 0) \
                and (doublons[0][-1] == len(ratio)-1):  # ca veut dire qu'il detecte une coda ou du moins un ratio>4 et on choisi une longueur de  au moins 20 seconde
                coda = st_smooth[imax:imax+int(Lw/new_dt)]  # donnee lissee
                tcoda = t_smooth[imax:imax+int(Lw/new_dt)]
                # raw=st.data[imax:imax+int(Lw/new_dt)]# donnee brut
                ##### test sur la coda pour voir si on a bien une "pente" : on joue avec le coeff de correlation

                # tr is the coda trace
                coda = np.log10(coda)  # on travaille en log avec la coda pour avoir une pente
                Npts = len(coda)  # nombre de point dans la coda
                # window=5
                # rec=2.5
                wdw_npts = int(subwdw_length / new_dt)  # nombre de pts dans la fenetre de 5 seconde
                wdw_rec = int(subwdw_length_rec / new_dt)  # nombre de point pour la fenetre de recouvrement
                Nmax = np.floor(Npts/wdw_npts)  # borne maximale a atteindre pour faire des fenetres de 5 seconde  
                start = 0
                end = wdw_npts

                moy = [];
                xmoy = []
                k = 0
                while end < Nmax * wdw_npts:
                    moy.append(np.mean(coda[start: end]))
                    xmoy.append(k)
                    k = k + 1
                    start = start + wdw_rec
                    end = end + wdw_rec
                slope, intercept, R, pvalue, stderr = sc.stats.linregress(xmoy, moy)
                start_time = st.stats.starttime + t_smooth[imax]
                ret_vals = (start_time, slope, intercept, R, pvalue, stderr)

        return ret_vals

#                     return slope, intercept, R, pvalue, stderr
#                     ok = 1
#                     print R
#                     if R < 0.9:  # si on a pas une excellent regression lineaire alors on rejette
#                         ok=0
#                 
#                     return ok
#                     
#                     
#                     
#                     
#                     test = test_coda(coda,new_dt,subwdw_length,subwdw_length_rec) # si le test est a 0 c'est qu'on decroit pas, si le test est a 1 c'est que la trace semble bonne
#                     if test == 1:
#                         plt.semilogy(t_smooth,st_smooth,'k')
#                         plt.semilogy(tcoda,coda,'b')
#                         plt.semilogy(t_smooth[0:sec],databruit,'m')
#                         plt.show()
#                     del test
                         























