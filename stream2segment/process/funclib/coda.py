"""
Module for coda analysis.
Not yet implemented (future feature in new versions)

Created on Jul 25, 2016

.. moduleauthor:: Jessie mMayor
.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
import numpy as np
import scipy

from obspy.signal.filter import bandpass
from obspy.signal.trigger import classic_sta_lta


# cycle=nombre de periode "moyenne" dans une fenetre; signal est un array,
# fm=freq moyenne du signal (filtre), dt=pas d'echant
def mysmooth(signal, time, fm, cycle, dt):
    """
        Return the envelop of the signal and its corresponding time, smoothed from natural
        variations thanks to an average moving window of length the number of cycle.
        Note that the signal is under-sampled depending on the number of cycles.

        :param signal: energy computed as the squared of the velocigram (from Obspy.core.Trace)
        :type signal: array (units depends on the input trace)
        :param time: time corresponding to the trace (st.times() for an Obspy trace object)
        :type time: array in seconds
        :param fm: mean frequency of the band passe filter
        :type fm: float in Hertz
        :param cycle: number of cycle in the moving window (1 cycle = 1 period)
        :type cycle: float number (adimensionnal)
        :param dt: sampling rate of the data
        :type dt: float in seconds
    """
    signal = list(signal)
    # longueur de la fenetre en temps
    window = cycle / fm
    # nombre de points dans la fenetre glissante=duree de la fenetre divise par le pas de tps
    npts = int(window // dt)
    helf_npts = npts // 2  # as int
    signal_smooth = []
    time_smooth = []
    for i in range(0, len(signal) - helf_npts, helf_npts):  # data c'est chaque point du signal
        end_ = i + npts
        signal_smooth.append(np.mean(signal[i:end_]))
        time_smooth.append(time[i + helf_npts])
    return (signal_smooth, np.array(time_smooth))


def group(indices_list):
    """
         Extract the first and the last part of a list components
        :param indices_list: list of indices
        :type indices_list: list
    """
    first = last = indices_list[0]
    for n in indices_list[1:]:
        if n - 1 == last:  # Part of the group, bump the end
            last = n
        else:  # Not part of the group, yield current group and start a new
            yield first, last
            first = last = n
    yield first, last  # Yield the last group


def analyze_coda(trace, fm=6, cycle=10, noise_level=16, Lw=50, noise_duration=5, subwdw_length=5,
                 subwdw_length_rec=2.5):
    """
        Return the correlation coefficient of the coda part of the signal : the onset of the coda
        is selected as the maximum amplitude time and the coda duration is Lw.

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
        sec = int(noise_duration // new_dt)  # on prend 10seconde de debut de signal
        noise = st_smooth[0:sec]  # on prend 5 seconde pour la moyenne de bruit
        # df=st.stats.sampling_rate
        # df = 1/new_dt

        # valeur que j'ai prise= 2 et 5 (en echantillon)
        cft = classic_sta_lta(noise, nsta=2, nlta=5)
        stalta = np.where(cft > 3)[0]  # valeur que j'ai prise =1.5
        # si on detecte effectivement du signal dans la fenetre de bruit: ca va pas
        if len(stalta) > 0:
            return None  # on ne peut pas definir une bonne moyenne de bruit
        else:
            noisedata = noise
        # ----fin definition moyenne du bruit ----------------------------------------
        # ##### duree de la coda = du maximum de l'enveloppe ------> ratio signal/bruit<4 #######
        j = 0
        start = imax
        end_ = start + int(subwdw_length // new_dt)  # on prend 5s de fenetre glissante
        # rec_window = new_dt/2.  # 50% de recouvrement
        n_rec = int(subwdw_length_rec // new_dt)  # nombre de pts de recouvrement : on choisit 2.5s
        ratio = []
        while j < len(st_smooth[imax:imax+int(Lw // new_dt)]):
            ratio.append(np.mean(st_smooth[start:end_]) / np.mean(noisedata))
            j = j+n_rec
            start = start+n_rec
            end_ = start + int(subwdw_length // new_dt)
        # ou est ce que le signal dans les 80s de fenetre de coda est superieur au niveau de bruit
        indok = np.where(np.array(ratio) > noise_level)[0]
        ret_vals = None
        if len(indok) > 0:
            doublons = list(group(indok))
            if (len(doublons) == 1) and (doublons[0][-1] == len(ratio)-1) or (doublons[0][0] == 0) \
                    and (doublons[0][-1] == len(ratio)-1):
                # ca veut dire qu'il detecte une coda ou du moins un ratio>4 et
                # on choisi une longueur de  au moins 20 seconde
                coda = st_smooth[imax:imax+int(Lw // new_dt)]  # donnee lissee

                # tcoda = t_smooth[imax:imax+int(Lw/new_dt)]

                # raw=st.data[imax:imax+int(Lw/new_dt)]# donnee brut

                # test sur la coda pour voir si on a bien une "pente" :
                # on joue avec le coeff de correlation

                # tr is the coda trace
                coda = np.log10(coda)  # on travaille en log avec la coda pour avoir une pente
                n_pts = len(coda)  # nombre de point dans la coda
                # window=5
                # rec=2.5

                # nombre de pts dans la fenetre de 5 seconde
                wdw_npts = int(subwdw_length // new_dt)
                # nombre de point pour la fenetre de recouvrement:
                wdw_rec = int(subwdw_length_rec // new_dt)
                # borne maximale a atteindre pour faire des fenetres de 5 seconde:
                n_max = int(n_pts // wdw_npts)
                start = 0
                end = wdw_npts

                means = []
                x_means = []
                k = 0
                while end < n_max * wdw_npts:
                    means.append(np.mean(coda[start: end]))
                    x_means.append(k)
                    k = k + 1
                    start = start + wdw_rec
                    end = end + wdw_rec
                slope, intercept, R, pvalue, stderr = scipy.stats.linregress(x_means, means)  # @UndefinedVariable
                start_time = st.stats.starttime + t_smooth[imax]
                ret_vals = (start_time, slope, intercept, R, pvalue, stderr)

        return ret_vals
