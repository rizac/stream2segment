'''
Created on Sep 5, 2017

@author: riccardo
'''
from __future__ import print_function
from os.path import join, dirname, isfile
import numpy as np
from stream2segment.download.traveltimes.ttloader import TTTable
from itertools import izip, product, count
from stream2segment.download.utils import get_min_travel_time


def test():
    '''Tets the models we have in test/data'''
    NSAMPLES = 1

    iasp91_05sec = TTTable("/Users/riccardo/work/gfz/projects/sources/python/stream2segment/stream2segment/resources/traveltimestables/iasp91_05sec.npz")
    iasp91_10sec = TTTable("/Users/riccardo/work/gfz/projects/sources/python/stream2segment/stream2segment/resources/traveltimestables/iasp91_10sec.npz")

    s = str(iasp91_05sec)
    j = 9
    # get source depths from a normal distribution where the standard deviation is 230 (km)
    # in order to have three standard dev (99.7%) at 690 km
    # sources_depth_km = np.concatenate(([0, 700],
    #                                  np.abs(np.random.normal(loc=0.0, scale=200, size=NSAMPLES))))
    # get source depths from a normal distribution where the standard deviation is 200 (km)
    # in order to have three standard dev (99.7%) at 90 deg
    # distances_deg = np.concatenate(([0, 180],
    #                               np.abs(np.random.normal(loc=0.0, scale=30, size=NSAMPLES))))
    # sources_depth_km.sort()
    # distances_deg.sort()

    # get source depths and distances in degree according roughly to the distribution
    # of our db
    sources_depth_km = np.array([0, 1, 2, 3, 4, 5, 10, 20, 30, 100, 300, 500, 700], dtype=float)
    distances_deg = np.array([0, 0.5, 1, 2, 3, 4, 5, 10, 30, 90, 180], dtype=float)

    # add some noise, except the first and last elements which are boundaries
    sources_depth_km[1:-1] = sources_depth_km[1:-1] + (sources_depth_km[1:-1]/1000.0) * (2*np.random.random(len(sources_depth_km)-2)-0.5)
    distances_deg[1:-1] = distances_deg[1:-1] + (distances_deg[1:-1]/1000.0) * (2*np.random.random(len(distances_deg)-2)-0.5)
    

    values = np.array([(s, 0.0, d) for s, d in product(sources_depth_km, distances_deg)])

    iasp91_10sec_linear = iasp91_10sec.min(values[:, 0], values[:, 1], values[:, 2], method='linear')
    iasp91_05sec_linear = iasp91_05sec.min(values[:, 0], values[:, 1], values[:, 2], method='linear')
    iasp91_10sec_nearest = iasp91_10sec.min(values[:, 0], values[:, 1], values[:, 2], method='nearest')
    iasp91_05sec_nearest = iasp91_05sec.min(values[:, 0], values[:, 1], values[:, 2], method='nearest')
    iasp91_10sec_cubic = iasp91_10sec.min(values[:, 0], values[:, 1], values[:, 2], method='cubic')
    iasp91_05sec_cubic = iasp91_05sec.min(values[:, 0], values[:, 1], values[:, 2], method='cubic')
    normal = np.array([get_min_travel_time(s, d, traveltime_phases=('ttp+',),
                                           receiver_depth_in_km=0,
                                           model='iasp91') for s, d in product(sources_depth_km,
                                                                            distances_deg)])
    
    
    err05l, err05n, err05c = np.abs(normal-iasp91_05sec_linear), np.abs(normal-iasp91_05sec_nearest), np.abs(normal-iasp91_05sec_cubic)
    err10l, err10n, err10c = np.abs(normal-iasp91_10sec_linear), np.abs(normal-iasp91_10sec_nearest), np.abs(normal-iasp91_10sec_cubic)
    
    # print
    print("iasp91. Rows: error (theoretical error)")
    print("---")
    print("%19s %19s %19s %19s" % ("", "linear", "nearest", "cubic"))
    print("---")
    print("%19s %19s %19s %19s" % ("err=05 max", str(np.nanmax(err05l)), str(np.nanmax(err05n)), str(np.nanmax(err05c))))
    print("%19s %19s %19s %19s" % ("err=10 max", str(np.nanmax(err10l)), str(np.nanmax(err10n)), str(np.nanmax(err10c))))
    print("---")
    print("%19s %19s %19s %19s" % ("err=05 median", str(np.nanmedian(err05l)), str(np.nanmedian(err05n)), str(np.nanmedian(err05c))))
    print("%19s %19s %19s %19s" % ("err=10 median", str(np.nanmedian(err10l)), str(np.nanmedian(err10n)), str(np.nanmedian(err10c))))
    print("---")
    print("%19s %19s %19s %19s" % ("err=05 mean", str(np.nanmean(err05l)), str(np.nanmean(err05n)), str(np.nanmean(err05c))))
    print("%19s %19s %19s %19s" % ("err=10 mean", str(np.nanmean(err10l)), str(np.nanmean(err10n)), str(np.nanmean(err10c))))

    print('')
    print('On model 05sec:')
    print('')
    # print error table
    coords = np.around(values, 3)
    vals = np.around(np.abs(normal-iasp91_05sec_cubic), 3)
    fcollen = 12
    ret = []
    print(" ".join(["Src depth".rjust(fcollen, ' '), "Error(s):".rjust(fcollen, ' ')]))
    for i, c, v in izip(count(), coords, vals):
        if i % len(distances_deg) == 0:
            ret.append("\n")
            ret.append(str(c[0]).rjust(fcollen, ' '))
        ret.append(str(np.around(v, 3)).rjust(fcollen, ' '))
    print(" ".join(ret))
    print('Dist (deg) ->'.rjust(fcollen+1, ' ') + " ".join(str(np.around(d, 3)).rjust(fcollen, ' ') for d in np.around(distances_deg,3)))

if __name__ == '__main__':
    test()