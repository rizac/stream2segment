'''
Created on Aug 23, 2017

@author: riccardo
'''
from multiprocessing import Pool

import numpy as np
from obspy.taup.helper_classes import SlownessModelError
from obspy.taup.tau_model import TauModel

from stream2segment.download.utils import get_min_travel_time, get_taumodel
import time
import math
from datetime import timedelta
from itertools import izip

# global vars
SD_MAX = 700  # in km
RD_MAX = 2  # in km
DIST_MAX = 20  # in degrees
MAX_TIME_ERR_TOL = 0.5  # in seconds
PWAVEVELOCITY = 6  # in km/sec
DEG2KM = 110
# space = (DIST_STEP * DEG2KM) / time = MAX_TIME_ERR_TOL = velocity = PWAVEVELOCITY
# then calculate the distance step, in degrees, rounded to the third decimal place (min=0.001 deg)
# this gives the maximum granularity of travel times every 110 meters (not below). Increase to 4
# if you want to go down to 11 meters, and so on
DIST_STEP = round((PWAVEVELOCITY * MAX_TIME_ERR_TOL) / DEG2KM, 3)


# array retrieved from global vars above:
_SOURCE_DEPTHS = np.concatenate((np.arange(0, 50, 1), np.arange(500, SD_MAX, 2.5), [SD_MAX]))
_RECEIVER_DEPTHS = np.concatenate((np.arange(0, 0.006, 0.001), np.arange(.01, .1, .01),
                                   np.arange(.1, RD_MAX, .1), [RD_MAX]))
_DISTANCES = np.concatenate((np.arange(0, DIST_MAX, DIST_STEP), [DIST_MAX]))
# the indices used for comparison (calculate only a portion of distances for speed reasons):
_CMP_DIST_INDICES = np.array([0, 11, 21, 31, 101, len(_DISTANCES)-1])
# the remaining indices (when two arrays are not close according to _CMP_DIST_INDICES, calculate
# remaining one and build the complete travel times array):
_REM_DIST_INDICES = np.array(sorted(list(set(xrange(len(_DISTANCES))) - set(_CMP_DIST_INDICES))))


def ttequal(traveltimes1, traveltimes2, maxerr):
    '''function defining if two traveltimes array are equal. Uses np.allclose'''
    return np.allclose(traveltimes1, traveltimes2, rtol=0, atol=maxerr, equal_nan=True)


def min_traveltime(model, source_depth_in_km, receiver_depth_in_km, distance_in_degree):
    try:
        return get_min_travel_time(source_depth_in_km, distance_in_degree, ("ttp+",),
                                   receiver_depth_in_km=receiver_depth_in_km, model=model)
    except (ValueError, SlownessModelError):
        return np.nan


# def _min_traveltime(model, source_depth_in_km, receiver_depth_in_km, distance_in_degree,
#                     index, array):
#     array[index] = min_traveltime(model, source_depth_in_km, receiver_depth_in_km,
#                                   distance_in_degree)


def newarray(basearray):
    return np.full(basearray.shape, np.nan)


def mp_callback(index, array):
    def _(result):
        array[index] = result
    return _


def itercreator(modelname,
                source_depths_in_km=_SOURCE_DEPTHS,
                receiver_depths_km=_RECEIVER_DEPTHS,
                max_abs_err_tol_in_sec=MAX_TIME_ERR_TOL):
    # set an array of distances whereby we compute if two traveltimes are equal.
    # The more the points, the more accuracy, but the slower the computation
    distances_in_degree = np.asarray(_DISTANCES)
    cmp_distances_in_degree = distances_in_degree[_CMP_DIST_INDICES]
    rem_distances_in_degree = distances_in_degree[_REM_DIST_INDICES]
    source_depths_in_km = np.asarray(source_depths_in_km)
    receiver_depths_km = np.asarray(receiver_depths_km)

    # as numpy allclose returns true if the error is <= max_abs_err_tol_in_sec, we want a strict
    # inequality, thus:
    max_err = abs(max_abs_err_tol_in_sec)
    max_err -= max_err * 0.00001

    model = TauModel.from_file(modelname)
    leng = len(source_depths_in_km) * len(receiver_depths_km)
    last_saved_traveltimes = None
    len_rds = len(receiver_depths_km)

    count = 0
    for sd in source_depths_in_km:
        args = [(sd, rd) for rd in receiver_depths_km]

        last_ttimes = newarray(cmp_distances_in_degree)
        pool = Pool()
        sd, rd = args[-1]
        for i, d in enumerate(cmp_distances_in_degree):
            pool.apply_async(min_traveltime, (model, sd, rd, d),
                             callback=mp_callback(i, last_ttimes))
        pool.close()
        pool.join()

        if last_saved_traveltimes is not None and \
                ttequal(last_saved_traveltimes, last_ttimes, max_err):
            count += len_rds
            continue

        # need to calculate all receiver depths:
        pool = Pool()
        ttimes = []
        for (sd, rd) in args[:-1]:
            last_ttimes = newarray(cmp_distances_in_degree)
            ttimes.append(last_ttimes)
            for i, d in enumerate(cmp_distances_in_degree):
                pool.apply_async(min_traveltime, (model, sd, rd, d),
                                 callback=mp_callback(i, last_ttimes))
        pool.close()
        pool.join()

        ttimes.append(last_ttimes)

        for (sd, rd), current_traveltimes in izip(args, ttimes):
            count += 1
            if last_saved_traveltimes is None or \
                    not ttequal(last_saved_traveltimes, current_traveltimes, max_err):
                real_maxerr = 0 if last_saved_traveltimes is None \
                    else np.nanmax(np.abs(last_saved_traveltimes - current_traveltimes))
                # calculate ALL traveltimes, set nan for element still to compute
                # the others are filled with the already computed ones
                complete_tt = newarray(distances_in_degree)
                pool = Pool()
                for i, d in izip(_REM_DIST_INDICES, rem_distances_in_degree):
                    pool.apply_async(min_traveltime, (model, sd, rd, d),
                                     callback=mp_callback(i, complete_tt))
                pool.close()
                pool.join()
                # add already calculated:
                complete_tt[_CMP_DIST_INDICES] = current_traveltimes
                yield sd, rd, complete_tt, real_maxerr, count, leng
                last_saved_traveltimes = current_traveltimes

#             if sd == source_depths_in_km[5]:
#                 break



# class Calculator(object):
#     '''callable class that calculates the given travel times'''
#     def __init__(self, model):
#         try:
#             self.m = TauModel.from_file(model)
#         except:
#             self.m = model
# 
#     def get_traveltime(self, source_depth_in_km, receiver_depth_in_km, distance_in_degree):
#         model = self.m
#         try:
#             return get_min_travel_time(source_depth_in_km, distance_in_degree, ("ttp+",),
#                                        receiver_depth_in_km=receiver_depth_in_km, model=model)
#         except (ValueError, SlownessModelError):
#             return np.nan
# 
#     def __call__(self, arg):
#         '''same as get_traveltimes but account for pool.map which accepts a single argument'''
#         sd, rd, distances = arg
#         return arg, np.array([self.get_traveltime(sd, rd, d) for d in distances])



def gettable(model, maxerr=MAX_TIME_ERR_TOL, isterminal=False):
    start = time.time()
    data = []
    sd_rd = []
    if isterminal:
        print("Calculating travel times for model '%s'" % str(model))
        print("The algorithm basically iterates over each source depth and receiver depths pair")
        print("   and saves the associated travel times array (t)")
        print("   when it exceeds %f seconds with the previous computed t" % maxerr)
        print("tc below refers to a chunk of each saved t showing the values at distances:")
        print("   %s" % (_DISTANCES[_CMP_DIST_INDICES].tolist()))

        # typical header should be (roughly):
        # "src_depth  rec_depth                     travel times (max err with previous)   done              eta"
        #        0.0        0.0  [0.0, 21.088, 36.402, 50.154, 146.266, 370.264] (0.000)  0.02%  19:15:33.394541

        frmt = "%9s  %9s  %55s  %5s  %16s"
        header = frmt % ('src_depth', 'rec_depth', 'tc (max err with previous tc)', '%done', 'eta')
        print("-"*len(header))
        print(header)
        print("-"*len(header))

    for sd, rd, tts, maxerr, count, total in itercreator(model):  # 'iasp91' 'ak135'
        if isterminal:
            percentdone = int(0.5 + (100.0 * count) / total)
            eta = int(0.5 + (total-count) * (float(time.time() - start) / count))
            tt_list = np.around(tts[_CMP_DIST_INDICES], decimals=1).tolist()  # round to 0.1 sec
            print(frmt % (sd, rd, '%s %.3f' % (tt_list, maxerr), percentdone,
                          str(timedelta(seconds=eta))))
#             print("%d) %.2f%% done, eta: %s" % (count, percentdone, str(timedelta(seconds=eta))))
#             print("src_depth=%s, rec_depth=%s, max_err=%.3f" % (sd, rd, maxerr))
#             print("travel_times(cmp)=%s" % str(np.around(tts[_CMP_DIST_INDICES],
#                                                          decimals=1).tolist()))
#             print("\n")
        sd_rd.append(sd)
        sd_rd.append(rd)
        data.append(tts)

    return np.reshape(sd_rd, newshape=(len(sd_rd)/2, 2)), \
        np.reshape(data, newshape=(len(data), len(data[0])))


if __name__ == '__main__':
    gettable('iasp91', isterminal=True)  # 'iasp91' 'ak135'

