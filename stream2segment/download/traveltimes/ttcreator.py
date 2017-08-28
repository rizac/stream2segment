'''
Created on Aug 23, 2017

@author: riccardo
'''
from multiprocessing import Pool
import time
from datetime import timedelta
from itertools import izip

import numpy as np
from click.termui import progressbar
from obspy.taup.helper_classes import SlownessModelError
from obspy.taup.tau_model import TauModel

from stream2segment.download.utils import get_min_travel_time

# global vars
SD_MAX = 700  # in km
RD_MAX = 1.2  # in km
DIST_MAX = 180  # in degrees
MAX_TIME_ERR_TOL = 0.5  # in seconds
# determine the distances step (in degrees). We need to set two variables first:
PWAVEVELOCITY = 5  # in km/sec
DEG2KM = 110
# then calculate the distance step, in degrees, rounded to the third decimal place (min=0.001 deg)
# this gives the maximum granularity of travel times every 110 meters (not below). Increase to 4
# if you want to go down to 11 meters, and so on
# space = (DIST_STEP * DEG2KM) / time = MAX_TIME_ERR_TOL = velocity = PWAVEVELOCITY
DIST_STEP = round((PWAVEVELOCITY * MAX_TIME_ERR_TOL) / DEG2KM, 3)


# array retrieved from global vars above:  DO NOT CHANGE VARIABLES BELOW UNLESS
# STUDY ON DISTRIBUTIONS OF SOURCE DEPTHS, RECEIVER DEPTHS OR DISTANCES SUGGESTS TO CHANGE THEM
_SOURCE_DEPTHS = np.concatenate((np.arange(0, 700, 1), [SD_MAX]))
_RECEIVER_DEPTHS = np.concatenate((np.arange(0, 0.006, 0.001), np.arange(.01, .1, .01),
                                   np.arange(.1, RD_MAX, .1), [RD_MAX]))
_DISTANCES = np.concatenate((np.arange(0, DIST_MAX, DIST_STEP), [DIST_MAX]))
# the indices used for comparison (calculate only a portion of distances for speed reasons):
_CMP_DIST_INDICES = np.array([0, 1,
                              int(len(_DISTANCES)/4), int(len(_DISTANCES)/2), len(_DISTANCES)-1])


def min_traveltimes(modelname, source_depth_km, receiver_depth_km, distances_in_deg,
                    traveltime_phases=("ttp+",)):
    model = taumodel(modelname)
    tts = []
    ds = []
    for d in distances_in_deg:
        ds.append(d)
        tts.append(min_traveltime(model, source_depth_km, receiver_depth_km, d,
                                  traveltime_phases))
    return ds, tts


def printtimes(modelname, source_depth_km, receiver_depth_km, all=False):
    ds, tts = min_traveltimes(modelname, source_depth_km, receiver_depth_km,
                              _DISTANCES if all else [_CMP_DIST_INDICES])
    print("distances: %s" % str(np.around(ds, decimals=3).tolist()))
    print("   ttimes: %s" % str(np.around(tts, decimals=3).tolist()))


def plottimes(modelname, source_depth_km, receiver_depth_km, distances_in_deg,
              traveltime_phases=("ttp+",)):
    import matplotlib
    bckend = matplotlib.get_backend()
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    ds, tts = min_traveltimes(modelname, source_depth_km, receiver_depth_km, distances_in_deg,
                              traveltime_phases)
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    ax.plot(ds, tts)
    plt.show(block=True)
    matplotlib.use(bckend)


def ttequal(traveltimes1, traveltimes2, maxerr):
    '''function defining if two traveltimes array are equal. Uses np.allclose'''
    return np.allclose(traveltimes1, traveltimes2, rtol=0, atol=maxerr, equal_nan=True)


def min_traveltime(model, source_depth_in_km, receiver_depth_in_km, distance_in_degree,
                   traveltime_phases=("ttp+",)):
    try:
        return get_min_travel_time(source_depth_in_km, distance_in_degree, traveltime_phases,
                                   receiver_depth_in_km=receiver_depth_in_km, model=model)
    except (ValueError, SlownessModelError):
        return np.nan


# def _min_traveltime(model, source_depth_in_km, receiver_depth_in_km, distance_in_degree,
#                     index, array):
#     array[index] = min_traveltime(model, source_depth_in_km, receiver_depth_in_km,
#                                   distance_in_degree)


def newarray(basearray):
    return np.full(basearray.shape, np.nan)


def mp_callback(index, array, bar=None):
    def _(result):
        array[index] = result
        if bar:
            bar.update(1)
    return _


def taumodel(model):
    try:
        return TauModel.from_file(model)
    except:
        return model


def itercreator(model,
                source_depths_in_km=_SOURCE_DEPTHS,
                receiver_depths_km=_RECEIVER_DEPTHS,
                max_abs_err_tol_in_sec=MAX_TIME_ERR_TOL):
    # set an array of distances whereby we compute if two traveltimes are equal.
    # The more the points, the more accuracy, but the slower the computation
    distances_in_degree = np.asarray(_DISTANCES)
    cmp_distances_in_degree = distances_in_degree[_CMP_DIST_INDICES]
    source_depths_in_km = np.asarray(source_depths_in_km)
    receiver_depths_km = np.asarray(receiver_depths_km)

    # as numpy allclose returns true if the error is <= max_abs_err_tol_in_sec, we want a strict
    # inequality, thus:
    max_err = abs(max_abs_err_tol_in_sec)
    max_err -= max_err * 0.00001

    model = taumodel(model)
    leng = len(source_depths_in_km) * len(receiver_depths_km)
    last_saved_traveltimes = None
    len_rds = len(receiver_depths_km)

    count = 0
    for sd in source_depths_in_km:  # [:int(len(source_depths_in_km)/10)]:
        args = [(sd, rd) for rd in receiver_depths_km]

        ttimes = np.full((len(args), len(cmp_distances_in_degree)), np.nan)

        # calculate first and last: if both are the same as the last computed, go on
        pool = Pool()
        for idx in [0, len(args)-1]:
            sd, rd = args[idx]
            tmp_ttimes = ttimes[idx]
            for i, d in enumerate(cmp_distances_in_degree):
                pool.apply_async(min_traveltime, (model, sd, rd, d),
                                 callback=mp_callback(i, tmp_ttimes))
        pool.close()
        pool.join()

        if last_saved_traveltimes is not None and \
                ttequal(last_saved_traveltimes, ttimes[0], max_err) and \
                ttequal(last_saved_traveltimes, ttimes[-1], max_err):
            count += len_rds
            continue

        # need to calculate all receiver depths:
        pool = Pool()
        for idx in xrange(1, len(args)-1):
            sd, rd = args[idx]
            tmp_ttimes = ttimes[idx]
            for i, d in enumerate(cmp_distances_in_degree):
                pool.apply_async(min_traveltime, (model, sd, rd, d),
                                 callback=mp_callback(i, tmp_ttimes))
        pool.close()
        pool.join()

        for (sd, rd), current_traveltimes in izip(args, ttimes):
            count += 1
            if last_saved_traveltimes is None or \
                    not ttequal(last_saved_traveltimes, current_traveltimes, max_err):
                real_maxerr = 0 if last_saved_traveltimes is None \
                    else np.nanmax(np.abs(last_saved_traveltimes - current_traveltimes))
                yield sd, rd, current_traveltimes, real_maxerr, count, leng
                last_saved_traveltimes = current_traveltimes


def get_sdrd_steps(model, maxerr=MAX_TIME_ERR_TOL, isterminal=False):
    start = time.time()
    data = []
    sd_rd = []
    if isterminal:
        print("Calculating source and receiver depth steps")
        print("- The algorithm basically iterates over each source depth and receiver depths pair")
        print("  and saves the associated travel times array (t)")
        print("  when it exceeds %f seconds with the previous computed t" % maxerr)
        print("- tc below refers to a chunk of each saved t showing the values at distances:")
        print("  %s" % (_DISTANCES[_CMP_DIST_INDICES].tolist()))

        # typical header should be (roughly):
        # "src_depth  rec_depth                     travel times (max err with previous)   done              eta"
        #        0.0        0.0  [0.0, 21.088, 36.402, 50.154, 146.266, 370.264] (0.000)  0.02%  19:15:33.394541

        frmt = "%4s %9s %9s %60s %5s %16s"
        header = frmt % ('#', 'src_depth', 'rec_depth', 'tc (max err with previous tc)', '%done',
                         'eta')
        print("-"*len(header))
        print(header)
        print("-"*len(header))

    # pool = Pool()
    # already_calculated_tt_indices = set(_CMP_DIST_INDICES)
    idx = 1
    for sd, rd, tts, maxerr, count, total in itercreator(model):  # 'iasp91' 'ak135'
        if isterminal:
            percentdone = int(0.5 + (100.0 * count) / total)
            eta = int(0.5 + (total-count) * (float(time.time() - start) / count))
            tt_list = np.around(tts, decimals=1).tolist()  # round to 0.1 sec
            print(frmt % (idx, sd, rd, '%s (%8.3f)' % (tt_list, maxerr), percentdone,
                          str(timedelta(seconds=eta))))
            idx += 1

        sd_rd.append(sd)
        sd_rd.append(rd)

        # NOW compute all remaining travel times. Use apply async so that we can go on inspecting
        # new source, receiver pairs
        complete_tts = newarray(_DISTANCES)
        data.append(complete_tts)
        # add already calculated:
        complete_tts[_CMP_DIST_INDICES] = tts

    return np.reshape(sd_rd, newshape=(len(sd_rd)/2, 2)), \
        np.reshape(data, newshape=(len(data), len(data[0])))


def computetts(model, sdrd_array, tts_matrix, isterminal=False):
    model = taumodel(model)
    numttpts = tts_matrix.shape[1]
    numtts = tts_matrix.shape[0]
    tocomputeperrow = np.isnan(tts_matrix[0])
    totalpts2compute = np.sum(tocomputeperrow) * numtts
    indices = np.array(range(numttpts))[tocomputeperrow]
    print("Calculating remaining travel times points:")
    print("- the algorithm re-computes the travel time for points that are nan")
    print("%d traveltimes arrays found" % numtts)
    print("%d points to compute for each array" % len(indices))
    print("%d total points to compute" % totalpts2compute)

    # dummy progressabr if isterminal is False:
    class Dummypbar(object):

        def update(self, *a, **kw):
            pass  # ignore the data

        def __enter__(self, *a, **kw):
            return self

        def __exit__(self, *a, **kw):
            pass

    pool = Pool()
    with Dummypbar() if not isterminal else progressbar(length=totalpts2compute) as bar:
        for sdrd, tts in izip(sdrd_array, tts_matrix):
            sd, rd = sdrd
            for i in indices:
                pool.apply_async(min_traveltime, (model, sd, rd, _DISTANCES[i]),
                                 callback=mp_callback(i, tts, bar if isterminal else None))

        pool.close()
        pool.join()


def computeall(modelname, fileout, maxerr=MAX_TIME_ERR_TOL, isterminal=False):
    if isterminal:
        print("Computing and saving travel times for model '%s'" % modelname)
    model = taumodel(modelname)
    sdrd_array, tt_matrix = get_sdrd_steps(model, isterminal=True)  # 'iasp91' 'ak135'
    if isterminal:
        print("")
    kwargs = dict(file=fileout, src_depth_bounds=[0, SD_MAX], rc_depth_bounds=[0, RD_MAX],
                  distances_bounds=[0, DIST_MAX], distances_step=[DIST_STEP], modelname=modelname,
                  tt_errtol=maxerr, sdrd_array=sdrd_array, tt_matrix=tt_matrix)
    # save now so in case we can interrupt somehow the computation
    np.savez_compressed(**kwargs)
    # FIXME: UNCOMMENT NEXT TWO LINES WHEN DONE!
    # computetts(model, sdrd_array, tt_matrix, isterminal=True)
    # np.savez_compressed(**kwargs)
    if isterminal:
        print("")
        print("Done")
        print("Computed %d travel times arrays associated to "
              "%d (source_depth, receiver_depth) pairs" % (tt_matrix.shape[0], tt_matrix.shape[0]))
        print("Each travel times array has %d points" % tt_matrix.shape[1])
        print("All travel times arrays compose a %dx%d matrix where:" %
              (tt_matrix.shape[0], tt_matrix.shape[1]))
        print("  a row represents a given (source_depth, receiver_depth) pair")
        print("  a column represents a given distance (in degree)")
        herr = np.abs(np.diff(tt_matrix, axis=1))
        verr = np.abs(np.diff(tt_matrix, axis=0))
        print("Time errors:")
        print("           %8s %8s %8s %8s" % ("min", "median", "mean", "max"))
        print("----------------------------------------------")
        print("horizontal %8.3f %8.3f %8.3f %8.3f (all in sec)" %
              (np.min(herr), np.median(herr), np.mean(herr), np.max(herr)))
        print("vertical   %8.3f %8.3f %8.3f %8.3f (all in sec)" %
              (np.min(verr), np.median(verr), np.mean(verr), np.max(verr)))
        print("(horizontal: time difference between two adjacent points on same row)")
        print("(vertical: time difference between two adjacent points on same column)")
        print("Bounds:")
        print("Source depths [min, max]: [%.3f, %.3f] km" % (0, SD_MAX))
        print("Receiver depths [min, max]: [%.3f, %.3f] km" % (0, RD_MAX))
        print("Distances [min : step: max]: [%.3f: %.3f: %.3f] deg" % (0, DIST_STEP, DIST_MAX))


if __name__ == '__main__':
    plottimes('iasp91', 0, 0.01, np.arange(0, 180, 10), ("P", "pP"))
#     import os
#     import sys
#     argv = sys.argv
#     if len(sys.argv) != 2:
#         _ = os.path.basename(sys.argv[0])
#         print("ERROR: run this script with a given model name, e.g:\n%s ak135\n%s iasp91\n..."
#               % (_, _))
#         sys.exit(1)
#     modelname = sys.argv[1]
#     fileout = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "resources",
#                                            "traveltimestables", "%s.npz" % modelname))
#     if not os.path.isdir(os.path.dirname(fileout)):
#         print("ERROR: not a directory: %s" % os.path.dirname(fileout))
#     maxerr = MAX_TIME_ERR_TOL
#     computeall(modelname, fileout, maxerr, isterminal=True)
#     sys.exit(0)
