'''
Created on Aug 23, 2017

@author: riccardo
'''
import os
from multiprocessing import Pool
import time
from datetime import timedelta
from itertools import izip, count, product

import numpy as np
from click.termui import progressbar
from obspy.taup.helper_classes import SlownessModelError
from obspy.taup.tau_model import TauModel
import click
from click.exceptions import BadParameter
from click import IntRange

from stream2segment.download.utils import get_min_travel_time
import sys

# global vars
DEFAULT_SD_MAX = 700.0  # in km
DEFAULT_RD_MAX = 0.0  # in km
DEFAULT_DIST_MAX = 180.0  # in degrees
DEFAULT_PWAVEVELOCITY = 5.0  # in km/sec  PLEASE SPECIFY A FLOAT!!
DEFAULT_DEG2KM = 111


def timemaxdecimaldigits(time_err_tolerance):
    numdigits = 0
    _ = time_err_tolerance
    while int(_) != _:
        _ *= 10
        numdigits += 1
        if numdigits > 3:
            raise ValueError("MAX_TIME_ERR_TOL cannot be lower than 0.001 (one millisecond)")
    return numdigits


def min_traveltimes(modelname, source_depth_km, receiver_depth_km, distances_in_deg,
                    traveltime_phases=("ttp+",)):
    model = taumodel(modelname)
    tts = []
    for d in distances_in_deg:
        tts.append(min_traveltime(model, source_depth_km, receiver_depth_km, d,
                                  traveltime_phases))
    return tts


def printtimes(modelname, source_depth_km, receiver_depth_km, distances_start=0,
               distances_end=180, distances_step=1):
    distances = np.linspace(distances_start, distances_step,
                            1+int(distances_end-distances_start), endpoint=True)
    tts = min_traveltimes(modelname, source_depth_km, receiver_depth_km, distances)
    print("distances: %s" % str(np.around(distances, decimals=3).tolist()))
    print("   ttimes: %s" % str(np.around(tts, decimals=3).tolist()))


def plottimes(modelname, source_depth_km, receiver_depth_km, distances_in_deg, phases):
    import matplotlib
    bckend = matplotlib.get_backend()
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    tts = min_traveltimes(modelname, source_depth_km, receiver_depth_km, distances_in_deg, phases)
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    ax.plot(distances_in_deg, tts)
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


def _cmp_indices(distances):  # returns the indices used for comparison from the given array arg.
    return np.unique([0, 1, 5, 10, int(len(distances)/4.0), int(len(distances)/2.0),
                      int(3*len(distances)/4.0), len(distances)-1])


def linspace(endvalue, tt_errtol, pwavevelocity=DEFAULT_PWAVEVELOCITY, deg2km=DEFAULT_DEG2KM,
             unit='deg'):
    '''
    :return: a numpy array of linearly spaced values from 0 to `endvalue` (bounds included),
            calculating the step according to tt_errtol.
    '''
    # Calculate the distance step, in degrees, rounded to the third decimal place (min=0.001 deg)
    # this gives the maximum granularity of travel times every 110 meters (not below). Increase to 4
    # if you want to go down to 11 meters, and so on
    # space = (DIST_STEP * DEG2KM) / time = MAX_TIME_ERR_TOL = velocity = PWAVEVELOCITY
    dist_step = np.around(float(pwavevelocity * tt_errtol), 3)
    if unit == 'deg':  # convert to degree (approx)
        dist_step = np.around(np.true_divide(dist_step, deg2km), 3)
    # calculate the number of points:
    numpts = int(np.ceil(float(endvalue)/dist_step))
    return np.linspace(0, endvalue, numpts, endpoint=True)


def itercreator(model, tt_errtol, phases, distances, maxsourcedepth=DEFAULT_SD_MAX,
                maxreceiverdepth=DEFAULT_RD_MAX,
                pwavevelocity=DEFAULT_PWAVEVELOCITY, deg2km=DEFAULT_DEG2KM):

    sourcedepths = linspace(maxsourcedepth, tt_errtol, pwavevelocity, deg2km, unit='km')
    receiverdepths = np.array([0]) if maxreceiverdepth == 0 else \
        linspace(maxreceiverdepth, tt_errtol, pwavevelocity, deg2km, unit='km')
    # The arrays above might need a finer step, because we iterate over it to check
    # when to store arrays. Set receiver depth at least 500mt and source depth
    # 1 km (assuming maxreceiverdepth and maxsourcedepths are ints, otherwise cast to int)
    # Note the +1 because we want to use numpy linspace with endpoint=True
    rd_numpts = max(len(receiverdepths), 1 + int(maxreceiverdepth*2))
    sd_numpts = max(len(sourcedepths), 1 + int(maxsourcedepth))
    if sd_numpts > len(sourcedepths):
        sourcedepths = np.linspace(0, maxsourcedepth, sd_numpts, endpoint=True)
    if rd_numpts > len(receiverdepths):
        receiverdepths = np.linspace(0, maxreceiverdepth, rd_numpts, endpoint=True)

    # the indices used for comparison (calculate only a portion of distances for speed reasons)
    # Be more granular for small distances (FIXME: why?)
    cmp_indices = _cmp_indices(distances)
    # and relative distances to be used for calculation:
    cmp_distances_in_degree = distances[cmp_indices]

    # as numpy allclose returns true if the error is <= max_abs_err_tol_in_sec, we want a strict
    # inequality, thus:
    max_err = abs(tt_errtol)
    max_err -= max_err * 0.00001

    model = taumodel(model)
    leng = len(sourcedepths) * len(receiverdepths)
    last_saved_traveltimes = None
    len_rds = len(receiverdepths)

    count = 0
    for sd in sourcedepths:  # [:int(len(source_depths_in_km)/10)]:
        args = [(sd, rd) for rd in receiverdepths]

        # ttimes is a matrix of [receiver_depts X distances]
        ttimes = np.full((len(args), len(cmp_distances_in_degree)), np.nan)

        # calculate first and last: if both are the same as the last computed, go on
        # don't do it for first and last iteration, as we need to save them anyway
        indices2precalculate = []
        if sd != sourcedepths[0] and sd != sourcedepths[-1]:
            indices2precalculate = [0] if len(args) == 1 else [0, len(args)-1]
            pool = Pool()
            for idx in indices2precalculate:
                sd, rd = args[idx]
                tmp_ttimes = ttimes[idx]
                for i, d in enumerate(cmp_distances_in_degree):
                    pool.apply_async(min_traveltime, (model, sd, rd, d),
                                     callback=mp_callback(i, tmp_ttimes))
            pool.close()
            pool.join()

            if ttequal(last_saved_traveltimes[0], ttimes[0], max_err) and \
                ttequal(last_saved_traveltimes[-1], ttimes[-1], max_err) and \
                    ttequal(ttimes[0], ttimes[-1], max_err):
                count += len_rds
                continue

        # need to calculate all receiver depths:
        range2calculate = sorted(list(set(xrange(len(args))) - set(indices2precalculate)))
        if range2calculate:
            pool = Pool()
            for idx in range2calculate:  # xrange(1, len(args)-1):
                sd, rd = args[idx]
                tmp_ttimes = ttimes[idx]
                for i, d in enumerate(cmp_distances_in_degree):
                    pool.apply_async(min_traveltime, (model, sd, rd, d),
                                     callback=mp_callback(i, tmp_ttimes))
            pool.close()
            pool.join()

        # Note that if we are here we have some (sd, rd) point that exceeds the err tolerance
        # Since we use scipy.griddata, we need to store the start and end point in order
        # to make a "cubic" grid, in order to avoid nan's for in bounds points
        # So we always store the first and last point of this loop, i.e. when
        # rd in (receiver_depths_km[0], receiver_depths_km[-1])
        traveltimes_mark = None if last_saved_traveltimes is None else last_saved_traveltimes[0]
        for (sd, rd), current_traveltimes in izip(args, ttimes):
            count += 1
            if rd in (receiverdepths[0], receiverdepths[-1]) \
                    or not ttequal(traveltimes_mark, current_traveltimes, max_err):
                real_maxerr = 0 if traveltimes_mark is None \
                    else np.nanmax(np.abs(traveltimes_mark - current_traveltimes))
                yield sd, rd, current_traveltimes, real_maxerr, count, leng
                traveltimes_mark = current_traveltimes

        last_saved_traveltimes = [ttimes[0], ttimes[-1]]


def get_sdrd_steps(model, tt_errtol, phases, maxsourcedepth=DEFAULT_SD_MAX,
                   maxreceiverdepth=DEFAULT_RD_MAX, maxdistance=DEFAULT_DIST_MAX,
                   pwavevelocity=DEFAULT_PWAVEVELOCITY, deg2km=DEFAULT_DEG2KM, isterminal=False):
    start = time.time()
    data = []
    sds, rds = [], []
    # calculate distances array:
    distances = linspace(maxdistance, tt_errtol, pwavevelocity, deg2km, unit='deg')
    # get the indices used for comparison in our algorithm
    cmp_indices = _cmp_indices(distances)
    if isterminal:
        print("Calculating source and receiver depth steps")
        print("- The algorithm basically iterates over each source depth and receiver depths pair")
        print("  and saves the associated travel times array (t)")
        print("  when it exceeds %f seconds with the previous computed t" % tt_errtol)
        print("- tc below refers to a chunk of each saved t showing the values at distances (deg):")
        print("  %s" % (np.around(distances[cmp_indices], decimals=3).tolist()))

        # typical header should be (roughly):
        # src_depth  rec_depth    travel times (max err with previous)   done              eta
        #        0.0        0.0  [0.0, 21.088, 36.402, 50.154] (0.000)  0.02%  19:15:33.394541
        # ...

        frmt = "%4s %9s %9s %60s %5s %16s"
        header = frmt % ('#', 'src_depth', 'rec_depth', 'tc (max err with previous tc)', '%done',
                         'eta')
        print("-"*len(header))
        print(header)
        print("-"*len(header))

    # pool = Pool()
    # already_calculated_tt_indices = set(_CMP_DIST_INDICES)
    idx = 1
    for sd, rd, tts, maxerr, count, total in itercreator(model, tt_errtol, phases, distances,
                                                         maxsourcedepth, maxreceiverdepth,
                                                         pwavevelocity, deg2km):
        if isterminal:
            percentdone = int(0.5 + (100.0 * count) / total)
            eta = int(0.5 + (total-count) * (float(time.time() - start) / count))
            # round to 0.1 sec:
            tt_list = np.around(tts, decimals=timemaxdecimaldigits(tt_errtol)+1).tolist()
            print(frmt % (idx, np.around(sd, 3),
                          np.around(rd, 3), '%s (%8.3f)' % (tt_list, maxerr), percentdone,
                          str(timedelta(seconds=eta))))
            idx += 1

        sds.append(sd)
        rds.append(rd)

        # NOW compute all remaining travel times. Use apply async so that we can go on inspecting
        # new source, receiver pairs
        complete_tts = newarray(distances)
        data.append(complete_tts)
        # add already calculated:
        complete_tts[cmp_indices] = tts

    return np.array(sds), np.array(rds), distances, \
        np.reshape(data, newshape=(len(data), len(data[0])))


def computetts(model, sourcedepths, receiverdepths, distances, tts_matrix, isterminal=False):
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
        for sd, rd, tts in izip(sourcedepths, receiverdepths, tts_matrix):
            for i in indices:
                pool.apply_async(min_traveltime, (model, sd, rd, distances[i]),
                                 callback=mp_callback(i, tts, bar if isterminal else None))
        pool.close()
        pool.join()


def computeall(fileout, model, tt_errtol, phases, maxsourcedepth=DEFAULT_SD_MAX,
               maxreceiverdepth=DEFAULT_RD_MAX, maxdistance=DEFAULT_DIST_MAX,
               pwavevelocity=DEFAULT_PWAVEVELOCITY, deg2km=DEFAULT_DEG2KM, isterminal=True):
    if not os.path.isdir(os.path.dirname(fileout)):
        raise OSError("File directory does not exist: '%s'" % str(os.path.dirname(fileout)))

    modelname = model
    if isterminal:
        print("Computing and saving travel times for model '%s'" % modelname)
    model = taumodel(modelname)
    sd, rd, d, tt_matrix = get_sdrd_steps(model, tt_errtol, phases, maxsourcedepth,
                                          maxreceiverdepth, maxdistance,
                                          pwavevelocity, deg2km, isterminal)  # 'iasp91' 'ak135'
    tt_matrix = tt_matrix.astype(np.float32)
    if isterminal:
        print("")
    kwargs = dict(file=fileout, modelname=modelname, sourcedepth_bounds_km=[0, maxsourcedepth],
                  receiverdepth_bounds_km=[0, maxreceiverdepth],
                  distances_bounds_deg=[d[0], d[-1]],
                  distances_step_deg=d[1]-d[0],
                  tt_errtol=tt_errtol, distances=d,
                  pwave_velocity=pwavevelocity, deg2km=deg2km,
                  sourcedepths=sd, receiverdepths=rd, traveltimes=tt_matrix,
                  phases=phases)
    # save now so in case we can interrupt somehow the computation
    np.savez_compressed(**kwargs)
    # FIXME: UNCOMMENT NEXT TWO LINES WHEN DONE!
    computetts(model, sd, rd, d, tt_matrix, isterminal=True)
    np.savez_compressed(**kwargs)
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
        print("Source depths [min, max]: [%.3f, %.3f] km" % (0, maxsourcedepth))
        print("Receiver depths [min, max]: [%.3f, %.3f] km" % (0, maxreceiverdepth))
        print("Distances [min : step: max]: [%.3f: %.3f: %.3f] deg" % (0, d[1]-d[0], d[-1]))
        print("")
        print("Travel times table written to '%s'" % fileout)


def _filepath(fileout, model, phases):
    if os.path.isdir(fileout):
        fileout = os.path.join(fileout, model + "_" + "_".join(phases))
    return fileout


@click.command(short_help='Creates via obspy routines travel time table, i.e. a grid of points '
               'in a 3-D space, where each point is '
               'associated to pre-computed minima travel times arrays. Stores the '
               'resulting file as .npz compressed numpy format. The resulting file, opened with '
               'the dedicated program class, allows to compute approximate travel times in a '
               '*much* faster way than using obspy routines')
@click.option('-o', '--output', is_eager=True, required=True,
              help=('The output file. If directory, the file name will be automatically '
                    'created inside the directory. Otherwise must denote a valid writable '
                    'file name. The extension .npz will be added automatically'))
@click.option("-m", "--model",
              help="the model name, e.g. iasp91, ak135, ..")
@click.option('-p', '--phases', multiple=True,  required=True,
              help=("The phases used, e.g. ttp+, tts+. Can be typed multiple times, e.g."
                    "-m P -m p"))
@click.option('-t', '--tt_errtol', type=float, required=True,
              help=('The error tolerance (in seconds). The algorithm will try to store grid points '
                    'whose distance is close to this value. Decrease this value to increase '
                    'precision, increase this value to increase the execution speed of this '
                    'command'))
@click.option('-s', '--maxsourcedepth', type=float, default=DEFAULT_SD_MAX,
              help=('The maximum source depth (in km) used for the grid generation. '
                    'Optional: defaults to 700 when missing. '
                    'When loaded, the relative model can calculate travel times for source depths '
                    'lower or equal to this value'))
@click.option('-r', '--maxreceiverdepth', type=float, default=DEFAULT_RD_MAX,
              help=('The maximum source depth (in km) used for the grid generation. '
                    'Optional: defaults to 0 when missing (assume all receiver depths as zero). '
                    'When loaded, the relative model can calculate travel times for receiver '
                    'depths lower or equal to this value'))
@click.option('-d', '--maxdistance', type=float, default=DEFAULT_DIST_MAX,
              help=('The maximum distance (in degrees) used for the grid generation. '
                    'Optional: defaults to 180 when missing. '
                    'When loaded, the relative model can calculate travel times for receiver '
                    'depths lower or equal to this value'))
@click.option('-P', '--pwavevelocity', type=float, default=DEFAULT_PWAVEVELOCITY,
              help=('The P-wave velocity (in km/sec). Used for the grid generation '
                    'to assess the step of the distances arrays.'
                    'Optional: defaults to 5 when missing. '
                    'As P-wave velocity varies from [6-13] km according to the region of the '
                    'Earth\'s interior, 5 is the value that assures '
                    'grid precision for small source depths and avoids having too many '
                    'redundant data for higher source depths'))
@click.option('-D', '--deg2km', type=float, default=DEFAULT_DEG2KM,
              help=('The (approximate) length (in km) of a degree. Used for the grid generation '
                    'to assess the step of the distances arrays.'
                    'Optional: defaults to 111 when missing.'))
def run(output, model, phases, tt_errtol, maxsourcedepth, maxreceiverdepth, maxdistance,
        pwavevelocity, deg2km):
    try:
        output = _filepath(output, model, phases)
        computeall(output, model, tt_errtol, phases, maxsourcedepth, maxreceiverdepth, maxdistance,
                   pwavevelocity, deg2km, isterminal=True)
        sys.exit(0)
    except Exception as exc:
        print("ERROR: %s" % str(exc))
        sys.exit(1)

if __name__ == '__main__':
    run()  # pylint: disable=E1120

