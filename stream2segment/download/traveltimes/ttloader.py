'''
Created on Sep 1, 2017

@author: riccardo
'''
from itertools import izip

import numpy as np
from scipy.interpolate.ndgriddata import griddata


class TTTable(object):
    '''Class handling the computation of the travel times from pre-computed travel times stored
    in .npz numpy format'''

    def __init__(self, filepath):
        '''Initializes a new TTTable from a pre-computed .npz numpy compressed file'''
        self.filepath = filepath
        _data = np.load(filepath)
        self._attrs = _data.files
        for f in self._attrs:
            setattr(self, "_" + f, _data[f])
        # create grid points (not most efficient way, but this way is more readable
        # and perfs should be irrelevant)
        # Note that if receiver depths are not set (i.e. all zero) we need to suppress
        # that dimension otherwise we have an internal error in self.min
        # on the algorithm using scipy.griddata
        self._unique_receiver_depth = np.unique(self._receiverdepths)[0] if \
            len(np.unique(self._receiverdepths)) == 1 else None
        gridpts = []
        for s, r in izip(self._normalize(self._sourcedepths),
                         self._normalize(self._receiverdepths)):
            for _ in self._distances:
                gridpts.append([s, r, _] if self._unique_receiver_depth is None else [s, _])
        # store in class attributes:
        self._gridpts = np.array(gridpts)
        self._gridvals = self._traveltimes.reshape(1, -1).flatten()

    def _normalize(self, array):
        return array / 110.0  # normalize to degrees

    @staticmethod
    def _normalize_data(source_depths, receiver_depths, distances):
        # broadcast arrays:
        source_depths, receiver_depths, distances = \
            np.broadcast_arrays(source_depths, receiver_depths, distances)
        # correct source depths and receiver depths
        source_depths[source_depths < 0] = 0
        receiver_depths[receiver_depths < 0] = 0
        # correct distances to be compatible with obpsy traveltimes calculations:
        distances = distances % 360
        _mask = distances > 180
        if _mask.any():  # does this speeds up (allocate mask array once)? FIXME: check
            distances[_mask] = 360 - distances[_mask]
        return source_depths, receiver_depths, distances

    def min(self, source_depths, receiver_depths, distances, method='linear'):
        '''
        Returns the minimum (minima) travel time(s) for the point(s) identified by
        (source_depths, receiver_depths, distances) by building a grid and
        interpolating on the points identified by each
        ```
            P[i] = (source_depths[i], receiver_depths[i], distances[i])
        ```
        if the source file has been built with
        receiver depths == [0]. It uses a 2d linear interpolation on a grid. It uses
        scipy griddata
        :param source_depths: numeric or numpy array of length n: the source depth(s), in km
        :param receiver_depths: numeric or numpy array of length n: the receiver depth(s), in km.
        For most applications, this value can be set to zero
        :param distances: numeric or numpy array of length n: the distance(s), in degrees
        :param method: forwarded to `scipy.griddata` function
        :return: a numpy array of length n denoting the minimum travel time(s) of this model for
        each P
        '''
        # normalize data:
        source_depths, receiver_depths, distances = \
            self._normalize_data(source_depths, receiver_depths, distances)
        # create values to interpolate. Note that if receiver depths are mono-dimensional (i.e.
        # all zero) we create a 2dimensional grid instead of a 3 dimensional grid
        if self._unique_receiver_depth is None:
            values = np.hstack((self._normalize(source_depths).reshape(-1, 1),
                                self._normalize(receiver_depths).reshape(-1, 1),
                                distances.reshape(-1, 1)))
        else:
            # if no receiver depths are set (i.e., all zero), we do not have a way to
            # distinguish from scipy griddata if receiver depths are out of bounds. Do it now:
            receiver_depths[receiver_depths != self._unique_receiver_depth] = np.nan
            # get values without receiver depth dimension:
            values = np.hstack((self._normalize(source_depths).reshape(-1, 1),
                                distances.reshape(-1, 1)))

        return griddata(self._gridpts, self._gridvals, values,
                        method=method, rescale=False, fill_value=np.nan)

    def __str__(self, *args, **kwargs):
        maxrows = 6

        def r_(num):
            return str(np.around(num, decimals=3)).rjust(8, ' ')

        def echorow(array):
            ret = []
            for i in xrange(maxrows/2):
                ret.append(r_(array[i]))
            if len(array) > maxrows:
                ret.append("...")
            for i in xrange(-maxrows/2, 0):
                ret.append(r_(array[i]))
            return " ".join(ret)

        collen = [8, 8, len(echorow(self._traveltimes[-1]))]
        _frmt = "%{:d}s %{:d}s %{:d}s".format(*collen)
        hline = " ".join(c * "-" for c in collen)

        ret = ["Model: '%s'" % (self._modelname), "Phases: %s" % (self._phases),
               "Input error tolerance: %f" % self._tt_errtol,
               "Data:", hline, _frmt % ("Source", "Receiver", ""),
               _frmt % ("depth", "depth", "Travel times"), hline]
        for i in xrange(maxrows/2):
            s, r = r_(self._sourcedepths[i]), r_(self._receiverdepths[i])
            ret.append(_frmt % (s, r, echorow(self._traveltimes[i])))

        if len(self._sourcedepths) > maxrows:
            ret.append(_frmt % ("...", "...", "..."))

        for i in xrange(-maxrows/2, 0):
            s, r = r_(self._sourcedepths[i]), r_(self._receiverdepths[i])
            ret.append(_frmt % (s, r, echorow(self._traveltimes[i])))

        ret.append(hline)
        ret.append(("%{:d}s %s".format(collen[0]+collen[1]+1)) %
                   ("Distances->", echorow(self._distances)))
        return "\n".join(ret)
