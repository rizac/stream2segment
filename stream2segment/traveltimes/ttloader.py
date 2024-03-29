"""
Module holding the TTTable class, a class ehich loads compressed numpy data previously
created via the `:ref:ttcreator` module and that allows the calculation of the
minimum theoretical travel times by means of a pre-computed grid of points
using linear, cubic or nearest sample approximation

:date: Sep 1, 2017
.. moduleauthor:: <rizac@gfz-potsdam.de>
"""
import numpy as np
try:
    from scipy.interpolate import griddata
except ImportError:
    from scipy.interpolate.ndgriddata import griddata


class TTTable:
    """Class handling the computation of the travel times from pre-computed travel times
    stored in .npz numpy format
    """

    def __init__(self, filepath):
        """Initializes a new TTTable from a pre-computed .npz numpy compressed file"""
        self.filepath = filepath
        _data = np.load(filepath)
        self._attrs = _data.files
        # set manually the attributes, a little verbose but pylint is happier:
        self._deg2km = _data['deg2km']
        self._swave_velocity = _data['swave_velocity']
        self._phases = _data['phases']
        self._modelname = _data['modelname']
        self._sourcedepth_bounds_km = _data['sourcedepth_bounds_km']
        self._sourcedepths = _data['sourcedepths']
        self._distances = _data['distances']
        self._tt_errtol = _data['tt_errtol']
        self._receiverdepth_bounds_km = _data['receiverdepth_bounds_km']
        self._receiverdepths = _data['receiverdepths']
        self._distances_bounds_deg = _data['distances_bounds_deg']
        self._distances_step_deg = _data['distances_step_deg']
        self._traveltimes = _data['traveltimes']
        self._pwave_velocity = _data['pwave_velocity']

        # create grid points (not most efficient way, but this way is more readable
        # and perfs should be irrelevant)
        # Note that if receiver depths are not set (i.e. all zero) we need to suppress
        # that dimension otherwise we have an internal error in self.min
        # on the algorithm using scipy.griddata
        self._unique_receiver_depth = True if \
            len(np.unique(self._receiverdepths)) == 1 else False
        gridpts = []
        for src, rec in zip(self._km2deg(self._sourcedepths),
                            self._km2deg(self._receiverdepths)):
            for _ in self._distances:
                gridpts.append([src, _] if self._unique_receiver_depth else [src, rec, _])
        # store in class attributes:
        self._gridpts = np.array(gridpts)
        self._gridvals = self._traveltimes.reshape(1, -1).flatten()

    def _km2deg(self, array):
        return np.true_divide(array, self._deg2km)  # normalize to degrees

    def __call__(self, source_depths, receiver_depths, distances, method='linear'):
        return self.min(source_depths, receiver_depths, distances, method)

    def min(self, source_depths, receiver_depths, distances, method='linear'):
        """Return the minimum (minima) travel time(s) for the point(s) identified by
        (source_depths, receiver_depths, distances) by building a grid and
        interpolating on the points identified by each
        ```
            P[i] = (source_depths[i], receiver_depths[i], distances[i])
        ```
        if the source file has been built with receiver depths == [0]. It uses a 2d
        linear interpolation on a grid. It uses scipy `griddata`

        :param source_depths: numeric or numpy array of length n: the source depth(s),
            in km
        :param receiver_depths: numeric or numpy array of length n: the receiver
            depth(s), in km. For most applications, this value can be set to zero
        :param distances: numeric or numpy array of length n: the distance(s), in degrees
        :param method: forwarded to `scipy.griddata` function
        :return: a numpy array of length n denoting the minimum travel time(s) of this
            model for each P
        """
        # Handle the case some arguments are scalars and some arrays:
        source_depths, receiver_depths, distances = \
            np.broadcast_arrays(source_depths, receiver_depths, distances)
        # copy arrays as numpy after 1.15 (I guess) issues warnings when
        # writing on a view (similar to pandas set with copy warning):
        source_depths = np.copy(source_depths)
        receiver_depths = np.copy(receiver_depths)
        distances = np.copy(distances)
        # handle the case all arguments scalars (https://stackoverflow.com/a/29319864):
        allscalars = all(_.ndim == 0 for _ in (source_depths, receiver_depths, distances))
        if source_depths.ndim == 0:
            source_depths = source_depths[None]  # Makes x 1D
        if receiver_depths.ndim == 0:
            receiver_depths = receiver_depths[None]  # Makes x 1D
        if distances.ndim == 0:
            distances = distances[None]  # Makes x 1D
        # correct source depths and receiver depths
        source_depths[source_depths < 0] = 0
        receiver_depths[receiver_depths < 0] = 0
        # correct distances to be compatible with obpsy traveltimes calculations:
        distances = distances % 360
        # set values symmetric to 180 degrees if greater than 180:
        _mask = distances > 180
        if _mask.any():  # does this speeds up (allocate mask array once)? FIXME: check
            distances[_mask] = 360 - distances[_mask]
        # set source depths to nan if out of bounds. This prevent method = 'nearest'
        # to return non nan values and be consistent with 'linear' and 'cubic'
        if self._unique_receiver_depth:
            # get values without receiver depth dimension:
            values = np.hstack((self._km2deg(source_depths).reshape(-1, 1),
                                distances.reshape(-1, 1)))
        else:
            values = np.hstack((self._km2deg(source_depths).reshape(-1, 1),
                                self._km2deg(receiver_depths).reshape(-1, 1),
                                distances.reshape(-1, 1)))

        ret = griddata(self._gridpts, self._gridvals, values,
                       method=method, rescale=False, fill_value=np.nan)

        # ret is almost likely a float, so we can set now NaNs for out of bound values
        # we cannot do it before on any input array because int arrays don't support NaN
        ret[(source_depths > self._sourcedepth_bounds_km[1]) |
            (receiver_depths > self._receiverdepth_bounds_km[1])] = np.nan
        # return scalar if inputs are scalar, array oitherwise
        return np.squeeze(ret) if allscalars else ret

    @property
    def model(self):
        """Return the model name whereby this object has been built

        :return: the model name (string)
        """
        return self._py23str(self._modelname.item())

    @property
    def phases(self):
        """Returns the travel times phases whereby this object has been built

        :return: a list of strings representing the travel time phases
        """
        return [self._py23str(p) for p in self._phases.tolist()]

    @staticmethod
    def _py23str(stringorbytes):
        # Because we might have created the underlying .npz file with python2, some
        # expected str values might be bytes (e.g. `self.phases` and `self.model`):
        if not isinstance(stringorbytes, str):
            stringorbytes = stringorbytes.decode('utf8')
        return stringorbytes

    def __str__(self):
        maxrows = 6

        def num2str(num):
            '''rounds num and returns its str representation with indentation'''
            return str(np.around(num, decimals=3)).rjust(8, ' ')

        def array2str(array):
            '''rounds each element of array and returns its str representation with indentation'''
            ret = []
            for i in range(int(maxrows / 2)):
                ret.append(num2str(array[i]))
            if len(array) > maxrows:
                ret.append("...")
            for i in range(int(-maxrows / 2), 0):
                ret.append(num2str(array[i]))
            return " ".join(ret)

        collen = [8, 8, len(array2str(self._traveltimes[-1]))]
        _frmt = "%{:d}s %{:d}s %{:d}s".format(*collen)
        hline = " ".join(c * "-" for c in collen)

        ret = ["Model: '%s'" % (self.model), "Phases: %s" % (self.phases),
               "Input error tolerance: %f" % self._tt_errtol,
               "Data:", hline, _frmt % ("Source", "Receiver", ""),
               _frmt % ("depth", "depth", "Travel times"), hline]
        for i in range(int(maxrows / 2)):
            src, rec = num2str(self._sourcedepths[i]), num2str(self._receiverdepths[i])
            ret.append(_frmt % (src, rec, array2str(self._traveltimes[i])))

        if len(self._sourcedepths) > maxrows:
            ret.append(_frmt % ("...", "...", "..."))

        for i in range(int(-maxrows / 2), 0):
            src, rec = num2str(self._sourcedepths[i]), num2str(self._receiverdepths[i])
            ret.append(_frmt % (src, rec, array2str(self._traveltimes[i])))

        ret.append(hline)
        ret.append(("%{:d}s %s".format(collen[0]+collen[1]+1)) %
                   ("Distances->", array2str(self._distances)))
        return "\n".join(ret)
