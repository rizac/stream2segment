"""
Event-based station search functions

:date: Dec 3, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
# make the following(s) behave like python3 counterparts if running from py2.7
# (http://python-future.org/imports.html#explicit-imports):
from builtins import zip

from itertools import cycle
from datetime import timedelta

import numpy as np
import pandas as pd

from stream2segment.download.db import Station, Channel, Event, Segment
from stream2segment.download.modules.utils import formatmsg
from stream2segment.download.exc import FailedDownload
from stream2segment.io.cli import get_progressbar
from stream2segment.io.db.pdsql import mergeupdate, dfrowiter


# # logger: do not use logging.getLogger(__name__) but point to
# # stream2segment.download.logger: this way we preserve the logging namespace
# # hierarchy
# # (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial) when
# # calling logging functions of stream2segment.download.utils:
# from stream2segment.download import logger  # @IgnorePep8

import logging
logger = logging.getLogger(__name__)


def merge_events_stations(events_df, channels_df, search_radius,
                          tttable, show_progress=False):
    """Merge `events_df` and `channels_df` by returning a new dataframe
    representing all channels within a specific search radius. *Each row of the
    returned data frame is basically a segment to be potentially donwloaded*.
    The returned dataframe will be the same as `channels_df` with one or more
    rows repeated (some channels might be in the search radius of several
    events), plus a column "event_id" (`Segment.event_id`) representing the
    event associated to that channel and two columns 'event_distance_deg',
    'time' (representing the *event* time) and 'depth_km' (representing the
    event depth in km)

    :param channels_df: pandas DataFrame resulting from `get_channels_df`
    :param events_df: pandas DataFrame resulting from `get_events_df`
    """
    # For convenience and readability, define once the mapped column names
    # representing the dataframe columns that we need:
    EVT_ID = Event.id.key  # noqa
    EVT_MAG = Event.magnitude.key  # noqa
    EVT_LAT = Event.latitude.key  # noqa
    EVT_LON = Event.longitude.key  # noqa
    EVT_TIME = Event.time.key  # noqa
    EVT_DEPTH = Event.depth_km.key  # noqa
    STA_LAT = Station.latitude.key  # noqa
    STA_LON = Station.longitude.key  # noqa
    STA_STIME = Station.start_time.key  # noqa
    STA_ETIME = Station.end_time.key  # noqa
    CHA_ID = Channel.id.key  # noqa
    CHA_STAID = Channel.station_id.key  # noqa
    SEG_EVID = Segment.event_id.key  # noqa
    SEG_EVDIST = Segment.event_distance_deg.key  # noqa
    SEG_ATIME = Segment.arrival_time.key  # noqa
    SEG_DCID = Segment.datacenter_id.key  # noqa
    SEG_CHAID = Segment.channel_id.key  # noqa

    channels_df = channels_df.rename(columns={CHA_ID: SEG_CHAID})
    # get unique stations, rename Channel.id into Segment.channel_id now so we
    # do not bother later
    stations_df = channels_df.drop_duplicates(subset=[CHA_STAID]).copy()

    ret = []

    sourcedepths, eventtimes = [], []

    with get_progressbar(show_progress, length=len(events_df)) as pbar:

        min_radia, max_radia = get_serarch_radia(search_radius,
                                                 events_df[EVT_MAG].values)

        radia_event_iter = zip(min_radia, max_radia,
                               dfrowiter(events_df, [EVT_ID, EVT_LAT, EVT_LON,
                                                     EVT_TIME, EVT_DEPTH]))

        oneday = timedelta(days=1)
        for min_radius, max_radius, evt_dic in radia_event_iter:
            l2d = locations2degrees(stations_df[STA_LAT], stations_df[STA_LON],
                                    evt_dic[EVT_LAT], evt_dic[EVT_LON])
            condition = (stations_df[STA_STIME] <= evt_dic[EVT_TIME]) & \
                        (pd.isnull(stations_df[STA_ETIME]) |
                         (stations_df[STA_ETIME] >= evt_dic[EVT_TIME] + oneday))
            # l2d is a distance, thus non negative. We can add the min radius
            # condition only if it is >=0. Evaluate to false in case min_radius
            # is None (legacy code):
            if min_radius:
                condition &= (l2d >= min_radius)
            # for max_radius, None means: skip
            if max_radius is not None:
                condition &= (l2d <= max_radius)

            pbar.update(1)
            if not np.any(condition):
                continue

            # Set (or re-set from second iteration on) as NaN SEG_EVDIST
            # columns. This is important cause from second loop on we might
            # have some elements not-NaN which should be NaN now
            channels_df[SEG_EVDIST] = np.nan
            # set locations2 degrees
            stations_df[SEG_EVDIST] = l2d
            # Copy distances calculated on stations to their channels
            # (match along column CHA_STAID shared between the reletive
            # dataframes). Set values only for channels whose stations are
            # within radius (stations_df[condition]):
            cha_df = mergeupdate(channels_df, stations_df[condition],
                                 [CHA_STAID], [SEG_EVDIST],
                                 drop_other_df_duplicates=False)
            # Note above: duplicates already dropped
            # Now drop channels which are not related to station within radius:
            cha_df = cha_df.dropna(subset=[SEG_EVDIST], inplace=False).copy()
            # ...and add "safely" SEG_EVID values:
            cha_df[SEG_EVID] = evt_dic[EVT_ID]
            # append to arrays (calculate arrival times in one shot a t the
            # end, it's faster):
            sourcedepths += [evt_dic[EVT_DEPTH]] * len(cha_df)
            eventtimes += [np.datetime64(evt_dic[EVT_TIME])] * len(cha_df)
            # Append only relevant columns:
            ret.append(cha_df[[SEG_CHAID, SEG_EVID, SEG_DCID, SEG_EVDIST]])

    # create total segments dataframe:
    # first check we have data:
    if not ret:
        raise FailedDownload(formatmsg("No segments to process",
                                       "No station within search radia"))
    # now concat:
    ret = pd.concat(ret, axis=0, ignore_index=True, copy=True)
    # compute travel times. Doing it on a single array is much faster
    sourcedepths = np.array(sourcedepths)
    distances = ret[SEG_EVDIST].values
    traveltimes = tttable(sourcedepths, 0, distances)
    # assign to column (should be of type  '<M8[us]' or any datetime dtype):
    eventtimes = np.array(eventtimes)
    # now to compute arrival times: eventtimes + traveltimes does not work
    # (we cannot sum np.datetime64 and np.float). Convert traveltimes to
    # np.timedelta: we first multiply by 1000000 to preserve the millisecond
    # resolution and then we write traveltimes.astype("m8[us]") which means:
    # 8bytes timedelta with microsecond resolution (10^-6). Side note: all
    # numpy timedelta constructors (as well as "astype") round to int argument,
    # at least in numpy13.
    ret[SEG_ATIME] = eventtimes + (traveltimes*1000000).astype("m8[us]")
    # drop nat values
    oldlen = len(ret)
    ret.dropna(subset=[SEG_ATIME], inplace=True)
    if oldlen > len(ret):
        logger.info(formatmsg("%d of %d segments discarded", "Travel times NaN"),
                    oldlen-len(ret), oldlen)
        if ret.empty:
            raise FailedDownload(formatmsg("No segments to process",
                                           "All travel times NaN"))
    return ret


def locations2degrees(lat1, lon1, lat2, lon2):
    """Same as ObsPy `locations2degree` but works with numpy arrays.

    From the doc:
    Convenience function to calculate the great circle distance between two
    points on a spherical Earth.

    This method uses the Vincenty formula in the special case of a spherical
    Earth. For more accurate values use the geodesic distance calculations of
    geopy (https://github.com/geopy/geopy).

    :param lat1: (numpy numeric array). Latitude(s) of point 1 in degrees
    :param lon1: (numpy numeric array). Longitude(s) of point 1 in degrees
    :param lat2: (numpy numeric array). Latitude(s) of point 2 in degrees
    :param lon2: (numpy numeric array). Longitude(s) of point 2 in degrees

    :return: Distance in degrees as a numpy numeric array.
    """
    # (Note: this function, exactly this one, is now in obspy, thanks to a PR
    # we issued long ago. We still have it here because prefer to decouple
    # ObsPy from the download package

    # Convert to radians.
    lat1 = np.radians(np.asarray(lat1))
    lat2 = np.radians(np.asarray(lat2))
    lon1 = np.radians(np.asarray(lon1))
    lon2 = np.radians(np.asarray(lon2))
    long_diff = lon2 - lon1
    deg, atan2, cos, sin, sqrt = np.degrees, np.arctan2, np.cos, np.sin, np.sqrt
    ret = deg(atan2(sqrt((cos(lat2) * sin(long_diff)) ** 2 +
                         (cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(long_diff)) ** 2),
                    sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(long_diff)))
    return ret


def get_serarch_radia(search_radius, magnitudes):
    """Return two iterables denoting the minima and maxima radia for
    stations search. Any element of the iterables might be None to indicate:
    no restriction for that element
    """
    if 'min' not in search_radius and 'max' not in search_radius:
        return cycle([None]), get_magdep_search_radius(magnitudes,
                                                       search_radius['minmag'],
                                                       search_radius['maxmag'],
                                                       search_radius['minmag_radius'],
                                                       search_radius['maxmag_radius'])
    return cycle([search_radius['min']]), cycle([search_radius['max']])


def get_magdep_search_radius(mag, minmag, maxmag, minmag_radius, maxmag_radius):
    """From a given magnitude, return the max radius/radia (in degrees).
    Given minmag_radius and maxmag_radius and minmag and maxmag, this
    function returns D from the f below:

                  |
    maxmag_radius +                oooooooooooo
                  |              o
                  |            o
                  |          o
    minmag_radius + oooooooo
                  |
                  ---------+-------+------------
                        minmag     maxmag


    :param mag: (numeric or list or numbers/numpy.array) the magnitude
    :param minmag: (int, float) the minimum magnitude
    :param maxmag: (int, float) the maximum magnitude
    :param minmag_radius: (int, float) the radius for `min_mag` (in degrees)
    :param maxmag_radius: (int, float) the radius for `max_mag` (in degrees)
    :return: the max radius/radia (in degrees)
    """
    mag = np.asarray(mag)  # do NOT copies data for existing arrays
    isscalar = not mag.shape
    if isscalar:
        mag = np.array(mag, ndmin=1)  # copies data, assures an array of dim=1

    if minmag == maxmag:
        dist = np.array(mag)
        dist[mag < minmag] = minmag_radius
        dist[mag >= minmag] = maxmag_radius
    else:
        dist = minmag_radius + \
            np.true_divide(maxmag_radius - minmag_radius, maxmag - minmag) * (mag - minmag)
        dist[dist < minmag_radius] = minmag_radius
        dist[dist > maxmag_radius] = maxmag_radius

    return dist[0] if isscalar else dist
