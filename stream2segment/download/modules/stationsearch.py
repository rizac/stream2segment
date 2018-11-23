'''
Download module for segments download

:date: Dec 3, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import map, next, zip, range, object

from datetime import timedelta

import numpy as np
import pandas as pd

from stream2segment.io.db.models import Station, Channel, Event, Segment
from stream2segment.download.utils import FailedDownload, formatmsg
from stream2segment.utils import get_progressbar
from stream2segment.io.db.pdsql import mergeupdate, dfrowiter


# logger: do not use logging.getLogger(__name__) but point to stream2segment.download.logger:
# this way we preserve the logging namespace hierarchy
# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial) when calling logging
# functions of stream2segment.download.utils:
from stream2segment.download import logger  # @IgnorePep8


def merge_events_stations(events_df, channels_df, minmag, maxmag, minmag_radius, maxmag_radius,
                          tttable, show_progress=False):
    """
        Merges `events_df` and `channels_df` by returning a new dataframe representing all
        channels within a specific search radius. *Each row of the returned data frame is
        basically a segment to be potentially donwloaded*.
        The returned dataframe will be the same as `channels_df` with one or more rows repeated
        (some channels might be in the search radius of several events), plus a column
        "event_id" (`Segment.event_id`) representing the event associated to that channel
        and two columns 'event_distance_deg', 'time' (representing the *event* time) and
        'depth_km' (representing the event depth in km)
        :param channels_df: pandas DataFrame resulting from `get_channels_df`
        :param events_df: pandas DataFrame resulting from `get_events_df`
    """
    # For convenience and readability, define once the mapped column names representing the
    # dataframe columns that we need:
    EVT_ID = Event.id.key  # pylint: disable=invalid-name
    EVT_MAG = Event.magnitude.key  # pylint: disable=invalid-name
    EVT_LAT = Event.latitude.key  # pylint: disable=invalid-name
    EVT_LON = Event.longitude.key  # pylint: disable=invalid-name
    EVT_TIME = Event.time.key  # pylint: disable=invalid-name
    EVT_DEPTH = Event.depth_km.key  # pylint: disable=invalid-name
    STA_LAT = Station.latitude.key  # pylint: disable=invalid-name
    STA_LON = Station.longitude.key  # pylint: disable=invalid-name
    STA_STIME = Station.start_time.key  # pylint: disable=invalid-name
    STA_ETIME = Station.end_time.key  # pylint: disable=invalid-name
    CHA_ID = Channel.id.key  # pylint: disable=invalid-name
    CHA_STAID = Channel.station_id.key  # pylint: disable=invalid-name
    SEG_EVID = Segment.event_id.key  # pylint: disable=invalid-name
    SEG_EVDIST = Segment.event_distance_deg.key  # pylint: disable=invalid-name
    SEG_ATIME = Segment.arrival_time.key  # pylint: disable=invalid-name
    SEG_DCID = Segment.datacenter_id.key  # pylint: disable=invalid-name
    SEG_CHAID = Segment.channel_id.key  # pylint: disable=invalid-name

    channels_df = channels_df.rename(columns={CHA_ID: SEG_CHAID})
    # get unique stations, rename Channel.id into Segment.channel_id now so we do not bother later
    stations_df = channels_df.drop_duplicates(subset=[CHA_STAID]).copy()

    ret = []
    max_radia = get_search_radius(events_df[EVT_MAG].values, minmag, maxmag,
                                  minmag_radius, maxmag_radius)

    sourcedepths, eventtimes = [], []

    with get_progressbar(show_progress, length=len(max_radia)) as bar:
        for max_radius, evt_dic in zip(max_radia, dfrowiter(events_df, [EVT_ID, EVT_LAT, EVT_LON,
                                                                        EVT_TIME, EVT_DEPTH])):
            l2d = locations2degrees(stations_df[STA_LAT], stations_df[STA_LON],
                                    evt_dic[EVT_LAT], evt_dic[EVT_LON])
            condition = (l2d <= max_radius) & (stations_df[STA_STIME] <= evt_dic[EVT_TIME]) & \
                        (pd.isnull(stations_df[STA_ETIME]) |
                         (stations_df[STA_ETIME] >= evt_dic[EVT_TIME] + timedelta(days=1)))

            bar.update(1)
            if not np.any(condition):
                continue

            # Set (or re-set from second iteration on) as NaN SEG_EVDIST columns. This is important
            # cause from second loop on we might have some elements not-NaN which should be NaN now
            channels_df[SEG_EVDIST] = np.nan
            # set locations2 degrees
            stations_df[SEG_EVDIST] = l2d
            # Copy distances calculated on stations to their channels
            # (match along column CHA_STAID shared between the reletive dataframes). Set values
            # only for channels whose stations are within radius (stations_df[condition]):
            cha_df = mergeupdate(channels_df, stations_df[condition], [CHA_STAID], [SEG_EVDIST],
                                 drop_other_df_duplicates=False)  # dupes already dropped
            # drop channels which are not related to station within radius:
            cha_df = cha_df.dropna(subset=[SEG_EVDIST], inplace=False).copy()
            cha_df[SEG_EVID] = evt_dic[EVT_ID]  # ...and add "safely" SEG_EVID values
            # append to arrays (calculate arrival times in one shot a t the end, it's faster):
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
    # assign to column:
    eventtimes = np.array(eventtimes)  # should be of type  '<M8[us]' or whatever datetime dtype
    # now to compute arrival times: eventtimes + traveltimes does not work (we cannot
    # sum np.datetime64 and np.float). Convert traveltimes to np.timedelta: we first multiply by
    # 1000000 to preserve the millisecond resolution and then we write traveltimes.astype("m8[us]")
    # which means: 8bytes timedelta with microsecond resolution (10^-6)
    # Side note: that all numpy timedelta constructors (as well as "astype") round to int
    # argument, at least in numpy13.
    ret[SEG_ATIME] = eventtimes + (traveltimes*1000000).astype("m8[us]")
    # drop nat values
    oldlen = len(ret)
    ret.dropna(subset=[SEG_ATIME], inplace=True)
    if oldlen > len(ret):
        logger.info(formatmsg("%d of %d segments discarded", "Travel times NaN"),
                    oldlen-len(ret), oldlen)
        if ret.empty:
            raise FailedDownload(formatmsg("No segments to process", "All travel times NaN"))
    return ret


def locations2degrees(lat1, lon1, lat2, lon2):
    """
    Same as obspy `locations2degree` but works with numpy arrays. NOTE: thanks to our PR ;)
    the current obspy version supports this function, but we prefer to decouple obspy from
    the download package

    From the doc:
    Convenience function to calculate the great circle distance between two
    points on a spherical Earth.

    This method uses the Vincenty formula in the special case of a spherical
    Earth. For more accurate values use the geodesic distance calculations of
    geopy (https://github.com/geopy/geopy).

    :type lat1: numpy numeric array
    :param lat1: Latitude(s) of point 1 in degrees
    :type lon1: numpy numeric array
    :param lon1: Longitude(s) of point 1 in degrees
    :type lat2: numpy numeric array
    :param lat2: Latitude(s) of point 2 in degrees
    :type lon2: numpy numeric array
    :param lon2: Longitude(s) of point 2 in degrees
    :rtype: numpy numeric array
    :return: Distance in degrees as a floating point number.

    """
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


def get_search_radius(mag, minmag, maxmag, minmag_radius, maxmag_radius):
    """From a given magnitude, determines and returns the max radius (in degrees).
        Given minmag_radius and maxmag_radius and minmag and maxmag (FIXME: TO BE CALIBRATED!),
        this function returns D from the f below:

                      |
        maxmag_radius +                oooooooooooo
                      |              o
                      |            o
                      |          o
        minmag_radius + oooooooo
                      |
                      ---------+-------+------------
                            minmag     maxmag

    :return: the max radius (in degrees)
    :param mag: (numeric or list or numbers/numpy.array) the magnitude
    :param minmag: (int, float) the minimum magnitude
    :param maxmag: (int, float) the maximum magnitude
    :param minmag_radius: (int, float) the radius for `min_mag` (in degrees)
    :param maxmag_radius: (int, float) the radius for `max_mag` (in degrees)
    :return: a scalar if mag is scalar, or an numpy.array
    """
    mag = np.asarray(mag)  # do NOT copies data for existing arrays
    isscalar = not mag.shape
    if isscalar:
        mag = np.array(mag, ndmin=1)  # copies data, assures an array of dim=1

    if minmag == maxmag:
        dist = np.array(mag)
        dist[mag < minmag] = minmag_radius
        dist[mag > minmag] = maxmag_radius
        dist[mag == minmag] = np.true_divide(minmag_radius+maxmag_radius, 2)
    else:
        dist = minmag_radius + \
            np.true_divide(maxmag_radius - minmag_radius, maxmag - minmag) * (mag - minmag)
        dist[dist < minmag_radius] = minmag_radius
        dist[dist > maxmag_radius] = maxmag_radius

    return dist[0] if isscalar else dist
