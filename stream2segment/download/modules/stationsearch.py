"""
Event-based station search functions

:date: Dec 3, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
from itertools import cycle
from datetime import timedelta
import logging

import numpy as np
import pandas as pd

from stream2segment.io.db.models import Channel, Event, Segment
from stream2segment.download.modules.utils import formatmsg
from stream2segment.download.exc import FailedDownload
from stream2segment.io.cli import get_progressbar
# from stream2segment.io.db.pdsql import mergeupdate


# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial):
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
    CHA_LAT = Channel.latitude.key  # noqa
    CHA_LON = Channel.longitude.key  # noqa
    CHA_STIME = Channel.start_time.key  # noqa
    CHA_ETIME = Channel.end_time.key  # noqa
    CHA_ID = Channel.id.key  # noqa
    # CHA_STAID = Channel.station_id.key  # noqa
    SEG_EVID = Segment.event_id.key  # noqa
    SEG_EVDIST = Segment.event_distance_deg.key  # noqa
    SEG_ATIME = Segment.arrival_time.key  # noqa
    SEG_WS_ID = Segment.webservice_id.key  # noqa
    SEG_CHAID = Segment.channel_id.key  # noqa
    CHA_NET = Channel.network_code.jey
    CHA_STA = Channel.station_code.key
    CHA_LOC = Channel.location_code.key
    CHA_CHA = Channel.channel_code.key

    # channels_df = channels_df.rename(columns={CHA_ID: SEG_CHAID})
    # get unique stations, rename Channel.id into Segment.channel_id now so we
    # do not bother later
    # stations_df = channels_df.drop_duplicates(subset=[CHA_STAID]).copy()

    ret = []

    # sourcedepths, eventtimes = [], []

    with get_progressbar(len(events_df) if show_progress else 0) as pbar:

        min_radia, max_radia = get_search_radia(search_radius,
                                                events_df[EVT_MAG].values)

        # radia_event_iter = zip(min_radia, max_radia,
        #                        events_df[[EVT_ID, EVT_LAT, EVT_LON, EVT_TIME,
        #                                   EVT_DEPTH]].itertuples(index=False, name=None))

        oneday = timedelta(days=1)
        # for min_radius, max_radius, (ev_id, ev_lat, ev_lon, ev_time, ev_depth) \
        #         in radia_event_iter:
        for min_radius, max_radius, ev_id, ev_lat, ev_lon, ev_time, ev_depth in \
                zip(min_radia,
                    max_radia,
                    events_df[EVT_ID],
                    events_df[EVT_LAT],
                    events_df[EVT_LON],
                    events_df[EVT_TIME],
                    events_df[EVT_DEPTH]):

            l2d = locations2degrees(channels_df[CHA_LAT], channels_df[CHA_LON],
                                    ev_lat, ev_lon)
            # set condition incrementally. First, distance must be finite:
            condition = pd.notna(l2d) & (np.abs(l2d) != np.inf)
            # channel start time matches event time:
            condition &= (channels_df[CHA_STIME] <= ev_time)
            # channel end time None or matches event time:
            condition &= (
                    pd.isnull(channels_df[CHA_ETIME]) |
                    (channels_df[CHA_ETIME] >= ev_time + oneday)
            )
            # add conditions based on matching radia:
            if min_radius:  # not None (legacy code) or 0:
                condition &= (l2d >= min_radius)
            # for max_radius, None means: skip
            if max_radius is not None:
                condition &= (l2d <= max_radius)

            pbar.update(1)
            matching_items = condition.sum()
            if matching_items == 0:
                continue
            if matching_items < len(channels_df):
                channels_df = channels_df[condition]
                l2d = l2d[condition]

            cha_df = channels_df.copy()
            cha_df[SEG_EVDIST] = l2d
            # add scalar broadcasted to all elements:
            cha_df[SEG_EVID] = ev_id
            cha_df[EVT_DEPTH] = [ev_depth] * len(cha_df)
            cha_df[EVT_TIME] = [ev_time] * len(cha_df)
            ret.append(cha_df)

            # FIXME remove?

            # sourcedepths += [ev_depth] * len(cha_df)
            # eventtimes += [ev_time] * len(cha_df)
            # # Set (or re-set from second iteration on) as NaN SEG_EVDIST
            # # columns. This is important cause from second loop on we might
            # # have some elements not-NaN which should be NaN now
            # channels_df[SEG_EVDIST] = np.nan
            # # set locations2 degrees
            # stations_df[SEG_EVDIST] = l2d
            # # Copy distances calculated on stations to their channels
            # # (match along column CHA_STAID shared between the reletive
            # # dataframes). Set values only for channels whose stations are
            # # within radius (stations_df[condition]):
            # cha_df = mergeupdate(channels_df, stations_df[condition],
            #                      [CHA_STAID], [SEG_EVDIST],
            #                      drop_other_df_duplicates=False)
            # # Note above: duplicates already dropped
            # # Now drop channels which are not related to station within radius:
            # cha_df = cha_df.dropna(subset=[SEG_EVDIST], inplace=False).copy()
            # # ...and add "safely" SEG_EVID values:
            # cha_df[SEG_EVID] = ev_id
            # # append to arrays (calculate arrival times in one shot a t the
            # # end, it's faster):
            # sourcedepths += [ev_depth] * len(cha_df)
            # eventtimes += [ev_time] * len(cha_df)
            # # Append only relevant columns:
            # ret.append(cha_df[[SEG_CHAID, SEG_EVID, SEG_DCID, SEG_EVDIST,
            #                    STA_NET, STA_STA, CHA_LOC, CHA_CHA]])

    # create total segments dataframe:
    # first check we have data:
    if not ret:
        raise FailedDownload(formatmsg("No segments to process",
                                       "No station within search radia"))
    # now concat:
    ret = pd.concat(ret, axis=0, ignore_index=True, copy=True)

    # check categoricals are preserved:
    for c in [CHA_NET, CHA_STA, CHA_LOC, CHA_CHA]:
        if not pd.api.types.is_categorical_dtype(ret[c]):
            ret[c] = ret[c].astype('category')

    # compute travel times. Doing it on a single array is much faster
    sourcedepths = ret.pop(EVT_DEPTH).values
    distances = ret[SEG_EVDIST].values
    traveltimes = tttable(sourcedepths, 0, distances)
    # eventtimes = np.array(eventtimes, dtype='datetime64[us]')  # or "M8[us]"
    eventtimes = ret.pop(EVT_TIME)
    if not pd.api.types.is_datetime64_any_dtype:  # safety check? FIXME: needed?
        eventtimes = pd.to_datetime(eventtimes)
    # now to compute arrival times: eventtimes + traveltimes does not work
    # (we cannot sum np.datetime64 and np.float). Convert traveltimes to
    # np.timedelta: we first multiply by 1000000 to preserve the millisecond
    # resolution and then we write traveltimes.astype("m8[us]") which means:
    # 8bytes timedelta with microsecond resolution (10^-6). Side note: all
    # numpy timedelta constructors (as well as "astype") round to int argument,
    # at least in numpy13.
    ret[SEG_ATIME] = eventtimes.values + (traveltimes*1000000).astype("m8[us]")
    # drop nat values
    oldlen = len(ret)
    # another safety check (arival times NaT):
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
    ret = deg(
        atan2(
            sqrt(
                (cos(lat2) * sin(long_diff)) ** 2 +
                (cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(long_diff)) ** 2
            ),
            sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(long_diff)
        )
    )
    return ret


def get_search_radia(search_radius, magnitudes):
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
