"""
Event-based station search functions
"""
# :date: Dec 3, 2017
from itertools import cycle
from datetime import timedelta
import logging

import numpy as np
import pandas as pd

from stream2segment.download.modules.utils import NothingToDownload, FailedDownload
from stream2segment.io.db.models import Event, Channel, Segment
from stream2segment.io.utils import get_progressbar
from stream2segment.download.modules.events import (
    lat_col as ev_lat_col, lon_col as ev_lon_col, mag_col as mag_col,
    depth_col as ev_depth_col, time_col as ev_time_col,
)
from stream2segment.download.modules.channels import (
    lat_col as ch_lat_col, lon_col as ch_lon_col, net_col, sta_col, loc_col,
    start_col, end_col, band_col, inst_col, orient_col, url_col
)

atime_col = "arrival_time"
dist_col = Segment.event_distance_deg.key
ev_id_col = Segment.event_id.key
ch_id_col = Segment.channel_id.key

# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial):
logger = logging.getLogger(__name__)


def merge_events_stations(
    *,
    events: pd.DataFrame,
    channels: pd.DataFrame,
    search_radius,
    tttable,
    show_progress=False
):
    """
    Merge `events_df` and `channels_df` by returning a new dataframe
    representing all channels within a specific search radius. *Each row of the
    returned data frame is basically a segment to be potentially donwloaded*.
    The returned dataframe will be the same as `channels_df` with one or more
    rows repeated (some channels might be in the search radius of several
    events), plus a column "event_id" (`Segment.event_id`) representing the
    event associated to that channel and two columns 'event_distance_deg',
    'time' (representing the *event* time) and 'depth_km' (representing the
    event depth in km)

    :param channels: pandas DataFrame resulting from `get_channels_df`
    :param events: pandas DataFrame resulting from `get_events_df`
    """
    ret = []

    with get_progressbar(len(events) if show_progress else 0) as pbar:

        min_radia, max_radia = get_search_radia(
            search_radius, events[mag_col].values
        )

        oneday = timedelta(days=1)
        # for min_radius, max_radius, (ev_id, ev_lat, ev_lon, ev_time, ev_depth) \
        #         in radia_event_iter:
        for min_radius, max_radius, ev_id, ev_lat, ev_lon, ev_time, ev_depth in zip(
            min_radia,
            max_radia,
            events[Event.id.key],
            events[ev_lat_col],
            events[ev_lon_col],
            events[ev_time_col],
            events[ev_depth_col]
        ):
            pbar.update(1)

            l2d = locations2degrees(
                channels[ch_lat_col], channels[ch_lon_col], ev_lat, ev_lon
            )
            # set condition incrementally. First, distance must be finite:
            condition = pd.notna(l2d) & (np.abs(l2d) != np.inf)
            # channel start time matches event time:
            condition &= (channels[start_col] <= ev_time)
            # channel end time None or matches event time:
            condition &= (channels[end_col] > ev_time + oneday)
            # add conditions based on matching radia:
            if min_radius:  # not None (legacy code) or 0:
                condition &= (l2d >= min_radius)
            # for max_radius, None means: skip
            if max_radius is not None:
                condition &= (l2d <= max_radius)

            chs = channels[condition]
            if chs.empty:
                continue

            chs["event_distance_deg"] = l2d[condition]
            # add scalar broadcasted to all elements:
            chs[ev_id_col] = ev_id
            chs[ch_id_col] = chs[Channel.id.key]
            chs["_.event_depth._"] = ev_depth
            chs["_.event_time._"] = ev_time
            ret.append(chs)

    # create total segments dataframe:
    # first check we have data:
    if not ret:
        raise NothingToDownload("No station within events search area")
    # now concat:
    ret = pd.concat(ret, axis=0, ignore_index=True, copy=True)

    # check categorical dtypes are preserved (for safety):
    for c in [
        net_col, sta_col, loc_col, band_col, inst_col, orient_col, url_col
    ]:
        if not pd.api.types.is_categorical_dtype(ret[c]):
            ret[c] = ret[c].astype('category')

    # compute travel times. Doing it on a single array is much faster
    source_depths = ret.pop("_.event_depth._").values
    distances = ret["event_distance_deg"].values
    traveltimes = tttable(source_depths, 0, distances)
    # event_times = np.array(event_times, dtype='datetime64[us]')  # or "M8[us]"
    event_times = ret.pop("_.event_time._")
    if not pd.api.types.is_datetime64_any_dtype:  # safety check? FIXME: needed?
        event_times = pd.to_datetime(event_times)
    # now to compute arrival times: event_times + traveltimes does not work
    # (we cannot sum np.datetime64 and np.float). Convert traveltimes to
    # np.timedelta: we first multiply by 1000000 to preserve the millisecond
    # resolution, and then we write traveltimes.astype("m8[us]") which means:
    # 8bytes timedelta with microsecond resolution (10^-6). Side note: all
    # numpy timedelta constructors (as well as "astype") round to int argument,
    # at least in numpy13.
    ret[atime_col] = event_times.values + (traveltimes*1000000).astype("m8[us]")
    # drop nat values
    old_len = len(ret)
    # another safety check (arrival times NaT):
    ret.dropna(subset=[atime_col], inplace=True)
    if old_len > len(ret):
        if ret.empty:
            raise FailedDownload("No segments to process (all travel times NaN)")
        else:
            logger.info(
                f"{old_len - len(ret):,} of {old_len:,} segments discarded (travel times NaN)"
            )

    # convert to event_distance_km:
    degrees = ret.pop('event_distance_deg')
    _overflow = degrees > 180
    if _overflow.any():
        degrees[_overflow] =  360 - degrees[_overflow]
    radius = 6371.0
    event_distance_km =  np.around(
        degrees * 2.0 * radius * np.pi / 360.0, 0
    ).astype(np.int16)
    ret[dist_col] = event_distance_km

    return ret[[
        url_col,
        net_col,
        sta_col,
        loc_col,
        band_col,
        inst_col,
        orient_col,
        atime_col,
        dist_col,
        # Channel.data_webservice_id.key,
        ch_id_col,
        ev_id_col
    ]]


def locations2degrees(lat1, lon1, lat2, lon2):
    """
    Vectorized replacement of legacy ObsPy `locations2degree`, still kept here to
    avoid importing ObsPy in the download package
    """
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
    """
    Return two iterables denoting the minima and maxima radia for
    stations search. Any element of the iterables might be None to indicate:
    no restriction for that element
    """
    if 'min' not in search_radius and 'max' not in search_radius:
        return cycle([None]), get_mag_dependent_radius(magnitudes,
                                                       search_radius['minmag'],
                                                       search_radius['maxmag'],
                                                       search_radius['minmag_radius'],
                                                       search_radius['maxmag_radius'])
    return cycle([search_radius['min']]), cycle([search_radius['max']])


def get_mag_dependent_radius(mag, minmag, maxmag, minmag_radius, maxmag_radius):
    """
    From a given magnitude, return the max radius/radia (in degrees).
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
    is_scalar = not mag.shape
    if is_scalar:
        mag = np.array(mag, ndmin=1)  # copies data, assures an array of dim=1

    if minmag == maxmag:
        dist = np.array(mag)
        dist[mag < minmag] = minmag_radius
        dist[mag >= minmag] = maxmag_radius
    else:
        dist = minmag_radius + (maxmag_radius - minmag_radius) * np.true_divide(
            mag - minmag, maxmag - minmag
        )
        dist[dist < minmag_radius] = minmag_radius
        dist[dist > maxmag_radius] = maxmag_radius

    return dist[0] if is_scalar else dist
