# -*- coding: utf-8 -*-
# from __future__ import print_function

"""query_utils: utilities of the package

   :Platform:
       Mac OSX, Linux
   :Copyright:
       Deutsches GFZ Potsdam <XXXXXXX@gfz-potsdam.de>
   :License:
       To be decided!
"""

# standard imports:
from StringIO import StringIO
import sys
import logging
from datetime import timedelta, datetime
# third party imports:
import numpy as np
import pandas as pd
import yaml
from click import progressbar

from stream2segment.utils import url_read, tounicode  # , Progress
from stream2segment.s2sio import db
from stream2segment import __version__ as program_version
from stream2segment.classification import UNKNOWN_CLASS_ID
from stream2segment.classification import class_labels_df

# IMPORT OBSPY AT END! IT MESSES UP WITH IMPORTS!
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees
from obspy.taup.helper_classes import TauModelError
from stream2segment.s2sio.db import DbHandler
# from stream2segment.s2sio.db import DbHandler


def get_min_travel_time(source_depth_in_km, distance_in_degree, model='ak135'):
    """
        Assess and return the travel time of P phases.
        Uses obspy.getTravelTimes
        :param source_depth_in_km: Depth in kilometer.
        :type source_depth_in_km: float
        :param distance_in_degree: Distance in degrees.
        :type distance_in_degree: float
        :param model: Either ``'iasp91'`` or ``'ak135'`` velocity model.
         Defaults to 'ak135'.
        :type model: str, optional
        :return the number of seconds of the assessed arrival time, or None in case of error
        :raises: ValueError (wrapping TauModel error in case)
    """
    taupmodel = TauPyModel(model)
    try:
        tt = taupmodel.get_travel_times(source_depth_in_km, distance_in_degree)
        # return min((ele['time'] for ele in tt if (ele.get('phase_name') or ' ')[0] == 'P'))

        # Arrivals are returned already sorted by time!
        return tt[0].time

        # return min(tt, key=lambda x: x.time).time
        # return min((ele.time for ele in tt))
    except (TauModelError, ValueError, AttributeError) as err:
        raise ValueError(("Unable to find minimum travel time (dist=%s, depth=%s, model=%s). "
                          "Source error: %s: %s"),
                         str(distance_in_degree), str(source_depth_in_km), str(model),
                         err.__class__.__name__, str(err))


def get_arrival_time(distance_in_degrees, ev_depth_km, ev_time):
    """
        Returns the tuple w,c where w is the waveform from the given parameters, and c is the
        relative channel
        :param distance_in_degrees: the distance in degrees
        :type distance_in_degrees: float. See obspy.locations2degrees
        :param dc: the datacenter to query from
        :type dc: string
        :param st: the station to query from
        :type st: string
        :param listCha: the list of channels, e.g. ['HL?', 'SL?', 'BL?']. The function iterates
            over the given channels and returns the first available data
        :type listCha: iterable (e.g., list)
        :param arrivalTime: the query time. The request will be built with a time start and end of
            +-minBeforeP (see below) minutes from arrivalTime
        :type arrivalTime: date or datetime
        :param minBeforeP: the minutes before P wave arrivalTime
        :type minBeforeP: float
        :param minAfterP: the minutes after P wave arrivalTime
        :type minAfterP: float
        :return: the tuple data, channel (bytes and string)
        :raises: ValueError
    """
    travel_time = get_min_travel_time(ev_depth_km, distance_in_degrees)
    arrival_time = ev_time + timedelta(seconds=float(travel_time))
    return arrival_time


def get_arrival_times(distances_series, ev_depth_km, ev_time):
    """returns a Series object """
    def atime(dista):
        """applies get_arrival_time to the given value"""
        try:
            return get_arrival_time(dista, ev_depth_km, ev_time)
        except ValueError:
            return None
            # logging.info('arrival time error: %s' % str(verr))
            # continue
    return distances_series.apply(atime)


def get_time_range(orig_time, days=0, hours=0, minutes=0, seconds=0):
    """
        Returns the tuple (orig_time - timeDeltaBefore, orig_time + timeDeltaAfter), where the deltas
        are built according to the given parameters. Any of the parameters can be an int
        OR an iterable (list, tuple) of two elements specifying the days before and after,
        respectively

        :Example:
            - get_time_range(t, seconds=(1,2)) returns the tuple with elements:
                - t minus 1 second
                - t plus 2 seconds
            - get_time_range(t, minutes=4) returns the tuple with elements:
                - t minus 4 minutes
                - t plus 4 minutes
            - get_time_range(t, days=1, seconds=(1,2)) returns the tuple with elements:
                - t minus 1 day and 1 second
                - t plus 1 day and 2 seconds

        :param days: the day shift from orig_time
        :type days: integer or tuple of positive integers (of length 2)
        :param minutes: the minutes shift from orig_time
        :type minutes: integer or tuple of positive integers (of length 2)
        :param seconds: the second shift from orig_time
        :type seconds: integer or tuple of positive integers (of length 2)
        :return: the tuple (timeBefore, timeAfter)
        :rtype: tuple of datetime objects (timeBefore, timeAfter)
    """
    td1 = []
    td2 = []
    for val in (days, hours, minutes, seconds):
        try:
            td1.append(val[0])
            td2.append(val[1])
        except TypeError:
            td1.append(val)
            td2.append(val)

    start = orig_time - timedelta(days=td1[0], hours=td1[1], minutes=td1[2], seconds=td1[3])
    endt = orig_time + timedelta(days=td2[0], hours=td2[1], minutes=td2[2], seconds=td2[3])
    return start, endt


def get_search_radius(mag, mmin=3, mmax=7, dmin=1, dmax=5):
    """From a given magnitude, determines and returns the max radius (in degrees).
        Given dmin and dmax and mmin and mmax (FIXME: TO BE CALIBRATED!),
        this function returns D from the f below:

             |
        dmax +                oooooooooooo
             |              o
             |            o
             |          o
        dmin + oooooooo
             |
             ---------+-------+------------
                    mmin     mmax

    """
    if mag < mmin:
        radius = dmin
    elif mag > mmax:
        radius = dmax
    else:
        radius = dmin + (dmax - dmin) / (mmax - mmin) * (mag - mmin)
    return radius


def get_events(**kwargs):
    """
        Returns a tuple of two elements: the first one is the DataFrame representing the stations
        read from the specified arguments. The second is the the number of rows (denoting stations)
        which where dropped from the url query due to errors in parsing
        :param kwargs: a variable length list of arguments, including:
            eventws (string): the event web service
            minmag (float): the minimum magnitude
            start (string): the event start, in string format (e.g., datetime.isoformat())
            end (string): the event end, in string format (e.g., datetime.isoformat())
            minlon (float): the event min longitude
            maxlon (float): the event max longitude
            minlat (float): the event min latitude
            maxlat (float): the event max latitude
        :raise: ValueError, TypeError, IOError
    """
    event_query = ('%(eventws)squery?minmagnitude=%(minmag)1.1f&start=%(start)s'
                   '&minlon=%(minlon)s&maxlon=%(maxlon)s&end=%(end)s'
                   '&minlat=%(minlat)s&maxlat=%(maxlat)s&format=text') % kwargs

    result = url_read(event_query, decoding='utf8')

    return evt_to_dframe(result)


def evt_to_dframe(event_query_result):
    """
        :return: the tuple dataframe, dropped_rows (int >=0)
        raises: ValueError
    """
    dfr = query2dframe(event_query_result)
    oldlen = len(dfr)
    if not dfr.empty:
        for key, cast_func in {'Time': pd.to_datetime,
                               'Depth/km': pd.to_numeric,
                               'Latitude': pd.to_numeric,
                               'Longitude': pd.to_numeric,
                               'Magnitude': pd.to_numeric,
                               }.iteritems():
            dfr[key] = cast_func(dfr[key], errors='coerce')

        dfr.dropna(inplace=True)

    return dfr, oldlen - len(dfr)


def get_datacenters(start_time, end_time):
    dcs_query = ('http://geofon.gfz-potsdam.de/eidaws/routing/1/query?service=station&'
                 'start=%s&end=%s&format=post' % (start_time.isoformat(), end_time.isoformat()))
    dc_result = url_read(dcs_query, decoding='utf8')
    dc_result = [k for k in dc_result.split("\n") if k[:7] == "http://"]
    return dc_result


def get_stations(datacenter, channels_list, orig_time, lat, lon, max_radius, level='channel'):
    """
        Returns a tuple of two elements: the first one is the DataFrame representing the stations
        read from the specified arguments. The second is the the number of rows (denoting stations)
        which where dropped from the url query due to errors in parsing
        :param datacenter: the datacenter, e.g.: "http://ws.resif.fr/fdsnws/station/1/query"
        :type datacenter: string
        :param channels_list: the list of channels, e.g. ['HL?', 'SL?', 'BL?'].
        :type channels_list: iterable (e.g., list)
        :param orig_time: the origin time. The request will be built with a time start and end of
            +-1 day from orig_time
        :type orig_time: date or datetime
        :param lat: the latitude
        :type lat: float
        :param lon: the longitude
        :type lon: float
        :param max_radius: the radius distance from lat and lon, in degrees FIXME: check!
        :type max_radius: float
        :return: the DataFrame representing the stations, and the stations dropped (int)
        :raise: ValueError, TypeError, IOError
    """

    start, endt = get_time_range(orig_time, days=1)
    station_query = ('%s?latitude=%3.3f&longitude=%3.3f&'
                     'maxradius=%3.3f&start=%s&end=%s&channel=%s&format=text&level=%s')
    aux = station_query % (datacenter, lat, lon, max_radius, start.isoformat(),
                           endt.isoformat(), ','.join(channels_list), level)
    dc_result = url_read(aux, decoding='utf8')

    return station_to_dframe(dc_result)


def station_to_dframe(stations_query_result):
    """
        :return: the tuple dataframe, dropped_rows (int >=0)
        raises: ValueError
    """
    dfr = query2dframe(stations_query_result)
    oldlen = len(dfr)
    if not dfr.empty:
        for key, cast_func in {'StartTime': pd.to_datetime,
                               'Elevation': pd.to_numeric,
                               'Latitude': pd.to_numeric,
                               'Longitude': pd.to_numeric,
                               'Depth': pd.to_numeric,
                               'Azimuth': pd.to_numeric,
                               'Dip': pd.to_numeric,
                               'SampleRate': pd.to_numeric,
                               'Scale': pd.to_numeric,
                               'ScaleFreq': pd.to_numeric,
                               }.iteritems():
            dfr[key] = cast_func(dfr[key], errors='coerce')

        dfr.dropna(inplace=True)
        dfr['EndTime'] = pd.to_datetime(dfr['EndTime'], errors='coerce')

    return dfr, oldlen - len(dfr)


def query2dframe(query_result_str):
    """
        Returns a pandas dataframne fro the given query_result_str
        :param: query_result_str
        :raise: ValueError in case of errors
    """
    if not query_result_str:
        return pd.DataFrame()

    events = query_result_str.splitlines()

    data = None
    columns = [e.strip() for e in events[0].split("|")]
    for evt in events[1:]:
        evt_list = evt.split('|')
        # Use numpy and then build the dataframe
        # For info on other solutions:
        # http://stackoverflow.com/questions/10715965/add-one-row-in-a-pandas-dataframe:
        if data is None:
            data = [evt_list]
        else:
            data = np.append(data, [evt_list], axis=0)

    if data is not None:
        # check that data rows and columns have the same length
        # cause DataFrame otherwise might do some weird stuff (e.g., one
        # column and rows of N>1 elemens, the DataFrame is built with
        # a single column packing those N elements as list in it)
        # Note that if we are here we are sure data rows are the same length
        np.append(data, [columns], axis=0)

    return pd.DataFrame(data=data, columns=columns)


def get_wav_query(datacenter, network, station_name, location, channel, start_time, end_time):
    """Returns the wav query from the arguments, all strings except the last two (datetime)"""
    qry = '%s/dataselect/1/query?network=%s&station=%s&location=%s&channel=%s&start=%s&end=%s'
    return qry % (datacenter, network, station_name, location, channel, start_time.isoformat(),
                  end_time.isoformat())


def get_wav_queries(dc_series, network_series, station_name_series, location_series, channel_series,
                    start_time_series, end_time_series):
    """Returns the wav query from the arguments, all pandas Series"""

    pddf = pd.DataFrame({'dc': dc_series, 'channel': channel_series, 'network': network_series,
                         'station_name': station_name_series, 'location': location_series,
                         'start_time': start_time_series, 'end_time': end_time_series})

    def func(row):
        """return the wav query from a  dataframe row"""

        return get_wav_query(row['dc'], row['network'], row['station_name'], row['location'],
                             row['channel'], row['start_time'], row['end_time'])

    query_series = pddf.apply(func, axis=1)
    return query_series


def get_distances(latitude_series, longitude_series, ev_lat, ev_lon):
    """returns a DataFrame of distances derived from the given arguments"""
    return pd.DataFrame({'lat': latitude_series,
                         'lon': longitude_series}).apply(lambda row: locations2degrees(ev_lat,
                                                                                       ev_lon,
                                                                                       row['lat'],
                                                                                       row['lon']),
                                                         axis=1)


def get_time_ranges(arrival_times_series, days=0, hours=0, minutes=0, seconds=0):
    """returns two series objects with 'StartTime' 'EndTime' """
    def func(val):
        """returns a dict of 'start' and 'end' keys mapped to the respective times"""
        try:
            tim1, tim2 = get_time_range(val['start'], days=days, hours=hours, minutes=minutes,
                                        seconds=seconds)
        except TypeError:
            tim1, tim2 = None, None
        val['start'], val['end'] = tim1, tim2
        return val

    retval = pd.DataFrame({'start': arrival_times_series,
                           'end': arrival_times_series}).apply(func, axis=1)
    # http://pandas.pydata.org/pandas-docs/stable/dsintro.html#name-attribute
    # The Series name will be assigned automatically in many cases, in particular when taking 1D
    # slices of DataFrame (as it is now). Problem: the constructor
    # (pd.DataFrame(series, columns=[new_col]) will produce a DataFrame with  NaN data in it if
    # new_col is not the same as series name. Solution 1: use pd.DataFrame({'new_name':series}) but
    # for safety there is also the rename method:
    return retval['start'].rename(None), retval['end'].rename(None)


def read_wav_data(query_str):
    """Returns the wav data (mseed binary data) from the given query_str (string)"""
    try:
        return url_read(query_str)
    except (IOError, ValueError, TypeError) as _:
        return None


def pd_str(dframe):
    """Returns a dataframe to string with all rows and all columns, used for printing to log"""
    with pd.option_context('display.max_rows', len(dframe),
                           'display.max_columns', len(dframe.columns),
                           'max_colwidth', 50, 'expand_frame_repr', False):
        return str(dframe)


class LoggerHandler(object):
    """Object handling the root loggers and two Handlers: one writing to StringIO (verbose, being
    saved to db) the other writing to stdout (or stdio) (less verbose, not saved).
    This class has all four major logger methods info, warning, debug and error, plus a save
    method to save the logger text to a database"""
    def __init__(self, out=sys.stdout):
        """
            Initializes a new LoggerHandler, attaching to the root logger two handlers
        """
        root_logger = logging.getLogger()
        root_logger.setLevel(10)
        stringio = StringIO()
        file_handler = logging.StreamHandler(stringio)
        root_logger.addHandler(file_handler)
        console_handler = logging.StreamHandler(out)
        console_handler.setLevel(20)
        root_logger.addHandler(console_handler)
        self.rootlogger = root_logger
        self.errors = 0
        self.warnings = 0
        self.stringio = stringio

    def info(self, *args, **kw):
        """forwards the arguments to L.info, where L is the root Logger"""
        self.rootlogger.info(*args, **kw)

    def debug(self, *args, **kw):
        """forwards the arguments to L.debug, where L is the root Logger"""
        self.rootlogger.debug(*args, **kw)

    def warning(self, *args, **kw):
        """forwards the arguments to L.debug (with "WARNING: " inserted at the beginning of the log
        message), where L is the root logger. This allows this kind of log messages
        to be printed to the db log but NOT on the screen (less verbose)"""
        args = list(args)  # it's a tuple ...
        args[0] = "WARNING: " + args[0]
        self.warnings += 1
        self.rootlogger.debug(*args, **kw)

    def error(self, *args, **kw):
        """forwards the arguments to L.error, where L is the root Logger"""
        self.errors += 1
        self.rootlogger.error(*args, **kw)

    def to_df(self, seg_found, seg_written, config_text=None, close_stream=True,
              datetime_now=None):
        """Saves the logger informatuon to database"""
        if datetime_now is None:
            datetime_now = datetime.utcnow()
        pddf = pd.DataFrame([[datetime_now, tounicode(self.stringio.getvalue()), self.warnings,
                              self.errors, seg_found, seg_written, seg_found - seg_written,
                              tounicode(config_text) if config_text else tounicode(""),
                              ".".join(str(v) for v in program_version)]],
                            columns=["Id", "Log", "Warnings", "Errors", "SegmentsFound",
                                     "SegmentsWritten", "SegmentsSkipped", "Config",
                                     "ProgramVersion"])
        if close_stream:
            self.stringio.close()
        return pddf


def save_waveforms(eventws, minmag, minlat, maxlat, minlon, maxlon, search_radius_args,
                   channels, start, end, ptimespan, min_sample_rate,
                   outpath):
    """
        Downloads waveforms related to events to a specific path
        :param eventws: Event WS to use in queries. E.g. 'http://seismicportal.eu/fdsnws/event/1/'
        :type eventws: string
        :param minmaa: Minimum magnitude. E.g. 3.0
        :type minmaa: float
        :param minlat: Minimum latitude. E.g. 30.0
        :type minlat: float
        :param maxlat: Maximum latitude E.g. 80.0
        :type maxlon: float
        :param minlon: Minimum longitude E.g. -10.0
        :type minlon: float
        :param maxlon: Maximum longitude E.g. 60.0
        :type maxlon: float
        :param search_radius_args: The arguments required to get the search radius R whereby all
            stations within R will be queried from a given event location E_lat, E_lon
        :type search_radius_args: list or iterable of numeric values:
            (min_magnitude, max_magnitude, min_distance, max_distance)
        :param datacenters_dict: a dict of data centers as a dictionary of the form
            {name1: url1, ..., nameN: urlN} where url1, url2,... are strings
        :type datacenters_dict dict of key: string entries
        :param channels: iterable (e.g. list) of channels (as strings), e.g.
            ['HH?', 'SH?', 'BH?', 'HN?', 'SN?', 'BN?']
        :type channels: iterable of strings
        :param start: Limit to events on or after the specified start time
            E.g. (date.today() - timedelta(days=1))
        :type start: datetime
        :param end: Limit to events on or before the specified end time
            E.g. date.today().isoformat()
        :type end: datetime
        :param ptimespan: the minutes before and after P wave arrival for the waveform query time
            span
        :type ptimespan: iterable of two float
        :param min_sample_rate: the minimum sample rate required to download data
        channels with a field 'SampleRate' lower than this value (in Hz) will be discarded and
        relative data not downloaded
        :type min_sample_rate: float
        :param outpath: path where to store mseed files E.g. '/tmp/mseeds'
        :type outpath: string
    """
    _args_ = dict(locals())  # this must be the first statement, so that we catch all arguments and
    # no local variable (none has been declared yet). Note: dict(locals()) avoids problems with
    # variables created inside loops, when iterating over _args_ (see below)

    logger = LoggerHandler()

    # print local vars:
    yaml_content = StringIO()
    yaml_content.write(yaml.dump(_args_, default_flow_style=False))
    logger.info("Arguments:")
    tab = "   "
    logger.info(tab + yaml_content.getvalue().replace("\n", "\n%s" % tab))

    # a little bit hacky, but convert to dict as the function gets dictionaries
    # Note: we might want to use dict(locals()) as above but that does NOT
    # preserve order and tests should be rewritten. It's too much pain for the moment
    args = {"eventws": eventws,
            "minmag": minmag,
            "minlat": minlat,
            "maxlat": maxlat,
            "minlon": minlon,
            "maxlon": maxlon,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "outpath": outpath}

    logger.debug("")
    logger.info("STEP 1/3: Querying Event WS")

    # initialize our Database handler:
    dbwriter = db.DbHandler(outpath)
    try:
        events_df, skipped = get_events(**args)
        # raise ValueError()
    except (IOError, ValueError, TypeError) as err:
        logger.error(str(err))
        log_dframe = logger.to_df(seg_found=0, seg_written=0,
                                  config_text=yaml_content.getvalue())
        dbwriter.write(log_dframe, dbwriter.tables.runs)
        return 1
    else:
        if skipped > 0:
            logger.warning(("%d events skipped (possible cause: bad formatting, "
                            "e.g. invalid datetimes or numbers") % skipped)

    logger.info('%s events found', len(events_df))
    logger.debug('Events: %s', pd_str(events_df))

    # get all datacenters:
    datacenters = get_datacenters(start, end)

    data_df = None
    logger.debug("")
    msg = "STEP 2/3: Querying Station WS (level=channel)"
    logger.info(msg)
    n_step = 0

    # First define column names once (avoid typos):
    atime_col = "ArrivalTime"
    dist_col = "EventDistance/deg"
    stime_col = "DataStartTime"
    etime_col = "DataEndTime"
    lat_col = 'Latitude'
    lon_col = 'Longitude'
    mag_col = 'Magnitude'
    srate_col = 'SampleRate'
    cha_col = 'Channel'
    sta_col = 'Station'
    net_col = '#Network'
    loc_col = 'Location'

    with progressbar(length=len(events_df) * len(datacenters)) as _bar:
        for _, row in events_df.iterrows():
            ev_mag = row[mag_col]
            ev_id = row['#EventID']
            ev_loc_name = row['EventLocationName']
            ev_time = row['Time']
            ev_lat = row[lat_col]
            ev_lon = row[lon_col]
            ev_depth_km = row['Depth/km']

            max_radius = get_search_radius(ev_mag,
                                           search_radius_args[0],
                                           search_radius_args[1],
                                           search_radius_args[2],
                                           search_radius_args[3])

            for dcen in datacenters:

                n_step += 1
                _bar.update(n_step)

                msg = ("Event %s (%s): querying stations within %5.3f deg. "
                       "to %s") % (ev_id, ev_loc_name, max_radius, dcen)

                logger.debug("")
                logger.debug(msg)

                try:
                    stations_cha_level, skipped = get_stations(dcen, channels, ev_time, ev_lat,
                                                               ev_lon, max_radius)
                except (IOError, ValueError, TypeError) as exc:
                    logger.warning(exc.__class__.__name__ + ": " + str(exc))
                    continue

                logger.debug('%d stations found (data center: %s, channel: %s)',
                             len(stations_cha_level), str(dcen), str(channels))

                if skipped > 0:
                    logger.warning(("%d stations skipped (possible cause: bad formatting, "
                                    "e.g. invalid datetimes or numbers") % skipped)

                if min_sample_rate > 0 and not stations_cha_level.empty:
                    tmp = stations_cha_level[stations_cha_level[srate_col] >= min_sample_rate]
                    if len(tmp) != len(stations_cha_level):
                        logger.warning(("%d stations skipped (sample rate < %s Hz") %
                                       (len(stations_cha_level) - len(tmp), str(min_sample_rate)))
                        stations_cha_level = tmp

                if stations_cha_level.empty:
                    continue

                # Now calculate. As arrival_times is computationally expensive. We might have
                # DUPLICATED stations so we select only those unique according to Latitude and
                # longitude
                stations_unique = stations_cha_level.drop_duplicates(subset=(lat_col, lon_col))
                # NOTE: the function above calculates duplicated if ALL subset(s) are equal, if any
                # is equal then does not drop them (what we want)

                # set stations_unique as "not a copy" to suppress pandas warning, as that warning
                # does tell us that we are not modifying the original stations dataframe, which is
                # what we are aware of
                stations_unique.is_copy = False  # suppress warning

                # add a column distances, arrival times etcetera to stations_unique
                stations_unique.loc[:, dist_col] = get_distances(stations_unique[lat_col],
                                                                 stations_unique[lon_col],
                                                                 ev_lat, ev_lon)
                stations_unique.loc[:, atime_col] = get_arrival_times(stations_unique[dist_col],
                                                                      ev_depth_km, ev_time)
                stations_unique.loc[:, stime_col], \
                    stations_unique.loc[:, etime_col] = get_time_ranges(stations_unique[atime_col],
                                                                        minutes=ptimespan)

                # print stations unique. It has infos about times and distances, we drop all channel
                # info (this might speed up rendering for long DataFrame and make things more
                # readable)
                logger.debug("Downloaded stations (unique, i.e. showing with level=station):")
                sts = stations_unique[dbwriter.STATION_TBL_COLUMNS]
                logger.debug(pd_str(sts))

                # build our DataFrame (extension of stations DataFrame):
                wdf = stations_cha_level
                # add specific segments columns
                # it's important to initialize some to na (None NaT or NaN) as we will drop those
                # values later (na means some error, thus warn in the log)
                wdf.insert(0, '#EventID', ev_id)
                wdf.insert(1, atime_col, pd.NaT)
                wdf.insert(2, dist_col, np.NaN)
                wdf.insert(3, stime_col, pd.NaT)
                wdf.insert(4, etime_col, pd.NaT)
                wdf.insert(5, 'QueryStr', dcen)  # this is the Datacenter (for the moment) later the
                # query string (see below)
                wdf.insert(6, 'ClassId', UNKNOWN_CLASS_ID)
                wdf.insert(7, 'ClassIdHandLabeled', False)
                wdf.insert(8, 'RunId', pd.NaT)

                # set in wdf 'stations-event distances', 'arrival time' and 'time_ranges' columns:
                def func(sur):
                    """populate our dataframe with the unique values, acounting for duplicates stations
                       due to different channels. sur = stations_unique_row"""
                    row_selector_df = (wdf[lat_col] == sur[lat_col]) & \
                        (wdf[lon_col] == sur[lon_col])  # pylint: disable=W0640
                    wdf.loc[row_selector_df, atime_col] = sur[atime_col]  # pylint: disable=W0640
                    wdf.loc[row_selector_df, stime_col] = sur[stime_col]  # pylint: disable=W0640
                    wdf.loc[row_selector_df, etime_col] = sur[etime_col]  # pylint: disable=W0640
                    wdf.loc[row_selector_df, dist_col] = sur[dist_col]  # pylint: disable=W0640

                stations_unique.apply(func, axis=1)  # apply is generally faster than iterrows

                # dropna D from distances, arr_times, time_ranges which are na
                dict_ = {(dist_col,): "station-event distance",
                         (atime_col,): "arrival time",
                         (stime_col, etime_col): "time-range around arrival time"}
                for subset, reason in dict_.iteritems():
                    _l_ = len(wdf)
                    wdf.dropna(subset=subset, inplace=True)
                    _l_ -= len(wdf)
                    if _l_ > 0:
                        logger.warning("%d stations removed (reason %s)" % (_l_, reason))

                # reset index so that we have nonnegative ordered natural numbers 0, ... N:
                wdf.reset_index(inplace=True, drop=True)

                # NOTE: wdf['QueryStr'] is the data center (will be filled with the query string
                # below). FIXME: HORRIBLE HACK!!!
                if 'location' in wdf.columns:
                    wdf = wdf.rename(columns={'location': 'Location'})
                wdf.loc[:, 'QueryStr'] = get_wav_queries(wdf['QueryStr'], wdf[net_col],
                                                         wdf[sta_col], wdf[loc_col],
                                                         wdf[cha_col], wdf[stime_col],
                                                         wdf[etime_col])

                # skip when the dataframe is empty. Moreover, this apparently avoids shuffling
                # column order
                if not wdf.empty:
                    data_df = wdf if data_df is None else data_df.append(wdf, ignore_index=True)

    logger.debug("")
    logger.info("STEP 3/3: Querying Datacenter WS")

    total = 0
    skipped_error = 0
    skipped_empty = 0
    skipped_already_saved = 0

    # set data_df to empty if None (makes life easier by checking if data.df.empty leter on)
    if data_df is None:
        data_df = pd.DataFrame([])

    if not data_df.empty:

        # FIXME: REMOVE THIS COMMENT IF NOT RELEVANT ANYMORE!!!
        # append reorders the columns, so set them as we wanted
        # Note that wdf is surely defined
        # Note also that now column order is not anymore messed up, but do this for safety:
        # data_df = data_df[wdf.columns]

        # purge wav_data (this creates a column id primary key):
        original_data_len = len(data_df)
        data_df = dbwriter.purge(data_df, dbwriter.T_SEG)
        skipped_already_saved = original_data_len - len(data_df)

        logger.debug("Downloading and saving %d of %d waveforms (%d already saved)",
                     len(data_df), original_data_len, skipped_already_saved)

        # it turns out that now wav_data is a COPY of data_df
        # any further operation on it raises a SettingWithCopyWarning, thus avoid issuing it:
        # http://stackoverflow.com/questions/23688307/settingwithcopywarning-even-when-using-loc
        data_df.is_copy = False
        data_df.reset_index(drop=True, inplace=True)

        logger.debug("")

        with progressbar(length=len(data_df)) as bar:
            # insert binary data (empty)
            def func_dwav(row_series):
                query_str = row_series['QueryStr']
                data = read_wav_data(query_str)
                msg = "%7d bytes downloaded from: %s" % (len(data), query_str)
                logger.debug(msg)
                bar.update(row_series.name + 1)  # series name is the original dframe index
                return data

            binary_data_series = data_df.apply(func_dwav, axis=1)

        data_df.insert(1, 'Data', binary_data_series)

        # purge stuff which is not good:
        _len_data_df = len(data_df)
        data_df.dropna(subset=['Data'], inplace=True)
        skipped_error = _len_data_df - len(data_df)

        # purge empty stuff:
        _len_data_df = len(data_df)
        data_df = data_df[data_df['Data'] != b'']
        skipped_empty = _len_data_df - len(data_df)

    logger.debug("")
    if logger.warnings:
        print "%d warnings (check log for details)" % logger.warnings

    seg_written = total-skipped_empty-skipped_error-skipped_already_saved
    logger.info(("%d segments written to '%s', "
                 "%d skipped (%d already saved, %d due to url error, %d empty). "
                 "Total number of segments found: %d"),
                seg_written,
                outpath,
                total - seg_written,
                skipped_already_saved,
                skipped_error,
                skipped_empty,
                total)

    # write the class labels:
    dbwriter.write(class_labels_df, dbwriter.T_CLS, if_exists='skip')  # fail means: do nothing
    # write events:
    # first purge them then write
    new_events_df = dbwriter.purge(events_df, dbwriter.T_EVT)
    dbwriter.write(new_events_df, dbwriter.T_EVT)
    # write data:
    now_ = datetime.utcnow()  # set a common datetime now for runs and data
    if not data_df.empty:
        stations_df = data_df[dbwriter.STATION_TBL_COLUMNS]
        pkeycol = dbwriter.table_settings[dbwriter.T_STA_NAME]['pkey']
        stations_df.insert(0, pkeycol, stations_df[net_col] + "." + stations_df[sta_col])
        stations_df = stations_df.drop_duplicates(subset=[pkeycol])
        stations_df = dbwriter.write(data_df, dbwriter.T_STA_NAME, purge_first=True)

        channels_df = data_df[dbwriter.CHANNEL_TBL_COLUMNS]
        channels_df.insert(0, "StationId", channels_df[net_col] + "." + channels_df[sta_col])
        pkeycol = dbwriter.table_settings[dbwriter.T_CHA_NAME]['pkey']
        channels_df.insert(0, "Id", channels_df[net_col] + "." + channels_df[sta_col] + "." +
                           channels_df[loc_col] + "." + channels_df[cha_col])
        channels_df = channels_df.drop_duplicates(subset=[pkeycol])
        channels_df = dbwriter.write(data_df, dbwriter.T_CHA_NAME, purge_first=True)

        non_segs_col = dbwriter.CHANNEL_TBL_COLUMNS + dbwriter.STATION_TBL_COLUMNS
        id_tmp_cols = (net_col, sta_col, loc_col, cha_col)  # columns to be kept (temporarily)
        non_segs_col = [k for k in non_segs_col if k not in id_tmp_cols]
        segments_df = data_df.drop(non_segs_col, axis=1)
        segments_df['RunId'] = now_
        pkeycol = dbwriter.table_settings[dbwriter.T_SEG_NAME]['pkey']
        segments_df.insert(0, 'ChannelId', segments_df[net_col] + "." + segments_df[sta_col] + "." +
                           segments_df[loc_col] + "." + segments_df[cha_col])
        segments_df.insert(0, pkeycol, None)

        def myfunc(row):
            row[pkeycol] = hash((row['#EventID'], row[net_col], row[sta_col],
                                 row[loc_col], row[cha_col],
                                 row[stime_col].isoformat(),
                                 row[etime_col].isoformat()))
            return row

        segments_df = segments_df.apply(myfunc, axis=1)
        segments_df = segments_df.drop(id_tmp_cols, axis=1)
        segments_df = segments_df.rename({stime_col: 'StartTime', etime_col: 'EndTime'})
        segments_df = dbwriter.write(segments_df, dbwriter.T_SEG_NAME)

    # write log:
    log_df = logger.to_df(seg_found=total, seg_written=seg_written,
                          config_text=yaml_content.getvalue(), datetime_now=now_)
    dbwriter.write(log_df, dbwriter.T_RUN)

    return 0
