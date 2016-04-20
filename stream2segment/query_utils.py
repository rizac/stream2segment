#!/usr/bin/python
# event2wav: First draft to download waveforms related to events
#
# (c) 2015 Deutsches GFZ Potsdam
# <XXXXXXX@gfz-potsdam.de>
#
# ----------------------------------------------------------------------
"""utils: utilities of the package

   :Platform:
       Mac OSX, Linux
   :Copyright:
       Deutsches GFZ Potsdam <XXXXXXX@gfz-potsdam.de>
   :License:
       To be decided!
"""

# standard imports:
import logging
from matplotlib.dates import date2num
# from datetime import datetime
from datetime import timedelta
from stream2segment.utils import datetime as dtime, EstRemTimer, url_read
from stream2segment import io as s2sio
import numpy as np
import pandas as pd
from sqlalchemy import BLOB
from pandas.core import indexing
# third party imports:
# from obspy.taup.taup import getTravelTimes
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees
from obspy.taup.helper_classes import TauModelError


def get_min_travel_time(source_depth_in_km, distance_in_degree, model='ak135'):  # FIXME: better!
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
        return min((ele.time for ele in tt))
    except (TauModelError, ValueError) as err:
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
    arrivalTime = ev_time + timedelta(seconds=float(travel_time))
    return arrivalTime


def get_time_range(origTime, days=0, hours=0, minutes=0, seconds=0):
    """
        Returns the tuple (origTime - timeDeltaBefore, origTime + timeDeltaAfter), where the deltas
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

        :param days: the day shift from origTime
        :type days: integer or tuple of positive integers (of length 2)
        :param minutes: the minutes shift from origTime
        :type minutes: integer or tuple of positive integers (of length 2)
        :param seconds: the second shift from origTime
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

    start = origTime - timedelta(days=td1[0], hours=td1[1], minutes=td1[2], seconds=td1[3])
    endt = origTime + timedelta(days=td2[0], hours=td2[1], minutes=td2[2], seconds=td2[3])
    return start, endt


def get_search_radius(mag, mmin=3, mmax=7, dmin=1, dmax=5):  # FIXME: better!
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
    eventQuery = ('%(eventws)squery?minmagnitude=%(minmag)1.1f&start=%(start)s'
                  '&minlon=%(minlon)s&maxlon=%(maxlon)s&end=%(end)s'
                  '&minlat=%(minlat)s&maxlat=%(maxlat)s&format=text') % kwargs

    result = url_read(eventQuery, decoding='utf8')

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


def get_stations(dc, listCha, origTime, lat, lon, max_radius):
    """
        Returns a tuple of two elements: the first one is the DataFrame representing the stations
        read from the specified arguments. The second is the the number of rows (denoting stations)
        which where dropped from the url query due to errors in parsing
        :param dc: the datacenter
        :type dc: string
        :param listCha: the list of channels, e.g. ['HL?', 'SL?', 'BL?'].
        :type listCha: iterable (e.g., list)
        :param origTime: the origin time. The request will be built with a time start and end of +-1
            day from origTime
        :type origTime: date or datetime
        :param lat: the latitude
        :type lat: float
        :param lon: the longitude
        :type lon: float
        :param max_radius: the radius distance from lat and lon, in degrees FIXME: check!
        :type max_radius: float
        :return: the DataFrame representing the stations, and the stations dropped (int)
        :raise: ValueError, TypeError, IOError
    """

    dcResult = ''
    try:
        # start, endt = get_time_range(origTime, timedelta(days=1))
        start, endt = get_time_range(origTime, days=1)
    except TypeError:
        logging.error('Cannot convert origTime parameter (%s).', origTime)
    else:
        stationQuery = ('%s/station/1/query?latitude=%3.3f&longitude=%3.3f&'
                        'maxradius=%3.3f&start=%s&end=%s&channel=%s&format=text&level=station')
        aux = stationQuery % (dc, lat, lon, max_radius, start.isoformat(),
                              endt.isoformat(), ','.join(listCha))
        dcResult = url_read(aux, decoding='utf8')

    return station_to_dframe(dcResult)


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
    for ev in events[1:]:
        evt_list = ev.split('|')
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


def get_wav_id(event_id, network, station, location, channel, start_time, end_time):
    """
        Returns a unique id from the given arguments. The hash of the tuple built from the
        arguments will be returned. No argument can be None
        :param: event_id: the event_id
        :type: event_id: string
        :param: network: the station network
        :type: network: string
        :param: station: the given station
        :type: station: string
        :param: location: the given location
        :type: location: string
        :param: channel: the given channel
        :type: channel: string
        :param: start_time: the wav start time
        :type: start_time: datetime object, or a string representing a datetime
            (e.g. datetime.isoformat())
        :param: end_time: the wav end time, or a string representing a datetime
            (e.g. datetime.isoformat())
        :type: end_time: datetime object
        :return: a unique integer denoting the given wav.
        Two wavs with the same argument have the same id
        :rtype: integer
    """
    val = (event_id, network, station, channel, location, dtime(start_time), dtime(end_time))
    if None in val:
        raise ValueError("No None value in get_wav_id")
    return hash(val)


def get_wav_dframe(stations_dataframe, dc,
                   chList, ev_id, ev_lat, ev_lon, ev_depth_km, ev_time, ptimespan):
    data = []
    qry = '%s/dataselect/1/query?station=%s&channel=%s&start=%s&end=%s'

    if not stations_dataframe.empty:
        for st in stations_dataframe.values:
            st_name = st[1]
            st_lat = st[2]
            st_lon = st[3]
            st_network = st[0]
            st_location = ''  # FIXME: ask

            dista = locations2degrees(ev_lat, ev_lon, st_lat, st_lon)
            try:
                arrivalTime = get_arrival_time(dista, ev_depth_km, ev_time)
            except ValueError as verr:
                logging.info('arrival time error: %s' % str(verr))
                continue

            try:
                start_time, end_time = get_time_range(arrivalTime,
                                                      minutes=(ptimespan[0], ptimespan[1]))
            except TypeError:
                logging.error('Cannot convert arrivalTime parameter (%s).', arrivalTime)
                continue

            # now build the data frames
            for cha in chList:
                wav_query = qry % (dc, st_name, cha, start_time.isoformat(), end_time.isoformat())
                data_row = [None, ev_id, cha, "", start_time, end_time, dc,  dista, arrivalTime,
                            wav_query, b'']
                data_row.extend(st)
                data_row[0] = get_wav_id(ev_id, st_network, st_name, st_location, cha,
                                         start_time, end_time)
                data.append(data_row)

    colz = ['Id', '#EventID_fk', 'Channel', 'Location', 'StartTime', 'EndTime', 'DataCenter',
            'Distance/deg', 'ArrivalTime', 'QueryStr', 'Data']
    colz.extend(['Station_' + s for s in stations_dataframe.columns])
    wav_dframe = pd.DataFrame(columns=colz) if not data else pd.DataFrame(columns=colz, data=data)
    return wav_dframe


def read_wav_data(query_str):
    try:
        return url_read(query_str)
    except (IOError, ValueError, TypeError) as exc:
        # logging.warning(str(exc))
        return None


def save_waveforms(eventws, minmag, minlat, maxlat, minlon, maxlon, search_radius_args,
                   datacenters_dict, channelList, start, end, ptimespan, outpath):
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
        :param channelList: iterable (e.g. list) of channels. Each channels is in turn an iterable
            of strings, e.g. ['HH?', 'SH?', 'BH?']
            Thus, channelList might be [['HH?', 'SH?', 'BH?'], ['HN?', 'SN?', 'BN?']]
        :type channelList: iterable of iterables of strings
        :param start: Limit to events on or after the specified start time
            E.g. (date.today() - timedelta(days=1)).isoformat()
        :type start: datetime or string, as returned from datetime.isoformat()
        :param end: Limit to events on or before the specified end time
            E.g. date.today().isoformat()
        :type end: datetime or string, as returned from datetime.isoformat()
        :param ptimespan: the minutes before and after P wave arrival for the waveform query time
            span
        :type ptimespan: iterable of two float
        :param outpath: path where to store mseed files E.g. '/tmp/mseeds'
        :type outpath: string
    """

    # print local vars:
    logging.info("Arguments:")
    for arg, varg in dict(locals()).iteritems():
        # Note: locals() might be updated inside the loop and throw an
        # error, as it stores all local variables.
        # Thus the need of dict(locals())
        logging.info("\t%s = %s", str(arg), str(varg))

    # a little bit hacky, but convert to dict as the function gets dictionaries
    # Note: we might want to use dict(locals()) as above but that does NOT
    # preserve order and tests should be rewritten. It's topo pain for the moment
    args = {"eventws": eventws,
            "minmag": minmag,
            "minlat": minlat,
            "maxlat": maxlat,
            "minlon": minlon,
            "maxlon": maxlon,
            "start": start,
            "end": end,
            "outpath": outpath}

    # Get events in text format. '|' separated values
    logging.info("Querying Event WS:")
    try:
        events, skipped = get_events(**args)
    except (IOError, ValueError, TypeError) as err:
        logging.error(str(err))
        return
    else:
        if skipped > 0:
            logging.warning(("%d events skipped (possible cause: bad formatting, "
                             "e.g. invalid datetimes or numbers") % skipped)

    logging.info('%s events found', len(events))
    logging.debug('Events: %s', events)

    db_handler = s2sio.DbHandler()
    events_table_name = "events"
    events_pkey = '#EventID'
    new_events = db_handler.purge(events, events_table_name, events_pkey)
    db_handler.write(new_events, events_table_name, events_pkey)

    wav_dframe = None
    ert = EstRemTimer(len(events))

    for ev in events.values:
        ev_mag = ev[10]
        ev_id = ev[0]
        ev_loc_name = ev[12]
        ev_time = ev[1]
        ev_lat = ev[2]
        ev_lon = ev[3]
        ev_depth_km = ev[4]
        remtime = ert.get()
        evtMsg = 'Processing event %s %s (mag=%s)' % (ev_id, ev_loc_name, ev_mag)
        etr_msg = ('%s done. '
                   'Est. remaining time: '
                   '%s') % (ert.percent(), "unknown" if remtime is None else str(remtime))
        strmsg = etr_msg + ". " + evtMsg
        logging.info("")
        logging.info(strmsg)

        max_radius = get_search_radius(ev_mag,
                                       search_radius_args[0],
                                       search_radius_args[1],
                                       search_radius_args[2],
                                       search_radius_args[3])

        logging.info(('Querying Station WS (selected data-centers and channels)'
                      ' for stations within %s degrees:') % max_radius)

        for DCID, dc in datacenters_dict.iteritems():
            for chName, chList in channelList.iteritems():
                try:
                    stations, skipped = get_stations(dc, chList, ev_time, ev_lat, ev_lon,
                                                     max_radius)
                except (IOError, ValueError, TypeError) as exc:
                    logging.warning(str(exc))
                    continue

                if not stations.empty:
                    logging.info('%d stations found (data center: %s, channel: %s)',
                                 len(stations), str(DCID), str(chList))

                if skipped > 0:
                    logging.warning(("%d stations skipped (possible cause: bad formatting, "
                                     "e.g. invalid datetimes or numbers") % skipped)

                if stations.empty:
                    continue

                logging.debug('Stations: %s', stations)

                # get distance
                wdf = get_wav_dframe(stations, dc, chList, ev_id, ev_lat, ev_lon, ev_depth_km,
                                     ev_time, ptimespan)

                # skip when the dataframe is empty. Moreover, this apparently avoids shuffling
                # column order
                if not wdf.empty:
                    wav_dframe = wdf if wav_dframe is None else wav_dframe.append(wdf,
                                                                                  ignore_index=True)

    total = 0
    skipped_error = 0
    skipped_empty = 0
    skipped_already_saved = 0
    if wav_dframe is not None:
        total = len(wav_dframe)
        data_table_name = "data"
        data_pkey = "Id"
        data_dtype = {'Data': BLOB}

        # append reorders the columns, so set them as we wanted
        # Note that wdf is surely defined
        # Note also that now column order is not anymore messed up, but do this for safety:
        wav_dframe = wav_dframe[wdf.columns]

        wav_data = db_handler.purge(wav_dframe, data_table_name, data_pkey)
        skipped_already_saved = total - len(wav_data)
        logging.info("")
        logging.info(("Querying Datacenter WS: downloading and saving %d of %d waveforms"
                      "(%d already saved)") %
                     (len(wav_data), len(wav_dframe), len(wav_dframe) - len(wav_data)))

        ert = EstRemTimer(len(wav_data))

        # it turns out that now wav_data is a COPY of wav_dframe
        # any further operation on it raises a SettingWithCopyWarning, thus avoid issuing it:
        # http://stackoverflow.com/questions/23688307/settingwithcopywarning-even-when-using-loc
        wav_data.is_copy = False

        # do a loop on the index. Inefficient, but in any case we need to perform
        # url queries so the benefits of using pandas methods such as e.g. apply might vanish
        # (therefore avoid premature optimization, a test scenario should be built)
        for i in wav_data.index:
            query_str = wav_data.loc[i, 'QueryStr']
            etr_msg = ('%s done. '
                       'Est. remaining time: '
                       '%s (%s)') % (ert.percent(), remtime, query_str)
            logging.info(etr_msg)
            data = read_wav_data(query_str)
            wav_data.loc[i, 'Data'] = data

        # purge stuff which is not good:
        wav_data.dropna(subset=['Data'], inplace=True)
        skipped_error = (total - skipped_already_saved) - len(wav_data)
        wav_data = wav_data[wav_data['Data'] != b'']
        skipped_empty = (total - skipped_already_saved - skipped_error) - len(wav_data)
        db_handler.write(wav_data, data_table_name, data_pkey, dtype=data_dtype)

    logging.info("")
    logging.info(("DONE: %d of %d waveforms (mseed files) written to '%s', "
                  "%d skipped (%d already saved, %d due to url error, %d empty)"),
                 total-skipped_empty-skipped_error-skipped_already_saved,
                 total,
                 outpath,
                 skipped_empty+skipped_error+skipped_already_saved,
                 skipped_already_saved,
                 skipped_error,
                 skipped_empty)


# Event WS
# Station WS
# Dataselect WS