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

# third party imports:
# from obspy.taup.taup import getTravelTimes
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees
from obspy.taup.helper_classes import TauModelError


def get_travel_time(source_depth_in_km, distance_in_degree, model='ak135'):  # FIXME: better!
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
    """
    taupmodel = TauPyModel(model)
    try:
        tt = taupmodel.get_travel_times(source_depth_in_km, distance_in_degree)
        # return min((ele['time'] for ele in tt if (ele.get('phase_name') or ' ')[0] == 'P'))
        return min((ele.time for ele in tt))
    except (TauModelError, ValueError):
        logging.error("Unable to find arrival time. Phase names (dist=%s, depth=%s, model=%s):\n%s",
                      str(distance_in_degree), str(source_depth_in_km), str(model),
                      ','.join(ele.get('phase_name') for ele in tt))
        return None
    # ttsel=[ele for ele in tt if ele.get('phase_name') in ['Pg','Pn','Pb']]
    # ttime=[ele['time'] for ele in ttsel]
    # arrtime=min(ttime)
    # return arrtime


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
        Return the events resulting from a query in a list
        :param kwargs: a variable length list of arguments, including:
            eventws (string): the event web service
            minmag (float): the minimum magnitude
            start (string): the event start, in string format (e.g., datetime.isoformat())
            end (string): the event end, in string format (e.g., datetime.isoformat())
            minlon (float): the event min longitude
            maxlon (float): the event max longitude
            minlat (float): the event min latitude
            maxlat (float): the event max latitude
    """
    eventQuery = ('%(eventws)squery?minmagnitude=%(minmag)1.1f&start=%(start)s'
                  '&minlon=%(minlon)s&maxlon=%(maxlon)s&end=%(end)s'
                  '&minlat=%(minlat)s&maxlat=%(maxlat)s&format=text') % kwargs

    result = url_read(eventQuery, decoding='utf8')

    return evt_to_dframe(result)


def evt_to_dframe(event_query_result):
    if not event_query_result:
        return pd.DataFrame()

    events = event_query_result.splitlines()
    dframe = pd.DataFrame(index=np.arange(0, len(events)-1),
                          columns=(e.strip() for e in events[0].split("|")))
    for i, ev in enumerate(events[1:]):
        evt_list = ev.split('|')
        evt_list[1] = dtime(evt_list[1], on_err_return_none=True)
        if evt_list[1] is None:
            logging.error("Couldn't convert origTime parameter (%s).", evt_list[1])
            continue

        try:
            evt_list[2] = float(evt_list[2])
            evt_list[3] = float(evt_list[3])
            evt_list[4] = float(evt_list[4])
            evt_list[10] = float(evt_list[10])
            # http://stackoverflow.com/questions/10715965/add-one-row-in-a-pandas-dataframe:
            # loc or iloc both work here since the index is natural numbers
            dframe.loc[i] = evt_list
        except (ValueError, IndexError) as err_:
            logging.error(str(err_))

    return dframe


def get_stations(dc, listCha, origTime, lat, lon, dist):
    """
        Returns the list of stations from a specified arguments:
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
        :param dist: the radius distance from lat and lon, in km
        :type dist: float
        :return: the list of stations
    """

    listResult = list()
    try:
        # start, endt = get_time_range(origTime, timedelta(days=1))
        start, endt = get_time_range(origTime, days=1)
    except TypeError:
        logging.error('Cannot convert origTime parameter (%s).', origTime)
        return listResult

    stationQuery = ('%s/station/1/query?latitude=%3.3f&longitude=%3.3f&'
                    'maxradius=%3.3f&start=%s&end=%s&channel=%s&format=text&level=station')

    aux = stationQuery % (dc, lat, lon, dist, start.isoformat(),
                          endt.isoformat(), ','.join(listCha))

    dcResult = url_read(aux, decoding='utf8')

    return station_to_dframe(dcResult)


def station_to_dframe(stations_query_result):
    if not stations_query_result:
        return pd.DataFrame()

    stations = stations_query_result.splitlines()
    dframe = pd.DataFrame(index=np.arange(0, len(stations)-1),
                          columns=(e.strip() for e in stations[0].split("|")))

    for i, st in enumerate(stations[1:]):
        splSt = st.split('|')
        splSt[6] = dtime(splSt[6], on_err_return_none=True)
        if splSt[6] is None:
            logging.error("Could not convert start time attribute (%s).", splSt[6])
            continue

        # parse end time, it can be None ()
        splSt[7] = dtime(splSt[7], on_err_return_none=True) or ''

        splSt[2] = float(splSt[2])
        splSt[3] = float(splSt[3])
        splSt[4] = float(splSt[4])

        # http://stackoverflow.com/questions/10715965/add-one-row-in-a-pandas-dataframe
        # loc or iloc both work here since the index is natural numbers
        dframe.loc[i] = splSt

    return dframe


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
    """
    # added info for the tau-p
    travel_time = get_travel_time(ev_depth_km, distance_in_degrees)
    if travel_time is None:
        return None
    arrivalTime = ev_time + timedelta(seconds=float(travel_time))
    # shall we avoid numpy? before was: timedelta(seconds=numpy.float64(travel_time))

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
        :return: a unique integer denoting the given wav. Two wavs with the same argument have the
        same id
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
            arrivalTime = get_arrival_time(dista, ev_depth_km, ev_time)
            if arrivalTime is None:
                logging.info('arrival time is None, skipping')
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
                            wav_query, buffer('')]
                data_row.extend(st)
                data_row[0] = get_wav_id(ev_id, st_network, st_name, st_location, cha,
                                         start_time, end_time)
                data.eppend(data_row)

    colz = ['Id', '#EventID_fk', 'Channel', 'Location', 'StartTime', 'EndTime', 'DataCenter',
            'Distance/deg', 'ArrivalTime', 'QueryStr', 'Data']
    colz.extend(['Station_' + s for s in stations_dataframe.columns])
    wav_dframe = pd.DataFrame(columns=colz) if not data else pd.DataFrame(columns=colz, data=data)
    return wav_dframe


def read_wav(row, ert=None):
    if ert is not None:
        remtime_last = ert.get(False)
        remtime = ert.get()
        msg = " (%s)" % row['QueryStr']
        if remtime != remtime_last:
            etr_msg = ('%s done. '
                       'Est. remaining time: '
                       '%s' + msg) % (ert.percent(), remtime)
            logging.info(etr_msg)
    return url_read(row['QueryStr'])


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
        events = get_events(**args)
    except (IOError, ValueError, TypeError) as err:
        logging.error(str(err))
        return

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

        distFromEvent = get_search_radius(ev_mag,
                                          search_radius_args[0],
                                          search_radius_args[1],
                                          search_radius_args[2],
                                          search_radius_args[3])

        logging.info(('Querying Station WS (selected data-centers and channels)'
                      ' for stations within %s degrees:') % distFromEvent)

        for DCID, dc in datacenters_dict.iteritems():
            for chName, chList in channelList.iteritems():
                try:
                    stations = get_stations(dc, chList, ev_time, ev_lat, ev_lon, distFromEvent)
                except (IOError, ValueError, TypeError) as exc:
                    logging.warning(str(exc))
                    continue

                if len(stations):
                    logging.info('%d stations found (data center: %s, channel: %s)', len(stations),
                                 str(DCID), str(chList))
                logging.debug('Stations: %s', stations)

                wdf = get_wav_dframe(stations, dc, chList, ev_id, ev_lat, ev_lon, ev_depth_km,
                                     ev_time, ptimespan)

                # skip when the dataframe is empty. Moreover, this apparently avoids shuffling
                # column order
                if not wdf.empty:
                    wav_dframe = wdf if wav_dframe is None else wav_dframe.append(wdf,
                                                                                  ignore_index=True)

    written = 0
    if wav_dframe is not None:

        data_table_name = "data"
        data_pkey = "Id"
        data_dtype = {'Data': BLOB}

        # append reorders the columns, so set them as we wanted
        # Note that wdf is surely defined
        # Note also that now column order is not anymore messed up, but do this for safety:
        wav_dframe = wav_dframe[wdf.columns]

        wav_data = db_handler.purge(wav_dframe, data_table_name, data_pkey)

        logging.info("")
        logging.info(("Querying Datacenter WS: downloading and saving %d of %d waveforms"
                      "(%d already saved)") %
                     (len(wav_data), len(wav_dframe), len(wav_dframe) - len(wav_data)))

        ert = EstRemTimer(len(wav_data))

        def readmseed(row, ert):
            try:
                return read_wav(row['QueryStr'], ert)
            except (IOError, ValueError, TypeError) as exc:
                logging.warning(str(exc))
                return None

        wav_data.loc[:, "Data"] = wav_data.apply(readmseed, axis=1, ert=ert)
        wav_data = wav_data[~wav_data['Data'] is None]
        db_handler.write(wav_data, data_table_name, data_pkey, dtype=data_dtype)
        written = len(wav_data)

    logging.info("")
    logging.info("DONE: %d waveforms (mseed files) written to '%s'", written, outpath)


# Event WS
# Station WS
# Dataselect WS