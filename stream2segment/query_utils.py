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
import os
import calendar
import logging
import time
from matplotlib.dates import date2num
# from datetime import datetime
from datetime import timedelta
from stream2segment.utils import to_datetime, estremttime, EstRemTimer
from stream2segment import io as s2sio
import numpy as np
import pandas as pd
# Python 3 compatibility
# try:
#     import urllib.request as ul
# except ImportError:
#     import urllib2 as ul

import urllib2 as ul

# third party imports:
# from obspy.taup.taup import getTravelTimes
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees
from obspy.taup.helper_classes import TauModelError


def getTravelTime(source_depth_in_km, distance_in_degree, model='ak135'):  # FIXME: better!
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


def getSearchRadius(mag, mmin=3, mmax=7, dmin=1, dmax=5):  # FIXME: better!
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


def getEvents(**kwargs):
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

    result = url_read(eventQuery, 'Event WS')

    return evt_to_dframe(result)
#     listResult = list()
#     for ev in result.splitlines()[1:]:
#         splEv = ev.split('|')
#         splEv[1] = to_datetime(splEv[1])
#         if splEv[1] is None:
#             logging.error("Couldn't convert origTime parameter (%s).", splEv[1])
#             continue
# 
#         try:
#             splEv[2] = float(splEv[2])
#             splEv[3] = float(splEv[3])
#             splEv[4] = float(splEv[4])
#             splEv[10] = float(splEv[10])
# 
#             listResult.append(splEv)
#         except (ValueError, IndexError) as err_:
#             logging.error(str(err_))
# 
#     return listResult

def evt_to_dframe(event_query_result):
    if not event_query_result:
        return pd.DataFrame()

    events = event_query_result.splitlines()
    dframe = pd.DataFrame(index=np.arange(0, len(events)-1),
                          columns=(e.strip() for e in events[0].split("|")))
    for i, ev in enumerate(events[1:]):
        evt_list = ev.split('|')
        evt_list[1] = to_datetime(evt_list[1])
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


def getStations(dc, listCha, origTime, lat, lon, dist):
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
        # start, endt = getTimeRange(origTime, timedelta(days=1))
        start, endt = getTimeRange(origTime, days=1)
    except TypeError:
        logging.error('Cannot convert origTime parameter (%s).', origTime)
        return listResult

    stationQuery = ('%s/station/1/query?latitude=%3.3f&longitude=%3.3f&'
                    'maxradius=%3.3f&start=%s&end=%s&channel=%s&format=text&level=station')

    aux = stationQuery % (dc, lat, lon, dist, start.isoformat(),
                          endt.isoformat(), ','.join(listCha))

    dcResult = url_read(aux, 'Station WS')

    return station_to_dframe(dcResult)
#     for st in dcResult.splitlines()[1:]:
#         splSt = st.split('|')
#         splSt[6] = to_datetime(splSt[6])
#         if splSt[6] is None:
#             logging.error("Couldn't convert start time attribute (%s).", splSt[6])
#             continue
# 
#         # FIXME: why shouldn't this log any error?
#         splSt[7] = to_datetime(splSt[7])
# 
#         splSt[2] = float(splSt[2])
#         splSt[3] = float(splSt[3])
#         splSt[4] = float(splSt[4])
# 
#         listResult.append(splSt)
# 
#     return listResult


def station_to_dframe(stations_query_result):
    if not stations_query_result:
        return pd.DataFrame()

    stations = stations_query_result.splitlines()
    dframe = pd.DataFrame(index=np.arange(0, len(stations)-1),
                          columns=(e.strip() for e in stations[0].split("|")))

    for i, st in enumerate(stations[1:]):
        splSt = st.split('|')
        splSt[6] = to_datetime(splSt[6])  # parse start time
        if splSt[6] is None:
            logging.error("Could not convert start time attribute (%s).", splSt[6])
            continue

        # parse end time, it can be None ()
        splSt[7] = to_datetime(splSt[7]) or ''

        splSt[2] = float(splSt[2])
        splSt[3] = float(splSt[3])
        splSt[4] = float(splSt[4])

        # http://stackoverflow.com/questions/10715965/add-one-row-in-a-pandas-dataframe
        # loc or iloc both work here since the index is natural numbers
        dframe.loc[i] = splSt

    return dframe


def getWaveforms(dc, st, listCha, origTime, minBeforeP, minAfterP):
    """
        Returns the tuple w,c where w is the waveform from the given parameters, and c is the
        relative channel
        :param dc: the datacenter to query from
        :type dc: string
        :param st: the station to query from
        :type st: string
        :param listCha: the list of channels, e.g. ['HL?', 'SL?', 'BL?']. The function iterates
            over the given channels and returns the first available data
        :type listCha: iterable (e.g., list)
        :param origTime: the query time. The request will be built with a time start and end of
            +-minBeforeP (see below) minutes from origTime
        :type origTime: date or datetime
        :param minBeforeP: the minutes before P wave origTime
        :type minBeforeP: float
        :param minAfterP: the minutes after P wave origTime
        :type minAfterP: float
        :return: the tuple data, channel (bytes and string)
    """
    dsQuery = '%s/dataselect/1/query?station=%s&channel=%s&start=%s&end=%s'

    try:
        start, endt = getTimeRange(origTime, minutes=(minBeforeP, minAfterP))
    except TypeError:
        logging.error('Cannot convert origTime parameter (%s).', origTime)
        return '', ''

    for cha in listCha:
        aux = dsQuery % (dc, st, cha, start.isoformat(), endt.isoformat())
        dcResult = url_read(aux, 'Dataselect WS')

        if dcResult:
            return cha.replace('*', 'X').replace('?', 'X'), dcResult

    return '', ''


def get_wav_dframe_FIXME(dc, st, listCha, arrivalTime, minBeforeP, minAfterP, dframe=None):
    """
        Returns the tuple w,c where w is the waveform from the given parameters, and c is the
        relative channel
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
    if dframe is None:
        dframe = pd.DataFrame(  # index=np.array(xrange(len(listCha))),
                              # Note above: np array or list, the former has more flexibility
                              # if we want to modify it later
                              columns=('dataCenter', 'channel', 'arrivalTime',
                                       'startTime', 'endTime', 'queryStr'))

    dsQuery = '%s/dataselect/1/query?station=%s&channel=%s&start=%s&end=%s'

    try:
        start, endt = getTimeRange(arrivalTime, minutes=(minBeforeP, minAfterP))
    except TypeError:
        logging.error('Cannot convert arrivalTime parameter (%s).', arrivalTime)
        return dframe

    for cha in listCha:
        query_str = dsQuery % (dc, st, cha, start.isoformat(), endt.isoformat())
        datalist = [dc, cha, arrivalTime, start.isoformat(), endt.isoformat(), query_str]
        datadict = {dframe.columns[i]: datalist[i] for i in xrange(len(datalist))}
        dframe = dframe.append(datadict, ignore_index=True)

    return dframe


def get_wav_queries_FIXME(dc, st, listCha, start_time, end_time):
    """
        Returns the tuple w,c where w is the waveform from the given parameters, and c is the
        relative channel
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

    qry = '%s/dataselect/1/query?station=%s&channel=%s&start=%s&end=%s'
    return [qry % (dc, st, cha, start_time.isoformat(), end_time.isoformat()) for cha in listCha]
# 
#     for cha in listCha:
#         query_str = dsQuery % (dc, st, cha, start_time.isoformat(), end_time.isoformat())
#         yield query_str


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
    travel_time = getTravelTime(ev_depth_km, distance_in_degrees)
    if travel_time is None:
        return None
    arrivalTime = ev_time + timedelta(seconds=float(travel_time))
    # shall we avoid numpy? before was: timedelta(seconds=numpy.float64(travel_time))

    return arrivalTime


def getTimeRange(origTime, days=0, hours=0, minutes=0, seconds=0):
    """
        Returns the tuple (origTime - timeDeltaBefore, origTime + timeDeltaAfter), where the deltas
        are built according to the given parameters. Any of the parameters can be an int
        OR an iterable (list, tuple) of two elements specifying the days before and after,
        respectively

            :Example:
            getTimeRange(t, seconds=(1,2)) returns (t - 1second, t + 2seconds)
            getTimeRange(t, minutes=4) returns (t - 4minutes, t + 4minutes)
            getTimeRange(t, days=1, seconds=(1,2)) returns (t - 1dayAnd1second, t + 1dayAnd2seconds)

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


def url_read(url, name, blockSize=1024*1024):
    """
        Reads and return data from the given url. Note that in case of IOException, the  data
        read until the exception is returned
        :param url: a valid url
        :type url: string
        :param name: the name of the request (will be displayed in errors, as of january 2016
            redirected to log)
        :type name: string
        :param blockSize: the block size while reading, defaults to 1024 ** 2
            (at most in chunks of 1 MB)
        :type blockSize: integer
        :return the data read, or empty string if None
        :rtype bytes of data
    """
    dcBytes = 0
    dcResult = ''
    logging.debug('Reading url: %s', url)
    req = ul.Request(url)
    urlopen_ = None

    try:
        urlopen_ = ul.urlopen(req)
    except (IOError, OSError) as e:
        # note: urllib2 raises urllib2.URLError (subclass of IOError),
        # in python3 raises urllib.errorURLError (subclass of OSError)
        # in both cases there might be a reason or code attributes, which we
        # want to print
        str_ = ''
        if hasattr(e, 'reason'):
            str_ = '%s (Reason: %s)' % (e.__class__.__name__, e.reason)  # pylint:disable=E1103
        elif hasattr(e, 'code'):
            str_ = '%s (The server couldn\'t fulfill the request. Error code: %s)' % \
                    (e.__class__.__name__, e.code)  # pylint:disable=E1103
        else:
            str_ = '%s (%s)' % (e.__class__.__name__, str(e))

        logging.error('%s - %s', url, str_)

    except (TypeError, ValueError) as e:
        # see https://docs.python.org/2/howto/urllib2.html#handling-exceptions
        logging.error('%s', e)

    if urlopen_ is None:
        return dcResult

    # Read the data in blocks of predefined size
    while True:
        try:
            buf = urlopen_.read(blockSize)
        except IOError:  # urlopen behaves as a file-like obj.
            # Thus we catch the file-like exception IOError,
            # see https://docs.python.org/2.4/lib/bltin-file-objects.html
            logging.error('Error while querying the %s', name)
            buf = ''  # for safety (break the loop here below)

        if not buf:
            break
        dcBytes += len(buf)
        dcResult += buf

    # Close the connection to avoid overloading the server
    urlopen_.close()

    # logging.debug('%s bytes read from %s', dcBytes, url)
    return dcResult


def timestamp(utc_dt):
    """Returns matplotlib.dates.date2num(utc-dt), where the argument is
    an utc datetime"""
#     return calendar.timegm(utc_dt.utctimetuple())
    return date2num(utc_dt)


def get_wav_dframe(stations_dataframe, dc,
                   chList, ev_id, ev_lat, ev_lon, ev_depth_km, ev_time, ptimespan, column='id'):
    data = []
    if not stations_dataframe.empty:
        # first of all get the network and station index for building the data id later
        # Do it once here. Use index defined in python lists cause googling seems not easy to get
        # a relative method (if exists) on the dataframe columns
        stations_col_list = list(stations_dataframe.columns)
        network_index = stations_col_list.index(u'#Network')
        station_index = stations_col_list.index(u'Station')

        qry = '%s/dataselect/1/query?station=%s&channel=%s&start=%s&end=%s'

        for st in stations_dataframe.values:
            st_name = st[1]
            st_lat = st[2]
            st_lon = st[3]

            # FIXME: move this message below
            # logging.info('Querying data-center (station=%s) for data', st_name)

            dista = locations2degrees(ev_lat, ev_lon, st_lat, st_lon)
            arrivalTime = get_arrival_time(dista, ev_depth_km, ev_time)
            if arrivalTime is None:
                logging.info('arrival time is None, skipping')
                continue

            try:
                start_time, end_time = getTimeRange(arrivalTime,
                                                    minutes=(ptimespan[0], ptimespan[1]))
            except TypeError:
                logging.error('Cannot convert arrivalTime parameter (%s).', arrivalTime)
                continue

            # wav_queries = get_wav_queries(dc, st_name, chList, start_time, end_time)

            # now build the data frames
            for cha in chList:
                wav_query = qry % (dc, st_name, cha, start_time.isoformat(), end_time.isoformat())
                data_ = [None, ev_id, cha, "", start_time, end_time, dc,  dista, arrivalTime,
                         wav_query, buffer('')]
                data_.extend(st)
                # make id. FIXME: very very prone to errors!!!!
                data_[0] = "|". join([ev_id, st[network_index], st[station_index], "",
                                     cha, start_time.isoformat(), end_time.isoformat()])
                try:
                    data_ = [d.decode('utf8') if isinstance(d, basestring) else d for d in data_]
                    data.append(data_)
                    # FIXME: raise logging error!
                except Exception:
                    g = 9

                # wav_dframe = wav_dframe.append([data])

    colz = ['Id', '#EventID_fk', 'Channel', 'Location',
            'StartTime',
            'EndTime', 'DataCenter',
            'Distance/deg', 'ArrivalTime', 'QueryStr', 'Data']
    colz.extend(['Station_' + s for s in stations_dataframe.columns])
    wav_dframe = pd.DataFrame(columns=colz) if not data else pd.DataFrame(columns=colz, data=data)
    return wav_dframe


def saveWaveforms(eventws, minmag, minlat, maxlat, minlon, maxlon, search_radius_args,
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
    # check path where to store stuff:
    if not os.path.exists(outpath):
        logging.error('"%s" does not exist', outpath)
        return

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
    logging.info("Querying events:")
    events = getEvents(**args)
    logging.info('%s events found', len(events))
    logging.debug('Events: %s', events)

    db_handler = s2sio.DbHandler()
    new_events = db_handler.purge(events, "events", '#EventID')
    db_handler.write(new_events, 'events', '#EventID')

    wav_dframe = None
    ert = EstRemTimer(len(events))

    for ev in events.values:  # itertuples(index=False, name=None):
        ev_mag = ev[10]
        ev_id = ev[0]
        ev_loc_name = ev[12]
        ev_time = ev[1]
        ev_lat = ev[2]
        ev_lon = ev[3]
        ev_depth_km = ev[4]
        remtime = ert.get()
        evtMsg = 'Processing event %s %s (mag=%s)' % (
                                                      ev_id,
                                                      ev_loc_name,
                                                      ev_mag,
                                                      )
        etr_msg = '%d%% Done. Est. remaining time: %s' % (
                                                          round(100*float(ert.done)/ert.total),
                                                          "unknown" if remtime is None else str(remtime),
                                                          )
        # dec_ = ('=' * max(3, len(evtMsg) - len(etr_msg) -3))
        # strh = "[" + etr_msg + "] "
        # logging.info("")
        # logging.info(strh)
        strmsg = "[" + etr_msg + "] " + evtMsg
        logging.info("")  # "-" * len(strmsg))
        logging.info(strmsg)

        # FIXME: this does not need anymore to be a parameter right?!!
        # use ev[10], that is the magnitude, to determine the radius distFromEvent
        distFromEvent = getSearchRadius(ev_mag,
                                        search_radius_args[0],
                                        search_radius_args[1],
                                        search_radius_args[2],
                                        search_radius_args[3])

#         stMsg = ('Querying '
#                  'data-centers %s (channels %s)') % (str(datacenters_dict.keys()),
#                                                      str(channelList.keys()))
#         logging.info(stMsg)
        logging.info(('Querying selected data-centers and channels'
                      ' for stations within %s degrees:') % distFromEvent)
        for DCID, dc in datacenters_dict.iteritems():
            # logging.info('Querying %s', str(DCID))
            for chName, chList in channelList.iteritems():
                # try with all channels in channelList
                # stMsg = 'Querying data-center %s (channels=%s) for stations' % (str(DCID),
                #                                                                str(chList))

                # logging.info('-' * len(stMsg))
                # logging.info(stMsg)

                stations = getStations(dc, chList, ev_time, ev_lat, ev_lon, distFromEvent)

                if len(stations):
                    logging.info('%d stations found (data center: %s, channel: %s)', len(stations),
                                 str(DCID), str(chList))
                logging.debug('Stations: %s', stations)

                wdf = get_wav_dframe(stations, dc, chList, ev_id, ev_lat, ev_lon, ev_depth_km,
                                     ev_time, ptimespan)

                # indices are not increasing automatically, so when iterating over iloc for
                # instance we night have problem. We can thus pass ignore_index=True in the append
                # function but this shuffles the columns order. Thus we will 
                # skip the ignore_index argument here and call reset_index later
                wav_dframe = wdf if wav_dframe is None else wav_dframe.append(wdf)
                                                                              #,ignore_index=True)

#                 for st in stations.values:
#                     st_name = st[1]
#                     st_lat = st[2]
#                     st_lon = st[3]
# 
#                     # FIXME: move this message below
#                     logging.info('Querying data-center %s (station=%s) for data',
#                                  str(DCID), st_name)
# 
#                     dista = locations2degrees(ev_lat, ev_lon, st_lat, st_lon)
#                     arrivalTime = get_arrival_time(dista, 
#                                                    ev_depth_km,
#                                                    ev_time)
#                     if arrivalTime is None:
#                         logging.info('arrival time is None, skipping')
#                         continue
# 
#                     try:
#                         start_time, end_time = getTimeRange(arrivalTime,
#                                                             minutes=(ptimespan[0], ptimespan[1]))
#                     except TypeError:
#                         logging.error('Cannot convert arrivalTime parameter (%s).', arrivalTime)
#                         continue
# 
#                     wav_queries = get_wav_queries(dc, st_name, chList, start_time, end_time)
# 
#                     # now build the data frames
#                     if wav_dframe is None:
#                         colz = ['fk_' + events.columns[0], 'Channel', 'Location',
#                                 'StartTime',
#                                 'EndTime', 'DataCenter',
#                                 'Distance/deg', 'ArrivalTime/sec', 'QueryStr']
#                         colz.extend(['Station_' + s for s in stations.columns])
#                         wav_dframe = pd.DataFrame(columns=colz)
#
#                     data = []
#                     for wqu, cha in zip(wav_queries, chList):
#                         data_ = [ev_id, cha, "", start_time, end_time, dc,  dista, arrivalTime, wqu]
#                         data_.extend(st)
#                         data.append(data_)
#                         # wav_dframe = wav_dframe.append([data])
#
#                     wav_dframe = wav_dframe.append(pd.DataFrame(data=data, columns=wav_dframe.columns))

                    # added info for the tau-p
#                     dista = locations2degrees(ev_lat, ev_lon, st_lat, st_lon)
#                     travel_time = getTravelTime(ev_depth_km, dista)
#                     if travel_time is None:
#                         continue
#                     arrivalTime = ev_time + timedelta(seconds=float(travel_time))
                    # shall we avoid numpy? before was: timedelta(seconds=numpy.float64(travel_time))

                    # HERE WE ARE!!!!

#                     wav_dframe = get_wav_dframe(dc,
#                                                 st_name,
#                                                 chList,
#                                                 arrivalTime,
#                                                 ptimespan[0],
#                                                 ptimespan[1],
#                                                 wav_dframe)

#                cha, wav = "", []
#                     cha, wav = getWaveforms(dc,
#                                             st_name,
#                                             chList,
#                                             arrivalTime,
#                                             ptimespan[0],
#                                             ptimespan[1])

                    # FIXME Here ev[1] must be replaced for the tau-p
                    # cha, wav = getWaveforms(dc, st[1], chList, ev[1])

                    # logging.info('%s, channel %s: %s', st[1], origCha, 'Data found' if len(wav) else "No data found")
#                 if len(wav):
#                     logging.info('Data found on channel %s', cha)
#                 else:
#                     logging.info("No data found")
# 
#                 # logging.debug('stations: %s', stations)
#                 if len(wav):
#                     complete_path = os.path.join(outpath,
#                                                  'ev-%s-%s-%s-origtime_%s.mseed' %
#                                                  (ev_id, st_name, cha, str(timestamp(arrivalTime)))
#                                                  )
#                     logging.debug('Writing wav to %s', complete_path)
#                     fout = open(complete_path, 'wb')
#                     fout.write(wav)
#                     fout.close()
#                     written_files += 1



#     wav_dframe.reset_index(drop=True, inplace=True)  # drop: remove old index (otherwise we will have
#     # a new index column
#     wav_data = dbh.purge(wav_dframe, 'data', 'Id')

    # WRONG: calls url_read with a series object
    # wav_data["data"] = url_read(wav_data['QueryStr'], 'Dataselect WS')

#     def readurl(row):
#         try:
#             row.set_value('data', buffer(url_read(row['QueryStr'], 'Dataselect WS')))
#         except Exception as exc:
#             g = 9
# 
# #     wav_data["data"] = wav_data.apply(readurl,
# #                                       axis=1)

#     def f(row):
#         return buffer(url_read(row['QueryStr'], 'Dataselect WS'))
#         # return "c{}n{}".format(row["condition"], row["no"])
# 
#     wav_data["Data"] = wav_data.apply(f, axis=1)

    # wav_data["data"] = wav_data.apply(lambda row: row.set_value('data', buffer(url_read(row['QueryStr'], 'Dataselect WS'))))

#     from sqlite3 import Binary
#     for i in xrange(len(wav_data)):
#         data = buffer(url_read(wav_data.iloc[i]['QueryStr'], 'Dataselect WS'))
#         print str(len(data)) + " " +wav_data.iloc[i]['QueryStr']
#         wav_data = wav_data.set_value(i, 'Data', data)
        
    
#     def func(arg):
#         arg['data'] = url_read(arg['QueryStr'], 'Dataselect WS')
#         j = 9
#         
#     wav_data.apply(func, axis=1)
#     for i in len(wav_data):
#         querystr = wav_data.iloc[i]['QueryStr']
#         dcResult = url_read(querystr, 'Dataselect WS')
#         wav_data.iloc[i]['mseed'] = bytes(dcResult)

#     k = 9
#     dbh.write(wav_data, 'data', 'id')
    written = 0
    if wav_dframe is not None:
        wav_dframe.reset_index(drop=True, inplace=True)  # drop: remove old index (otherwise we will have
        # a new index column

        # wav_dframe = wav_dframe[1:2]
        wav_data = db_handler.purge(wav_dframe, 'data', 'Id')

#         def f(row):
#             return url_read(row['QueryStr'], 'Dataselect WS')
            # return "c{}n{}".format(row["condition"], row["no"])

        wav_data.loc[:, "Data"] = \
            wav_data.apply(lambda row: url_read(row['QueryStr'], 'Dataselect WS'), axis=1)
        db_handler.write(wav_data, 'data', 'Id')
        written = len(wav_data)
    logging.info("DONE: %d waveforms (mseed files) written to '%s'", written, outpath)
