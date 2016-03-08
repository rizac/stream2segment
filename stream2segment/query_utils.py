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
from stream2segment.utils import to_datetime, estremttime

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


def getArrivalTime(source_depth_in_km, distance_in_degree, model='ak135'):  # FIXME: better!
    """
        Assess and return the arrival time of P phases.
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

    listResult = list()
    for ev in result.splitlines()[1:]:
        splEv = ev.split('|')
        splEv[1] = to_datetime(splEv[1])
        if splEv[1] is None:
            logging.error("Couldn't convert origTime parameter (%s).", splEv[1])
            continue

        try:
            splEv[2] = float(splEv[2])
            splEv[3] = float(splEv[3])
            splEv[4] = float(splEv[4])
            splEv[10] = float(splEv[10])

            listResult.append(splEv)
        except (ValueError, IndexError) as err_:
            logging.error(str(err_))

    return listResult


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

    for st in dcResult.splitlines()[1:]:
        splSt = st.split('|')
        splSt[6] = to_datetime(splSt[6])
        if splSt[6] is None:
            logging.error("Couldn't convert start time attribute (%s).", splSt[6])
            continue

        # FIXME: why shouldn't this log any error?
        splSt[7] = to_datetime(splSt[7])

        splSt[2] = float(splSt[2])
        splSt[3] = float(splSt[3])
        splSt[4] = float(splSt[4])

        listResult.append(splSt)

    return listResult


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
        logging.info("   %s = %s", str(arg), str(varg))

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
    events = getEvents(**args)

    logging.info('%s events found', len(events))
    logging.debug('Events: %s', events)

    evtCounter = 0
    est_rt = 'unknown'
    written_files = 0
    start_time = time.time()
    for ev in events:
        evtCounter += 1
        evtMsg = 'Processing event %s %s %s (%d of %d, est.rem.time %s)' % (ev[10],
                                                                            ev[0],
                                                                            ev[12],
                                                                            evtCounter,
                                                                            len(events),
                                                                            est_rt,
                                                                            )
        strh = '=' * len(evtMsg)
        logging.info(strh)
        logging.info(evtMsg)

        # FIXME: this does not need anymore to be a parameter right?!!
        # use ev[10], that is the magnitude, to determine the radius distFromEvent
        distFromEvent = getSearchRadius(ev[10],
                                        search_radius_args[0],
                                        search_radius_args[1],
                                        search_radius_args[2],
                                        search_radius_args[3])

        for DCID, dc in datacenters_dict.iteritems():
            # logging.info('Querying %s', str(DCID))
            for chName, chList in channelList.iteritems():
                # try with all channels in channelList
                stMsg = 'Querying data-center %s (channels=%s) for stations' % (str(DCID), str(chList))

                logging.info('-' * len(stMsg))
                logging.info(stMsg)

                stations = getStations(dc, chList, ev[1], ev[2], ev[3], distFromEvent)

                logging.info('%d stations found', len(stations))
                logging.debug('Stations: %s', stations)

                for st in stations:
                    logging.info('Querying data-center %s (station=%s) for data',
                                 str(DCID), st[1])

                    # added info for the tau-p
                    dista = locations2degrees(ev[2], ev[3], st[2], st[3])
                    arrtime = getArrivalTime(ev[4], dista)
                    if arrtime is None:
                        continue
                    origTime = ev[1] + timedelta(seconds=float(arrtime))
                    # shall we avoid numpy? before was: timedelta(seconds=numpy.float64(arrtime))
                    cha, wav = getWaveforms(dc,
                                            st[1],
                                            chList,
                                            origTime,
                                            ptimespan[0],
                                            ptimespan[1])

                    # FIXME Here ev[1] must be replaced for the tau-p
                    # cha, wav = getWaveforms(dc, st[1], chList, ev[1])

                    # logging.info('%s, channel %s: %s', st[1], origCha, 'Data found' if len(wav) else "No data found")
                    if len(wav):
                        logging.info('Data found on channel %s', cha)
                    else:
                        logging.info("No data found")

                    # logging.debug('stations: %s', stations)
                    if len(wav):
                        complete_path = os.path.join(outpath,
                                                     'ev-%s-%s-%s-origtime_%s.mseed' %
                                                     (ev[0], st[1], cha, str(timestamp(origTime)))
                                                     )
                        logging.debug('Writing wav to %s', complete_path)
                        fout = open(complete_path, 'wb')
                        fout.write(wav)
                        fout.close()
                        written_files += 1
        elapsed = time.time() - start_time
        est_rt = str(estremttime(elapsed, evtCounter, len(events)))
    logging.info("DONE: %d waveforms (mseed files) written to '%s'", written_files, outpath)
