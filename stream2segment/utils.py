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
import os
import logging
from datetime import datetime
from datetime import timedelta
from obspy.taup.taup import getTravelTimes
from obspy.geodetics.base import locations2degrees
# FIXME: this should be the import! but pydev trhows unwanted errors:
# from obspy.core.util import locations2degrees

# Python 3 compatibility
try:
    import urllib.request as ul
except ImportError:
    import urllib2 as ul


def getArrivalTime(dist, depth, model='ak135'):
    """
        Assess and return the arrival time of P phases.
        Uses obspy.getTravelTimes
        :param dist: Distance in degrees.
        :type dist: float
        :param depth: Depth in kilometer.
        :type depth: float
        :param model: Either ``'iasp91'`` or ``'ak135'`` velocity model.
         Defaults to 'ak135'.
        :type model: str, optional
    """
    tt = getTravelTimes(delta=dist, depth=depth, model=model)
    return min((ele['time'] for ele in tt if ele.get('phase_name') in ['Pg', 'Pn', 'Pb']))
    # ttsel=[ele for ele in tt if ele.get('phase_name') in ['Pg','Pn','Pb']]
    # ttime=[ele['time'] for ele in ttsel]
    # arrtime=min(ttime)
    # return arrtime


def getSearchRadius(mag, mmin=3, mmax=7, dmin=1, dmax=5):
    """From a given magnitude, determines and returns tha max radius (in degrees).
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


def to_datetime(date_str):
    """
        Converts a date in string format (as returned by a fdnsws query) into
        a datetime python object
        Example:
        to_datetime("2016-06-01T09:04:00.5600Z")
    """
    try:
        date_str = date_str.replace('-', ' ').replace('T', ' ')\
            .replace(':', ' ').replace('.', ' ').replace('Z', '').split()
        return datetime(*(int(value) for value in date_str))
    except (AttributeError, IndexError, ValueError, TypeError):
        # logging.error("Couldn't convert start time attribute (%s).", splSt[6])
        return None


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
    eventQuery = '%(eventws)squery?minmagnitude=%(minmag)1.1f&start=%(start)s' \
        + '&minlon=%(minlon)s&maxlon=%(maxlon)s&end=%(end)s' \
        + '&minlat=%(minlat)s&maxlat=%(maxlat)s&format=text' % kwargs

    result = url_read(eventQuery, 'Event WS')

    listResult = list()
    for ev in result.splitlines()[1:]:
        splEv = ev.split('|')
        splEv[1] = to_datetime(splEv[1])
        if splEv[1] is None:
            logging.error("Couldn't convert origTime parameter (%s).", splEv[1])
            continue

#         try:
#             splEv[1] = splEv[1].replace('-', ' ').replace('T', ' ')
#             splEv[1] = splEv[1].replace(':', ' ').replace('.', ' ')
#             splEv[1] = splEv[1].replace('Z', '').split()
#             splEv[1] = datetime(*map(int, splEv[1]))
#         except (TypeError, ValueError):
#             logging.error("Couldn't convert origTime parameter (%s).", splEv[1])
#             continue

        splEv[2] = float(splEv[2])
        splEv[3] = float(splEv[3])
        splEv[4] = float(splEv[4])
        splEv[10] = float(splEv[10])

        listResult.append(splEv)

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
    stationQuery = '%s/station/1/query?latitude=%3.3f&longitude=%3.3f&' + \
        'maxradius=%3.3f&start=%s&end=%s&channel=%s&format=text&level=station'

    listResult = list()
    try:
        # start, endt = getTimeRange(origTime, timedelta(days=1))
        start, endt = getTimeRange(origTime, days=1)
    except TypeError:
        logging.error('Cannot convert origTime parameter (%s).', origTime)
        return listResult

    aux = stationQuery % (dc, lat, lon, dist, start.isoformat(),
                          endt.isoformat(), ','.join(listCha))

    dcResult = url_read(aux, 'Station WS')

    for st in dcResult.splitlines()[1:]:
        splSt = st.split('|')
        splSt[6] = to_datetime(splSt[6])
        if splSt[6] is None:
            logging.error("Couldn't convert start time attribute (%s).", splSt[6])
            continue
#         try:
#             splSt[6] = splSt[6].replace('-', ' ').replace('T', ' ')
#             splSt[6] = splSt[6].replace(':', ' ').replace('.', ' ')
#             splSt[6] = splSt[6].replace('Z', '').split()
#             splSt[6] = datetime(*map(int, splSt[6]))
#         except (AttributeError, IndexError):
#             logging.error("Couldn't convert start time attribute (%s).", splSt[6])
#             continue

        splSt[7] = to_datetime(splSt[7])
#         try:
#             splSt[7] = splSt[7].replace('-', ' ').replace('T', ' ')
#             splSt[7] = splSt[7].replace(':', ' ').replace('.', ' ')
#             splSt[7] = splSt[7].replace('Z', '').split()
#             splSt[7] = datetime(*map(int, splSt[7]))
#         except (AttributeError, IndexError):
#             splSt[7] = None

        splSt[2] = float(splSt[2])
        splSt[3] = float(splSt[3])
        splSt[4] = float(splSt[4])

        listResult.append(splSt)

    logging.info('%s stations found', len(listResult))

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
            logging.info('Data found from channel %s', cha)
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
        Reads and return data from the given url.
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
    logging.debug('Querying %s', url)
    req = ul.Request(url)

    try:
        u = ul.urlopen(req)

        # Read the data in blocks of predefined size
        try:
            buf = u.read(blockSize)
        except:
            logging.error('Error while querying the %s', name)

        if not len(buf):
            logging.debug('Error code: %s', u.getcode())

        while len(buf):
            dcBytes += len(buf)
            # Return one block of data
            dcResult += buf
            try:
                buf = u.read(blockSize)
            except:
                logging.error('Error while querying the %s', name)
            logging.debug('%s bytes from %s', dcBytes, url)

        # Close the connection to avoid overloading the server
        logging.debug('%s bytes from %s', dcBytes, url)
        u.close()

    except ul.URLError as e:
        if hasattr(e, 'reason'):
            logging.error('%s - Reason: %s', url, e.reason)
        elif hasattr(e, 'code'):
            logging.error('The server couldn\'t fulfill the request')
            logging.error('Error code: %s', e.code)  # pylint:disable=E1103

    except (TypeError, ValueError) as e:
        # see https://docs.python.org/2/howto/urllib2.html#handling-exceptions
        logging.error('%s', e)

    return dcResult


def saveWaveforms(eventws, minmag, minlat, maxlat, minlon, maxlon, distFromEvent, datacenters_dict,
                  channelList, start, end, minBeforeP, minAfterP, outpath):
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
        :param distFromEvent: the distance, in km, from an event E.g. 5.0
        :type distFromEvent: float
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
        :param minBeforeP: the minutes before P wave arrival for the waveform query time span
        :type minBeforeP: float
        :param minAfterP: the minutes after P wave arrival for the waveform query time span
        :type minAfterP: float
        :param outpath: path where to store mseed files E.g. '/tmp/mseeds'
        :type outpath: string
    """
    # check path where to store stuff:
    if not os.path.exists(outpath):
        logging.error('"%s" does not exist', outpath)
        return
    # a little bit hacky, but convert to dict as the function gets dictionaries
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
    logging.debug('%s', events)

    for ev in events:
        logging.info('Processing event %s (%s) %s', ev[10], ev[0], ev[12])

        for DCID, dc in datacenters_dict.iteritems():
            logging.info('Querying %s', str(DCID))
            for chList in channelList:
                # try with all channels in channelList
                logging.info('(Querying %s channels)', str(chList))

                stations = getStations(dc, chList, ev[1], ev[2], ev[3],
                                       distFromEvent)

                logging.debug('stations: %s', stations)

                for st in stations:
                    logging.info('Processing station %s', st[1])

                    # added info for the tau-p
                    dista = locations2degrees(ev[2], ev[3], st[2], st[3])
                    arrtime = getArrivalTime(dista, ev[4])
                    origTime = ev[1] + timedelta(seconds=float(arrtime))
                    # shall we avoid numpy? before was: timedelta(seconds=numpy.float64(arrtime))
                    cha, wav = getWaveforms(dc, st[1], chList, origTime, minBeforeP, minAfterP)

                    # FIXME Here ev[1] must be replaced for the tau-p
                    # cha, wav = getWaveforms(dc, st[1], chList, ev[1])

                    if len(wav):
                        complete_path = os.path.join(outpath,
                                                     'ev-%s-%s-%s.mseed' % (ev[0], st[1], cha))
                        logging.debug('Writing to %s', complete_path)
                        fout = open(complete_path, 'wb')
                        fout.write(wav)
                        fout.close()
