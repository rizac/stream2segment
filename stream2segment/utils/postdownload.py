'''
Module documentation to be implemented

:date: Sep 19, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''

# provide some imports to let python3 syntax work also in python 2.7+ effortless. 
# Any of the defaults import below can be safely removed if python2+
# compatibility is not needed

# standard python imports (must be the first import)
from __future__ import absolute_import, division, print_function

# future direct imports (needs future package installed, otherwise remove):
# (http://python-future.org/imports.html#explicit-imports)
from builtins import (ascii, chr, dict, filter, hex, input,
                      int, map, next, oct, open, pow, range, round,
                      super, zip)

from io import BytesIO

from sqlalchemy.orm.session import object_session
from obspy.core.stream import read

from stream2segment.io.utils import loads_inv, dumps_inv
from stream2segment.utils.url import urlread
from stream2segment.utils import urljoin
from stream2segment.io.db.models import Segment
from sqlalchemy.exc import SQLAlchemyError


def get_stream(segment):
    """Returns a Stream object relative to the given segment.
        :param segment: a model ORM instance representing a Segment (waveform data db row)
    """
    data = segment.data
    if not data:
        raise ValueError('no data')
    return read(BytesIO(data))


def get_inventory(station, asbytes=False, **urlread_kwargs):
    """Gets the inventory object for the given station, downloading it and saving it
    if not data is empty/None.
    Raises any utils.url.URLException for any url related errors, ValueError if inventory data
    is empty
    """
    data = station.inventory_xml
    if not data:
        data = download_inventory(station, **urlread_kwargs)
    if not data:
        raise ValueError('Inventory data is empty (0 bytes)')

    return data if asbytes else loads_inv(data)


def download_inventory(station, **urlread_kwargs):
    '''downloads the given inventory. Raises utils.url.URLException on error'''
    query_url = get_inventory_url(station)
    return urlread(query_url, **urlread_kwargs)[0]


def get_inventory_url(station):
    return urljoin(station.datacenter.station_url, station=station.station,
                   network=station.network, level='response')


def save_inventory(session, station, downloaded_bytes_data):
    """Saves the inventory. Raises SqlAlchemyError if station's session is None,
    or (most likely IntegrityError) on failure
    """
    station.inventory_xml = dumps_inv(downloaded_bytes_data)
    session.commit()


class SegmentWrapper(object):

    def __init__(self, session, segment_id, save_station_inventory=True, cache_size=1):
        self.__sess = session
        self.__saveinv = save_station_inventory
        self.__streams = dict()
        self.__invs = dict()
        self.__cachesize = cache_size
        self.reinit(segment_id)

    def reinit(self, segment_id):
        self.__segid = segment_id
        self.__inv = None
        self.__segment = None
        self.__sess.expunge_all()
        self.__sess.close()

    @staticmethod
    def __returnorraise(obj):
        if isinstance(obj, Exception):
            raise obj
        return obj

    @property
    def __getseg(self):
        if self.__segment is None:
            try:
                self.__segment = self.__sess.query(Segment).filter(Segment.id == self.__segid).one()
            except Exception as exc:
                self.__segment = exc
        return self.__returnorraise(self.__segment)

    def stream(self):
        stream = self.__streams.get(self.__segid, None)
        if stream is None:
            try:
                stream = get_stream(self)
            except Exception as exc:
                stream = Exception("MiniSeed (`obspy.Stream` object) error: %s" %
                                   (str(exc) or str(exc.__class__.__name__)))
            self.__setitem(self.__segid, stream, self.__streams, self.__cachesize)
        return self.__returnorraise(stream)

    def inventory(self):
        inv = self.__inv
        if inv is None:  # not accessed inventory sofar, but we might have one in cache
            sta_id = None
            try:
                sta = self.station
                sta_id = sta.id
                inv = self.__invs.get(sta_id, None)
                if inv is None:  # inventory not in cache
                    inv = get_inventory(sta, asbytes=self.__saveinv)
                    # saving the inventory must NOT prevent the continuation.
                    # If we are here we have non-empty data:
                    if self.__saveinv:
                        try:
                            save_inventory(self.__sess, sta, inv)
                        except SQLAlchemyError as exc:
                            pass  # FIXME: how to handle it?
                        inv = loads_inv(inv)  # convert from bytes into obspy object
                else:
                    sta_id = None  # do not insert inv in the cache dict
            except Exception as exc:
                inv = Exception("Station inventory (xml) error: %s" %
                                (str(exc) or str(exc.__class__.__name__)))
            self.__inv = inv
            if sta_id is not None:
                self.__setitem(sta_id, inv, self.__invs, self.__cachesize)

        return self.__returnorraise(inv)

    @staticmethod
    def __setitem(key, val, dic, cachesize):
        if cachesize <= 1:
            dic.clear()
        elif len(dic) and len(dic) >= cachesize:
            del dic[next(iter(dic.keys()))]  # delete first item
        dic[key] = val

    def __getattr__(self, name):
        return getattr(self._SegmentWrapper__getseg, name)

