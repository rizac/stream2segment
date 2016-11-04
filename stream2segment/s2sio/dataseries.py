#!/usr/local/bin/python2.7
# encoding: utf-8
'''
stream2segment.s2sio.dataseries -- sdfsdf

stream2segment.s2sio.dataseries is a description

It defines classes_and_methods

@author:     user_name

@copyright:  2016 organization_name. All rights reserved.

@license:    license

@contact:    user_email
@deffield    updated: Updated
'''

# import sys
# import os
from StringIO import StringIO
from collections import OrderedDict as odict
import gzip
import zipfile
import zlib
import bz2
try:
    import cPickle as pickle
except ImportError:
    import pickle
# import pickle  # uncomment for testing
import numpy as np
from obspy.core import Stream
from obspy.core import Trace
from obspy.core.stream import read as obspy_read


class Object(object):  # pylint: disable=too-few-public-methods
    """Simple python object. the same would have been acquired via the builtin `type(...)`
    function but the latter is not pickable"""

    def __init__(self, **attrs):
        """Sets the attributes of this object
        :param attrs: the attributes
        """
        self.update(**attrs)

    def update(self, **attrs):
        """updates the attributes of this object overriding existing ones, if any
        :param attrs: the attributes
        :return: this object
        """
        for k, j in attrs.iteritems():
            setattr(self, k, j)
        return self


class LightTrace(Object):  # pylint: disable=too-few-public-methods
    """A light obspy trace. A Light trace is a simple python object with the same structure of an
    `obspy.Trace` object (i.e., with the fields `data` - numeric array, and `stats`, an
    object with Trace attributes). The purpose is to allow a lighter container for IO operations
    and to allow different data series to have the same structure of the Trace they originated from
    (e.g. after applying an fft to a given Trace, see subclass :ref: `FreqTrace`)"""
    def __init__(self, data, **stats_attrs):
        """
            Initializes a new light trace.
            :param data: [iterable | array | Stream | Trace] the data. If Stream, it must hold only
            a single trace, and the trace `data` attribute will be used. If Trace, the `trace.data`
            attribute will be used
            :param stats_attrs: custom attributes to be set in the attribute `self.stats` (which is
            a :ref Object type)

            :Example:
            l = LightTrace([1,2,3], 'att1':15, 'att2': 34.5)
            l.data  # returns numpy.array([1,2,3])
            l.stats # returns an object with fields 'att1'=15 and 'att2'=34.5, e.g.:
            l.stats.att1 # returns 15

            l = LightTrace(obspy_trace, 'att1':15, 'att2': 34.5)
            l.data # returns obspy_trace.data
            l.stats.starttime # raises AttributeError
            l.stats.att1 # returns 15
        """
        if isinstance(data, Stream):
            if len(data) != 1:
                raise ValueError("Only single trace streams for light traces")
            data = data.traces[0].data
        elif isinstance(data, Trace):
            data = data.data
        super(LightTrace, self).__init__(data=data)  # basically sets self.data
        self.stats = Object(**stats_attrs)


class FreqTrace(LightTrace):  # pylint: disable=too-few-public-methods
    """An obspy Trace-like object with frequencies on the "x axis".
    It is built from a data array and has two mandatory fields: `startfreq` and `delta`.
    This object extends `Light trace` thus has the same structure of an
    `obspy.Trace` object (i.e., with the fields `data` - numeric array, and `stats`, an
    object with attributes)"""
    def __init__(self, data, startfreq, delta, y_unit=None, **stats_attrs):
        """
            Initializes a new FreqTrace.
            :param data: [iterable | array | Stream | Trace] the data, usually after applying
            some sort of fft algorithm on a trace `data` attribute. This parameter can be also
            Stream and /or Trace for backward (superclass) compatibility, although it should not
            make much sense to pass *time*-series data. However, if Stream, it must hold only
            a single trace, and the trace `data` attribute will be used. If Trace, the `trace.data`
            attribute will be used
            :param stats_attrs: custom attributes to be set in the attribute `self.stats` (which is
            a :ref Object type). 'startfreq' and 'delta' are set by default as stats attributes

            :Example:
            f = FreqTrace([1,2,3], 0, 1.5, 'att1':15, 'att2': 34.5)
            f.data  # returns numpy.array([1,2,3])
            f.stats # returns an object with fields 'att1'=15 and 'att2'=34.5, e.g.:
            f.stats.startfreq  # returns 0
            f.stats.delta  # returns 1.5
            f.stats.att1 # returns 15
        """
        stats_attrs.update(startfreq=startfreq, delta=delta, y_unit=y_unit)
        super(FreqTrace, self).__init__(data, **stats_attrs)


def dumps(obj, pickleprotocol=pickle.HIGHEST_PROTOCOL, compression='gzip',
          compresslevel=9):
    """
        Serializes `obj` (with optional compression) returning a byte sequence to be, e.g.,
        saved to file or db and loaded back by this module level function :ref: `loads`.
        :param obj: A Python object
        :param pickle protocol: the pickle protocol passed to pickle.dumps (default:
        `pickle.HIGHEST_PROTOCOL`, which gives lighter object sizes). Ignored if `compression`
        is 'numpy' (see below). For more information, see:
        https://docs.python.org/2/library/pickle.html#data-stream-format
        :param compression: String, either ['bz2', 'zlib', 'gzip', 'zip' or 'numpy'. Default:
        'gip'], The compression library to use for compressing the data. If None or empty string,
        no compression is applied (thus generally increasing the size of the returned byte string).
        The compression is applied on the byte string returned by 'pickle.dumps` *except* if this
        parameter is 'numpy' (in that case, `pickle.dumps` is not applied at all).
        :param compresslevel: integer (9 by default). Ignored if `compression` is None, empty,
        'numpy' or 'zip', this parameter controls the level of compression; 1 is fastest and
        produces the least compression, and 9 is slowest and produces the most compression
    """
    if compression == 'numpy':
        sio = StringIO()
        np.savez_compressed(sio, obj)
        ret = sio.getvalue()
        sio.close()
        return ret
    pickle_bytes = pickle.dumps(obj, protocol=pickleprotocol)
    if compression == 'bz2':
        return bz2.compress(pickle_bytes, compresslevel=compresslevel)
    elif compression == 'zlib':
        return zlib.compress(pickle_bytes, compresslevel)
    else:
        sio = StringIO()
        if compression == 'gzip':
            with gzip.GzipFile(mode='wb', fileobj=sio, compresslevel=compresslevel) as gzip_obj:
                gzip_obj.write(pickle_bytes)
                # Note: DO NOT return sio.getvalue() WITHIN the with statement, the gzip file obj
                # needs to be closed first. FIXME: ref?
        elif compression == 'zip':
            with zipfile.ZipFile(sio, 'w') as zip_obj:
                zip_obj.writestr(__name__ + " object", pickle_bytes)
        elif compression:
            raise ValueError("compression '%s' not in ('gzip', 'zlib', 'bz2', 'zip', 'numpy')" %
                             str(compression))
        return sio.getvalue()

    return pickle_bytes


def loads(bytestr):
    """De-serializes the given byte string into a python object. The argument must be a byte string
    as returned from the `dumps` method of this module, or a bytes string representing an obspy
    Stream object (using `obspy.core.stream.read` function)
    :param bytestr: a sequence of bytes from e.g., file or database.
    :return: a python object
    """
    try:  # try as obspy read:
        return obspy_read(StringIO(bytestr))
    except TypeError:
        pass
    d_bytestr = None  # decompressed byte string
    if bytestr.starts_with("\x1f\x8b\x08"):  # gzip
        with gzip.GzipFile(mode='rb', fileobj=StringIO(bytestr)) as gzip_obj:
            d_bytestr = gzip_obj.read()
    elif bytestr.starts_with("\x42\x5a\x68"):  # bz2
        d_bytestr = bz2.decompress(bytestr)
    elif bytestr.starts_with("\x50\x4b\x03\x04"):  # zip
        with zipfile.ZipFile(StringIO(bytestr), 'r') as zip_obj:
            namelist = zip_obj.namelist()
            if len(namelist) != 1:
                raise ValueError("Found zipped content with %d archives, `loads` "
                                 "can only uncompress single archive content" % len(namelist))
            d_bytestr = zip_obj.read(namelist[0])
    else:
        # zlib seems not to have a 100% reliable way. See:
        # http://stackoverflow.com/questions/5322860/how-to-detect-quickly-if-a-string-is-zlib-compressed
        try:
            d_bytestr = zlib.decompress(bytestr)
        except:  # pylint: disable=bare-except
            # try numpy compressed data:
            try:
                data = np.load(StringIO(bytestr))
                if len(data.keys()) != 1:
                    raise ValueError("Found numpy compressed content with %d archives, `loads` can "
                                     "only uncompress single archive content" % len(data.keys()))
                return data[data.keys()[0]]
            except:
                pass

    return pickle.loads(bytestr if d_bytestr is None else d_bytestr)


def _load_np(obj):
    data = np.load(StringIO(obj))
    ret = odict()
    for key in data.keys():
        ret[key] = data[key]
    return ret


def _loads_gzip(obj):
    with gzip.GzipFile(mode='rb', fileobj=StringIO(obj)) as gzip_obj:
        return gzip_obj.read()


def _loads_zip(obj):
    with zipfile.ZipFile(StringIO(obj), 'r') as zip_obj:
        ret = odict()
        for name in zip_obj.namelist():
            ret[name] = zip_obj.read(name)
        return ret


def _loads_zlib(obj):
    return zlib.decompress(obj)


def _loads_bz2(obj):
    return bz2.decompress(obj)
