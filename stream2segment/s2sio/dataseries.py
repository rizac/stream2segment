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
from obspy.core.inventory.inventory import read_inventory
try:
    import cPickle as pickle
except ImportError:
    import pickle
# import pickle  # uncomment for testing
import numpy as np
from obspy.core import Stream
from obspy.core import Trace
from obspy.core.stream import read as obspy_read
# from obspy.core.stream import write


# class Object(object):  # pylint: disable=too-few-public-methods
#     """Simple python object. the same would have been acquired via the builtin `type(...)`
#     function but the latter is not pickable"""
# 
#     def __init__(self, **attrs):
#         """Sets the attributes of this object
#         :param attrs: the attributes
#         """
#         self.update(**attrs)
# 
#     def update(self, **attrs):
#         """updates the attributes of this object overriding existing ones, if any
#         :param attrs: the attributes
#         :return: this object
#         """
#         for k, j in attrs.iteritems():
#             setattr(self, k, j)
#         return self
# 
#     def __str__(self):
#         return self.__dict__.__str__()
# 
# 
# class LightTrace(Object):  # pylint: disable=too-few-public-methods
#     """A light obspy trace. A Light trace is a simple python object with the same structure of an
#     `obspy.Trace` object (i.e., with the fields `data` - numeric array, and `stats`, an
#     object with Trace attributes). The purpose is to allow a lighter container for IO operations
#     and to allow different data series to have the same structure of the Trace they originated from
#     (e.g. after applying an fft to a given Trace, see subclass :ref: `FreqTrace`)"""
#     def __init__(self, data, **stats_attrs):
#         """
#             Initializes a new light trace.
#             :param data: [iterable | array | Stream | Trace] the data. If Stream, it must hold only
#             a single trace, and the trace `data` attribute will be used. If Trace, the `trace.data`
#             attribute will be used
#             :param stats_attrs: custom attributes to be set in the attribute `self.stats` (which is
#             a :ref Object type)
# 
#             :Example:
#             l = LightTrace([1,2,3], 'att1':15, 'att2': 34.5)
#             l.data  # returns numpy.array([1,2,3])
#             l.stats # returns an object with fields 'att1'=15 and 'att2'=34.5, e.g.:
#             l.stats.att1 # returns 15
# 
#             l = LightTrace(obspy_trace, 'att1':15, 'att2': 34.5)
#             l.data # returns obspy_trace.data
#             l.stats.starttime # raises AttributeError
#             l.stats.att1 # returns 15
#         """
#         if isinstance(data, Stream):
#             if len(data) != 1:
#                 raise ValueError("Only single trace streams for light traces")
#             data = data.traces[0].data
#         elif isinstance(data, Trace):
#             data = data.data
#         super(LightTrace, self).__init__(data=data)  # basically sets self.data
#         self.stats = Object(**stats_attrs)
# 
# 
# class FreqTrace(LightTrace):  # pylint: disable=too-few-public-methods
#     """An obspy Trace-like object with frequencies on the "x axis".
#     It is built from a data array and has two mandatory fields: `startfreq` and `delta`.
#     This object extends `Light trace` thus has the same structure of an
#     `obspy.Trace` object (i.e., with the fields `data` - numeric array, and `stats`, an
#     object with attributes)"""
#     def __init__(self, data, delta, startfreq=0, y_unit='', **stats_attrs):
#         """
#             Initializes a new FreqTrace.
#             :param data: [iterable | array | Stream | Trace] the data, usually after applying
#             some sort of fft algorithm on a trace `data` attribute. This parameter can be also
#             Stream and /or Trace for backward (superclass) compatibility, although it should not
#             make much sense to pass *time*-series data. However, if Stream, it must hold only
#             a single trace, and the trace `data` attribute will be used. If Trace, the `trace.data`
#             attribute will be used
#             :param delta: the sampling frequency (the distance between two points on the frequency
#             domain, in Herz)
#             :param startfreq: integer, defaults to zero. Equivalent to stats.starttime for an `obspy`
#             Trace, the frequency value of the first point in data
#             :param y_unit (str): empty by default, provide the optional y unit. Typical values might
#             be 'fft' for a simple Fft (with complex-values data), 'amplitude' or 'power' for
#             amplitude and power spectra (with real-values data), or whatever is meaningful to the
#             user
#             :param stats_attrs: custom attributes to be set in the attribute `self.stats` (which is
#             a :ref Object type). 'startfreq' and 'delta' are set by default as stats attributes
# 
#             :Example:
#             f = FreqTrace([1,2,3], 0, 1.5, 'att1':15, 'att2': 34.5)
#             f.data  # returns numpy.array([1,2,3])
#             f.stats # returns an object with fields 'att1'=15 and 'att2'=34.5, e.g.:
#             f.stats.startfreq  # returns 0
#             f.stats.delta  # returns 1.5
#             f.stats.att1 # returns 15
#         """
#         stats_attrs.update(startfreq=startfreq, delta=delta, y_unit=y_unit)
#         super(FreqTrace, self).__init__(data, **stats_attrs)


def dumps(obj, format=None,   # @ReservedAssignment  # pylint: disable=redefined-builtin
          compression='gzip', compresslevel=9):
    """
        Serializes `obj` (with optional compression) returning a byte sequence to be, e.g.,
        saved to file or db and loaded back by the module function :ref: `loads`.
        :param obj: An `obspy` `Stream` or `Trace` object or anything implementing the obspy Stream
        `write` method
        :param format: The format argument passed to obspy.write. If None, it defaults to 'MSEED'
        and 'pickle', meaning that the two format are tried in that order (obspy pickle protocol
        defaults to 2 as of November 2016)
        :param compression: String, either ['bz2', 'zlib', 'gzip', 'zip'. Default:
        'gip'], The compression library to use (after serializing `obj` with the given format)
        on the serialized data.
        If None or empty string, no compression is applied (thus generally increasing the size of
        the returned byte string).
        :param compresslevel: integer (9 by default). Ignored if `compression` is None, empty
        or 'zip' (the latter does not accept this argument), this parameter
        controls the level of compression; 1 is fastest and
        produces the least compression, and 9 is slowest and produces the most compression
    """
    sio = StringIO()
    if not format:
        try:
            format = obj.stats._format  # @ReservedAssignment # pylint: disable=protected-access
        except AttributeError:
            try:
                format = obj[0].stats._format  # @IgnorePep8 # @ReservedAssignment # pylint: disable=protected-access
            except (AttributeError, TypeError):
                format = "MSEED"  # @ReservedAssignment # pylint: disable=protected-access

    try:
        obj.write(sio, format=format)
        pickle_bytes = sio.getvalue()
        sio.close()
    except (AttributeError, TypeError, ValueError):
        # these are exceptions we should NEVER get, regardless of the format
        raise
    except:  # pylint: disable=bare-except
        # obspy raises a bare Exception, we cannot do anything against!
        if format.lower() == 'pickle':
            raise
        obj.write(sio, format='pickle')
        pickle_bytes = sio.getvalue()
        sio.close()

    return compress(pickle_bytes, compression, compresslevel)


def compress(bytestr, compression='gzip', compresslevel=9):
    """
        Compresses `bytestr` returning a new compressed byte sequence
        :param bytestr: (string) a sequence of bytes to be compressed
        :param compression: String, either ['bz2', 'zlib', 'gzip', 'zip'. Default:
        'gip'], The compression library to use (after serializing `obj` with the given format)
        on the serialized data.
        If None or empty string, no compression is applied, and `bytestr` is returned as it is
        :param compresslevel: integer (9 by default). Ignored if `compression` is None, empty
        or 'zip' (the latter does not accept this argument), this parameter
        controls the level of compression; 1 is fastest and
        produces the least compression, and 9 is slowest and produces the most compression
    """
    if compression == 'bz2':
        return bz2.compress(bytestr, compresslevel=compresslevel)
    elif compression == 'zlib':
        return zlib.compress(bytestr, compresslevel)
    elif compression:
        sio = StringIO()
        if compression == 'gzip':
            with gzip.GzipFile(mode='wb', fileobj=sio, compresslevel=compresslevel) as gzip_obj:
                gzip_obj.write(bytestr)
                # Note: DO NOT return sio.getvalue() WITHIN the with statement, the gzip file obj
                # needs to be closed first. FIXME: ref?
        elif compression == 'zip':
            # In this case, use the compress argument to ZipFile to compress the data,
            # since writestr() does not take compress as an argument. See:
            # https://pymotw.com/2/zipfile/#writing-data-from-sources-other-than-files
            with zipfile.ZipFile(sio, 'w', compression=zipfile.ZIP_DEFLATED) as zip_obj:
                zip_obj.writestr("x", bytestr)  # first arg must be a nonempty str
        else:
            raise ValueError("compression '%s' not in ('gzip', 'zlib', 'bz2', 'zip')" %
                             str(compression))

        return sio.getvalue()

    return bytestr


def loads(bytestr):
    """De-serializes the given byte string into a python object. The argument must be a byte string
    as returned from the `dumps` method of this module, or a bytes string representing an obspy
    Stream object (using `obspy.core.stream.read` function)
    :param bytestr: a sequence of bytes from e.g., file or database.
    :return: a python object
    :raises: ValueError if bytestr is compressed and a compression error occurred, or if `bytestr`
    could not be read as `Stream` object
    """
    try:
        bytestr = decompress(bytestr)
    except(IOError, zipfile.BadZipfile, zlib.error) as exc:
        pass  # we might actually have uncompressed data which starts accidentally with the wrong
        # bytes
    try:
        return obspy_read(StringIO(bytestr))
    except (TypeError, AttributeError, ValueError) as exc:
        raise ValueError(str(exc))


def decompress(bytestr):
    """De-compresses bytestr (a sequence of bytes) trying to guess the compression format. If no
    guess can be made, returns bytestr. Otherwise, returns the de-compressed sequence of bytes.
    Raises IOError, zipfile.BadZipfile, zlib.error if compression is detected but did not work. Note
    that this might happen if (accidentally) the sequence of bytes is not compressed but starts
    with bytes denoting a compression type. Thus function caller should not necessarily raise
    exceptions if this function does, but try to read `bytestr` as if it was not compressed
    """
    # check if the data is compressed. Note: this is a hint! For info see:
    # http://stackoverflow.com/questions/19120676/how-to-detect-type-of-compression-used-on-the-file-if-no-file-extension-is-spe
    if bytestr.startswith("\x1f\x8b\x08"):  # gzip
        # raises IOError in case
        with gzip.GzipFile(mode='rb', fileobj=StringIO(bytestr)) as gzip_obj:
            bytestr = gzip_obj.read()
    elif bytestr.startswith("\x42\x5a\x68"):  # bz2
        bytestr = bz2.decompress(bytestr)  # raises IOError in case
    elif bytestr.startswith("\x50\x4b\x03\x04"):  # zip
        # raises zipfile.BadZipfile in case
        with zipfile.ZipFile(StringIO(bytestr), 'r') as zip_obj:
            namelist = zip_obj.namelist()
            if len(namelist) != 1:
                raise ValueError("Found zipped content with %d archives, `loads` "
                                 "can only uncompress single archive content" % len(namelist))
            bytestr = zip_obj.read(namelist[0])
    else:
        byte1 = ord(bytestr[0])
        byte2 = ord(bytestr[1])
        if (byte1 * 256 + byte2) % 31 == 0 and (byte1 & 143) == 8:  # zlib. 143=int('10001111', 2)
            bytestr = zlib.decompress(bytestr)  # raises zlib.error in case
    return bytestr


def loads_inv(bytestr):  # FIXME: move to different package?
    try:
        bytestr = decompress(bytestr)
    except(IOError, zipfile.BadZipfile, zlib.error) as _:
        pass  # we might actually have uncompressed data which starts accidentally with the wrong
        # bytes
#     s = StringIO(bytestr)
#     s.seek(0)  # FIXME: necessary ??
#     try:
#         inv = read_inventory(StringIO(bytestr), format="STATIONXML")
#     finally:
#         s.close()
#     return inv
    return read_inventory(StringIO(bytestr), format="STATIONXML")


def dumps_inv(bytestr, compression='gzip', compresslevel=9):
    return compress(bytestr, compression, compresslevel)


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

# ================================================================================================
# def _dumps(obj, format=None,  # @ReservedAssignment  # pylint: disable=redefined-builtin
#           compression='gzip', compresslevel=9):
#     """
#         Serializes `obj` (with optional compression) returning a byte sequence to be, e.g.,
#         saved to file or db and loaded back by the module function :ref: `loads`.
#         :param obj: A Python object (Stream, Trace, or any object)
#         :param format: The format to be used when serializing obj, i.e., for producing a sequence
#         of bytes representing `obj`. If integer, data will be
#         serialized via pickle dumps and this argument
#         is `protocol` argument. If 'numpy', then numpy.savez_compressed will be used, otherwise
#         it is the format string passed to obspy.Stream (or obspy.core.Trace) `write` method (in this
#         latter case, `obj` needs to be either a Stream or a Trace object). If None, it infers the
#         format from `obj`: if the latter is either a Stream or a Trace, it defaults to 'MSEED',
#         otherwise to pickle.HIGHEST_PROTOCOL (integer). Note that, according to the docs of pickle,
#         a negative int defaults always to pickle.HIGHEST_PROTOCOL (for more information, see:
#         https://docs.python.org/2/library/pickle.html#data-stream-format)
#         :param compression: String, either ['bz2', 'zlib', 'gzip', 'zip'. Default:
#         'gip'], The compression library to use (after serializing `obj` with the given format)
#         on the serialized data.
#         If None or empty string, no compression is applied (thus generally increasing the size of
#         the returned byte string).
#         Note that if format is 'numpy' this compression might be redundant, as 'numpy' already
#         compresses data.
#         :param compresslevel: integer (9 by default). Ignored if `compression` is None, empty,
#         'numpy' or 'zip' (the latter two do not accept compresslevel arguments), this parameter
#         controls the level of compression; 1 is fastest and
#         produces the least compression, and 9 is slowest and produces the most compression
#     """
#     if format is None:
#         if isinstance(obj, Stream) or isinstance(obj, Trace):
#             format = 'MSEED'  # @ReservedAssignment
#         else:
#             format = pickle.HIGHEST_PROTOCOL  # @ReservedAssignment
# 
#     if format == 'numpy':
#         sio = StringIO()
#         np.savez_compressed(sio, obj)
#         pickle_bytes = sio.getvalue()
#         sio.close()
#     elif isinstance(format, int):
#         pickle_bytes = pickle.dumps(obj, protocol=format)
#     else:
#         sio = StringIO()
#         obj.write(sio, format=format)
#         pickle_bytes = sio.getvalue()
#         sio.close()
# 
#     if compression == 'bz2':
#         return bz2.compress(pickle_bytes, compresslevel=compresslevel)
#     elif compression == 'zlib':
#         return zlib.compress(pickle_bytes, compresslevel)
#     elif compression:
#         sio = StringIO()
#         if compression == 'gzip':
#             with gzip.GzipFile(mode='wb', fileobj=sio, compresslevel=compresslevel) as gzip_obj:
#                 gzip_obj.write(pickle_bytes)
#                 # Note: DO NOT return sio.getvalue() WITHIN the with statement, the gzip file obj
#                 # needs to be closed first. FIXME: ref?
#         elif compression == 'zip':
#             # In this case, use the compress argument to ZipFile to compress the data,
#             # since writestr() does not take compress as an argument. See:
#             # https://pymotw.com/2/zipfile/#writing-data-from-sources-other-than-files
#             with zipfile.ZipFile(sio, 'w', compression=zipfile.ZIP_DEFLATED) as zip_obj:
#                 zip_obj.writestr("x", pickle_bytes)  # first arg must be a nonempty str
#         else:
#             raise ValueError("compression '%s' not in ('gzip', 'zlib', 'bz2', 'zip')" %
#                              str(compression))
# 
#         return sio.getvalue()
# 
#     return pickle_bytes
# 
# 
# def _loads(bytestr):
#     """De-serializes the given byte string into a python object. The argument must be a byte string
#     as returned from the `dumps` method of this module, or a bytes string representing an obspy
#     Stream object (using `obspy.core.stream.read` function)
#     :param bytestr: a sequence of bytes from e.g., file or database.
#     :return: a python object
#     """
#     # check if the data is compressed. Note: this is a hint! For info see:
#     # http://stackoverflow.com/questions/19120676/how-to-detect-type-of-compression-used-on-the-file-if-no-file-extension-is-spe
#     d_bytestr = None  # decompressed byte string
#     if bytestr.startswith("\x1f\x8b\x08"):  # gzip
#         try:
#             with gzip.GzipFile(mode='rb', fileobj=StringIO(bytestr)) as gzip_obj:
#                 d_bytestr = gzip_obj.read()
#         except IOError:
#             pass
#     elif bytestr.startswith("\x42\x5a\x68"):  # bz2
#         try:
#             d_bytestr = bz2.decompress(bytestr)
#         except IOError:
#             pass
#     elif bytestr.startswith("\x50\x4b\x03\x04"):  # zip
#         try:
#             with zipfile.ZipFile(StringIO(bytestr), 'r') as zip_obj:
#                 namelist = zip_obj.namelist()
#                 if len(namelist) != 1:
#                     raise ValueError("Found zipped content with %d archives, `loads` "
#                                      "can only uncompress single archive content" % len(namelist))
#                 d_bytestr = zip_obj.read(namelist[0])
#         except zipfile.BadZipfile:
#             pass
#     else:
#         byte1 = ord(bytestr[0])
#         byte2 = ord(bytestr[1])
#         if (byte1 * 256 + byte2) % 31 == 0 and (byte1 & 143) == 8:  # zlib. 143=int('10001111', 2)
#             try:
#                 d_bytestr = zlib.decompress(bytestr)
#             except zlib.error:
#                 pass
#     # try with obspy read:
#     try:
#         return obspy_read(StringIO(bytestr if d_bytestr is None else d_bytestr))
#     except TypeError:
#         pass
# 
#     # try with pickle:
#     try:
#         return pickle.loads(bytestr if d_bytestr is None else d_bytestr)
#     except (pickle.UnpicklingError, AttributeError,  EOFError, ImportError, IndexError):
#         # http://stackoverflow.com/questions/33307623/python-exception-safe-pickle-use
#         pass
# 
#     # try with numpy:
#     try:
#         data = np.load(StringIO(bytestr))  # allow_pickle=True by default
#         if len(data.keys()) != 1:
#             raise ValueError("Found numpy compressed content with %d archives, `loads` can "
#                              "only uncompress single archive content" % len(data.keys()))
#         return data[data.keys()[0]]
#     except IOError:  # Note: ValueError ia also raised but only if allow_pickle=False
#         pass
# 
#     raise ValueError("Unable to loads data")



