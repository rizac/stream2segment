#!/usr/local/bin/python2.7
# encoding: utf-8
'''
stream2segment.io.utils -- utilities

stream2segment.io.utils tmp module

It defines classes_and_methods

@author:     user_name

@copyright:  2016 organization_name. All rights reserved.

@license:    license

@contact:    user_email
@deffield    updated: Updated
'''

# import sys
# import os
from io import BytesIO
# from collections import OrderedDict as odict
import gzip
import zipfile
import zlib
import bz2
# try:
#     import cPickle as pickle
# except ImportError:
#     import pickle
# import pickle  # uncomment for testing
# import numpy as np
# from obspy.core import Trace, Stream
# from obspy.core.stream import read as obspy_read
from obspy.core.inventory.inventory import read_inventory
# from stream2segment.analysis.mseeds import utcdatetime


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
        sio = BytesIO()
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
        with gzip.GzipFile(mode='rb', fileobj=BytesIO(bytestr)) as gzip_obj:
            bytestr = gzip_obj.read()
    elif bytestr.startswith("\x42\x5a\x68"):  # bz2
        bytestr = bz2.decompress(bytestr)  # raises IOError in case
    elif bytestr.startswith("\x50\x4b\x03\x04"):  # zip
        # raises zipfile.BadZipfile in case
        with zipfile.ZipFile(BytesIO(bytestr), 'r') as zip_obj:
            namelist = zip_obj.namelist()
            if len(namelist) != 1:
                raise ValueError("Found zipped content with %d archives, "
                                 "can only uncompress single archive content" % len(namelist))
            bytestr = zip_obj.read(namelist[0])
    else:
        byte1 = ord(bytestr[0])
        byte2 = ord(bytestr[1])
        if (byte1 * 256 + byte2) % 31 == 0 and (byte1 & 143) == 8:  # zlib. 143=int('10001111', 2)
            bytestr = zlib.decompress(bytestr)  # raises zlib.error in case
    return bytestr


def loads_inv(bytestr):
    '''Returns the inventory object given an input bytes sequence representing an inventory (xml)
    from, e.g., downloaded data
    :param bytestr: the sequence of bytes. It can be compressed with any of the function
    defined here. The method will first try to de-compress data. Then, the de-compressed data
    (if de-compression does not fail) or the data passed as argument will be passed to obspy
    `read_inventory`
    :return: an `class: obspy.core.inventory.inventory.Inventory` object
    '''
    try:
        bytestr = decompress(bytestr)
    except(IOError, zipfile.BadZipfile, zlib.error) as _:
        pass  # we might actually have uncompressed data which starts accidentally with the wrong
    return read_inventory(BytesIO(bytestr), format="STATIONXML")


def dumps_inv(bytestr, compression='gzip', compresslevel=9):
    '''Compresses the bytes sequence representing an inventory (xml) with the given
    compression algorithm
    :param bytestr: the sequence of bytes
    :param compression: string, either ['bz2', 'zlib', 'gzip', 'zip'. Default:
    'gip'], The compression library to use.
    If None or empty string, no compression is applied, and `bytestr` is returned as it is
    :param compresslevel: integer (9 by default). Ignored if `compression` is None or
    or 'zip' (the latter does not accept this argument), this parameter
    controls the level of compression; 1 is fastest and
    produces the least compression, and 9 is slowest and produces the most compression
    :return: a new bytes sequence the same type of `bytestr`, compressed with the given algorithm
    '''
    return compress(bytestr, compression, compresslevel)


# def dumps(obj, format=None,   # @ReservedAssignment  # pylint: disable=redefined-builtin
#           compression='gzip', compresslevel=9):
#     """
#         Serializes `obj` (with optional compression) returning a byte sequence to be, e.g.,
#         saved to file or db and loaded back by the module function :ref: `loads`.
#         :param obj: An `obspy` `Stream` or `Trace` object or anything implementing the obspy Stream
#         `write` method
#         :param format: The format argument passed to obspy.write. If None, it defaults to 'MSEED'
#         and 'pickle', meaning that the two format are tried in that order (obspy pickle protocol
#         defaults to 2 as of November 2016)
#         :param compression: String, either ['bz2', 'zlib', 'gzip', 'zip'. Default:
#         'gip'], The compression library to use (after serializing `obj` with the given format)
#         on the serialized data.
#         If None or empty string, no compression is applied (thus generally increasing the size of
#         the returned byte string).
#         :param compresslevel: integer (9 by default). Ignored if `compression` is None, empty
#         or 'zip' (the latter does not accept this argument), this parameter
#         controls the level of compression; 1 is fastest and
#         produces the least compression, and 9 is slowest and produces the most compression
#     """
#     sio = BytesIO()
#     if not format:
#         try:
#             format = obj.stats._format  # @ReservedAssignment # pylint: disable=protected-access
#         except AttributeError:
#             try:
#                 format = obj[0].stats._format  # @IgnorePep8 # @ReservedAssignment # pylint: disable=protected-access
#             except (AttributeError, TypeError):
#                 format = "MSEED"  # @ReservedAssignment # pylint: disable=protected-access
# 
#     try:
#         obj.write(sio, format=format)
#         pickle_bytes = sio.getvalue()
#         sio.close()
#     except (AttributeError, TypeError, ValueError):
#         # these are exceptions we should NEVER get, regardless of the format
#         raise
#     except:  # pylint: disable=bare-except
#         # obspy raises a bare Exception, we cannot do anything against!
#         if format.lower() == 'pickle':
#             raise
#         obj.write(sio, format='pickle')
#         pickle_bytes = sio.getvalue()
#         sio.close()
# 
#     return compress(pickle_bytes, compression, compresslevel)
# 
# 
# def loads(bytestr):
#     """De-serializes the given byte string into a python object. The argument must be a byte string
#     as returned from the `dumps` method of this module, or a bytes string representing an obspy
#     Stream object (using `obspy.core.stream.read` function)
#     :param bytestr: a sequence of bytes from e.g., file or database.
#     :return: a python object
#     :raises: ValueError if bytestr is compressed and a compression error occurred, or if `bytestr`
#     could not be read as `Stream` object
#     """
#     try:
#         bytestr = decompress(bytestr)
#     except(IOError, zipfile.BadZipfile, zlib.error) as exc:
#         pass  # we might actually have uncompressed data which starts accidentally with the wrong
#         # bytes
#     try:
#         return obspy_read(BytesIO(bytestr))
#     except (TypeError, AttributeError, ValueError) as exc:
#         raise ValueError(str(exc))
    

# def loads_time(timeobj):
#     """Loads timeobj converting it to the relative obspy UTCDateTime
#     :param timeobj: a datetime object, an ISO string representing a date time object, a float
#     representing a timestamp, or None (in which case None is returned)
#     """
#     return utcdatetime(timeobj, None)
# 
# 
# def dumps_time(obspyutctime):
#     """converts UtcDateTime to datetime, returns None if arg is None. This function allows to
#     write obpsy UTCDateTime's to a database in the form of a python datetime object
#     (without timestamp)"""
#     return None if obspyutctime is None else obspyutctime.datetime


# def _load_np(obj):
#     data = np.load(StringIO(obj))
#     ret = odict()
#     for key in data.keys():
#         ret[key] = data[key]
#     return ret
# 
# 
# def _loads_gzip(obj):
#     with gzip.GzipFile(mode='rb', fileobj=StringIO(obj)) as gzip_obj:
#         return gzip_obj.read()
# 
# 
# def _loads_zip(obj):
#     with zipfile.ZipFile(StringIO(obj), 'r') as zip_obj:
#         ret = odict()
#         for name in zip_obj.namelist():
#             ret[name] = zip_obj.read(name)
#         return ret
# 
# 
# def _loads_zlib(obj):
#     return zlib.decompress(obj)
# 
# 
# def _loads_bz2(obj):
#     return bz2.decompress(obj)


