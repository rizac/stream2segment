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
        'gzip'], The compression library to use (after serializing `obj` with the given format)
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
