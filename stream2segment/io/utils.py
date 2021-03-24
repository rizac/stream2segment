# encoding: utf-8
"""
IO utilities and compression functions

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
import re
# we could simply import urlparse from stream2segment.utils, but we want this module
# to be standalone. thus:
try:  # py3:
    from urllib.parse import urlparse
except ImportError:  # py2
    from urlparse import urlparse  # noqa

# from io import BytesIO
#
# import gzip
# import zipfile
# import zlib
# import bz2
#
# from obspy.core.inventory.inventory import read_inventory


class Fdsnws(object):
    """Fdsn w(eb) s(ervice) URL normalizer. Gets any URL, checks its
    correctness and allows retrieving all other FDSN URLs easily and safely.
    Example:
    ```
    fdsn = Fdsnws(url)
    station_query_url = fdsn.url(Fdsnws.STATION)
    dataselect_query_url = fdsn.url(Fdsnws.DATASEL)
    dataselect_queryauth_url = fdsn.url(Fdsnws.DATASEL, method=Fdsnws.QUERYAUTH)
    ```
    """
    # equals to the string 'station', used in urls for identifying the FDSN
    # station service:
    STATION = 'station'
    # equals to the string 'dataselect', used in urls for identifying the FDSN
    # data service:
    DATASEL = 'dataselect'
    # equals to the string 'event', used in urls for identifying the FDSN event
    # service:
    EVENT = 'event'
    # equals to the string 'query', used in urls for identifying the FDSN
    # service query method:
    QUERY = 'query'
    # equals to the string 'queryauth', used in urls for identifying the FDSN
    # service query method (with authentication):
    QUERYAUTH = 'queryauth'
    # equals to the string 'auth', used  (by EIDA only?) in urls for querying
    # username and password with provided token:
    AUTH = 'auth'
    # equals to the string 'version', used in urls for identifying the FDSN
    # service query method:
    VERSION = 'version'
    # equals to the string 'application.wadl', used in urls for identifying the
    # FDSN service application wadl method:
    APPLWADL = 'application.wadl'

    def __init__(self, url):
        """Initialize a Fdsnws object from a FDSN URL

        :param url: string denoting the Fdsn web service url
            Example of valid urls (the scheme 'https://' might be omitted
            and will default to 'http://'. An ending '/' or '?' will be ignored
            if present):
            https://www.mysite.org/fdsnws/<station>/<majorversion>
            http://www.mysite.org/fdsnws/<station>/<majorversion>/<method>
        """
        # do not use urlparse as we should import from stream2segment.url for
        # py2 compatibility but this will cause circular imports:

        obj = urlparse(url)
        if not obj.scheme:
            obj = urlparse('http://' + url)
        if not obj.netloc:
            raise ValueError('no domain specified or invalid scheme, '
                             'check typos')

        self.site = "%s://%s" % (obj.scheme, obj.netloc)

        pth = obj.path
        #  urlparse has already removed query char '?' and params and fragment
        # from the path. Now check the latter:
        reg = re.match("^(?:/.+)*/fdsnws/(?P<service>[^/]+)/"
                       "(?P<majorversion>[^/]+)(?P<method>.*)$",
                       pth)
        try:
            self.service = reg.group('service')
            self.majorversion = reg.group('majorversion')
            method = reg.group('method')

            if self.service not in [self.STATION, self.DATASEL, self.EVENT]:
                raise ValueError("Invalid <service> '%s' in '%s'" %
                                 (self.service, pth))
            try:
                float(self.majorversion)
            except ValueError:
                raise ValueError("Invalid <majorversion> '%s' in '%s'" %
                                 (self.majorversion, pth))
            if method not in ('', '/'):
                method = method[1:] if method[0] == '/' else method
                method = method[:-1] if len(method) > 1 and method[-1] == '/' \
                    else method
                if method not in ['', self.QUERY, self.QUERYAUTH, self.AUTH,
                                  self.VERSION, self.APPLWADL]:
                    raise ValueError("Invalid method '%s' in '%s'" %
                                     (method, pth))
        except ValueError:
            raise
        except Exception:
            raise ValueError("Invalid FDSN URL '%s': it should be "
                             "'[site]/fdsnws/<service>/<majorversion>', "
                             "check potential typos" % str(url))

    def url(self, service=None, majorversion=None, method=None):
        """Build a new url from this object url. Arguments which are 'None'
        will default to this object's url passed in the constructor. The
        returned URL denotes the base url (with no query parameter and no
        trailing '?' or '/') in order to build queries to a FDSN web service

        :param service: None or one of this class static attributes:
            `STATION`, `DATASEL`, `EVENT`
        :param majorversion: None or numeric value or string parsable to number
            denoting the service major version. Defaults to 1 when None
            `STATION`, `DATASEL`, `EVENT`
        :param method: None or one of the class static attributes
            `QUERY` (the default when None), `QUERYAUTH`, `VERSION`, `AUTH`
            or `APPLWADL`
        """
        return "%s/fdsnws/%s/%s/%s" % (self.site, service or self.service,
                                       str(majorversion or self.majorversion),
                                       method or self.QUERY)

    def __str__(self):
        return self.url('<service>', None, '<method>')


# def compress(bytestr, compression='gzip', compresslevel=9):
#     """
#         Compresses `bytestr` returning a new compressed byte sequence
#         :param bytestr: (string) a sequence of bytes to be compressed
#         :param compression: String, either ['bz2', 'zlib', 'gzip', 'zip'. Default:
#         'gzip'], The compression library to use (after serializing `obj` with the given format)
#         on the serialized data.
#         If None or empty string, no compression is applied, and `bytestr` is returned as it is
#         :param compresslevel: integer (9 by default). Ignored if `compression` is None, empty
#         or 'zip' (the latter does not accept this argument), this parameter
#         controls the level of compression; 1 is fastest and
#         produces the least compression, and 9 is slowest and produces the most compression
#     """
#     if compression == 'bz2':
#         return bz2.compress(bytestr, compresslevel=compresslevel)
#     elif compression == 'zlib':
#         return zlib.compress(bytestr, compresslevel)
#     elif compression:
#         sio = BytesIO()
#         if compression == 'gzip':
#             with gzip.GzipFile(mode='wb', fileobj=sio, compresslevel=compresslevel) as gzip_obj:
#                 gzip_obj.write(bytestr)
#                 # Note: DO NOT return sio.getvalue() WITHIN the with statement, the gzip file obj
#                 # needs to be closed first. FIXME: ref?
#         elif compression == 'zip':
#             # In this case, use the compress argument to ZipFile to compress the data,
#             # since writestr() does not take compress as an argument. See:
#             # https://pymotw.com/2/zipfile/#writing-data-from-sources-other-than-files
#             with zipfile.ZipFile(sio, 'w', compression=zipfile.ZIP_DEFLATED) as zip_obj:
#                 zip_obj.writestr("x", bytestr)  # first arg must be a nonempty str
#         else:
#             raise ValueError("compression '%s' not in ('gzip', 'zlib', 'bz2', 'zip')" %
#                              str(compression))
#
#         return sio.getvalue()
#
#     return bytestr
#
#
# def decompress(bytestr):
#     """De-compresses bytestr (a sequence of bytes) trying to guess the compression format. If no
#     guess can be made, returns bytestr. Otherwise, returns the de-compressed sequence of bytes.
#     Raises IOError, zipfile.BadZipfile, zlib.error if compression is detected but did not work. Note
#     that this might happen if (accidentally) the sequence of bytes is not compressed but starts
#     with bytes denoting a compression type. Thus function caller should not necessarily raise
#     exceptions if this function does, but try to read `bytestr` as if it was not compressed
#     """
#     # check if the data is compressed. Note: this is a hint! For info see:
#     # http://stackoverflow.com/questions/19120676/how-to-detect-type-of-compression-used-on-the-file-if-no-file-extension-is-spe
#     if bytestr.startswith(b"\x1f\x8b\x08"):  # gzip
#         # raises IOError in case
#         with gzip.GzipFile(mode='rb', fileobj=BytesIO(bytestr)) as gzip_obj:
#             bytestr = gzip_obj.read()
#     elif bytestr.startswith(b"\x42\x5a\x68"):  # bz2
#         bytestr = bz2.decompress(bytestr)  # raises IOError in case
#     elif bytestr.startswith(b"\x50\x4b\x03\x04"):  # zip
#         # raises zipfile.BadZipfile in case
#         with zipfile.ZipFile(BytesIO(bytestr), 'r') as zip_obj:
#             namelist = zip_obj.namelist()
#             if len(namelist) != 1:
#                 raise ValueError("Found zipped content with %d archives, "
#                                  "can only uncompress single archive content" % len(namelist))
#             bytestr = zip_obj.read(namelist[0])
#     else:
#         barray = bytearray(bytestr[:2])  # make things python 2+ 3 compatible:
#         # https://stackoverflow.com/questions/41843579/typeerror-ord-expected-string-of-length-1-but-int-found
#         byte1 = barray[0]
#         byte2 = barray[1]
#         if (byte1 * 256 + byte2) % 31 == 0 and (byte1 & 143) == 8:  # zlib. 143=int('10001111', 2)
#             bytestr = zlib.decompress(bytestr)  # raises zlib.error in case
#     return bytestr
#
#
# def loads_inv(bytestr):
#     '''Returns the inventory object given an input bytes sequence representing an inventory (xml)
#     from, e.g., downloaded data
#     :param bytestr: the sequence of bytes. It can be compressed with any of the function
#     defined here. The method will first try to de-compress data. Then, the de-compressed data
#     (if de-compression does not fail) or the data passed as argument will be passed to obspy
#     `read_inventory`
#     :return: an `class: obspy.core.inventory.inventory.Inventory` object
#     '''
#     try:
#         bytestr = decompress(bytestr)
#     except(IOError, zipfile.BadZipfile, zlib.error) as _:
#         pass  # we might actually have uncompressed data which starts accidentally with the wrong
#     return read_inventory(BytesIO(bytestr), format="STATIONXML")
#
#
# def dumps_inv(bytestr, compression='gzip', compresslevel=9):
#     '''Compresses the bytes sequence representing an inventory (xml) with the given
#     compression algorithm
#     :param bytestr: the sequence of bytes
#     :param compression: string, either ['bz2', 'zlib', 'gzip', 'zip'. Default:
#     'gip'], The compression library to use.
#     If None or empty string, no compression is applied, and `bytestr` is returned as it is
#     :param compresslevel: integer (9 by default). Ignored if `compression` is None or
#     or 'zip' (the latter does not accept this argument), this parameter
#     controls the level of compression; 1 is fastest and
#     produces the least compression, and 9 is slowest and produces the most compression
#     :return: a new bytes sequence the same type of `bytestr`, compressed with the given algorithm
#     '''
#     return compress(bytestr, compression, compresslevel)
