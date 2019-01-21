'''
Created on 17 Jan 2019

@author: riccardo
'''
import pytest
from itertools import product
from stream2segment.io.db.pdsql import colnames
from stream2segment.io.db.models import Event
from stream2segment.utils import strptime

try:  # py3:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse

import re

'''
RATIONAE

in the eventws, the user can supply:
- a string 'iris' 'isc' 'emsc' OR
- an url which represents the url of a webservice query url without the trailing '?' and query parameters,
    e.g. 'https://service.iris.edu/fdsnws/event/1/query'
    or 'service.iris.edu/fdsnws/event/1/query' (will be converted to 'service.iris.edu/fdsnws/event/1/query')


In the dataws, the user can supply:
- a string 'iris' or 'eida', OR
- an url which represents the full portion of the webservice url that will be concatenated with
    the query parameters. It must the in the form
    scheme://host/fdsnws/<service or dataselect>/<majorversion>/
    scheme://host/  (will default to scheme://host/fdsnws/dataselect/1/)
    host/  (will default to http://host/fdsnws/dataselect/1/)
    The last slash is optional. Also, 'dataselect' can be replaced by 'station'
    as the program will use 
    be fdsnws compliant:
    <site>/fdsnws/<service>/<majorversion>/
    where [service] must be either 'station' or 'dataselect'
    Valid examples:
    'https://service.iris.edu/fdsnws/station/1/'
    'https://service.iris.edu/fdsnws/station/1'
    'https://service.iris.edu/'  (will set major
    'https://service.iris.edu/fdsnws/station/1/'
    
    
    'https://service.iris.edu/fdsnws/station/1/'
    'https://service.iris.edu/fdsnws/station/1'
    'https://service.iris.edu' (will default [majorversion] to 1)
    The examples above can be provided also without the schema ('https://') but in that
    case 'http://' will be used.
    Example:
    'service.iris.edu' will use:
    'http://service.iris.edu/fdsnws/dataselect/1/query' for the station search
    'http://service.iris.edu/fdsnws/dataselect/1/query' for the waveform download

'''

def split_url(url):
    obj = urlparse(url)
    if not obj.scheme:
        obj = urlparse('http://' + url)
    if not obj.netloc:
        raise ValueError('no domain specified, e.g. http://<domain>')
    return obj.scheme + "://", obj.netloc, '' if obj.path == '/' else obj.path


class Fdsnws(object):
    '''simple class parsing an fdsn url and allowing to build any new well-formed url
    associated to services and methods
    of the site url. Raises ValueError if the url is not a valid fdsn url of the form
    '<site>'
    '<site>/fdsnws/<service>/<majorversion>'
    '<site>/fdsnws/<service>/<majorversion>/query'
    All three forms will ignore the trailing (ending) slash, if present
    (if site has no scheme, it will default to "http")

    Examples:
    ```
        fdsn = Fdsnws('...')
        normalized_station_query_url = fdsn.url(Fdsnws.STATION)
        normalized_dataselect_query_url = fdsn.url(Fdsnws.DATASEL)
        site_url = fdsn.site  # the portion of text before '/fdsnws/....', must start with
            http: or https:
        majorversion = fdsn.majorversion  # int
    ```
    '''
    # equals to the string 'station', used in urls for identifying the fdsn station service:
    STATION = 'station'
    # equals to the string 'dataselect', used in urls for identifying the fdsn data service:
    DATASEL = 'dataselect'
    # equals to the string 'event', used in urls for identifying the fdsn event service:
    EVENT = 'event'
    # equals to the string 'query', used in urls for identifying the fdsn service query method:
    QUERY = 'query'
    # equals to the string 'queryauth', used in urls for identifying the fdsn service query
    # method (with authentication):
    QUERYAUTH = 'queryauth'
    # equals to the string 'auth', used  (by EIDA only?) in urls for querying username and
    # password with provided token:
    AUTH = 'auth'
    # equals to the string 'version', used in urls for identifying the fdsn service
    # query method:
    VERSION = 'version'
    # equals to the string 'application.wadl', used in urls for identifying the fdsn service
    # application wadl method:
    APPLWADL = 'application.wadl'

    def __init__(self, url, default_service=None, default_majorversion=1):
        '''initializes a Fdsnws object from a fdsn url

        If url does not contain the <service> and <majorversion> tokens in the
        url path, then they will default to the defaults provided (see below)

        :param url: string denoting the Fdsn web service url
        :param default_service: None or string as one of the possible class static attributes:
            `DATASEL` (default when None), `STATION`, `EVENT`. Used only `url`
            has no path to set this class default service
        :param default_majorversion: integer denoting the majorversion of the web service,
            defaults to 1. Used only if `url` has no path to set this object
            default majorversion
        '''
        # do not use urlparse as we should import from stream2segment.url for py2 compatibility
        # but this will cause circular imports:

        obj = urlparse(url)
        if not obj.scheme:
            obj = urlparse('http://' + url)
        if not obj.netloc:
            raise ValueError('no domain specified, e.g. http://<domain>')
        self.site = "%s://%s" % (obj.scheme, obj.netloc)
        self.service = default_service or self.DATASEL
        self.majorversion = str(default_majorversion)
        if obj.path and not obj.path == '/':   # ignore cases where obj.path is empty or '/'
            pth = obj.path
            reg = re.match("^/(?:fdsnws)/(?P<service>.*?)/(?P<majorversion>\\d+)(:?/query/*|/*)$",
                           pth)
            try:
                service, majorversion = reg.group('service'), reg.group('majorversion')
                if service not in [self.STATION, self.DATASEL, self.EVENT]:
                    raise ValueError("Invalid <service> in '%s'" % pth)
                try:
                    majorversion = int(majorversion)
                except ValueError:
                    raise ValueError("Invalid <majorversion> in '%s'" % pth)
                self.service = service
                self.majorversion = str(majorversion)
            except ValueError:
                raise
            except Exception:
                raise ValueError("Invalid FDSN path '%s' should be "
                                 "'/fdsnws/<service>/<majorversion>', "
                                 "check potential typos" % str(obj.path))

    def url(self, service=None, majorversion=None, method=None):
        '''builds a new url from this object url. Arguments which are 'None' will default
        to this object's url passed in the constructor. The returned url
        denotes the base url (with no query parameter and no trailing '?' or '/') in
        order to build queries to a fdsn web service

        :param service: None or one of this class static attributes:
            `STATION`, `DATASEL`, `EVENT`
        :param majorversion: None or a numeric integer (or string parsable to int)
            denoting the service major version. Defaults to 1 when None
            `STATION`, `DATASEL`, `EVENT`
        :param method: None or one of the class static attributes
            `QUERY` (the default when None), `QUERYAUTH`,
            `VERSION`, `AUTH` or `APPLWADL`
        '''
        return "%s/fdsnws/%s/%d/%s" % (self.site, service or self._service,
                                       majorversion or self._majorversion,
                                       method or self.QUERY)


def tst_models_fdsn_url():
    for url in ["https://mock/fdsnws/station/1/query",
                "https://mock/fdsnws/station/1/query?",
                "https://mock/fdsnws/station/1/", "https://mock/fdsnws/station/1",
                "https://mock/fdsnws/station/1/abcde?h=8&b=76",
                "https://mock/fdsnws/station/1/whatever/abcde?h=8&b=76"]:
        fdsn = Fdsnws(url)
        assert fdsn.site == 'https://mock'
        assert fdsn.service == Fdsnws.STATION
        assert fdsn.majorversion == 1
        normalizedurl = fdsn.url()
        assert normalizedurl == 'https://mock/fdsnws/station/1/query'
        for service in [Fdsnws.STATION, Fdsnws.DATASEL, Fdsnws.EVENT, 'abc']:
            assert fdsn.url(service) == normalizedurl.replace('station', service)

        assert fdsn.url(majorversion=55) == normalizedurl.replace('1', '55')

        for method in [Fdsnws.QUERY, Fdsnws.QUERYAUTH, Fdsnws.APPLWADL, Fdsnws.VERSION,
                       'abcdefg']:
            assert fdsn.url(method=method) == normalizedurl.replace('query', method)

    for url in ["fdsnws/station/1/query",
                "/fdsnws/station/1/query",
                "http://www.google.com",
                "https://mock/fdsnws/station/abc/1/whatever/abcde?h=8&b=76",
                "https://mock/fdsnws/station/", "https://mock/fdsnws/station"]:
        with pytest.raises(ValueError):
            Fdsnws(url)


def tst_fdsn_real_url():
    baseurl = 'earthquake.usgs.gov'
    def_scheme = 'http'
    for url in [baseurl, baseurl + '/']:
        for def_service, def_mv in product([Fdsnws.EVENT, Fdsnws.STATION, Fdsnws.DATASEL],
                                           [1, 2]):
            assert Fdsnws(url, def_service, def_mv).url() == \
                '%s://%s/fdsnws/%s/%s/%s' % (def_scheme, url, def_service, def_mv, Fdsnws.QUERY)

    baseurl = 'http://earthquake.usgs.gov'
    for url in [baseurl, baseurl + '/']:
        for def_service, def_mv in product([Fdsnws.EVENT, Fdsnws.STATION, Fdsnws.DATASEL],
                                           [1, 2]):
            assert Fdsnws(url, def_service, def_mv).url() == \
                'http://%s/fdsnws/%s/%s/%s' % (url, def_service, def_mv, Fdsnws.QUERY)
    
    url = 'http://earthquake.usgs.gov/'
    

def _get(list_, index):
    return list_[index].strip() if -len(list_) <= index < len(list_) else None


def response2df_isc(response_data):
    buf = []
    expects_event_header = expects_event_data = False
    reg = re.compile(r'(.*)\n')
    reg2 = re.compile(' +')
    data = []
    row = {}
    columns = list(colnames(Event, pkey=False, fkey=False))
    for match in reg.finditer(response_data):
        line = match.group(1)
        if line.startwith('Event'):
            row = {_: None for _ in columns}
            row[Event.catalog.key] = row[Event.contributor.key] = 'ISC'
            expects_event_header = expects_event_data = False
            elements = reg2.split(line)
            if not _get(elements, 1):
                continue
            row[Event.event_id.key] = row[Event.contributor_id.key] = _get(elements, 1)
            row[Event.event_location_name.key] = _get(elements, 2)
            expects_event_header = True
            continue
        elif expects_event_header:
            expects_event_header = False
            elements = reg2.split(line)
            if _get(elements, 0) == 'Date':
                expects_event_data = True
            continue
        elif expects_event_data:
            expects_event_header = False
            elements = reg2.split(line)
            dat, tme = _get(elements, 0), _get(elements, 1)
            try:
                dtime = strptime(dat + 'T' + tme)
                row[Event.time.key] = dtime.strftime('%Y-%m-%dT%H-%M-%S')
            except (TypeError, ValueError):
                pass
            row[Event.latitude.key] = _get(elements, 4)
            row[Event.longitude.key] = _get(elements, 5)
            depth = _get(elements, 9)
            if depth is not None and depth[-1] == 'f':
                depth = depth[:-1]
            row[Event.depth_km.key] = depth
            row[Event.author.key] = _get(elements, 17)
            
            
            
            
            
            
def text_isp_reader(data):
    data.read('isc_response.txt')
    
    
