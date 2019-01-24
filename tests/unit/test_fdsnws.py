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

    we want to make http:// (scheme) and the '/query' part optional
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
        if obj.path in ('', '/'):
            raise ValueError("Invalid path (/fdsnws/<service>/</majorversion>) in '%s'" % url)

        self.site = "%s://%s" % (obj.scheme, obj.netloc)

        pth = obj.path
        reg = re.match("^/(?:fdsnws)/(?P<service>.*?)/(?P<majorversion>\\d+)(:?/query/*|/*)$",
                       pth)
        try:
            self.service, self.majorversion = reg.group('service'), reg.group('majorversion')
            if self.service not in [self.STATION, self.DATASEL, self.EVENT]:
                raise ValueError("Invalid <service> '%s' in '%s'" % (self.service, pth))
            try:
                int(self.majorversion)
            except ValueError:
                raise ValueError("Invalid <majorversion> '%s' in '%s'" % (self.majorversion, pth))
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
        return "%s/fdsnws/%s/%d/%s" % (self.site, service or self.service,
                                       majorversion or self.majorversion,
                                       method or self.QUERY)


@pytest.mark.parametrize(['url'],[
                        ('abc.org/fdsnws/station/1/query',),
                        ('abc.org/fdsnws/station/1/query/',),
                        ('abc.org/fdsnws/station/1',),
                        ('abc.org/fdsnws/station/1/',),
                        ])
def test_models_fdsn_url(url):
    fdsn = Fdsnws(url)
    assert fdsn.site == 'https://mock'
    assert fdsn.service == Fdsnws.STATION
    assert fdsn.majorversion == 1

    normalizedurl = fdsn.url()
    for service in [Fdsnws.STATION, Fdsnws.DATASEL, Fdsnws.EVENT, 'abc']:
        assert fdsn.url(service) == normalizedurl.replace('station', service)

    assert fdsn.url(majorversion=55) == normalizedurl.replace('1', '55')

    for method in [Fdsnws.QUERY, Fdsnws.QUERYAUTH, Fdsnws.APPLWADL, Fdsnws.VERSION,
                   'abcdefg']:
        assert fdsn.url(method=method) == normalizedurl.replace('query', method)


#     for url in ["fdsnws/station/1/query",
#                 "/fdsnws/station/1/query",
#                 "http://www.google.com",
#                 "https://mock/fdsnws/station/abc/1/whatever/abcde?h=8&b=76",
#                 "https://mock/fdsnws/station/", "https://mock/fdsnws/station"]:
#         with pytest.raises(ValueError):
#             Fdsnws(url)


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
