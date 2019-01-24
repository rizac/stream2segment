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

    def __init__(self, url):
        '''initializes a Fdsnws object from a fdsn url

        If url does not contain the <service> and <majorversion> tokens in the
        url path, then they will default to the defaults provided (see below)

        :param url: string denoting the Fdsn web service url
            Example of valid values ('dataselect' can also be 'station', the
            scheme 'http[s]://' might be omitted and will default to 'http://'):
                https://www.mysite.org/fdsnws/dataselect/1
                https://www.mysite.org/fdsnws/dataselect/1/
                https://www.mysite.org/fdsnws/dataselect/1/query
        '''
        # do not use urlparse as we should import from stream2segment.url for py2 compatibility
        # but this will cause circular imports:

        obj = urlparse(url)
        if not obj.scheme:
            obj = urlparse('http://' + url)
        if not obj.netloc:
            raise ValueError('no domain specified, e.g. http://<domain>')

#         if obj.path in ('', '/'):
#             raise ValueError("Invalid path (/fdsnws/<service>/</majorversion>) in '%s'" % url)

        self.site = "%s://%s" % (obj.scheme, obj.netloc)

        pth = obj.path
        reg = re.match("^/(?:fdsnws)/(?P<service>.*?)/(?P<majorversion>.*?)(:?/query/*|/*)$",
                       pth)
        try:
            self.service, self.majorversion = reg.group('service'), reg.group('majorversion')
            if self.service not in [self.STATION, self.DATASEL, self.EVENT]:
                raise ValueError("Invalid <service> '%s' in '%s'" % (self.service, pth))
            try:
                float(self.majorversion)
            except ValueError:
                raise ValueError("Invalid <majorversion> '%s' in '%s'" % (self.majorversion, pth))
        except ValueError:
            raise
        except Exception:
            raise ValueError("Invalid FDSN path in '%s': it should be "
                             "'[site]/fdsnws/<service>/<majorversion>', "
                             "check potential typos" % str(url))

    def url(self, service=None, majorversion=None, method=None):
        '''builds a new url from this object url. Arguments which are 'None' will default
        to this object's url passed in the constructor. The returned url
        denotes the base url (with no query parameter and no trailing '?' or '/') in
        order to build queries to a fdsn web service

        :param service: None or one of this class static attributes:
            `STATION`, `DATASEL`, `EVENT`
        :param majorversion: None or numeric value or string parsable to number
            denoting the service major version. Defaults to 1 when None
            `STATION`, `DATASEL`, `EVENT`
        :param method: None or one of the class static attributes
            `QUERY` (the default when None), `QUERYAUTH`,
            `VERSION`, `AUTH` or `APPLWADL`
        '''
        return "%s/fdsnws/%s/%s/%s" % (self.site, service or self.service,
                                       str(majorversion or self.majorversion),
                                       method or self.QUERY)


@pytest.mark.parametrize(['url_'],
                         [
                          ('abc.org/fdsnws/station/1/query/',),
                          ('abc.org/fdsnws/station/1/query',),
                          ('abc.org/fdsnws/station/1',),
                          ('abc.org/fdsnws/station/1/',),])
def test_models_fdsn_url(url_):
    for url in [url_, 'http://' + url_, 'https://'+url_]:
        fdsn = Fdsnws(url)
        if url.startswith('https'):
            assert fdsn.site == 'https://abc.org'
        else:
            assert fdsn.site == 'http://abc.org'
        assert fdsn.service == Fdsnws.STATION
        assert fdsn.majorversion == '1'

        normalizedurl = fdsn.url()
        for service in [Fdsnws.STATION, Fdsnws.DATASEL, Fdsnws.EVENT, 'abc']:
            assert fdsn.url(service) == normalizedurl.replace('station', service)

        assert fdsn.url(majorversion=55) == normalizedurl.replace('1', '55')

        for method in [Fdsnws.QUERY, Fdsnws.QUERYAUTH, Fdsnws.APPLWADL, Fdsnws.VERSION,
                       'abcdefg']:
            assert fdsn.url(method=method) == normalizedurl.replace('query', method)


@pytest.mark.parametrize(['url_'],
                         [
                          ('',),
                          ('/fdsnws/station/1',),
                          ('fdsnws/station/1/',),
                          ('fdsnws/station/1/query',),
                          ('fdsnws/station/1/query/',),
                          ('abc.org',),
                          ('abc.org/',),
                          ('abc.org/fdsnws',),
                          ('abc.org/fdsnws/',),
                          ('abc.org/fdsnws/bla',),
                          ('abc.org/fdsnws/bla/',),
                          ('abc.org/fdsnws/station/a',),
                          ('abc.org/fdsnws/station/b/',),
                          ('abc.org//fdsnws/station/1.1/',),])
def test_models_bad_fdsn_url(url_):
    for url in [url_, 'http://' + url_, 'https://'+url_]:
        with pytest.raises(ValueError):
            fdsn = Fdsnws(url)
