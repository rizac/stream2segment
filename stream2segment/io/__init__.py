import re

try:  # py3:
    from urllib.parse import urlparse
except ImportError:  # py2
    from urlparse import urlparse  # noqa


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