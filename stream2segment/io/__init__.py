import re

import yaml

try:  # py3:
    from urllib.parse import urlparse
except ImportError:  # py2
    from urlparse import urlparse  # noqa

# this can not apparently be fixed with the future package:
# The problem is io.StringIO accepts unicode in python2 and strings in Py3:
import sys
PY2 = sys.version_info[0] == 2
if PY2:
    from cStringIO import StringIO  # noqa
else:
    from io import StringIO  # noqa


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


def yaml_safe_dump(data, stream=None, default_flow_style=False,
                   sort_keys=False, **kwds):
    """Call `yaml.safe_dump` with default shortcuts

    :param default_flow_style: boolean, tells if collections (lists/dicts) should
        to serialized in the flow style, i.e. `b: {c: 3, d: 4}`. False by default
    :param sort_keys: whether to sort keys (param names). Defaults to False. Might
        not work in PyYAML version < 5.1 (in that case, it is ignored)
    :return: None (if stream is not None). **If stream is None, returns
        the produced string instead**
    """
    kwds['default_flow_style'] = default_flow_style
    kwds['sort_keys'] = sort_keys
    try:
        return yaml.safe_dump(data, stream, **kwds)
    except TypeError:
        # we might have a PyYaml version < 5.1 where sort_keys was not
        # supported: try to remove the argument. Note however that in that
        # case safe_dump will sort dict keys
        kwds.pop('sort_keys', None)
        return yaml.safe_dump(data, stream, **kwds)