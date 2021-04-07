import re
from os.path import join, normpath, isabs, isdir, abspath, dirname

import yaml
from future.utils import PY2, string_types, viewkeys, viewitems

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

# http://python-future.org/imports.html#explicit-imports
from builtins import open as compatible_open  # py2+3


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
    # URL path string denoting a station service ('station')
    STATION = 'station'
    # URL path string denoting a data service ('dataselect')
    DATASEL = 'dataselect'
    # URL path string denoting an event service ('event')
    EVENT = 'event'
    # URL path string denoting a query method ('query')
    QUERY = 'query'
    # URL path string denoting a query with authorization method ('queryauth')
    QUERYAUTH = 'queryauth'
    # URL path string denoting an authorization method ('auth')
    AUTH = 'auth'
    # URL path string denoting a version method ('version')
    VERSION = 'version'
    # URL path string denoting an application.wadl method ('application.wadl')
    APPLWADL = 'application.wadl'

    # all services, as frozenset (non optimal iteration speed, but faster search)
    SERVICES = frozenset([STATION, DATASEL, EVENT])

    # all services, as frozenset (non optimal iteration speed, but faster search)
    METHODS = frozenset([QUERY, QUERYAUTH, AUTH, VERSION, APPLWADL])

    def __init__(self, url, strict_path=True):
        """Initialize a Fdsnws object from a FDSN URL:

        [site]/fdsnws/<service>/<majorversion>/

        E.g.:

        http://www.mysite.org/fdsnws/station/1/query

        :param url: string denoting the Fdsn web service url. The scheme (e.g.,
            'https://') might be omitted and will default to 'http://'. An ending
            '/' or '?' will be ignored, if present
        :param strict_path: boolean (default True) whether the URL path should start
            with "/fdsnws". If False, paths are allowed before "/fdsnws", so
            "http://www.mysite.org/mypath/fdsnws/station/1/query" would be valid
        """
        obj = urlparse(url)
        if not obj.scheme:
            obj = urlparse('http://' + url)
        if not obj.netloc:
            raise ValueError('no domain specified or invalid scheme, '
                             'check typos')

        self.site = "%s://%s" % (obj.scheme, obj.netloc)

        pth = obj.path
        # urlparse has already removed query char '?' and params and fragment
        # from the path (which starts with '/'). Now check the path:
        reg = re.match("^(?P<path_prefix>.*/)fdsnws/(?P<service>[^/]+)/"
                       "(?P<majorversion>[^/]+)(?P<method>.*)$",
                       pth)

        try:
            self.service = reg.group('service')
            self.majorversion = reg.group('majorversion')
            method = reg.group('method')

            if self.service not in self.SERVICES:
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
                if method and method not in self.METHODS:
                    raise ValueError("Invalid method '%s' in '%s'" %
                                     (method, pth))

            # check path_prefix at end:
            path_pre = reg.group('path_prefix')
            if path_pre and path_pre[-1] == '/':
                path_pre = path_pre[:-1]
            if path_pre:
                if strict_path:
                    raise ValueError('Invalid "%s" before "fdsnws"' % path_pre)
                # add a slash to path if missing, and add path to self.site:
                if path_pre[0] != '/':
                    path_pre = '/' + path_pre
                self.site += path_pre
        except Exception as exc:
            raise ValueError("FDSN URL error: %s (URL: %s)" %
                             (str(exc).lower(), str(url)))

    def _find_str(self, tokens_list):
        try:
            return
        except Exception as exc:
            raise

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


def yaml_safe_dump(data, stream=None, default_flow_style=False, sort_keys=False, **kwds):
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


def open2writetext(file, **kw):
    """Python 2+3 compatible function for writing **text** files with `str`
    types to file (i.e., object of `<type str>` in *both* python2 and 3).
    This function should be used with the csv writer or when we provide an
    input string which is `str` type in both python2 and 3 (e.g., by writing
    a = 'abc'). This function basically returns the python3 `open` function
    where the 'mode' argument is 'wb' in Python2 and 'w' in Python3. In the
    latter case, 'errors' and 'encoding' will be removed from `kw`, if any,
    because not compatible with 'wb' mode.
    Using `io.open(mode='w',..)` in py2 and `open(mode='w', ...)` in py3
    provides compatibility across function **signatures**, but the user must
    provide `unicodes` in python2 and `str` in py3. If this is not the case
    (e.g., we created a string such as a="abc" and we write it to a file, or we
    use the csv module) this function takes care of using the correct 'mode' in
    `open`

    :param file: the file. It is the first argument of the builtin `open`
        function
    :param kw: keyword arguments as for the python3 open function. 'mode' will
        be replaced if present ('wb' for Python2, 'w' for Python 3). An
        optional 'append' argument (True or False) will ad 'a' to the 'mode'
        (i.e., 'wba' for Python2, 'wa' for Python 3). If python2, 'encoding',
        'newline' and 'errors' will be removed as not compatible with the 'wb'
        mode (they raise if present)
    :return: the python3 open function for writing `str` types into text file
    """
    append = kw.pop('append', False)
    if PY2:
        kw.pop('encoding', None)
        kw.pop('errors', None)
        kw.pop('newline', None)
        kw['mode'] = 'wb'
    else:
        kw['mode'] = 'w'
    if append:
        kw['mode'] = kw['mode'].replace('w', 'a')
    return compatible_open(file, **kw)


def yaml_load(filepath, **updates):
    """Load a yaml file into a dict (if `filepath` is a `dict`, skips loading).
    Then:
    1. If `filepath` denotes a file path (and not a dict),
       normalizes non-absolute sqlite path values relative to `filepath`, if any
    2. updates the dict values with `updates` and returns the yaml dict. The
       update is recursive, meaning that nested dict values will be updated
       recursively and not completely overwritten. Example:
       param. a = {'b': 1, 'c' :2}, updates a = {'b': 2, 'd' :3} =>
       result = {'b': 2, 'c': 2, 'd': 3}

    :param filepath: str, dict or file-like object. If str, it must denote a
        path to an existing .yaml file
    :param updates: arguments which will updates the yaml dict before it is
        returned
    """
    if isinstance(filepath, string_types):
        with open(filepath, 'r') as stream:
            ret = yaml.safe_load(stream)
    elif isinstance(filepath, dict):
        ret = filepath
    elif hasattr(filepath, 'read'):
        ret = yaml.safe_load(filepath)
    else:
        raise TypeError('required file path (string), file object or dict, '
                        '%s found' % str(type(filepath)))

    # update recursively (which means sub-dicts are updated as well and not
    # overwritten):
    def update(dic1, dic2):
        """update dic1 with dic2 recursively"""
        # Terminology: If the same key exists in both dicts and is mapped to
        # two dictionaries (not necessarily equal), the latter are called
        # "shared dicts"

        # 1. Move shared dicts from `dic2` in a temporary dictionary 'dickeys':
        dickeys = {k: dic2.pop(k) for k in viewkeys(dic1)
                   if isinstance(dic1[k], dict) and
                   isinstance(dic2.get(k, None), dict)}
        # 2. `dic1` and `dic2` have no shared dicts, update dicts "normally":
        dic1.update(dic2)
        # 3. Update shared dicts recursively:
        for k in dickeys:
            update(dic1[k], dickeys[k])

    update(ret, updates)

    if isinstance(filepath, string_types):
        # convert relative sqlite path to absolute, assuming they are relative
        # to the config:
        sqlite_prefix = 'sqlite:///'
        # we cannot modify a dict while in iteration, thus create a new dict of
        # possibly modified sqlite paths and use later dict.update
        newdict = {}
        for key, val in viewitems(ret):
            try:
                if val.startswith(sqlite_prefix) and ":memory:" not in val:
                    dbpath = val[len(sqlite_prefix):]
                    npath = absrelpath(dbpath, filepath)
                    if npath != dbpath:
                        newdict[key] = sqlite_prefix + npath
            except AttributeError:
                pass

        ret.update(newdict)
    return ret


def absrelpath(path, base):
    """Normalize `path` with respect to `base`, returning the absolute path
    by joining `base` and `path`. If `path` is already absolute, returns it as it is

    :param path: the path
    :param base: the base directory path. If file, `dirname(file)` will be
        used
    """
    if isabs(path):
        return path
    if not isdir(base):
        base = dirname(base)
    return abspath(normpath(join(base, path)))