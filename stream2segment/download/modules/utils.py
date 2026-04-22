"""
Utilities for the download routine
"""
# date Nov 25, 2016
import os
import sys
import re
from io import StringIO
from datetime import datetime, date, timezone
import logging
from urllib.parse import urlencode, unquote, urlparse, urlunparse
from urllib.request import Request

import pandas as pd

from stream2segment.download.url import responses, get_host, urlread

# (https://docs.python.org/2/howto/logging.html#advanced-logging-tutorial):
logger = logging.getLogger(__name__)


def formatmsg(action=None, errmsg=None, url=None):  # FIXME remove
    """Format a message in order to have normalized message types across the
    program (e.g., in logging utilities). The argument can contain new
    (e.g., "{}") but also old-style format keywords (such as '%s', '%d') for
    usage within the logging functions, e.g.:
    `logging.warning(msg('%d segments discarded', 'no response'), 3)`.
    The resulting string message will be in any of the following formats
    (according to how many arguments are non-empty):
    ```
        "{action} ({errmsg}). url: {url}"
        "{action} ({errmsg})"
        "{action}"
        "{errmsg}. url: {url}"
        "{errmsg}"
        "{url}"
        ""
    ```
    :param action: string or None: what has been done
        (e.g. "discarded 3 events")
    :param errmsg: string or Exception: the Exception or error message which
        caused the action
    :param url: the url (string) or `urllib2.Request` object: the url
        originating the message, if the latter was issued from a web request
    """
    msg = action.strip()
    if errmsg:
        strerr = err2str(errmsg)
        msg = "{} ({})".format(msg, strerr) if msg else strerr
    if url:
        urlmsg = url2str(url, maxlen=200).strip()
        msg = "{}. url: {}".format(msg, urlmsg) if msg else urlmsg
    return msg


def err2str(err):
    """Return the string representation of `err`

    :param err: string or Exception denoting the error
    """
    # This class basically does two things: convert KeyErrors into
    # "KeyError: 'a'" and not simply "a",
    # and in case of exceptions which produce the empty string, return their
    # class name instead (e.g. socket.timeout returns 'timeout' instead of '')
    errclass = err.__class__
    if errclass == KeyError:
        return "%s: %s" % (str(errclass), str(err))
    if errclass == str:  # if we passed a string, just return it
        return err
    return (str(err) or str(errclass.__name__)).strip()


def url2str(obj, maxlen=None):
    """Convert an url or `urllib2.Request` object to string. In the latter
    case, the format is:
    "{obj.get_full_url()}" if `obj.data` is falsy
    "{obj.get_full_url()}, data: '{obj.get_data()}'"
    if `obj.data` has no newlines, or
    "{obj.get_full_url()}, data: '{obj.get_data()[:I]}'" otherwise
    (I=obj.get_data().find('\n')`)
    """
    # unquote removes % characters which might be confused with str interpolation
    try:
        url = unquote(str(obj.get_full_url()))
        data = str(obj.data or '')
        if data:
            url = "%s, POST data:\n%s" % (url, data)
    except AttributeError:
        url = unquote(str(obj))
    if maxlen is not None and len(url) > maxlen + 10:
        url = url[:maxlen] + \
               f"... ({len(url) - maxlen:,} remaining characters not shown)"
    return url


class IdOnceLogFilter(logging.Filter):
    """
    logging Filter that expects an 'ID' attribute on each record
    and filters out already processed record (comparing by same 'ID').

    # Usage:

    log_filter = IdOnceLogFilter()
    logger.addFilter(log_filter)
    logger.warn('message', extra={'ID': (1, 'geofon.gfz.de')})  # logged
    logger.warn('another message', extra={'ID': (1, 'geofon.gfz.de')})  # not logged
    # eventually, you can optionally remove the filter (freeing memory):
    logger.removeFilter(log_filter)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processed_ids = set()

    def filter(self, record):
        _id = getattr(record, "ID", None)
        if _id is None:
            return True

        if _id in self.processed_ids:
            return False

        self.processed_ids.add(_id)
        return True


def fdsn_response_text_to_df(response: str):
    """
    Convert a response content obtained from a FDSN webservice with format=text
    into a pandas DataFrame of type str (no casting performed)
    """
    # read csv but do not let pandas infer data types (`_harmonize_columns` does that
    # later): `dtype=str` reads everything as strings (this prevents `event_id`s in
    # catalogs to be inadvertently casted as int), `na_values` + `keep_default_na` reads
    # empty cells as "" (this prevents channels `location` to be NULL and the relative
    # row to be dropped in `_harmonize_columns` because NULL is not allowed)
    return pd.read_csv(
        StringIO(response),
        sep='|',
        header=None,
        comment='#',
        dtype=str,
        keep_default_na=False,
        na_values=[
            '#N/A', '#NA', '-NaN', '-nan', '<NA>', 'N/A',
            'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null'
        ]
    )


EVENTWS_MAPPING = {
    'emsc':  'http://www.seismicportal.eu/fdsnws/event/1/query',
    'isc':   'http://www.isc.ac.uk/fdsnws/event/1/query',
    'iris':  'http://service.iris.edu/fdsnws/event/1/query',
    'ncedc': 'http://service.ncedc.org/fdsnws/event/1/query',
    'scedc': 'http://service.scedc.caltech.edu/fdsnws/event/1/query',
    'usgs':  'http://earthquake.usgs.gov/fdsnws/event/1/query',
}


EVENTWS_SAFE_PARAMS = ['minlatitude', 'minlat', 'maxlatitude', 'maxlat',
                       'minlongitude', 'minlon', 'maxlongitude', 'maxlon',
                       'minmagnitude', 'minmag', 'maxmagnitude', 'maxmag',
                       'mindepth', 'maxdepth']


class Authorizer(dict[str, tuple[str, str]]):
    """Class handling authorization/authentication. It subclasses dict
    returns """

    def __init__(self, token):
        """Initialize a new Authorizer, a class handling authorization and
        authentication for restricted data

        :param token: a filepath (to a token), the token data (bytes),
            or a tuple (username, password). If None, this authorizer is no-op
        """
        super().__init__()
        self._token = None
        self.user_pass = None, None
        token_file = None
        if isinstance(token, (tuple, list)):
            if len(token) != 2 or not all(isinstance(_, str) for _ in token):
                raise ValueError('provide username and password as '
                                 'list/tuple of two strings')
            self._user_pass = tuple(token)
        else:
            # check if there's a local file that matches the provided str
            token_file = token if os.path.isfile(token) else None
            if token_file is not None:
                with open(token_file, 'rb') as fhd:
                    self._token = fhd.read()
            valid_eida_token = re.search(
                pattern=rb'\bBEGIN PGP\b', string=self._token, flags=re.IGNORECASE
            )
            if not valid_eida_token:
                raise ValueError("Invalid token. "
                                 "If you passed a file path, "
                                 "check that the file is a valid token")

    def add_url(self, url):
        """Adds the given url as restricted data download

        :param url: an FDSN url (method and query majorversion will be ignored)
        """
        # hostname = get_host(url)
        if self._token:
            req = Request(
                fdsn_url(url, new_service='dataselect', new_method='auth'),
                data=self._token
            )
            response = urlread(req)
            if response.error:
                raise response.error
            data = response.data
            if ':' not in data:
                raise ValueError('Invalid user and password returned. '
                                 'This could be a data-center bug')
            else:
                user_pass = tuple(data.split(':'))
        else:
            user_pass = self.user_pass
        self[get_host(url, include_scheme=True)] = user_pass

    # @staticmethod   # FIXME REMOVE
    # def _validate_eida_token(token):
    #     """Along the lines of ObsPy: basic check to test that a token is ok"""
    #     if re.search(pattern=r'\bBEGIN PGP\b', string=token,
    #                  flags=re.IGNORECASE):  # @UndefinedVariable
    #         return True
    #     return False
    #
    # @property
    # def token(self):
    #     """Return the token (as bytes), or None. You can safely use this method
    #     also in an if statement: `if auth.token`, as the token can not be empty
    #     """
    #     return self._token
    #
    # @property
    # def userpass(self):
    #     """Return the tuple (user, password), or None, You can safely use
    #     this method also in an if statement: `if auth.userpass`
    #     """
    #     if (self._uname, self._pswd) == (None, None):
    #         return None
    #     return self._uname, self._pswd


def strptime(obj):
    """Convert `obj` to a `datetime` object **in UTC without tzinfo** (if the datetime
    is timezone aware, it will be converted to UTC and then its tzinfo removed).

    :param obj: datetime, date or datetime-string string in ISO format

    :return: a datetime object in UTC, with the tzinfo removed
    :raise: TypeError or ValueError
    """
    dtime = obj
    if isinstance(obj, str):
        dtime = _fromisoformat(obj)

    if not isinstance(dtime, datetime):
        if isinstance(dtime, date):  # note: check here (a datetime is also a date!)
            dtime = datetime(year=dtime.year, month=dtime.month, day=dtime.day)
        else:
            raise TypeError(f'string or datetime required, found {type(obj)}')

    if dtime.tzinfo is not None:
        # if a time zone is specified, convert to utc and remove the timezone
        dtime = dtime.astimezone(timezone.utc).replace(tzinfo=None)

    # the datetime has no timezone provided AND is in UTC:
    return dtime


if sys.version_info[0] == 3 and sys.version_info[1] < 11:
    import dateutil.parser

    def _fromisoformat(string):
        """fix py<3.11 datetime.fromisoformat where, e.g. microseconds given not in
        6 digits would raise. Use dateutil for that"""
        # https://stackoverflow.com/a/15228038
        try:
            return dateutil.parser.isoparse(string)
        except ValueError:  # make msg consistent with datetime.fromisoformat:
            raise ValueError(f"Invalid isoformat string: '{string}'")
else:
    def _fromisoformat(string):
        return datetime.fromisoformat(string)


def fdsn_url(
    url: str, new_service: str = None, new_method: str = None, check_scheme=True
):
    """Check that the given url is a valid FDSN URL and return it (with new service and
    method substrings, if given). Raise ValueError if the url is invalid. The URL query
    string, if present, will not be checked and returned as it is

    :param url: a valid FDSN url, with or without query string or schema
    :param new_service: the new service. None will leave the service of `url`. Must be
        a string in ('station', 'dataselect', 'event')
    :param new_method: the new method, None will leave the url method. Must be a string
        in ('query', 'queryauth', 'auth', 'version', 'application.wadl')
    :param check_scheme: if True (the default) check that the url is prefixed with a
        scheme (e.g. https://), raising ValueError if missing. If False, urls can also
        miss the schema
    """
    services = {'station', 'dataselect', 'event'}
    if new_service is not None and new_service not in services:
        raise ValueError(f'Invalid argument service: {new_service}')

    methods = {'query', 'queryauth', 'auth', 'version', 'application.wadl'}
    if new_method is not None and new_method not in methods:
        raise ValueError(f'Invalid argument method: {new_method}')

    parsed_url = urlparse(url)
    if not parsed_url.scheme and check_scheme:
        raise ValueError('url starts with no scheme, e.g. "https://")')

    if not parsed_url.netloc:
        raise ValueError('url has no valid domain, e.g. "geofon.gfz.de"')

    path = parsed_url.path
    # urlparse has already removed query char '?' and params and fragment
    # from the path (which starts with '/'). Now check the path:
    reg = re.match(
        "^/fdsnws/(?P<service>[^/]+)/(?P<majorversion>[^/]+)/(?P<method>.+)$", path
    )

    if not reg:
        raise ValueError('url has no path "fdsnws/<service>/<majorversion>/<method>')

    service = reg.group('service')
    if service not in services:
        raise ValueError(f"Invalid service in url: {service}")

    majorversion = reg.group('majorversion')
    try:
        float(majorversion)
    except ValueError:
        raise ValueError(f"Invalid major version in url: {majorversion}")

    method = reg.group('method')
    if method not in methods:
        raise ValueError(f"Invalid method in url: {method}")

    change_service = new_service is not None and new_service != service
    change_method = new_method is not None and new_method != method
    if change_service or change_method:
        path2 = f'/fdsnws/{new_service}/{majorversion}/{new_method}'
        new_parsed = parsed_url._replace(path=path2)
        url = urlunparse(new_parsed)

    return url


def fdsn_url_qs(base_url: str, **query_args):
    """Build a valid FDSN URL appending to `base_url` the query string (qs) built
    from the FDSN parameters in `query_args`.

    Note: any query parameter (keys of `query_args`) mapped to None will be removed.
    Any other value will be encoded using its string representation, except for the
    following parameters:
    - net, sta, loc, cha (and their long-name variants, e.g. network)
      will be ignored if '*'.
    - start, end (and their long-name variants) will be converted to ISO format strings
      if Python datetime or date
    Duplicates (e.g. 'mag', 'magnitude') are not checked for and will be both encoded,
    whether the encoded URL is valid depends on the server, but in principle it
    should be rejected
    """
    qs = {}
    # date and time params:
    d_params = {
        'start', 'starttime', 'end', 'endtime', 'startbefore', 'startafter',
        'endbefore', 'endafter'
    }
    # special text params (needing * treated specially):
    c_params = {'net', 'network', 'sta', 'station', 'cha', 'channel', 'loc', 'location'}
    # Numeric params. Keep track of them just for ref (maybe needed in future):
    n_params = {
        'lat', 'latitude', 'minlat', 'minlatitude', 'maxlat', 'maxlatitude',
        'lon', 'longitude', 'minlon', 'minlongitude', 'maxlon', 'maxlongitude',
        'mag', 'magnitude', 'minmag', 'minmagnitude', 'maxmag', 'maxmagnitude',
        'mindepth', 'maxdepth', 'minradius', 'maxradius'
    }
    safe_chars = set()  # avoid encoding these chars (improve debug and copy/paste url)
    for k, v in query_args.items():
        if v is None or (k in c_params and v.strip() == '*'):
            continue
        if k in d_params and isinstance(v, (date, datetime)):
            if isinstance(v, date):
                v = datetime(v.year, v.month, v.day)
            v = v.isoformat('T')
            safe_chars.update(set('-:') & set(v))  # don't encode ":-" if in v
        else:
            v = str(v)
            if k in c_params:
                safe_chars.update(set(',?*') & set(v))  # don't encode ',?*' if in v
            elif k in n_params:
                safe_chars.update(set('.') & set(v))  # don't encode '.' if in v

        qs[k] = v
    return f'{base_url}?{urlencode(qs, safe="".join(safe_chars))}'
