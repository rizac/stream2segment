"""
Http requests with multi-threading
"""
# :date: Apr 15, 2017
from collections import deque
from dataclasses import dataclass
from threading import Condition, current_thread, main_thread, Lock, Event
import signal
import socket
import os
import ssl
from enum import IntEnum
from multiprocessing.pool import ThreadPool
from typing import Any

from urllib.parse import urlparse  # , urlencode
from urllib.error import HTTPError, URLError
from http.client import HTTPException, responses as builtin_responses
from urllib.request import (urlopen, build_opener, HTTPPasswordMgrWithDefaultRealm,
                            HTTPDigestAuthHandler, Request)


# https://docs.python.org/3/library/urllib.request.html#request-objects
def get_host(url: str | Request, default_scheme="https://") -> str:
    """
    Return the host (domain name, lower case, no user info, no port) from an
    urllib.request.Request object or URL str. Calling again this function with the
    hostname returned here will be consistent and return always the same result,
    (see default_scheme).

    get_host("https://geofon.gfz.de/fdsnws") -> "geofon.gfz.de"
    get_host("geofon.gfz.de") -> "geofon.gfz.de"
    """
    url = getattr(url, 'full_url', url)  # if Request, URL is in the `full_url` attr
    parsed = urlparse(url)
    if not parsed.scheme:
        parsed = urlparse(f'{default_scheme}{url}')
    return parsed.hostname


def _get_opener(base_url, user, password):
    """Return an opener to be used for downloading data with a given user and password.
    All arguments should be strings. For info see:


    :param base_url: the base url (domain name, or scheme+domain_name, if e.g. "http"
        and "https" have to be considered differently). The opener's password manager
        normalizes this internally and matches against request URLs).
        See also https://docs.python.org/3/howto/urllib2.html?utm_source=chatgpt.com#id5
    :param: user: string, the username
    :param password: the password

    :return: an urllib opener
    """
    # parsed_url = urlparse(url)
    # base_url = "%s://%s" % (parsed_url.scheme, parsed_url.netloc)
    # base_url = get_host(url, include_scheme=True)
    # handlers = []
    password_mgr = HTTPPasswordMgrWithDefaultRealm()
    password_mgr.add_password(None, base_url, user, password)
    return build_opener(HTTPDigestAuthHandler(password_mgr))
    # handlers.append(HTTPDigestAuthHandler(password_mgr))
    # return build_opener(*handlers)


# custom codes:

class CustomResponseCode(IntEnum):
    URL_ERROR = -1001
    TIMEOUT_ERROR = -1002
    GET_ADDR_INFO_ERROR = -1003
    CONNECTION_ERROR = -1004
    HTTP_EXC_ERROR = -1005
    SSL_ERROR = -1006


responses = dict(builtin_responses)

responses[CustomResponseCode.URL_ERROR] = "Network error"
responses[CustomResponseCode.TIMEOUT_ERROR] = "Server takes too long"
responses[CustomResponseCode.GET_ADDR_INFO_ERROR] = "Address resolution failure"
responses[CustomResponseCode.CONNECTION_ERROR] = "Server connection error"
responses[CustomResponseCode.HTTP_EXC_ERROR] = "HTTP response malformed"
responses[CustomResponseCode.SSL_ERROR] = "SSL/TLS handshake failure"


@dataclass(slots=True, frozen=True)
class Response:
    """
    Lightweight data class representing a Response object with two arguments:

    - data (`bytes` or `str` or `Exception`) is the response content, usually bytes or,
      if data read was decoded, `str`. If an exception was raised, data is the
      exception object (with __traceback__, __context__ and __cause__
      all set to None for performance reason). `data` can also be manually set to
      any type (e.g. to `dict` via `json.loads(response.data)`)
    - status_code (int) is the response HTTP status code (extended), including
      normal http_codes (if exception is an HTTPError) and custom codes that
      generally denote network-related errors (usually integers > 1000, see
      the module enum class `CustomResponseCode` for details)
    - request: The request (URL string or Request object) generating the Response
    """

    data: str | bytes | dict
    status_code: int
    request: str | Request

    @property
    def is_ok(self):
        return 200 <= self.status_code <= 299


def urlread(
    url, blocksize=-1, decode=None, timeout=None, opener=None, **kwargs
) -> Response:
    """Read and return data from the given `url` using Python `urllib.open`.
    Return the tuple `(data, error, status_code)` (see below for details)

    :param url: (str or ``urllib.request..Request`)
    :param blocksize: int, default: -1. The block size while reading, -1 means:
        read entire content at once
    :param decode: string or None, default: None. The string used for decoding (e.g.,
        'utf8'). If None, the result is a `bytes` object, otherwise `str`
    :param timeout: timeout parameter specifies a timeout in seconds for blocking
        operations like the connection attempt (if not specified, None or non-positive,
        the global default timeout setting will be used). This actually only works for
        HTTP, HTTPS and FTP connections.
    :param opener: a custom opener. When None (the default), the default urllib opener is
        used. See :func:`get_opener` for, e.g., creating an opener from a base url, user
        and password
    :param kwargs: optional arguments to be passed to the underlying python `urlopen`
        function. These arguments are ignored if a custom `opener` argument is provided

    :return: a Response object
    """
    try:
        ret = b''
        # set default for timeout: timeout in urlopen defaults to
        # socket._GLOBAL_DEFAULT_TIMEOUT so we unfortunately either pass it or skip it.
        # As we allow for non-negative numbers normalize it first to None. If None,
        # don't pass it to urlopen
        if timeout is not None and timeout > 0:
            kwargs['timeout'] = timeout

        if opener is None:
            open_conn = urlopen(url, **kwargs)
        else:
            open_conn = opener.open(url, **kwargs)

        with open_conn as conn:
            if blocksize < 0:
                ret = conn.read()
            else:
                while True:
                    buf = conn.read(blocksize)
                    if not buf:
                        break
                    ret += buf
        if decode:
            ret = ret.decode(decode)
        return Response(ret, conn.code, url)
    except HTTPError as exc:
        return Response(str(exc), exc.code, url)
    except HTTPException as exc:
        return Response(str(exc), CustomResponseCode.HTTP_EXC_ERROR, url)
    except URLError as err:
        code = CustomResponseCode.URL_ERROR
        if isinstance(err.reason, (socket.timeout, TimeoutError)):
            code = CustomResponseCode.TIMEOUT_ERROR
        elif isinstance(err.reason, (ConnectionError, ConnectionRefusedError)):
            code = CustomResponseCode.CONNECTION_ERROR
        elif isinstance(err.reason, ssl.SSLError):
            code = CustomResponseCode.SSL_ERROR
        elif isinstance(err.reason, socket.gaierror):
            code = CustomResponseCode.GET_ADDR_INFO_ERROR
        return Response(str(err), code, url)
    except (socket.timeout, TimeoutError) as err:
        return Response(str(err), CustomResponseCode.TIMEOUT_ERROR, url)
    except (ConnectionError, ConnectionRefusedError) as err:
        return Response(str(err), CustomResponseCode.CONNECTION_ERROR, url)
    except ssl.SSLError as err:
        return Response(str(err), CustomResponseCode.SSL_ERROR, url)
    except socket.gaierror as err:
        return Response(str(err), CustomResponseCode.GET_ADDR_INFO_ERROR, url)
    except socket.error as err:
        return Response(str(err), CustomResponseCode.URL_ERROR, url)
    except Exception as err:
        asd = 9  # FIXME REMOVE (only for debug)
        raise


def read_async(
    iterable,
    *,
    max_global_concurrency=None,
    max_concurrency=8,
    error_limit=25,
    consecutive_error_limit=10,
    blocksize=-1,
    decode=None,
    timeout=None,
    unordered=True,
    credentials=None,
    **kwargs
):
    """Download data asynchronously from different urls iteratively. Specifically
    designed for large downloads, handles concurrency (`threading.Pool`) globally
    and per URL-domain, stopping at specific errors iteratively received.

    For each item `obj` of iterable, this function yields the tuple
    `(obj: [Any], response [Response])`

    :param iterable: an iterable strings (URLs) or `Request` objects
    :param max_global_concurrency: integer or None (the default) denoting the max
        parallel downloads globally. This corresponds to the maximum worker (sub)
        threads used. When None, the threads allocated are relative to the machine CPU
        (should be around 16-32)
    :param max_concurrency: integer denoting the max parallel downloads per url domain.
        Defaults to 4. This parameter might be adjusted and decreased when
        `consecutive_error_limit` errors are returned
    :param error_limit: int denoting the error limit per-domain: if no download is
        successful for `error_limit` times, regardless of the error type, the downloads
        from that domain are suspended and nothing is yielded anymore. Default: 25
    :param consecutive_error_limit: int denoting the (same) error limit per-domain: if
        the same error type is returned for `consecutive_error_limit` times from the
        same url domain, the concurrency for that domain is decreased until it reaches
        0 (in that case, downloads from that domain are suspended and nothing is
        yielded anymore). Default: 10
    :param blocksize: integer defaulting to 1024*1024 specifying, when connecting to one
        of the given urls, the maximum number of bytes to be read at each call of
        `urlopen.read`. If the size argument is negative or omitted, read all data until
        EOF is reached
    :param decode: string or None (default: None) optional decoding (e.g., 'utf-8') to
        convert the result of the url request from `bytes` (the default) into `str`
    :param timeout: timeout parameter specifies a timeout in seconds for blocking
        operations like the connection attempt (if not specified, None or non-positive,
        the global default timeout setting will be used). This actually only works for
        HTTP, HTTPS and FTP connections.
    :param unordered: boolean (default True): tells whether the download results are
        yielded in the same order they are input in `iterable`. Theoretically (tests did
        not show any remarkable difference), False (the default) might execute faster,
        but results are not guaranteed to be yielded in the same order as `iterable`.
    :param credentials: credentials for downloading non-open data. It can be a tuple of
        two strings (user, password) or a dict of URL *domains* optionally prefixed with
        the http scheme (i.e., "https://mydomain.org" or "mydomain.org") mapped to a
        (user, password)
    :param kwargs: optional arguments to be passed to the underlying python `urlopen`
        function. These arguments are ignored if a custom `openers` function is provided

    killing threads / handling exceptions: this function handles any kind of unexpected
    exception (particularly relevant in case of e.g., `KeyboardInterrupt`) by canceling
    all worker threads before raising
    """
    if max_global_concurrency is None:
        max_global_concurrency = get_os_max_thread_count()

    max_concurrency = min(max_concurrency, max_global_concurrency)

    user, pswd, openers = None, None, None
    openers = {}
    if credentials is not None:
        if isinstance(credentials, tuple):
            user, pswd = credentials
        else:
            # Store just the domain name in openers, as we would do for (user, pswd):
            for k, user_pswd in credentials.items():
                # check that credentials does not hav conflicting keys (same domain
                # name, e.g. "geofon.de" and "https://geofon.de" and different passw.):
                base_url = get_host(k)
                if base_url in openers and openers[base_url] != user_pswd:
                    raise ValueError(f'Credentials conflict for {base_url}')
                openers[base_url] = _get_opener(base_url, *user_pswd)

    stop_event = None
    t_pool = None
    t_map = map

    concurrency_is_on = max_global_concurrency > 1

    if concurrency_is_on:
        # flag for CTRL-C or cancelled tasks
        stop_event = Event()

        def signal_handler(sig, frame):
            stop_event.set()

        signal.signal(signal.SIGINT, signal_handler)

        t_pool = ThreadPool(max_global_concurrency)
        t_map = t_pool.imap_unordered if unordered else t_pool.imap
        # note above: chunksize argument for threads (not processes)
        # seems to slow down download. Omit the argument and leave chunksize=1 (default)

    try:

        per_domain_lock = thread_lock_factory()
        aborted_download_domains = set()
        limiters: dict[str, DynamicLimiter] = {}

        def url_wrapper(url):
            if stop_event is not None and stop_event.is_set():
                return None
            # get the opener (restricted data):
            domain = get_host(url)  # noqa
            with per_domain_lock(domain):
                if domain in aborted_download_domains:
                    return None

                if pswd is not None:
                    opener = openers.setdefault(domain, _get_opener(domain, user, pswd))
                else:
                    opener = openers.get(domain, None)

                hostname_limiter = limiters.setdefault(
                    domain, DynamicLimiter(max_concurrency)
                )
                hostname_limiter.acquire()
                try:
                    resp = urlread(url, blocksize, decode, timeout, opener, **kwargs)
                finally:
                    hostname_limiter.release()

                return domain, resp

        last_n_errors = {}

        # perform download:
        for resp_tuple in t_map(url_wrapper, iterable):
            if stop_event is not None and stop_event.is_set():
                continue
            if resp_tuple is None:
                continue
            domain, response = resp_tuple

            if 200 <= response.status_code < 300:
                yield resp_tuple
                if domain in last_n_errors:
                    resp_queue = last_n_errors[domain]
                    while len(resp_queue):
                        yield resp_queue.pop()
                continue

            # error response. Append to queue:
            resp_queue = last_n_errors.setdefault(domain, deque(maxlen=error_limit))
            resp_queue.appendleft(resp_tuple)

            if len(resp_queue) < consecutive_error_limit:
                # threshold not yet reached, go on:
                continue

            if len({_.status_code for _ in resp_queue[:consecutive_error_limit]}) == 1:
                # same error got more than threshold. Decrease domain concurrency:
                new_limit = limiters[domain].adjust_limit(
                    -max(1, limiters[domain].limit // 2)
                )
                resp_queue.clear()
                if new_limit <= 0:
                    # cannot decrease further: discard domain downloads
                    with per_domain_lock(domain):
                        aborted_download_domains.add(domain)
                continue

            if len(resp_queue) >= error_limit:
                # too many errors (any error): discard domain downloads
                with per_domain_lock(domain):
                    aborted_download_domains.add(domain)
                resp_queue.clear()

        # yield suspended results:
        for domain, resp_queue in last_n_errors.items():
            if domain in aborted_download_domains:
                continue
            for obj, response in resp_queue:
                yield obj, response

    finally:
        if t_pool is not None:
            t_pool.close()
            t_pool.join()


def get_os_max_thread_count():
    """
    Return the maximum number of concurrent downloads adjusting the argument
    in order not to exceed the computer CPU
    """
    # Now adjust with the computer capacity (algorithm copied from
    # concurrent.futures.ThreadPoolExecutor):
    return min(32, os.cpu_count() + 4)


def thread_lock_factory():
    """Create a function F so that F(key:str) returns a unique key-based
    threading.Lock, meaning that calling the function with the same key again will
    return the same Lock"""

    thread_locks = {}
    global_lock = Lock()

    def get_thread_lock(domain):
        lock = thread_locks.get(domain)
        if lock is not None:
            return lock
        with global_lock:
            return thread_locks.setdefault(domain, Lock())

    return get_thread_lock


class DynamicLimiter:
    def __init__(self, limit: int):
        self.limit = limit
        self.active = 0
        self.cond = Condition()

    def adjust_limit(self, delta):
        with self.cond:
            new_limit = self.limit + delta
            # prevent invalid state
            if new_limit < 0:
                new_limit = 0
            self.limit = new_limit
            if new_limit > self.active:
                # wake up waiters in case capacity increased
                self.cond.notify_all()
        return new_limit

    def acquire(self):
        with self.cond:
            while self.active >= self.limit:
                self.cond.wait()
            self.active += 1

    def release(self):
        with self.cond:
            self.active -= 1
            self.cond.notify_all()


def _ismainthread():
    """Mainly uised for testing, returns True if we are currently executing in the
    mainv thread
    """
    # https://stackoverflow.com/q/23206787
    return current_thread() is main_thread()
