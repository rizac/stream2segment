"""
Http requests with multi-threading

:date: Apr 15, 2017

.. moduleauthor:: <rizac@gfz-potsdam.de>
"""
from itertools import chain
from threading import Semaphore, current_thread, main_thread, Lock, Event
import signal
import socket
import os
import ssl
from enum import IntEnum
from multiprocessing.pool import ThreadPool

from urllib.parse import urlparse  # , urlencode
from urllib.error import HTTPError, URLError
from http.client import HTTPException, responses as builtin_responses
from urllib.request import (urlopen, build_opener, HTTPPasswordMgrWithDefaultRealm,
                            HTTPDigestAuthHandler)


# https://docs.python.org/3/library/urllib.request.html#request-objects
def get_host(url_or_request, include_scheme=False) -> str:
    """Returns the host (domain name, lower case) from a urllib.request.Request object
    or str (URL). If hostname is not found, return the full url "untouched"
    get_host("https://GEOFON.de/fdsnws") -> "geofon.de"
    get_host("https://GEOFON.de/fdsnws", True) -> "https://geofon.de"
    get_host("geofon.de.invalid_url") -> "geofon.de.invalid_url"
    """
    # Handle both url as Request obj. (use attr. host) or string (use urlparse):
    url = getattr(url_or_request, 'full_url', url_or_request)
    parsed = urlparse(url)
    if not parsed.hostname:
        return url
    if include_scheme and parsed.scheme:  # scheme is present
        return f"{parsed.scheme}://{parsed.hostname}"
    return parsed.hostname  # noqa


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
    URL_ERROR = 1001
    TIMEOUT_ERROR = 1002
    GET_ADDR_INFO_ERROR = 1003
    CONNECTION_REFUSED_ERROR = 1004
    HTTP_EXC_ERROR = 1005
    SSL_ERROR = 1006
    DOWNLOAD_SUSPENDED = 1007


responses = dict(builtin_responses)

responses[CustomResponseCode.URL_ERROR] = \
    "Catch-all network error not matching any known error"
responses[CustomResponseCode.TIMEOUT_ERROR] = \
    "Server takes too long to respond (connection or read timeout)"
responses[CustomResponseCode.GET_ADDR_INFO_ERROR] = \
    "Hostname/address resolution failure (e.g., DNS)"
responses[CustomResponseCode.CONNECTION_REFUSED_ERROR] = \
    "Server actively refuses the connection (e.g., port closed, server down)"
responses[CustomResponseCode.HTTP_EXC_ERROR] = \
    "HTTP response malformed or incomplete (bad headers, truncated data)"
responses[CustomResponseCode.SSL_ERROR] = \
    "SSL/TLS handshake failure (bad certificate, hostname mismatch, expired cert)"
responses[CustomResponseCode.DOWNLOAD_SUSPENDED] = \
    "Too many identical failures, download suspended"


class Response:
    """lightweight data class representing a Response object with three arguments:

    - data (`bytes` or `str`) is the response content. If `decode` is given,
      it is a `str`. It is None in case of request/response error (see `error` below)
    - error: the response error in form of Python exception raised (either
      HTTPException, URLError, HTTPError). Always None if the request/response
      exchange was successful
    - status_code (int) is the response HTTP status code (extended), including
      normal http_codes (if exception is an HTTPError) and custom codes that
      generally denote network-related errors. These codes, usually integers > 1000
      are available as items of the module enum class `CustomResponseCode`, their
      explanation is available using the global variable `responses` that extends
      `http.client.responses`
    """
    __slots__ = ('data', 'error', 'status_code')

    def __init__(self, data, error, status_code: int):
        self.data = data
        self.status_code = status_code
        self.error = error

    @property
    def is_ok(self):
        return self.data is not None


def urlread(
        url, blocksize=-1, decode=None, timeout=None, opener=None, **kwargs
) -> Response:
    """Read and return data from the given `url` using Python `urllib.open`.
    Return the tuple `(data, error, status_code)` (see below for details)

    :param url: (str or ``urllib.request..Request`)
    :param blocksize: int, default: -1. The block size while reading, -1 means:
        read entire content at once
    :param: decode: string or None, default: None. The string used for decoding (e.g.,
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
            if blocksize < 0:  # https://docs.python.org/2.4/lib/bltin-file-objects.html
                ret = conn.read()
            else:
                while True:
                    buf = conn.read(blocksize)
                    if not buf:
                        break
                    ret += buf
        if decode:
            ret = ret.decode(decode)
        return Response(ret, None, conn.code)
    except HTTPError as exc:
        return Response(None, exc, exc.code)
    except URLError as u_err:
        code = CustomResponseCode.URL_ERROR
        if isinstance(u_err.reason, (socket.timeout, TimeoutError)):
            code = CustomResponseCode.TIMEOUT_ERROR
        elif isinstance(u_err.reason, socket.gaierror):
            code = CustomResponseCode.GET_ADDR_INFO_ERROR
        elif isinstance(u_err.reason, ConnectionRefusedError):
            code = CustomResponseCode.CONNECTION_REFUSED_ERROR
        elif isinstance(u_err.reason, ssl.SSLError):
            code = CustomResponseCode.SSL_ERROR
        return Response(None, u_err, code)
    except HTTPException as h_exc:
        # (socket.error is the superclass of all socket exc)
        return Response(None, h_exc, CustomResponseCode.HTTP_EXC_ERROR)


def read_async(
        iterable,
        url_callback=None,
        max_concurrency=None,
        max_concurrency_per_domain=8,
        slowdown=True,
        slowdown_trigger=3,
        suspend_trigger=25,
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

    :param iterable: an iterable of objects representing the urls addresses to be read:
        if its elements are neither strings nor `Request` objects, the `url_callback`
        argument must be specified to map each element to a valid url string or Request
    :param url_callback: function or None. When None (the default), all elements of
        `iterable` must be url strings or Request objects. If callable, it will be
        called with each element of `iterable` as argument, and must return the mapped
        url address or Request.
    :param max_concurrency: integer or None (the default) denoting the max parallel
        downloads. This corresponds to the maximum worker (sub) threads used. When None,
        the threads allocated are relative to the machine CPU (should be around 16-32)
    :param max_concurrency_per_domain: integer (default: 8) denoting the max parallel
        downloads per domain. Setting this value >0 means that - while `max_concurrency`
        will allow a certain number of parallel downloads *globally*, you will be assured
        that at most *max_concurrency_per_domain* will be from the same URL domain,
        possibly avoiding errors due to concurrent requests limit configured on the
        servers
    :param slowdown: True to enable slowing down concurrent downloads for those domains
        consistently returning the same download error at least `slowdown_trigger` times
        (download from other domains are not affected): `max_concurrency_per_domain` will
        be halved until it is greater than 1 (1 basically denoting sequential download).
        This parameter can be also a tuple of integers with the download error codes that
        should be considered: whereas some errors are more likely to indicate too many
        requests (i.e., `(429, 500, 503, CustomResponseCode.TIMEOUT_ERROR)`) setting this
        parameter to True is safer. If this parameter is the empty tuple or None,
        no halving is applied and `max_concurrency_per_domain` will stay constant
    :param slowdown_trigger: the number of consecutive errors that have to be
        received per domain to trigger halving of `max_concurrency_per_domain`.
        Defaults to 3, ignored if `slowdown` is empty or None
    :param suspend_trigger: int denoting the maximum downloads from the same
        domain if the same error is repeatedly returned by the server. Default: 25.
        After that, the tuple `(None, exc, CustomResponseCode.DOWNLOAD_SUSPENDED)`
        will be returned (`exc` is the Exception returned bu the last error response)
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
    :param unordered: boolean (default False): tells whether the download results are
        yielded in the same order they are input in `iterable`. Theoretically (tests did
        not show any remarkable difference), False (the default) might execute faster,
        but results are not guaranteed to be yielded in the same order as `iterable`.
    :param credentials: credentials for downloading non-open data. It can be a tuple of
        two strings (user, password) or a dict of URL *domains* optionally prefixed with
        the http scheme (i.e., "https://mydomain.org" or "mydomain.org") mapped to a
        (user, password)
    :param kwargs: optional arguments to be passed to the underlying python `urlopen`
        function. These arguments are ignored if a custom `openers` function is provided

    Implementation details:

    ThreadPool vs ThreadPoolExecutor: this function changed from using
    `concurrent.futures.ThreadPoolExecutor` into the "old"
    `multiprocessing.pool.ThreadPool`: the latter consumes in most cases less memory
    (about 30% less), especially if `iterable` is not a list in memory but a python
    iterable (`concurrent.futures.ThreadPoolExecutor` builds a `set` of `Future`s object
    from `iterable`, whereas `multiprocessing.pool.ThreadPool` seems just to execute each
    element in iterable)

    killing threads / handling exceptions: this function handles any kind of unexpected
    exception (particularly relevant in case of e.g., `KeyboardInterrupt`) by canceling
    all worker threads before raising
    """
    max_concurrency = adjust_max_concurrent_downloads(max_concurrency)
    max_concurrency_per_domain = min(max_concurrency_per_domain, max_concurrency)

    user, pwsd, openers = None, None, None
    if credentials is not None:
        if isinstance(credentials, tuple):
            user, pswd = credentials
            openers = {}
        else:
            # Store just the domain name in openers, as we would do for (user, pswd):
            openers = {}
            for k, v in credentials.items():
                # check that credentials does not hav conflicting keys (same domain
                # name, e.g. "geofon.de" and "https://geofon.de" and different passw.):
                base_url = get_host(k)
                if base_url in openers and openers[base_url] != v:
                    raise ValueError(f'Credentials conflict for {base_url}')
                openers[base_url] = v

    def prepare_url_read_args(obj) -> tuple:
        url = obj
        if url_callback is not None:
            url = url_callback(obj)
        host = get_host(url)
        opener = None
        if openers is not None:
            if pswd is not None:
                opener = openers.setdefault(host, _get_opener(host, user, pswd))
            else:
                opener = openers.get(host, None)
        return url, opener, host

    stop_event = None
    t_pool = None
    t_map = map

    concurrency_is_on = max_concurrency > 1
    concurrency_per_domain_is_on = concurrency_is_on and max_concurrency_per_domain > 0

    if concurrency_is_on:
        # flag for CTRL-C or cancelled tasks
        stop_event = Event()

        def signal_handler(sig, frame):
            stop_event.set()

        signal.signal(signal.SIGINT, signal_handler)

        t_pool = ThreadPool(max_concurrency)
        t_map = t_pool.imap_unordered if unordered else t_pool.imap
        # note above: chunksize argument for threads (not processes)
        # seems to slow down download. Omit the argument and leave chunksize=1 (default)

    try:

        domain_state = {}

        semaphore_lock = thread_lock_factory()
        skip_response_lock = thread_lock_factory()

        # set slodown codes:
        if slowdown is True:
            slowdown = {
                k for k in responses.keys() if 400 <= k < 600 or k in CustomResponseCode
            }
            slowdown.remove(CustomResponseCode.DOWNLOAD_SUSPENDED)
        else:
            slowdown = set(slowdown or [])

        class DomainState:
            """Memory-efficient container handling download state for a URL domain"""

            # define class-level attrs:
            _slowdown_codes: slowdown
            _slowdown_trigger = slowdown_trigger
            _max_concurrency = max_concurrency_per_domain
            _suspend_trigger = suspend_trigger

            # memory efficient container for a domain state
            # (better than dict, might use dataclass but keep it simple)
            __slots__ = (
                'semaphore', 'semaphore_limit', 'fail_count', 'fail_code',
                'skip_response', 'retry_downloads', 'pending_responses'
            )

            def __init__(self):
                self.semaphore = Semaphore(self._max_concurrency)
                self.semaphore_limit = self._max_concurrency
                self.fail_code = 0
                self.fail_count = 0
                self.skip_response = None
                self.retry_downloads = []
                self.pending_responses = []

            def update(
                    self,
                    semaphore: Semaphore,
                    domain: str,
                    obj,
                    response: Response
            ):
                """Update the current state and yield the downloaded data in form
                 of (obj, response) tuples, depending on the current state.
                 Calling this method assumes that `self.skip_response is None`
                """
                if semaphore is None:
                    # either no concurrency_per_domain,
                    # or download suspended (skip_response is set)
                    yield obj, response
                    return

                code = response.status_code
                if semaphore is not self.semaphore:
                    # we slowed down, and the passed semaphore is an old one
                    if code not in self._slowdown_codes:
                        # if download is ok, yield:
                        yield obj, response
                    else:
                        # download not ok, tricky case: for safety, retry later:
                        self.retry_downloads.append(obj)
                    return

                if code not in self._slowdown_codes:
                    # download is ok (no slowdown potential issues)
                    # yield current response:
                    yield obj, response
                    # reset variables only if we have slodown codes (otherwise skip):
                    if self._slowdown_codes:
                        self.fail_count = 0
                        self.fail_code = 0
                        # all pending responses can be yielded:
                        yield from self.pending_responses
                        self.pending_responses.clear()
                    return

                if code != self.fail_code:
                    # status code denotes a different error than previous
                    self.fail_count = 1
                    self.fail_code = code
                    # yield pending responses (old status code):
                    yield from self.pending_responses
                    # add current response as pending:
                    self.pending_responses.clear()
                    self.pending_responses.append((obj, response))
                    return

                # status denotes the same error as previous one:
                self.fail_count += 1

                if self.fail_count >= self._suspend_trigger:
                    # too many errors, suspend downloads:
                    with skip_response_lock(domain):
                        self.skip_response = Response(
                            response.data,
                            response.error,
                            CustomResponseCode.DOWNLOAD_SUSPENDED
                        )
                    # yield current and pending responses (all with the same code)
                    yield obj, response
                    yield from self.pending_responses
                    self.pending_responses.clear()
                    return

                if self.fail_count < self._slowdown_trigger:
                    # no slowdown triggered, put current download as pending:
                    self.pending_responses.append((obj, response))
                    return

                if self.semaphore_limit <= 1:
                    # Slowdown triggered but not possible, simply yield
                    yield obj, response
                    return

                # slow down triggered and possible:

                # put pending responses in retry stage (they might work later, with
                # less concurrency, i.e. a lower semaphore_limit):
                self.retry_downloads.extend(
                    _[0] for _ in self.pending_responses
                )
                # add current response in pending responses:
                self.pending_responses.clear()
                self.pending_responses.append((obj, response))
                new_limit = max(1, self.semaphore_limit // 2)
                with semaphore_lock(domain):
                    self.semaphore = Semaphore(new_limit)
                    self.semaphore_limit = new_limit
                self.fail_count = 0
                self.fail_code = 0

        def url_wrapper(obj):
            if stop_event is not None and stop_event.is_set():
                return None
            url, opener, domain = prepare_url_read_args(obj)
            with skip_response_lock(domain):
                resp = domain_state.setdefault(domain, DomainState()).skip_response
            semaph = None
            if resp is None:
                if concurrency_per_domain_is_on:
                    with semaphore_lock(domain):
                        semaph = domain_state.setdefault(domain, DomainState()).semaphore
                    with semaph:
                        resp = urlread(url, blocksize, decode, timeout, opener, **kwargs)
                else:
                    resp = urlread(url, blocksize, decode, timeout, opener, **kwargs)
            return semaph, domain, obj, resp

        # perform download:
        for semaphore, domain, obj, response in t_map(url_wrapper, iterable):
            if stop_event is not None and stop_event.is_set():
                continue
            yield from domain_state[domain].update(semaphore, domain, obj, response)

        # yield pending responses not yielded:
        yield from chain.from_iterable(
            state.pending_responses for state in domain_state.values()
        )

        # re-download pending objs that were queued in domain state, possibly now they
        # work due to lowered per-domain concurrency:
        pending_objs = chain.from_iterable(
            state.retry_downloads for state in domain_state.values()
        )
        for semaphore, domain, obj, response in t_map(url_wrapper, pending_objs):
            if stop_event is not None and stop_event.is_set():
                continue
            yield obj, response

    finally:
        if t_pool is not None:
            t_pool.close()
            t_pool.join()


def adjust_max_concurrent_downloads(preferred_max_concurrent_downloads=None):
    """Return the maximum number of concurrent downloads adjusting the argument
    in order not to exceed the computer CPU

    :param preferred_max_concurrent_downloads: int denoting the preferred
        number of concurrent downloads. <=0 or None means: no preferred number, infer
        and return the max number of concurrent downloads from the computer CPU
    """
    # Now adjust with the computer capacity (algorithm copied from
    # concurrent.futures.ThreadPoolExecutor):
    os_max_concurrent_downloads = min(32, os.cpu_count() + 4)
    if not preferred_max_concurrent_downloads or preferred_max_concurrent_downloads < 0:
        return os_max_concurrent_downloads
    return min(os_max_concurrent_downloads, preferred_max_concurrent_downloads)


def thread_lock_factory():
    """Create a function that called with a key:str argument return a unique key-based
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


def _ismainthread():
    """Mainly uised for testing, returns True if we are currently executing in the
    mainv thread
    """
    # https://stackoverflow.com/q/23206787
    return current_thread() is main_thread()
