"""
Http requests with multi-threading

:date: Apr 15, 2017

.. moduleauthor:: <rizac@gfz-potsdam.de>
"""
from threading import Semaphore, current_thread, main_thread, Lock, Event
import signal
import socket
import os
import ssl
from enum import IntEnum
# from contextlib import nullcontext
from multiprocessing.pool import ThreadPool

from urllib.parse import urlparse  # , urlencode
from urllib.error import HTTPError, URLError
from http.client import HTTPException, responses as builtin_responses
from urllib.request import (urlopen, build_opener,HTTPPasswordMgrWithDefaultRealm,
                            HTTPDigestAuthHandler)


# https://docs.python.org/3/library/urllib.request.html#request-objects
def get_host(url_or_request) -> str:
    """Returns the host (string) from a urllib.request.Request object or str (URL)"""
    # Handle both url as Request obj. (use attr. host) or string (use urlparse):
    return urlparse(getattr(url_or_request, 'full_url', url_or_request)).hostname  # FIXME check


def get_opener(url, user, password):
    """Return an opener to be used for downloading data with a given user and password.
    All arguments should be strings.

    :param url: the domain name of the given url
    :param: user: string, the user name
    :param password: the password

    :return: an urllib opener
    """
    parsed_url = urlparse(url)
    base_url = "%s://%s" % (parsed_url.scheme, parsed_url.netloc)
    handlers = []
    password_mgr = HTTPPasswordMgrWithDefaultRealm()
    password_mgr.add_password(None, base_url, user, password)
    handlers.append(HTTPDigestAuthHandler(password_mgr))
    return build_opener(*handlers)


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


def urlread(url, blocksize=-1, decode=None, timeout=None, opener=None, **kwargs):
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

    :return: the tuple (data, error, status_code), where:

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
        return ret, None, conn.code
    except HTTPError as exc:
        return None, exc, exc.code
    except URLError as u_err:
        code = CustomResponseCode.URL_ERROR
        if isinstance(u_err.reason, socket.timeout):
            code = CustomResponseCode.TIMEOUT_ERROR
        elif isinstance(u_err.reason, socket.gaierror):
            code = CustomResponseCode.GET_ADDR_INFO_ERROR
        elif isinstance(u_err.reason, ConnectionRefusedError):
            code = CustomResponseCode.CONNECTION_REFUSED_ERROR
        elif isinstance(u_err.reason, ssl.SSLError):
            code = CustomResponseCode.SSL_ERROR
        return None, u_err, code
    except HTTPException as h_exc:
        # (socket.error is the superclass of all socket exc)
        return None, h_exc, CustomResponseCode.HTTP_EXC_ERROR


def read_async(iterable,
               url_callback=None,
               max_workers=None,
               max_workers_per_domain=8,
               worker_error_codes=(429, 500, 503, CustomResponseCode.TIMEOUT_ERROR),
               max_retry_on_same_error=25,
               blocksize=-1,
               decode=None,
               timeout=None, unordered=True, openers=None, **kwargs):  # noqa
    """Wrapper around `multiprocessing.pool.ThreadPool()` for  downloading
    data asynchronously from different urls iteratively. Specifically designed for
    large downloads, each download is executed on a separate *worker thread*, yielding
    the result of each `url` read.

    For each item `obj` of iterable, this function yields the tuple:
    ```
        obj: [Any],
        response_data: str | bytes | None,
        response_error: Exception | None,
        response_code: int | None
    ```
    Notes:

      - either `response_data` and `response_error` are None, but not both. If the latter
        is not None, then the request failed. `response_error` can be any of the
        following URL-related exceptions: `urllib.error.URLError`,
        `http.client.HTTPException`, `socket.error` `urllib.error.HTTPError`. Any other
        Exception raises "normally"
      - `response_code` is the int denoting the status code (e.g. 200), which might be
         None (e.g., a failed request with `response_error` not `urllib.error.HTTPError`)

    :param iterable: an iterable of objects representing the urls addresses to be read:
        if its elements are neither strings nor `Request` objects, the `url_callback`
        argument must be specified to map each element to a valid url string or Request
    :param url_callback: function or None. When None (the default), all elements of
        `iterable` must be url strings or Request objects. If callable, it will be
        called with each element of `iterable` as argument, and must return the mapped
        url address or Request.
    :param max_workers: integer or None (the default) denoting the max worker threads
        used. When None, the threads allocated are relative to the machine CPU
    :param max_workers_per_domain: integer or None (default: 8) denoting the max
        worker threads *per URL domain*. Defaults to 8
    :param worker_error_codes: http status codes that denote too many requests and
        cause max_workers_per_domain to be halved, if received from the same server
        more than 3 times in sequence. If empty or None, no halving is applied
    :param max_retry_on_same_error: in denoting the maximum downloads from the same
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
    :param openers: a function behaving like `urlkey`, should return a specific opener
        for the given item of iterable. When None, the default opener is used. See
        :func:`get_opener` for creating an opener from given base URL, user and password
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
    max_concurrency = adjust_max_concurrent_downloads(max_workers)

    def new_domain_lock_factory():
        domain_locks = {}
        global_lock = Lock()

        def get_domain_lock(domain):
            lock = domain_locks.get(domain)
            if lock is not None:
                return lock
            with global_lock:
                return domain_locks.setdefault(domain, Lock())

        return get_domain_lock

    def update_failed_response(domain: str, response: tuple):
        pass

    def get_skip_response(domain: str):
        return None

    if max_retry_on_same_error > 0:

        class FailedResponse:
            # memory efficient container for a skip response
            # (better than dict, might use dataclass but keep it simple)
            __slots__ = ('fail_count', 'fail_code', 'response')

            def __init__(self):
                self.fail_code = 0
                self.response: tuple = None  # same obj returned by `urlread` # noqa
                self.fail_count = 0

        failed_response: dict[str, FailedResponse] = {}

        failed_response_lock = new_domain_lock_factory()

        def get_skip_response(domain: str):
            ret = failed_response.get(domain)
            if ret is not None:
                with failed_response_lock(domain):
                    return ret.response  # tuple or None
            return None

        def update_failed_response(domain: str, response: tuple):
            exc = response[1]
            if exc is not None:
                code = response[2]
                with failed_response_lock(domain):
                    skip_resp = failed_response.setdefault(domain, FailedResponse())
                    if skip_resp.fail_code != code:
                        skip_resp.fail_code = code
                        skip_resp.fail_count = 1
                    else:
                        skip_resp.fail_count += 1
                        if skip_resp.fail_count > max_retry_on_same_error:
                            skip_resp.response = (
                                None, exc, CustomResponseCode.DOWNLOAD_SUSPENDED
                            )

    def prepare_url_read_args(obj) -> tuple:
        url = obj
        if url_callback is not None:
            url = url_callback(obj)
        opener = openers(obj) if openers is not None else None
        return url, opener, get_host(url)

    # 1) handle the case 'no concurrency' (simple loop)
    ###################################################

    if max_concurrency <= 1:
        for obj in iterable:
            url, opener, domain = prepare_url_read_args(obj)
            resp = get_skip_response(domain)
            if resp is None:
                resp = urlread(url, blocksize, decode, timeout, opener, **kwargs)
                update_failed_response(domain, resp)
            yield (obj,) + resp
        return

    # 2) Handle the case 'concurrency' (multi threading)
    ####################################################

    # flag for CTRL-C or cancelled tasks
    stop_event = Event()

    def signal_handler(sig, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)

    tpool = ThreadPool(max_concurrency)
    threadpoolmap = tpool.imap_unordered if unordered else tpool.imap
    # note above: chunksize argument for threads (not processes)
    # seems to slow down download. Omit the argument and leave chunksize=1 (default)

    try:

        slowdown_error_codes = set(worker_error_codes or [])
        downgrade_trigger = 3
        domain_state = {}

        domain_state_lock = new_domain_lock_factory()

        class DomainState:
            # memory efficient container for a domain state
            # (better than dict, might use dataclass but keep it simple)
            __slots__ = ('semaphore', 'semaphore_limit', 'fail_count', 'fail_code')

            def __init__(self):
                self.semaphore = Semaphore(max_workers_per_domain)
                self.semaphore_limit = max_workers_per_domain
                self.fail_code = 0
                self.fail_count = 0

        def url_wrapper(obj):
            if stop_event.is_set():
                return None
            url, opener, domain = prepare_url_read_args(obj)
            resp = get_skip_response(domain)
            sem = None
            if resp is None:
                if max_workers_per_domain <= 1:
                    # no workers per domain
                    resp = urlread(url, blocksize, decode, timeout, opener, **kwargs)
                else:
                    with domain_state_lock(domain):
                        sem = domain_state.setdefault(domain, DomainState()).semaphore
                    with sem:
                        resp = urlread(url, blocksize, decode, timeout, opener, **kwargs)
                update_failed_response(domain, resp)
            return sem, domain, (obj,) + resp

        # this try is for the keyboard interrupt, which will be caught inside the
        # as_completed below
        for result in threadpoolmap(url_wrapper, iterable):
            if stop_event.is_set():
                continue
            sem, domain, result = result
            if sem is not None:
                state: DomainState = domain_state[domain]
                # Only allow the most recent semaphore instance to affect state
                if sem is state.semaphore and state.semaphore_limit > 1:
                    status = result[2]
                    if status not in slowdown_error_codes:
                        state.fail_count = 0
                        state.fail_code = 0
                    elif status != state.fail_code:
                        state.fail_count = 1
                        state.fail_code = status
                    else:
                        state.fail_count += 1
                        if state.fail_count >= downgrade_trigger:
                            new_limit = max(1, state.semaphore_limit // 2)
                            with domain_state_lock(domain):
                                state.semaphore = Semaphore(new_limit)
                            state.semaphore_limit = new_limit
                            state.fail_count = 0
                            state.fail_code = 0
            yield result

    finally:
        tpool.close()
        tpool.join()


def adjust_max_concurrent_downloads(preferred_max_concurrent_downloads=None):
    """Return the maximum number of concurrent downloads adjusting the argument
    in order not to exceed the computer CPU

    :param preferred_max_concurrent_downloads: int denoting the preferred
        number of concurrent downloads. <=0 or None means: no preferred number, infer
        and return the max number of concurrent downloads from the computer CPU
    """
    # Now adjust with the computer capacity (we use the algorithm here:
    # https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor
    os_max_concurrent_downloads = min(32, os.cpu_count() + 4)
    if not preferred_max_concurrent_downloads or preferred_max_concurrent_downloads < 0:
        return os_max_concurrent_downloads
    return min(os_max_concurrent_downloads, preferred_max_concurrent_downloads)


def _ismainthread():
    """Mainly uised for testing, returns True if we are currently executing in the
    mainv thread
    """
    # https://stackoverflow.com/q/23206787
    return current_thread() is main_thread()
