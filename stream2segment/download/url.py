"""
Http requests with multi-threading
"""
# :date: Apr 15, 2017
from collections.abc import Iterable, Callable
from contextlib import nullcontext
from typing import Any
from dataclasses import dataclass
from threading import current_thread, main_thread, Lock, Event, Semaphore
import signal
import socket
import sys
import os
import ssl
from enum import IntEnum
from multiprocessing.pool import ThreadPool

from urllib.parse import urlsplit
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
    if isinstance(url, Request):
        url = url.full_url
    parsed = urlsplit(url)
    if not parsed.scheme:
        parsed = urlsplit(f'{default_scheme}{url}')
    return parsed.hostname


def _get_opener(base_url, user, password):
    """
    Return an opener to be used for downloading data with a given user and password.
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


@dataclass(slots=True)
class Response:
    """
    Lightweight data class representing a Response object with two arguments:

    - data (any object, usually `bytes` or `str`) is the response content.
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

    data: str | bytes
    status_code: int
    request: str | Request
    meta: Any = None

    @property
    def is_ok(self):
        return self.status_code == 200 and self.data

    @property
    def request_url(self) -> str:
        url = self.request
        if isinstance(self.request, Request):
            url = self.request.full_url
        return url

    def __str__(self):
        if self.is_ok:
            return f'Data successfully downloaded from {self.request_url}'

        msg = "unknown cause"
        if self.status_code in responses:
            msg = responses[self.status_code].lower()
            if self.status_code in builtin_responses:
                msg += f' (http code {self.status_code})'

        return f'Download unsuccessful, {msg}. Source: {self.request_url}'

    def __repr__(self):
        return f'{self.__class__.__name__}({self.status_code}, {self.request!r})'


def read_url(
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


def build_and_read_urls(
    url_builder: Callable[..., [str | Request]] | None,
    iterable: Iterable,
    *,
    max_global_concurrency: int | None = None,
    max_concurrency: int | None = 8,
    error_limit: int | None = 25,
    same_error_limit: int | None = 10,
    skip_on_errors: set = frozenset({400, 401, 403, 405, 414, 501}),
    blocksize=-1,
    decode=None,
    timeout=None,
    unordered=True,
    credentials=None,
    **kwargs
) -> Iterable[Response]:
    """
    Download data (optionally asynchronously) from different urls build from the given
    iterable of objects, yielding Response objects. Specifically designed for large
    downloads, handles concurrency (`threading.Pool`) globally and per URL-domain,
    stopping at specific errors iteratively received.

    :param url_builder: a callable which takes one object of iterable and converts it
        to a download URL string or Request object
    :param iterable: an iterable of objects. Each element will be returned in the
        `Response.meta` attribute so that the input object can be easily retrieved,
        if needed
    :param max_global_concurrency: integer or None (the default) denoting the max
        parallel downloads globally. This corresponds to the maximum worker (sub)
        threads used. When None or <1, the threads allocated are relative to the machine
        CPU (should be around 16-32, usually > 4)
    :param max_concurrency: integer denoting the max parallel downloads per url domain.
        Defaults to 8. None or values < 1 will disable per-domain concurrency, leaving
        only the global one active
    :param error_limit: int (default: 25) denoting the error limit per-domain: if no
        download is successful for `error_limit` times, the downloads are not yielded
        and all pending domain downloads will be skipped; users are responsible to handle
        the retry of failed downloads in case. Unsuccessful downloads are HTTP error
        with codes in the range (400-599) and not included in `skip_on_errors` and other
        errors for which a custom unique code is assigned (e.g., timeout or network
        errors). All other HTTP codes are yielded normally, e.g. 'OK' (200), 'No data'
        (204).
        When this parameter is None or < 1 the error code check will be disabled, and all
        downloads will be performed and yielded, which might slow down time
        unnecessarily and increase server workload
    :param same_error_limit: int denoting the (same) error limit per-domain: if
        the same error type is returned for `consecutive_error_limit`, the downloads
        are not yielded and all pending domain downloads will be skipped; users are
        responsible to handle the retry of failed downloads in case. Two errors are of
        the same type if they share the same code (int): see parameter `error_limit` for
        more details. When this parameter is None or < 1 the error code check will be
        disabled, and downloads will be performed regardless of their successful state
    :param skip_on_errors: a set of ints denoting the http errors that should
        be skipped from the error counts. These are errors that denote a
        "not worth retrying" message. Defaults to:
        400 Bad Request — request is malformed.
        401 Unauthorized — credentials/access problem.
        403 Forbidden — access denied.
        405 Method Not Allowed — client is using the wrong HTTP method.
        414 URI Too Long — request construction issue.
        501 Not Implemented — server does not support that functionality
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
    if max_global_concurrency is None or max_global_concurrency < 1:
        max_global_concurrency = get_os_max_thread_count()
    max_global_concurrency = max(1, max_global_concurrency)

    if max_concurrency is not None:
        if max_concurrency < 1:
            max_concurrency = None
        else:
            max_concurrency = min(max_concurrency, max_global_concurrency)
            max_concurrency = max(1, max_concurrency)

    if error_limit is None:
        error_limit = float('inf')

    if same_error_limit is None:
        same_error_limit = float('inf')

    user = None
    pswd = None
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

    # locks (for modifying objects in sub-threads safely):
    aborted_download_domains_lock = Lock()
    openers_lock = Lock()
    null_context = nullcontext()
    if max_concurrency is not None:
        semaphores_lock = Lock()
    else:
        semaphores_lock = nullcontext()

    # flag for CTRL-C or cancelled tasks
    stop_event = Event()

    def signal_handler(sig, frame):
        stop_event.set()
        if sys.stdout.isatty():
            print('\nCancelling pending tasks, please wait', file=sys.stdout)
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, signal_handler)

    t_pool = ThreadPool(max_global_concurrency)
    t_map = t_pool.imap_unordered if unordered else t_pool.imap
    # note above: chunksize argument for threads (not processes)
    # seems to slow down download. Omit the argument and leave chunksize=1 (default)

    try:
        aborted_download_domains = set()
        semaphores: dict[str, Semaphore] = {}

        def url_wrapper(item):
            if stop_event.is_set():
                return None
            url = item if url_builder is None else url_builder(item)

            # get the opener (restricted data):
            domain = get_host(url)  # noqa
            with aborted_download_domains_lock:
                if domain in aborted_download_domains:
                    return None

            opener = None
            if credentials is not None:
                with openers_lock:
                    if pswd is not None and domain not in openers:
                        openers[domain] = _get_opener(domain, user, pswd)
                    opener = openers.get(domain, None)

            semaphore = null_context
            if max_concurrency is not None:
                with semaphores_lock:
                    if domain not in semaphores:
                        semaphores[domain] = Semaphore(max_concurrency)
                    semaphore = semaphores[domain]

            with semaphore:
                if stop_event.is_set():
                    return None
                resp = read_url(url, blocksize, decode, timeout, opener, **kwargs)
                resp.meta = item

            return domain, resp

        last_n_errors = {}

        # perform download:
        for resp_tuple in t_map(url_wrapper, iterable):
            if stop_event.is_set() or resp_tuple is None:
                continue

            domain, response = resp_tuple
            if domain in aborted_download_domains:
                continue

            if 200 <= response.status_code < 300 or response.status_code in skip_on_errors:
                yield response
                resp_queue = last_n_errors.get(domain, [])
                for resp in resp_queue:
                    if (
                        200 <= resp.status_code < 300 or
                        resp.status_code in skip_on_errors
                    ):
                        yield resp_queue.pop()
                resp_queue.clear()
                continue

            # error response. Append to queue:
            resp_queue = last_n_errors.setdefault(domain, [])
            resp_queue.append(response)

            same_err_limit_reached = (
                len(resp_queue) >= same_error_limit and
                len({_.status_code for _ in resp_queue[-same_error_limit:]}) == 1
            )
            err_limit_reached = len(resp_queue) >= error_limit

            if same_err_limit_reached or err_limit_reached:
                with aborted_download_domains_lock:
                    aborted_download_domains.add(domain)
                resp_queue.clear()

        # yield suspended results:
        for domain, resp_queue in last_n_errors.items():
            if domain in aborted_download_domains:
                continue
            for response in resp_queue:
                yield response

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


def _is_main_thread():
    """
    Used for testing, returns True if we are currently executing in the main Thread
    """
    # https://stackoverflow.com/q/23206787
    return current_thread() is main_thread()
