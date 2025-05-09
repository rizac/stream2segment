"""
Http requests with multi-threading

:date: Apr 15, 2017

.. moduleauthor:: <rizac@gfz-potsdam.de>
"""
from threading import Semaphore, current_thread, main_thread
import socket
import os
from contextlib import nullcontext
from multiprocessing.pool import ThreadPool

from urllib.parse import urlparse, urlencode
from urllib.error import HTTPError, URLError
from http.client import HTTPException, responses
from urllib.request import (urlopen, build_opener,HTTPPasswordMgrWithDefaultRealm,
                            HTTPDigestAuthHandler)


# https://docs.python.org/3/library/urllib.request.html#request-objects
def get_host(url_or_request):
    """Returns the host (string) from a Request object or url path as str"""
    # Handle both url as Request obj. (use attr. host) or string (use urlparse):
    return getattr(url_or_request, 'host', urlparse(url_or_request).netloc)


def get_opener(url, user, password):
    """Return an opener to be used for downloading data with a given user and password.
    All arguments should be strings.

    :param url: the domain name of the given url
    :param: string, the user name
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
          HTTPException, URLError, socket.error - e.g. socket.timeout - or HTTPError).
          It is always None if the request/response exchange was successful
        - status_code (int) is the response HTTP status code. None if the code could not
          be inferred (e.g. `error` is given but not instance of HTTPError). Note: it
          could be a string representing the status code instead of an int
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
    except (HTTPException, URLError, socket.error) as exc:
        # (socket.error is the superclass of all socket exc)
        return None, exc, None


def read_async(iterable,
               url_callback=None,
               max_workers=None,
               max_cuncurrent_per_domain=2,
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
    :param max_cuncurrent_per_domain: integer or None (default: 2) denoting the max
        worker threads *per URL domain*. Increasing this number leads faster downloads
        but also potentially fewer data received, if it overcomes the maximum concurrent
        requests configured in some server, which might be as low as 2 or 3.
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

    semaphores = None
    null_context = nullcontext()  # no-op with statement to mimic a semaphore
    if max_cuncurrent_per_domain and max_cuncurrent_per_domain > 0:
        semaphores = {}

    # flag for CTRL-C or cancelled tasks
    kill = False

    # function called from within urlread to check if go on or not
    def url_wrapper(obj):
        if kill:
            return None
        url = obj
        if url_callback is not None:
            url = url_callback(obj)
        opener = openers(obj) if openers is not None else None
        if semaphores is None:
            sem = null_context
        else:
            sem = semaphores.setdefault(
                get_host(url), Semaphore(max_cuncurrent_per_domain)
            )
        with sem:
            return (obj,) + urlread(url, blocksize, decode, timeout, opener, **kwargs)

    tpool = ThreadPool(adjust_max_concurrent_downloads(max_workers))
    threadpoolmap = tpool.imap_unordered if unordered else tpool.imap
    # note above: chunksize argument for threads (not processes)
    # seems to slow down download. Omit the argument and leave chunksize=1 (default)
    try:
        # this try is for the keyboard interrupt, which will be caught inside the
        # as_completed below
        for result_tuple in threadpoolmap(url_wrapper, iterable):
            if result_tuple is not None:  # (just for extreme safety: see urlwrapper)
                yield result_tuple
    except:  # noqa
        # According to https://stackoverflow.com/a/29237343, after a `KeyboardInterrupt`
        # this method does not return until all working threads have finished. Set
        # `kill = True` to make them finish quicker (see `urlwrapper` above):
        kill = True
        # (the time from now until 'raise' below is the time taken to finish all threads)
        raise
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
