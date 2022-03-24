"""
Http requests with multi-threading

:date: Apr 15, 2017

.. moduleauthor:: <rizac@gfz-potsdam.de>
"""
import threading
import socket
from multiprocessing.pool import ThreadPool

from urllib.parse import urlparse, urlencode
from urllib.error import HTTPError, URLError
from http.client import HTTPException, responses
from urllib.request import (urlopen, build_opener,HTTPPasswordMgrWithDefaultRealm,
                            HTTPDigestAuthHandler)


# https://docs.python.org/3/library/urllib.request.html#request-objects
def get_host(request):
    """Returns the host (string) from a Request object"""
    return request.host


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


def urlread(url, blocksize=-1, decode=None, wrap_exceptions=True,
            raise_http_err=True, timeout=None, opener=None, **kwargs):
    """Read and return data from the given `url`. Wrapper around Python `urllib`
    `open` with some features added. Returns the tuple `(content_read, status, message)`

    :param url: (string or `Request` object) a valid url or an `urllib2.Request` object
    :param blocksize: int, default: -1. The block size while reading, -1 means:
        read entire content at once
    :param: decode: string or None, default: None. The string used for decoding (e.g.,
        'utf8'). If None, the result is a `bytes` object, otherwise `str`
    :param wrap_exceptions: if True (the default), all url-related exceptions:
        `urllib.error.HTTPError`, `urllib.error.URLError`, `http.client.HTTPException`,
        `socket.error` will be caught and wrapped into a :class:`url.URLException` that
        will be raised (the original exception is available via the `.exc` attribute)
    :param raise_http_err: If True (the default) `HTTPError`s will be raised normally as
        exceptions. Otherwise, they will be treated as response object and the tuple
        `(None, status, message)` will be returned, where `status` (int) is the HTTP
        status code (most likely in the range [400-599]) and `message` (`str`) is the
        string denoting the status message
    :param timeout: timeout parameter specifies a timeout in seconds for blocking
        operations like the connection attempt (if not specified, None or non-positive,
        the global default timeout setting will be used). This actually only works for
        HTTP, HTTPS and FTP connections.
    :param opener: a custom opener. When None (the default), the default urllib opener is
        used. See :func:`get_opener` for, e.g., creating an opener from a base url, user
        and passowrd
    :param kwargs: optional arguments to be passed to the underlying python `urlopen`
        function. These arguments are ignored if a custom `opener` argument is provided

    :return: the tuple (content_read, status_code, status_message), where:

        - content_read (`bytes` or `str`) is the response content. If `decode` is given,
          it is a `str`. If the response is issued from an HTTPError and `raise_http_err`
          is False, `content_read` is None
        - status_code (int) is the response HTTP status code
        - status_message (string) is the response HTTP status message

    :raise: `URLException` if `wrap_exceptions` is True. Otherwise any of the following:
        `urllib.error.HTTPError`, `urllib.error.URLError`, `http.client.HTTPException`,
        `socket.error`
    """
    try:
        ret = b''
        # set default for timeout: timeout in urlopen defaults to
        # socket._GLOBAL_DEFAULT_TIMEOUT so we unfortunately either pass it or skip it.
        # As we allow for non-negative numbers normalize it first to None. If None,
        # don't pass it to urlopen
        if timeout is not None and timeout > 0:
            kwargs['timeout'] = timeout

        with urlopen(url, **kwargs) if opener is None else opener.open(url, **kwargs) \
                as conn:
            if blocksize < 0:  # https://docs.python.org/2.4/lib/bltin-file-objects.html
                ret = conn.read()  # pylint: disable=no-member
            else:
                while True:
                    buf = conn.read(blocksize)  # pylint: disable=no-member
                    if not buf:
                        break
                    ret += buf
        if decode:
            ret = ret.decode(decode)
        return ret, conn.code, conn.msg  # pylint: disable=no-member
    except HTTPError as exc:
        if not raise_http_err:
            return None, exc.code, exc.msg
        else:
            if wrap_exceptions:
                raise URLException(exc)
            else:
                raise exc
    except (HTTPException, URLError, socket.error) as exc:
        # (socket.error is the superclass of all socket exc)
        if wrap_exceptions:
            raise URLException(exc)
        else:
            raise exc


class URLException(Exception):
    """Custom exception wrapping any url/http related exception `urllib.error.HTTPError`,
    `urllib.error.URLError`, `http.client.HTTPException`, `socket.error`.
    The original exception is retrievable via the `exc` attribute of this object.
    """
    def __init__(self, original_exception):
        # this constructor makes str(self) behave like str(original_exception) in both
        # py 2.7.14 and 3.6.2: But in principle, this class should act as container and
        # any operation be performed on self.exc
        super(URLException, self).__init__(original_exception)
        self.exc = original_exception


def read_async(iterable, urlkey=None, max_workers=None, blocksize=1024*1024, decode=None,
               raise_http_err=True, timeout=None, unordered=True, openers=None,
               **kwargs):  # pylint:disable=too-many-arguments
    """Wrapper around `multiprocessing.pool.ThreadPool()` for downloading
    data from urls in `iterable` asynchronously with remarkable performance boost for
    large downloads. Each download is executed on a separate *worker thread*, yielding
    the result of each `url` read.

    Yields the tuple:
    ```
        obj, result, exc, url
    ```
    where:

      - `obj` is the element of `iterable` which originated the `urlread` call
      - `result` is the result of `urlread`, it is None in case of errors (see `exc`
        below). Otherwise, it is the tuple
        ```(data, status_code, message)```
         where:
         * `data` is the data read (as bytes or string if `decode != None`). It can be
            None when `raise_http_err=False` and an HTTPException occurred
         * `status_code` is the integer denoting the status code (e.g. 200), and
         * `message` the string denoting the status message (e.g., 'OK').
      - `exc` is the exception raised by `urlread`, if any. **Either `result` or `exc`
         are None, but not both**. Note that `exc` is one of the following URL-related
         exceptions: `urllib.error.URLError`, `http.client.HTTPException`, `socket.error`
         and optionally, `urllib.error.HTTPError` (when `raise_http_err` is `False`).
         Any other exception is raised and will stop the download
      - `url` is the original url (either string or Request object). If `iterable` is an
        iterable of `Request` objects or url strings, then `url` is equal to `obj`

    Note that if `raise_http_err=False` then `urllib.error.HTTPError`s are treated as
    'normal' response and will be yielded in `result` as a tuple where `data=None` and
    `status_code` is most likely >= 400.
    Finally, this function can cleanly cancel *worker threads* (still to be processed)
    via Ctrl+C if executed from the command line. In the following we will simply refer
    to `urlread` to indicate the `urllib2.urlopen.read` function.

    :param iterable: an iterable of objects representing the urls addresses to be read:
        if its elements are neither strings nor `Request` objects, the `urlkey` argument
        (see below) must be specified to map each element to a valid url string or
        Request
    :param urlkey: function or None. When None (the default), all elements of `iterable`
        must be url strings or Request objects. When function, it will be called with
        each element of `iterable` as argument, and must return the mapped url address or
        Request.
    :param max_workers: integer or None (the default) denoting the max worker threads
        used. When None, the threads allocated are relative to the machine CPU
    :param blocksize: integer defaulting to 1024*1024 specifying, when connecting to one
        of the given urls, the mximum number of bytes to be read at each call of
        `urlopen.read`. If the size argument is negative or omitted, read all data until
        EOF is reached
    :param decode: string or None (default: None) optional decoding (e.g., 'utf-8') to
        convert the result of the url request from `bytes` (the default) into `str`
    :param raise_http_err: boolean (True by default) tells whether `HTTPError`s should
        be yielded as exceptions or not. When False, `HTTPError`s are yielded as normal
        responses in `result` as the tuple `(None, status_code, message)` (where
        `status_code` is most likely >= 400)
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
    # flag for CTRL-C or cancelled tasks
    kill = False

    # function called from within urlread to check if go on or not
    def urlwrapper(obj):
        if kill:
            return None
        url = urlkey(obj) if urlkey is not None else obj
        opener = openers(obj) if openers is not None else None
        try:
            return obj, \
                urlread(url, blocksize, decode, True, raise_http_err, timeout, opener,
                        **kwargs), \
                None, url
        except URLException as urlexc:
            return obj, None, urlexc.exc, url

    tpool = ThreadPool(max_workers)
    threadpoolmap = tpool.imap_unordered if unordered else tpool.imap
    # note above: chunksize argument for threads (not processes)
    # seems to slow down download. Omit the argument and leave chunksize=1 (default)
    try:
        # this try is for the keyboard interrupt, which will be caught inside the
        # as_completed below
        for result_tuple in threadpoolmap(urlwrapper, iterable):
            if kill:
                continue  # (for safety: we should never enter here)
            yield result_tuple
    except:
        # According to https://stackoverflow.com/a/29237343, after a `KeyboardInterrupt`
        # this method does not return until all working threads have finished. Set
        # `kill = True` to make them finish quicker (see `urlwrapper` above):
        kill = True
        # (the time from now until 'raise' below is the time taken to finish all threads)
        raise

    tpool.close()


def _ismainthread():
    """Mainly uised for testing, returns True if we are currently executing in the
    mainv thread
    """
    # https://stackoverflow.com/q/23206787
    return isinstance(threading.current_thread(), threading._MainThread)
