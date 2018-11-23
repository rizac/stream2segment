'''
Http requests with multi-threading (async) utilities

:date: Apr 15, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
from contextlib import closing
import threading
# import http.client
import socket
from multiprocessing.pool import ThreadPool
import os

from future.utils import PY2

# Python 2 and 3: Futures (http://python-future.org/imports.html#aliased-imports) backports
# to python2 are buggy when used with ThreadPools (like here). As there seem to be no particular
# difference in function signature but only import placement, we do the old way
# ALSO, ALL IMPORTS REQUIRING ANY OF THE MODULES/CLASSES BELOW SHOULD IMPORT FROM HERE
# TO GUARANTEE PY2+3 COMPATIBILITY
try:  # py3:
    from urllib.parse import urlparse, urlencode  # pylint: disable=unused-import
    from urllib.request import urlopen, Request, \
        build_opener, HTTPPasswordMgrWithDefaultRealm, HTTPDigestAuthHandler  # pylint: disable=unused-import
    from urllib.error import HTTPError, URLError
    from http.client import HTTPException, responses  # pylint: disable=ungrouped-imports
except ImportError:
    from urlparse import urlparse  # @UnusedImport pylint: disable=bad-option-value
    from urllib import urlencode  # @UnusedImport pylint: disable=ungrouped-imports
    from urllib2 import urlopen, Request, HTTPError, URLError, \
        build_opener, HTTPPasswordMgrWithDefaultRealm, HTTPDigestAuthHandler  # @UnusedImport
    from httplib import HTTPException
    from BaseHTTPServer import BaseHTTPRequestHandler
    # responses values are tuples, map to the response message (1st tuple item, str)
    # and make this response compatible with the python3 response above:
    responses = {k: v[0] for k, v in BaseHTTPRequestHandler.responses.iteritems()}


def get_opener(url, user, password):
    '''Returns an opener to be used for downloading data with a given user and password.
    All arguments should be strings.

    :param url: the domain name of the given url
    :param: string, the user name
    :param password: the password

    :return: an urllib opener
    '''
    parsed_url = urlparse(url)
    base_url = "%s://%s" % (parsed_url.scheme, parsed_url.netloc)
    handlers = []
    password_mgr = HTTPPasswordMgrWithDefaultRealm()
    password_mgr.add_password(None, base_url, user, password)
    handlers.append(HTTPDigestAuthHandler(password_mgr))
    return build_opener(*handlers)


def urlread(url, blocksize=-1, decode=None, wrap_exceptions=True,
            raise_http_err=True, timeout=None, opener=None, **kwargs):
    """
    Reads and return data from the given url. Wrapper around urllib2.open with some
    features added. Returns the tuple (content_read, status, message)

    :param url: (string or `Request` object) a valid url or an `urllib2.Request` object
    :param blockSize: int, default: -1. The block size while reading, -1 means:
        read entire content at once
    :param: decode: string or None, default: None. The string used for decoding to string
        (e.g., 'utf8'). If None, the result is returned as it is (type `bytes`, note that in
        Python2 this is equivalent to `str`), otherwise as unicode string (`str` in python3+)
    :param wrap_exceptions: if True (the default), all url-related exceptions (python2)
        ```urllib2.HTTPError, urllib2.URLError, httplib.HTTPException, socket.error```
        or the equivalent (python3):
        ```urllib.error.HTTPError, urllib.error.URLError, http.client.HTTPException, socket.error```
        will be caught and wrapped into an :class:`url.URLException` object E that will be
        raised (the original exception can always be retrieved via `E.exc`)
    :param raise_http_err: If True (the default) `HTTPError`s will be raised normally as
        exceptions. Otherwise, they will treated as response object and the
        tuple (None, status, message) will be returned, where `status` (int) is the
        `HTTPError` status code (most likely in the range [400-599]) and `message`
        (string) is the string denoting the status message, respectively
    :param timeout: timeout parameter specifies a timeout in seconds for blocking operations
        like the connection attempt (if not specified, None or non-positive, the global
        default timeout setting will be used). This actually only works for HTTP, HTTPS
        and FTP connections.
    :param opener: a custom opener. When None (the default), the default urllib opener is used.
        See :func:`get_opener` for, e.g., creating an opener from a base url, user and passowrd
    :param kwargs: optional arguments to be passed to the underlying python `urlopen` function.
        These arguments are ignored if a custom `opener` argument is provided

    :return: the tuple (content_read, status_code, status_message), where:

        - content_read (bytes) is the response content (if `decode` is given,
          it is a str (py3) / unicode (py2). If the response is issued from an HTTPError
          and raise_http_err=False, it is None)

        - status_code (int) is the response HTTP status code

        - status_message (string) is the response HTTP status message

    :raise: `URLException` if `wrap_exceptions` is True. Otherwise any of the following:
        `urllib2.HTTPError`, `urllib2.URLError`, `httplib.HTTPException`, `socket.error`
        or the equivalent (python3):
        ```urllib.error.HTTPError, urllib.error.URLError, http.client.HTTPException, socket.error```
    """
    try:
        ret = b''
        # set default for timeout: timeout in urlopen defaults to socket._GLOBAL_DEFAULT_TIMEOUT
        # so we unfortunately either pass it or skip it. As we allow for non-negative numbers
        # normalize it fiorst to None. If None, don't pass it to urlopen
        if timeout is not None and timeout > 0:
            kwargs['timeout'] = timeout

        # urlib2 does not support with statement in py2. See:
        # http://stackoverflow.com/questions/3880750/closing-files-properly-opened-with-urllib2-urlopen
        # https://docs.python.org/2.7/library/contextlib.html#contextlib.closing
        with closing(urlopen(url, **kwargs) if opener is None else opener.open(url, **kwargs)) \
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
    except (HTTPException,  # @UndefinedVariable
            URLError, socket.error) as exc:  # socket.error is the superclass of all socket exc.
        if wrap_exceptions:
            raise URLException(exc)
        else:
            raise exc


class URLException(Exception):
    """Custom exception wrapping any url/http related exception:
    `urllib2.HTTPError`, `urllib2.URLError`, `httplib.HTTPException`, `socket.error`
    The original exception is retrievable via the `self.exc` attribute
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
    """
    Wrapper around `multiprocessing.pool.ThreadPool()` for downloading
    data from urls in `iterable` asynchronously with remarkable performance boost for large
    downloads. Each download is executed on a separate *worker thread*, yielding the result of
    each `url` read.

    Yields the tuple:
    ```
        obj, result, exc, url
    ```
    where:

      - `obj` is the element of `iterable` which originated the `urlread` call
      - `result` is the result of `urlread`, it is None or the tuple
        ```(data, status_code, message)```
         where:
         * `data` is the data read (as bytes or string if `decode != None`). It can be None,
            e.g., when `raise_http_err=True` and an http-like exception has been raised
         * `status_code` is the integer denoting the status code (e.g. 200), and
         * `messsage` the string denoting the status message (e.g., 'OK').
      - exc is the exception raised by `urlread`, if any. **Either `result` or `exc` are None,
      but not both**. Note that `exc` is one of the following URL-related exceptions:
      ```urllib2.URLError, httplib.HTTPException, socket.error```
      Any other exception is raised and will stop the download
      - url: the original url (either string or Request object). If `iterable` is an iterable
      of `Request` objects or url strings, then `url` is equal to `obj`

    Note that if `raise_http_err=False` then `HTTPError`s are treated as 'normal'
    response and will be yielded in `result` as a tuple where `data=None` and `status_code`
    is most likely greater or equal to 400.
    Finally, this function can cleanly cancel yet-to-be-processed *worker threads* via Ctrl+C
    if executed from the command line. In the following we will simply refer to `urlread`
    to indicate the `urllib2.urlopen.read` function.

    :param iterable: an iterable of objects representing the urls addresses to be read
        (either strings or `urllib2.Request` objects). If the elements of `iterable` are neither
        strings nor Request objects, the `urlkey` argument (see below) must be specified to
        return valid url strings or Request objects
    :param urlkey: a function of one argument or None (the default) that is used to extract
        an url (string) or Request object from each `iterable` element. When None, it returns
        the argument, i.e. assumes that `iterable` is an iterable of valid url addresses or
        Request objects.
    :param max_workers: integer or None (the default) denoting the max workers of the
        `ThreadPoolExecutor`. When None, the theads allocated are relative to the machine cpu
    :param blocksize: integer defaulting to 1024*1024 specifying, when connecting to one of
        the given urls, the mximum number of bytes to be read at each call of `urlopen.read`.
        If the size argument is negative or omitted, read all data until EOF is reached
    :param decode: string or None (default: None) optional argument specifying if the content
        of the url must be decoded. None means: return the byte string as it was read.
        Otherwise, use this argument for string content (not bytes) by supplying a decoding,
        such as e.g. 'utf8'
    :param raise_http_err: boolean (True by default) tells whether `HTTPError`s should
        be yielded as exceptions or not. When False, `HTTPError`s are yielded as normal
        responses in `result` as the tuple `(None, status_code, message)`  (where `status_code`
        is most likely greater or equal to 400)
    :param timeout: timeout parameter specifies a timeout in seconds for blocking operations
        like the connection attempt (if not specified, None or non-positive, the global default
        timeout setting will be used). This actually only works for HTTP, HTTPS and FTP
        connections.
    :param unordered: boolean (default False): tells whether the download results are yielded
        in the same order they are input in `iterable`, i.e. if the i-th download is relative
        to the i-th element of iterable. Theoretically, False (the default) might execute faster,
        but results are not guaranteed to be yielded in the same order as `iterable`. Although
        tests did not show any relevant performance increase with `ThreadPool`s (maybe it's a
        feature of `ProcessPool`s) this argument is False by default
    :param openers: a function behaving like `urlkey`, should return a specific opener
        for the given item of iterable. When None, the default urllib opener is used
        See :func:`get_opener` for, e.g., creating an opener from a base url, user and passowrd
    :param kwargs: optional arguments to be passed to the underlying python `urlopen` function.
        These arguments are ignored if a custom `openers` function is provided

    Notes:
    ======

    ThreadPool vs ThreadPoolExecutor
    --------------------------------

    This function changed from using `concurrent.futures.ThreadPoolExecutor` into
    the "old" `multiprocessing.pool.ThreadPool`: the latter consumes in most cases
    less memory (about 30% less), especially if `iterable` is not a list in memory but
    a python iterable (`concurrent.futures.ThreadPoolExecutor` builds a `set` of
    `Future`s object from `iterable`, whereas `multiprocessing.pool.ThreadPool` seems just
    to execute each element in iterable)

    killing threads / handling exceptions
    -------------------------------------

    this function handles any kind of unexpected exception (particularly relevant in case of
    e.g., `KeyboardInterrupt`) by canceling all worker threads before raising. As
    ThreadPoolExecutor returns (or raises) after all worker threads have finished, an internal
    boolean flag makes all remaining worker threads quit as soon as possible, making the
    function return (or raise) much more quickly
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
    threadpoolmap = tpool.imap_unordered if unordered else tpool.imap  # (func, iterable, chunksize)
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
        # According to this post:
        # http://stackoverflow.com/questions/29177490/how-do-you-kill-futures-once-they-have-started,
        # after a KeyboardInterrupt this method does not return until all
        # working threads have finished. Thus, we implement the `kill` flag
        # which makes them exit immediately, and hopefully this function will return within
        # seconds at most. We catch  a bare except cause we want the same to apply to all
        # other exceptions which we might raise (see few line above)
        kill = True
        # the time here before executing 'raise' below is the time taken to finish all threads.
        # Without the line above, it might be a lot (minutes, hours), now it is much shorter
        # (in the order of few seconds max) and the command below can be executed quickly:
        raise

    tpool.close()


def _ismainthread():
    """
    utility function for testing, returns True if we are currently executing in the main thread"""
    # see:
    # http://stackoverflow.com/questions/23206787/check-if-current-thread-is-main-thread-in-python
    return isinstance(threading.current_thread(), threading._MainThread)
