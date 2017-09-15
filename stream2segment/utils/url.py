'''
Created on Apr 15, 2017

@author: riccardo
'''
from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import map
from builtins import str
from contextlib import closing
import threading
import urllib.request, urllib.error, urllib.parse  # @UnresolvedImport
import http.client
import socket
from multiprocessing.pool import ThreadPool
import psutil
import os
# import time


def urlread(url, blocksize=-1, decode=None, wrap_exceptions=True,
            raise_http_err=True, timeout=None, **kwargs):
    """
        Reads and return data from the given url. Wrapper around urllib2.open with some
        features added. Returns the tuple (content_read, status, message)

        :param url: (string or urllib2.Request) a valid url or an `urllib2.Request` object
        :param blockSize: int, default: -1. The block size while reading, -1 means:
        read entire content at once
        :param: decode: string or None, default: None. The string used for decoding to string
        (e.g., 'utf8'). If None, the result is returned as it is (type `bytes`, note that in
        Python2 this is equivalent to `str`), otherwise as unicode string (`str` in python3+)
        :param wrap_exceptions: if True (the default), all url-related exceptions
        ```urllib2.HTTPError, urllib2.URLError, httplib.HTTPException, socket.error```
        will be caught and wrapped into an `URLException` object E that will be raised
        (the original exception can always be retrieved via `E.exc`)
        :param raise_http_err: If True (the default) `urllib2.HTTPError`s will be treated as
        normal exceptions. Otherwise, they will treated as response object and the
        tuple (None, status, message) will be returned, where
        `status` (int) is the http status code (from the doc, most likely
        in the range 4xx-5xx) and `message` (string) is the string denoting the status message,
        respectively
        :param timeout: timeout parameter specifies a timeout in seconds for blocking operations
        like the connection attempt (if not specified, None or non-positive, the global default
        timeout setting will be used). This actually only works for HTTP, HTTPS and FTP connections.
        :param kwargs: optional arguments for `urllib2.urlopen` function (e.g., timeout=60)
        :return: the tuple (content_read, status code, status message), where the first item is
        the bytes sequence read (can be None if `raise_http_err=False`, is string - unicode in
        python2 - if `decode` is given, i.e. not None),
        the second item is the int denoting the http status code (int), and the third the http
        status message (string)
        :raise: `URLException` if `wrap_exceptions` is True. Otherwise any of the following:
        `urllib2.HTTPError`, `urllib2.URLError`, `httplib.HTTPException`, `socket.error`
        (the latter is the superclass of all `socket` exceptions such as `socket.timeout`)
    """
    try:
        ret = b''
        # set default for timeout: timeout in urlopen defaults to socket._GLOBAL_DEFAULT_TIMEOUT
        # so we unfortunately either pass it or skip it. As we allow for non-negative numbers
        # normalize it fiorst to None. If None, don't pass it to urlopen
        if timeout is None or timeout <= 0:
            timeout = None
        # urlib2 does not support with statement in py2. See:
        # http://stackoverflow.com/questions/3880750/closing-files-properly-opened-with-urllib2-urlopen
        # https://docs.python.org/2.7/library/contextlib.html#contextlib.closing
        with closing(urllib.request.urlopen(url, **kwargs) if timeout is None else
                     urllib.request.urlopen(url, timeout=timeout, **kwargs)) as conn:
            if blocksize < 0:  # https://docs.python.org/2.4/lib/bltin-file-objects.html
                ret = conn.read()
            else:
                while True:
                    buf = conn.read(blocksize)
                    if not buf:
                        break
                    ret += buf
        return ret.decode(decode) if decode else ret, conn.code, conn.msg
    except urllib.error.HTTPError as exc:
        if not raise_http_err:
            return None, exc.code, exc.msg
        else:
            if wrap_exceptions:
                raise URLException(exc)
            else:
                raise exc
    except (urllib.error.URLError, http.client.HTTPException, socket.error) as exc:
        if wrap_exceptions:
            raise URLException(exc)
        else:
            raise exc


class URLException(Exception):
    """Custom exceptions which wraps any url/http related exception:
    `urllib2.HTTPError`, `urllib2.URLError`, `httplib.HTTPException`, `socket.error`
    The original exception is retrievable via the `exc` attribute
    """
    def __init__(self, original_exception):
        self.exc = original_exception

    def __str__(self):
        """Represent this object as the original exception"""
        return str(self.exc)


def _mem_percent(process=None):
    return 0 if process is None else process.memory_percent()


def read_async(iterable, urlkey=None, max_workers=None, blocksize=1024*1024,
               decode=None, raise_http_err=True, timeout=None, max_mem_consumption=90,
               unordered=True,
               **kwargs):  # pylint:disable=too-many-arguments
    """
        Wrapper around `multiprocessing.pool.ThreadPool()` for downloading
        data from urls in `iterable` asynchronously (i.e., most likely faster).
        Each download is executed on a separate *worker thread*, yielding the result of each
        `url` read.

        Yields the tuple
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
          bit not both**. Note that `exc` is one of the following URL-related exceptions:
          ```urllib2.URLError, httplib.HTTPException, socket.error```
          Any other exception is raised and will stop the download
          - url: the original url (either string or Request object). If `iterable` is an iterable
          of `Request` objects or strings, then `url` is equal to `obj`

        Note that if `raise_http_err=False` then `urllib2.HTTPError` are treated as 'normal'
        response and will return a tuple where `data=None` and `status_code` is most likely greater
        or equal to 400

        Finally, this function can cleanly cancel yet-to-be-processed *worker threads* via Ctrl+C
        if executed from the command line. In the following we will simply refer to `urlread`
        to indicate the `urllib2.urlopen.read` function.

        :param iterable: an iterable of objects representing the urls addresses to be read
        (either strings or `urllib2.Request` objects). If the elements of `iterable` are neither
        strings nor Request objects, the `urlkey` argument (see below) must be specified to
        return valid url strings or Request objects
        :param urlkey: a function of one argument or None (the default) that is used to extract
        an url (string) or Request object from each `iterable` element. When None, it returns the
        argument, i.e. assumes that `iterable` is an iterable of valid url addresses or Request
        objects.
        :param max_workers: integer or None (the default) denoting the max workers of the
        `ThreadPoolExecutor`. When None, the theads allocated are relative to the machine cpu
        :param blocksize: integer defaulting to 1024*1024 specifying, when connecting to one of
        the given urls, the mximum number of bytes to be read at each call of `urlopen.read`.
        If the size argument is negative or omitted, read all data until EOF is reached
        :param decode: string or None (default: None) optional argument specifying if the content
        of the url must be decoded. None means: return the byte string as it was read. Otherwise,
        use this argument for string content (not bytes) by supplying a decoding, such as
        e.g. 'utf8'
        :param raise_http_err: boolean (True by default) tells whether `urllib2.HTTPError` should
        be raised as url-like exceptions and passed as the argument `exc` in `ondone`. When False,
        `urllib2.HTTPError`s are treated as 'normal' response and passed as the argument `result`
        in `ondone` as a tuple `(None, status_code, message)` (where `status_code` is most likely
        greater or equal to 400)
        :param timeout: timeout parameter specifies a timeout in seconds for blocking operations
        like the connection attempt (if not specified, None or non-positive, the global default
        timeout setting will be used). This actually only works for HTTP, HTTPS and FTP connections.
        :param max_mem_consumption: integer in ]0 100], default: 90. For big downloads with a log
        of elements in `iterable`, this function might kill the python program: Set the maximum
        memory percentage (90% by default): if the programs overcomes that percentage, it returns
        and raises a `MemoryError`: This lets the caller handle the case (e.g., closing db
        connection, saving what has been downloaded etcetera) without loosing data
        :param unordered: boolean (default False): tells whether the download results are yielded
        in the same order they are input in `iterable`, i.e. if the i-th download is relative
        to the i-th element of iterable. Theoretically, False (the default) might execute faster,
        the drawback being that the results are not guaranteed to be returned in the same order
        as `iterable`. (Although with `ThreadPool`s we could not
        assess it clearly. Maybe it holds `ProcessPool`s, maybe is due to chunksize)
        :param kwargs: optional keyword arguments passed to `urllib2.urlopen` function (except
        the `timeout` argument, see above). NOT TESTED. For info see
        https://docs.python.org/2/library/urllib2.html#urllib2.urlopen

        Notes:
        ======

        ThreadPool vs ThreadPoolExecutor
        --------------------------------

        This function changed from using `concurrent.futures.ThreadPoolExecutor` into
        the "old" `multiprocessing.pool.ThreadPool`: the latter consumes in most cases
        less memory (about 30% less), especially if `iterable` is not a list in memory but
        a python iterable (`concurrent.futures.ThreadPoolExecutor` allocates `set`s of
        `Future`s object, whereas `multiprocessing.pool.ThreadPool` seems just to execute
        each element in iterable)

        killing threads / handling exceptions
        -------------------------------------

        this function handles any kind of unexpected exception (particularly relevant in case of
        e.g., `KeyboardInterrupt`) or the case when `ondone` returns True, by canceling all worker
        threads before raising. As ThreadPoolExecutor returns (or raises) after all worker
        threads have finished, an internal boolean flag makes all remaining worker threads quit as
        soon as possible, making the function return (or raise) much more quickly
    """
    if not (max_mem_consumption > 0 and max_mem_consumption < 100):
        max_mem_consumption = -1

    process = psutil.Process(os.getpid()) if max_mem_consumption != -1 else None

    # flag for CTRL-C or cancelled tasks
    kill = False

    # function called from within urlread to check if go on or not
    def urlwrapper(obj):
        if kill:
            return None
        url = urlkey(obj) if urlkey is not None else obj
        try:
            return obj, \
                urlread(url, blocksize, decode, True, raise_http_err, timeout, **kwargs), None, url
        except URLException as urlexc:
            return obj, None, urlexc.exc, url

    tpool = ThreadPool(max_workers)
    imap = tpool.imap_unordered if unordered else tpool.imap  # (func, iterable, chunksize)
    # note above: chunksize argument for threads (not processes)
    # seems to slow down download. Omit the argument and leave chunksize=1 (default)

    try:
        # this try is for the keyboard interrupt, which will be caught inside the
        # as_completed below
        for result_tuple in map(urlwrapper, iterable):
            if process is not None:
                mem_percent = _mem_percent(process)
                if mem_percent > max_mem_consumption:
                    raise MemoryError("Memory overflow: %.2f%% (used) > %.2f%% (threshold)" %
                                      (mem_percent, max_mem_consumption))

            if kill:
                continue  # (for safety: should never happen as long as kill=True in 'except' below)
            yield result_tuple
    except:
        # According to this post:
        # http://stackoverflow.com/questions/29177490/how-do-you-kill-futures-once-they-have-started,
        # after a KeyboardInterrupt this method does not return until all
        # working threads have finished. Thus, we implement the urlreader._kill flag
        # which makes them exit immediately, and hopefully this function will return within
        # seconds at most. We catch  a bare except cause we want the same to apply to all
        # other exceptions which we might raise (see few line above)
        kill = True  # pylint:disable=protected-access
        # the time here before executing 'raise' below is the time taken to finish all threads.
        # Without the line above, it might be a lot (minutes, hours), now it is much shorter
        # (in the order of few seconds max) and the command below can be executed quickly:
        raise


def _ismainthread():
    """
    utility function for testing, returns True if we are currently executing in the main thread"""
    # see:
    # http://stackoverflow.com/questions/23206787/check-if-current-thread-is-main-thread-in-python
    return isinstance(threading.current_thread(), threading._MainThread)
