'''
Created on Nov 18, 2016

@author: riccardo
'''


import concurrent.futures
from contextlib import closing
import threading
import urllib2
import httplib
import socket
# import time

# maybe used in the future
CONNECTION_ERRORS = (urllib2.HTTPError, urllib2.URLError, httplib.HTTPException, socket.timeout)


# Retrieve a single page and report the url and contents
def _load_url_default(url, timeout=60, blockSize=1024*1024, decode=None):
    conn = urllib2.urlopen(url, timeout=timeout)
    return conn.read().decode(decode) if decode else conn.read()


def _urlread(url, blocksize=-1, decode=None, **kwargs):
    ret = b''
    with closing(urllib2.urlopen(url, **kwargs)) as conn:
        if blocksize < 0:  # https://docs.python.org/2.4/lib/bltin-file-objects.html
            ret = conn.read()
        else:
            while True:
                buf = conn.read(blocksize)
                if not buf:
                    break
                ret += buf
    return ret.decode(decode) if decode else ret


def url_read(url, blocksize=-1, decode=None, on_exc=None, **kwargs):
    """
        Reads and return data from the given url. Returns the bytes read or the string read (if
        decode is specified). Returns None if on_exc is a callable and a specific "connection"
        exception is raised (see below), otherwise raises it
        :param url: a valid url
        :type url: string
        :param blockSize: int, default: -1. The block size while reading, -1 means:
            read entire content at once
        :param: decode: string or None, default: None. The string used for decoding to string
        (e.g., 'utf8'). If None, the result is returned as it is (byte string, note that in
        Python2 this is equivalent to string), otherwise as unicode string
        :param on_exc: callable or None, default: None. A callable which has a single argument,
        the exception thrown, that will be called in case of IOExceptions. Any other exception
        will be raised normally. If None, also IOExceptions will be raised normally
        :param kwargs: optional arguments for `urllib2.urlopen` function (e.g., timeout=60)
        :return: the bytes read. If on_exc is a callable and an IOException is raised, returns None
        :rtype bytes of data (equivalent to string in python2), or unicode string, or the tuple
        bytes of data or unicode string, exception (the latter might be None)
        :raise: `urllib2.HTTPError`, `urllib2.URLError`, `httplib.HTTPException`, `socket.error``
        (the latter is the superclass of all `socket` exceptions such as `socket.timeout` etcetera)
        Note that socket errors have to be caught because they are not included in the other
        exceptions (not in all python versions).
        See https://docs.python.org/2/howto/urllib2.html#handling-exceptions for more information,
        especially on error codes for the urllib2 module.
        in case of connection errors
    """
    if on_exc is None or not hasattr(on_exc, "__call__"):
        return _urlread(url, blocksize, decode, **kwargs)
    else:
        try:
            return _urlread(url, blocksize, decode, **kwargs)
        except (urllib2.HTTPError, urllib2.URLError, httplib.HTTPException, socket.error) as exc:
            # http://stackoverflow.com/questions/666022/what-errors-exceptions-do-i-need-to-handle-with-urllib2-request-urlopen
            # Note that socket.timeout (subclass of socket.error) is not an urllib2.URLError
            # in py2.7. See:
            # http://stackoverflow.com/questions/2712524/handling-urllib2s-timeout-python
            on_exc(exc)
            return None


def read_async(urls, onsuccess, onerror, max_workers=5, blocksize=1024*1024, decode=None,
               **kwargs):
    """
        Reads each url in `urls` asynchronously on different *worker threads*, executing `onsuccess`
        or `onerror` *on the main thread* for each `url`, as soon as it has been read.
        This function uses the `ThreadPoolExecutor` class (standard in python 3.4+) and the urllib2
        python library (see https://docs.python.org/2/library/urllib2.html and
        https://docs.python.org/2/howto/urllib2.html)
        This function **blocks and returns when all urls are read** or killed (see `onsuccess`
        below)
        :param urls: an iterable of strings representing the urls addresses to be read. (Note that
        if `dict`, its keys must be url strings and are iterated over)
        :param onsuccess: a function *executed on the main thread* after an url has been
        successfully read. It is called with three arguments: `data`, `url` and `index`, where
        `data` is the data read from `url`, `index` is an incrementing integer
        (``index < len(urls)``) useful for, e.g., displaying progress. Note that due to
        multi-threading it is **not** guaranteed that `url` is the index-th element obtained by
        iterating over `urls`.
        `onsuccess` needs not to return a value. However, if it returns False then all working
        threads **are** (kind of) **killed** : more precisely as all worker threads have to be
        finished, a "global" flag is set which tells all remaining working threads to quit without
        performing any job and **not** to call `onsuccess` and `onerror` anymore.
        Thus, returning False does not guarantee that this function will return immediately
        (more probably, "very soon")
        If `onsuccess` raises an exception it behaves as if it returned False. The exception is
        propagated and must be handled outside this function.
        :param onerror: a function *executed on the main thread* after an url has *not* been
        successfully read, i.e. when one of the following exceptions is raised:
        ```urllib2.HTTPError, urllib2.URLError, httplib.HTTPException, socket.error```
        (the latter is the superclass of all `socket` exceptions such as `socket.timeout` etcetera)
        Note that socket errors have to be caught because they are not included in the other
        exceptions (not in all python versions).
        See https://docs.python.org/2/howto/urllib2.html#handling-exceptions for more information,
        especially on error codes for the urllib2 module.
        It behaves exactly as `onsuccess` above with one difference: as first argument of this
        function is passed the exception raised (one of the above) and not the data read.
        :param max_workers: integer defaulting to 5 denoting the max workers of the
        `ThreadPoolExecutor`
        :param blocksize: integer defaulting to 1024*1024 specifying, when connecting to one of
        the given urls, how many bytes have to be read at each call of `urlopen.read` function
        (less if the `urlopen.read` hits EOF before obtaining size bytes). If the size argument is
        negative or omitted, read all data until EOF is reached
        :param decode: string or None (default: None) optional argument specifying if the content of
        the url must be decoded. None means: return the byte string as it was read. Otherwise, use
        this argument for string content (not bytes) by supplying a decoding, such as e.g. 'utf8'
        :param kwargs: optional keyword arguments passed to `urllib2.urlopen` function. For
        instance, `timeout=60`. See https://docs.python.org/2/library/urllib2.html#urllib2.urlopen

        :Example:

        ```
        urls = [...]  # list of url strings

        datas = []
        errors = []

        def onsuccess(data, url, index):
            # executed on the main thread, so it is safe the following:
            datas.append(data)
            print "%d of %d done" % (index+1, len(urls))

        def onerror(error, url, index):
            # executed on the main thread, so it is safe the following:
            errors.append(error)
            print "%d of %d done" % (index+1, len(urls))

        # read all urls (wrapping in a simple and relatively common try-catch statement)
        try:
            urls_read_async(urls, onsuccess, onerror, timeout=60)
        except KeyboardInterrupt:
            print "interrupted by user"

        # now you can manipulate datas and errors. This line is executed once **all** urls have been
        # visited or the Ctrl+C keyboard has been pressed on the terminal
        ```
    """
    # flag to "kill" threads. See
    # http://stackoverflow.com/questions/29177490/how-do-you-kill-futures-once-they-have-started
    cancel = False

    def __urlread__(url, blocksize, decode, **kwargs):
        """Custom function which handles url read and checks the cancel flag"""
        # this is executed in a separate thread (thus not thread safe):
        if cancel:
            return
        ret = b''
        # urlib2 does not support with statement in py2. See:
        # http://stackoverflow.com/questions/3880750/closing-files-properly-opened-with-urllib2-urlopen
        # https://docs.python.org/2.7/library/contextlib.html#contextlib.closing
        with closing(urllib2.urlopen(url, **kwargs)) as conn:
            if blocksize < 0:  # https://docs.python.org/2.4/lib/bltin-file-objects.html
                ret = conn.read()
            else:
                while not cancel:
                    buf = conn.read(blocksize)
                    if not buf:
                        break
                    ret += buf
        if cancel:
            return
        return ret.decode(decode) if decode else ret

    try:
        # We can use a with statement to ensure threads are cleaned up promptly
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Start the load operations and mark each future with its URL
            future_to_url = {executor.submit(__urlread__, url, blocksize, decode, **kwargs): url
                             for url in urls}
            for index, future in enumerate(concurrent.futures.as_completed(future_to_url)):
                # this is executed in the main thread (thus is thread safe):
                if cancel:
                    continue
                url = future_to_url[future]
                try:
                    data = future.result()
                except concurrent.futures.CancelledError:  # we should never fall here. However,
                    # a future that has been cancelled should be skipped silently
                    pass
                except (urllib2.HTTPError, urllib2.URLError, httplib.HTTPException,
                        socket.error) as exc:
                    # http://stackoverflow.com/questions/666022/what-errors-exceptions-do-i-need-to-handle-with-urllib2-request-urlopen
                    # Note that socket.timeout is not an urllib2.URLError in py2.7. See:
                    # http://stackoverflow.com/questions/2712524/handling-urllib2s-timeout-python
                    ret = onerror(exc, url, index)
                else:
                    ret = onsuccess(data, url, index)
                if ret is False:
                    cancel = True
    except:
        cancel = True  # this should in principle make all pending working threads skip their job,
        # and return almost immediately. This is because all worker threads have to be finished
        # before my main thread (this one) can exit. See:
        # http://stackoverflow.com/questions/29177490/how-do-you-kill-futures-once-they-have-started
        raise


def _ismainthread():
    """
    utility function for testing, returns True if we are currently executing in the main thread"""
    # see:
    # http://stackoverflow.com/questions/23206787/check-if-current-thread-is-main-thread-in-python
    return isinstance(threading.current_thread(), threading._MainThread)

