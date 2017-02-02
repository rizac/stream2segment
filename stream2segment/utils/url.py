'''
Created on Nov 18, 2016

@author: riccardo
'''
from contextlib import closing
import threading
import urllib2
import httplib
import socket
import concurrent.futures
# import time

# maybe used in the future
# CONNECTION_ERRORS = (urllib2.HTTPError, urllib2.URLError, httplib.HTTPException, socket.timeout)

# Retrieve a single page and report the url and contents
# def _load_url_default(url, timeout=60, blockSize=1024*1024, decode=None):
#     conn = urllib2.urlopen(url, timeout=timeout)
#     return conn.read().decode(decode) if decode else conn.read()


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


class _urlreader(object):

    def __init__(self):
        self._kill = False

    def __call__(self, url, blocksize=-1, decode=None, **kwargs):
        """Custom function which handles url read and checks the cancel flag"""
        # this is executed in a separate thread (thus not thread safe):
        if self._kill:
            return None
        ret = b''
        # urlib2 does not support with statement in py2. See:
        # http://stackoverflow.com/questions/3880750/closing-files-properly-opened-with-urllib2-urlopen
        # https://docs.python.org/2.7/library/contextlib.html#contextlib.closing
        with closing(urllib2.urlopen(url, **kwargs)) as conn:
            if blocksize < 0:  # https://docs.python.org/2.4/lib/bltin-file-objects.html
                ret = conn.read()
            else:
                while not self._kill:
                    buf = conn.read(blocksize)
                    if not buf:
                        break
                    ret += buf
        if self._kill:
            return None
        return ret.decode(decode) if decode else ret


def url_read(url, blocksize=-1, decode=None, on_exc=None, **kwargs):
    """
        Reads and return data from the given url. Returns the bytes read or the string read (if
        decode is specified). Returns None if on_exc is a callable and a specific "connection"
        exception is raised (see below), otherwise raises it
        :param url: a valid url or an urllib2.Request object
        :type url: string or urllib2.Request
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
    # The if branch below is a little bit verbose, but this way we preserve stack trace
    if not hasattr(on_exc, "__call__"):  # note: None evaluates to False
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


def read_async(iterable, ondone, oncanc=lambda *a, **v: None,
               urlkey=lambda obj: obj,
               max_workers=5, blocksize=1024*1024,
               decode=None, **kwargs):  # pylint:disable=too-many-arguments
    """
        Reads each url in `iterables` asynchronously on different *worker threads*, executing
        `ondone` *on the main thread* for each `url`, as soon as it has been read.
        This function uses the `ThreadPoolExecutor` class (standard in python 3.4+) and the urllib2
        python library (see https://docs.python.org/2/library/urllib2.html and
        https://docs.python.org/2/howto/urllib2.html)
        This function **blocks and returns when all urls are read**, can cancel
        yet-to-be-processed *worker threads* (see `ondone` below), and supports Ctrl+C if executed
        via command line. In the following we will simply refer to `urlread` to indicate the
        `urllib2.urlopen.read` function.
        :param iterable: an iterable of objects representing the urls addresses to be read. If not
        strings representing url addresses, the `urlkey` argument should be specified in order to
        extract a valid url string from each element of iterable
        :param ondone: a function *executed on the main thread* after `urlread` has completed.
        It is called with the following arguments:
        ```
            def ondone(obj, exception, result, url)
        ```
        where:
          - `obj` is the element of `iterable` which originated the `urlread` call
          - exception: the exception raised from `urlread`. If not None, `result` is a boolean
          telling if `exception` is an url-like exception:
          ```urllib2.HTTPError, urllib2.URLError, httplib.HTTPException, socket.error```
          or not. If None, then no exception was raised from `urlread` and `result` is the data
          read from the url as bytes or string (depending on the `decode` argument)
          - `result` is the data read by `urlread`, as bytes or string (depending on the `decode`
          argument) *if exception is None*.
          If exception is not None, it is a boolean telling if exception is a url-exception:
          ```urllib2.HTTPError, urllib2.URLError, httplib.HTTPException, socket.error```,
          False otherwise. Thus, `result` has two different purposes and returned types,
          because when exception is not None we need to know if the "result" of `urlread` was
          still good or not: an url-like exception (result=True) might most likely be
          handled differently (e.g., log or print the exception as a warning) than
          an unexpected exception (e.g., result=False: raise exception)
          - url: the url address (string) associated to `obj`. Might be equal to `obj` when
          no custom `urlkey` is provided.

        Returning a value from ondone
        -----------------------------
        `ondone` might raise any kind of exception. In this case, this function will return
        as fast as possible (no further call to `ondone` or `oncanc` will be made) before
        the exception is raised. This is an option if the user wants to break the download
        "brutally" and use this function in a try-catch statement to handle the `ondone` exception.
        On the other hand, the user might choose to cancel all or only a part of all
        remaining threads, while still being notified where they are cancelled (thus, keeping
        this function alive). In this case, the argument `oncanc` should be implemented
        (see below) and the return value of
        `ondone` should be a lambda function L of one argument that will be called with all not-yet
        processed remaining objects of `iterable`. For any object such that `L(obj) =True`,
        then the relative *worker thread* will be cancelled and `oncanc(obj)` will be
        called

        :param oncanc: a function, defaults to `lambda *a, **v: None` (noop). A funtion
        *executed on the main thread*
        of two arguments `obj`, `url`, where `obj` is the element of iterable whose worker thread
        has been cancelled and `url` is the relative url address (string). If no custom `urlkey` is
        provided, then `obj` = `url`.
        This function is called only because the user provided a return value of `ondone`.
        If `ondone` never returns a lambda function, then this function is never called
        :param urlkey: a function of one argument that is used to extract an url (string) from
        each `iterable` element. By default, returns the argument, i.e. assumes that `iterable`
        is an iterable of valid url addresses.
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

        Note that this function handles any kind of exception by canceling all worker
        threads before raising. Without this feature, any exception
        (e.g., `KeyboardInterrupt`, or any exception to be raised if `errors='raise'`) would
        be raised after a potentially huge amount of time, as all worker threads must be finished
        before this function returns. By setting an internal flag, when an exception should
        be raised all remaining worker threads quit as soon as possible, making the function
        return much more quickly before raising the relative exception

        :Example:

        ```
        iterable = [{'url': 'blabla', ...}, ...]  # list of objects strings

        datas = []
        urlerrors = []

        def ondone(obj, exc, res, *unused):
            # this executed on the main thread, so it is safe append to datas and errors
            # (however, due to the GIL only one python thread at a time is allowed to be run)
            if exc is None:
                datas.append(res)
            elif res:
                # url-like exceptions
                urlerrors.append(exc)
            else:
                raise exc

        # read all urls. This will raise an unexpected exception, if any
        urls_read_async(urls, ondone, urlkey=lambda obj: obj['url'], timeout=60)

        # now you can manipulate datas and errors. This line is executed once **all** urls have been
        # visited (if the Ctrl+C keyboard has been pressed on the terminal, a flag terminates
        # quickly all remaining threads)

        # suppose we have some sort of function F which must handle all objects of iterable,
        # but we still want unexpected errors to discard remaining threads
        # oncanc
        def ondone(obj, exc, res, *unused):
            # execute F here
            if exc is None:
                datas.append(res)
            elif res:
                # url-like exceptions
                urlerrors.append(exc)
            else:
                # unexpected, cancel all remaining threads
                return lambda obj: True

        def oncanc(obj, url):
            # execute F here

        # read all urls. This will never raise unless executed from terminal and Ctrl+C is hit
        urls_read_async(urls, ondone, oncanc, urlkey=lambda obj: obj['url'], timeout=60)
        ```
    """
    urlreader = _urlreader()  # for KeyboardInterrupt or other weird stuff
    pendingcancelledfutures = {}
    # We can use a with statement to ensure threads are cleaned up promptly
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        try:
            # Start the load operations and mark each future with its iterable item and URL
            future_to_obj = {}
            for obj in iterable:
                url = urlkey(obj)
                future_to_obj[executor.submit(urlreader, url, blocksize, decode, **kwargs)] =\
                    obj, url
            for future in concurrent.futures.as_completed(future_to_obj):
                # this is executed in the main thread (thus is thread safe):
                if urlreader._kill:  # pylint:disable=protected-access
                    continue
                ret = None
                obj, url = future_to_obj.pop(future)
                try:
                    if pendingcancelledfutures.pop(future, None) is not None:
                        raise concurrent.futures.CancelledError()
                    data = future.result()
                except concurrent.futures.CancelledError:  # might be the case, see below
                    oncanc(obj, url)
                except (urllib2.HTTPError, urllib2.URLError, httplib.HTTPException, socket.error) \
                        as urlexc:
                    ret = ondone(obj, urlexc, True, url)
                except Exception as exc:  # pylint:disable=broad-except
                    ret = ondone(obj, exc, False, url)
                else:
                    ret = ondone(obj, None, data, url)

                if hasattr(ret, "__call__"):
                    for future in future_to_obj:
                        if future not in pendingcancelledfutures and ret(future_to_obj[future][0]):
                            if not future.cancel():
                                pendingcancelledfutures[future] = True  # will be handled later

        except:
            # According to this post:
            # http://stackoverflow.com/questions/29177490/how-do-you-kill-futures-once-they-have-started,
            # after a KeyboardInterrupt this method does not return until all
            # working threads have finished. Thus, we implement the urlreader._kill flag
            # which makes them exit immediately, and hopefully this function will return within
            # seconds at most. We catch  a bare except cause we want the same to apply to all
            # other exceptions which we might raise (see few line above)
            urlreader._kill = True  # pylint:disable=protected-access
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

