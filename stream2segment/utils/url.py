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
import multiprocessing


def urlread(url, blocksize=-1, decode=None, urlexc=True, killfunc=None, **kwargs):
    """
        Reads and return data from the given url, featuring some afvanced options.
        Returns the bytes read or the string read (if
        decode is specified). Returns None if killfunc is a callable which will
        evaluate to True somewhere the function

        :param url: (string or urllib2.Request) a valid url or an urllib2.Request object
        :param blockSize: int, default: -1. The block size while reading, -1 means:
            read entire content at once
        :param: decode: string or None, default: None. The string used for decoding to string
        (e.g., 'utf8'). If None, the result is returned as it is (byte string, note that in
        Python2 this is equivalent to string), otherwise as unicode string
        :param urlexc: if True (the default), all url-related exceptions
        ```urllib2.HTTPError, urllib2.URLError, httplib.HTTPException, socket.error```
        will be caught and an `URLException` will be raised (the original exception can always
        be retrieved via 'URLException.exc')
        :param killfunc: (function or None). Advanced parameter (in most the cases, this argument
        is not relevant for the user, drop it or set it to None) used for
        multi-threading when the user wants to kill worker threads rapidly
        A flag-function which is called repeatedly with url as argument to check if the
        function should return. Used by `read_async`
        :param kwargs: optional arguments for `urllib2.urlopen` function (e.g., timeout=60)
        :return: the bytes read. If on_exc is a callable and an IOException is raised, returns None
        :rtype bytes of data (equivalent to string in python2), or unicode string, or the tuple
        bytes of data or unicode string, exception (the latter might be None)
        :raise: `URLException` if `urlexc` is True. Otherwise
        `urllib2.HTTPError`, `urllib2.URLError`, `httplib.HTTPException`, `socket.error``
        (the latter is the superclass of all `socket` exceptions such as `socket.timeout` etcetera)
    """
    try:
        # this is executed in a separate thread (thus not thread safe):
        if killfunc is not None and killfunc(url):
            return None
        ret = b''
        # urlib2 does not support with statement in py2. See:
        # http://stackoverflow.com/questions/3880750/closing-files-properly-opened-with-urllib2-urlopen
        # https://docs.python.org/2.7/library/contextlib.html#contextlib.closing
        with closing(urllib2.urlopen(url, **kwargs)) as conn:
            if blocksize < 0:  # https://docs.python.org/2.4/lib/bltin-file-objects.html
                ret = conn.read()
            else:
                while killfunc is None or not killfunc(url):
                    buf = conn.read(blocksize)
                    if not buf:
                        break
                    ret += buf
        if killfunc is not None and killfunc(url):
            return None
        return ret.decode(decode) if decode else ret
    except (urllib2.HTTPError, urllib2.URLError, httplib.HTTPException, socket.error) as exc:
        if urlexc:
            raise URLException(exc)
        else:
            raise exc


class URLException(Exception):
    def __init__(self, original_exception):
        self.exc = original_exception

    def __str__(self):
        return str(self.exc)


def read_async(iterable, ondone, cancel=False,
               urlkey=lambda obj: obj,
               max_workers=None, blocksize=1024*1024,
               decode=None, **kwargs):  # pylint:disable=too-many-arguments
    """
        Wrapper around `concurrent.futures.ThreadPoolExecutor` for downloading asynchronously
        data from urls in `iterable`. Each download is executed on a separate *worker thread*,
        calling `ondone` *on the main thread* for each `url`, as soon as it has been read.
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
            def ondone(obj, result, exc, cancelled)
        ```
        where:

          - `obj` is the element of `iterable` which originated the `urlread` call. If the source
          url is needed, either you keep a track of `urlkey` or, even better, you should store it
          in each `obj` before calling this method and specify `urlkey` accordingly
          - `result` is the data read by `urlread`, as bytes or string (depending on the `decode`
          argument). If None, either `urlread` did not downloaded succesfully, or the relative
          *worker thread* was cancelled
          - exc is the exception raised by `urlread`, if any. If None, then the download completed
          succesfully ot the relative *worker thread* was cancelled. Note that exc is one of
          the following URL-related exceptions:
          ```urllib2.HTTPError, urllib2.URLError, httplib.HTTPException, socket.error```
          Any other exception is raised and will stop the download
          - cancelled: a boolean flag telling if the *worker thread* executing the download
          has been cancelled. This happens only if `cancel=True` and the user returned a custom
          function from a previous call to `ondone`, as explained below

        Returning a value from ondone
        -----------------------------
        `ondone` might raise any kind of exception. In this case, as well as when `urlread`
        raises a non-url exception, or the user hits a CTRL-C key (if this function is run from
        terminal), this function will return as fast as possible (no further call to `ondone` or
        `oncanc` will be made) by means of an internal flag, and the exception will be raised.
        If the user wants to cancel only a part of all
        remaining worker threads, while still being notified where they are cancelled (thus,
        keeping this function alive, then `cancel` should be set as True and `oncanc` should
        return a function L of one argument that will be called with all not-yet
        processed remaining objects of `iterable`. For any object such that `L(obj) =True`,
        then the relative *worker thread* will be cancelled and `oncanc(obj)` will be
        called. Note that returning `lambda obj: True` cancels *all* remaining downloads but,
        contrary to an exception raised, `ondone(obj, None, None, True)` will be called for all
        remaining objects

        :param cancel: a boolean (default False) telling if the returned value of `ondone`
        should be checked to cancel remaining downloads
        :param urlkey: a function of one argument or None (the default) that is used to extract
        an url (string) from each `iterable` element. When None, it returns the argument,
        i.e. assumes that `iterable` is an iterable of valid url addresses.
        :param max_workers: integer or None (the default) denoting the max workers of the
        `ThreadPoolExecutor`. When None, the htread allocated are relative to the machine cpu
        :param blocksize: integer defaulting to 1024*1024 specifying, when connecting to one of
        the given urls, how many bytes have to be read at each call of `urlopen.read` function
        (less if the `urlopen.read` hits EOF before obtaining size bytes). If the size argument is
        negative or omitted, read all data until EOF is reached
        :param decode: string or None (default: None) optional argument specifying if the content
        of the url must be decoded. None means: return the byte string as it was read. Otherwise,
        use this argument for string content (not bytes) by supplying a decoding, such as
        e.g. 'utf8'
        :param kwargs: optional keyword arguments passed to `urllib2.urlopen` function.
        NOT TESTED. However, you can provide e.g. `timeout=60`.
        See https://docs.python.org/2/library/urllib2.html#urllib2.urlopen

        Note that this function handles any kind of exception by canceling all worker
        threads before raising. Without this feature, any exception
        (e.g., `KeyboardInterrupt`, or any exception not caught, i.e. non url-related exceptions)
        would be raised after a potentially huge amount of time, as all worker threads must be
        finished before this function returns. By setting an internal flag, when an exception
        should be raised all remaining worker threads quit as soon as possible, making the
        function return much more quickly before raising the relative exception

        :Example:

        ```
        iterable = [{'url': 'blabla', ...}, ...]  # list of objects strings

        data = []
        urlerrors = []

        def ondone(obj, res, exc, cancelled):
            # this executed on the main thread, so it is safe append to datas and errors
            # (however, due to the GIL only one python thread at a time is allowed to be run)
            if exc is not None:
                # url-like exceptions
                urlerrors.append(exc)
            elif res is not None:
                data.append(res)
            else:
                # unexpected exception: stops all remaining threads and raise
                raise exc

        # read all urls. This will raise an unexpected exception, if any
        urls_read_async(urls, ondone, urlkey=lambda obj: obj['url'], timeout=60)

        # now you can manipulate datas and errors. This line is executed once **all** urls have been
        # visited (if the Ctrl+C keyboard has been pressed on the terminal, a flag terminates
        # quickly all remaining threads)

        # Now we want to stop at the first url error, but we want to execute ondone for all
        # objects (like e.g., displaying progress)
        def ondone(obj, exc, res, cancelled):
            if cancelled:
                # execute yoir code here, or return
            elif exc is not None:
                # url-like exceptions
                urlerrors.append(exc)
                # cancel all remaining threads
                return lambda obj: True
            else:
                # data surely not None
                data.append(res)

        # read all urls
        urls_read_async(urls, ondone, cancel=True, urlkey=lambda obj: obj['url'])
        ```
    """
    # working futures which cannot be cancelled set their url here so that maybe
    # the urlread returns faster
    cancelled_urls = set()
    # flag for CTRL-C
    kill = False

    # function called from within urlread to check if go on or not
    def killf(url):
        return kill or (url in cancelled_urls)

    # set for futures that should be cancelled but they couldnt in order to know how to
    # process them when finished (they might NOT raise CancelledError)
    cancelled_futures = set()

    # we experienced some problems if max_workers is None. The doc states that it is the number
    # of processors on the machine, multiplied by 5, assuming that ThreadPoolExecutor is often
    # used to overlap I/O instead of CPU work and the number of workers should be higher than the
    # number of workers for ProcessPoolExecutor. But the source code seems not to set this value
    # at all!! (at least in python2, probably in pyhton 3 is fine). So let's do it manually:
    if max_workers is None:
        max_workers = 5 * multiprocessing.cpu_count()

    # We can use a with statement to ensure threads are cleaned up promptly
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        try:
            # Start the load operations and mark each future with its iterable item and URL
            future_to_obj = {}
            for obj in iterable:
                url = urlkey(obj)
                future_to_obj[executor.submit(urlread, url, blocksize, decode, True,
                                              killf, **kwargs)] = (obj, url)
            for future in concurrent.futures.as_completed(future_to_obj):
                # this is executed in the main thread (thus is thread safe):
                if kill:  # pylint:disable=protected-access
                    continue
                ret = None
                obj, url = future_to_obj.pop(future)
                if future in cancelled_futures:
                    ondone(obj, None, None, True)
                    continue
                try:
                    data = future.result()
                except concurrent.futures.CancelledError:
                    ondone(obj, None, None, True)
                except URLException as urlexc:
                    ret = ondone(obj, None, urlexc.exc, False)
                else:
                    ret = ondone(obj, data, None, False)

                if hasattr(ret, "__call__"):
                    for future in future_to_obj:
                        if future not in cancelled_futures and ret(future_to_obj[future][0]):
                            future.cancel()
                            if not future.cancelled():
                                # this might be let url return faster if stuck in a loop:
                                cancelled_urls.add(future_to_obj[future][1])
                                cancelled_futures.add(future)  # will be handled later

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

