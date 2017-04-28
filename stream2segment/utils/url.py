'''
Created on Apr 15, 2017

@author: riccardo
'''
from contextlib import closing
import threading
import urllib2
import httplib
import socket
import concurrent.futures
import multiprocessing


def urlread(url, blocksize=-1, decode=None, wrap_exceptions=True,
            raise_http_err=True, **kwargs):
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
        # urlib2 does not support with statement in py2. See:
        # http://stackoverflow.com/questions/3880750/closing-files-properly-opened-with-urllib2-urlopen
        # https://docs.python.org/2.7/library/contextlib.html#contextlib.closing
        with closing(urllib2.urlopen(url, **kwargs)) as conn:
            if blocksize < 0:  # https://docs.python.org/2.4/lib/bltin-file-objects.html
                ret = conn.read()
            else:
                while True:
                    buf = conn.read(blocksize)
                    if not buf:
                        break
                    ret += buf
        return ret.decode(decode) if decode else ret, conn.code, conn.msg
    except urllib2.HTTPError as exc:
        if not raise_http_err:
            return None, exc.code, exc.msg
        else:
            if wrap_exceptions:
                raise URLException(exc)
            else:
                raise exc
    except (urllib2.URLError, httplib.HTTPException, socket.error) as exc:
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


def read_async(iterable, ondone, urlkey=None, max_workers=None, blocksize=1024*1024,
               decode=None, raise_http_err=True, **kwargs):  # pylint:disable=too-many-arguments
    """
        Wrapper around `concurrent.futures.ThreadPoolExecutor` for downloading asynchronously
        data from urls in `iterable`. Each download is executed on a separate *worker thread*,
        calling `ondone` *on the main thread* for each `url`, as soon as it has been read.
        This function **blocks and returns when all urls are read**, can cancel
        yet-to-be-processed *worker threads* (see `ondone` below), and supports Ctrl+C if executed
        via command line. In the following we will simply refer to `urlread` to indicate the
        `urllib2.urlopen.read` function.
        :param iterable: an iterable of objects representing the urls addresses to be read
        (either strings or `urllib2.Request` objects). If not
        strings nor Request objects, the `urlkey` argument should be specified as a function
        accepting an element of iterable and returning a valid url string or Request object from it
        :param ondone: a function *executed on the main thread* after `urlread` has completed.
        It is called with the following arguments:
        ```
            def ondone(obj, result, exc, url)
        ```
        where:

          - `obj` is the element of `iterable` which originated the `urlread` call
          - `result` is the result of `urlread`, if not None is the tuple
          ```(data, status_code, message)```
          where `data` is the data read (as bytes or string if `decode != None`), `status_code` is
          the integer denoting the status code (e.g. 200), and `messsage` the string denoting the
          status message (e.g., 'OK'). `data` can be None, e.g., when `raise_http_err=True`
          (see below)
          - exc is the exception raised by `urlread`, if any. **Either `result` or `exc` are None,
          bit not both**. Note that exc is one of the following URL-related exceptions:
          ```urllib2.URLError, httplib.HTTPException, socket.error```
          Any other exception is raised and will stop the download
          - url: the original url (either string or Request object)

        Note that if t`raise_http_err=False` then `urllib2.HTTPError` are treated as 'normal'
        response and will return a tuple where `data=None` and `status_code` is most likely greater
        or equal to 400

        if `ondone` returns True, then the download will stop. To know more, read the
        `killing threads / handling exceptions` section below

        :param cancel: a boolean (default False) telling if the returned value of `ondone`
        should be checked to cancel remaining downloads
        :param urlkey: a function of one argument or None (the default) that is used to extract
        an url (string) or Request object from each `iterable` element. When None, it returns the
        argument, i.e. assumes that `iterable` is an iterable of valid url addresses or Request
        objects.
        :param max_workers: integer or None (the default) denoting the max workers of the
        `ThreadPoolExecutor`. When None, the theads allocated are relative to the machine cpu
        :param blocksize: integer defaulting to 1024*1024 specifying, when connecting to one of
        the given urls, how many bytes have to be read at each call of `urlopen.read` function
        (less if the `urlopen.read` hits EOF before obtaining size bytes). If the size argument is
        negative or omitted, read all data until EOF is reached
        :param decode: string or None (default: None) optional argument specifying if the content
        of the url must be decoded. None means: return the byte string as it was read. Otherwise,
        use this argument for string content (not bytes) by supplying a decoding, such as
        e.g. 'utf8'
        :param raise_http_err: boolean (True by default) tells whether `urllib2.HTTPError` should
        be raised as url-like exceptions and passed as the argument `exc` in `ondone`. When False,
        `urllib2.HTTPError`s are treated as 'normal' response and passed as the argument `result`
        in `ondone` as a tuple `(None, status_code, message)` (where `status_code` is most likely
        greater or equal to 400)

        :param kwargs: optional keyword arguments passed to `urllib2.urlopen` function.
        NOT TESTED. However, you can provide e.g. `timeout=60`.
        See https://docs.python.org/2/library/urllib2.html#urllib2.urlopen

        killing threads / handling exceptions
        =====================================

        this function handles any kind of unexpected exception (particularly relevant in case of
        e.g., `KeyboardInterrupt`) or the case when `ondone` returns True, by canceling all worker
        threads before raising. As ThreadPoolExecutor returns (or raises) after all worker
        threads have finished, an internal boolean flag makes all remaining worker threads quit as
        soon as possible, making the function return (or raise) much more quickly
    """
    # flag for CTRL-C or cancelled tasks
    kill = False

    # function called from within urlread to check if go on or not
    def urlwrapper(obj, urlkey, blocksize, decode, raise_http_err, **kw):
        if kill:
            return None
        url = urlkey(obj)
        try:
            return obj, urlread(url, blocksize, decode, True, raise_http_err, **kw), None, url
        except URLException as urlexc:
            return obj, None, urlexc.exc, url

    if urlkey is None:
        urlkey = lambda obj: obj  # @IgnorePep8

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
            future_to_obj = (executor.submit(urlwrapper, obj, urlkey, blocksize, decode,
                                             raise_http_err, **kwargs) for obj in iterable)
            for future in concurrent.futures.as_completed(future_to_obj):
                # this is executed in the main thread (thus is thread safe):
                if kill:  # pylint:disable=protected-access
                    continue
                elif ondone(*future.result()) is True:
                    kill = True
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

# def read_async(iterable, ondone, cancel=False,
#                urlkey=lambda obj: obj,
#                max_workers=None, blocksize=1024*1024,
#                decode=None, **kwargs):  # pylint:disable=too-many-arguments
#     """
#         Wrapper around `concurrent.futures.ThreadPoolExecutor` for downloading asynchronously
#         data from urls in `iterable`. Each download is executed on a separate *worker thread*,
#         calling `ondone` *on the main thread* for each `url`, as soon as it has been read.
#         This function **blocks and returns when all urls are read**, can cancel
#         yet-to-be-processed *worker threads* (see `ondone` below), and supports Ctrl+C if executed
#         via command line. In the following we will simply refer to `urlread` to indicate the
#         `urllib2.urlopen.read` function.
#         :param iterable: an iterable of objects representing the urls addresses to be read. If not
#         strings representing url addresses, the `urlkey` argument should be specified in order to
#         extract a valid url string from each element of iterable
#         :param ondone: a function *executed on the main thread* after `urlread` has completed.
#         It is called with the following arguments:
#         ```
#             def ondone(obj, result, exc, cancelled)
#         ```
#         where:
# 
#           - `obj` is the element of `iterable` which originated the `urlread` call. If the source
#           url is needed, either you keep a track of `urlkey` or, even better, you should store it
#           in each `obj` before calling this method and specify `urlkey` accordingly
#           - `result` is the data read by `urlread`, as bytes or string (depending on the `decode`
#           argument). If None, either `urlread` did not downloaded succesfully, or the relative
#           *worker thread* was cancelled
#           - exc is the exception raised by `urlread`, if any. If None, then the download completed
#           succesfully ot the relative *worker thread* was cancelled. Note that exc is one of
#           the following URL-related exceptions:
#           ```urllib2.HTTPError, urllib2.URLError, httplib.HTTPException, socket.error```
#           Any other exception is raised and will stop the download
#           - cancelled: a boolean flag telling if the *worker thread* executing the download
#           has been cancelled. This happens only if `cancel=True` and the user returned a custom
#           function from a previous call to `ondone`, as explained below
# 
#         Returning a value from ondone
#         -----------------------------
#         `ondone` might raise any kind of exception. In this case, as well as when `urlread`
#         raises a non-url exception, or the user hits a CTRL-C key (if this function is run from
#         terminal), this function will return as fast as possible (no further call to `ondone` or
#         `oncanc` will be made) by means of an internal flag, and the exception will be raised.
#         If the user wants to cancel only a part of all
#         remaining worker threads, while still being notified where they are cancelled (thus,
#         keeping this function alive, then `cancel` should be set as True and `oncanc` should
#         return a function L of one argument that will be called with all not-yet
#         processed remaining objects of `iterable`. For any object such that `L(obj) =True`,
#         then the relative *worker thread* will be cancelled and `oncanc(obj)` will be
#         called. Note that returning `lambda obj: True` cancels *all* remaining downloads but,
#         contrary to an exception raised, `ondone(obj, None, None, True)` will be called for all
#         remaining objects
# 
#         :param cancel: a boolean (default False) telling if the returned value of `ondone`
#         should be checked to cancel remaining downloads
#         :param urlkey: a function of one argument or None (the default) that is used to extract
#         an url (string) from each `iterable` element. When None, it returns the argument,
#         i.e. assumes that `iterable` is an iterable of valid url addresses.
#         :param max_workers: integer or None (the default) denoting the max workers of the
#         `ThreadPoolExecutor`. When None, the htread allocated are relative to the machine cpu
#         :param blocksize: integer defaulting to 1024*1024 specifying, when connecting to one of
#         the given urls, how many bytes have to be read at each call of `urlopen.read` function
#         (less if the `urlopen.read` hits EOF before obtaining size bytes). If the size argument is
#         negative or omitted, read all data until EOF is reached
#         :param decode: string or None (default: None) optional argument specifying if the content
#         of the url must be decoded. None means: return the byte string as it was read. Otherwise,
#         use this argument for string content (not bytes) by supplying a decoding, such as
#         e.g. 'utf8'
#         :param kwargs: optional keyword arguments passed to `urllib2.urlopen` function.
#         NOT TESTED. However, you can provide e.g. `timeout=60`.
#         See https://docs.python.org/2/library/urllib2.html#urllib2.urlopen
# 
#         Note that this function handles any kind of exception by canceling all worker
#         threads before raising. Without this feature, any exception
#         (e.g., `KeyboardInterrupt`, or any exception not caught, i.e. non url-related exceptions)
#         would be raised after a potentially huge amount of time, as all worker threads must be
#         finished before this function returns. By setting an internal flag, when an exception
#         should be raised all remaining worker threads quit as soon as possible, making the
#         function return much more quickly before raising the relative exception
# 
#         :Example:
# 
#         ```
#         iterable = [{'url': 'blabla', ...}, ...]  # list of objects strings
# 
#         data = []
#         urlerrors = []
# 
#         def ondone(obj, res, exc, cancelled):
#             # this executed on the main thread, so it is safe append to datas and errors
#             # (however, due to the GIL only one python thread at a time is allowed to be run)
#             if exc is not None:
#                 # url-like exceptions
#                 urlerrors.append(exc)
#             elif res is not None:
#                 data.append(res)
#             else:
#                 # unexpected exception: stops all remaining threads and raise
#                 raise exc
# 
#         # read all urls. This will raise an unexpected exception, if any
#         urls_read_async(urls, ondone, urlkey=lambda obj: obj['url'], timeout=60)
# 
#         # now you can manipulate datas and errors. This line is executed once **all** urls have been
#         # visited (if the Ctrl+C keyboard has been pressed on the terminal, a flag terminates
#         # quickly all remaining threads)
# 
#         # Now we want to stop at the first url error, but we want to execute ondone for all
#         # objects (like e.g., displaying progress)
#         def ondone(obj, exc, res, cancelled):
#             if cancelled:
#                 # execute yoir code here, or return
#             elif exc is not None:
#                 # url-like exceptions
#                 urlerrors.append(exc)
#                 # cancel all remaining threads
#                 return lambda obj: True
#             else:
#                 # data surely not None
#                 data.append(res)
# 
#         # read all urls
#         urls_read_async(urls, ondone, cancel=True, urlkey=lambda obj: obj['url'])
#         ```
#     """
#     # working futures which cannot be cancelled set their url here so that maybe
#     # the urlread returns faster
#     cancelled_urls = set()
#     # flag for CTRL-C
#     kill = False
# 
#     # function called from within urlread to check if go on or not
#     def urlwrapper(*a, **kw):
#         if kill:
#             return None
#         return urlread(*a, **kw)
# 
#     # set for futures that should be cancelled but they couldnt in order to know how to
#     # process them when finished (they might NOT raise CancelledError)
#     cancelled_futures = set()
# 
#     # we experienced some problems if max_workers is None. The doc states that it is the number
#     # of processors on the machine, multiplied by 5, assuming that ThreadPoolExecutor is often
#     # used to overlap I/O instead of CPU work and the number of workers should be higher than the
#     # number of workers for ProcessPoolExecutor. But the source code seems not to set this value
#     # at all!! (at least in python2, probably in pyhton 3 is fine). So let's do it manually:
#     if max_workers is None:
#         max_workers = 5 * multiprocessing.cpu_count()
# 
#     # We can use a with statement to ensure threads are cleaned up promptly
#     with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#         try:
#             # Start the load operations and mark each future with its iterable item and URL
#             future_to_obj = {}
#             for obj in iterable:
#                 url = urlkey(obj)
#                 future_to_obj[executor.submit(urlwrapper, url, blocksize, decode, True,
#                                               **kwargs)] = (obj, url)
#             for future in concurrent.futures.as_completed(future_to_obj):
#                 # this is executed in the main thread (thus is thread safe):
#                 if kill:  # pylint:disable=protected-access
#                     continue
#                 ret = None
#                 obj, url = future_to_obj.pop(future)
#                 if future in cancelled_futures:
#                     ondone(obj, None, None, True)
#                     continue
#                 try:
#                     data = future.result()
#                 except concurrent.futures.CancelledError:
#                     ondone(obj, None, None, True)
#                 except URLException as urlexc:
#                     ret = ondone(obj, None, urlexc.exc, False)
#                 else:
#                     ret = ondone(obj, data, None, False)
# 
#                 if hasattr(ret, "__call__"):
#                     for future in future_to_obj:
#                         if future not in cancelled_futures and ret(future_to_obj[future][0]):
#                             future.cancel()
#                             if not future.cancelled():
#                                 # this might be let url return faster if stuck in a loop:
#                                 cancelled_urls.add(future_to_obj[future][1])
#                                 cancelled_futures.add(future)  # will be handled later
# 
#         except:
#             # According to this post:
#             # http://stackoverflow.com/questions/29177490/how-do-you-kill-futures-once-they-have-started,
#             # after a KeyboardInterrupt this method does not return until all
#             # working threads have finished. Thus, we implement the urlreader._kill flag
#             # which makes them exit immediately, and hopefully this function will return within
#             # seconds at most. We catch  a bare except cause we want the same to apply to all
#             # other exceptions which we might raise (see few line above)
#             kill = True  # pylint:disable=protected-access
#             # the time here before executing 'raise' below is the time taken to finish all threads.
#             # Without the line above, it might be a lot (minutes, hours), now it is much shorter
#             # (in the order of few seconds max) and the command below can be executed quickly:
#             raise


def _ismainthread():
    """
    utility function for testing, returns True if we are currently executing in the main thread"""
    # see:
    # http://stackoverflow.com/questions/23206787/check-if-current-thread-is-main-thread-in-python
    return isinstance(threading.current_thread(), threading._MainThread)
