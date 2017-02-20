'''
Created on Feb 15, 2017

@author: riccardo
'''
import concurrent.futures
# from concurrent.futures import Future
from concurrent.futures import CancelledError
import multiprocessing
import time


def run_async(iterable, func, ondone, func_posargs=None, func_kwargs=None,
              use_thread=False, max_workers=5, timeout=None):
    """
        Executes asynchronously
        `func(obj, *func_posargs, **func_kwargs)` for each obj in iterable in separate
        threads/processes and calls on the main thread/process 
        `ondone` with the results of the functions as they are completed. This functions
        uses the `ProcessPoolExecutor` or  the `ThreadPoolExecutor` class (standard in python 3.4+):
        it blocks until all `func`s are executed, can cancel
        yet-to-be-processed processes/threads (see `ondone` below),
        and supports nicely Ctrl+C if executed via command line.

        :param iterable: an iterable of objects to be passed as first
        argument to `func`
        :param func: the function to be executed on a separate process/thread. IT MUST BE
        PICKABLE (normal module level functions are pickable) if processes are used.
        It is called with each iterable element as first argument, and optional positional
        and keyword arguments `func_posargs` and `func_kwargs`
        :param ondone: a function *executed on the main thread/process* after `func` has completed.
        It is called with the following arguments:
        ```
            def ondone(obj, future)
        ```
        where:
          - `obj` is the element of `iterable` passed as first argument of `func`
          - `future` is a Future-like object
             (https://docs.python.org/3/library/concurrent.futures.html#future-objects) for
             compatibility with the python libray implementing the following methods:
             - `future.result()`: Return the value returned by `func`. If the future is cancelled
               before completing then `concurrent.futures.CancelledError` will be raised.
               If `func` raised, this method will raise the same exception.
             - `future.exception()`: Return the exception raised by `func`. If the future is
               cancelled
               before completing then `concurrent.futures.CancelledError` will be raised.
               If `func` completed without raising, None is returned.
             - `future.cancelled()`: Return True if the call was successfully cancelled, see
               `ondone` return value below
            (for compatibility with the `Future` object, also `future.running()` and `future.done()`
            are available, although they should be useless as the `Future` has completed and thus
            be always False and True, respectively)

        Returning a value from ondone
        -----------------------------
        `ondone` might raise any kind of exception. In this case, this function will return
        as fast as possible (no further call to `ondone` will be made) before
        the exception is raised. This is an option if the user wants to break the run
        "brutally".
        On the other hand, the user might choose to cancel all
        remaining processes/threads, while still being notified where they are cancelled
        (thus, keeping this function alive). In this case,
        `ondone` should return True. From that point on, `ondone` will be always called with
        a future-like object where `future.cancelled()` will be True
        One can also return a function for selecting which, among remaining threads/subprocesses,
        should be cancelled: in this case, the function should have a single argument and will
        be called with all remaining objects
        of `iterable`: for any object for which it returns True, the thread/subprocess associated
        to that object will be cancelled (Note that already completed threads/subprocesses not
        already processed by `ondone` will have the `cancelled()` method set as True although the
        underlying Future object was not cancelled)

        :param func_posargs: a list of positional arguments to be passed to `func`
        :param func_kwargs: a dict of keyword arguments to be passed to `func`
        :param use_thread: boolean (False by default): Use `ThreadPoolExecutor` instead of
        `ProcessPoolExecutor`
        :param max_workers: integer defaulting to 5 denoting the max workers of the
        `ThreadPoolExecutor` or `ProcessPoolExecutor`
        :param timeout: the timeout to be set for each worker processes/threads to finish. For
        each `ondone` completed if the next `func` result is not available after timeout seconds
        from the start of the process/thread pool then a `concurrent.futures.TimeoutError` is
        raised, thus, in principle, stopping remaining threads / subprocesses. THIS FUNCTION HAS
        NOT BEEN TESTED, SETTING A VALUE DIFFERENT THAN None is deprecated

        Note that this function handles any kind of exception by canceling all
        processes/threads before raising. Without this feature, any exception
        (e.g., `KeyboardInterrupt`) would
        be raised after a potentially huge amount of time, as all processes/threads must be finished
        before this function returns. By setting an internal flag, when an exception should
        be raised all remaining processes/threads quit as soon as possible, making the function
        return much more quickly before raising the relative exception
    """
    pool_executor = concurrent.futures.ThreadPoolExecutor if use_thread else \
        concurrent.futures.ProcessPoolExecutor
    pendingcancelledfutures = {}

    mngr = multiprocessing.Manager()
    lock = mngr.Lock()
    func_wrap = wrapper(lock)

    args, kwargs = func_posargs or [], func_kwargs or {}

    # We can use a with statement to ensure threads/subprocesses are cleaned up promptly
    with pool_executor(max_workers=max_workers) as executor:
        try:
            # Start the load operations and mark each future with its iterable item
            future_to_obj = {executor.submit(func_wrap, func, obj, *args, **kwargs): obj
                             for obj in iterable}
            for future in concurrent.futures.as_completed(future_to_obj, timeout):
                # this is executed in the main thread (thus is thread safe):
                if func_wrap._kill:
                    continue

                obj = future_to_obj.pop(future)
                cancelled = None if pendingcancelledfutures.pop(future, None) is None else True
                ret = ondone(obj, future if cancelled is None else FutureLikeObj(future, cancelled))

                if ret is True:
                    ret = lambda obj: True  # @IgnorePep8
                elif not hasattr(ret, "__call__"):
                    continue

                for future in future_to_obj:
                    if future.cancelled() or future in pendingcancelledfutures:
                        continue
                    if ret(future_to_obj[future]) is True:
                        future.cancel()
                        if not future.cancelled():
                            pendingcancelledfutures[future] = True  # will be handled later

        except:
            # According to this post:
            # http://stackoverflow.com/questions/29177490/how-do-you-kill-futures-once-they-have-started,
            # after a KeyboardInterrupt this method does not return until all
            # working threads/processes have finished. Thus, we implement the urlreader._kill flag
            # which makes them exit immediately, and hopefully this function will return within
            # seconds at most. We catch  a bare except cause we want the same to apply to all
            # other exceptions which we might raise (see few line above)
            func_wrap.kill()  # pylint:disable=protected-access
            # the time here before executing 'raise' below is the time taken to finish all threads/
            # processes.
            # Without the line above, it might be a lot (minutes, hours), now it is much shorter
            # (in the order of few seconds max) and the command below can be executed quickly:
            raise


class FutureLikeObj(object):

    def __init__(self, future, force_cancelled=None):
        self._future = future
        self._force_cancelled = True if force_cancelled is True else False

    def cancelled(self):
        return self._force_cancelled or self._future.cancelled()

    def exception(self):
        if self._force_cancelled:
            return CancelledError()
        return self._future.exception()

    def result(self):
        if self._force_cancelled:
            raise CancelledError()
        return self._future.result()

    def running(self):
        return self._future.running()

    def done(self):
        return self._future.done()


class wrapper(object):

    def __init__(self, lock):
        self._kill = False
        self._lock = lock

    def kill(self):
        with self._lock:
            print "killing"
            self._kill = True

    def __call__(self, func, iterable_elm, *args, **kwargs):
        with self._lock:
            if self._kill:
                print "killed"
                return
        return func(iterable_elm, *args, **kwargs)
