"""
Class handling logger for downloading and processing

:date: Feb 20, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import object  # pylint: disable=redefined-builtin
import time
from datetime import datetime, timedelta


class LevelFilter(object):  # pylint: disable=too-few-public-methods
    """Logging filter that logs only messages in a set of levels (the base filter
    class only allows events which are below a certain point in the logger hierarchy).

    Usage `logger.addFilter(LevelFilter(20, 50, 50))`
    """

    # note: looking at the code, it seems that we do not need to inherit from
    # logging.Filter
    def __init__(self, levels):
        """Initialize a LevelFilter

        :param levels: iterable of `int`s representing different logging levels:
            ```
            CRITICAL 50
            ERROR    40
            WARNING  30
            INFO     20
            DEBUG    10
            NOTSET    0
            ```
        """
        self.levels = set(levels)

    def filter(self, record):
        """Filter record according to its level number"""
        return True if record.levelno in self.levels else False


def logfilepath(filepath):
    """Return a log file associated to the given `filepath`, i.e.:
    `filepath + "[now].log"` where [now] is the current date-time in ISO
    format, rounded to the closest second

    :param filepath: a file path serving as base for the log file path. The
        file does not need to exist but if you want to use the returned file
        for logging (the usual case), its parent directory must exist
    """
    _now = datetime.utcnow().replace(microsecond=0).isoformat()
    return filepath + (".%s.log" % _now)


def close_logger(logger):
    """Close all logger handlers and removes them from logger"""
    handlers = logger.handlers[:]
    for handler in handlers:
        try:
            handler.close()  # maybe already closed? pass in case
        except Exception:  # noqa
            pass
        logger.removeHandler(handler)


def elapsed_time(t0_sec, t1_sec=None):
    """Time elapsed from `t0_sec` until `t1_sec`, as `timedelta` object rounded
    to seconds. If `t1_sec` is None, it will default to `time.time()` (the
    current time since the epoch, in seconds)

    :param t0_sec: (float) the start time in seconds. Usually it is the result
        of a previous call to `time.time()`, before starting a process that
        had to be monitored
    :param t1_sec: (float) the end time in seconds. If None, it defaults to
        `time.time()` (current time since the epoch, in seconds)

    :return: a timedelta object, rounded to seconds
    """
    return timedelta(seconds=round((time.time() if t1_sec is None else t1_sec) - t0_sec))


