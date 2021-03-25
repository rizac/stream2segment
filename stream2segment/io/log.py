"""
Class handling logger for downloading and processing

:date: Feb 20, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import object  # pylint: disable=redefined-builtin

import logging
from datetime import datetime
import sys


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


class SysOutStreamHandler(logging.StreamHandler):
    """Handler that prints to screen the logging messages.
    It implements a LevelFilter so that only special levels (not only levels
    "up to") are printed to screen. By default, these levels are 20 (info),
    40 (error) and 50 (critical)
    """
    def __init__(self, out=sys.stdout, levels=(20, 40, 50)):
        super(SysOutStreamHandler, self).__init__(out)
        self.setLevel(min(levels))
        # custom filtering: do not print certain levels (default: print info
        # and critical):
        self.addFilter(LevelFilter(levels))
        # this should be the default, but for safety set it again:
        self.setFormatter(logging.Formatter('%(message)s'))


def configlog4processing(logger, logfile_path='', verbose=False):
    """Configures the logger, setting it to a `INFO` level with a list of
    default handlers:

    - If `logfile_path` is given (not empty), a :class:`logging.FileHandler` (
      streaming to that file) will capture all messages of at least level INFO
      (e.g., INFO, WARNING, ERROR).
      See :func:`logfilepath` if you want to create automatically a log file
      path in the same directory of a given processing file.

    - If `verbose` = True, a :class:`StreamHandler` (streaming to standard
      output) will capture ONLY messages of level INFO (20) and ERROR (40) and
      CRITICAL (50), ideal for showing relevant information to the user on a
      terminal

    The returned list can thus contain 0, 1 or 2 loggers depending on the
    arguments.

    Implementation detail: this method modifies these values for performance
    reason:
    ```
    logging._srcfile = None
    logging.logThreads = 0
    logging.logProcesses = 0
    ```

    :return: a list of handlers added to the logger
    """
    # https://docs.python.org/2/howto/logging.html#optimization:
    logging._srcfile = None  # pylint: disable=protected-access
    logging.logThreads = 0
    logging.logProcesses = 0

    logger.setLevel(logging.INFO)  # necessary to forward to handlers
    handlers = []
    if logfile_path:
        handlers.append(logging.FileHandler(logfile_path, mode='w'))
    if verbose:
        handlers.append(SysOutStreamHandler(sys.stdout))
    for hand in handlers:
        logger.addHandler(hand)
    return handlers


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


def closelogger(logger):
    """Close all logger handlers and removes them from logger"""
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
