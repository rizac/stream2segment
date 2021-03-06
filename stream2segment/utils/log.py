"""
Class handling logger for downloading and processing

:date: Feb 20, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import object  # pylint: disable=redefined-builtin

import os
import logging
from datetime import timedelta, datetime
import sys

from stream2segment.io.db.models import Download

# CRITICAL    50
# ERROR    40
# WARNING    30
# INFO    20
# DEBUG    10
# NOTSET    0


class DbStreamHandler(logging.FileHandler):
    """A `logging.FileHandler` which counts errors and warnings. See
    https://stackoverflow.com/q/812477. This class takes in the constructor an
    id of the table 'downloads' (referring to the current download), and when
    closed writes the content of the file to the database, deleting the
    handler's file. You should always explicitly call close() to assure the log
    is written to the database**. For an example using SQL-Alchemy log rows
    (slightly different case but informative) see:
    http://docs.pylonsproject.org/projects/pyramid_cookbook/en/latest/logging/sqlalchemy_logger.html
    """
    def __init__(self, filepath, min_level=20):
        """
        Initialize a DbStreamHandler

        :param min_level: this handlers level
        (https://docs.python.org/3/library/logging.html#logging.Handler.setLevel)
        """
        # w+: allows to read without closing first:
        super(DbStreamHandler, self).__init__(filepath, mode='w+')
        # access the stream with self.stream
        self.errors = 0
        self.warnings = 0
        self.criticals = 0  # one should be enough
        # configure level and formatter
        self.setLevel(min_level)
        self.setFormatter(logging.Formatter('[%(levelname)s]  %(message)s'))

    def emit(self, record):
        if record.levelno == 30:
            self.warnings += 1
        elif record.levelno == 40:
            self.errors += 1
        elif record.levelno == 50:
            self.criticals += 1
        super(DbStreamHandler, self).emit(record)
        # (superclass logging.FileHandler flushes every emit)

    def finalize(self, session, download_id, removefile=True):
        """Write to db, closes this handler
        and optionally removes the underlying file"""
        # the super-class sets the stream to None when closing, so we might
        # check this to see if we closed it already:
        if self.stream is None:
            return
        # we experienced the NoneType error which we could not test
        # deterministically so the if above serves to this, especially because
        # we know self.stream == None => already closed

        super(DbStreamHandler, self).flush()  # for safety
        self.stream.seek(0)  # offset of 0
        logcontent = self.stream.read()   # read again
        try:
            super(DbStreamHandler, self).close()
        except:
            pass
        if removefile:
            try:
                os.remove(self.baseFilename)
            except:
                pass
        session.query(Download).filter(Download.id == download_id).\
            update({Download.log.key: logcontent,
                    Download.errors.key: self.errors,
                    Download.warnings.key: self.warnings})
        session.commit()


class LevelFilter(object):  # pylint: disable=too-few-public-methods
    """This is a filter extending standard logging filter in that it handles
    logging messages not "up to a certain level" but for a particular set of
    levels only.
    """
    def __init__(self, levels):
        """Initialize a LevelFilter

        :param levels: iterable of integers representing different logging
            levels:
            CRITICAL    50
            ERROR    40
            WARNING    30
            INFO    20
            DEBUG    10
            NOTSET    0
        """
        self._levels = set(levels)

    def filter(self, record):
        """Return True or False to dictate whether the given record must be
        processed or not"""
        return True if record.levelno in self._levels else False


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


def configlog4download(logger, logfile_path='', verbose=False):
    """"Configures the logger, setting it to a `INFO` level with a list of
    default handlers:

    - If `logfile_path` is not the empty str, a :class:`DbStreamHandler`
      (streaming to that file) will capture all INFO, ERROR and WARNING level
      messages, and when its finalize() method is called, flushes the file
      content to the database (deleting the file if needed. This assures that
      if `DbStreamHandler.finalize` is not called, possibly due to an
      exception, the file can be inspected). See :func:`logfilepath` if you
      want to create automatically a log file path in the same directory of a
      given download config file.

    - If `verbose` is True (False by default), a :class:`StreamHandler`
      (streaming to standard output) will capture ONLY messages of level INFO
      (20) and ERROR (40) and CRITICAL (50), ideal for showing relevant
      information to the user on a terminal

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
    # custom StreamHandler: count errors and warnings:
    handlers = []
    if logfile_path:
        handlers.append(DbStreamHandler(logfile_path))
    if verbose:
        handlers.append(SysOutStreamHandler(sys.stdout))
    for hand in handlers:
        logger.addHandler(hand)
    return handlers


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
