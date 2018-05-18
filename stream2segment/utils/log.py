'''
Class handling logger for downloading and processing

:date: Feb 20, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import object

import os
import tempfile
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
    http://stackoverflow.com/questions/812477/how-many-times-was-logging-error-called
    This class takes in the constructor an id of the table 'downloads' (referring to
    the current download), and when closed writes the content of the file to the database,
    deleting the handler's file. You should always explicitly call close() to assure the log
    is written to the database**.
    For an example usijng sql-alchemy log rows (slightly different case but informative) see:
    http://docs.pylonsproject.org/projects/pyramid_cookbook/en/latest/logging/sqlalchemy_logger.html
    """
    def __init__(self, min_level=20):
        """
        :param download_id: the id of the database instance reflecting a row of the Download table.
        THE INSTANCE MUST BE ADDED TO THE DATABASE ALREADY. It will be
        notified with each error and warning issued by this log
        """
        # w+: allows to read without closing first:
        super(DbStreamHandler, self).__init__(tmpfilepath(prefix='s2s_d'), mode='w+')
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
        super(DbStreamHandler, self).emit(record)  # logging.FileHandler flushes every emit

    def finalize(self, session, download_id, removefile=True):
        '''writes to db, closes and optionally removes the underlying file'''
        # the super-class sets the stream to None when closing, so we might check this to
        # see if we closed it already:
        if self.stream is None:
            return
        # we experienced the NoneType error which we could not test deterministically
        # so the if above serves to this, especially because we
        # know self.stream == None => already closed

        # Completed succesfully, 45 warnings (execution ime: ...)
        # Not completed succesfully, 3 error, 45 warning (total execution time: )

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
    """
    This is a filter extending standard logging filter in that it handles logging messages
    not "up to a certain level" but for a particular set of levels only.
    """
    def __init__(self, levels):
        '''Init a LevelFilter
        :param levels: iterable of integers representing different logging levels:
            CRITICAL    50
            ERROR    40
            WARNING    30
            INFO    20
            DEBUG    10
            NOTSET    0
        '''
        self._levels = set(levels)

    def filter(self, record):
        """returns True or False to dictate whether the given record must be processed or not"""
        return True if record.levelno in self._levels else False


class SysOutStreamHandler(logging.StreamHandler):
    """Handler that prints to screen the loggin messages.
    It implements a LevelFilter so that only special levels (not only levels "up to")
    Are printed to screen. By default, these levels are 20 (info) 40 (error) and 50 (critical)"""

    def __init__(self, out=sys.stdout, levels=(20, 40, 50)):
        super(SysOutStreamHandler, self).__init__(out)
        self.setLevel(min(levels))
        # custom filtering: do not print certain levels (default: print info and critical):
        self.addFilter(LevelFilter(levels))
        # this should be the default, but for safety set it again:
        self.setFormatter(logging.Formatter('%(message)s'))


def configlog4processing(logger, outcsvfile, isterminal):
    """Configures a set of default handlers, add them to `logger` amd returns them as list:

       - A logging.FileHandler redirecting to a file named `outcsvfile`+".log"
         if outcsvfile is given (i.e. not falsy) OR temporary file with the current timestamp in it

       - If `isterminal` = True, a StreamHandler which redirects to standard output ONLY messages
         of level INFO (20) and ERROR (40) and CRITICAL (50): i.e., it does not print DEBUG
         WARNING messages

    Implementation detail: this method modifies permanently these values for performance reason:
    ```
    logging._srcfile = None
    logging.logThreads = 0
    logging.logProcesses = 0
    ```
    If you run the download inside a process re-using logging, store those values and re-set them
    as needed
    """
    # https://docs.python.org/2/howto/logging.html#optimization:
    logging._srcfile = None  # pylint: disable=protected-access
    logging.logThreads = 0
    logging.logProcesses = 0

    logger.setLevel(logging.INFO)  # this is necessary to configure logger HERE, otherwise the
    handlers = []
    if outcsvfile:
        handlers.append(logging.FileHandler(outcsvfile + ".log", mode='w'))
    else:
        handlers.append(logging.FileHandler(tmpfilepath(prefix='s2s_p'), mode='w'))
    if isterminal:
        # configure print to stdout (by default only info errors and critical messages)
        handlers.append(SysOutStreamHandler(sys.stdout))
    for hand in handlers:
        logger.addHandler(hand)
    return handlers


def tmpfilepath(prefix='s2s', extension='log'):
    """returns a file path with dirname `tempfile.gettempdir()` and basename:
        <prefix>_<now>.<extension>
    where <now> is the iso format represenation of the current date-time (in UTC).
    The returned path can be opened for writing temporary stuff (e.g. log filehandler)
    """
    return os.path.join(tempfile.gettempdir(),
                        "%s_%s.%s" % (prefix,
                                      datetime.utcnow().replace(microsecond=0).isoformat(),
                                      extension))


def configlog4download(logger, isterminal=False):
    """Configures a set of default handlers, add them to `logger` amd teturns them as list:

    - A DbStreamHandler which will capture all INFO, ERROR and WARNING level messages, and when
      its finalize() method is called, flushes the content of its file to the database (deleting
      the file if needed. This assures that if `finalize` is not called, possibly due to an
      exception, the file can be inspected)

    - If `isterminal` = True, a StreamHandler which prints to standard output ONLY messages of
      level INFO (20) and ERROR (40) and CRITICAL (50): i.e., it does not rpint DEBUG and WARNING
      messages

    Implementation detail: this method modifies permanently these values for performance reason:
    ```
    logging._srcfile = None
    logging.logThreads = 0
    logging.logProcesses = 0
    ```
    If you run the download inside a process re-using logging, store those values and re-set them
    as needed
    """
    # https://docs.python.org/2/howto/logging.html#optimization:
    logging._srcfile = None  # pylint: disable=protected-access
    logging.logThreads = 0
    logging.logProcesses = 0

    logger.setLevel(logging.INFO)  # necessary to forward to handlers
    # custom StreamHandler: count errors and warnings:
    handlers = [DbStreamHandler()]
    if isterminal:
        # configure print to stdout (by default only info errors and critical messages)
        handlers.append(SysOutStreamHandler(sys.stdout))
    for hand in handlers:
        logger.addHandler(hand)
    return handlers
