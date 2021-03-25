"""
Log utilities for the download routine
"""

import logging
import os
import sys

from stream2segment.download import db as ddb
from stream2segment.io.log import LevelFilter


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

    # add handlers:
    db_streamer, sysout_streamer = None, None

    if logfile_path:
        db_streamer = DbStreamHandler(logfile_path)
        logger.addHandler(db_streamer)

    if verbose:
        sysout_streamer = logging.StreamHandler(sys.stdout)
        sysout_streamer.setFormatter(logging.Formatter('%(message)s'))
        # configure the levels we want to print (20: info, 40: error, 50: critical)
        l_filter = LevelFilter((20, 40, 50))
        sysout_streamer.addFilter(l_filter)
        # set minimum level (for safety):
        sysout_streamer.setLevel(min(l_filter.levels))
        logger.addHandler(sysout_streamer)

    return db_streamer, sysout_streamer

    # custom StreamHandler: count errors and warnings:
    # handlers = []
    # if logfile_path:
    #     handlers.append(DbStreamHandler(logfile_path))
    # if verbose:
    #     handlers.append(SysOutStreamHandler(sys.stdout))
    # for hand in handlers:
    #     logger.addHandler(hand)
    # return handlers


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
        Download = ddb.Download
        session.query(Download).filter(Download.id == download_id).\
            update({Download.log.key: logcontent,
                    Download.errors.key: self.errors,
                    Download.warnings.key: self.warnings})
        session.commit()