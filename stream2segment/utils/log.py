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
from contextlib import contextmanager
import time

from stream2segment.utils import timedeltaround
from stream2segment.io.db.models import Download

# CRITICAL    50
# ERROR    40
# WARNING    30
# INFO    20
# DEBUG    10
# NOTSET    0


class DbStreamHandler(logging.FileHandler):
    """A StreamHandler which counts errors and warnings. See
    http://stackoverflow.com/questions/812477/how-many-times-was-logging-error-called
    NOTE: we might want to commit immediately each log message to a database, but
    we should update (concat) the log field: don't know
    how much is efficient. Better keep it in a StringIO and commit at the end for the
    moment (see how an object of this class is instantiated few lines below)
    PLEASE CALL close() OTHERWISE
    FOR AN EXAMPLE USING LOG AND SQLALCHEMY, SEE:
    http://docs.pylonsproject.org/projects/pyramid_cookbook/en/latest/logging/sqlalchemy_logger.html
    """
    def __init__(self, session, download_id, min_level=20):
        """
        :param download_id: the id of the database instance reflecting a row of the Download table.
        THE INSTANCE MUST BE ADDED TO THE DATABASE ALREADY. It will be
        notified with each error and warning issued by this log
        """
        # w+: allows to read without closing first:
        super(DbStreamHandler, self).__init__(gettmpfile(), mode='w+')
        # access the stream with self.stream
        self.session = session
        self.download_id = download_id
        self.errors = 0
        self.warnings = 0
        # configure level and formatter
        self.setLevel(min_level)
        self.setFormatter(logging.Formatter('[%(levelname)s]  %(message)s'))

    def emit(self, record):
        if record.levelno == 30:
            self.warnings += 1
        elif record.levelno == 40:
            self.errors += 1
        super(DbStreamHandler, self).emit(record)  # logging.FileHandler flushes every emit

    def close(self):
        super(DbStreamHandler, self).flush()  # for safety
        self.stream.seek(0)  # offset of 0
        logcontent = self.stream.read()   # read again
        try:
            super(DbStreamHandler, self).close()
        except:
            pass
        try:
            os.remove(self.baseFilename)
        except:
            pass
        self.session.query(Download).filter(Download.id == self.download_id).\
            update({Download.log.key: logcontent,
                    Download.errors.key: self.errors,
                    Download.warnings.key: self.warnings})
        self.session.commit()


class LevelFilter(object):
    """
    This is a filter extending standard logging filter in that it handles logging messages
    not "up to a certain level" but for a particular set of levels only.
    This
    """
    def __init__(self, levels):
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
    # config logger (FIXME: merge with download logger?):
    logger.setLevel(logging.INFO)  # this is necessary to configure logger HERE, otherwise the
    # handler below does not work. FIXME: better implementation!!
    logger_handler = logging.FileHandler(outcsvfile + ".log", mode='w')
    logger_handler.setLevel(logging.DEBUG)
    logger.addHandler(logger_handler)
    if isterminal:
        # configure print to stdout (by default only info warning errors and critical messages)
        logger.addHandler(SysOutStreamHandler(sys.stdout))


def gettmpfile():
    """returns a file under `tempfile.gettempdir()` with a datetimestamp prefixed with s2s
    """
    return os.path.join(tempfile.gettempdir(), "s2s_%s.log" % datetime.utcnow().isoformat())


def configlog4download(logger, db_session, download_id, isterminal):
    """configs for download and returns the handler used to store the log to the db
    and to a tmp file. The file is accessible via logger..baseFilename
    """
    logger.setLevel(logging.INFO)  # necessary to forward to handlers
    # custom StreamHandler: count errors and warnings:
    dbstream_handler = DbStreamHandler(db_session, download_id)
    logger.addHandler(dbstream_handler)
    if isterminal:
        # configure print to stdout (by default only info and critical messages)
        logger.addHandler(SysOutStreamHandler(sys.stdout))
    return dbstream_handler


# def configlog4stdout(logger):
#     logger.setLevel(logging.INFO)  # necessary to forward to handlers
#     # configure print to stdout (by default only info and critical messages):
#     logger.addHandler(SysOutStreamHandler(sys.stdout))


@contextmanager
def elapsedtime2logger_when_finished(logger, method='info'):
    """contextmanager to be used in a with statement, will print to the logger how much time it
    required to
    execute the code in the with statement. At the end (if no exception is raised)
    `logger.info` (or whatever specified in `method`) will be called with a
    "Completed in ..." message with a nicely formatted timedelta
    :param logger: a logger object
    :param method: the string of the method to invoke. Defaults to 'info'. Possible other values
    are 'warning', 'error', 'debug' and 'critical'
    """
    starttime = time.time()
    yield
    getattr(logger, method)("(Completed in %s)",
                            str(timedeltaround(timedelta(seconds=time.time()-starttime))))

# class MyFormatter(logging.Formatter):
#         """Extends formatter to print different formatted messages according to levelname
#         (or levelno). Warning: don't do too complex stuff as logger gets hard to mantain then
#         """
#         indent_n = 9
#         indent = "%-{:d}s".format(indent_n)
#
#         def format(self, record):
#             string = super(MyFormatter, self).format(record)  # defaults to "%(message)s"
#             if record.levelno != 20:  # insert levelname, indent newlines
#                 indent = self.indent
#                 lname = indent % record.levelname
#                 if "\n" in string:
#                     string = "\n{}".format(indent % "").join(string.split("\n"))
#                 string = "%s%s" % (lname, string)
#             return string
