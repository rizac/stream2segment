'''
Class handling logger for downloading and processing
Created on Feb 20, 2017

@author: riccardo
'''
import logging
from datetime import timedelta
from cStringIO import StringIO
import sys
from contextlib import contextmanager
import time
from stream2segment.io.db.pd_sql_utils import commit
from stream2segment.utils import timedeltaround

# CRITICAL    50
# ERROR    40
# WARNING    30
# INFO    20
# DEBUG    10
# NOTSET    0


class DbStreamHandler(logging.StreamHandler):
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
    def __init__(self, session, run_instance, min_level=20, close_session_on_close=False):
        super(DbStreamHandler, self).__init__(stream=StringIO())
        # access the stream with self.stream
        self.session = session
        self.run_row = run_instance
        self.csoc = close_session_on_close
        # configure level and formatter
        self.setLevel(min_level)
        self.setFormatter(logging.Formatter('[%(levelname)s] - %(message)s'))

    def emit(self, record):
        if record.levelno == 30:
            self.run_row.warnings += 1
        elif record.levelno == 40:
            self.run_row.errors += 1
        super(DbStreamHandler, self).emit(record)

    def close(self):
        self.run_row.log = self.stream.getvalue()
        commit(self.session)  # does not throw, so we can call super.close() here:
        super(DbStreamHandler, self).close()
        self.stream.close()
        if self.csoc:
            self.session.close()


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
    Are printed to screen. By default, these levels are 20 (info) and 50 (critical)"""

    def __init__(self, out=sys.stdout, levels=(20, 50)):
        super(SysOutStreamHandler, self).__init__(out)
        self.setLevel(min(levels))
        # custom filtering: do not print certain levels (default: print info and critical):
        self.addFilter(LevelFilter(levels))


def configlog4processing(logger, outcsvfile, isterminal):
    # config logger (FIXME: merge with download logger?):
    logger.setLevel(logging.INFO)  # this is necessary to configure logger HERE, otherwise the
    # handler below does not work. FIXME: better implementation!!
    logger_handler = logging.FileHandler(outcsvfile + ".log")
    logger_handler.setLevel(logging.DEBUG)
    logger.addHandler(logger_handler)
    if isterminal:
        # configure print to stdout (by default only info and critical messages)
        logger.addHandler(SysOutStreamHandler(sys.stdout))


def configlog4download(logger, db_session, run_instance, isterminal):

    logger.setLevel(logging.INFO)  # necessary to forward to handlers

    # custom StreamHandler: count errors and warnings
    logger.addHandler(DbStreamHandler(db_session, run_instance))
    if isterminal:
        # configure print to stdout (by default only info and critical messages)
        logger.addHandler(SysOutStreamHandler(sys.stdout))


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
    getattr(logger, method)("Completed in %s",
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
