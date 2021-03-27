"""
Log utilities for the process routine
"""
import logging
import sys

from stream2segment.io.log import LevelFilter


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
        logger.addHandler(logging.FileHandler(logfile_path, mode='w'))
    if verbose:
        # handlers.append(SysOutStreamHandler(sys.stdout))
        sysout_streamer = logging.StreamHandler(sys.stdout)
        sysout_streamer.setFormatter(logging.Formatter('%(message)s'))
        # configure the levels we want to print (20: info, 40: error, 50: critical)
        l_filter = LevelFilter((20, 40, 50))
        sysout_streamer.addFilter(l_filter)
        # set minimum level (for safety):
        sysout_streamer.setLevel(min(l_filter.levels))
        logger.addHandler(sysout_streamer)

    for hand in handlers:
        logger.addHandler(hand)
    # return handlers