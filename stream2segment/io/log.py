"""
log utilities
"""
# :date: Feb 20, 2017

class LevelFilter:  # noqa
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


def close_logger(logger):
    """Close all logger handlers and removes them from logger"""
    for handler in logger.handlers[:]:
        try:
            handler.flush()
        except Exception:  # noqa
            pass
        try:
            handler.close()  # maybe already closed? pass in case
        except Exception:  # noqa
            pass
        logger.removeHandler(handler)
