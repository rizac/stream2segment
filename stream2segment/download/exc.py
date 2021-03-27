"""Download exceptions module"""


class QuitDownload(Exception):
    """This is an abstract-like class representing an Exception to be raised
    as soon as something causes no segments to be downloaded.

    This class should not be called directly. Rather, the user should re-raise
    a :class:`NothingToDownload` or :class:`FailedDownload` (see their
    documentation)
    """

    def __init__(self, exc_or_msg):
        """Create a new QuitDownload instance

        :param exc_or_msg: an Exception or a message string. If string, it is
            usually passed via the :function:`formatmsg` function in order to
            provide harmonized message formats
        """
        if isinstance(exc_or_msg, KeyError):  # just re-format key errors
            exc_or_msg = 'KeyError: %s' % str(exc_or_msg)
        super(QuitDownload, self).__init__(str(exc_or_msg))


class NothingToDownload(QuitDownload):
    """Exception that should be raised whenever the download process has no
    segments to download according to the user's settings. Currently,
    stream2segments catches these Exceptions logging their message as level
    INFO and returning a 0 (=successful) status code

    This class and :class:`FailedDownload` both inherit from
    :class:`QuitDownload`.
    """
    pass


class FailedDownload(QuitDownload):
    """Exception that should be raised whenever the download process could not
    proceed. E.g., a download error (e.g., no internet connection) prevents to
    fetch any data. Currently, stream2segments catches these Exceptions logging
    their message as level CRITICAL or ERROR and returning a nonzero
    (=unsuccessful) status code

    This class and :class:`NothingToDownload` both inherit from
    :class:`QuitDownload`
    """
    pass