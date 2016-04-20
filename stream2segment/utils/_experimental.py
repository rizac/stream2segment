
# old function unused. Will be removed soon
def _oserr(errnotype, msg=''):  # FIXME: check msg
    """
        Returns an OSError raised by the file argument.
        :param errnotype: the error type, see errno package for details (e.g., errno.ENOENT)
        :param file: the file
    """
    return OSError(strerror(errnotype) + " " + str(msg))