class SkipSegment(Exception):
    """Stream2segment exception indicating a segment processing error that should
    resume to the next segment without interrupting the whole routine
    """
    pass  # (we can also pass an exception in the __init__, superclass converts it)
