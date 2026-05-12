"""
Command line interface (cli) utilities
"""

import sys
from contextlib import contextmanager
from itertools import chain

from click import progressbar as click_progressbar


class BadParam(Exception):
    """
    Exception describing a bad input parameter that can be caught in
    command line applications as well as working as normal Exception in codebase
    """
    pass


def ascii_decorate(string, frame=None):
    """
    Decorate the string with a frame in Unicode decoration characters,
    and returns the decorated string

    :param string: a single- or multi-line string
    :param frame: list of characters or string. The string/list can have length 1,3 or 7:
        1 character/list defines the decorator character. E.g. '#' or ('#',)
        3 characters/lists define the (top, mid, bottom) characters. E.g. ("=", "|", "-")
        7 characters define the (topleft, topcenter, topright, midleft, midright
          bottomleft, bottomcenter, bottomright) characters. When None or missing,
          this argument defaults to "╔═╗║║╚═╝"
    """
    if not string:
        return ''
    if not frame:
        frame = "╔", "═", "╗", "║", "║", "╚", "═", "╝"
    if len(frame) == 1:
        frame = frame * 8
    elif len(frame) == 3:
        frame = [frame[0]*3, frame[1]*2, frame[2]*3]

    lines = string.splitlines()
    max_len = max(len(l) for l in lines)
    frmt = "%s {:<%d} %s" % (frame[3], max_len, frame[4])
    header_line = frame[0] + frame[1] * (max_len + 2) + frame[2]
    footer_line = frame[-3] + frame[-2] * (max_len + 2) + frame[-1]

    return "\n".join(
        chain(
            [header_line], (frmt.format(l) for l in lines), [footer_line]
        )
    )


class NoOp:
    """
    No-op placeholder object. Used as a drop-in contextmanager with arbitrary
    attributes and methods that do nothing
    """
    # https://stackoverflow.com/a/24946360

    @staticmethod
    def __nop(*args, **kw):
        pass

    def __getattr__(self, _):
        return self.__nop


@contextmanager
def get_progressbar(length, **kw):
    """
    Wrapper around `click.progressbar` (to be used in a `with` statement),
    if `length>0` and the argument 'iterable' is not provided, return a No-op object
    still usable in a with statement. Example
    ```
    with get_progressbar(<N>, ...) as bar:
        # do your stuff ... and then:
        bar.update(1)
    ```
    """
    if not length and 'iterable' not in kw:
        yield NoOp(**kw)
    else:
        # some custom setup if missing:
        # (note that progressbar characters render differently across OSs:
        # after some attempts, I found out the best for Mac - which is the
        # default - and Ubuntu):
        is_linux = sys.platform.startswith('linux')
        kw.setdefault('fill_char', "▮" if is_linux else "●")
        kw.setdefault('empty_char', "▯" if is_linux else "○")
        kw.setdefault('bar_template', '%(label)s %(bar)s %(info)s')
        if length:
            kw['length'] = length
        with click_progressbar(**kw) as pbar:
            yield pbar


@contextmanager
def start_logging(logger, handlers):
    for h in handlers:
        logger.addHandler(h)
    try:
        yield
    finally:
        for handler in handlers:
            try:
                handler.flush()
            except Exception:  # noqa
                pass
            try:
                handler.close()  # maybe already closed? pass in case
            except Exception:  # noqa
                pass
            logger.removeHandler(handler)
