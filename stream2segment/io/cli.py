"""
Command line interface IO utilities

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""

import sys
from contextlib import contextmanager
from itertools import chain

from click import progressbar as click_progressbar


def ascii_decorate(string):
    """Decorate the string with a frame in unicode decoration characters,
    and returns the decorated string

    :param string: a signle- or multi-line string
    """
    if not string:
        return ''

    # defined the frame characters:
    # (topleft, top, topright, left, right, bottomleft, bottom, bottomright):
    # note that top and bottom must be 1-length strings, and
    # topleft+left=bottomleft must have the same length, as well as
    # topright+right+bottomright

    frame = "╔", "═", "╗", "║", "║", "╚", "═", "╝"
    # frame = "###", "#", "###", "###", "###", "###", "#", "###"

    linez = string.splitlines()
    maxlen = max(len(l) for l in linez)
    frmt = "%s {:<%d} %s" % (frame[3], maxlen, frame[4])
    hline_top = frame[0] + frame[1] * (maxlen + 2) + frame[2]
    hline_bottom = frame[-3] + frame[-2] * (maxlen + 2) + frame[-1]

    return "\n".join(chain([hline_top],
                           (frmt.format(l) for l in linez),
                           [hline_bottom]))


class Nop(object):
    """Dummy class (no-op), used to yield a contextmanager where each method
    is no-op. Used in `get_progressbar`
    """
    # https://stackoverflow.com/a/24946360
    def __init__(self, *a, **kw):
        pass

    @staticmethod
    def __nop(*args, **kw):
        pass

    def __getattr__(self, _):
        return self.__nop


@contextmanager
def get_progressbar(show, **kw):
    """Return a `click.progressbar` if `show` is True, otherwise a No-op
    class, so that we can run programs by simply doing:
    ```
    isterminal = True  # or False for no-op class
    with get_progressbar(isterminal, length=..., ...) as bar:
        # do your stuff ... and then:
        bar.update(num_increments)  # this is no-op if `isterminal` is False
    ```
    """
    if not show or kw.get('length', 1) == 0:
        yield Nop(**kw)
    else:
        # some custom setup if missing:
        # (note that progressbar characters render differently across OSs:
        # after some attempts, I found out the best for mac - which is the
        # default - and Ubuntu):
        is_linux = sys.platform.startswith('linux')
        kw.setdefault('fill_char', "▮" if is_linux else "●")
        kw.setdefault('empty_char', "▯" if is_linux else "○")
        kw.setdefault('bar_template', '%(label)s %(bar)s %(info)s')
        with click_progressbar(**kw) as pbar:
            yield pbar