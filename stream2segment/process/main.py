'''
Main module for the segment processing and .csv output

Created on Feb 2, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
from __future__ import print_function

# future direct imports (needs future package installed, otherwise remove):
# (http://python-future.org/imports.html#explicit-imports)
from builtins import (ascii, chr, dict, filter, hex, input,
                      int, map, next, oct, open, pow, range, round,
                      super, zip)

# iterating over dictionary keys with the same set-like behaviour on Py2.7 as on Py3:
from future.utils import viewkeys

import logging
import csv


from stream2segment.io.db.models import Segment
from stream2segment.process.core import run as process_core_run

logger = logging.getLogger(__name__)


def run(session, pyfunc, config_dict, outcsvfile=None, isterminal=False):
    if outcsvfile is None:
        process_core_run(session, pyfunc, None, config_dict, isterminal)
        return

    kwargs = dict(delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    flush_num = [1, 10]  # determines when to flush (not used. We use the
    # last argument to open which tells to flush line-wise. To add custom flush, see commented
    # lines at the end of the with statement and uncomment them
    # ------------------------
    # cols always written (1 for the moment, the id): Segment ORM table attribute name(s):
    col_headers = [Segment.id.key]
    CHEAD_FRMT = "Segment.%s"  # try avoiding overridding user defined keys
    csvwriter = [None, None]  # bad hack: in python3, we might use 'nonlocal' @UnusedVariable

    with open(outcsvfile, 'w', 1) as csvfile:

        def ondone(segment, result):  # result is surely not None
            if csvwriter[0] is None:  # instantiate writer according to first input
                isdict = isinstance(result, dict)
                csvwriter[1] = isdict
                # write first column(s):
                if isdict:
                    # we need to pass a list and not an iterable cause the iterable needs
                    # to be consumed twice (the doc states differently, however...):
                    fieldnames = [(CHEAD_FRMT % c) for c in col_headers]
                    fieldnames.extend(viewkeys(result))
                    csvwriter[0] = csv.DictWriter(csvfile, fieldnames=fieldnames, **kwargs)
                    csvwriter[0].writeheader()
                else:
                    csvwriter[0] = csv.writer(csvfile,  **kwargs)

            csv_writer, isdict = csvwriter
            if isdict:
                result.update({(CHEAD_FRMT % c): getattr(segment, c) for c in col_headers})
            else:
                # we might have numpy arrays, we should support variable types (numeric, strings,..)
                res = [getattr(segment, c) for c in col_headers]
                res.extend(result)
                result = res

            csv_writer.writerow(result)

            # if flush_num[0] % flush_num[1] == 0:
            #    csvfile.flush()  # this should force writing so if errors we have something
            #    # http://stackoverflow.com/questions/3976711/csvwriter-not-saving-data-to-file-why
            # flush_num[0] += 1

        process_core_run(session, pyfunc, ondone, config_dict, isterminal)
