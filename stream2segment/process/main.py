'''
Main module for the segment processing and .csv output

Created on Feb 2, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
from __future__ import print_function

# future direct imports (needs future package installed, otherwise remove):
# (http://python-future.org/imports.html#explicit-imports)
# from builtins import (ascii, chr, dict, filter, hex, input,
#                       int, map, next, oct, open, pow, range, round,
#                       super, zip)

import logging
import csv

# iterating over dictionary keys with the same set-like behaviour on Py2.7 as on Py3:
from future.utils import viewkeys, PY2

from stream2segment.io.db.models import Segment
from stream2segment.process.core import run as process_core_run
from stream2segment.utils import open2writetext

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


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
    seg_id_attname = Segment.id.key
    seg_id_colname = "Segment.db.%s" % seg_id_attname  # try avoiding overridding user defined keys
    csvwriter = [None, None]  # bad hack: in python3, we might use 'nonlocal' @UnusedVariable

    # py2 compatibility of csv library: open in 'wb'. If py3, open in 'w' mode:
    # See utils module
    with open2writetext(outcsvfile, buffering=1, encoding='utf8', errors='replace',
                        newline='') as csvfile:

        def ondone(segment_id, result):  # result is surely not None
            if csvwriter[0] is None:  # instantiate writer according to first input
                isdict = isinstance(result, dict)
                csvwriter[1] = isdict
                # write first column(s):
                if isdict:
                    # we need to pass a list and not an iterable cause the iterable needs
                    # to be consumed twice (the doc states differently, however...):
                    fieldnames = [seg_id_colname]
                    fieldnames.extend(viewkeys(result))
                    csvwriter[0] = csv.DictWriter(csvfile, fieldnames=fieldnames, **kwargs)
                    csvwriter[0].writeheader()
                else:
                    csvwriter[0] = csv.writer(csvfile,  **kwargs)

            csv_writer, isdict = csvwriter
            if isdict:
                result[seg_id_colname] = segment_id
            else:
                # we might have numpy arrays, we should support variable types (numeric, strings,..)
                res = [segment_id]
                res.extend(result)
                result = res

            csv_writer.writerow(result)

            # if flush_num[0] % flush_num[1] == 0:
            #    csvfile.flush()  # this should force writing so if errors we have something
            #    # http://stackoverflow.com/questions/3976711/csvwriter-not-saving-data-to-file-why
            # flush_num[0] += 1

        process_core_run(session, pyfunc, ondone, config_dict, isterminal)
