'''
Created on Feb 2, 2017

@author: riccardo
'''
from __future__ import print_function
from io import BytesIO
import os
import sys
import logging
from cStringIO import StringIO
from contextlib import contextmanager
import warnings
import re
# from collections import OrderedDict as odict
import traceback
# from sqlalchemy import func, distinct
# from sqlalchemy.orm import load_only
import csv
from obspy.core.stream import read
from stream2segment.utils import yaml_load, get_progressbar, load_source, secure_dburl
from stream2segment.io.db.models import Segment  # , Station
from stream2segment.download.utils import get_inventory
from stream2segment.process.utils import segstr
from stream2segment.io.db.queries import getquery4process


logger = logging.getLogger(__name__)


# THE FUNCTION BELOW REDIRECTS STANDARD ERROR/OUTPUT FROM EXTERNAL PROGRAM
# http://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python
# there's a second one easier to understand but does not restore old std/err stdout
# Added comments from
# http://stackoverflow.com/questions/8804893/redirect-stdout-from-python-for-c-calls
@contextmanager
def redirect(src=sys.stdout, dst=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''

    # some tools (e.g., pytest) change sys.stderr. In that case, we do want this
    # function to yield and return without changing anything:
    try:
        file_desc = src.fileno()
    except AttributeError:
        yield
        return

    # # assert that Python and C stdio write using the same file descriptor
    # assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == file_desc == 1

    def _redirect_stderr(to):
        sys.stderr.close()  # + implicit flush()
        os.dup2(to.fileno(), file_desc)  # file_desc writes to 'to' file
        sys.stderr = os.fdopen(file_desc, 'w')  # Python writes to file_desc

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), file_desc)  # file_desc writes to 'to' file
        sys.stdout = os.fdopen(file_desc, 'w')  # Python writes to file_desc

    _redirect_ = _redirect_stderr if src is sys.stderr else _redirect_stdout

    with os.fdopen(os.dup(file_desc), 'w') as old_:
        with open(dst, 'w') as fopen:
            _redirect_(to=fopen)
        try:
            yield  # allow code to be run with the redirected stdout/err
        finally:
            # restore stdout. buffering and flags such as CLOEXEC may be different:
            _redirect_(to=old_)


def load_proc_cfg(configsourcefile):
    """Returns the dict represetning the processing yaml file"""
    # Simply call the default "yaml to dict" function (yaml_load). Originally,
    # this function also modified the returned a dictionary to return an object where keys where
    # accessible via attributes (attrdict), but this would apply to the main config only (and not
    # to nested dictionaries), thus confusing non expert users
    return yaml_load(configsourcefile)


def run(session, pysourcefile, ondone, configsourcefile=None, show_progress=False):
    reg = re.compile("^(.*):([a-zA-Z_][a-zA-Z0-9_]*)$")
    m = reg.match(pysourcefile)
    if m and m.groups():
        pysourcefile = m.groups()[0]
        funcname = m.groups()[1]
    else:
        funcname = 'main'

    # not implemented, but the following var is used below for exceptions info
    # pysourcefilename = os.path.basename(pysourcefile)

    try:
        pyfunc = load_source(pysourcefile).__dict__[funcname]
    except Exception as exc:
        msg = "Error while importing '%s' from '%s': %s" % (funcname, pysourcefile, str(exc))
        logger.critical(msg)
        return 0

    try:
        config = {} if configsourcefile is None else load_proc_cfg(configsourcefile)
    except Exception as exc:
        msg = "Error while reading config file '%s': %s" % (configsourcefile,  str(exc))
        logger.critical(msg)
        return 0

    # suppress obspy warnings. Doing process-wise is more feasible FIXME: do it?
    warnings.filterwarnings("default")  # https://docs.python.org/2/library/warnings.html#the-warnings-filter @IgnorePep8
    s = StringIO()
    logger_handler = logging.StreamHandler(s)
    logger_handler.setLevel(logging.WARNING)
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.addHandler(logger_handler)
    logger.info("Executing '%s' in '%s'", funcname, pysourcefile)
    logger.info(" for all segments in '%s", secure_dburl(str(session.bind.engine.url)))
    logger.info("Config. file: %s", str(configsourcefile))

    save_station_inventory = config.get('save_downloaded_inventory', False)
    inventory_required = config.get('inventory', False)

    load_stream = True

    # multiprocess with sessions is a mess. So we have two choices: either we build a dict from
    # each segment object, or we simply do not use multiprocess. We will opt for the second choice
    # (maybe implement tests in the future to see which is faster)

    # do an iteration on the main process to check when AsyncResults is ready
    done = 0

    # attributes to load only for stations and segments (sppeeds up db queries):
    SEG_ID = Segment.id.key
    seg_atts = [SEG_ID]  # ["channel_id", "start_time", "end_time", "id"]
    segs_staids = getquery4process(session, config.get('segment_select', {}), *seg_atts)

    # get total segment length:
    seg_len = segs_staids.count()
    # actually, this is better as it should be optimized, but how to translate for the query we
    # have? comment for the moment:
    # seg_len = session.query(func.count(Segment.id)).filter(seg_filter).scalar()
    logger.info("%d segments found to process", seg_len)

    current_inv = None
    current_sta_id = None

    stations_discarded = 0
    segments_discarded = 0
    stations_saved = 0

    with redirect(sys.stderr):
        with get_progressbar(show_progress, length=seg_len) as pbar:
            try:
                for seg, sta_id in segs_staids:
                    pbar.update(1)
                    if inventory_required:
                        if sta_id != current_sta_id:
                            current_sta_id = sta_id
                            # try loading inventory
                            try:
                                current_inv = get_inventory(seg.station, save_station_inventory)
                            except Exception as exc:
                                stations_discarded += 1
                                current_inv = None
                                logger.error(exc)
                        if current_inv is None:
                            segments_discarded += 1
                            continue
                    try:
                        if load_stream:
                            try:
                                mseed = read(BytesIO(seg.data))
                            except Exception as exc:
                                raise ValueError("Error while reading mseed: " + str(exc))
                        array = pyfunc(seg, mseed, current_inv, config)
                        if array is not None:
                            ondone(seg, array)
                        done += 1
                    except (ImportError, NameError, AttributeError, SyntaxError, TypeError) as _:
                        raise  # sys.exc_info()
                    except Exception as generr:
                        logger.warning("%s: %s", segstr(seg), str(generr))
            except:
                err_msg = traceback.format_exc()
                logger.critical(err_msg)
                return 0

    captured_warnings = s.getvalue()
    if captured_warnings:
        logger.info("(external warnings captured, please see log for details)")
        logger.info("")
        logger.warning("Captured external warnings:")
        logger.warning("%s", captured_warnings)
        logger.warning("(only the first occurrence of an external warning for each location where "
                       "the warning is issued is reported. Because of maintainability and "
                       "performance potential issues, the segment id which originated "
                       "these warnings cannot be shown. However, in most cases the process "
                       "completed successfully, and if you want to check the correctness of the "
                       "data please check the results)")

    logging.captureWarnings(False)  # form the docs the redirection of warnings to the logging
    # system will stop, and warnings will be redirected to their original destinations
    # (i.e. those in effect before captureWarnings(True) was called).

    s.close()  # maybe not really necessary

    if stations_discarded:
        logger.info("%d stations (and %d relative segments) unprocessed: "
                    "error in reading inventory)", stations_discarded, segments_discarded)
    if stations_saved:
        logger.info("station inventories saved: %d", stations_saved)

    logger.info("%d of %d segments successfully processed\n" % (done, seg_len))


def to_csv(outcsvfile, session, pysourcefile, configsourcefile, isterminal):
    kwargs = dict(delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    flush_num = [1, 10]  # determines when to flush (not used. We use the
    # last argument to open which tells to flush line-wise. To add custom flush, see commented
    # lines at the end of the with statement and uncomment them
    # ------------------------
    # cols always written (1 for the moment, the id): Segment ORM table attribute name(s):
    col_headers = [Segment.id.key]
    CHEAD_FRMT = "_segment_db_%s_"  # try avoiding overridding user defined keys
    csvwriter = [None, None]  # bad hack: in python3, we might use 'nonlocal' @UnusedVariable

    with open(outcsvfile, 'wb', 1) as csvfile:

        def ondone(segment, result):  # result is surely not None
            if csvwriter[0] is None:  # instanitate writer according to first input
                isdict = isinstance(result, dict)
                csvwriter[1] = isdict
                # write first column(s):
                if isdict:
                    # we need to pass a list and not an iterable cause the iterable needs
                    # to be consumed twice (the doc states differently, however...):
                    fieldnames = [(CHEAD_FRMT % c) for c in col_headers]
                    fieldnames.extend(result.iterkeys())
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

        run(session, pysourcefile, ondone, configsourcefile, isterminal)
