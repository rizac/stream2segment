'''
Created on Feb 2, 2017

@author: riccardo
'''
from __future__ import print_function
import os
import sys
import logging
from cStringIO import StringIO
from contextlib import contextmanager
import warnings
import re
from collections import OrderedDict as odict
import traceback
from sqlalchemy.orm import load_only
from obspy.core.stream import read
from stream2segment.utils import yaml_load, get_progressbar, load_source, secure_dburl
from stream2segment.io.db import models
from stream2segment.download.utils import get_inventory
from stream2segment.io.db.pd_sql_utils import withdata
from stream2segment.process.utils import segstr

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


def getid(segment):
    return str(segment.channel.id), segment.start_time.isoformat(), segment.end_time.isoformat(),\
            str(segment.id)


def load_proc_cfg(configsourcefile):
    """Returns the dict represetning the processing yaml file"""
    # Simply call the default "yaml to dict" function (yaml_load). Originally,
    # this function also modified the returned a dictionary to return an object where keys where
    # accessible via attributes (attrdict), but this would apply to the main config only (and not
    # to nested dictionaries), thus confusing non expert users
    return yaml_load(configsourcefile)


def run(session, pysourcefile, ondone, configsourcefile=None, isterminal=False):
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
        func = load_source(pysourcefile).__dict__[funcname]
    except Exception as exc:
        msg = "Error while importing '%s' from '%s': %s" % (funcname, pysourcefile, str(exc))
        logger.error(msg)
        raise Exception(msg)

    try:
        config = {} if configsourcefile is None else load_proc_cfg(configsourcefile)
    except Exception as exc:
        msg = "Error while reading config file '%s': %s" % (configsourcefile,  str(exc))
        logger.error(msg)
        raise Exception(msg)

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

    # store how many inventories we have, just to warn later how many we saved (if any):
    inv_ok = session.query(models.Station).filter(withdata(models.Station.inventory_xml)).count()

    seg_filter = withdata(models.Segment.data)

    # LEVE NOTE HERE EVEN IF WE DO NOT USE THREADING NOR MULTIPROCESSING FOR NOW:
    # another thing experienced with ThreadPoolExecutor:
    # we experienced some problems if max_workers is None. The doc states that it is the number
    # of processors on the machine, multiplied by 5, assuming that ThreadPoolExecutor is often
    # used to overlap I/O instead of CPU work and the number of workers should be higher than the
    # number of workers for ProcessPoolExecutor. But the source code seems not to set this value
    # at all!! (at least in python2, probably in pyhton3 is fine). So let's do it manually
    # (remember, for multiprocessing don't multiply to 5):
    # max_workers = 5 * multiprocessing.cpu_count()

    # attributes to load only for stations and segments (sppeeds up db queries):
    sta_atts = ['id']
    seg_atts = ["channel_id", "start_time", "end_time", "id"]

    def _ztr(itm):
        """function returning the id attrs defined in seg_atts mapped to their str represenation"""
        try:
            return itm.isoformat()
        except AttributeError:
            return str(itm)

    # do an iteration on the main process to check when AsyncResults is ready
    done = 0
    progressbar = get_progressbar(isterminal)
    sta_query = session.query(models.Station).\
        options(load_only(*(sta_atts + ['inventory_xml'] if inventory_required else sta_atts))).\
        filter(models.Station.segments.any(seg_filter))  # @UndefinedVariable

    seg_len = session.query(models.Segment).filter(seg_filter).count()

    with redirect(sys.stderr):
        with progressbar(length=seg_len) as pbar:
            try:
                for sta in sta_query:
                    segs = sta.segments.filter(seg_filter)
                    segs_count = segs.count()

                    inv = None
                    if inventory_required:
                        try:
                            inv = get_inventory(sta, save_station_inventory)
                        except Exception as exc:
                            logger.error(exc)
                            pbar.update(segs_count)
                            logger.warning("(%d segments discarded)", segs_count)
                            continue

                    for seg in segs.options(load_only(*(seg_atts + ['data'] if load_stream
                                                        else seg_atts))):
                        pbar.update(1)
                        try:
                            if load_stream:
                                try:
                                    mseed = read(StringIO(seg.data))
                                except Exception as exc:
                                    raise ValueError("Error while reading mseed: " + str(exc))
                            array = func(seg, mseed, inv, config)
                            if isinstance(array, dict):
                                ddd = odict([('_segment_' + at, _ztr(getattr(seg, at)))
                                             for at in seg_atts])
                                ddd.update(array)
                                ondone(ddd)
                            elif hasattr(array, '__iter__') and not isinstance(array, str):
                                ondone(list(_ztr(getattr(seg, at)) for at in seg_atts) + list(array))
                            elif array is not None:
                                msg = ("'%s' in '%s' cannot return '%s' objects") \
                                       % (funcname, pysourcefile, str(type(array)))
                                logger.error(msg)
                                raise SyntaxError(msg)
                            done += 1
                        except (ImportError, NameError, SyntaxError, TypeError) as stopexc:
                            raise stopexc
                        except Exception as generr:
                            logger.warning("%s: %s", segstr(seg), str(generr))
            except Exception as stopexc:
                err_msg = traceback.format_exc()
                logger.error(err_msg)
                raise sys.exc_info()

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

    inv_ok2 = session.query(models.Station).filter(withdata(models.Station.inventory_xml)).count()

    if inv_ok2 > inv_ok:
        logger.info("station inventories saved: %d", inv_ok2 - inv_ok)

    logger.info("%d of %d segments successfully processed\n" % (done, seg_len))
