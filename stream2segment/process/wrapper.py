'''
Created on Feb 2, 2017

@author: riccardo
'''
from __future__ import print_function
import logging
from cStringIO import StringIO
import os
import sys
import inspect
from obspy.core.stream import read
from stream2segment.utils import get_session, yaml_load, get_progressbar
from stream2segment.io.db import models
from stream2segment.download.utils import get_inventory_query
from stream2segment.utils.url import url_read
from stream2segment.io.utils import loads_inv, dumps_inv
from contextlib import contextmanager
import warnings
import re
from collections import OrderedDict as odict
from stream2segment.utils.poolexecutor import run_async
import multiprocessing
import types
from stream2segment.download.query import save_inventories
from stream2segment.io.db.pd_sql_utils import withdata

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


def get_inventory(seg_or_sta, session=None, **kwargs):
    """raises tons of exceptions (see main). FIXME: write doc
    :param session: if **not** None but a valid sqlalchemy session object, then
    the inventory, if downloaded because not present, will be saveed to the db (compressed)
    """
    try:
        data = seg_or_sta.inventory_xml
        station = seg_or_sta
    except AttributeError:
        station = seg_or_sta.channel.station
        data = station.inventory_xml

    if not data:
        query_url = get_inventory_query(station)
        data = url_read(query_url, **kwargs)
        if session and data:
            station.inventory_xml = dumps_inv(data)
            session.commit()
        elif not data:
            raise ValueError("No data from server")
    return loads_inv(data)


def load_source(pyfilepath):
    if sys.version_info[0] == 2:
        import imp  # @UnresolvedImport
        return imp.load_source('processing_module_name', pyfilepath)
    else:
        import importlib.util  # @UnresolvedImport
        spec = importlib.util.spec_from_file_location('processing_module_name', pyfilepath)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        return foo


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
        logger.error("Error while importing '%s' from '%s': %s", funcname, pysourcefile, str(exc))
        return

    try:
        config = {} if configsourcefile is None else load_proc_cfg(configsourcefile)
    except Exception as exc:
        logger.error("Error while reading config file '%s': %s", configsourcefile,  str(exc))
        return

    # suppress obspy warnings. Doing process-wise is more feasible FIXME: do it?
    warnings.filterwarnings("default")  # https://docs.python.org/2/library/warnings.html#the-warnings-filter @IgnorePep8
    s = StringIO()
    logger_handler = logging.StreamHandler(s)
    logger_handler.setLevel(logging.WARNING)
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.addHandler(logger_handler)

    query = session.query(models.Segment.channel_id,
                          models.Segment.start_time, models.Segment.end_time, models.Segment.id)
    seg_len = query.count()
    logger.info("Executing '%s' in '%s' for all segments", funcname, pysourcefile)
    logger.info("(iteratively for all segments in '%s)", str(session.bind.engine.url))
    logger.info("Config. file: %s", str(configsourcefile))

    save_station_inventory = config.get('inventory', False)

    inv_ok = session.query(models.Station).filter(withdata(models.Station.inventory_xml)).count()

    done = [0]
    progressbar = get_progressbar(isterminal)
    with redirect(sys.stderr):
        with progressbar(length=seg_len) as pbar:

            def ondone_(light_segment, future):
                pbar.update(1)
                if future.cancelled():
                    return
                exc = future.exception()
                if exc:
                    excinfo = ''
                    # EXPERIMENTAL (commented): get file name and line number. UNCOMMENT TO SHOW IT
                    ## get line number and function name:
                    # traceback = sys.exc_info()[-1]
                    # while traceback.tb_frame:
                    #     finfo = inspect.getframeinfo(traceback.tb_frame)
                    #     if os.path.basename(finfo.filename) == pysourcefilename and \
                    #         finfo.function == funcname:
                    #         excinfo = ' (File "<%s>", line %d, in <%s>)' % \
                    #             (finfo.filename, finfo.lineno, finfo.function)
                    #         break
                    #     traceback = traceback.tb_next

                    # print to log:
                    logger.warning("%s: %s%s", str(light_segment), str(exc), excinfo)
                else:
                    array = future.result()
                    if isinstance(array, dict):
                        ddd = odict()
                        for att, val in light_segment.items(to_str=True):
                            ddd['_segment_' + att] = val
                        ddd.update(array)
                        ondone(ddd)
                    elif hasattr(array, '__iter__') and not isinstance(array, str):
                        ondone(list(light_segment.values(to_str=True)) + list(array))
                    elif array is not None:
                        msg = ("'%s' in '%s' must return None (=skip item), "
                               "an iterable (list, tuple, ...) or a dict") \
                               % (funcname, pysourcefile)
                        logger.error(msg)
                        raise ValueError(msg)  # this should kill (by quitting rapidly)
                        # all remaining processes
                    done[0] += 1

            mngr = multiprocessing.Manager()
            lock = mngr.Lock()
            iterable = (LightSegment(ids_tuple) for ids_tuple in query)
            run_async(iterable, func_wrapper, ondone_, func_posargs=[func, lock, config,
                                                                     str(session.bind.engine.url),
                                                                     save_station_inventory],
                      func_kwargs={}, use_thread=False, max_workers=5, timeout=None)

    captured_warnings = s.getvalue()
    if captured_warnings:
        logger.info("(external warnings captured, please see log for details)")
        logger.info("")
        logger.warning("Captured external warnings:")
        logger.warning("%s", captured_warnings)
        logger.warning("(only the first occurrence of an external warning for each location where "
                       "the warning is issued is reported. Displaying the segment id which "
                       "originated these warnings would require too much effort and performance "
                       "issues compared to the benefits: in most cases, the process "
                       "completed successfully, and if you want to check the correctness of the "
                       "data please check the results)")

    logging.captureWarnings(False)  # form the docs the redirection of warnings to the logging
    # system will stop, and warnings will be redirected to their original destinations
    # (i.e. those in effect before captureWarnings(True) was called).

    s.close()  # maybe not really necessary

    inv_ok2 = session.query(models.Station).filter(withdata(models.Station.inventory_xml)).count()

    if inv_ok2 > inv_ok:
        logger.info("station inventories saved: %d", inv_ok2 - inv_ok)

    logger.info("%d of %d segments successfully processed\n" % (done[0], seg_len))


_inventories = {}
"""private dict to store inventory objects. It is accessed by subprocesses with a lock. Do
not access directly"""


def _inventory(seg, lock, session=None):
    """return the obsoy inventory object from a given segment. Do not call directly, this
    method is called from within suboprocesses with a manager.lock"""
    sta = seg.channel.station
    with lock:
        inv = _inventories.get(sta.id, None)
        if inv is None:
            try:
                inv = get_inventory(sta, session)
            except Exception as exc:
                inv = Exception("Unable to load inventory: %s: %s" %
                                (str(exc.__class__.__name__), str(exc)))
            # logger.warning("loaded " + str(inv))
            _inventories[sta.id] = inv
    if isinstance(inv, Exception):
        raise inv
    else:
        return inv


def func_wrapper(light_segment, func, lock, config, dburl, save_inventories_if_needed):
    """Function executed in each subprocess. Sets-up the sqlalchemy session, bounds some
    custom methods to the segment object and calls the user defined processing
    function

    Notes: sqlalchemy tells to use one engine per subprocess. Here we use a session, which seems
    to work. Maybe check if the performances might be improved by using an engine (at which cost?)
    Check also if attaching bound methods to each segment is the way to go
    """
    seg_id = light_segment.id
    try:
        # http://stackoverflow.com/questions/9619789/sqlalchemy-proper-session-handling-in-multi-thread-applications
        session = get_session(dburl, True)
        segment = session.query(models.Segment).filter(models.Segment.id == seg_id).first()

        segment.stream = types.MethodType(lambda self: read(StringIO(self.data)), segment)
        sess_or_none = session if save_inventories_if_needed else None
        segment.inventory = \
            types.MethodType(lambda self: _inventory(self, lock, session=sess_or_none), segment)

        return func(segment, config)
    finally:
        session.close()
        session.bind.dispose()


class LightSegment(object):
    """a simple light container which wraps a tuple (channel_id, start_time, end_time, dbId)
    identifying a segment. It is used mainly to print the segment refs in various part of the
    program (and whose values are returned in the array/dict of each processing function)"""
    def __init__(self, ids):
        self.channel_id = ids[0]
        self.start_time = ids[1]
        self.end_time = ids[2]
        self.id = ids[3]

    def items(self, to_str=False):
        return [('channel_id', str(self.channel_id) if to_str else self.channel_id),
                ('start_time', self.start_time.isoformat() if to_str else self.start_time),
                ('end_time', self.end_time.isoformat() if to_str else self.end_time),
                ('id', str(self.id) if to_str else self.id)]

    def values(self, to_str=False):
        return [a[1] for a in self.items(to_str)]

    def keys(self):
        return [a[0] for a in self.items()]

    def __str__(self):
        return "segment '{}' [{}, {}] (id: {})".format(*self.values(True))
