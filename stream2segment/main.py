#!/usr/bin/python
# event2wav: First draft to download waveforms related to events
#
# (c) 2015 Deutsches GFZ Potsdam
# <XXXXXXX@gfz-potsdam.de>
#
# ----------------------------------------------------------------------
"""
   :Platform:
       Linux, Mac OSX
   :Copyright:
       Deutsches GFZ Potsdam <XXXXXXX@gfz-potsdam.de>
   :License:
       To be decided!
"""
from sqlalchemy.engine import create_engine
from stream2segment.s2sio.db.models import Base
from sqlalchemy.orm.session import sessionmaker
from stream2segment.s2sio.db import models
from stream2segment.processing import process, process_all
import logging
from StringIO import StringIO
from stream2segment.s2sio.db.pd_sql_utils import flush, commit
import sys
import yaml
import click
import datetime as dt
from stream2segment.query import main as query_main
from stream2segment.utils import datetime as dtime, tounicode
from sqlalchemy.sql import func


class LoggerHandler(object):
    """Object handling the root loggers and two Handlers: one writing to StringIO (verbose, being
    saved to db) the other writing to stdout (or stdio) (less verbose, not saved).
    This class has all four major logger methods info, warning, debug and error, plus a save
    method to save the logger text to a database"""
    def __init__(self, out=sys.stdout):
        """
            Initializes a new LoggerHandler, attaching to the root logger two handlers
        """
        root_logger = logging.getLogger()
        root_logger.setLevel(10)
        stringio = StringIO()
        file_handler = logging.StreamHandler(stringio)
        root_logger.addHandler(file_handler)
        console_handler = logging.StreamHandler(out)
        console_handler.setLevel(20)
        root_logger.addHandler(console_handler)
        self.rootlogger = root_logger
        self.errors = 0
        self.warnings = 0
        self.stringio = stringio

    def info(self, *args, **kw):
        """forwards the arguments to L.info, where L is the root Logger"""
        self.rootlogger.info(*args, **kw)

    def debug(self, *args, **kw):
        """forwards the arguments to L.debug, where L is the root Logger"""
        self.rootlogger.debug(*args, **kw)

    def warning(self, *args, **kw):
        """forwards the arguments to L.debug (with "WARNING: " inserted at the beginning of the log
        message), where L is the root logger. This allows this kind of log messages
        to be printed to the db log but NOT on the screen (less verbose)"""
        args = list(args)  # it's a tuple ...
        args[0] = "WARNING: " + args[0]
        self.warnings += 1
        self.rootlogger.debug(*args, **kw)

    def error(self, *args, **kw):
        """forwards the arguments to L.error, where L is the root Logger"""
        self.errors += 1
        self.rootlogger.error(*args, **kw)

    def get_log(self):
        return tounicode(self.stringio.getvalue())


def valid_date(string):
    """does a check on string to see if it's a valid datetime string.
    Returns the string on success, throws an ArgumentTypeError otherwise"""
    return dtime(string, on_err=click.BadParameter)
    # return string


def get_def_timerange():
    """ Returns the default time range when  not specified, for downloading data
    the reutnred tuple has two datetime objects: yesterday, at midniight and
    today, at midnight"""
    dnow = dt.datetime.utcnow()
    endt = dt.datetime(dnow.year, dnow.month, dnow.day)
    startt = endt - dt.timedelta(days=1)
    return startt, endt


def load_def_cfg(filepath='config.yaml', raw=False):
    """Loads default config from yaml file"""
    with open(filepath, 'r') as stream:
        ret = yaml.load(stream) if not raw else stream.read()
    # load config file. This might be better implemented in the near future
    return ret


cfg_dict = load_def_cfg()


def run(action, dbpath, eventws, minmag, minlat, maxlat, minlon, maxlon, ptimespan,
        search_radius_args,
        channels,
        start, end, min_sample_rate,
        processing_args_dict):

    _args_ = dict(locals())  # this must be the first statement, so that we catch all arguments and
    # no local variable (none has been declared yet). Note: dict(locals()) avoids problems with
    # variables created inside loops, when iterating over _args_ (see below)

    if action == 'gui':
        from stream2segment.gui import main as main_gui
        main_gui.run_in_browser(dbpath)
        return 0

    # init the session:
    engine = create_engine(dbpath)
    Base.metadata.create_all(engine)
    # create a configured "Session" class
    Session = sessionmaker(bind=engine)
    # create a Session
    session = Session()

    # add run row with current datetime (utcnow, see models)
    run_row = models.Run()

    # create logger handler
    logger = LoggerHandler()

    # print local vars:
    yaml_content = StringIO()
    yaml_content.write(yaml.dump(_args_, default_flow_style=False))
    config_text = yaml_content.getvalue()
    logger.info("Arguments:")
    tab = "   "
    logger.info(tab + config_text.replace("\n", "\n%s" % tab))
    run_row.config = tounicode(config_text)

    session.add(run_row)
    session.flush()  # udpate run row

    ret = 0
    try:
        segments = []
        if 'd' in action:
#             main(session, run_id, eventws, minmag, minlat, maxlat, minlon, maxlon, search_radius_args,
#          channels, start, end, ptimespan, min_sample_rate, logger=None):
            ret = query_main(session, run_row.id, eventws, minmag, minlat, maxlat, minlon, maxlon,
                             search_radius_args, channels,
                             start, end, ptimespan, min_sample_rate, logger)

        if 'p' in action.lower() and ret == 0:
            segments = session.query(models.Segment).all()

#            parse processing dict:
            pro_sublists_keys = ['bandpass', 'remove_response', 'snr', 'multi_event', 'coda']
            pro_args = {k: processing_args_dict[k] for k in processing_args_dict
                        if k not in pro_sublists_keys}
            for key in pro_sublists_keys:
                subvalues = processing_args_dict.get(key, {})
                for k, v in subvalues.iteritems():
                    pro_args[key + "_" + k] = v

            ret_vals = process_all(session, segments, run_row.id, overwrite_all='P' in action,
                                   logger=logger, **pro_args)

    except Exception as exc:
        logger.error(str(exc))
        ret = 1

    run_row.log = logger.get_log()
    run_row.errors = logger.errors
    run_row.warnings = logger.warnings
    if commit(session):
        return ret

    return 1


@click.command()
@click.option('--action', '-a', type=click.Choice(['d', 'p', 'P', 'dp', 'dP', 'gui']),
              help=('Action to be taken for the program. Choices are:'
                    '\nd   : download data only, no processing'
                    '\np   : no download, process only non-processed data (if any)'
                    '\nP   : no download, clear processing and re-process *all* data'
                    '\ndp  : download data, process only non-processed data (if any)'
                    '\ndP  : download data, clear processing and re-process *all* data'
                    '\ngui : show gui'
                    '\n'
                    '\ne.g.: stream2segment --action dp'
                    '\n      stream2segment --action gui'),
              default=cfg_dict['action'])
@click.option('-e', '--eventws', default=cfg_dict['eventws'],
              help='Event WS to use in queries.')
@click.option('--minmag', default=cfg_dict['minmag'],
              help='Minimum magnitude.', type=float)
@click.option('--minlat', default=cfg_dict['minlat'], type=float,
              help='Minimum latitude.')
@click.option('--maxlat', default=cfg_dict['maxlat'], type=float,
              help='Maximum latitude.')
@click.option('--minlon', default=cfg_dict['minlon'], type=float,
              help='Minimum longitude.')
@click.option('--maxlon', default=cfg_dict['maxlon'], type=float,
              help='Maximum longitude.')
@click.option('--ptimespan', nargs=2, type=float,
              help='Minutes to account for before and after the P arrival time.',
              default=cfg_dict['ptimespan'])
@click.option('--search_radius_args', default=cfg_dict['search_radius_args'], type=float, nargs=4,
              help=('arguments to the function returning the search radius R whereby all '
                    'stations within R will be queried from given event location. '
                    'args are: min_mag max_mag min_distance_deg max_distance_deg'),
              )
@click.option('-o', '--outpath', default=cfg_dict.get('outpath', ''),
              help='Db path where to store waveforms, or db path from where to read the'
                   ' waveforms, if --gui is specified.')
@click.option('-f', '--start', default=cfg_dict.get('start', get_def_timerange()[0]),
              type=valid_date,
              help='Limit to events on or after the specified start time.')
@click.option('-t', '--end', default=cfg_dict.get('end', get_def_timerange()[1]),
              type=valid_date,
              help='Limit to events on or before the specified end time.')
@click.option('--min_sample_rate', default=cfg_dict['min_sample_rate'],
              help='Limit to segments on a sample rate higher than a specific threshold')
def main(action, eventws, minmag, minlat, maxlat, minlon, maxlon, ptimespan, search_radius_args,
         outpath, start, end, min_sample_rate):

    try:
        ret = run(action, outpath, eventws, minmag, minlat, maxlat, minlon, maxlon, ptimespan,
                  search_radius_args, cfg_dict['channels'], start, end, min_sample_rate,
                  cfg_dict['processing'])
        sys.exit(ret)
    except KeyboardInterrupt:
        sys.exit(1)


if __name__ == '__main__':
    main()  # pylint: disable=E1120
