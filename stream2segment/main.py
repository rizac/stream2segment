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
import os
from sqlalchemy.engine import create_engine
from stream2segment import __version__ as s2s_version
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
from stream2segment.download.query import main as query_main
from stream2segment.utils import datetime as dtime, tounicode, load_def_cfg, get_session


# CRITICAL    50
# ERROR    40
# WARNING    30
# INFO    20
# DEBUG    10
# NOTSET    0
def config_logger(db_session, out=sys.stdout, out_levels=(20, 50), logfile_level=10):
    class LevelFilter(object):
        """
        This is a filter which handles standard output: print only critical and info
        messages (it might inherit by logging.Filter but the docs says it's enough to
        implement a filter method)
        """
        def __init__(self, levels):
            self._levels = set(levels)

        def filter(self, record):
            """returns True or False to dictate whether the given record must be processed or not"""
            return True if record.levelno in self._levels else False

    class CounterStreamHandler(logging.StreamHandler):
        """A StreamHandler which counts errors and warnings. See
        http://stackoverflow.com/questions/812477/how-many-times-was-logging-error-called
        NOTE: we might want to send immediately the log to a database, but that requires a lot
        of refactoring and problem is, we should update (concat) the log field: don't know
        how much is efficient. Better keep it in a StringIO and commit at the end for the
        moment (see how an object of this class is instantiated few lines below)
        FOR AN EXAMPLE USING LOG AND SQLALCHEMY, SEE:
        http://docs.pylonsproject.org/projects/pyramid_cookbook/en/latest/logging/sqlalchemy_logger.html
        """
        def __init__(self, session):
            super(CounterStreamHandler, self).__init__(stream=StringIO())
            # access the stream with self.stream
            self.session = session
            self.run_row = models.Run()  # database model instance
            self.run_row.errors = 0
            self.run_row.warnings = 0
            self.run_row.program_version = ".".join(str(x) for x in s2s_version)
            session.add(self.run_row)
            session.commit()

        def emit(self, record):
            if record.levelno == 30:
                self.run_row.warnings += 1
            elif record.levelno == 40:
                self.run_row.errors += 1
            super(CounterStreamHandler, self).emit(record)

        def close(self):
            self.run_row.log = self.stream.getvalue()
            commit(self.session)  # does not throw, so we can call super.close() here:
            super(CounterStreamHandler, self).close()

    class MyFormatter(logging.Formatter):
        """Extends formatter to print custom messages according to levelname (or levelno)
        Warning: don't do too complex stuff as logger gets hard to mantain then
        """
        indent_n = 9
        indent = "%-{:d}s".format(indent_n)

        def format(self, record):
            string = super(MyFormatter, self).format(record)  # defaults to "%(message)s"
            if record.levelno != 20:  # insert levelname, indent newlines
                indent = self.indent
                lname = indent % record.levelname
                if "\n" in string:
                    string = "\n{}".format(indent % "").join(string.split("\n"))
                string = "%s%s" % (lname, string)
            return string

    root_logger = logging.getLogger()
    root_logger.setLevel(min(min(out_levels), logfile_level))  # necessary to forward to handlers

    # =============================
    # configure log file/db handler:
    # =============================
    # custom StreamHandler: count errors and warnings
    db_handler = CounterStreamHandler(db_session)
    db_handler.setLevel(logfile_level)
    db_handler.setFormatter(MyFormatter())
    root_logger.addHandler(db_handler)

    # ==========================================
    # configure out (by default, stdout) handler:
    # ==========================================
    console_handler = logging.StreamHandler(out)
    console_handler.setLevel(min(out_levels))
    # custom filtering: do not print certain levels (default: print info and critical):
    console_handler.addFilter(LevelFilter(out_levels))
    # console_handler.setLevel(20) don't set level, we use filter above
    console_handler.setFormatter(MyFormatter())
    root_logger.addHandler(console_handler)

    return root_logger, db_handler.run_row


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


# a bit hacky maybe, should be checked:
cfg_dict = load_def_cfg()


def run(action, dburi, eventws, minmag, minlat, maxlat, minlon, maxlon, ptimespan, stimespan,
        search_radius_args, channels, start, end, min_sample_rate, processing, isterminal=False):
    """
        Main run method. KEEP the ARGUMENT THE SAME AS THE config.yaml OTHERWISE YOU'LL GET
        A DIFFERENT CONFIG SAVED IN THE DB
        :param processing: a dict as load from the config
    """
    _args_ = dict(locals())  # this must be the first statement, so that we catch all arguments and
    # no local variable (none has been declared yet). Note: dict(locals()) avoids problems with
    # variables created inside loops, when iterating over _args_ (see below)

    if action == 'gui':
        from stream2segment.gui import main as main_gui
        main_gui.run_in_browser(dburi)
        return 0

    # init the session: FIXME: call utils function!
    session = get_session(dburi)

    # create logger handler
    logger, run_row = config_logger(session)

    yaml_dict = load_def_cfg()
    # update with our current variables (only those present in the config_yaml):
    for arg in _args_:
        if arg in yaml_dict:
            yaml_dict[arg] = _args_[arg]

    # print local vars:
    yaml_content = StringIO()
    # use safe_dump to avoid python types. See:
    # http://stackoverflow.com/questions/1950306/pyyaml-dumping-without-tags
    yaml_content.write(yaml.safe_dump(yaml_dict, default_flow_style=False))
    config_text = yaml_content.getvalue()
    if isterminal:
        print("Arguments:")
        tab = "   "
        print(tab + config_text.replace("\n", "\n%s" % tab))
    run_row.config = tounicode(config_text)
    session.commit()  # udpate run row. flush might be also used but we prever sotring to db

    ret = 0
    try:
        segments = []
        if 'd' in action:
            ret = query_main(session, run_row.id, eventws, minmag, minlat, maxlat, minlon, maxlon,
                             search_radius_args, channels,
                             start, end, ptimespan, stimespan, min_sample_rate, isterminal)

        if 'p' in action.lower() and ret == 0:
            segments = session.query(models.Segment).all()

            pro_sublists_keys = ['bandpass', 'remove_response', 'snr', 'multi_event', 'coda']
            pro_args = {k: processing[k] for k in processing
                        if k not in pro_sublists_keys}
            for key in pro_sublists_keys:
                subvalues = processing.get(key, {})
                for k, v in subvalues.iteritems():
                    pro_args[key + "_" + k] = v

            ret_vals = process_all(session, segments, run_row.id, overwrite_all='P' in action,
                                   logger=logger, **pro_args)

    except Exception as exc:
        logger.critical(str(exc))
        raise
    finally:
        logger.info("")
        logger.info("Done: %d error(s), %d warning(s)" % (run_row.errors, run_row.warnings))
        logger.info("")
        for handler in logger.handlers:
            try:
                handler.close()
            except (AttributeError, TypeError, IOError, ValueError):
                pass

    return 0


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
              help='Minutes to account for before and after the P arrival time',
              default=cfg_dict['ptimespan'])
@click.option('--stimespan', nargs=2, type=float,
              help='Hours to account for before and after an event is found to set the start and '
              'end time of the stations search', default=cfg_dict['stimespan'])
@click.option('--search_radius_args', default=cfg_dict['search_radius_args'], type=float, nargs=4,
              help=('arguments to the function returning the search radius R whereby all '
                    'stations within R will be queried from given event location. '
                    'args are: min_mag max_mag min_distance_deg max_distance_deg'),
              )
@click.option('-d', '--dburi', default=cfg_dict.get('dburi', ''),
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
def main(action, eventws, minmag, minlat, maxlat, minlon, maxlon, ptimespan, stimespan,
         search_radius_args, dburi, start, end, min_sample_rate):

    try:
        ret = run(action, dburi, eventws, minmag, minlat, maxlat, minlon, maxlon, ptimespan,
                  stimespan,
                  search_radius_args, cfg_dict['channels'], start, end, min_sample_rate,
                  cfg_dict['processing'], isterminal=True)
        sys.exit(ret)
    except KeyboardInterrupt:
        sys.exit(1)


if __name__ == '__main__':
    main()  # pylint: disable=E1120
