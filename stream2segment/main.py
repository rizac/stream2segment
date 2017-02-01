#!/usr/bin/python
# event2wav: First draft to download waveforms related to events
#
# (c) 2015 Deutsches GFZ Potsdam
# <XXXXXXX@gfz-potsdam.de>
#
# ----------------------------------------------------------------------
import logging
import sys
from StringIO import StringIO
import datetime as dt
import yaml
import time
import os
from sqlalchemy.exc import SQLAlchemyError
import click
from click.exceptions import BadParameter
"""
   :Platform:
       Linux, Mac OSX
   :Copyright:
       Deutsches GFZ Potsdam <XXXXXXX@gfz-potsdam.de>
   :License:
       To be decided!
"""
from stream2segment.io.db import models
from stream2segment.process.processing import main as process_main
from stream2segment.io.db.pd_sql_utils import commit
from stream2segment.download.query import main as query_main
from stream2segment.utils import tounicode, load_def_cfg, get_session, strptime

# set root logger if we are executing this module as script, otherwise as module name following
# logger conventions. Discussion here:
# http://stackoverflow.com/questions/30824981/do-i-need-to-explicitly-check-for-name-main-before-calling-getlogge
# howver, based on how we configured entry points in config, the name is (as november 2016)
# 'stream2segment.main', which messes up all hineritances. So basically setup a main logger
# with the package name
logger = logging.getLogger("stream2segment")

# CRITICAL    50
# ERROR    40
# WARNING    30
# INFO    20
# DEBUG    10
# NOTSET    0


def config_logger_and_return_run_instance(db_session, isterminal, out=sys.stdout,
                                          out_levels=(20, 50),
                                          logfile_level=20):
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

            try:
                with open(os.path.join(os.path.dirname(__file__), "..", "version")) as _:
                    self.run_row.program_version = _.read()
            except IOError:
                pass
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
        """Extends formatter to print different formatted messages according to levelname
        (or levelno). Warning: don't do too complex stuff as logger gets hard to mantain then
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

    root_logger = logger
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
    if isterminal:
        console_handler = logging.StreamHandler(out)
        console_handler.setLevel(min(out_levels))
        # custom filtering: do not print certain levels (default: print info and critical):
        console_handler.addFilter(LevelFilter(out_levels))
        # console_handler.setLevel(20) don't set level, we use filter above
        console_handler.setFormatter(MyFormatter())
        root_logger.addHandler(console_handler)

    return db_handler.run_row


def valid_date(string):
    """does a check on string to see if it's a valid datetime string.
    Returns the string on success, throws an ArgumentTypeError otherwise"""
    try:
        return strptime(string)
    except ValueError as exc:
        raise BadParameter(str(exc))
    # return string


def get_def_timerange():
    """ Returns the default time range when  not specified, for downloading data
    the reutnred tuple has two datetime objects: yesterday, at midniight and
    today, at midnight"""
    dnow = dt.datetime.utcnow()
    endt = dt.datetime(dnow.year, dnow.month, dnow.day)
    startt = endt - dt.timedelta(days=1)
    return startt, endt


def tdstr(timdelta):
    """Returns a formatted timedelta with seconds rounded up or down"""
    # remainder. timedelta has already a nicer formatting with its str method:
    # str(timedelta(hours=15000,seconds=4500))
    # >>> '625 days, 1:15:00'
    # str(timedelta(seconds=4500) - timedelta(microseconds=1))
    # >>> '1:14:59.999999'
    # so we just need to append 'hours' and round microseconds
    add = 1 if timdelta.microseconds >= 500000 else 0
    str_ = str(dt.timedelta(days=timdelta.days, seconds=timdelta.seconds+add, microseconds=0))
    spl = str_.split(":")
    return str_ if len(spl) != 3 else "%sh:%sm:%ss" % (spl[0], spl[1], spl[2])

# a bit hacky maybe, should be checked:
cfg_dict = load_def_cfg()


# IMPORTANT
# IMPORTANT: THE ARGUMENT NAMES HERE MUST BE THE SAME AS THE CONFIG FILE!!!
# IMPORTANT
def run(action, dburl, start, end, eventws, eventws_query_args, stimespan,
        search_radius,
        channels, min_sample_rate, traveltime_phases, wtimespan,
        processing, advanced_settings, class_labels=None, isterminal=False):
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
        main_gui.run_in_browser(dburl)
        return 0

    session = get_session(dburl, scoped=True)  # FIXME: is it necessary for multiprocessing in processing?

    # create logger handler
    run_row = config_logger_and_return_run_instance(session, isterminal)

    yaml_dict = load_def_cfg()
    # update with our current variables (only those present in the config_yaml):
    yaml_dict.update(**{k: v for k, v in _args_.iteritems() if k in yaml_dict})

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
        if 'd' in action.lower():
            starttime = time.time()
            ret = query_main(session, run_row.id, start, end, eventws, eventws_query_args,
                             stimespan, search_radius['minmag'],
                             search_radius['maxmag'], search_radius['minradius'],
                             search_radius['maxradius'], channels,
                             min_sample_rate, 'i' in action, traveltime_phases, wtimespan,
                             'D' in action, advanced_settings, class_labels, isterminal)
            logger.info("Download completed in %s",
                        tdstr(dt.timedelta(seconds=time.time()-starttime)))

        if 'p' in action.lower() and ret == 0:
            starttime = time.time()
            if 'P' in action:
                try:
                    _ = session.query(models.Processing).delete()  # returns num rows deleted
                    session.commit()
                except SQLAlchemyError:
                    session.rollback()
                    raise Exception("Unable to delete all processing (internal db error). Please"
                                    "try to run again the program")
            segments = session.query(models.Segment).\
                filter(~models.Segment.processings.any()).all()  # @UndefinedVariable
            process_main(session, segments, run_row.id, isterminal, **processing)
            logger.info("Processing completed in %s",
                        tdstr(dt.timedelta(seconds=time.time()-starttime)))
        logger.info("")
        logger.info("%d total error(s), %d total warning(s)", run_row.errors, run_row.warnings)

    except Exception as exc:
        logger.critical(str(exc))
        raise
    finally:
        for handler in logger.handlers:
            try:
                handler.close()
            except (AttributeError, TypeError, IOError, ValueError):
                pass

    return 0


@click.command()
@click.option('--action', '-a',  # type=click.Choice(['d', 'p', 'P', 'dp', 'dP', 'gui']),
              help=('action to be taken for the program. Possible values are a combination of '
                    'the following values (without square brackets):'
                    '\n[d]: download data (skip already downloaded segments). '
                    '[D]: download data (already downloaded segments: retry downloading if '
                    'empty/with errors, otherwise skip). '
                    '[i]: (in conjunction with d or D, otherwise ignored) download station '
                    'inventories (skipping already downloaded). '
                    '[p]: Process segments (skip already processed segments). '
                    '[P]: Process segments (all: already processed segments are processed again). '
                    '[gui]: show gui. this option cannot be used in combination with the '
                    'other options (i.e. dgui is invalid). '
                    '\n'
                    '\ne.g.: stream2segment --action dp'
                    '\n      stream2segment --action gui'),
              default=cfg_dict['action'])
@click.option('-s', '--start', default=cfg_dict.get('start', get_def_timerange()[0]),
              type=valid_date,
              help='Limit to events on or after the specified start time.')
@click.option('-e', '--end', default=cfg_dict.get('end', get_def_timerange()[1]),
              type=valid_date,
              help='Limit to events on or before the specified end time.')
@click.option('-E', '--eventws', default=cfg_dict['eventws'],
              help='Event WS to use in queries.')
@click.option('--wtimespan', nargs=2, type=float,
              help='Waveform segment time window: specify two positive integers denoting the '
              'minutes to account for before and after the calculated arrival time',
              default=cfg_dict['wtimespan'])
@click.option('--stimespan', nargs=2, type=float,
              help='Stations time window: specify two positive integers denoting the hours'
              'before and after each event time, to set the start and '
              'end time of the stations search', default=cfg_dict['stimespan'])
@click.option('-d', '--dburl', default=cfg_dict.get('dburl', ''),
              help='Db path where to store waveforms, or db path from where to read the'
                   ' waveforms, if --gui is specified.')
@click.option('--min_sample_rate', default=cfg_dict['min_sample_rate'],
              help='Limit to segments on a sample rate higher than a specific threshold')
def main(action, start, end, eventws, wtimespan, stimespan,
         dburl, min_sample_rate):
    try:
        ret = run(action, dburl, start, end, eventws, cfg_dict['eventws_query_args'],
                  stimespan, cfg_dict['search_radius'], cfg_dict['channels'], min_sample_rate,
                  cfg_dict['traveltime_phases'], wtimespan, cfg_dict['processing'],
                  cfg_dict['advanced_settings'], cfg_dict.get('class_labels', {}),
                  isterminal=True)
        sys.exit(ret)
    except KeyboardInterrupt:
        sys.exit(1)


if __name__ == '__main__':
    main()  # pylint: disable=E1120
