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
import click
from click.exceptions import BadParameter
from contextlib import contextmanager
import csv
"""
   :Platform:
       Linux, Mac OSX
   :Copyright:
       Deutsches GFZ Potsdam <XXXXXXX@gfz-potsdam.de>
   :License:
       To be decided!
"""
from stream2segment.io.db import models
from stream2segment.io.db.pd_sql_utils import commit
from stream2segment.process.wrapper import run as process_run
from stream2segment.download.query import main as query_main
from stream2segment.utils import tounicode, yaml_load, get_session, strptime, yaml_load_doc

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
cfg_dict = yaml_load()
cfg_doc = yaml_load_doc()


def showgui(dburl):
    from stream2segment.gui import main as main_gui
    main_gui.run_in_browser(dburl)
    return 0


# IMPORTANT !!!
# IMPORTANT: THE ARGUMENT NAMES HERE MUST BE THE SAME AS THE CONFIG FILE!!! SEE FUNCTION DOC BELOW
# IMPORTANT !!!
def download(dburl, start, end, eventws, eventws_query_args, stimespan,
             search_radius,
             channels, min_sample_rate, inventory, traveltime_phases, wtimespan,
             processing, retry, advanced_settings, class_labels=None, isterminal=False):
    """
        Main run method. KEEP the ARGUMENT THE SAME AS THE config.yaml OTHERWISE YOU'LL GET
        A DIFFERENT CONFIG SAVED IN THE DB
        :param processing: a dict as load from the config
    """
    _args_ = dict(locals())  # this must be the first statement, so that we catch all arguments and
    # no local variable (none has been declared yet). Note: dict(locals()) avoids problems with
    # variables created inside loops, when iterating over _args_ (see below)

    with closing(dburl) as session:
        # create logger handler
        run_row = config_logger_and_return_run_instance(session, isterminal)
        yaml_dict = yaml_load()
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
        session.commit()  # udpate run row. flush might be also used but we prefer storing to db

        starttime = time.time()
        ret = query_main(session, run_row.id, start, end, eventws, eventws_query_args,
                         stimespan, search_radius['minmag'],
                         search_radius['maxmag'], search_radius['minradius'],
                         search_radius['maxradius'], channels,
                         min_sample_rate, inventory, traveltime_phases, wtimespan,
                         retry, advanced_settings, class_labels, isterminal)
        logger.info("Download completed in %s",
                    tdstr(dt.timedelta(seconds=time.time()-starttime)))
        logger.info("")
        logger.info("%d total error(s), %d total warning(s)", run_row.errors, run_row.warnings)

    return 0


def process(dburl, pysourcefile, configsourcefile, outcsvfile, isterminal=False):
    """
        Main run method. KEEP the ARGUMENT THE SAME AS THE config.yaml OTHERWISE YOU'LL GET
        A DIFFERENT CONFIG SAVED IN THE DB
        :param processing: a dict as load from the config
    """
    with closing(dburl) as session:
        starttime = time.time()
        # config logger (FIXME: merge with download logger?):
        logger.setLevel(logging.INFO)  # this is necessary to configure logger HERE, otherwise the
        # handler below does not work. FIXME: better implementation!!
        logger_handler = logging.FileHandler(outcsvfile + ".log")
        logger_handler.setLevel(logging.DEBUG)
        logger.addHandler(logger_handler)

        csvwriter = [None]  # bad hack: in python3, we might use 'nonlocal' @UnusedVariable
        kwargs = dict(delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        with open(outcsvfile, 'wb') as csvfile:

            def ondone(result):
                if csvwriter[0] is None:
                    if isinstance(result, dict):
                        csvwriter[0] = csv.DictWriter(csvfile, fieldnames=result.keys(), **kwargs)
                        csvwriter[0].writeheader()
                    else:
                        csvwriter[0] = csv.writer(csvfile,  **kwargs)
                csvwriter[0].writerow(result)

            process_run(session, pysourcefile, ondone, configsourcefile, isterminal)

            logger.info("Processing completed in %s",
                        tdstr(dt.timedelta(seconds=time.time()-starttime)))
    return 0


@contextmanager
def closing(dburl, scoped=False, close_logger=True):
    """Opens a sqlalchemy session and closes it. Also closes and removes all logger handlers if
    close_logger is True (the default)
    :example:
        # configure logger ...
        with closing(dburl) as session:
            # ... do stuff, print to logger etcetera ...
        # session is closed and also the logger handlers
    """
    try:
        session = get_session(dburl, scoped=scoped)
        yield session
    except Exception as exc:
        logger.critical(str(exc))
        raise
    finally:
        try:
            session.close()
            session.bind.dispose()
        except NameError:
            pass
        if close_logger:
            handlers = logger.handlers[:]  # make a copy:
            for handler in handlers:
                try:
                    handler.close()
                    logger.removeHandler(handler)
                except (AttributeError, TypeError, IOError, ValueError):
                    pass


def click_option(*args, **kwargs):
    """Returns a click.option doing some pre-process: provides custom help from config.example and
    sets the default as config.example OR the default provided in kwargs"""
    for name in args:
        if name[:2] == '--':
            name = name[2:]
            break

    if 'default' in kwargs:
        kwargs['default'] = cfg_dict.get(name, kwargs['default'])
    else:
        kwargs['default'] = cfg_dict[name]

    kwargs['help'] = cfg_doc[name]
    return click.option(*args, **kwargs)


@click.group()
def main():
    """stream2segment is a program to download, process, visualize or annotate EIDA web services
    waveform data segments.
    According to the given command, segments can be:

    \b
    - efficiently downloaded (with metadata) in a custom database without polluting the filesystem
    - processed with little implementation effort by supplying a custom python file
    - visualized and annotated in a web browser

    Type:

    \b
    stream2segment COMMAND --help

    \b
    for details"""
    pass


@main.command(short_help='Efficiently download waveform data segments')
@click_option('-s', '--start', default=get_def_timerange()[0], type=valid_date)
@click_option('-e', '--end', default=get_def_timerange()[1], type=valid_date)
@click_option('-E', '--eventws')
@click_option('--wtimespan', nargs=2, type=int)
@click_option('--stimespan', nargs=2, type=int)
@click_option('-d', '--dburl')
@click_option('--min_sample_rate')
@click_option('-r', '--retry', default=False, is_flag=True)
@click_option('-i', '--inventory', default=False, is_flag=True)
@click.argument('eventws_query_args', nargs=-1, type=click.UNPROCESSED)
def d(start, end, eventws, wtimespan, stimespan, dburl, min_sample_rate, retry, inventory,
      eventws_query_args):
    """Efficiently download waveform data segments and relative events, stations and channels
    metadata (plus additional class labels, if needed)
    into a specified database for further processing or visual inspection in a
    browser. Options are listed below: when not specified, their default
    values are those set in the config file (`config.yaml`).
    [EVENTWS_QUERY_ARGS] is an optional list of space separated arguments to be passed
    to the event web service query (exmple: 'minmag 5.5 minlon 34.5') and will be merged with
    (overriding if needed) the arguments of `eventws_query_args` specified in in the config file,
    if any.
    All FDSN query arguments are valid
    *EXCEPT* 'start', 'end' and 'format' (the first two are set via the relative options, the
    format will default in most cases to 'text' for performance reasons)
    """
    # merge eventws_query_args
    eventws_query_args_ = cfg_dict.get('eventws_query_args', {})
    # stupid way to iterate as pair (key, value) in eventws_query_args as the latter is supposed to
    # be in the form (key, value, key, value,...):
    key = None
    for val in eventws_query_args:
        if key is None:
            key = val
        else:
            eventws_query_args_[key] = val
            key = None

    try:
        ret = download(dburl, start, end, eventws, eventws_query_args_,
                       stimespan, cfg_dict['search_radius'], cfg_dict['channels'], min_sample_rate,
                       inventory,
                       cfg_dict['traveltime_phases'], wtimespan, cfg_dict['processing'], retry,
                       cfg_dict['advanced_settings'], cfg_dict.get('class_labels', {}),
                       isterminal=True)
        sys.exit(ret)
    except KeyboardInterrupt:
        sys.exit(1)


@main.command(short_help='Process downloaded waveform data segments')
@click.argument('pyfile')
@click.argument('configfile')
@click.argument('outfile')
@click_option('-d', '--dburl')
def p(pyfile, configfile, outfile, dburl):
    """Process downloaded waveform data segments via a custom python file and a configuration
    file. Options are listed below. When missing, they default to the values provided in the
    config file `config.yaml`"""
    process(dburl, pyfile, configfile, outfile, isterminal=True)


@main.command(short_help='Visualize downloaded waveform data segments in a browser')
@click_option('-d', '--dburl')
def v(dburl):
    """Visualize downloaded waveform data segments in a browser.
    Options are listed below. When missing, they default to the values provided in the
    config file `config.yaml`"""
    showgui(dburl)


if __name__ == '__main__':
    main()  # pylint: disable=E1120
