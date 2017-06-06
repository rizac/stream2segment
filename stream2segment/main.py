# -*- coding: utf-8 -*-
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
from __future__ import print_function  # , unicode_literals
import logging
import sys
from StringIO import StringIO
import datetime as dt
import yaml
import os
import click
from click.exceptions import BadParameter
from contextlib import contextmanager
import csv
import shutil
from stream2segment.utils.log import configlog4download, configlog4processing,\
    elapsedtime2logger_when_finished, configlog4stdout
from stream2segment.download.utils import run_instance
# from stream2segment.utils.resources import get_proc_template_files
# from stream2segment.io.db import models
# from stream2segment.io.db.pd_sql_utils import commit
from stream2segment.process.main import run as run_process
from stream2segment.download.main import run as run_download
from stream2segment.utils import tounicode, yaml_load, get_session, strptime, yaml_load_doc,\
    indent, secure_dburl
from stream2segment.utils.resources import get_templates_fpath


# set root logger if we are executing this module as script, otherwise as module name following
# logger conventions. Discussion here:
# http://stackoverflow.com/questions/30824981/do-i-need-to-explicitly-check-for-name-main-before-calling-getlogge
# howver, based on how we configured entry points in config, the name is (as november 2016)
# 'stream2segment.main', which messes up all hineritances. So basically setup a main logger
# with the package name
logger = logging.getLogger("stream2segment")


class click_stuff(object):
    """just a wrapper (to make code more readable) around
    click stuff used for validating/defaults etcetera"""

    @staticmethod
    def valid_date(string):
        """does a check on string to see if it's a valid datetime string.
        This is executed only if the relative Option is given
        Returns the datetime on success, throws an BadParameter otherwise"""
        try:
            return strptime(string)
        except ValueError as exc:
            raise BadParameter(str(exc))
        # return string

    @staticmethod
    def set_help_from_download_yaml(ctx, param, value):
        """
        Attach this function as `callback` argument to an Option (`click.Option`), and it will set
        an automatic help for all Options of the same command, which do not have an `help`
        specified and are found in the default config file for downloading
        (currently `download.yaml`).
        The Option having as callback this function must also have `is_eager=True`.
        Example:
        Assuming opt1, opt2, opt3 are variables of the config yaml file, and opt4 not, this
        sets the default help for opt1 and opt2:
        ```
        \@click.Option('--opt1', ..., callback=set_help_from_download_yaml, is_eager=True,...)
        \@click.Option('--opt2'...)
        \@click.Option('--opt3'..., help='my custom help do not set the config help')
        \@click.Option('--opt4'...)
        ...
        ```
        """
        # define iterator over options (no arguments):
        def _optsiter():
            for option in ctx.command.params:
                if option.param_type_name == 'option':
                    yield option

        cfg_doc = yaml_load_doc(get_templates_fpath("download.yaml"))

        for option in _optsiter():
            if option.help is None:
                option.help = cfg_doc[option.name]

        return value

    @staticmethod
    def proc_eventws_args(ctx, param, value):
        """parses optional event query args into a dict for the 'd' command"""
        # stupid way to iterate as pair (key, value) in eventws_query_args as the latter is
        # supposed to be in the form (key, value, key, value,...):
        ret = {}
        key = None
        for val in value:
            if key is None:
                key = val
            else:
                ret[key] = val
                key = None
        return ret

    @staticmethod
    def set_dburl(ctx, param, value):
        """
        For all non-download options, returns the database path by reading it from
        `value`, if the latter is a file, or returning `value` otherwise (assuming in this
        case it is already a db path). Sets also the help for the option
        """

        param.help = """Database path where to fetch the segments.
It can be specified also as a yaml file path with the variable 'dburl' defined therein
(for instance, the config file previously used for downloading the data).
In any case, the db url must denote an sql database (currently supported are sqlite or postresql).
If sqlite, just write the path to your local file prefixed with 'sqlite:///'
(e.g., 'sqlite:////home/my_folder/db.sqlite'). If read from a config file,
non-absolute paths will be relative to the config file. If specified from the command line,
they should be relative to the current working directory, although this has not be tested.
If not sqlite, the syntax is:
dialect+driver://username:password@host:port/database
(e.g.: 'postgresql://smith:Hw_6,9@hh21.uni-northpole.org/stream2segment_db')
(for info see: http://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls)"""

        if value and os.path.isfile(value):
            value = yaml_load(get_templates_fpath("download.yaml"))['dburl']

        return value
# other utility functions:


def get_def_timerange():
    """ Returns the default time range when  not specified, for downloading data
    the returned tuple has two datetime objects: yesterday, at midniight and
    today, at midnight"""
    dnow = dt.datetime.utcnow()
    endt = dt.datetime(dnow.year, dnow.month, dnow.day)
    startt = endt - dt.timedelta(days=1)
    return startt, endt


def get_template_config_path(filepath):
    root, _ = os.path.splitext(filepath)
    outconfigpath = root + ".config.yaml"
    return outconfigpath


def create_template(outpath):
    pyfile, configfile =  'caz', 'wat'  #  FIXME: FIX THIS
    shutil.copy2(pyfile, outpath)
    outconfigpath = get_template_config_path(outpath)
    shutil.copy2(configfile, outconfigpath)
    return outpath, outconfigpath


# main functionalities:

def visualize(dburl):
    from stream2segment.gui import main as main_gui
    main_gui.run_in_browser(dburl)
    return 0


def data_aval(dburl, outfile, max_gap_ovlap_ratio=0.5):
    from stream2segment.gui.da_report.main import create_da_html
    # errors are printed to terminal:
    configlog4stdout(logger)
    with closing(dburl) as session:
        create_da_html(session, outfile, max_gap_ovlap_ratio, True)
    if os.path.isfile(outfile):
        import webbrowser
        webbrowser.open_new_tab('file://' + os.path.realpath(outfile))


def download(isterminal=False, **yaml_dict):
    """
        Downloads the given segment providing a set of keyword arguments to match those of the
        config file (see confi.example.yaml for details)
    """
    dburl = yaml_dict['dburl']
    with closing(dburl) as session:
        # print local vars: use safe_dump to avoid python types. See:
        # http://stackoverflow.com/questions/1950306/pyyaml-dumping-without-tags
        run_inst = run_instance(session, config=tounicode(yaml.safe_dump(yaml_dict,
                                                                         default_flow_style=False)))

        if isterminal:
            print("Arguments:")
            # replace dbrul passowrd for printing to terminal
            # Note that we remove dburl from yaml_dict cause query_main gets its session object
            # (which we just built)
            yaml_safe = dict(yaml_dict, dburl=secure_dburl(yaml_dict.pop('dburl')))
            print(indent(yaml.safe_dump(yaml_safe, default_flow_style=False), 2))

        configlog4download(logger, session, run_inst, isterminal)
        with elapsedtime2logger_when_finished(logger):
            run_download(session=session, run_id=run_inst.id, isterminal=isterminal, **yaml_dict)
            logger.info("%d total error(s), %d total warning(s)", run_inst.errors,
                        run_inst.warnings)

    return 0


def process(dburl, pysourcefile, configsourcefile, outcsvfile, isterminal=False):
    """
        Process the segment saved in the db and saves the results into a csv file
        :param processing: a dict as load from the config
    """
    with closing(dburl) as session:
        if isterminal:
            print("Processing, please wait")
        logger.info('Output file: %s', outcsvfile)

        configlog4processing(logger, outcsvfile, isterminal)
        csvwriter = [None]  # bad hack: in python3, we might use 'nonlocal' @UnusedVariable
        kwargs = dict(delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        flush_num = [1, 10]  # determines when to flush (not used. We use the
        # last arg to open whihc tells to flush line-wise. To add custom flush, see commented
        # lines at the end of the with statement and uncomment them
        with open(outcsvfile, 'wb', 1) as csvfile:

            def ondone(result):
                if csvwriter[0] is None:
                    if isinstance(result, dict):
                        csvwriter[0] = csv.DictWriter(csvfile, fieldnames=result.keys(),
                                                      **kwargs)
                        csvwriter[0].writeheader()
                    else:
                        csvwriter[0] = csv.writer(csvfile,  **kwargs)
                csvwriter[0].writerow(result)
                # if flush_num[0] % flush_num[1] == 0:
                #    csvfile.flush()  # this should force writing so if errors we have something
                #    # http://stackoverflow.com/questions/3976711/csvwriter-not-saving-data-to-file-why
                # flush_num[0] += 1

            with elapsedtime2logger_when_finished(logger):
                run_process(session, pysourcefile, ondone, configsourcefile, isterminal)

    return 0


@contextmanager
def closing(dburl, scoped=False, close_logger=True, close_session=True):
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
        print("caught!")
        logger.critical(str(exc))
        raise
    finally:
        if close_logger:
            handlers = logger.handlers[:]  # make a copy
            for handler in handlers:
                try:
                    handler.close()
                    logger.removeHandler(handler)
                except (AttributeError, TypeError, IOError, ValueError):
                    pass
        if close_session:
            # close the session at the **real** end! we might need it above when closing loggers!!!
            try:
                session.close()
                session.bind.dispose()
            except NameError:
                pass


# click commands:

@click.group()
def main():
    """stream2segment is a program to download, process, visualize or annotate massive amounts of
    seismic waveform data segments.
    According to the given command, segments can be:

    \b
    - efficiently downloaded (with metadata) in a custom database without polluting the filesystem
    - processed with little implementation effort by supplying a custom python file
    - visualized and annotated in a web browser

    For details, type:

    \b
    stream2segment COMMAND --help

    \b
    where COMMAND is one of the commands listed below"""
    pass


@main.command(short_help='Efficiently download waveform data segments')
@click.option("-c", "--configfile",
              help=("The path to the configuration file. For creating a default config file, "
                    "run the program with the 't' option first ('t --help' for help)"),
              type=click.Path(exists=True, file_okay=True, dir_okay=False, writable=False,
                              readable=True), is_eager=True,
              callback=click_stuff.set_help_from_download_yaml)
@click.option('-d', '--dburl')
@click.option('-t0', '--start', type=click_stuff.valid_date)
@click.option('-t1', '--end', type=click_stuff.valid_date)
@click.option('-s', '--service')
@click.option('--wtimespan', nargs=2, type=int)
@click.option('--min_sample_rate')
@click.option('-r1', '--retry_url_errors', is_flag=True, default=None)
@click.option('-r2', '--retry_mseed_errors', is_flag=True, default=None)
@click.option('-r3', '--retry_no_code', is_flag=True, default=None)
@click.option('-r4', '--retry_4xx', is_flag=True, default=None)
@click.option('-r5', '--retry_5xx', is_flag=True, default=None)
@click.option('-i', '--inventory', is_flag=True, default=None)
@click.argument('eventws_query_args', nargs=-1, type=click.UNPROCESSED,
                callback=click_stuff.proc_eventws_args)
def d(configfile, dburl, start, end, service, wtimespan, min_sample_rate, retry_no_code,
      retry_url_errors, retry_mseed_errors, retry_4xx, retry_5xx, inventory,
      eventws_query_args):
    """Efficiently download waveform data segments and relative events, stations and channels
    metadata (plus additional class labels, if needed)
    into a specified database for further processing or visual inspection in a
    browser. Options are listed below: when not specified, their default
    values are those set in the value of the configfile option.
    [EVENTWS_QUERY_ARGS] is an optional list of space separated arguments to be passed
    to the event web service query (example: minmag 5.5 minlon 34.5) and will be added to
    the arguments of `eventws_query_args` specified in in the config file,
    if any. In case of conflicts, the values command line values supplied here will override the
    config ones
    All FDSN query arguments are valid
    *EXCEPT* 'start', 'end' and 'format' (the first two are set via 't0' or 'start' and 't1' or
    'end'), the format will default in most cases to 'text' for performance reasons)
    """
    _ = dict(locals())
    cfg_dict = yaml_load(_.pop('configfile'))

    # set start and end as default if not provided. If specified in the command line, they will
    # be overidden below
    start_def, end_def = get_def_timerange()
    cfg_dict['start'] = cfg_dict.get('start', start_def)
    cfg_dict['end'] = cfg_dict.get('end', end_def)

    # override with command line values, if any:
    for var, val in _.iteritems():
        if val:
            cfg_dict[var] = val

    try:
        ret = download(isterminal=True, **cfg_dict)
        sys.exit(ret)
    except KeyboardInterrupt:  # this except avoids printing traceback
        sys.exit(1)  # exit with 1 as normal python exceptions


@main.command(short_help='Process downloaded waveform data segments')
@click.argument('pyfile')
@click.argument('configfile')
@click.argument('outfile')
@click.option('-d', '--dburl', callback=click_stuff.set_dburl, is_eager=True)
def p(pyfile, configfile, outfile, dburl):
    """Process downloaded waveform data segments via a custom python file and a configuration
    file. The argument --dburl (or -d) can be specified also as the config file path used for
    downloading data. In that case it will default to the 'dburl' variable defined therein"""
    try:
        process(dburl, pyfile, configfile, outfile, isterminal=True)
    except KeyboardInterrupt:  # this except avoids printing traceback
        sys.exit(1)  # exit with 1 as normal python exceptions


@main.command(short_help='Visualize downloaded waveform data segments in a browser')
@click.option('-d', '--dburl', callback=click_stuff.set_dburl, is_eager=True)
@click.option('-c', '--configfile', type=click.Path(exists=True, file_okay=True, dir_okay=False,
                                                    writable=False,
                                                    readable=True))
def v(dburl, configgile):
    """Visualize downloaded waveform data segments in a browser.
    Options are listed below. When missing, they default to the values provided in the
    config file `config.yaml`"""
    visualize(dburl)


@main.command(short_help='Create a data availability html file showing downloaded data '
                         'quality on a map')
@click.option('-d', '--dburl', callback=click_stuff.set_dburl, is_eager=True)
@click.option('-m', '--max_gap_ovlap_ratio', help="""Sets the maximum gap/overlap ratio.
Mark segments has 'corrupted' because of gaps/overlaps if they exceed this threshold.
Defaults to 0.5 (half of the segment sampling frequency)""", default=0.5)
@click.argument('outfile')
def a(dburl, max_gap_ovlap_ratio, outfile):
    """Creates a data availability html file, where the user can interactively inspect the
    quality of the waveform data downloaded"""
    data_aval(dburl, outfile, max_gap_ovlap_ratio)


@main.command(short_help='Creates template/config files in a specified directory')
@click.argument('outfile')
def t(outfile):
    """Creates template/config files which can be inspected and edited for launching download and
    processing.
    A config file in the same path is also created with the same name and suffix 'config.yaml'.
    If either file already exists, the program will ask for confirmation
    """
    try:
        outconfigfile = get_template_config_path(outfile)
        msgs = ["'%s' already exists" % outfile if os.path.isfile(outfile) else "",
                "'%s' already exists" % outconfigfile if os.path.isfile(outconfigfile) else ""]
        msgs = [m for m in msgs if m]  # remove empty strings
        if not msgs or click.confirm("%s.\nOverwrite?" % "\n".join(msgs)):
            out1, out2 = create_template(outfile)
            sys.stdout.write("template processing python file written to '%s'\n" % out1)
            sys.stdout.write("template config yaml file written to '%s'\n" % out2)
    except Exception as exc:
        sys.stderr.write("%s\n" % str(exc))


if __name__ == '__main__':
    main()  # pylint: disable=E1120
