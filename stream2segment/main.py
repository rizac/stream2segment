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
# from StringIO import StringIO
import datetime as dt
import os
from contextlib import contextmanager
# import csv
import shutil

import yaml
import click
from click.exceptions import BadParameter, ClickException, MissingParameter

from stream2segment.utils.log import configlog4download, configlog4processing,\
    elapsedtime2logger_when_finished, configlog4stdout
from stream2segment.download.utils import run_instance
# from stream2segment.utils.resources import get_proc_template_files
from stream2segment.io.db.models import Segment, Run
# from stream2segment.io.db.pd_sql_utils import commit
from stream2segment.process.main import run as run_process, to_csv
from stream2segment.download.main import run as run_download
from stream2segment.utils import tounicode, get_session, strptime,\
    indent, secure_dburl
from stream2segment.utils.resources import get_templates_fpath, yaml_load, yaml_load_doc,\
    get_templates_fpaths


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

    _external_file_suffix = """To set up a directory with all necessary files run the program with
the 't' option first ('t --help' for help): the files are ready-to-use as program arguments, but
can be customized editing them and following the help instructions implemented therein"""

    @staticmethod
    def get_config_help():
        return ("The path to the configuration file in yaml format "
                "(https://learn.getgrav.org/advanced/yaml). "
                "%s" % click_stuff._external_file_suffix)

    @staticmethod
    def get_v_pyfile_help():
        return ("The path to the python file "
                "where to implement the plots for the GUI. %s" % click_stuff._external_file_suffix)

    @staticmethod
    def get_p_pyfile_help():
        return ("The path to the python file where to implement the processing function. "
                "The function will be then automatically called iteratively on each segment to "
                "create the csv file. %s" % click_stuff._external_file_suffix)

    @staticmethod
    def set_help_from_download_yaml(ctx, param, value):
        """
        When attaching this function as `callback` argument to an Option (`click.Option`),
        it will set
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

    dburl_opt_help = """Database path where to fetch the segments.
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

    @staticmethod
    def check_dburl(ctx, param, value):
        """
        For all non-download click Options, returns the database path from 'value':
        'value' can be a file (in that case is assumed to be a yaml file with the
        'dburl' key in it) or the database path otherwise
        """
        if value and os.path.isfile(value):
            try:
                value = yaml_load(get_templates_fpath("download.yaml"))['dburl']
            except Exception:
                raise BadParameter("'dburl' not found in '%s'" % value)
        if not value:
            raise MissingParameter("dburl")
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


# main functionalities:


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

        fileout = configlog4download(logger, session, run_inst, isterminal)

        if isterminal:
            print("Log messages will be written to table '%s' (column '%s')" %
                  (Run.__tablename__, Run.log.key))
            print("and to '%s'" % str(fileout))

        with elapsedtime2logger_when_finished(logger):
            run_download(session=session, run_id=run_inst.id, isterminal=isterminal, **yaml_dict)
            logger.info("%d total error(s), %d total warning(s)", run_inst.errors,
                        run_inst.warnings)

    return 0


def process(dburl, pyfile, configfile, outcsvfile, isterminal=False):
    """
        Process the segment saved in the db and saves the results into a csv file
        :param processing: a dict as load from the config
    """
    with closing(dburl) as session:
        if isterminal:
            print("Processing, please wait")
        logger.info('Output file: %s', outcsvfile)

        configlog4processing(logger, outcsvfile, isterminal)
        with elapsedtime2logger_when_finished(logger):
            to_csv(outcsvfile, session, pyfile, configfile, isterminal)

    return 0


def visualize(dburl, pyfile, configfile):
    from stream2segment.gui import main as main_gui
    main_gui.run_in_browser(dburl, pyfile, configfile)
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


def create_templates(outpath, prompt=True, *filenames):
    # get the template files. Use all files except those with more than one dot
    # This might be better implemented
    template_files = get_templates_fpaths(*filenames)
    if prompt:
        existing_files = [t for t in template_files if os.path.isfile(t)]
        if existing_files:
            msg = ("The following file(s) "
                   "already exist on '%s':\n%s"
                   "\n\nOverwrite?") % (outpath, "\n".join([os.path.basename(_)
                                                           for _ in existing_files]))
            if not click.confirm(msg):
                return []
    copied_files = []
    for tfile in template_files:
        shutil.copy2(tfile, outpath)
        copied_files.append(os.path.join(outpath, os.path.basename(tfile)))
    return copied_files


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
    except:
        logger.critical(sys.exc_info()[1])
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
              help=click_stuff.get_config_help(),
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
@click.option('-e', '--eventws')
@click.argument('eventws_query_args', nargs=-1, type=click.UNPROCESSED,
                callback=click_stuff.proc_eventws_args)
def d(configfile, dburl, start, end, service, wtimespan, min_sample_rate, retry_no_code,
      retry_url_errors, retry_mseed_errors, retry_4xx, retry_5xx, inventory, eventws,
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
@click.option('-d', '--dburl', callback=click_stuff.check_dburl, help=click_stuff.dburl_opt_help)
@click.option("-c", "--configfile",
              help=click_stuff.get_config_help(),
              type=click.Path(exists=True, file_okay=True, dir_okay=False, writable=False,
                              readable=True))
@click.option("-p", "--pyfile",
              help=click_stuff.get_v_pyfile_help(),
              type=click.Path(exists=True, file_okay=True, dir_okay=False, writable=False,
                              readable=True))
@click.argument('outfile')
def p(dburl, configfile, pyfile, outfile):
    """Process downloaded waveform data segments via a custom python file and a configuration
    file"""
    try:
        process(dburl, pyfile, configfile, outfile, isterminal=True)
    except KeyboardInterrupt:  # this except avoids printing traceback
        sys.exit(1)  # exit with 1 as normal python exceptions


@main.command(short_help='Visualize downloaded waveform data segments in a browser')
@click.option('-d', '--dburl', callback=click_stuff.check_dburl, help=click_stuff.dburl_opt_help)
@click.option("-c", "--configfile",
              help=click_stuff.get_config_help(),
              type=click.Path(exists=True, file_okay=True, dir_okay=False, writable=False,
                              readable=True))
@click.option("-p", "--pyfile",
              help=click_stuff.get_v_pyfile_help(),
              type=click.Path(exists=True, file_okay=True, dir_okay=False, writable=False,
                              readable=True))
def v(dburl, configfile, pyfile):
    """Visualize downloaded waveform data segments in a browser"""
    visualize(dburl, pyfile, configfile)


# @main.command(short_help='Create a data availability html file showing downloaded data '
#                          'quality on a map')
# @click.option('-d', '--dburl', callback=click_stuff.set_dburl, is_eager=True)
# @click.option('-m', '--max_gap_ovlap_ratio', help="""Sets the maximum gap/overlap ratio.
# Mark segments has 'corrupted' because of gaps/overlaps if they exceed this threshold.
# Defaults to 0.5 (half of the segment sampling frequency)""", default=0.5)
# @click.argument('outfile')
# def a(dburl, max_gap_ovlap_ratio, outfile):
#     """Creates a data availability html file, where the user can interactively inspect the
#     quality of the waveform data downloaded"""
#     data_aval(dburl, outfile, max_gap_ovlap_ratio)


@main.command(short_help='Create template/config files in a specified directory')
@click.argument('outdir')
def t(outdir):
    """Creates template/config files which can be inspected and edited for launching download,
    processing and visualization.
    """
    helpdict = {'download.py': 'download configuration file (-c option)',
                'gui.py': 'visualization python file (-p option)',
                'gui.yaml': 'visualization configuration file (-c option)',
                'processing.py': 'processing python file (-p option)',
                'processing.yaml': 'processing configuration file (-c option)'}
    try:
        copied_files = create_templates(outdir, True, *helpdict)
        if not copied_files:
            print("No file copied")
        else:
            print("%d files copied in '%s':" % (len(copied_files), outdir))
            for fcopied in copied_files:
                bname = os.path.basename(fcopied)
                print("%s: %s" % (bname, helpdict.get(bname, "")))
            sys.exit(0)
    except Exception as exc:
        print("%s\n" % str(exc))
    sys.exit(1)


if __name__ == '__main__':
    main()  # pylint: disable=E1120
