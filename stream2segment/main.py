# -*- coding: utf-8 -*-
"""
Main module of the stream2segment package. Entry points and click commands are defined here

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
from __future__ import print_function

# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import object

import logging
import sys
import os
from contextlib import contextmanager
import shutil

# iterate over dictionary keys without list allocation in both py 2 and 3:
from future.utils import viewitems
from datetime import datetime, timedelta
import yaml
import click
from click.exceptions import BadParameter, ClickException, MissingParameter

from stream2segment.utils.log import configlog4download, configlog4processing,\
    elapsedtime2logger_when_finished, configlog4stdout
# from stream2segment.download.utils import run_instance
from stream2segment.utils.resources import version
from stream2segment.io.db.models import Segment, Download
from stream2segment.process.main import run as run_process, to_csv, default_funcname
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


class clickutils(object):
    """Container for Options validations, default settings so as not to pollute the click
    decorators"""

    TERMINAL_HELP_WIDTH = 90  # control width of help. 80 should be the default (roughly)
    NOW = datetime.utcnow()
    DEFAULTDOC = yaml_load_doc(get_templates_fpath("download.yaml"))
    DBURLDOC_SUFFIX = ("****IMPORTANT NOTE****: It can also be the path of a yaml file "
                       "containing the property 'dburl' (e.g., the yaml you used for "
                       "downloading, so as to avoid re-typing the database path)")

    @classmethod
    def valid_date(cls, obj):
        """does a check on string to see if it's a valid datetime string.
        If integer, is a datetime object relative to today, at midnight, plus
        `string` days (negative values are allowed)
        Returns the datetime on success, throws an BadParameter otherwise"""
        try:
            return strptime(obj)  # if obj is datetime, returns obj
        except ValueError as _:
            try:
                days = int(obj)
                endt = datetime(cls.NOW.year, cls.NOW.month, cls.NOW.day, 0, 0, 0, 0)
                return endt - timedelta(days=days)
            except Exception:
                pass
        raise BadParameter("Invalid date time or invalid integer")

    @classmethod
    def set_help_from_yaml(cls, ctx, param, value):
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
        \@click.Option('--opt1', ..., callback=set_help_from_yaml, is_eager=True,...)
        \@click.Option('--opt2'...)
        \@click.Option('--opt3'..., help='my custom help do not set the config help')
        \@click.Option('--opt4'...)
        ...
        ```
        """
        cfg_doc = cls.DEFAULTDOC
        for option in (opt for opt in ctx.command.params if opt.param_type_name == 'option'):
            if option.help is None:
                option.help = cfg_doc.get(option.name, None)

        return value

    @staticmethod
    def proc_eventws_args(ctx, param, value):
        """parses optional event query args (when the 'd' command is issued) into a dict"""
        # use iter to make a dict from a list whose even indices = keys, odd ones = values
        # https://stackoverflow.com/questions/4576115/convert-a-list-to-a-dictionary-in-python
        itr = iter(value)
        return dict(zip(itr, itr))

    @staticmethod
    def extract_dburl_if_yaml(ctx, param, value):
        """
        For all non-download click Options, returns the database path from 'value':
        'value' can be a file (in that case is assumed to be a yaml file with the
        'dburl' key in it) or the database path otherwise
        """
        if value and os.path.isfile(value):
            try:
                value = yaml_load(value)['dburl']
            except Exception:
                raise BadParameter("'dburl' not found in '%s'" % value)
        if not value:
            raise MissingParameter("dburl")
        return value


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
        download_inst = Download(config=tounicode(yaml.safe_dump(yaml_dict,
                                                                 default_flow_style=False)),
                                 # log by default shows error. If everything works fine, we replace
                                 # the content later
                                 log=('Content N/A: this is probably due to an unexpected'
                                      'and out-of-control interruption of the download process '
                                      '(e.g., memory error)'), program_version=version())

        session.add(download_inst)
        session.commit()
        download_id = download_inst.id
        session.close()  # frees memory?

        if isterminal:
            print("Arguments:")
            # replace dbrul passowrd for printing to terminal
            # Note that we remove dburl from yaml_dict cause query_main gets its session object
            # (which we just built)
            yaml_safe = dict(yaml_dict, dburl=secure_dburl(yaml_dict.pop('dburl')))
            print(indent(yaml.safe_dump(yaml_safe, default_flow_style=False), 2))

        loghandler = configlog4download(logger, session, download_id, isterminal)

        if isterminal:
            print("Log messages will be temporarily written to '%s'" % str(loghandler.baseFilename))
            print("If the program does not quit for external causes (e.g., memory overflow), "
                  "the file will be deleted before exiting and its content will "
                  "be written to the table '%s' (column '%s')" % (Download.__tablename__,
                                                                  Download.log.key))

        with elapsedtime2logger_when_finished(logger):
            run_download(session=session, download_id=download_id, isterminal=isterminal,
                         **yaml_dict)
            logger.info("%d total error(s), %d total warning(s)", loghandler.errors,
                        loghandler.warnings)

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


# def data_aval(dburl, outfile, max_gap_ovlap_ratio=0.5):
#     from stream2segment.gui.da_report.main import create_da_html
#     # errors are printed to terminal:
#     configlog4stdout(logger)
#     with closing(dburl) as session:
#         create_da_html(session, outfile, max_gap_ovlap_ratio, True)
#     if os.path.isfile(outfile):
#         import webbrowser
#         webbrowser.open_new_tab('file://' + os.path.realpath(outfile))


def create_templates(outpath, prompt=True, *filenames):
    # get the template files. Use all files except those with more than one dot
    # This might be better implemented
    if not os.path.isdir(outpath):
        os.makedirs(outpath)
        if not os.path.isdir(outpath):
            raise Exception("Unable to create '%s'" % outpath)
    template_files = get_templates_fpaths(*filenames)
    if prompt:
        existing_files = [t for t in template_files
                          if os.path.isfile(os.path.join(outpath, os.path.basename(t)))]
        non_existing_files = [t for t in template_files if t not in existing_files]
        if existing_files:
            suffix = ("Type:\n1: overwrite all files\n2: write only non-existing\n"
                      "0 or any other value: do nothing (exit)")
            msg = ("The following file(s) "
                   "already exist on '%s':\n%s"
                   "\n\n%s") % (outpath, "\n".join([os.path.basename(_)
                                                    for _ in existing_files]), suffix)
            val = click.prompt(msg)
            try:
                val = int(val)
                if val == 2:
                    if not len(non_existing_files):
                        raise ValueError()
                    else:
                        template_files = non_existing_files
                elif val != 1:
                    raise ValueError()
            except ValueError:
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
    - efficiently downloaded (with metadata) in a custom sqlite or postgres database
    - processed with little implementation effort by supplying a custom python file
    - visualized and annotated in a web browser

    For details, type:

    \b
    stream2segment COMMAND --help

    \b
    where COMMAND is one of the commands listed below"""
    pass


@main.command(short_help='Efficiently download waveform data segments',
              context_settings=dict(max_content_width=clickutils.TERMINAL_HELP_WIDTH))
@click.option("-c", "--configfile",
              help="The path to the configuration file in yaml format "
                   "(https://learn.getgrav.org/advanced/yaml).",
              type=click.Path(exists=True, file_okay=True, dir_okay=False, writable=False,
                              readable=True), is_eager=True,
              callback=clickutils.set_help_from_yaml)
@click.option('-d', '--dburl')
@click.option('-e', '--eventws')
@click.option('-t0', '--start', type=clickutils.valid_date)
@click.option('-t1', '--end', type=clickutils.valid_date)
@click.option('-ds', '--dataws')
@click.option('--min_sample_rate')
@click.option('-t', '--traveltimes_model')
@click.option('-w', '--wtimespan', nargs=2, type=float)
@click.option('-r1', '--retry_url_errors', is_flag=True, default=None)
@click.option('-r2', '--retry_mseed_errors', is_flag=True, default=None)
@click.option('-r3', '--retry_no_code', is_flag=True, default=None)
@click.option('-r4', '--retry_4xx', is_flag=True, default=None)
@click.option('-r5', '--retry_5xx', is_flag=True, default=None)
@click.option('-i', '--inventory', is_flag=True, default=None)
@click.argument('eventws_query_args', nargs=-1, type=click.UNPROCESSED,
                callback=clickutils.proc_eventws_args)
def d(configfile, dburl, eventws, start, end, dataws, min_sample_rate, traveltimes_model,
      wtimespan, retry_no_code, retry_url_errors, retry_mseed_errors, retry_4xx, retry_5xx,
      inventory, eventws_query_args):
    """Efficiently download waveform data segments and relative events, stations and channels
    metadata into a specified database for further processing or visual inspection in a
    browser. The -c option (required) sets the defaults for all other options below, which are
    optional.
    The argument 'eventws_query_args' is an optional list of space separated key and values to be
    passed to the event web service query (example: minmag 5.5 minlon 34.5). All FDSN query
    arguments are valid except 'start', 'end' (set them via -t0 and -t1) and 'format'
    """
    try:
        cfg_dict = yaml_load(configfile, **{k: v for k, v in locals().items()
                                            if v not in ((), {}, None, configfile)})
        # start and end might be integers. If we attach the conversion function
        # `clickutils.valid_date` to the relative clikc Option 'type' argument, the
        # function does not affect integer values in the config. Thus we need to set it here:
        cfg_dict['start'] = clickutils.valid_date(cfg_dict['start'])
        cfg_dict['end'] = clickutils.valid_date(cfg_dict['end'])
        ret = download(isterminal=True, **cfg_dict)
        sys.exit(ret)
    except KeyboardInterrupt:  # this except avoids printing traceback
        sys.exit(1)  # exit with 1 as normal python exceptions


@main.command(short_help='Process downloaded waveform data segments',
              context_settings=dict(max_content_width=clickutils.TERMINAL_HELP_WIDTH))
@click.option('-d', '--dburl', callback=clickutils.extract_dburl_if_yaml,
              help="%s.\n%s" % (clickutils.DEFAULTDOC['dburl'], clickutils.DBURLDOC_SUFFIX))
@click.option("-c", "--configfile",
              help="The path to the configuration file in yaml format "
                   "(https://learn.getgrav.org/advanced/yaml).",
              type=click.Path(exists=True, file_okay=True, dir_okay=False, writable=False,
                              readable=True))
@click.option("-p", "--pyfile",
              help="The path to the python file where to implement the processing function "
                   "which will be called iteratively on each segment "
                   "selected in the config file",
              type=click.Path(exists=True, file_okay=True, dir_okay=False, writable=False,
                              readable=True))
@click.option("-f", "--funcname",
              help="The name of the function to execute in the given python file. "
                   "Optional: defaults to '%s' when missing" % default_funcname(),
              )  # do not set default='main', so that we can test when arg is missing or not
@click.argument('outfile')
def p(dburl, configfile, pyfile, funcname, outfile):
    """Process downloaded waveform data segments via a custom python file and a configuration
    file"""
    try:
        process(dburl, pyfile+":"+funcname if funcname else pyfile, configfile, outfile,
                isterminal=True)
    except KeyboardInterrupt:  # this except avoids printing traceback
        sys.exit(1)  # exit with 1 as normal python exceptions


@main.command(short_help='Visualize downloaded waveform data segments in a browser',
              context_settings=dict(max_content_width=clickutils.TERMINAL_HELP_WIDTH))
@click.option('-d', '--dburl', callback=clickutils.extract_dburl_if_yaml,
              help="%s.\n%s" % (clickutils.DEFAULTDOC['dburl'], clickutils.DBURLDOC_SUFFIX))
@click.option("-c", "--configfile",
              help="The path to the configuration file in yaml format "
                   "(https://learn.getgrav.org/advanced/yaml).",
              type=click.Path(exists=True, file_okay=True, dir_okay=False, writable=False,
                              readable=True))
@click.option("-p", "--pyfile",
              help="The path to the python file with the plot functions implemented",
              type=click.Path(exists=True, file_okay=True, dir_okay=False, writable=False,
                              readable=True))
def v(dburl, configfile, pyfile):
    """Visualize downloaded waveform data segments in a browser"""
    visualize(dburl, pyfile, configfile)


# @main.command(short_help='Create a data availability html file showing downloaded data '
#                          'quality on a map')
# @click.option('-d', '--dburl', callback=clickutils.set_dburl, is_eager=True)
# @click.option('-m', '--max_gap_ovlap_ratio', help="""Sets the maximum gap/overlap ratio.
# Mark segments has 'corrupted' because of gaps/overlaps if they exceed this threshold.
# Defaults to 0.5 (half of the segment sampling frequency)""", default=0.5)
# @click.argument('outfile')
# def a(dburl, max_gap_ovlap_ratio, outfile):
#     """Creates a data availability html file, where the user can interactively inspect the
#     quality of the waveform data downloaded"""
#     data_aval(dburl, outfile, max_gap_ovlap_ratio)


@main.command(short_help='Create template/config files in a specified directory',
              context_settings=dict(max_content_width=clickutils.TERMINAL_HELP_WIDTH))
@click.argument('outdir')
def t(outdir):
    """Creates template/config files which can be inspected and edited for launching download,
    processing and visualization.
    """
    helpdict = {"download.yaml": "download configuration file ('s2s d' -c option)",
                "processing.py": "processing/gui python file ('s2s p' and 's2s v' -p option)",
                "processing.yaml": ("processing/gui configuration file "
                                    "('s2s p' and 's2s v' -c option)")}
    try:
        copied_files = create_templates(outdir, True, *helpdict)
        print('')
        if not copied_files:
            print("No file copied")
        else:
            print("%d file(s) copied in '%s':" % (len(copied_files), outdir))
            for fcopied in copied_files:
                bname = os.path.basename(fcopied)
                print("%s: %s" % (bname, helpdict.get(bname, "")))
            sys.exit(0)
    except Exception as exc:
        print('')
        print("error: %s" % str(exc))
    sys.exit(1)


if __name__ == '__main__':
    main()  # pylint: disable=E1120
