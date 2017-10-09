# -*- coding: utf-8 -*-
"""
Main module of the stream2segment package. Entry points and click commands are defined here

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
from __future__ import print_function

# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import object

import sys
import os

# iterate over dictionary keys without list allocation in both py 2 and 3:
from datetime import datetime, timedelta

import click
from click.exceptions import BadParameter, ClickException, MissingParameter

from stream2segment.process.main import default_funcname
from stream2segment.utils import strptime
from stream2segment.utils.resources import get_templates_fpath, yaml_load, yaml_load_doc

from stream2segment.core import helpmathiter, _TEMPLATE_FILES, create_templates, visualize,\
    download, process


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
                              readable=True),
              required=True  # type click.Path checks the existence only if option is provided.
              # Don't set required = True with eager=True: it suppresses --help
              )
@click.option("-p", "--pyfile",
              help="The path to the python file where to implement the processing function "
                   "which will be called iteratively on each segment "
                   "selected in the config file",
              type=click.Path(exists=True, file_okay=True, dir_okay=False, writable=False,
                              readable=True),
              required=True  # type click.Path checks the existence only if option is provided.
              # Don't set required = True with eager=True: it suppresses --help
              )
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
                              readable=True),
              required=True  # type click.Path checks the existence only if option is provided.
              # Don't set required = True with eager=True: it suppresses --help
              )
@click.option("-p", "--pyfile",
              help="The path to the python file with the plot functions implemented",
              type=click.Path(exists=True, file_okay=True, dir_okay=False, writable=False,
                              readable=True),
              required=True  # type click.Path checks the existence only if option is provided.
              # Don't set required = True with eager=True: it suppresses --help
              )
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
    processing and visualization. OUTDIR will be created if it does not exist
    """
    existing_files = [t for t in _TEMPLATE_FILES
                      if os.path.isfile(os.path.join(outdir, os.path.basename(t)))]
    copied_files = []
    if existing_files:
        suffix = ("Type:\n1: overwrite all files\n2: write only non-existing\n"
                  "0 or any other value: do nothing (exit)")
        msg = ("The following file(s) "
               "already exist on '%s':\n%s"
               "\n\n%s") % (outdir, "\n".join([os.path.basename(_)
                                               for _ in existing_files]), suffix)
        val = click.prompt(msg)
        copied_files = []
        if val in ('1', '2'):
            try:
                copied_files = create_templates(outdir, val == '1')
            except Exception as exc:
                print("ERROR: %s" % str(exc))
                sys.exit(1)

        if not copied_files:
            print("No file copied")
        else:
            print("%d file(s) copied in '%s':" % (len(copied_files), outdir))
            for fcopied in copied_files:
                bname = os.path.basename(fcopied)
                print("   %s: %s" % (bname, _TEMPLATE_FILES.get(bname, "")))
            sys.exit(0)


@main.command(short_help='Show quick help on stream2segment built-in math functions',
              context_settings=dict(max_content_width=clickutils.TERMINAL_HELP_WIDTH))
@click.option("-t", "--type", type=click.Choice(['numpy', 'obspy', 'all']), default='all',
              show_default=True,
              help="Show help only for the function matching the given type. Numpy indicates "
                    "functions operating on numpy arrays (module `stream2segment.analysis`). "
                    "Obspy (module `stream2segment.analysis.mseeds`) those operating on obspy "
                    "Traces, most of which are simply the numpy counterparts defined for Trace "
                    "objects")
@click.option("-f", "--filter", default='*', show_default=True,
              help="Show doc only for the function whose name matches the given filter. "
                    "Wildcards (* and ?) are allowed")
def h(type, filter):  # @ReservedAssignment
    for line in helpmathiter(type, filter):
        print(line)

if __name__ == '__main__':
    main()  # pylint: disable=E1120
