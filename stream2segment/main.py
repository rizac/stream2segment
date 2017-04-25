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
from stream2segment.utils.resources import get_proc_template_files, get_default_cfg_filepath
from stream2segment.io.db import models
from stream2segment.io.db.pd_sql_utils import commit
from stream2segment.process.wrapper import run as process_run
from stream2segment.download.query import main as query_main
from stream2segment.utils import tounicode, yaml_load, get_session, strptime, yaml_load_doc,\
    get_default_dbpath, printfunc, indent, secure_dburl

# set root logger if we are executing this module as script, otherwise as module name following
# logger conventions. Discussion here:
# http://stackoverflow.com/questions/30824981/do-i-need-to-explicitly-check-for-name-main-before-calling-getlogge
# howver, based on how we configured entry points in config, the name is (as november 2016)
# 'stream2segment.main', which messes up all hineritances. So basically setup a main logger
# with the package name
logger = logging.getLogger("stream2segment")


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
    pyfile, configfile = get_proc_template_files()
    shutil.copy2(pyfile, outpath)
    outconfigpath = get_template_config_path(outpath)
    shutil.copy2(configfile, outconfigpath)
    return outpath, outconfigpath


def visualize(dburl):
    from stream2segment.gui import main as main_gui
    main_gui.run_in_browser(dburl)
    return 0


def data_aval(dburl, outfile):
    from stream2segment.gui.da_report.main import create_da_html
    # errors are printed to terminal:
    configlog4stdout(logger)
    with closing(dburl) as session:
        create_da_html(session, outfile, True)
    if os.path.isfile(outfile):
        import webbrowser
        webbrowser.open_new_tab('file://' + os.path.realpath(outfile))


# IMPORTANT !!!
# IMPORTANT: THE ARGUMENT NAMES HERE MUST BE THE SAME AS THE CONFIG FILE!!! SEE FUNCTION DOC BELOW
# IMPORTANT !!!
def download(dburl, start, end, eventws, eventws_query_args, stimespan,
             search_radius,
             channels, min_sample_rate, inventory, traveltime_phases, wtimespan,
             retry, advanced_settings, class_labels=None, isterminal=False):
    """
        Main run method. KEEP the ARGUMENT THE SAME AS THE config.yaml OTHERWISE YOU'LL GET
        A DIFFERENT CONFIG SAVED IN THE DB
        :param processing: a dict as load from the config
    """
    yaml_dict = dict(locals())  # this must be the first statement, so that we catch all arguments
    # and no local variable (none has been declared yet). Note: dict(locals()) avoids problems with
    # variables created inside loops, when iterating over _args_ (see below)
    yaml_dict.pop('isterminal')  # not a yaml var
    # remove db url password when printing:

    with closing(dburl) as session:
        try:
            # print local vars: use safe_dump to avoid python types. See:
            # http://stackoverflow.com/questions/1950306/pyyaml-dumping-without-tags
            run_inst = run_instance(session, config=tounicode(yaml.safe_dump(yaml_dict,
                                                                             default_flow_style=False)))
    
            echo = printfunc(isterminal)  # no-op if argument is False
            echo("Arguments:")
            # replace dbrul passowrd for printing to terminal
            yaml_dict['dburl'] = secure_dburl(yaml_dict['dburl'])
            echo(indent(yaml.safe_dump(yaml_dict, default_flow_style=False), 2))
    
            configlog4download(logger, session, run_inst, isterminal)
            with elapsedtime2logger_when_finished(logger):
                query_main(session, run_inst.id, start, end, eventws, eventws_query_args,
                           stimespan, search_radius['minmag'],
                           search_radius['maxmag'], search_radius['minradius'],
                           search_radius['maxradius'], channels,
                           min_sample_rate, inventory, traveltime_phases, wtimespan,
                           retry, advanced_settings, class_labels, isterminal)
                logger.info("%d total error(s), %d total warning(s)", run_inst.errors,
                            run_inst.warnings)
        except Exception as exc:
            logger.critical(str(exc))
            raise

    return 0


def process(dburl, pysourcefile, configsourcefile, outcsvfile, isterminal=False):
    """
        Process the segment saved in the db and saves the results into a csv file
        :param processing: a dict as load from the config
    """
    with closing(dburl) as session:
        try:
            echo = printfunc(isterminal)  # no-op if argument is False
            echo("Processing, please wait")
            logger.info('Output file: %s', outcsvfile)

            configlog4processing(logger, outcsvfile, isterminal)
            csvwriter = [None]  # bad hack: in python3, we might use 'nonlocal' @UnusedVariable
            kwargs = dict(delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            flush_num = [1, 10]  # determines when to flush (not used. We use the
            # last arg to open tells to flush line-wise. To add custom flush, see commented
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
                    process_run(session, pysourcefile, ondone, configsourcefile, isterminal)
        except Exception as exc:
            logger.critical(str(exc))
            raise exc
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


@click.group()
def main():
    """stream2segment is a program to download, process, visualize or annotate EIDA web services
    waveform data segments.
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


def setup_help(ctx, param, value):
    """this function does not check the value (it simply returns it) but
    dynamically sets help and default values from config. It must be
    attached to an eager click.Option so that the option is executed FIRST (before even --help)
    """
    # define iterator over options (no arguments):
    def _optsiter():
        for option in ctx.command.params:
            if option.param_type_name == 'option':
                yield option

    # cfg_dict = yaml_load(get_default_cfg_filepath(filename='config.example.yaml'))
    cfg_doc = yaml_load_doc()

    for option in _optsiter():
        if option.help is None:
            option.help = cfg_doc[option.name]

    return value


def proc_e(ctx, param, value):
    """parses optional event query args into a dict"""
    # stupid way to iterate as pair (key, value) in eventws_query_args as the latter is supposed to
    # be in the form (key, value, key, value,...):
    ret = {}
    key = None
    for val in value:
        if key is None:
            key = val
        else:
            ret[key] = val
            key = None
    return ret


def config_defaults_when_missing():
    """defaults for download cannot be set via click cause they need to be set only if
    missing in the config file"""
    start_def, end_def = get_def_timerange()
    return dict(start=start_def, end=end_def, retry=False, inventory=False)


@main.command(short_help='Efficiently download waveform data segments')
@click.option("-c", "--configfile", default=get_default_cfg_filepath(),
              help=("The path to the configuration file. If missing, it defaults to `config.yaml` "
                    "in the stream2segment directory"), type=click.Path(exists=True,
                                                                        file_okay=True,
                                                                        dir_okay=False,
                                                                        writable=False,
                                                                        readable=True),
              is_eager=True, callback=setup_help)
@click.option('-d', '--dburl')
@click.option('-s', '--start', type=valid_date)
@click.option('-e', '--end', type=valid_date)
@click.option('-E', '--eventws')
@click.option('--wtimespan', nargs=2, type=int)
@click.option('--min_sample_rate')
@click.option('-r1', '--retry_no_code', is_flag=True)
@click.option('-r2', '--retry_url_errors', is_flag=True)
@click.option('-r3', '--retry_mseed_errors', is_flag=True)
@click.option('-r4', '--retry_4xx', is_flag=True)
@click.option('-r5', '--retry_5xx', is_flag=True)
@click.option('-i', '--inventory', is_flag=True)
@click.argument('eventws_query_args', nargs=-1, type=click.UNPROCESSED, callback=proc_e)
def d(configfile, dburl, start, end, eventws, wtimespan, min_sample_rate, retry_no_code, retry_url_errors,
                            retry_mseed_errors,
                            retry_4xx, retry_5xx,
      inventory, eventws_query_args):
    """Efficiently download waveform data segments and relative events, stations and channels
    metadata (plus additional class labels, if needed)
    into a specified database for further processing or visual inspection in a
    browser. Options are listed below: when not specified, their default
    values are those set in the value of the configfile option.
    [EVENTWS_QUERY_ARGS] is an optional list of space separated arguments to be passed
    to the event web service query (exmple: minmag 5.5 minlon 34.5) and will be added to
    (or override) the arguments of `eventws_query_args` specified in in the config file,
    if any.
    All FDSN query arguments are valid
    *EXCEPT* 'start', 'end' and 'format' (the first two are set via the relative options, the
    format will default in most cases to 'text' for performance reasons)
    """
    _ = dict(locals())
    cfg_dict = yaml_load(_.pop('configfile'))

    # override with command line values, if any:
    for var, val in _.iteritems():
        if val not in (None, (), {}, []):
            cfg_dict[var] = val
    # set defaults when missing. This cannot be set via click.Option(default=...) because
    # we wouldn't diistinguish the cases [provided as command line / not provided as command line]
    # when the option is also provided in the config
    for key, val in config_defaults_when_missing().iteritems():
        if key not in cfg_dict:
            cfg_dict[key] = val

    try:
        ret = download(isterminal=True, **cfg_dict)
        sys.exit(ret)
    except KeyboardInterrupt:
        sys.exit(1)


@main.command(short_help='Process downloaded waveform data segments')
@click.argument('pyfile')
@click.argument('configfile')
@click.argument('outfile')
@click.option('-d', '--dburl', callback=setup_help, is_eager=True, default=get_default_dbpath())
def p(pyfile, configfile, outfile, dburl):
    """Process downloaded waveform data segments via a custom python file and a configuration
    file. Options are listed below. When missing, they default to the values provided in the
    config file `config.yaml`"""
    process(dburl, pyfile, configfile, outfile, isterminal=True)


@main.command(short_help='Visualize downloaded waveform data segments in a browser')
@click.option('-d', '--dburl', callback=setup_help, is_eager=True, default=get_default_dbpath())
def v(dburl):
    """Visualize downloaded waveform data segments in a browser.
    Options are listed below. When missing, they default to the values provided in the
    config file `config.yaml`"""
    visualize(dburl)


@main.command(short_help='Create a data availability html file showing downloaded data '
                         'quality on a map')
@click.option('-d', '--dburl', callback=setup_help, is_eager=True, default=get_default_dbpath())
@click.argument('outfile')
def a(dburl, outfile):
    """Creates a data availability html file, where the user can interactively inspect the
    quality of the waveform data downloaded"""
    data_aval(dburl, outfile)


@main.command(short_help='Creates processing template file(s)')
@click.argument('outfile')
def t(outfile):
    """Creates a template python file which can be inspected and edited for launching processing.
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
