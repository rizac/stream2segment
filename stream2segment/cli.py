"""
Module implementing the Command line interface (cli) to access function in the main module

:date: Oct 8, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""

# provide some imports to let python3 syntax work also in python 2.7+ effortless.
# Any of the defaults import below can be safely removed if python2+
# compatibility is not needed

# standard python imports (must be the first import)
from __future__ import absolute_import, division, print_function

# future direct imports (needs future package installed, otherwise remove):
# (http://python-future.org/imports.html#explicit-imports)
from builtins import (ascii, bytes, chr, dict, filter, hex, input,
                      int, map, next, oct, open, pow, range, round,
                      str, super, zip)

# NOTE: do not use future aliased imports
# (http://python-future.org/imports.html#aliased-imports), they fail with urllib
# related  functions with multithreading (see utils.url module)

import sys
import os
import warnings
from collections import OrderedDict

import click

# from stream2segment import main

from stream2segment.utils.resources import get_templates_fpath, yaml_load_doc
from stream2segment.utils import inputvalidation


class clickutils(object):  # noqa
    """Container for Options validations, default settings so as not to
    pollute the click decorators
    """
    DEFAULTDOC = yaml_load_doc(get_templates_fpath("download.yaml"))
    EQA = "(event search parameter)"
    DBURL_OR_YAML_ATTRS = dict(type=inputvalidation.valid_dburl_or_download_yamlpath,
                               metavar='TEXT or PATH',
                               help=("Database URL where data has been saved. "
                                     "It can also be the path of a YAML file "
                                     "with the property 'dburl' (e.g., the "
                                     "config file used for downloading). "
                                     "IMPORTANT: if the URL contains passwords "
                                     "we STRONGLY suggest to use a file instead "
                                     "of typing the URL on the terminal"),
                               required=True)
    ExistingPath = click.Path(exists=True, file_okay=True, dir_okay=False,
                              writable=False, readable=True)

    @classmethod
    def options_missing_help_from_yaml(cls, command, *args, **kwargs):
        """Decorator to the `download`command to set missing options help
        from the relative YAML parameter, if found in the YAML file
        "download.yaml"
        """
        cfg_doc = cls.DEFAULTDOC
        for option in (opt for opt in command.params if
                       opt.param_type_name == 'option'):
            if option.help is None:
                option.help = cfg_doc.get(option.name, "")
                # remove implementation details from the cli (avoid too much information,
                # or information specific to the yaml file and not the cli):
                idx = option.help.find('Implementation details:')
                if idx > -1:
                    option.help = option.help[:idx]

        return command

    @staticmethod
    def _config_cmd_kwargs(**kwargs):
        """Configures a new Command (or Group) with default arguments"""
        # increase width of help on terminal (default ~= 80):
        context_settings = dict(max_content_width=85)
        kwargs.setdefault('context_settings', context_settings)
        kwargs.setdefault('options_metavar', '[options]')
        return kwargs

    class MyCommand(click.Command):
        """Class used for any click Command in this module"""
        def __init__(self, *arg, **kwargs):
            # configure default arguments:
            super().__init__(*arg, **clickutils._config_cmd_kwargs(**kwargs))

        def format_options(self, ctx, formatter):
            """Write all the options into the formatter if they exist.
            Overwrite super implementation to provide custom formatting
            """
            # same as superclass:
            opts = []
            for param in self.get_params(ctx):
                rv = param.get_help_record(ctx)
                if rv is None:
                    continue
                p_names, p_help = rv

                # is param multiple? then add the info to the p_help, wrapped in square
                # brackets at the end, as click does with all other extra info
                if getattr(param, 'multiple', False):
                    # check for extra info, i.e. a last chunk of the form " [...]"
                    idx = p_help.rstrip().rfind(' ')
                    if idx > -1:
                        extra_info_chunk = p_help[idx:].strip()
                        # is the last chunk a collection of extra information, whihc
                        # `click` wraps in square brackets)?
                        if extra_info_chunk[0] == '[' and extra_info_chunk[-1] == ']':
                            extra_info_chunk = extra_info_chunk[:-1] + "; "
                        else:
                            extra_info_chunk = ' ['  # create a "fake" last chunk
                            idx = None  # make first chunk the whole opt_help str

                        extra_info_chunk += 'multi-param: can be provided several times]'
                        p_help = p_help[idx:] + extra_info_chunk

                opts.append((p_names, p_help))

            # same as superclass, with slight modifications:
            if not opts:
                return

            # formatter.section handles indentation and paragraphs:
            with formatter.section('Options'):
                for opt, opt_help in opts:
                    formatter.write_paragraph()  # prints "\n"
                    # write_text handles indentation for us:
                    formatter.write_text(opt)
                    formatter.write_text(opt_help)

    class MyGroup(click.Group):
        """Class used for any click Group of this module"""

        def __init__(self, *arg, **kwargs):
            # configure default arguments:
            kwargs.setdefault('options_metavar', '')
            kwargs.setdefault('subcommand_metavar', "[command] [args]...")
            super().__init__(*arg, **clickutils._config_cmd_kwargs(**kwargs))

        def format_options(self, ctx, formatter):
            """Customize help formatting (print no options, only commands)"""
            # superclass (click.MultiCommand) code:
            # Command.format_options(self, ctx, formatter)  # <- ignore opt
            self.format_commands(ctx, formatter)

        def format_commands(self, ctx, formatter, parent_cmd_name=""):
            """Customize commands help formatting"""
            commands = []
            for cmd_name in list(self.commands):
                cmd = self.get_command(ctx, cmd_name)
                # What is this, the tool lied about a command.  Ignore it
                if cmd is None:
                    continue
                if cmd.hidden:
                    continue

                commands.append((cmd_name, cmd))

            if not commands:
                return

            # formatter.section handles indentation and paragraphs:
            with (formatter.section('Commands')
                    if not parent_cmd_name else formatter.indentation()):

                if not parent_cmd_name:
                    parent_cmd_name = ctx.command_path  # used below

                for cmd_name, cmd in commands:
                    formatter.write_paragraph()  # prints "\n"
                    formatter.write_text(cmd_name)

                    is_group = isinstance(cmd, click.Group)

                    if is_group:
                        formatter.write_text(cmd.get_short_help_str() +
                                             " (command group). Subcommands:")
                        cmd.format_commands(ctx, formatter,
                                            parent_cmd_name + " " + cmd_name)
                        continue

                    usage_pieces = [parent_cmd_name, cmd_name]
                    usage_pieces += list(cmd.collect_usage_pieces(ctx))
                    formatter.write_text('Usage: ' + " ".join(usage_pieces))
                    formatter.write_text(cmd.get_short_help_str())

        def command(self, *args, **kwargs):
            """Force to return my subclasses of command"""
            kwargs.setdefault('cls', clickutils.MyCommand)
            return super().command(*args, **kwargs)

        def group(self, *args, **kwargs):
            """Force to return my subclasses of group"""
            kwargs.setdefault('cls', clickutils.MyGroup)
            return super().group(*args, **kwargs)


@click.group(cls=clickutils.MyGroup)
def cli():
    """Stream2segment is a program to download, process, visualize or annotate
    massive amounts of event-based seismic waveform segments and their metadata.
    """
    pass


@cli.command(short_help='Create working example files with documentation to '
                        'start downloading and processing data')
@click.argument('outdir')
def init(outdir):
    """Create template files for launching download,
    processing and visualization. OUTDIR will be created if it does not exist
    """
    helpdict = OrderedDict([
        ("download.yaml",
         "Download configuration file (option -c of 's2s download')"),
        ("paramtable.py",
         "Processing python file for creating a parametric table (HDF, CSV). "
         "Option -p of 's2s process' and 's2s show'"),
        ("paramtable.yaml",
         "Processing configuration used in the associated Python file. "
         "Option -c of 's2s process' and 's2s show'"),
        ("save2fs.py",
         "Processing python file for saving waveform to filesystem. "
         "Option -p of 's2s process' and 's2s show'"),
        ("save2fs.yaml",
         "Processing configuration used in the associated Python file. "
         "Option -c of 's2s process' and 's2s show'"),
        ("jupyter.example.ipynb",
         "Jupyter notebook illustrating how to "
         "access downloaded data and run custom code. "
         "Run 'jupyter notebook jupyter.example.ipynb' for details "
         "(requires the installation of jupyter)"),
        ("jupyter.example.db",
         "Test database with few downloaded segments. Used in the "
         "associated jupyter notebook")
    ])
    # import here to improve slow click cli (at least when --help is invoked)
    # https://www.tomrochette.com/problems/2020/03/07
    from stream2segment import main

    try:
        copied_files = main.init(outdir, True, *helpdict)  # pass only helpdict keys
        if not copied_files:
            print("No file copied")
        else:
            print("%d file(s) copied in '%s':" % (len(copied_files), outdir))
            frmt = "- {:<%d} {}" % max(len(f) for f in helpdict.keys())
            for i, fcopied in enumerate(copied_files):
                if i in (0, 1, 5):
                    print("")
                bname = os.path.basename(fcopied)
                print(frmt.format(bname, helpdict.get(bname, "")))
            print("")
            sys.exit(0)
    except Exception as exc:  # pylint: disable=broad-except
        print('')
        print("error: %s" % str(exc))
    sys.exit(1)


# Short recap here (READ BEFORE EDITING OPTIONS BELOW):
# * option short name: any click option name starting with "-"
# * option long name: any click option name starting with "--"
#   IMPORTANT: Some YAML params accept different names (e.g., 'net', 'network'
#   or 'networks'). By convention, these name(s) must be provided here as
#   **long names**. Short names are not supposed to be used in the YAML and
#   can be safely modified here.
# * option default name: any click option name not starting with any "-"
#   (see http://click.pocoo.org/5/parameters/#parameter-names)
# * option help: the option help shown when issuing "--help" from the command
#   line.
#   IMPORTANT: Option help not provided here will be set as the docstring
#   of the same parameter in the YAML file ('download.yaml'), searching for the
#   Option **long name** or, if the Option has multiple long names, the option
#   **default name**. If no YAML parameter is found, or no Option default name
#   is provided, the Option help will be left empty.
# Reminder:
# * option flags should all have default=None which lets us know that the flag
#   is missing and use the corresponding yaml param values
# * Don't set required = True with eager=True in a click option, as it forces
#   that option to be always present, and thus raises if only --help is given
@clickutils.options_missing_help_from_yaml  # autofill options help. See function above
@cli.command(short_help='Download waveform data segments saving data into an '
                        'SQL database')
@click.option("-c", "--config",
              help="The path to the configuration file in yaml format "
                   "(https://learn.getgrav.org/advanced/yaml).",
              type=clickutils.ExistingPath, required=True)
@click.option('-d', '--dburl', is_eager=True)
@click.option('-es', '--eventws')
@click.option('-s', '--start', '--starttime', "starttime",
              type=inputvalidation.valid_date, metavar='DATE or DATETIME')
@click.option('-e', '--end', '--endtime', 'endtime',
              type=inputvalidation.valid_date, metavar='DATE or DATETIME', )
@click.option('-n', '--network', '--networks', '--net', 'network')
@click.option('-z', '--station', '--stations', '--sta', 'station')
@click.option('-l', '--location', '--locations', '--loc', 'location')
@click.option('-k', '--channel', '--channels', '--cha', 'channel')
@click.option('-msr', '--min-sample-rate', type=float)
@click.option('-ds', '--dataws', multiple=True)
@click.option('-t', '--traveltimes-model')
@click.option('-w', '--timespan', nargs=2, type=float)
@click.option('-u', '--update-metadata',
              type=click.Choice(['true', 'false', 'only']), default=None)
@click.option('-r1', '--retry-url-err', is_flag=True, default=None)
@click.option('-r2', '--retry-mseed-err', is_flag=True, default=None)
@click.option('-r3', '--retry-seg-not-found', is_flag=True, default=None)
@click.option('-r4', '--retry-client-err', is_flag=True, default=None)
@click.option('-r5', '--retry-server-err', is_flag=True, default=None)
@click.option('-r6', '--retry-timespan-err', is_flag=True, default=None)
@click.option('-i', '--inventory', is_flag=True, default=None)
@click.option('-minlat', '--minlatitude', type=float,
              help=clickutils.EQA + " Limit to events with a latitude larger "
                                    "than or equal to the specified minimum")
@click.option('-maxlat', '--maxlatitude', type=float,
              help=clickutils.EQA + " Limit to events with a latitude smaller "
                                    "than or equal to the specified maximum")
@click.option('-minlon', '--minlongitude', type=float,
              help=clickutils.EQA + " Limit to events with a longitude larger "
                                    "than or equal to the specified minimum")
@click.option('-maxlon', '--maxlongitude', type=float,
              help=clickutils.EQA + " Limit to events with a longitude smaller "
                                    "than or equal to the specified maximum")
@click.option('--mindepth', type=float,
              help=clickutils.EQA + " Limit to events with depth more than the "
                                    "specified minimum")
@click.option('--maxdepth', type=float,
              help=clickutils.EQA + " Limit to events with depth less than the "
                                    "specified maximum")
@click.option('-minmag', '--minmagnitude', type=float,
              help=clickutils.EQA + " Limit to events with a magnitude larger "
                                    "than the specified minimum")
@click.option('-maxmag', '--maxmagnitude', type=float,
              help=clickutils.EQA + " Limit to events with a magnitude smaller "
                                    "than the specified maximum")
def download(config, dburl, eventws, starttime, endtime, network,  # noqa
             station, location, channel, min_sample_rate,  # noqa
             dataws, traveltimes_model, timespan,  # noqa
             update_metadata, retry_url_err, retry_mseed_err,  # noqa
             retry_seg_not_found, retry_client_err,  # noqa
             retry_server_err, retry_timespan_err, inventory,  # noqa
             minlatitude, maxlatitude, minlongitude,  # noqa
             maxlongitude, mindepth, maxdepth, minmagnitude,  # noqa
             maxmagnitude):  # noqa
    """Download waveform data segments and their metadata on a SQL database.
    NOTE: The config file (-c option, see below) is the only required option.
    All other options, if provided, will overwrite the corresponding value in
    the config file
    """
    _locals = dict(locals())  # MUST BE THE FIRST STATEMENT

    # import here to improve slow click cli (at least when --help is invoked)
    # https://www.tomrochette.com/problems/2020/03/07
    from stream2segment import main

    # REMEMBER: NO LOCAL VARIABLES OTHERWISE WE MESS UP THE CONFIG OVERRIDES
    # ARGUMENTS
    try:
        overrides = {k: v for k, v in _locals.items()
                     if v not in ((), {}, None) and k != 'config'}
        with warnings.catch_warnings():  # capture (ignore) warnings
            warnings.simplefilter("ignore")
            ret = main.download(config, log2file=True, verbose=True,
                                **overrides)
    except inputvalidation.BadParam as err:
        print(err)
        ret = 2
    except:  # @IgnorePep8 pylint: disable=bare-except
        # do not print traceback, as we already did it by configuring loggers
        ret = 3
    # ret might return 0 or 1 the latter in case of QuitDownload, but tests
    # expect a non-zero value thus we skip this feature for the moment
    sys.exit(0 if ret <= 1 else ret)


@cli.command(short_help="Process downloaded waveform data segments by "
                        "executing custom code on a user-defined selection of "
                        "segments")
@click.option('-d', '--dburl', **clickutils.DBURL_OR_YAML_ATTRS)
@click.option("-c", "--config",
              help="The path to the configuration file in yaml format "
                   "(https://learn.getgrav.org/advanced/yaml).",
              type=clickutils.ExistingPath, required=True)
@click.option("-p", "--pyfile",
              help="The path to the Python file where the user-defined "
                   "processing function is implemented. The function will be "
                   "called iteratively on each segment selected in the config "
                   "file",
              type=clickutils.ExistingPath, required=True)
@click.option("-f", "--funcname",
              help="The name of the user-defined processing function in the "
                   "given python file. Defaults to '%s' when "
                   "missing" % inputvalidation.valid_default_processing_funcname())
@click.option("-a", "--append", is_flag=True, default=False,
              help="Append results to the output file (this flag is ignored if "
                   "no output file is provided. The output file will be "
                   "scanned to detect already processed segments and skip them: "
                   "for huge files, this might be time-consuming). When "
                   "missing, it defaults to false, meaning that an output file, "
                   "if provided, will be overwritten if it exists")
@click.option("--no-prompt", is_flag=True, default=False,
              help="Do not prompt the user when attempting to overwrite an "
                   "existing output file. This flag is false by default, i.e. "
                   "the user will be asked for confirmation before overwriting "
                   "an existing file. This flag is ignored if no output file is "
                   "provided, or the 'append' flag is given")
@click.option("-mp", "--multi-process", is_flag=True,
              default=None,  # default=None let us know when arg is missing or not
              help="Use parallel sub-processes to speed up the execution. "
                   "When missing, it defaults to false")
@click.option("-np", "--num-processes", type=int,
              default=None,  # default=None let us know when arg is missing or not
              help="The number of sub-processes. If missing, it is set as the "
                   "the number of CPUs in the system. This option is ignored "
                   "if --multi-process is not given")
@click.argument('outfile', required=False)
def process(dburl, config, pyfile, funcname, append, no_prompt,
            multi_process, num_processes,
            outfile):
    """Process downloaded waveform data segments via a custom python file and a
    configuration file.

    [OUTFILE] (optional): the path of the tabular file (CSV or HDF) where the
    output of the user-defined processing function will be written to
    (generally, one row per processed segment. If this argument is missing,
    then any output of the processing function will be ignored). The tabular
    format will be inferred from the file extension provided (.h5, .hdf5, .hdf
    for HDF files, anything else: CSV). All information, errors or warnings
    will be logged to the file [OUTFILE].[now].log (where [now] denotes the
    execution date and time. If no output file is provided, [OUTFILE] will be
    replaced with [pyfile])
    """
    _locals = dict(locals())  # MUST BE THE FIRST STATEMENT

    # import here to improve slow click cli (at least when --help is invoked)
    # https://www.tomrochette.com/problems/2020/03/07
    from stream2segment import main

    # REMEMBER: NO LOCAL VARIABLES OTHERWISE WE MESS UP THE CONFIG OVERRIDES
    # ARGUMENTS
    try:
        if not append and outfile and os.path.isfile(outfile) \
                and not no_prompt and \
                not click.confirm("'%s' already exists in '%s'.\nOverwrite?" %
                                  (os.path.basename(os.path.abspath(outfile)),
                                   os.path.dirname(os.path.abspath(outfile)))):
            ret = 1
        else:
            # override config values for multi_process and num_processes
            overrides = {k: v for k, v in _locals.items()
                         if v not in ((), {}, None) and k in
                         ('multi_process', 'num_processes')}
            if overrides:
                # if given, put these into 'advanced_settings' sub-dict. Note
                # that nested dict will be merged with the values of the config
                overrides = {'advanced_settings': overrides}
            with warnings.catch_warnings():  # capture (ignore) warnings
                warnings.simplefilter("ignore")
                ret = main.process(dburl, pyfile, funcname, config, outfile,
                                   log2file=True, verbose=True, append=append,
                                   **overrides)
    except inputvalidation.BadParam as err:
        print(err)
        ret = 2  # exit with 1 as normal python exceptions
    except:  # @IgnorePep8 pylint: disable=bare-except
        # do not print traceback, as we already did it by configuring loggers
        ret = 3
    sys.exit(ret)


@cli.command(short_help='Show waveform plots and metadata in the browser')
@click.option('-d', '--dburl', **clickutils.DBURL_OR_YAML_ATTRS)
@click.option("-c", "--configfile",
              help="Optional: The path to the configuration file in yaml "
                   "format (https://learn.getgrav.org/advanced/yaml).",
              type=clickutils.ExistingPath, required=False)
@click.option("-p", "--pyfile", help="Optional: The path to the Python file "
                                     "with the plot functions implemented",
              type=clickutils.ExistingPath, required=False)
def show(dburl, configfile, pyfile):
    """Show waveform plots and metadata in the browser,
    customizable with user-defined configuration and custom Plots
    """
    # import here to improve slow click cli (at least when --help is invoked)
    # https://www.tomrochette.com/problems/2020/03/07
    from stream2segment import main

    try:
        ret = 0
        with warnings.catch_warnings():  # capture (ignore) warnings
            warnings.simplefilter("ignore")
            main.show(dburl, pyfile, configfile)
    except inputvalidation.BadParam as err:
        print(err)
        ret = 2  # exit with 1 as normal python exceptions
    except:
        ret = 3
    sys.exit(ret)


@cli.group(short_help="Downloaded data analysis and inspection")
def dl():  # pylint: disable=missing-docstring
    pass


@dl.command(short_help='Produce download summary statistics in either plain '
                       'text or html format')
@click.option('-d', '--dburl', **clickutils.DBURL_OR_YAML_ATTRS)
@click.option('-did', '--download-id', multiple=True, type=int,
              help="Limit the download statistics to a specified set of "
                   "download ids (integers). when missing, all downloads are "
                   "shown. this option can be given multiple times: "
                   "... -did 1 --download_id 2 ...")
@click.option('-g', '--maxgap-threshold', type=float, default=0.5,
              help="Optional: set the threshold (in number of samples) "
                   "to identify segments with gaps/overlaps. Defaults to 0.5, "
                   "meaning that segments whose maximum gap is greater than "
                   "half a sample will be identified has having gaps, and "
                   "segments whose maximum gap is lower than minus half a "
                   "sample will be identified has having overlaps")
@click.option('-htm', '--html', is_flag=True,
              help="Generate an interactive dynamic web page where the "
                   "download info is visualized on a map, with statistics "
                   "on a per-station and data-center basis. A working internet "
                   "connection is needed to properly view the page")
@click.argument("outfile", required=False, type=click.Path(file_okay=True,
                                                           dir_okay=False,
                                                           writable=True,
                                                           readable=True))
def stats(dburl, download_id, maxgap_threshold, html, outfile):
    """Produce download summary statistics either in plain text or html format.

    [OUTFILE] (optional): the output file where the information will be saved
    to. If missing, results will be printed to screen or opened in a web
    browser (depending on the option '--html')
    """
    # import here to improve slow click cli (at least when --help is invoked)
    # https://www.tomrochette.com/problems/2020/03/07
    from stream2segment import main

    print('Fetching data, please wait (this might take a while depending on '
          'the db size and connection)')
    try:
        with warnings.catch_warnings():  # capture (ignore) warnings
            warnings.simplefilter("ignore")
            main.dstats(dburl, download_id or None, maxgap_threshold,
                        html, outfile)
        if outfile is not None:
            print("download statistics written to '%s'" % outfile)
        sys.exit(0)
    except inputvalidation.BadParam as err:
        print(err)
        sys.exit(1)  # exit with 1 as normal python exceptions


@dl.command(short_help="Return download information for inspection")
@click.option('-d', '--dburl', **clickutils.DBURL_OR_YAML_ATTRS)
@click.option('-did', '--download-id', multiple=True, type=int,
              help="Limit the download statistics to a specified set of "
                   "download ids (integers) when missing, all downloads are "
                   "shown. This option can be given multiple times: "
                   "... -did 1 --download_id 2 ...")
@click.option('-c', '--config', is_flag=True, default=None,
              help="Returns only the config used (in YAML syntax) of the "
                   "chosen download(s)")
@click.option('-l', '--log', is_flag=True, default=None,
              help="Returns only the log messages of the chosen download(s)")
@click.argument("outfile", required=False, type=click.Path(file_okay=True,
                                                           dir_okay=False,
                                                           writable=True,
                                                           readable=True))
def report(dburl, download_id, config, log, outfile):
    """Return download information.

    [OUTFILE] (optional): the output file where the information will be saved
    to. If missing, results will be printed to screen or opened in a web
    browser (depending on the option '--html')
    """
    # import here to improve slow click cli (at least when --help is invoked)
    # https://www.tomrochette.com/problems/2020/03/07
    from stream2segment import main

    print('Fetching data, please wait (this might take a while depending on the '
          'db size and connection)')
    try:
        # this is hacky but in case we want to restore the html
        # argument ...
        html = False
        with warnings.catch_warnings():  # capture (ignore) warnings
            warnings.simplefilter("ignore")
            main.dreport(dburl, download_id or None,
                         bool(config), bool(log), html, outfile)
        if outfile is not None:
            print("download report written to '%s'" % outfile)
        sys.exit(0)
    except inputvalidation.BadParam as err:
        print(err)
        sys.exit(1)  # exit with 1 as normal python exceptions


@cli.group(short_help="Database management")
def db():  # pylint: disable=missing-docstring
    pass


@db.command(short_help="Drop (delete) download executions and all associated "
                       "segments")
@click.option('-d', '--dburl', **clickutils.DBURL_OR_YAML_ATTRS)
@click.option('-did', '--download-id', multiple=True, type=int, required=True,
              help="The id(s) of the download execution(s) to be deleted. "
                   "This option can be given multiple "
                   "times: ... -did 1 --download_id 2 ...")
def drop(dburl, download_id):
    """Drop (deletes) download executions. WARNING: this command deletes also
    all segments, stations and channels downloaded with the given download
    execution
    """
    # import here to improve slow click cli (at least when --help is invoked)
    # https://www.tomrochette.com/problems/2020/03/07
    from stream2segment import main

    print('Fetching data, please wait (this might take a while depending on '
          'the db size and connection)')
    try:
        with warnings.catch_warnings():  # capture (ignore) warnings
            warnings.simplefilter("ignore")
            ret = main.ddrop(dburl, download_id, True)
        if ret is None:
            sys.exit(1)
        elif not ret:
            print('Nothing to delete')
        for key, val in ret.items():
            msg = 'Download id=%d: ' % key
            if isinstance(val, Exception):
                msg += "FAILED (%s)" % str(val)
            else:
                msg += "DELETED (%d associated segments deleted)" % val
            print(msg)
        sys.exit(0)
    except inputvalidation.BadParam as err:
        print(err)
        sys.exit(1)  # exit with 1 as normal python exceptions


@db.command(short_help="Add/rename/delete class labels from the database")
@click.option('-d', '--dburl', **clickutils.DBURL_OR_YAML_ATTRS)
@click.option('--add', multiple=True, nargs=2, type=str, required=False,
              help="Add a new class label: `--add label description`. You can "
                   "provide this arguments multiple times to add several "
                   "labels")
@click.option('--rename', multiple=True, nargs=3, type=str, required=False,
              help="Rename a class label: "
                   "`--rename old_label new_label new_description`. Set "
                   "new_description to \"\" or '' to rename the label only and "
                   "keep the old description. You can provide this argument "
                   "multiple times to rename several labels")
@click.option('--delete', multiple=True, type=str, required=False,
              help="Delete a new class label. Provide a single value (label to"
                   "be removed). You can provide this argument multiple times "
                   "to delete several labels. Note: this will also remove all "
                   "mappings (class labellings) between segments and their "
                   "associated label, if present")
@click.option("--no-prompt", is_flag=True, default=False,
              help="Do not prompt the user when attempting to "
                   "perform an operation")
def classlabel(dburl, add, rename, delete, no_prompt):
    """Add/Rename/delete class labels from the database. A class label is
    composed of a label name (e.g., LowS2N) and a short description (e.g.,
    "Segment has a low signal-to-noise ratio") and denote any user-defined
    characteristic that you want to assign to certain segments either manually
    in the GUI, or programmatically in the processing module or your code.
    Class labels can then be used for e.g., supervised classification problems,
    or tp perform custom selection on specific segments before processing.
    """
    # import here to improve slow click cli (at least when --help is invoked)
    # https://www.tomrochette.com/problems/2020/03/07
    from stream2segment.process.db import (configure_classlabels,
                                           get_classlabels)
    from stream2segment.main import input  # <- for mocking in testing

    add_arg, rename_arg, delete_arg = {}, {}, []
    try:
        if add:
            add_arg = {_[0]: _[1] for _ in add}
        if rename:
            rename_arg = {_[0]: (_[1], _[2] or None) for _ in rename}
        if delete:
            delete_arg = list(delete)

        if not no_prompt:
            msg = "Attempting to:"
            if add_arg:
                msg += '\nAdd %d class label(s)' % len(add_arg)
            if rename_arg:
                msg += '\nRename %d class label(s)' % len(rename_arg)
            if rename_arg:
                msg += '\nDelete %d class label(s)' % len(delete_arg)
            msg += '\nContinue (y/n)?'
            if input(msg) != 'y':
                sys.exit(1)

        session = inputvalidation.validate_param("dburl", dburl,
                                                 inputvalidation.valid_session,
                                                 for_process=False, scoped=False)
        configure_classlabels(session, add=add_arg, rename=rename_arg,
                              delete=delete_arg)
        print('Done. Current class labels on the database:')
        clabels = get_classlabels(session, include_counts=False)
        if not clabels:
            print('None')
        else:
            for clbl in clabels:
                print("%s (%s)" % (clbl['label'], clbl['description']))
        sys.exit(0)
    except inputvalidation.BadParam as err:
        print(err)
        sys.exit(1)  # exit with 1 as normal python exceptions


# Old click Group (not used anymore):

# @cli.group(short_help="Program utilities")
# def utils():  # noqa
#     pass


if __name__ == '__main__':
    cli()  # noqa
