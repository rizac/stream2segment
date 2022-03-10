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

# (http://python-future.org/imports.html#explicit-imports):
from builtins import (bytes, dict, int, open, str, super, input)

import sys
import os
import warnings
from collections import OrderedDict, defaultdict
from contextlib import contextmanager

import click

from stream2segment.resources import get_templates_fpath
from stream2segment.io.inputvalidation import BadParam
from stream2segment.io import yaml_load


class clickutils(object):  # noqa
    """Container for all `click` related stuff to be used here"""

    @staticmethod
    def valid_dburl_or_download_yamlpath(value, param_name='dburl'):
        """Return the database path from 'value': 'value' can be a file (in that
        case is assumed to be a yaml file with the `param_name` key in it, which
        must denote a db path) or the database path otherwise
        """
        if not isinstance(value, str):
            raise ValueError('Please provide a string')
        if os.path.isfile(value):
            try:
                yaml_dict = yaml_load(value)
            except Exception:
                raise ValueError('file exists but can not be read as YAML')
            try:
                return yaml_dict[param_name]
            except KeyError:
                raise ValueError('%d not found in YAML file %s' % (param_name, value))
        return value

    # shorthand string for event-related download params:
    EQA = "(event search parameter)"
    # Keyword attributes fot Options that accept a db URL also in form of the
    # file path to a download config (with the db url in it):
    DBURL_OR_YAML_ATTRS = dict(
        type=lambda val: clickutils.valid_dburl_or_download_yamlpath(val),
        metavar='TEXT or PATH',
        help=("Database URL where data has been saved. It can also be the path of a "
              "YAML file with the property 'dburl' (e.g., the config file used for "
              "downloading). WARNING: if the URL contains passwords it is safer to use "
              "a file instead of typing the URL on the terminal"),
        required=True
    )
    # custom type for Options accepting an existing File:
    ExistingPath = click.Path(exists=True, file_okay=True, dir_okay=False,
                              writable=False, readable=True)

    @classmethod
    def fill_missing_help_from_yaml_download_file(cls, command, *args, **kwargs):
        """Decorator to the `download`command to set missing options help
        from the relative YAML parameter, if found in the YAML file
        "download.yaml"
        """
        cfg_doc = cls.yaml_load_doc(get_templates_fpath("download.yaml"))
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

    @classmethod
    def yaml_load_doc(cls, filepath, varname=None, preserve_newlines=False):
        """Return the documentation from a YAML file. The returned object is
        the docstring of the given variable name (if `varname` is set),
        or a dict[str, str] (defaultdict("")) of all variables found, mapped to
        their docstring (a variable documentation is made up of all
        consecutive commented lines -  with *no* leading spaces - placed immediately
        before the variable). Only top-level variables can be parsed, nested ones
        are skipped.

        :param filepath: The YAML file to read the doc from
        :param varname: str or None (the default). Return the doc for this specific
            YAML variable. if None, returns a `defaultdict` with all top-level
            variables found.
        :param preserve_newlines: boolean. Whether to preserve newlines in comment
            or not. If False (the default), each variable comment is returned as a
            single line, concatenating parsed lines with a space
        """
        comments = []
        # reg_yaml_var = re.compile("^([^:]+):\\s.*")
        # reg_comment = re.compile("^#+(.*)")
        ret = defaultdict(str) if varname is None else ''
        isbytes = None
        with open(filepath, 'r') as stream:
            while True:
                line = stream.readline()
                # from the docs (https://docs.python.org/3/tutorial/inputoutput.html): if
                # f.readline() returns an empty string, the end of the file has been
                # reached, while a blank line is represented by '\n'
                if not line:
                    break
                if isbytes is None:
                    isbytes = isinstance(line, bytes)
                # is line a comment? do not use regexp, it's slower
                # m = reg_comment.match(line)
                if line.startswith('#'):
                    # the line is a comment, add the comment text.
                    # Note that the line does not include last newline, if present
                    comments.append(line[1:].strip())
                else:
                    # the line is not a comment line. Do we have parsed comments?
                    if comments:
                        # use string search and not regexp because faster:
                        idx = line.find(': ')
                        if idx == -1:
                            idx = line.find(':\n')
                        var_name = None if idx < 1 else line[:idx]
                        # We have parsed comments. Is the line a YAML parameter?
                        # m = reg_yaml_var.match(line)
                        # if m and m.groups():
                        if var_name:
                            # the line is a yaml variable, it's name is
                            # m.groups()[0]. Map the variable to its comment
                            # var_name = m.groups()[0]
                            join_char = "\n" if preserve_newlines else " "
                            comment = join_char.join(comments)
                            docstring = comment
                            if isbytes:
                                docstring = comment.decode('utf8')
                            if varname is None:
                                ret[var_name] = docstring
                            elif varname == var_name:
                                return docstring
                    # In any case, if not comment, reset comments:
                    comments = []
        return ret

    @staticmethod
    def _config_cmd_kwargs(**kwargs):
        """Shared default configurations settings for new Commands and Groups below"""
        # increase width of help on terminal (default ~= 80):
        context_settings = dict(max_content_width=85)
        kwargs.setdefault('context_settings', context_settings)
        kwargs.setdefault('options_metavar', '[options]')
        return kwargs

    class MyCommand(click.Command):
        """Class used for any click Command in this module"""
        def __init__(self, *arg, **kwargs):
            """Just (re)configure some default `kwargs`"""
            # configure default arguments:
            super().__init__(*arg, **clickutils._config_cmd_kwargs(**kwargs))

        def format_options(self, ctx, formatter):
            """Invoked by :meth:`self.format_help`, reformat here how options are printed
            for a :class:`click.Command`: 1. avoid the two columns layout, just print
            command name(s) and then help, and 2. mention when an Option is "multiple"
            (accepted several times) in the dedicated "extra information" chunk (within
            square brackets, e.g. "[required]") at the end of the help
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
                        # is the last chunk a collection of extra information, which
                        # `click` wraps in square brackets)?
                        if extra_info_chunk[0] == '[' and extra_info_chunk[-1] == ']':
                            extra_info_chunk = extra_info_chunk[:-1] + "; "
                        else:
                            extra_info_chunk = ' ['  # create a "fake" last chunk
                            idx = None  # make first chunk the whole opt_help str

                        extra_info_chunk += 'multi-param: can be provided several times]'
                        p_help = p_help[:idx] + extra_info_chunk

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
            """Just (re)configure some default `kwargs`"""
            # configure default arguments:
            kwargs.setdefault('options_metavar', '')
            kwargs.setdefault('subcommand_metavar', "[command] [args]...")
            super().__init__(*arg, **clickutils._config_cmd_kwargs(**kwargs))

        def format_options(self, ctx, formatter):
            """Invoked by :meth:`self.format_help`, reformat here how Options or Commands
            are printed for a :class:`click.Group`: 1. Skip printing the (only) Option
            "--help": it's too prominent and not really useful, and 2: customize
            Commands help calling :meth:`self.format_commands` (see method for details)
            """
            # superclass (click.MultiCommand) code:
            # Command.format_options(self, ctx, formatter)  # <- ignore opt
            self.format_commands(ctx, formatter)

        def format_commands(self, ctx, formatter, parent_cmd_name=""):
            """Invoked by :meth:`self.format_help`, reformat here how Commands are
            printed by 1: listing all commands recursively (not only direct children),
            2: using different layouts and descriptions for Commands vs. Groups, and 3:
            providing for each subcommand a short 'Usage' string, as the first line that
            would appear by navigating into the subcommand and typing "--help"
            """
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
         "Download configuration settings (option -c of 's2s download')"),
        ("paramtable.py",
         "Processing module for creating a parametric table (HDF, CSV). "
         "Option -p of 's2s process' and 's2s show'"),
        ("paramtable.yaml",
         "Processing configuration settings used in the associated module. "
         "Option -c of 's2s process' and 's2s show'"),
        # ("save2fs.py",
        #  "Processing python file for saving waveform to filesystem. "
        #  "Option -p of 's2s process' and 's2s show'"),
        # ("save2fs.yaml",
        #  "Processing configuration used in the associated Python file. "
        #  "Option -c of 's2s process' and 's2s show'"),
        ("Using-Stream2segment-in-your-Python-code.ipynb",
         "Jupyter notebook illustrating how to work with downloaded data "
         "(requires the installation of jupyter)"),
        ("example.db.sqlite",
         "Example database used in the associated notebook")
    ])
    # import here to improve slow click cli (at least when --help is invoked)
    # https://www.tomrochette.com/problems/2020/03/07

    try:
        copied_files = copy_example_files(outdir, True, *helpdict)  # pass only helpdict keys
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


def copy_example_files(outpath, prompt=True, *filenames):
    """Initialize an output directory writing therein the given template files

    :param prompt: bool (default: True) telling if a prompt message (Python
        `input` function) should be issued to warn the user when overwriting
        files. The user should return a string or integer where '1' means
        'overwrite all files', '2' means 'overwrite only non-existing', and any
        other value will return without copying.
    """
    import jinja2, shutil
    from stream2segment.resources import get_templates_fpaths
    from stream2segment.resources.templates import DOCVARS

    if not os.path.isdir(outpath):
        os.makedirs(outpath)
        if not os.path.isdir(outpath):
            raise Exception("Unable to create '%s'" % outpath)

    if prompt:
        existing_files = [f for f in filenames
                          if os.path.isfile(os.path.join(outpath, f))]
        non_existing_files = [f for f in filenames if f not in existing_files]
        if existing_files:
            suffix = ("Type:\n1: overwrite all files\n2: write only non-existing\n"
                      "0 or any other value: do nothing (exit)")
            msg = ("The following file(s) "
                   "already exist on '%s':\n%s"
                   "\n\n%s") % (outpath, "\n".join([_ for _ in existing_files]), suffix)
            val = input(msg)
            try:
                val = int(val)
                if val == 2:
                    if not non_existing_files:
                        raise ValueError()  # fall back to "exit" case
                    else:
                        filenames = non_existing_files
                elif val != 1:
                    raise ValueError()  # fall back to "exit" case
            except ValueError:
                return []

    srcfilepaths = get_templates_fpaths(*filenames)
    if srcfilepaths:
        basedir = os.path.dirname(srcfilepaths[0])
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(basedir),
                                 keep_trailing_newline=True)
        copied_files = []
        for srcfilepath in srcfilepaths:
            filename = os.path.basename(srcfilepath)
            outfilepath = os.path.join(outpath, filename)
            if os.path.splitext(filename)[1].lower() in ('.yaml', '.py'):
                env.get_template(filename).stream(DOCVARS).dump(outfilepath)
            else:
                shutil.copyfile(srcfilepath, outfilepath)
            copied_files.append(outfilepath)
    return copied_files


# Short recap here (READ IF YOU PLAN TO EDIT OPTIONS BELOW, SKIP OTHERWISE):
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
@clickutils.fill_missing_help_from_yaml_download_file  # autofill options help. See above
@cli.command(short_help='Download waveform data segments saving data into an '
                        'SQL database')
@click.option("-c", "--config",
              help="The path to the configuration file in yaml format "
                   "(https://learn.getgrav.org/advanced/yaml).",
              type=clickutils.ExistingPath, required=True)
@click.option('-d', '--dburl', is_eager=True)
@click.option('-es', '--eventws')
@click.option('-s', '--start', '--starttime', "starttime", metavar='DATE or DATETIME')
@click.option('-e', '--end', '--endtime', 'endtime', metavar='DATE or DATETIME', )
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
@click.option('--print-config-only', is_flag=True, default=False,
              help="Do not execute any download, but print the final YAML created by "
                   "merging the provided yaml file with the command line parameters "
                   "(default False when missing)")
def download(config, dburl, eventws, starttime, endtime, network,  # noqa
             station, location, channel, min_sample_rate,  # noqa
             dataws, traveltimes_model, timespan,  # noqa
             update_metadata, retry_url_err, retry_mseed_err,  # noqa
             retry_seg_not_found, retry_client_err,  # noqa
             retry_server_err, retry_timespan_err, inventory,  # noqa
             minlatitude, maxlatitude, minlongitude,  # noqa
             maxlongitude, mindepth, maxdepth, minmagnitude,  # noqa
             maxmagnitude, print_config_only):  # noqa
    """Download waveform data segments and their metadata on a SQL database.
    NOTE: The config file (-c option, see below) is the only required option.
    All other options, if provided, will overwrite the corresponding value in
    the config file (when providing negative numbers through the command line,
    you might need to escape the minus sign)
    """
    _locals = dict(locals())  # <- THIS MUST BE THE FIRST STATEMENT OF THIS FUNCTION!

    # import in function body to speed up the main module import:
    from stream2segment.download.main import download as _download

    try:
        overrides = {k: list(v) if type(v) == tuple else v for k, v in _locals.items()
                     if v not in ((), {}, None) and k not in
                     ('config', 'print_config_only')}
        with warnings.catch_warnings():  # capture (ignore) warnings
            warnings.simplefilter("ignore")
            ret = _download(config, log2file=True, verbose=True,
                            print_config_only=print_config_only,
                            **overrides)
    except BadParam as err:
        _print_badparam_and_exit(err)
    except:  # @IgnorePep8 pylint: disable=bare-except
        # do not print traceback, as we already did it by configuring loggers
        ret = 3  # 1 is reserved for FailedDownload
    sys.exit(ret)


@contextmanager
def _print_badparam_and_exit(bad_param_exception):
    print(bad_param_exception, file=sys.stderr)
    sys.exit(2)  # 1 is reserved for other stuff (e.g. FailedDownload)


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
                   "given python file. Defaults to 'main' when missing")
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
@click.option("--logfile", default=None,
              help="The path of the log file to record processing information such as "
                   "skipped segments errors. When missing, it is set as the output file "
                   "path suffixed with a timestamp to avoid conflicts and extension "
                   "'.log' (if no output file is given, the Python file is used as base "
                   "path). Provide 'skip' (with no quotes) to disable logging")
@click.argument('outfile', required=False)
def process(dburl, config, pyfile, funcname, append, no_prompt, multi_process,
            num_processes, logfile, outfile):
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
    _locals = dict(locals()) # <- THIS MUST BE THE FIRST STATEMENT OF THIS FUNCTION!

    # import in function body to speed up the main module import:
    from stream2segment.process.main import (process as _process, load_p_config,
                                             validate_param, valid_pyfile)

    try:
        if not append and outfile and os.path.isfile(outfile) \
                and not no_prompt and \
                not click.confirm("'%s' already exists in '%s'.\nOverwrite?" %
                                  (os.path.basename(os.path.abspath(outfile)),
                                   os.path.dirname(os.path.abspath(outfile)))):
            ret = 1
        else:
            overrides = {}
            if multi_process:
                multi_process = num_processes if num_processes else True
                # override config advanced_settings. Note that dict will be merged,
                # so other advanced settings will not be deleted:
                overrides = {'advanced_settings': {'multi_process': multi_process}}

            with warnings.catch_warnings():  # capture (ignore) warnings
                warnings.simplefilter("ignore")
                _, seg_sel, m_p, chunksize, w_options = load_p_config(config, **overrides)
                if funcname:
                    pyfile += '::' + funcname
                # Raise BadParam with the right param name (this will be done also in
                # `_process`, with different param name though):
                validate_param('pyfile', pyfile, valid_pyfile)
                # convert logfile argument:
                if logfile is None:
                    logfile = True
                elif logfile == 'skip':
                    logfile = False
                # execute now:
                ret = _process(pyfile, dburl, seg_sel, config, outfile, append=append,
                               writer_options=w_options, logfile=logfile, verbose=True,
                               multi_process=m_p, chunksize=chunksize)
    except BadParam as err:
        _print_badparam_and_exit(err)
    except:  # noqa
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

    # import in function body to speed up the main module import:
    from stream2segment.process.gui.main import show_gui

    try:
        ret = 0
        with warnings.catch_warnings():  # capture (ignore) warnings
            warnings.simplefilter("ignore")
            show_gui(dburl, pyfile, configfile)
    except BadParam as err:
        _print_badparam_and_exit(err)
    except:
        ret = 3
    sys.exit(ret)


@cli.group(short_help="Downloaded data analysis and inspection")
def dl():  # pylint: disable=missing-docstring
    pass


@dl.command(short_help='Produce download statistics in either plain text or html format',
            context_settings={'ignore_unknown_options': True})
@click.option('-d', '--dburl', **clickutils.DBURL_OR_YAML_ATTRS)
@click.option('-did', '--download-id', multiple=True, type=int,
              help="The unique id (integer) of the download execution(s) to select, "
                   "which can be supplied instead of or together with the list of "
                   "download indices")
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
@click.option("-o", "--outfile", required=False, type=click.Path(file_okay=True,
                                                                 dir_okay=False,
                                                                 writable=True,
                                                                 readable=True),
              help="The optional output file where the information will be saved to. "
                   "If missing, results will be printed to screen or opened in a web "
                   "browser, depending on the option '--html'")
@click.argument("download_indices", required=False, nargs=-1)
def stats(dburl, download_id, maxgap_threshold, html, outfile, download_indices):
    """Produce download statistics either in plain text or html format.

    [DOWNLOAD_INDICES] (optional): The space-separated indices of the download executions
    to inspect, where 0 indicates the first/oldest. To start counting from the end use a
    negative index, e.g., -1 for the last execution, -2 for the next-to-last.
    When no download index or id is provided, all download executions will be shown by
    default
    """
    # import in function body to speed up the main module import:
    from stream2segment.download.db.inspection.main import stats as _stats

    _print_waitmsg_while_fetching_data()

    try:
        with warnings.catch_warnings():  # capture (ignore) warnings
            warnings.simplefilter("ignore")
            _stats(dburl, download_indices or None, download_id or None,
                   maxgap_threshold, html, outfile)
        if outfile is not None:
            print("download statistics written to '%s'" % outfile, file=sys.stderr)
        sys.exit(0)
    except BadParam as err:
        _print_badparam_and_exit(err)


@dl.command(short_help="Show short summary of the given download execution(s)",
            context_settings={'ignore_unknown_options': True})
@click.option('-d', '--dburl', **clickutils.DBURL_OR_YAML_ATTRS)
@click.option('-did', '--download-id', multiple=True, type=int,
              help="The unique id (integer) of the download execution(s) to select, "
                   "which can be supplied instead of or together with the list of "
                   "download indices")
@click.argument("download_indices", required=False, nargs=-1)
def summary(dburl, download_id, download_indices):
    """Return a summary of the download execution

    [DOWNLOAD_INDICES] (optional): The space-separated indices of the download executions
    to inspect, where 0 indicates the first/oldest. To start counting from the end use a
    negative index, e.g., -1 for the last execution, -2 for the next-to-last.
    When no download index or id is provided, all download executions will be shown by
    default
    """
    # import in function body to speed up the main module import:
    from stream2segment.download.db.inspection.main import summary as _summary

    _print_waitmsg_while_fetching_data()

    try:
        with warnings.catch_warnings():  # capture (ignore) warnings
            warnings.simplefilter("ignore")
            _summary(dburl, download_indices or None, download_id or None)
        sys.exit(0)
    except BadParam as err:
        _print_badparam_and_exit(err)


@dl.command(short_help="Show the log file content of the given download execution(s)",
            context_settings={'ignore_unknown_options': True})
@click.option('-d', '--dburl', **clickutils.DBURL_OR_YAML_ATTRS)
@click.option('-did', '--download-id', multiple=True, type=int,
              help="The unique id (integer) of the download execution(s) to select, "
                   "which can be supplied instead of or together with the list of "
                   "download indices")
@click.argument("download_indices", required=False, nargs=-1)
def log(dburl, download_id, download_indices):
    """Return the log file(s) content with detailed information of the download execution

    [DOWNLOAD_INDICES] (optional): The space-separated indices of the download executions
    to inspect, where 0 indicates the first/oldest. To start counting from the end use a
    negative index, e.g., -1 for the last execution, -2 for the next-to-last.
    When no download index or id is provided, the last download execution (index -1) will
    be shown by default
    """
    # import in function body to speed up the main module import:
    from stream2segment.download.db.inspection.main import log as _log

    _print_waitmsg_while_fetching_data()

    try:
        with warnings.catch_warnings():  # capture (ignore) warnings
            warnings.simplefilter("ignore")
            if not download_id and not download_indices:
                download_indices = [-1]
            _log(dburl, download_indices or None, download_id or None, None)
        sys.exit(0)
    except BadParam as err:
        _print_badparam_and_exit(err)


@dl.command(short_help="Show the YAML config of the given download execution(s)",
            context_settings={'ignore_unknown_options': True})
@click.option('-d', '--dburl', **clickutils.DBURL_OR_YAML_ATTRS)
@click.option('-did', '--download-id', multiple=True, type=int,
              help="The unique id (integer) of the download execution(s) to select, "
                   "which can be supplied instead of or together with the list of "
                   "download indices")
@click.argument("download_indices", required=False, nargs=-1)
def config(dburl, download_id, download_indices):
    """Return the YAML configuration(s) used in previous download execution(s)

    [DOWNLOAD_INDICES] (optional): The space-separated indices of the download executions
    to inspect, where 0 indicates the first/oldest. To start counting from the end use a
    negative index, e.g., -1 for the last execution, -2 for the next-to-last.
    When no download index or id is provided, the last download execution (index -1) will
    be shown by default.
    With a single download execution to inspect, the output of this command can be piped
    into a YAML file and directly used in a new download
    """
    # import in function body to speed up the main module import:
    from stream2segment.download.db.inspection.main import config as _config

    _print_waitmsg_while_fetching_data()

    try:
        with warnings.catch_warnings():  # capture (ignore) warnings
            warnings.simplefilter("ignore")
            if not download_id and not download_indices:
                download_indices = [-1]
            _config(dburl, download_indices or None, download_id or None, None)
        sys.exit(0)
    except BadParam as err:
        _print_badparam_and_exit(err)


def _print_waitmsg_while_fetching_data(**kwargs):
    kwargs.setdefault('flush', True)
    kwargs.setdefault('file', sys.stderr)
    msg = ('Fetching data, please wait (this might take a while depending on the '
           'db size and connection, if db is remote)')
    print(msg, **kwargs)


@cli.group(short_help="Database management")
def db():  # pylint: disable=missing-docstring
    pass


@db.command(short_help="Drop (delete) download executions and all associated "
                       "segments")
@click.option('-d', '--dburl', **clickutils.DBURL_OR_YAML_ATTRS)
@click.option('-did', '--download-id', multiple=True, type=int, required=True,
              help="The id(s) of the download execution(s) to be deleted")
def drop(dburl, download_id):
    """Drop (delete) download executions. WARNING: this command deletes also
    all segments, stations and channels downloaded with the given download
    execution
    """
    # import in function body to speed up the main module import:
    from stream2segment.download.db.management import drop

    _print_waitmsg_while_fetching_data()

    try:
        with warnings.catch_warnings():  # capture (ignore) warnings
            warnings.simplefilter("ignore")
            ret = drop(dburl, download_id)
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
    except BadParam as err:
        _print_badparam_and_exit(err)


@db.command(short_help="Add/rename/delete class labels from the database")
@click.option('-d', '--dburl', **clickutils.DBURL_OR_YAML_ATTRS)
@click.option('--add', multiple=True, nargs=2, type=str, required=False,
              help="New class label(s) to be added (example: --add label description)")
@click.option('--rename', multiple=True, nargs=3, type=str, required=False,
              help="Class label(s) to be renamed "
                   "(example: --rename old_label new_label new_description). Set "
                   "new_description to \"\" or '' to rename the label only and "
                   "keep the old description")
@click.option('--delete', multiple=True, type=str, required=False,
              help="Class label(s) to be deleted (example: --delete label). Note: "
                   "this will also remove all existing mappings (class labellings) "
                   "between segments and their associated label")
@click.option("--no-prompt", is_flag=True, default=False,
              help="Do not prompt the user when attempting to "
                   "perform an operation")
def classlabel(dburl, add, rename, delete, no_prompt):
    """Add/Rename/delete class labels from the database. A class label is
    composed of a label name (e.g., LowS2N) and a short description (e.g.,
    "Segment has a low signal-to-noise ratio") and denotes any user-defined
    characteristic that you want to assign to certain segments either manually
    in the GUI, or programmatically in the processing module or your code.
    Class labels can then be used for e.g., supervised classification problems,
    or tp perform custom selection on specific segments before processing.
    """
    # import in function body to speed up the main module import:
    from stream2segment.download.db.management import classlabels

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

        c_labels = classlabels(dburl, add=add_arg, rename=rename_arg, delete=delete_arg)
        print('Done. Current class labels on the database:')
        if not c_labels:
            print('None')
        else:
            for c_lbl, c_dsc in c_labels.items():
                print("%s (%s)" % (c_lbl, c_dsc))
        sys.exit(0)
    except BadParam as err:
        _print_badparam_and_exit(err)


if __name__ == '__main__':
    cli()  # noqa
