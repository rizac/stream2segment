'''
Module implementing the Command line interface (cli) to access function in the main module

:date: Oct 8, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''

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

# future aliased imports(needs future package installed, otherwise remove):
# You want to import and use safely, e.g. collections.UserDict, collections.UserList,
# collections.UserString, urllib.parse, urllib.request, urllib.response, urllib.robotparser,
# urllib.error, itertools.filterfalse, itertools.zip_longest, subprocess.getoutput,
# subprocess.getstatusoutput, sys.intern (a full list available on
# http://python-future.org/imports.html#aliased-imports)
# If none of the above is needed, you can safely remove the next two lines
# from future.standard_library import install_aliases
# install_aliases()

import sys
import os
from collections import OrderedDict

import click
# from click.exceptions import BadParameter, MissingParameter

from stream2segment import main
from stream2segment.utils.resources import get_templates_fpath, yaml_load_doc
from stream2segment.traveltimes import ttcreator
from stream2segment.utils import inputargs


class clickutils(object):
    """Container for Options validations, default settings so as not to pollute the click
    decorators"""

    TERMINAL_HELP_WIDTH = 110  # control width of help. 80 should be the default (roughly)
    DEFAULTDOC = yaml_load_doc(get_templates_fpath("download.yaml"))
    EQA = "(eventws query argument)"
    DBURL_OR_YAML_ATTRS = dict(type=inputargs.extract_dburl_if_yamlpath,
                               metavar='TEXT or PATH',
                               help="%s. %s" % (DEFAULTDOC['dburl'],
                                                ("It can also be the path of a yaml file "
                                                 "containing the property 'dburl' "
                                                 "(e.g., the config file used for "
                                                 "downloading)")),
                               required=True)
    ExistingPath = click.Path(exists=True, file_okay=True, dir_okay=False, writable=False,
                              readable=True)

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
        click.option('--opt1', ..., callback=set_help_from_yaml, is_eager=True,...)
        click.option('--opt2'...)
        click.option('--opt3'..., help='my custom help. Do not fetch help from config')
        click.option('--opt4'...)
        ...
        ```
        """
        cfg_doc = cls.DEFAULTDOC
        for option in (opt for opt in ctx.command.params if opt.param_type_name == 'option'):
            if option.help is None:
                option.help = cfg_doc.get(option.name, "")
                # remove implementation details from the cli (avoid too much information,
                # or information specific to the yaml file and not the cli):
                idx = option.help.find('Implementation details:')
                if idx > -1:
                    option.help = option.help[:idx]

        return value


@click.group()
def cli():
    """stream2segment is a program to download, process, visualize or annotate massive amounts of
    event-based seismic waveform data segments.
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


@cli.command(short_help='Creates template/config files in a specified directory',
             context_settings=dict(max_content_width=clickutils.TERMINAL_HELP_WIDTH))
@click.argument('outdir')
def init(outdir):
    """Creates template files for launching download,
    processing and visualization. OUTDIR will be created if it does not exist
    """
    helpdict = OrderedDict([
        ("download.yaml",
         "download configuration file ('s2s download -c' option)"),
        ("paramtable.py",
         "processing python file for creating a parametric table in csv format"
         " ('s2s process -p' and 's2s show -p' option) [template#1]"),
        ("paramtable.yaml",
         "processing configuration file of [template#1]"
         " ('s2s process -c' and 's2s show -c' option)"),
        ("save2fs.py",
         "processing python file for saving waveform to filesystem"
         " ('s2s process -p' and 's2s show -p' option) [template#2]"),
        ("save2fs.yaml",
         "processing configuration file of [template#2]"
         " ('s2s process -c' and 's2s show -c' option)")
        ])

    try:
        copied_files = main.init(outdir, click.prompt, *helpdict)  # pass only helpdict keys
        if not copied_files:
            print("No file copied")
        else:
            print("%d file(s) copied in '%s':" % (len(copied_files), outdir))
            frmt = "- {:<%d} {}" % max(len(f) for f in helpdict.keys())
            for fcopied in copied_files:
                bname = os.path.basename(fcopied)
                print(frmt.format(bname, helpdict.get(bname, "")))
            sys.exit(0)
    except Exception as exc:  # pylint: disable=broad-except
        print('')
        print("error: %s" % str(exc))
    sys.exit(1)


# NOTES BELOW:
# First, naming conventions:
# * option short name: any click option name starting with "-"
# * option long name: any click option name starting with "--"
# * option default name: any click option name not starting with any "-"
#   (see http://click.pocoo.org/5/parameters/#parameter-names)
# * option help: the option help shown when issuing "--help" from the command line
# * yaml param help: the help in the docstring immediately preceeding a yaml param name,
#   (fetched from the default download.yaml file in the 'templates' folder)
# 1. we want to set each option help from the corresponding yaml param help, so that we keep docs
#    updated in only one place.
#    This is done by the method `clickutils.set_help_from_yaml`, attached
#    as callback to the option '--dburl' below, which has 'is_eager=True', meaning that the
#    callback is executed before all other options, even when invoking the command with --help.
# 2. Note: Don't set required = True with eager=True in a click option, as it forces that option
#    to be always present, and thus raises if only --help is given
# 3. For yaml param help, any string following "Implementation details:": will not be shown
#    in the corresponding option help.
# 4. Some yaml params accepts different names (e.g., 'net' will
#    be recognized as 'networks'): by convention, these are provided as option long names.
#    (Options short names can be changed without problems, in principle).
#    For these options, you need also to provide an option default name which MUST MATCH
#    the corresponding yaml param help, otherwise the option doc will not be found.
# 5. Option flags should all have default=None which lets us know that the flag is missing and use
#    the corresponding yaml param values
@cli.command(short_help='Downloads waveform data segments',
             context_settings=dict(max_content_width=clickutils.TERMINAL_HELP_WIDTH))
@click.option("-c", "--config",
              help="The path to the configuration file in yaml format "
                   "(https://learn.getgrav.org/advanced/yaml).",
              type=clickutils.ExistingPath, required=True)
@click.option('-d', '--dburl', is_eager=True, callback=clickutils.set_help_from_yaml)
@click.option('-es', '--eventws')
@click.option('-s', '--start', '--starttime', "start", type=inputargs.valid_date,
              metavar='DATE or DATETIME')
@click.option('-e', '--end', '--endtime', 'end', type=inputargs.valid_date,
              metavar='DATE or DATETIME',)
@click.option('-n', '--networks', '--network', '--net', 'networks', help='See channels')
@click.option('-z', '--stations', '--station', '--sta', 'stations', help='See channels')
@click.option('-l', '--locations', '--location', '--loc', 'locations', help='See channels')
@click.option('-k', '--channels', '--channel', '--chan', 'channels')
@click.option('-msr', '--min-sample-rate', type=float)
@click.option('-ds', '--dataws')
@click.option('-t', '--traveltimes-model')
@click.option('-w', '--timespan', nargs=2, type=float)
@click.option('-u', '--update-metadata', is_flag=True, default=None)
@click.option('-r1', '--retry-url-err', is_flag=True, default=None)
@click.option('-r2', '--retry-mseed-err', is_flag=True, default=None)
@click.option('-r3', '--retry-seg-not-found', is_flag=True, default=None)
@click.option('-r4', '--retry-client-err', is_flag=True, default=None)
@click.option('-r5', '--retry-server-err', is_flag=True, default=None)
@click.option('-r6', '--retry-timespan-err', is_flag=True, default=None)
@click.option('-i', '--inventory', is_flag=True, default=None)
@click.option('--minlat', '--minlatitude', type=float,
              help=(clickutils.EQA + " Limit to events with a latitude larger than "
                    "or equal to the specified minimum"))
@click.option('--maxlat', '--maxlatitude', type=float,
              help=(clickutils.EQA + " Limit to events with a latitude smaller than "
                    "or equal to the specified maximum"))
@click.option('--minlon', '--minlongitude', type=float,
              help=(clickutils.EQA + " Limit to events with a longitude larger than "
                    "or equal to the specified minimum"))
@click.option('--maxlon', '--maxlongitude', type=float,
              help=(clickutils.EQA + " Limit to events with a longitude smaller than "
                    "or equal to the specified maximum"))
@click.option('--lat', '--latitude', type=float,
              help=(clickutils.EQA + " Specify the latitude to be used for a radius search."))
@click.option('--lon', '--longitude', type=float,
              help=(clickutils.EQA + " Specify the longitude to be used for a radius search"))
@click.option('--minradius', type=float,
              help=(clickutils.EQA + " Limit to events within the specified minimum "
                    "number of degrees from the geographic point defined by the latitude and "
                    "longitude parameters"))
@click.option('--maxradius', type=float,
              help=(clickutils.EQA + " Limit to events within the specified maximum "
                    "number of degrees from the geographic point defined by the latitude and "
                    "longitude parameters"))
@click.option('--mindepth', type=float,
              help=(clickutils.EQA + " Limit to events with depth more than the "
                    "specified minimum"))
@click.option('--maxdepth', type=float,
              help=(clickutils.EQA + " Limit to events with depth less than the "
                    "specified maximum"))
@click.option('--minmag', '--minmagnitude', type=float,
              help=(clickutils.EQA + " Limit to events with a magnitude larger than "
                    "the specified minimum"))
@click.option('--maxmag', '--maxmagnitude', type=float,
              help=(clickutils.EQA + " Limit to events with a magnitude smaller than "
                    "the specified maximum"))
def download(config, dburl, eventws, start, end, networks,  # pylint: disable=unused-argument
             stations, locations, channels, min_sample_rate,  # pylint: disable=unused-argument
             dataws, traveltimes_model, timespan,  # pylint: disable=unused-argument
             update_metadata, retry_url_err, retry_mseed_err,  # pylint: disable=unused-argument
             retry_seg_not_found, retry_client_err,  # pylint: disable=unused-argument
             retry_server_err, retry_timespan_err, inventory,  # pylint: disable=unused-argument
             minlatitude, maxlatitude, minlongitude,  # pylint: disable=unused-argument
             maxlongitude, latitude, longitude, minradius,  # pylint: disable=unused-argument
             maxradius, mindepth, maxdepth, minmagnitude,  # pylint: disable=unused-argument
             maxmagnitude):  # pylint: disable=unused-argument
    """Downloads waveform data segments with metadata in a specified database.
    The -c option (required) sets the defaults for all other options below, **which are optional**
    """
    try:
        overrides = {k: v for k, v in locals().items()
                     if v not in ((), {}, None) and k != 'config'}
        # pre-process all event ws query arguments:
        eventws_dict = {par: overrides.pop(par) for par in ("minlatitude", "maxlatitude",
                                                            "minlongitude", "maxlongitude",
                                                            "latitude", "longitude", "minradius",
                                                            "maxradius", "mindepth", "maxdepth",
                                                            "minmagnitude", "maxmagnitude")
                        if par in overrides}
        if eventws_dict:
            overrides['eventws_query_args'] = eventws_dict

        sys.exit(main.download(config, verbosity=2, **overrides))
    except inputargs.BadArgument as aerr:
        print(aerr)
        sys.exit(1)
    except KeyboardInterrupt:  # this except avoids printing traceback
        sys.exit(2)


@cli.command(short_help='Processes downloaded waveform data segments',
             context_settings=dict(max_content_width=clickutils.TERMINAL_HELP_WIDTH))
@click.option('-d', '--dburl', **clickutils.DBURL_OR_YAML_ATTRS)
@click.option("-c", "--config",
              help="The path to the configuration file in yaml format "
                   "(https://learn.getgrav.org/advanced/yaml).",
              type=clickutils.ExistingPath, required=True
              )
@click.option("-p", "--pyfile",
              help="The path to the python file where the user-defined processing function "
                   "is implemented. The function will be called iteratively on each segment "
                   "selected in the config file", type=clickutils.ExistingPath, required=True
              )
@click.option("-f", "--funcname",
              help="The name of the user-defined processing function in the given python file. "
                   "Optional: defaults to '%s' when "
                   "missing" % inputargs.default_processing_funcname(),
              )  # do not set default='main', so that we can test when arg is missing or not
@click.option("-mp", "--multi-process", is_flag=True,
              help="Use parallel sub-processes to speed up the execution. "
                   "When missing, it defaults to false"
              )  # do not set default='main', so that we can test when arg is missing or not
@click.option("-np", "--num-processes", type=int, default=None,
              help="The number of sub-processes. If missing, it is set as the "
                   "the number of CPUs in the system. This option is ignored "
                   "if --multi-process is not given",
              )  # do not set default='main', so that we can test when arg is missing or not
@click.argument('outfile', required=False)
def process(dburl, config, pyfile, funcname,
            multi_process, num_processes,  # pylint: disable=unused-argument
            outfile):
    """Processes downloaded waveform data segments via a custom python file and a configuration
    file.


    [OUTFILE] (optional): the path of the .csv file where the output of the user-defined processing
    function F will be written to (one row per processed segment); all logging information,
    errors or warnings will be written to the file [OUTFILE].log.
    If this argument is missing, then the output of F (if any) will be discarded,
    and all logging messages will be redirected to the standard error
    """
    try:
        # override config values for multi_process and num_processes
        overrides = {k: v for k, v in locals().items()
                     if v not in ((), {}, None) and k in ('multi_process', 'num_processes')}
        if overrides:
            # if given, put these into 'advanced_settings' sub-dict. Note that
            # nested dict will be merged with the values of the config
            overrides = {'advanced_settings': overrides}
        sys.exit(main.process(dburl, pyfile, funcname, config, outfile, verbose=True, **overrides))
    except inputargs.BadArgument as aerr:
        print(aerr)
        sys.exit(1)  # exit with 1 as normal python exceptions
    except KeyboardInterrupt:  # this except avoids printing traceback
        sys.exit(2)


@cli.command(short_help='Shows raw and processed downloaded waveform\'s plots in a browser',
             context_settings=dict(max_content_width=clickutils.TERMINAL_HELP_WIDTH))
@click.option('-d', '--dburl', **clickutils.DBURL_OR_YAML_ATTRS)
@click.option("-c", "--configfile",
              help="The path to the configuration file in yaml format "
                   "(https://learn.getgrav.org/advanced/yaml).", type=clickutils.ExistingPath,
              required=True
              )
@click.option("-p", "--pyfile",
              help="The path to the python file with the plot functions implemented",
              type=clickutils.ExistingPath, required=True
              )
def show(dburl, configfile, pyfile):
    """Shows raw and processed downloaded waveform\'s plots in a browser"""
    main.show(dburl, pyfile, configfile)


@cli.group(short_help="Program utilities. Type --help to list available sub-commands")
def utils():  # pylint: disable=missing-docstring
    pass


@utils.command(short_help='Produces download information either in plain text or html format',
               context_settings=dict(max_content_width=clickutils.TERMINAL_HELP_WIDTH))
@click.option('-d', '--dburl', **clickutils.DBURL_OR_YAML_ATTRS)
@click.option('-did', '--download-id', multiple=True, type=int,
              help="Limit the download statistics to a specified set of download ids (integers) "
                   "when missing, all downloads are shown. this option can be given multiple "
                   "times: .. -did 1 --download_id 2 ...")
@click.option('-g', '--maxgap-threshold', type=float, default=0.5,
              help="Set the threshold (in number of samples relative to each segment) "
                   "to set which segments in the download statistics "
                   "have gaps/overlaps. "
                   "Defaults to 0.5, meaning that segments whose maximum gap is >0.5 will be "
                   "identified has having gaps, and segments whose maximum gap is <-0.5 will "
                   "be identified has having overlaps")
@click.option('-htm', '--html', is_flag=True, help="Generate an interactive "
              "dynamic web page where the download infos are visualized on a map, with statistics "
              "on a per-station and data-center basis. A working internet connection is needed to"
              "properly view the page")
@click.argument("outfile", required=False, type=click.Path(file_okay=True,
                                                           dir_okay=False, writable=True,
                                                           readable=True))
def dinfo(dburl, download_id, maxgap_threshold, html, outfile):
    """Produces download information either in plain text or html format.

    [OUTFILE] (optional): the output file where the information will be saved to.
    If missing, results will be printed to screen or opened in a web browser
    (depending on the option '--html')"""
    print('Fetching data, please wait (this might take a while depending on the '
          'db size and connection)')
    try:
        main.dinfo(dburl, download_id or None, maxgap_threshold, html, outfile)
        if outfile is not None:
            print("download info and statistics written to '%s'" % outfile)
        sys.exit(0)
    except inputargs.BadArgument as aerr:
        print(aerr)
        sys.exit(1)  # exit with 1 as normal python exceptions


@utils.command(short_help='Prints on screen quick help on stream2segment built-in math functions',
               context_settings=dict(max_content_width=clickutils.TERMINAL_HELP_WIDTH))
@click.option("-t", "--type", type=click.Choice(['numpy', 'obspy', 'all']), default='all',
              show_default=True,
              help="Show help only for the function matching the given type. Numpy indicates "
                    "functions operating on numpy arrays "
                    "(module `stream2segment.process.math.ndarrays`). "
                    "Obspy (module `stream2segment.process.math.traces`) the functions operating "
                    "on obspy Traces, most of which are simply the numpy counterparts defined "
                    "for Trace objects")
@click.option("-f", "--filter", default='*', show_default=True,
              help="Show doc only for the function whose name matches the given filter. "
                    "Wildcards (* and ?) are allowed")
def mathinfo(type, filter):  # @ReservedAssignment pylint: disable=redefined-outer-name
    '''Prints on screen the doc-strings of the math functions implemented in this package,
    according to the given type and filter'''
    for line in main.helpmathiter(type, filter):
        print(line)


@utils.command(short_help="Creates a travel time table for computing "
               "travel times (via linear or cubic interpolation, or nearest point) "
               "in a *much* faster way than using obspy routines directly for large number of "
               "points")
@click.option('-o', '--output', required=True,
              help=('The output file. If directory, the file name will be automatically '
                    'created inside the directory. Otherwise must denote a valid writable '
                    'file name. The extension .npz will be added automatically'))
@click.option("-m", "--model", required=True,
              help="the model name, e.g. iasp91, ak135, ..")
@click.option('-p', '--phases', multiple=True,  required=True,
              help=("The phases used, e.g. ttp+, tts+. Can be typed multiple times, e.g."
                    "-m P -m p"))
@click.option('-t', '--tt_errtol', type=float, required=True,
              help=('The error tolerance (in seconds). The algorithm will try to store grid points '
                    'whose distance is close to this value. Decrease this value to increase '
                    'precision, increase this value to increase the execution speed'))
@click.option('-s', '--maxsourcedepth', type=float, default=ttcreator.DEFAULT_SD_MAX,
              show_default=True,
              help=('Optional: the maximum source depth (in km) used for the grid generation. '
                    'When loaded, the relative model can calculate travel times for source depths '
                    'lower or equal to this value'))
@click.option('-r', '--maxreceiverdepth', type=float, default=ttcreator.DEFAULT_RD_MAX,
              show_default=True,
              help=('Optional: the maximum receiver depth (in km) used for the grid generation. '
                    'When loaded, the relative model can calculate travel times for receiver '
                    'depths lower or equal to this value. Note that setting this value '
                    'greater than zero might lead to numerical problems, e.g. times not '
                    'monotonically increasing with distances, especially for short distances '
                    'around the source'))
@click.option('-d', '--maxdistance', type=float, default=ttcreator.DEFAULT_DIST_MAX,
              show_default=True,
              help=('Optional: the maximum distance (in degrees) used for the grid generation. '
                    'When loaded, the relative model can calculate travel times for receiver '
                    'depths lower or equal to this value'))
@click.option('-P', '--pwavevelocity', type=float, default=ttcreator.DEFAULT_PWAVEVELOCITY,
              show_default=True,
              help=('Optional: the P-wave velocity (in km/sec), if the calculation of the P-waves '
                    'is required according to the argument `phases` (otherwise ignored). '
                    'As the grid points (in degree) of the distances axis '
                    'cannot be optimized, a fixed step S is set for which it holds: '
                    '`min(travel_times(D+step))-min(travel_times(D)) <= tt_errtol` for any point '
                    'D of the grid. The P-wave velocity is needed to asses such a step '
                    '(for info, see: '
                    'http://rallen.berkeley.edu/teaching/F04_GEO302_PhysChemEarth/Lectures/HellfrichWood2001.pdf)'))  # @IgnorePep8 pylint: disable=line-too-long
@click.option('-S', '--swavevelocity', type=float, default=ttcreator.DEFAULT_SWAVEVELOCITY,
              show_default=True,
              help=('Optional: the S-wave velocity (in km/sec), if the calculation of the S-waves '
                    '*only* is required, according to the argument `phases` (otherwise ignored). '
                    'As the grid points (in degree) of the distances axis '
                    'cannot be optimized, a fixed step S is set for which it holds: '
                    '`min(travel_times(D+step))-min(travel_times(D)) <= tt_errtol` for any point '
                    'D of the grid. If the calculation of the P-waves is also needed according to '
                    'the argument `phases` , the p-wave velocity value will be used and this '
                    'argument will be ignored. (for info, see: '
                    '(http://rallen.berkeley.edu/teaching/F04_GEO302_PhysChemEarth/Lectures/HellfrichWood2001.pdf)'))  # @IgnorePep8 pylint: disable=line-too-long
def ttcreate(output, model, phases, tt_errtol, maxsourcedepth, maxreceiverdepth, maxdistance,
             pwavevelocity, swavevelocity):
    """Creates a travel time table TT, i.e. a grid of
       source_depths, receiver_depths and distances asssociated to the corresponding
       travel time T, computed with obspy routines. This allows the calculation of the
       travel times (via linear or cubic interpolation, or nearest point)
       in a *much* faster way than using obspy routines directly for large number of
       points. Stores the
       resulting file as .npz compressed numpy format. The file path can be given
       as parameter in the download config to customize the travel times computation"""
    try:
        output = ttcreator._filepath(output, model, phases)
        ttcreator.computeall(output, model, tt_errtol, phases, maxsourcedepth, maxreceiverdepth,
                             maxdistance, pwavevelocity, swavevelocity, isterminal=True)
        sys.exit(0)
    except Exception as exc:  # pylint: disable=broad-except
        print("ERROR: %s" % str(exc))
        sys.exit(1)


if __name__ == '__main__':
    cli()  # pylint: disable=E1120
