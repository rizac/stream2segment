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
import sys
import yaml
import click
import datetime as dt
from stream2segment.query import main as query_main
from stream2segment.utils import datetime as dtime


def valid_date(string):
    """does a check on string to see if it's a valid datetime string.
    Returns the string on success, throws an ArgumentTypeError otherwise"""
    return dtime(string, on_err=click.BadParameter)
    # return string


def get_def_timerange():
    """ Returns the default time range when  not specified, for downloading data
    the reutnred tuple has two datetime objects: yesterday, at midniight and
    today, at midnight"""
    dnow = dt.datetime.utcnow()
    endt = dt.datetime(dnow.year, dnow.month, dnow.day)
    startt = endt - dt.timedelta(days=1)
    return startt, endt


def load_def_cfg(filepath='config.yaml', raw=False):
    """Loads default config from yaml file"""
    with open(filepath, 'r') as stream:
        ret = yaml.load(stream) if not raw else stream.read()
    # load config file. This might be better implemented in the near future
    return ret


cfg_dict = load_def_cfg()


def run(gui, eventws, minmag, minlat, maxlat, minlon, maxlon, ptimespan, search_radius_args,
        outpath, start, end, min_sample_rate, action, processing_on_exist,
        **processing_args):

    if gui is True:
        from stream2segment.gui import main
        main.run_in_browser(outpath)
        sys.exit(0)

    try:
        segments = []
        ret_val = 1
        if action != 'p':
            segments, ret_val = query_main(eventws, minmag, minlat, maxlat, minlon, maxlon,
                                           search_radius_args, cfg_dict['channels'],
                                           start, end, ptimespan, min_sample_rate, outpath)
        else:
            segments = 
        if action != 'd':

    except KeyboardInterrupt:
        sys.exit(1)


@click.command()
@click.option('--gui', is_flag=True,
              help='Launch GUI editor to annotate class ids on '
              'pre-saved data (using this tool). You can also provide optional '
              'class labels to be shown after --gui, '
              'e.g.: stream2segment --gui -1 0')
@click.option('-e', '--eventws', default=cfg_dict['eventws'],
              help='Event WS to use in queries.')
@click.option('--minmag', default=cfg_dict['minmag'],
              help='Minimum magnitude.', type=float)
@click.option('--minlat', default=cfg_dict['minlat'], type=float,
              help='Minimum latitude.')
@click.option('--maxlat', default=cfg_dict['maxlat'], type=float,
              help='Maximum latitude.')
@click.option('--minlon', default=cfg_dict['minlon'], type=float,
              help='Minimum longitude.')
@click.option('--maxlon', default=cfg_dict['maxlon'], type=float,
              help='Maximum longitude.')
@click.option('--ptimespan', nargs=2, type=float,
              help='Minutes to account for before and after the P arrival time.',
              default=cfg_dict['ptimespan'])
@click.option('--search_radius_args', default=cfg_dict['search_radius_args'], type=float, nargs=4,
              help=('arguments to the function returning the search radius R whereby all '
                    'stations within R will be queried from given event location. '
                    'args are: min_mag max_mag min_distance_deg max_distance_deg'),
              )
@click.option('-o', '--outpath', default=cfg_dict.get('outpath', ''),
              help='Db path where to store waveforms, or db path from where to read the'
                   ' waveforms, if --gui is specified.')
@click.option('-f', '--start', default=cfg_dict.get('start', get_def_timerange()[0]),
              type=valid_date,
              help='Limit to events on or after the specified start time.')
@click.option('-t', '--end', default=cfg_dict.get('end', get_def_timerange()[1]),
              type=valid_date,
              help='Limit to events on or before the specified end time.')
@click.option('--min_sample_rate', default=cfg_dict['min_sample_rate'],
              help='Limit to segments on a sample rate higher than a specific threshold')
def main(gui, eventws, minmag, minlat, maxlat, minlon, maxlon, ptimespan, search_radius_args,
         outpath, start, end, min_sample_rate):

    if gui is True:
        from stream2segment.gui import main
        main.run_in_browser(outpath)
        sys.exit(0)

    try:
        sys.exit(query_main(eventws, minmag, minlat, maxlat, minlon, maxlon,
                            search_radius_args, cfg_dict['channels'],
                            start, end, ptimespan, min_sample_rate, outpath,
                            do_processing=))
    except KeyboardInterrupt:
        sys.exit(1)


if __name__ == '__main__':
    main()  # pylint: disable=E1120
