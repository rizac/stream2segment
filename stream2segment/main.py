#!/usr/bin/python
# event2wav: First draft to download waveforms related to events
#
# (c) 2015 Deutsches GFZ Potsdam
# <XXXXXXX@gfz-potsdam.de>
#
# ----------------------------------------------------------------------


"""event2wav: First draft to download waveforms related to events

   :Platform:
       Linux
   :Copyright:
       Deutsches GFZ Potsdam <XXXXXXX@gfz-potsdam.de>
   :License:
       To be decided!
"""
import os
import sys
import yaml
import logging
import argparse
import datetime as dt
from stream2segment.query import save_waveforms
from stream2segment.utils import datetime as dtime


def valid_date(string):
    """does a check on string to see if it's a valid datetime string.
    Returns the string on success, throws an ArgumentTypeError otherwise"""
    return dtime(string, on_err=argparse.ArgumentTypeError)
    # return string


def load_def_cfg(filepath='config.yaml', raw=False):
    with open(filepath, 'r') as stream:
        ret = yaml.load(stream) if not raw else stream.read()
    # load config file. This might be better implemented in the near future
    return ret


def parse_args(description=sys.argv[0], args=sys.argv[1:], cfg_dict=load_def_cfg()):

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--gui',
                        help='Launch GUI editor to annotate class ids on '
                             'pre-saved data (using this tool)',
                        action='store_true')
    parser.add_argument('-e', '--eventws',
                        help='Event WS to use in queries.',
                        default=cfg_dict['eventws'])
    parser.add_argument('--minmag',
                        help='Minimum magnitude.', type=float,
                        default=cfg_dict['minmag'])
    parser.add_argument('--minlat',
                        help='Minimum latitude.', type=float,
                        default=cfg_dict['minlat'])
    parser.add_argument('--maxlat',
                        help='Maximum latitude.', type=float,
                        default=cfg_dict['maxlat'])
    parser.add_argument('--minlon',
                        help='Minimum longitude.', type=float,
                        default=cfg_dict['minlon'])
    parser.add_argument('--maxlon',
                        help='Maximum longitude.', type=float,
                        default=cfg_dict['maxlon'])
    parser.add_argument('--ptimespan', nargs=2, type=float,
                        help='Minutes to account for before and after the P arrival time.',
                        default=cfg_dict['ptimespan'])
    parser.add_argument('--search_radius_args', nargs=4,
                        help=('arguments to the function returning the search radius R whereby all '
                              'stations within R will be queried from given event location'),
                        type=float,
                        default=cfg_dict['search_radius_args'])

    # set the file, finally:
    # note that the relative parser are called if the argument is a string. If not supplied, i.e.
    # None, the parser won't be called. Thus, we supply a default argument
    parser.add_argument('-o', '--outpath',  # type=existing_directory,
                        help='Db path where to store waveforms, or db path from where to read the'
                             ' waveforms, if --gui is specified.',
                        default=cfg_dict.get('outpath', ''))
    _now = dt.datetime.utcnow()
    dtn = dt.datetime(_now.year, _now.month, _now.day, 0, 0, 0)  # set to today at midnight
    parser.add_argument('-f', '--start', type=valid_date,
                        default=cfg_dict.get('start',
                                             dtn-dt.timedelta(days=1)),
                        help='Limit to events on or after the specified start time.')
    parser.add_argument('-t', '--end', type=valid_date,
                        default=cfg_dict.get('end', dtn),
                        help='Limit to events on or before the specified end time.')

#     parser.add_argument('--version', action='version',
#                         version='event2wav %s' % version)
    args = parser.parse_args(args)
    return args


def main():

    cfg_dict = load_def_cfg()
    args = parse_args(description='Download waveforms related to events',
                      cfg_dict=cfg_dict)

    if args.gui:
        from stream2segment.gui import plot
        plot.main(args.outpath)
        sys.exit(0)

    vars_args = vars(args)
    # vars_args = config_logging(**vars_args)  # this also removes unwanted keys in saveWaveform

    vars_args['channels'] = cfg_dict['channels']
    vars_args['datacenters_dict'] = cfg_dict['datacenters']

    # remove unwanted args:
    vars_args.pop('gui', None)
    try:
        sys.exit(save_waveforms(**vars_args))
    except KeyboardInterrupt:
        print "wow"
        return 1

if __name__ == '__main__':
    main()
    # TODO: store arrival time in file name or mseed see obspy read
    # data[0].stats data[1].stats
    # see docs.obspy.org/packages/obspy.signal.html
    # FIXME: getTravelTime deprecated!!!
    # fix bug non existing folder