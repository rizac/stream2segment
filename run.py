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
from stream2segment.query_utils import saveWaveforms
from stream2segment.utils import to_datetime


def existing_directory(string):
    if not string:
        raise argparse.ArgumentTypeError(" not specified")

    path_ = string
    if not os.path.isabs(string):
        string = os.path.abspath(os.path.join(os.path.dirname(__file__), string))

    if not os.path.exists(string):
        os.makedirs(string)
        logging.warning('"%s" newly created (did not exist)', string)

    if not os.path.isdir(string):
        raise argparse.ArgumentTypeError(path_ + " is not an existing directory")
    return string


def valid_date(string):
    """does a check on string to see if it's a valid datetime string.
    Returns the string on success, throws an ArgumentTypeError otherwise"""
    if to_datetime(string) is None:
        raise argparse.ArgumentTypeError(str(string) + " " + str(type(str)) +
                                         " is not a valid date")
    return string


def load_def_cfg(filepath='config.yaml'):
    # load config file. This might be better implemented in the near future
    with open(filepath, 'r') as stream:
        cfg_dict = yaml.load(stream)
    return cfg_dict


def parse_args(description=sys.argv[0], args=sys.argv[1:], cfg_dict=load_def_cfg()):

    parser = argparse.ArgumentParser(description=description)

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
    parser.add_argument('-v', '--verbosity', action="count", default=cfg_dict['verbosity'],
                        help='Increase the verbosity level')

    # set the file, finally:
    # note that the relative parser are called if the argument is a string. If not supplied, i.e.
    # None, the parser won't be called. Thus, we supply a default argument
    parser.add_argument('-o', '--outpath', type=existing_directory,
                        help='Path where to store waveforms.',
                        default=cfg_dict.get('outpath', ''))
    dtn = dt.datetime.utcnow()
    parser.add_argument('-f', '--start', type=valid_date,
                        default=cfg_dict.get('start',
                                             dtn-dt.timedelta(days=1)).strftime("%Y-%m-%d"),
                        help='Limit to events on or after the specified start time.')
    parser.add_argument('-t', '--end', type=valid_date,
                        default=cfg_dict.get('end', dtn.strftime("%Y-%m-%d")),
                        help='Limit to events on or before the specified end time.')

#     parser.add_argument('--version', action='version',
#                         version='event2wav %s' % version)
    args = parser.parse_args(args)
    return args


def config_logging(verbosity=3):
    # Limit the maximum verbosity to 3 (DEBUG)
    # see http://stackoverflow.com/questions/2557168/how-do-i-change-the-default-format-of-log-messages-in-python-app-engine
    verbNum = max(0, min(3, verbosity))
    lvl = 40 - verbNum * 10
    logging.basicConfig(level=lvl,
                        format='%(levelname)-8s %(message)s',)


def main():
    # Version of this software
    cfg_dict = load_def_cfg()
    args = parse_args(description='First draft to download waveforms related to events',
                      cfg_dict=cfg_dict)

    config_logging(args.verbosity)

    try:
        # add last two parameters not customizable via arguments
        # FIXME: right way?
        vars_args = vars(args)
        vars_args['channelList'] = cfg_dict['channels']
        vars_args['datacenters_dict'] = cfg_dict['datacenters']
        # remove unwanted args:
        vars_args.pop('verbosity', None)
        vars_args.pop('version', None)
        saveWaveforms(**vars_args)
        sys.exit(0)
    except ValueError as verr:
        logging.error('Error while saving waveforms %s' % str(verr))

    sys.exit(1)

if __name__ == '__main__':
    main()
    # TODO: store arrival time in file name or mseed see obspy read
    # data[0].stats data[1].stats
    # see docs.obspy.org/packages/obspy.signal.html
    # FIXME: getTravelTime deprecated!!!
    # fix bug non existing folder