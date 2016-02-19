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
from stream2segment.utils import saveWaveforms


def main():
    # Version of this software
    version = '0.1a1'
    # load config file. This might be better implemented in the near future
    with open('config.yaml', 'r') as stream:
        cfg_dict = yaml.load(stream)

    description = 'First draft to download waveforms related to events'
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
    parser.add_argument('--minbeforep',
                        help='Minutes to account for before the P arrival time.', type=float,
                        default=cfg_dict['minBeforeP'])
    parser.add_argument('--minafterp',
                        help='Minutes to account for after the P arrival time.', type=float,
                        default=cfg_dict['minAfterP'])
    parser.add_argument('--distfromevent',
                        help='distance radius from event, in Km', type=float,
                        default=cfg_dict['distFromEvent'])
    parser.add_argument('-v', '--verbosity', action="count", default=cfg_dict['verbosity'],
                        help='Increase the verbosity level')

    # adding arguments with more complex defaults:
    # try to create the file if it does not exist:
    # check path where to store stuff:
    path_ = cfg_dict['outpath']
    if not os.path.isabs(path_):
        path_ = os.path.abspath(os.path.join(os.path.dirname(__file__), path_))

    if not os.path.exists(path_):
        os.makedirs(path_)
        logging.warning('"%s" newly created (did not exist)' % path_)
    # set the file, finally:
    parser.add_argument('-o', '--outpath',
                        help='Path where to store waveforms.',
                        default=path_)

    try:
        def_start = cfg_dict['start']
    except KeyError:
        def_start = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    parser.add_argument('-f', '--start',
                        default=def_start,
                        help='Limit to events on or after the specified start time.')

    try:
        def_end = cfg_dict['end']
    except KeyError:
        def_end = dt.date.today().isoformat()
    parser.add_argument('-t', '--end', default=def_end,
                        help='Limit to events on or before the specified end time.')

    parser.add_argument('--version', action='version',
                        version='event2wav %s' % version)
    args = parser.parse_args()

    # Limit the maximum verbosity to 3 (DEBUG)
    verbNum = 3 if args.verbosity >= 3 else args.verbosity
    lvl = 40 - verbNum * 10
    logging.basicConfig(level=lvl)

    try:
        # add last two parameters not customizable via arguments
        # FIXME: right way?
        vars_args = vars(args)
        vars_args['channels'] = cfg_dict['channels']
        vars_args['datacenters_dict'] = cfg_dict['datacenters']
        saveWaveforms(**vars_args)
        sys.exit(0)
    except ValueError as verr:
        logging.error('Error while saving waveforms %s' % str(verr))

    sys.exit(1)

if __name__ == '__main__':
    # enable printing to stdout if called as script
    logging.basicConfig(stream=sys.stdout)
    main()
