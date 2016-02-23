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
import dateutil.parser as dparser

def main():
    # Version of this software
    version = '0.1a1'

#     # load yaml config file with two default values for 'start' and 'end' (if missing from yaml)
#     cfg_dict = yaml2dict('config.yaml',
#                          start=lambda: (dt.date.today() - dt.timedelta(days=1)).isoformat(),
#                          end=dt.date.today().isoformat())

    # load config file. This might be better implemented in the near future
    with open('config.yaml', 'r') as stream:
        cfg_dict = yaml.load(stream)

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
        if to_datetime(string) is None:
            raise argparse.ArgumentTypeError(str(string) + " " + str(type(str)) +
                                             " is not a valid date")
        return string

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
    parser.add_argument('--ptimespan', nargs=2,
                        help='Minutes to account for before and after the P arrival time.', type=float,
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

    parser.add_argument('--version', action='version',
                        version='event2wav %s' % version)
    args = parser.parse_args()

    # Limit the maximum verbosity to 3 (DEBUG)
    verbNum = 3 if args.verbosity >= 3 else args.verbosity
    lvl = 40 - verbNum * 10
    logging.basicConfig(level=lvl, 
#                         format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',)
                        format='%(levelname)-8s %(message)s',)
#     if __name__ == "__main__":
#         # enable printing to stdout if called as script
#         # logging.basicConfig(stream=sys.stdout)
#         # create console handler and set level to debug
#         ch = logging.StreamHandler()
#         ch.setLevel(lvl)
# 
#         # create formatter
#         formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 
#         # add formatter to ch
#         ch.setFormatter(formatter)
# 
#         # add ch to logger
#         logging.getLogger('').addHandler(ch)
#     else:
    logging.basicConfig(level=lvl)

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
