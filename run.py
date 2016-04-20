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
from stream2segment.query_utils import save_waveforms
from stream2segment.utils import datetime as dtime


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
    if dtime(string, on_err_return_none=True) is None:
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

    # logging stuff
    parser.add_argument('-v', '--log_verbosity', action="count", default=cfg_dict['log_verbosity'],
                        help='Increase the verbosity level in log')
    parser.add_argument('-l', '--log_filename', default=cfg_dict.pop('log_filename', None),
                        help='log filename (if missing the program prints log to standard output)')
    parser.add_argument('--log_filemode', default=cfg_dict.get('log_filemode', None),
                        help='log filemode. Defaults to "a". Ignored if log_filename is missing')
    parser.add_argument('--log_format', default=cfg_dict.get('log_format', None),
                        help='log message format')
    parser.add_argument('--log_datefmt', default=cfg_dict.get('log_datefmt', None),
                        help='log message datetime format')

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


def config_logging(**kwargs):
    # see here: https://docs.python.org/2/library/logging.html#logging.basicConfig

    # Limit the maximum verbosity to 3 (DEBUG)
    # see http://stackoverflow.com/questions/2557168/how-do-i-change-the-default-format-of-log-messages-in-python-app-engine

    """configures the logging from kwargs. Argument are the same as logging.basicConfig
        with a "log_" prefix with two exceptions: 
        "log_verbosity" is "level" in baseConfig BUT goes from 0 to 3 in inverse order
            (0=error, 3=debug)
        "log_filename" and "log_stream", if not specified or None, default to stream=sys.stdout
        The other parameters:
        "log_filename", "log_filemode", "log_format", "log_datefmt"
        behaves like in logging.baseConfig
        :return: the input dictionary WITHOUT the keys used for configuruing logging
    """

    logging_dict = {}

    for key, val in {'log_format': 'format',
                     'log_datefmt': 'datefmt',
                     'log_verbosity': 'level',
                     'log_filename': 'filename',
                     'log_filemode': 'filemode'}.iteritems():
        var = kwargs.pop(key, None)
        if var is not None:
            logging_dict[val] = var

    if 'level' in logging_dict:
        verbNum = max(0, min(3, logging_dict['level']))
        lvl = 40 - verbNum * 10
        logging_dict['level'] = lvl

    if 'filename' not in logging_dict:
        logging_dict.pop('filemode', None)
        logging_dict['stream'] = sys.stdout

    logging.basicConfig(**logging_dict)

    return kwargs


def main():

    cfg_dict = load_def_cfg()
    args = parse_args(description='First draft to download waveforms related to events',
                      cfg_dict=cfg_dict)

    try:
        # add last two parameters not customizable via arguments
        # FIXME: right way?
        vars_args = vars(args)
        vars_args = config_logging(**vars_args)  # this also removes unwanted keys in saveWaveform

        vars_args['channelList'] = cfg_dict['channels']
        vars_args['datacenters_dict'] = cfg_dict['datacenters']
        # remove unwanted args:
        vars_args.pop('version', None)
        save_waveforms(**vars_args)
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