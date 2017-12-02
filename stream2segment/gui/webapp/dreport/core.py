'''
Core functionalities for the GUI web application (download report)

:date: Oct 12, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''

# make the following(s) behave like python3 counterparts if running from python2.7.x
# (http://python-future.org/imports.html#explicit-imports):
from builtins import map, zip

import re
from itertools import cycle, chain
from collections import OrderedDict

from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import load_only

from stream2segment.io.db.pdsql import colnames
from stream2segment.io.db.models import Segment, Class, Station, Channel, DataCenter, Event,\
    ClassLabelling, Download
from stream2segment.io.db.queries import query4gui, query4dreport, querystationinfo4dreport
# from stream2segment.io.db import sqlevalexpr
from stream2segment.utils.resources import yaml_load_doc, get_templates_fpath

from stream2segment.download.utils import custom_download_codes


def _getlabels(max_gap_overlap=(-0.5, 0.5)):
    urlexc, mseedexc, time_err, time_warn = custom_download_codes()
    c_empty = Segment.data.isnot(None) & (func.length(Segment.data) == 0)
    # sql between includes endpoints
    no_gaps = Segment.maxgap_numsamples.between(max_gap_overlap[0], max_gap_overlap[1])
    c_data = Segment.has_data == True  # @IgnorePep8
    c_gaps = c_data & ~no_gaps
    c_srate_mismatch = c_data & no_gaps & (Segment.sample_rate != Channel.sample_rate)
    return OrderedDict([['no code', (True, Segment.download_code.is_(None))],
                        ['url error', (True, Segment.download_code == urlexc)],
                        ['mseed error', (True, Segment.download_code == mseedexc)],
                        ['4xx HTTP code', (True, (Segment.download_code >= 400) &
                                                 (Segment.download_code < 500))],
                        ['5xx HTTP code', (True, Segment.download_code >= 500)],
                        ['empty data', (True, c_empty & ~
                                        (Segment.download_code == time_err))],
                        ['gaps/overlaps', (True, c_gaps)],
                        ['sample rate mismatch (channel vs. data)', (False, c_srate_mismatch)],
                        ['data completely out of request\'s time span',
                         (True, (Segment.download_code == time_err))],
                        ['data partially out of request\'s time span',
                         (False, (Segment.download_code == time_warn))]
                        ])


def selectablelabels():
    return [(k, v[0], 0) for k, v in _getlabels().items()]


def binexprs2count():
    return OrderedDict([(k, v[1]) for k, v in _getlabels().items()])


def get_data(session):
    binexprs2count_ = binexprs2count()
    query = query4dreport(session, **binexprs2count_)
    return query.all()


def get_station_data(session, station_id, selectedLabels):
    lbls = _getlabels()
    binexprs = {}
    for key, val in selectedLabels:
        if not val:
            continue
        binexprs[key] = lbls[key][1]

    query = querystationinfo4dreport(session, station_id, **binexprs)
    return query.all()
