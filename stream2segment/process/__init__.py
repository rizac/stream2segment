from stream2segment.io.db import close_session
from stream2segment.io.db.inspection import attnames, get_related_models
from stream2segment.process.db.models import Segment
from stream2segment.process.db.sqlevalexpr import exprquery
from stream2segment.process.main import process, imap, SkipSegment
from stream2segment.io import yaml_load
from stream2segment.process.db import get_session


def get_segments(dburl, conditions, orderby=None):
    """Return a query object (iterable of `Segment`s) from teh given conditions
    Example of conditions (dict):
    ```
    {
        'id' : '<6',
        'has_data': 'true'
    }
    ```
    :param conditions: a dict of string columns mapped to **string**
        expression, e.g. "column2": "[1, 45]" or "column1": "true" (note:
        string, not the boolean True). A string column is an expression
        denoting an attribute of the reference model class and can include
        relationships.
        Example: if the reference model tablename is 'mymodel', then a string
        column 'name' will refer to 'mymodel.name', 'name.id' denotes on the
        other hand a relationship 'name' on 'mymodel' and will refer to the
        'id' attribute of the table mapped by 'mymodel.name'. The values of
        the dict on the other hand are string expressions in the form
        recognized by `binexpr`. E.g. '>=5', '["4", "5"]' ...
        For each condition mapped to a falsy value (e.g., None or empty
        string), the condition is discarded. See note [*] below for auto-added
        joins  from columns
    :param orderby: a list of string columns (same format
        as `conditions` keys), or a list of tuples where the first element is
        a string column, and the second is either "asc" (ascending) or "desc"
        (descending). In the first case, the order is "asc" by default. See
        note [*] below for auto-added joins from orderby columns
    """
    sess = dburl
    close_sess = False
    try:
        if isinstance(sess, str):
            sess = get_session(dburl)
            close_sess = True
        yield from exprquery(sess.query(Segment), conditions, orderby)
    finally:
        if close_sess:
            close_session(sess)


# ======================================================================================
# Docstring help for the Segment object as lists of (name, description) tuples to
# preserve desired order (for older python version compatibility).
# Any attribute or method of :class:`stream2segment.process.db.models.Segment` should be
# here below. Otherwise, the function raises (to check, you can always run
# `test_segment_help` in `tests/misc/test_notebook.py`).
# Attributes/methods that are supposed to be HIDDEN SHOULD be coupled with a FALSY
# description.
# Descriptions should be in plain text with optionally partial markdown support
# (*, **, ` and \n that you can use to force newlines and is converted to '<br>' in html)
# ======================================================================================

_SEGMENT_ATTRS = [  # as formatting can use *, **, `, nothing else
    ["id", "int: segment (unique) db id"],
    ["has_data", "bool: if the segment has waveform data saved (at least one "
                 "byte of data). Often necessary in segment selection: "
                 "e.g., to skip processing segments with no data, then:\n"
                 "has_data: 'true'"],
    ["event_distance_deg", "float: distance between the segment station and the "
                           "event, in degrees"],
    ["event_distance_km", "float: distance between the segment station and the "
                          "event, in km, assuming a perfectly spherical earth "
                          "with a radius of 6371 km"],
    ["start_time", "datetime.datetime: waveform start time"],
    ["arrival_time", "datetime.datetime: waveform arrival time (value between "
                     "'start_time' and 'end_time')"],
    ["end_time", "datetime.datetime: waveform end time"],
    ["request_start", "datetime.datetime: waveform requested start time"],
    ["request_end", "datetime.datetime: waveform requested end time"],
    ["duration_sec", "float: waveform data duration, in seconds"],
    ["missing_data_sec", "float: number of seconds of missing data, as ratio of "
                         "the requested time window. It might also be negative "
                         "(more data received than requested). Useful in segment "
                         "selection: e.g., if we requested 5 minutes of data and "
                         "we want to process segments with at least 4 minutes of "
                         "downloaded data, then: missing_data_sec: '< 60'"],
    ["missing_data_ratio", "float: portion of missing data, as ratio of the "
                           "requested time window. It might also be negative "
                           "(more data received than requested). Useful in "
                           "segment selection: e.g., to process segments whose "
                           "time window is at least 90% of the requested one: "
                           "missing_data_ratio: '< 0.1'"],
    ["sample_rate", "float: waveform sample rate. It might differ from the "
                    "segment channel sample_rate"],
    ["maxgap_numsamples", "float: maximum gap/overlap (G/O) found in the waveform, "
                          "in number of points. If\n"
                          "0: segment has no G/O\n"
                          ">=1: segment has gaps\n"
                          "<=-1: segment has overlaps.\n"
                          "Values in (-1, 1) are difficult to interpret: a rule "
                          "of thumb is to consider no G/O if values are within "
                          "-0.5 and 0.5. Useful in segment selection: e.g., to "
                          "process segments with no gaps/overlaps:\n"
                          "maxgap_numsamples: '(-0.5, 0.5)'"],
    ["seed_id", "str: the seed identifier in the typical format "
                "[Network].[Station].[Location].[Channel]. For segments with "
                "waveform data, `data_seed_id` (see below) might be faster to "
                "fetch."],
    ["data_seed_id", "str: same as 'segment.seed_id', but faster to get because "
                     "it reads the value stored in the waveform data. The "
                     "drawback is that this value is null for segments with no "
                     "waveform data"],
    ["has_class", "bool: tells if the segment has (at least one) class label "
                  "assigned"],
    ["data", "bytes: the waveform (raw) data. Used by `segment.stream()`"],
    ["queryauth", "bool: if the segment download required authentication "
                  "(data is restricted)"],
    ["download_code", "int: the segment download status. For advanced users. "
                      "Useful in segment selection. E.g., to process segments "
                      "with non malformed waveform data (readable as miniSEED):\n"
                      "has_data: 'true'\n"
                      "download_code: '!=-2'\n"
                      "(for details on all download codes, see Table 2 in "
                      "https://doi.org/10.1785/0220180314)"],
    # ["event", "object (attributes below)"],
    ["event.id", "int"],
    ["event.event_id", "str: the id returned by the web service or catalog"],
    ["event.time", "datetime.datetime"],
    ["event.latitude", "float"],
    ["event.longitude", "float"],
    ["event.depth_km", "float"],
    ["event.author", "str"],
    ["event.catalog", "str"],
    ["event.contributor", "str"],
    ["event.contributor_id", "str"],
    ["event.mag_type", "str"],
    ["event.magnitude", "float"],
    ["event.mag_author", "str"],
    ["event.event_location_name", "str"],
    ['event.event_type', 'str: the event type (e.g. "earthquake")'],
    # ["channel", "object (attributes below)"],
    ["channel.id", "int"],
    ["channel.location", "str"],
    ["channel.channel", "str"],
    ["channel.depth", "float"],
    ["channel.azimuth", "float"],
    ["channel.dip", "float"],
    ["channel.sensor_description", "str"],
    ["channel.scale", "float"],
    ["channel.scale_freq", "float"],
    ["channel.scale_units", "str"],
    ["channel.sample_rate", "float"],
    ["channel.band_code", "str: the first letter of channel.channel"],
    ["channel.instrument_code", "str: the second letter of channel.channel"],
    ["channel.orientation_code", "str: the third letter of channel.channel"],
    ["channel.band_instrument_code", "str: the first two letters of channel.channel"],
    # ["channel.station", "object: same as segment.station (see below)"],
    # ["station", "object (attributes below)"],
    ["station.id", "int"],
    ["station.network", "str: the station's network code, e.g. 'AZ'"],
    ["station.station", "str: the station code, e.g. 'NHZR'"],
    ["station.netsta_code", "str: the network + station code, concatenated with "
                            "the dot, e.g.: 'AZ.NHZR'"],
    ["station.latitude", "float"],
    ["station.longitude", "float"],
    ["station.elevation", "float"],
    ["station.site_name", "str"],
    ["station.start_time", "datetime.datetime"],
    ["station.end_time", "datetime.datetime"],
    ["station.has_inventory", "bool: tells if the segment's station inventory "
                              "has data saved (at least one byte of data). "
                              "Useful in segment selection. E.g., to process "
                              "only segments with inventory downloaded:\n"
                              "station.has_inventory: 'true'"],
    ["station.datacenter", "object (same as segment.datacenter, see below)"],
    # ["datacenter", "object (attributes below)"],
    ["datacenter.id", "int"],
    ["datacenter.station_url", "str"],
    ["datacenter.dataselect_url", "str"],
    ["datacenter.organization_name", "str"],
    # ["download", "object (attributes below): the download execution"],
    ["download.id", "int"],
    ["download.run_time", "datetime.datetime"],
    ["classes.id", "int: the id(s) of the class labels assigned to the segment"],
    ["classes.label", "int: the unique name(s) of the class labels assigned to "
                      "the segment"],
    ["classes.description", "int: the description(s) of the class labels "
                            "assigned to the segment"],
    # attrs mapped to None are ignored:
    ["station.inventory_xml", None], # bytes
    ["download.log", None], # str
    ["download.warnings", None], # int
    ["download.errors", None], # int
    ["download.config", None], # str
    ["download.program_version", None], # str
]

# 2nd element falsy: no doc (hidden)
_SEGMENT_METHODS = [
    ['stream', Segment.stream.__doc__],
    ['inventory', Segment.inventory.__doc__],
    ['dbsession', False],
    ['sds_path', Segment.sds_path.__doc__],
    ['get_siblings', False],
    ['siblings', False],
    ['add_classes', False],
    ['del_classes', False],
    ['edit_classes', False],
    ['set_classes', False]
]


def get_segment_help(format='html', maxwidth=70, **print_kwargs):
    """Return the :class:`Segment` help (attributes and methods) as string

    :param format: Not supported yet, only html allopwed
    """
    import re, inspect

    # Get queryable attributes (no relationships):
    qatts = list(attnames(Segment, qatt=True, rel=False, fkey=False))
    # add relationships:
    for relname, relmodel in get_related_models(Segment).items():
        qatts.extend('%s.%s' % (relname, _)
                     for _ in attnames(relmodel, qatt=True, rel=False, fkey=False))

    # check that all documentable attributes are considered and we did not forget any
    # (might happen when adding new hybrid props or methods):
    all_documented_attrs = set(_[0] for _ in _SEGMENT_ATTRS) | \
        set(attnames(Segment, fkey=True))
    undocumented_attrs = set(qatts) - all_documented_attrs
    assert not undocumented_attrs, "Missing doc for attribute(s): %s" % undocumented_attrs

    # Get segment methods:
    meths = []
    for meth in list(attnames(Segment, qatt=False, rel=False)):
        if meth[:1] == '_':
            continue
        func = getattr(Segment, meth)
        if not callable(func):
            continue
        meths.append(meth)
    all_documented_meths = set(_[0] for _ in _SEGMENT_METHODS)
    undocumented_attrs = set(meths) - all_documented_meths
    assert not undocumented_attrs, "Missing doc for method(s): %s" % undocumented_attrs

    table = []
    # attrs:
    table += [('**Selectable attributes**', '**Type and optional description**')]
    table += _SEGMENT_ATTRS
    # methods (some work more, add signature to the method name):
    table += [('**Methods**', '**Description**')]
    for meth, doc in _SEGMENT_METHODS:
        if doc:
            # create
            func = getattr(Segment, meth)
            meth += '(' + str(inspect.signature(func))[7:]  # remove "(self, "
            table.append((meth, doc))

    format = format or ''

    lines = []
    # one format supported for the moment (html):
    if format.lower() in ('html', 'htm'):
        code_re = re.compile('`(.*?)`')
        br_re = re.compile('\n')
        b_re = re.compile(r'\*\*(.*?)\*\*')
        i_re = re.compile(r'\*(.*?)\*')

        def convert(string):
            string = code_re.sub('<code>\\1</code>', string)
            string = b_re.sub('<b>\\1</b>', string)
            string = i_re.sub('<i>\\1</i>', string)
            string = br_re.sub('<br/>', string)
            return string

        if maxwidth and maxwidth > 0:
            lines.append('<table style="max-width:%dem;>' % maxwidth)
        else:
            lines.append('<table>')
        for attname, attval in table:
            if not attval:
                continue
            attname = '<span style="white-space: nowrap">{0}</span>'.\
                format(convert(attname))
            # replace markdown with html:
            attval = convert(attval)
            # notebook prints everything right aligned. So let's force left align:
            style = ' style="text-align:left"'
            lines.append('<tr {0}><td>{1}</td><td {0}>{2}</td></tr>'.format(style,
                                                                            attname,
                                                                            attval))
        lines.append('</table>')
    else:
        raise ValueError('format "%s" not supported' % format)

    return '\n'.join(lines)
