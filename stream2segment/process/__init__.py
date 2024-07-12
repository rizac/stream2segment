from contextlib import contextmanager

from sqlalchemy import orm, func
import numpy as np

from stream2segment.process.db import get_session
from stream2segment.io.db import close_session
from stream2segment.process.db.models import (Segment, Event, Station, Channel,
                                              DataCenter, Download, Class, WebService)
from stream2segment.process.db.sqlevalexpr import exprquery
from stream2segment.process.main import (process as map, imap, SkipSegment,  # noqa
                                         get_default_segments_selection,
                                         create_processing_env as _create_processing_env)
from stream2segment.process.funclib import traces


# legacy code, allow map to be imported as `process`:
process = map


def get_db_items(db, item_type, conditions, *, load_only=None, defer=None, orderby=None):
    """Return a selection of items from the given database. `item_type` is the Python
    class representing the requested db table (e.g., Segment, Event), each yielded item
    is a class instance representing a table row matching the given selection conditions.
    The yielded objects are simple Python objects where the column values are accessed
    via the object atttributes. E.g.
    ```
        from stream2segment.process import get_db_items, Segment, Event

        for event in get_db_items('..', Event, { 'magnitude' : '>=5' }, ...):
            mag = event.magnitude

        for segment in get_db_items('..', Segment, { 'has_valid_data': 'true' }, ...):
            obspy_trace = segment.stream()
    ```

    :param db: the database URL, as string, or a `session` object already created from
        an given URL (see :func:`get_session`). URLs must be given in this format:
        https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls.
        NOTE: if `db` is a string, a db session is opened and closed just before this
        function returns: afterwards, attributes returning related db objects (e.g.,
        `event`, `station`) might not be accessible anymore
    :param conditions: a dict of Segment attribute names (string) mapped to
        an expression, *also as string* (so values must be quoted). Example:
        ```
        {
            'id' : "<6",
            'has_valid_data': 'true'
        }
        ```
    :param load_only: str or list of Segment attribute(s) or attribute name(s)
        denoting database Table columns, that need to be loaded. If None, all
        attributes are loaded, but this might be inefficient for huge queries.
        If specified, the Segment attributes can be accessed anyway, but will
        cost a new db query each time.
        Example: if you want only segment id, use `load_only='id'`
    :param defer: same as `load_only`, but specifies the columns NOT to load.
        Example: if you want to avoid loading the waveform data because youn work
        with metadata only, use `defer='data'`
    :param orderby: a list of string columns (same format
        as `conditions` keys), or a list of tuples where the first element is
        a string column, and the second is either "asc" (ascending) or "desc"
        (descending). In the first case, the order is "asc" by default
    """
    sess = get_session(db) if isinstance(db, str) else db
    try:
        qry = exprquery(sess.query(item_type), conditions, orderby)
        # As per-doc, `item_type` must be a model (Segment). Legacy / internal code is
        # still allowed to pass InstrumentedAttributes instead (e.g. Segment.id). So
        # let's assure that `item_type` is from now on a model instance:
        item_type = qry.column_descriptions[0]['entity']
        # (Note: as of v <=2.0, the parent model of an InstrumentedAttribute attr can be
        # also obtained via: attr.parent.entity or attr.parent.class_, )
        if load_only:
            if not isinstance(load_only, list):
                load_only = [load_only]
            # sqlalchemy <2 needs model attributes, not strings:
            load_only = [getattr(item_type, c) if isinstance(c, str) else c
                         for c in load_only]
            qry = qry.options(orm.load_only(*load_only))
        if defer:
            if not isinstance(defer, list):
                defer = [defer]
            # sqlalchemy <2 needs model attributes, not strings:
            defer = [getattr(item_type, c) if isinstance(c, str) else c
                     for c in defer]
            qry = qry.options(orm.defer(*defer))
        if defer and load_only and set(defer) & set(load_only):
            raise ValueError('You cannot supply the same column(s) in '
                             'both `load_only` and `defer`')
        yield from qry
    finally:
        if sess is not db:  # we created the session here, close it before returning
            close_session(sess)


def get_segments(db, conditions, *, load_only=None, defer=None, orderby=None):
    """Legacy code: yield Segments from the given database. See get_db_items for
    details"""
    yield from get_db_items(db, Segment, conditions, load_only=load_only, defer=defer,
                            orderby=orderby)


def get_db_items_count(db, item_type, conditions):
    """Yield Segments from the given database. See get_db_items for details"""
    # legacy code
    return get_db_items(db, func.count(item_type), conditions).scalar()  # noqa


@contextmanager
def terminal_environment(progressbar_total_length=0,
                         capture_external_stderr_printout=True,
                         warnings_filter=None):
    """Set up an environment for the execution of medium-to-long tasks on the terminal.
    This method should be called once before starting your tasks on the main process
    (i.e., if you use Python multiprocessing do NOT call this function in each child
    process). The created environment assures that:
    1. Undesired printouts (to stderr) that can easily pollute the terminal are captured
    and not shown. Note that also external non-Python libraries (as used by some ObsPy
    routines) are also captured
    2. A progressbar can be initialized, incremented and decremented, and will be updated
    on the terminal, showing also computed estimated time available. Use it when you have
    code running in loop / iterables, and the total loops count is known beforehand
    3. Undesired Python warnings can be filtered. By default, this is disabled (None)
    because if you are using Python multiprocessing, you should probably filter warnings
    on each child process. For possible string values, see:
    https://docs.python.org/3/library/warnings.html#the-warnings-filter

    Example:
    ```
    with terminal_environment(progressbar_total_length) as env:
        for ...:  # custom for loop
            # ... execute your code ...and update progressbar:
            env.increment_progressbar(1)

    # if you only need to capture external printouts to stderr:
    with terminal_environment():
        ... execute your code ...
    ```
    """
    with _create_processing_env(length=progressbar_total_length,
                                redirect_stderr=capture_external_stderr_printout,
                                warnings_filter=warnings_filter) as pbar:
        yield _TerminalEnv(pbar)


class _TerminalEnv:
    """Dummy private-like class for progressbar"""
    def __init__(self, pbar):
        self.pbar = pbar

    def increment_progressbar(self, number: int):
        self.pbar.update(number)


def get_classlabels(db):
    """Yields the Python objects representing each class label stored on the given db.
    The object main attributes are `label`, `description`, '`id` and `segments`, which
    can be used to yield the Segments assigned to the given class label.

    :param db: the database URL, as string, or a `session` object already created from
        an given URL (see :func:`get_session`). URLs must be given in this format:
        https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls.
        NOTE: if `db` is a string, a db session is opened and closed just before this
        function returns: afterwards, attributes returning related db objects (e.g.,
        `segments`) might not be accessible anymore
    """
    sess = get_session(db) if isinstance(db, str) else db
    try:
        yield from sess.query(Class)
    finally:
        if sess is not db:  # we created the session here, close it before returning
            sess.close()


def load_ints_from_txt(path, sep='\n', as_list=True):
    """Return a list of integers from a given file in text format.
    See also `save_ints_to_txt`

    :param path: string denoting the file path
    :param sep: the separator (default: "\n", i.e., one integer per line). No
        separator (e.g. "") is not allowed and will be replaced with "\n"
    :param as_list: boolean (default True). Return a Python list (otherwise numpy int
        array)
    """
    # note: the method below is faster than reading and casting line by line with Python,
    # and MUCH faster than np.loadtxt
    with open(path, 'r') as _:
        lst = np.fromstring(_.read(), sep=sep or '\n', dtype=int)
        if as_list:
            return lst.tolist()
        return lst


def save_ints_to_txt(path, integers, sep="\n"):
    """Return a list of integers from a given file in text format.
    See also `load_ints_from_txt`

    :param path: string denoting the file path
    :param integers: numpy int array or Python list of integers to be saved.
    :param sep: the number separator. Default to "\n" (one integer per line). No
        separator (e.g. "") is not allowed and will be replaced with "\n"
    """
    np.savetxt(path, np.asarray(integers, dtype=int),  # noqa
               delimiter=sep or "\n", newline=sep or "\n", fmt='%i')


def get_segment_help(format='html', maxwidth=79, **print_kwargs):
    """Return the :class:`Segment` help (attributes and methods) as string

    :param format: Not supported yet, only html allopwed
    """
    import re
    import inspect
    import textwrap
    from itertools import chain
    from stream2segment.io.db.inspection import attnames

    # ==================================================================================
    # Set Segment attributes documentation as list of `[attname, description]` items.
    # Falsy descriptions (None, '', False) mean the relative attribute will be hidden
    # from the doc. Otherwise, a description should start always with the Python type
    # (see below). Recognized special characters are: "\n" (newline), * (italic) and **
    # (bold)
    # ==================================================================================

    _SELECTABLE_ATTRS = [
        ["id", "int: segment (unique) db id"],
        ["has_data", "bool: if the segment waveform data is not empty, i.e. it has "
                     "at least 1 byte of data saved. This parameter or `has_valid_data` "
                     "are often necessary in segment selection, e.g.: \n"
                     "has_data: 'true'\n"
                     "Empty segments are those whose server did not return any data "
                     "and are stored anyway for collecting stats and allow to "
                     "customize what should be re-downloaded in further attempts"],
        ["has_valid_data", "bool: if the segment waveform data is not empty and "
                           "could be successfully read as miniSEED during "
                           "download. Often necessary in segment selection, e.g.: \n"
                           "has_valid_data: 'true'"],
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
        ["classlabels_count", "int: the number of class labels assigned "
                              "to this segment"],
        ["data", "bytes: the waveform (raw) data. Used by `segment.stream()`"],
        ["queryauth", "bool: if the segment download required authentication "
                      "(data is restricted)"],
        ["download_code", None],  # <- IGNORED
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
        # attrs mapped to None are ignored:
        ["classes.id", None],  # "int: the id(s) of the class labels assigned to
                               # the segment"],
        ["classes.label", None],  # "int: the unique name(s) of the class labels
                                  # assigned to  the segment"],
        ["classes.description", None],  # "int: the description(s) of the class labels
                                        #  assigned to the segment"],
        ["station.stationxml", None],  # bytes
        ["download.log", None],  # str
        ["download.warnings", None],  # int
        ["download.errors", None],  # int
        ["download.config", None],  # str
        ["download.program_version", None],  # str
    ]

    # Prepare a list of strings (aname, socstring) tuples:
    table = []

    # Append selectable attributes:
    table += [('**Selectable attributes**', '**Type and optional description**')]
    table += [_ for _ in _SELECTABLE_ATTRS if _[1]]

    # Append Standard methods/ attributes:
    table += [('**Standard attributes or methods**', '**Description**')]
    # Define the main attributes/methods to be shown first (those not listed
    # below will simply be shown next):
    _MAIN_ATTS = ('stream', 'inventory', 'url', 'sds_path', 'dbsession', 'classlabels')
    # Before looping through the Segment class, define what to skip:
    skip_attrs = set(attnames(Segment)) | {'metadata'}  # <- reserved att names
    signatures = {}
    # Now loop:
    for aname in chain(_MAIN_ATTS, dir(Segment)):
        if aname[:1] == '_' or aname in skip_attrs:
            continue
        # if aname is in _MAIN_ATTRS, avoid displaying it twice later
        # when handling dir(Segment):
        skip_attrs.add(aname)
        try:
            att = getattr(Segment, aname)
            docstr = (att.__doc__ or '').strip()
            if not docstr:
                continue
            # Append the method/ attribute for the moment:
            table.append((aname, docstr))
            # Get `att` signature, if method (<=> is callable):
            if callable(att):
                sig_str = '()'
                sig = inspect.signature(att)
                if len(sig.parameters) > 1:
                    sig_str = "(" + str(sig)[7:]  # remove "(self, "
                # sig_str might have special characters (*, **) which should
                # not be confused with their markdown meaning (italic, bold)
                # this is why we do not add `sig_str` to `aname` but keep all
                # signatures in a separate dict for the moment:
                signatures[aname] = sig_str
        except Exception:  # getattr might fail (e.g. for hybrid properties with no expr)
            pass

    format = format or ''

    lines = []
    # one format supported for the moment (html):
    if format.lower() in ('html', 'htm'):
        pre_code_re = re.compile(r'\n*```\n*(.*?)\n*```\n*', re.DOTALL)
        code_re = re.compile('`(.*?)`')
        br_re = re.compile('\n')
        b_re = re.compile(r'\*\*(.+?)\*\*')
        i_re = re.compile(r'\*(.+?)\*')
        param_re = re.compile(r'\:param (\w+)\:', re.MULTILINE)
        link_re = re.compile(r'(https?:\/\/[\w\~\-\.\?\&\=\%\/\#]+)',
                             re.IGNORECASE | re.MULTILINE)
        raises_re = re.compile(r'\:raises?\:')

        def convert(string):
            search = pre_code_re.search(string)
            if search and search.group(1):
                string = convert(string[:search.start()]) + \
                    '<p><pre><code>' +\
                    textwrap.dedent(search.group(1)) + '</code></pre></p>' +\
                    convert(string[search.end():])
            else:
                # string = pre_code_re.sub('<code>\\1</code>', string)
                string = code_re.sub('<code>\\1</code>', string)
                string = b_re.sub('<b>\\1</b>', string)
                string = i_re.sub('<i>\\1</i>', string)
                string = br_re.sub('<br/>', string)
                string = param_re.sub('<i>Parameter</i> <b>\\1</b>:', string)
                string = link_re.sub('<a target="_blank" href="\\1">\\1</a>', string)
                string = raises_re.sub('<i>Raises</i>', string)
            return string

        if maxwidth and maxwidth > 0:
            lines.append('<table class="s2s-segment-summary-table" '
                         'style="max-width:%dem;">' % maxwidth)
        else:
            lines.append('<table>')
        for aname, aval in table:
            if not aval:
                continue  # safety check (shoulw not happen)
            attname = '<span>{0}</span>'.format(convert(aname))
            if aname in signatures:
                attname = attname.replace(aname, aname+signatures[aname])
            # replace markdown with html:
            attval = convert(aval)
            # notebook prints everything right aligned. So let's force left align:
            # style = 'style="text-align:left"'
            style_td = 'style="text-align:left; border-width:1px; border-style:solid"'
            lines.append('<tr><td {0}>{1}</td><td {0}>{2}</td></tr>'.format(style_td,
                                                                            attname,
                                                                            attval))
        lines.append('</table>')
    else:
        raise ValueError('format "%s" not supported' % format)

    return '\n'.join(lines)
