"""
Input validation module

:date: Feb 27, 2018

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
from future.utils import string_types
from stream2segment.io.db import database_exists, get_session


class BadParam(Exception):
    """Exception describing a bad input parameter. The purpose of this class is twofold:
    provide clear exception messages to the user focused on the input parameter to fix,
    and provide a formatting style similar to :class:`click.exceptions.BadParameter`, to
    harmonize output when invoking commands from the terminal
    """

    P_CONFLICT = "Conflicting names"
    P_MISSING = "Missing value for"
    P_INVALID = "Invalid value for"
    P_UNKNOWN = "No such option"

    def __init__(self, preamble, param_name_or_names,
                 message='', param_sep=' / ', param_quote='"'):
        """Initialize a BadParam object. The formatted output string `str(self)` will be:

        'Error: %(preamble) "%(param_name_or_names)": %(message)'
        'Error: "%(param_name_or_names)": %(error)"'
        'Error: "%(param_name_or_names)"'

        :param preamble: the optional message preamble, as string. For a predefined set
            of preambles, see this class global variables `P_*`, e.g.
            `BadParam.P_INVALID`, `BadParam.P_CONFLICT`
        :param param_name_or_names: the parameter name (string), or a list of
            parameter names (if the parameter supports several optional names)
        :param message: the original exception, or a string message.
        :param param_sep: the separator used for printing the parameter name(s).
            Default: " / "
        :param param_quote: the quote character used when printing the parameter name(s).
            Default : '"'
        """
        super(BadParam, self).__init__(message)
        self.preamble = preamble
        self.params = tuple(self._vectorize(param_name_or_names))
        self._param_sep = param_sep
        self._param_quote = param_quote

    @staticmethod
    def _vectorize(param_name_or_names):
        if not hasattr(param_name_or_names, '__iter__') or \
                isinstance(param_name_or_names, (bytes, str)):
            return [param_name_or_names]
        return param_name_or_names

    @property
    def message(self):
        return str(self.args[0] or '')

    @message.setter
    def message(self, message):
        # in superclasses, self.args is a tuple and stores as 1st arg. the error message
        args = list(self.args)
        args[0] = message
        self.args = tuple(args)

    def __str__(self):
        """String representation of this object"""
        msg_preamble = self.preamble
        if msg_preamble:
            msg_preamble += ' '

        p_name = self._param_sep.join("{0}{1}{0}".format(self._param_quote, _)
                                      for _ in self.params)

        err_msg = self.message
        if err_msg:
            # lower case first letter as err_msg will follow a ":"
            err_msg = ": " + err_msg[0].lower() + err_msg[1:]

        full_msg = msg_preamble + p_name + err_msg
        return "Error: %s" % full_msg.strip()


# THIS FUNCTION SHOULD BE CALLED FROM ALL LOW LEVEL FUNCTIONS BELOW
def validate_param(param_name_or_names, value, validation_func, *v_args, **v_kwargs):
    """Validate a parameter calling and returning the value of
    `validation_func(value, *v_args, **v_kwargs)`. Any exception raised from the
    validation function is wrapped and re-raised as :class:`InvalidValue` providing
    the given parameter name(s) in the exception message
    """
    try:
        return validation_func(value, *v_args, **v_kwargs)
    except Exception as exc:
        preamble = BadParam.P_INVALID
        if isinstance(exc, TypeError):  # change type. FIXME: replace?
            preamble = preamble.replace('Invalid value', 'Invalid type')
        raise BadParam(preamble, param_name_or_names, message=exc, param_sep=" / ")


# to make None a passable argument to the next function
# (See https://stackoverflow.com/a/14749388/3526777):
_VALUE_NOT_FOUND_ = object()


def pop_param(dic, name_or_names, default=_VALUE_NOT_FOUND_):
    """Pop a parameter value from `dic`, supporting multiple optional parameter names.
    Return the tuple `(name, value)`. Raise :class:`BadParam` if either 1. no name is
    found and no default is provided, or 2: multiple names are found

    :param dic: the dict of parameter names and values
    :param name_or_names: str or list/tuple of strings (multiple names). Names
        with a dot will perform recursive search within sub-dicts, e.g.
        'advanced_settings.param' will first get the sub-dict  'advanced_settings'
        (setting it to `{}` if not found), and then the value of the key 'param' from
        the sub-dict. If multiple names are found, class:`BadParam` is raised.
    :param default: if provided, this is the value returned if no name is found.
        If not provided, and no name is found in `dic`, :class:`BadParam` is raised
    """
    return _param_tuple(dic, name_or_names, default, pop=True)


def get_param(dic, name_or_names, default=_VALUE_NOT_FOUND_):
    """Get a parameter value from `dic`, supporting multiple optional parameter names.
    Return the tuple `(name, value)`. Raise :class:`BadParam` if either 1. no name is
    found and no default is provided, or 2: multiple names are found

    :param dic: the dict of parameter names and values
    :param name_or_names: str or list/tuple of strings (multiple names). Names
        with a dot will perform recursive search within sub-dicts, e.g.
        'advanced_settings.param' will first get the sub-dict  'advanced_settings'
        (setting it to `{}` if not found), and then the value of the key 'param' from
        the sub-dict. If multiple names are found, class:`BadParam` is raised.
    :param default: if provided, this is the value returned if no name is found.
        If not provided, and no name is found in `dic`, :class:`BadParam` is raised
    """
    return _param_tuple(dic, name_or_names, default, pop=False)


def _param_tuple(dic, name_or_names, default=_VALUE_NOT_FOUND_, pop=False):
    """private base function used by the public `get` and `pop`"""
    names = BadParam._vectorize(name_or_names)
    keyval = {}  # copy all param -> value mapping here

    for name in names:
        _dic = dic
        _names = name.split('.')

        i = 0
        while isinstance(_dic, dict) and _names[i] in _dic:
            _name, i = _names[i], i + 1  # set _name and increment i
            if i == len(_names):  # last item
                keyval[name] = _dic.pop(_name) if pop else _dic[_name]
                break
            else:
                _dic = _dic[_name]

    if not keyval:
        if default is not _VALUE_NOT_FOUND_:
            return names[0], default
        raise BadParam(BadParam.P_MISSING, names[0])

    if len(keyval) != 1:
        raise BadParam(BadParam.P_CONFLICT, keyval.keys())

    # return the only key:
    p_name = next(iter(keyval.keys()))

    return p_name, keyval[p_name]


##############################################################################
# Loading config functions. These functions should validate the whole input
# of our main routines (download, process, show) and call validate_param()
# with the given parameters and the low level validation functions above
##############################################################################


def _extract_segments_selection(config):
    """Return the dict in `config` denoting the selection of segment. Validators
    should all call this method so that the valid parameter names are implemented in
    one place and can be easily modified.

    :param config: the config `dict` (e.g. resulting from a YAML config file used for
        processing, or visualization)
    """
    return pop_param(config, ['segments_selection', 'segment_select'], {})[1]


#####################################################################################
# Low level validation functions.
# IMPORTANT: By convention, these functions should start with "valid_" and
# NOT BE CALLED DIRECTLY but wrapped within :func:`validate_param`
######################################################################################


def valid_session(dburl, for_process=False, scoped=False, **engine_kwargs):
    """Create an SQL-Alchemy session from dburl. Raises if `dburl` is
    not a string, or any SqlAlchemy exception if the session could not be
    created.

    IMPORTANT: This function is intended to be called through `validate_param`,
    so that if the database session could not be created, a meaningful message
    with the parameter name (usually, "dburl" from the cli) can be raised. Example:
    ```
    session = validate_param('dburl', <variable_name>, get_session, *args, **kwargs)
    ```
    will raise in case of failure an error message like:
    "Error: invalid value for "dburl": <message>"

    :param dburl: string denoting a database url (currently postgres and sqlite
        supported
    :param for_process: boolean (default: False) whether the session should be
        used for processing, i.e. the database is supposed to exist already and
        the `Segment` model has ObsPy method such as `Segment.stream()`
    :param scoped: boolean (False by default) if the session must be scoped
        session
    :param engine_kwargs: optional keyword argument values for the
        `create_engine` method. E.g., let's provide two engine arguments,
        `echo` and `connect_args`:
        ```
        get_session(dbpath, ..., echo=True, connect_args={'connect_timeout': 10})
        ```
        For info see:
        https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine.params.connect_args
    """
    if not isinstance(dburl, string_types):
        raise TypeError('string required, %s found' % str(type(dburl)))
    # import in function to speed up module imports from cli:
    # FIXME: single func!
    # if for_process:
    #     # important, rename otherwise conflicts with this function name:
    #     from stream2segment.process.db import get_session as sess_func
    # else:
    #    # important, rename otherwise conflicts with this function name:
    #    from stream2segment.io.db import get_session as sess_func

    exists = database_exists(dburl)
    # the only case when we don't care if the database exists is when
    # we have sqlite and we are downloading. Thus
    if not dburl.startswith('sqlite') or for_process:
        if not exists:
            dbname = dburl[dburl.rfind('/')+1:]
            if for_process:
                raise ValueError('Database "%s" does not exist. Provide an existing '
                                 'database' % dbname)
            else:
                raise ValueError('Database "%s" needs to be created first' % dbname)

    sess = get_session(dburl, scoped=scoped, **engine_kwargs)

    if not for_process:
        # Note: this creates the SCHEMA, not the database
        from stream2segment.download.db.models import Base
        Base.metadata.create_all(sess.get_bind())

    # assert that the database exist. The only exception is when we

    # Check if database exist, which should not always be done (e.g.
    # for_processing=True). Among other methods
    # (https://stackoverflow.com/a/3670000
    # https://stackoverflow.com/a/59736414) this seems to do what we need
    # (we might also not check if the tables length > 0 sometime):
    # sess.bind.engine.table_names()
    return sess
