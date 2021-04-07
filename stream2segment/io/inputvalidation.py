"""
Input validation module

:date: Feb 27, 2018

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
from future.utils import string_types
from stream2segment.io.db import database_exists, get_session


class BadParam(Exception):
    """Exception describing a bad input parameter. The purpose of this class is twofold:
    provide clear exception messages to the user with info on the input parameter to fix,
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
        p_name = self._param_sep.join("{0}{1}{0}".format(self._param_quote, _)
                                      for _ in self.params)
        if msg_preamble and p_name:
            msg_preamble += ' '

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
    validation function is wrapped and re-raised as :class:`BadParam` with the
    given parameter name(s) in the exception message

    :paraqm param_name_or_names: str or list of strings denoting the parameter name(s)
        a list of strings denotes parameters with optional names
    :param value: the parameter value to validate
    :param validation_func: the validation function whose first argument must be
        `value`
    :param v_args: additional positional arguments to be passed to `validation_func`
    :param v_kwargs: additional keyword arguments to be passed to `validation_func`
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
    names = BadParam._vectorize(name_or_names)  # noqa
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


#####################################################################################
# Low level validation functions.
# IMPORTANT: By convention, these functions should start with "valid_" and
# NOT BE CALLED DIRECTLY but wrapped within :func:`validate_param`
######################################################################################


def valid_between(val, min, max, include_min=True, include_max=True, pass_if_none=True):
    if val is None:
        if pass_if_none:
            return val
        raise ValueError('value is None/null')

    is_ok = min is None or val > min or (include_min and val >= min)
    if not is_ok:
        raise ValueError('%s must be %s %s' %
                         (str(val), '>=' if include_min else '>', str(min)))

    is_ok = max is None or val < max or (include_max and val <= max)
    if not is_ok:
        raise ValueError('%s must be %s %s' %
                         (str(val), '<=' if include_max else '<', str(max)))
    return val
