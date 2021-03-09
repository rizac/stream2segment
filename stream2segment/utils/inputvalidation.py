"""
Module with utilities for checking / parsing / setting input arguments from
the command line interface (cli), e.g. download and process.

:date: Feb 27, 2018

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
import inspect
import os
# import sys
# import re
# import sys
from datetime import datetime, timedelta
# from itertools import chain
# import importlib

from future.utils import string_types
# from stream2segment.process import SkipSegment

from stream2segment.utils.resources import yaml_load, get_ttable_fpath, \
    get_templates_fpath, normalizedpath
from stream2segment.utils import strptime, load_source
from stream2segment.traveltimes.ttloader import TTTable
from stream2segment.io.db.models import Fdsnws
from stream2segment.download.utils import Authorizer, EVENTWS_MAPPING,\
    EVENTWS_SAFE_PARAMS


class BadParam(Exception):
    """Exception describing a bad configuration parameter, as
    :class:`click.exceptions.BadParameter`, provides similar messages when
    output in the terminal but it can be used outside a command line interface
    environment
    """

    P_CONFLICT = "Conflicting names"
    P_MISSING = "Missing value for"
    P_INVALID = "Invalid value for"
    P_UNKNOWN = "No such option"

    def __init__(self, preamble, param_name_or_names,
                 message='', param_sep=' / ', param_quote='"'):
        """Initialize a BadParam object. Formats the message according to
        the given parameters. The formatted output string, depending on the
        value of the arguments will be:

        'Error: %(msg_preamble) "%(param_name_or_names)": %(error)'
        'Error: "%(param_name_or_names)": %(error)"'
        'Error: "%(param_name_or_names)"'

        :param preamble: the optional message preamble, as string. If not
            provided (empty string by default), it will default to a string
            according to `error` type, e.g. KeyError => 'Missing value for'
        :param param_name_or_names: the parameter name (string), or a list of
            parameter names (if the parameter supports several optional names)
        :param message: the original exception, or a string message.

        """
        # if not preamble:
        #     preamble = "Invalid value for"
        #     if isinstance(error, KeyError):
        #         preamble = "Missing value for"
        #         error = ''
        #     elif isinstance(error, TypeError):
        #         preamble = "Invalid type for"
        #
        # if isinstance(error, BadParam):
        #     err_msg = error.message
        # else:
        #     err_msg = str(error)

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

    # @property
    # def _params2str(self):
    #     if isinstance(self.param_name, (list, tuple)):
    #         p_name = " / ".join('"%s"' % p for p in self.param_name)
    #     else:
    #         p_name = '"' + str(self.param_name) + '"'
    #     return p_name

    # @property
    # def message(self):
    #     # msg = '%s' if not self.msg_preamble else \
    #     #     self.msg_preamble.strip() + " %s"
    #     # Access the parent message (works in py2 and 3):
    #     if isinstance(self.param_name, (list, tuple)):
    #         p_name = " / ".join('"%s"' % p for p in self.param_name)[1:-1]
    #     else:
    #         p_name = str(self.param_name)
    #     p_name = '"' + p_name + '"'
    #     msg_preamble = self.msg_preamble
    #     if msg_preamble:
    #         msg_preamble += ' '
    #     err_msg = str(self.args[0] or '')  # noqa
    #     if err_msg:
    #         # lower case first letter as err_msg will follow a ":"
    #         err_msg = ": " + err_msg[0].lower() + err_msg[1:]
    #     ret = msg_preamble + p_name + err_msg
    #     return ret[0:1].upper() + ret[1:]

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


# class ConflictingNames(BadParam):
#
#     def __init__(self, param_name_or_names):
#         """param_names are synonyms: NO"""
#         super().__init__("Conflicting names", param_name_or_names, param_sep=" and ")
#
#
# class MissingValue(BadParam):
#
#     def __init__(self, param_name_or_names):
#         """param_names are synonyms: NO"""
#         super().__init__("Missing value(s) for", param_name_or_names, param_sep=" and ")
#
#
# class UnknownName(BadParam):
#
#     def __init__(self, param_name_or_names):
#         """param_names are synonyms: NO"""
#         super().__init__("No such option(s):", param_name_or_names, param_sep=" and ")
#
#
# class InvalidValue(BadParam):
#
#     def __init__(self, param_name, error):
#         """param_names are synonyms: YES"""
#         super().__init__("Invalid value for", param_name, message=error)


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
        if isinstance(exc, TypeError):
            preamble = preamble.replace('Invalid value', 'Invalid type')
        raise BadParam(preamble, param_name_or_names, message=exc, param_sep=" / ")


# to make None a passable argument to the next function
# (See https://stackoverflow.com/a/14749388/3526777):
_VALUE_NOT_FOUND_ = object()


def pop_param(dic, name_or_names, default=_VALUE_NOT_FOUND_):
    return _param_tuple(dic, name_or_names, default, pop=True)


def get_param(dic, name_or_names, default=_VALUE_NOT_FOUND_):
    return _param_tuple(dic, name_or_names, default, pop=False)


def _param_tuple(dic, name_or_names, default=_VALUE_NOT_FOUND_, pop=False):
    """Similar to `dic.pop` with optional (multi) keys. Raises
    :class:`BadParam` in case no default is provided and no key is found,
    or multiple keys are found

    :param dic: the source dict
    :param name_or_names: str or list/tuple of strings (multiple names). Names
        with a dot will perform recursive search within sub-dicts, e.g.
        'advanced_settings.param' will first get the sub-dict  'advanced_settings'
        (defaulting to `{}` if not found), and then the value of the key 'param' from
        the sub-dict.
    :param default: if provided and not None (the default), then
        this is the value returned if no name is found. If not provided, and no
        name is found in `dic`, :class:`BadParam` is raised
    """
    names = BadParam._vectorize(name_or_names)
    keyval = {}

    for name in names:  # recursive search
        _dic = dic
        _names = name.split('.')
        try:
            for _name in _names[:-1]:
                _dic = _dic[_name]
        except (AttributeError, TypeError, KeyError):
            _dic = {}
        # for _name in _names[:-1]:
        #     if not isinstance(_dic, dict):
        #         _dic = {}
        #         break
        #     _dic = _dic.get(_name, {})
        if _names[-1] in _dic:
            keyval[name] = _dic.pop(_names[-1]) if pop else _dic[_names[-1]]

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


def load_config_for_download(config, validate, **param_overrides):
    """Load download arguments from the given config (yaml file or dict) after
    parsing and checking some of the dict keys.

    :return: a dict loaded from the given `config` and with parsed arguments
        (dict keys)

    Raises `BadParam` in case of parsing errors, missing arguments,
    conflicts and so on
    """
    config_dict = validate_param("config", config, yaml_load, **param_overrides)

    if not validate:
        return config_dict

    # few variables:
    configfile = None
    if isinstance(config, string_types) and os.path.isfile(config):
        configfile = config

    # =====================
    # Parameters validation
    # =====================

    # put validated params into a new dict, which will be eventually returned:
    old_config, new_config = config_dict, {}
    validated_params = set()

    # validate dataws FIRST because it is used by other params later
    pname, pval = pop_param(old_config, 'dataws')
    validated_params.add(pname)
    if isinstance(pval, string_types):  # backward compatibility
        pval = [pval]
    dataws = validate_param(pname, pval,
                            lambda urls: [valid_fdsn(url, is_eventws=False)
                                          for url in urls])
    new_config[pname] = dataws

    pname, pval = pop_param(old_config, 'update_metadata')
    validated_params.add(pname)
    new_config[pname] = validate_param(pname, pval, valid_updatemetadata_param)

    pname, pval = pop_param(old_config, 'eventws')
    validated_params.add(pname)
    new_config[pname] = validate_param(pname, pval, valid_fdsn,
                                       is_eventws=True, configfile=configfile)

    pname, pval = pop_param(old_config, 'search_radius')
    validated_params.add(pname)
    new_config[pname] = validate_param(pname, pval, valid_search_radius)

    pname, pval = pop_param(old_config, 'min_sample_rate', default=0)
    validated_params.add(pname)
    new_config[pname] = validate_param(pname, pval, int)

    # parameters whose validation changes completely their type and should
    # returned separately from the new confg dict:

    pname, pval = pop_param(old_config, 'restricted_data')
    validated_params.add(pname)
    authorizer = validate_param(pname, pval, valid_authorizer, dataws, configfile)

    pname, pval = pop_param(old_config, 'dburl')
    validated_params.add(pname)
    session = validate_param(pname, pval, valid_session)

    pname, pval = pop_param(old_config, 'traveltimes_model')
    validated_params.add(pname)
    tt_table = validate_param(pname, pval,valid_tt_table)

    # parameters with multiple allowed names (use get_param_tuple to get which
    # param name is implemented among the allowed ones)

    pnames = ('starttime', 'start')
    validated_params.update(pnames)
    pname, pval = pop_param(old_config, pnames)
    new_config[pnames[0]] = validate_param(pname, pval, valid_date)

    pnames = ('endtime', 'end')
    validated_params.update(pnames)
    pname, pval = pop_param(old_config, pnames)
    new_config[pnames[0]] = validate_param(pname, pval, valid_date)

    pnames = ('network', 'net', 'networks')
    validated_params.update(pnames)
    pname, pval = pop_param(old_config, pnames, default=[])
    new_config[pnames[0]] = validate_param(pname, pval, valid_nslc)

    pnames = ('station', 'sta', 'stations')
    validated_params.update(pnames)
    pname, pval = pop_param(old_config, pnames, default=[])
    new_config[pnames[0]] = validate_param(pname, pval, valid_nslc)

    pnames = ('location', 'loc', 'locations')
    validated_params.update(pnames)
    pname, pval = pop_param(old_config, pnames, default=[])
    new_config[pnames[0]] = validate_param(pname, pval, valid_nslc)

    pnames = ('channel', 'cha', 'channels')
    validated_params.update(pnames)
    pname, pval = pop_param(old_config, pnames, default=[])
    new_config[pnames[0]] = validate_param(pname, pval, valid_nslc)

    # validate advanced_settings:
    pname = 'advanced_settings'
    validated_params.add(pname)
    # Call get_param with no default just for raising if param is missing:
    new_config[pname] = pop_param(old_config, pname)[1]
    _validate_download_advanced_settings(new_config, pname)

    # Validate eventws (event web service) params. These parameters can be supplied
    # in the main config but also in the eventws_params dict, which was formerly
    # named eventws_query_args:

    pnames = ('eventws_params', 'eventws_query_args')
    validated_params.update(pnames)
    # get eventws_param dict:
    pname, evt_params = get_param(old_config, pnames, default=None)
    # validate it (null is allowed and should be converted to {}):
    evt_params = validate_param(pnames, evt_params or {}, valid_type, {})
    if evt_params:
        # evt_params is not empty, validate some parameters:
        evt_params.update(_pop_event_params(old_config, pname))
    # Now remove evt_params dict from the old config:
    pop_param(old_config, pnames, default=None)
    # Ok, now do the same above on the main config:
    evt_params2 = _pop_event_params(old_config)
    # Now merge (but check conflicts beforehand):
    _conflicts = set(evt_params) & set(evt_params2)
    if _conflicts:
        # Issue a general warning (printing all conflicting params might be too verbose,
        # and we should print them as they were input,e.g. minlatitude or minlat?)
        raise BadParam(BadParam.P_CONFLICT.replace('names', 'name(s)'), _conflicts,
                       message='parameter(s) can be provided globally or in "%s", '
                               'not in both' % pname)
    evt_params.update(evt_params2)  # merge
    # now put evt_params into new_config:
    new_config[pnames[0]] = evt_params

    # evt_params = {}
    # pnames = ('eventws_params', 'eventws_query_args')
    # validated_params.update(pnames)
    # pname, pval = get_param(old_config, pnames, default=None)
    # if pval:
    #     validate_param(pname, pval, valid_type, {})
    #     evt_params = _pop_event_params(old_config, pname)
    # else:
    #     # remove param from old config:
    #     pop_param(old_config, pname, None)
    # # now pop all event params from the "outer" config_dict:
    # evt_params2 = _pop_event_params(old_config)
    # # and merge (but check conflicts beforehand):
    # _conflicts = set(evt_params) & set(evt_params2)
    # if _conflicts:
    #     # raise just the first conflict:
    #     raise BadParam(BadParam.P_CONFLICT, _conflicts,
    #                    message="check %s" % pname)
    # evt_params.update(evt_params2)  # merge
    # # now put evt_params into new_config:
    # new_config[pnames[0]] = evt_params

    # =========================================================
    # Done with parameter validation. Just perform final checks
    # =========================================================

    # load original config (default in this package) to perform some checks:
    orig_config = yaml_load(get_templates_fpath("download.yaml"))

    unknown_keys = set(old_config) - set(orig_config)
    if unknown_keys:
        raise BadParam(BadParam.P_UNKNOWN, unknown_keys)

    # Now check for params supplied here NOT in the default config, and supplied
    # here but with different type in the original config
    validated_params.update(old_config)
    for pname in list(old_config.keys()):
        pval = old_config.pop(pname)
        new_config[pname] = validate_param(pname, pval,
                                           valid_type, orig_config[pname])

    # And finally, check for params in the default config not supplied here:
    missing_keys = set(orig_config) - validated_params - set(EVENTWS_SAFE_PARAMS)
    if missing_keys:
        raise BadParam(BadParam.P_MISSING, missing_keys)

    return new_config, session, authorizer, tt_table


def _pop_event_params(evt_params_dict, prefix=None):
    """pops event params and returns a new dict FIXME: doc"""
    # define first default event params in order to avoid typos
    def_evt_params = EVENTWS_SAFE_PARAMS

    if prefix:
        def_evt_params = [prefix + '.' + _ for _ in def_evt_params]

    # returned dict:
    evt_params2 = {}

    pnames = def_evt_params[:2]  # ['minlatitude', 'minlat']
    pname, pval = pop_param(evt_params_dict, pnames, None)
    if pval is not None:
        new_pname = pnames[0].split('.')[-1]  # remove prefix, if any
        evt_params2[new_pname] = validate_param(pname, pval,
                                                valid_between, -90.0, 90.0)

    pnames = def_evt_params[2:4]  # ['maxlatitude', 'maxlat'],
    pname, pval = pop_param(evt_params_dict, pnames, None)
    if pval is not None:
        new_pname = pnames[0].split('.')[-1]  # remove prefix, if any
        evt_params2[new_pname] = validate_param(pname, pval,
                                                valid_between, -90.0, 90.0)

    pnames = def_evt_params[4:6]  # ['minlongitude', 'minlon'],
    pname, pval = pop_param(evt_params_dict, pnames, None)
    if pval is not None:
        new_pname = pnames[0].split('.')[-1]  # remove prefix, if any
        evt_params2[new_pname] = validate_param(pname, pval,
                                                valid_between, -180.0, 180.0)

    pnames = def_evt_params[6:8]  # ['maxlongitude', 'maxlon'],
    pname, pval = pop_param(evt_params_dict, pnames, None)
    if pval is not None:
        new_pname = pnames[0].split('.')[-1]  # remove prefix, if any
        evt_params2[new_pname] = validate_param(pname, pval,
                                                valid_between, -180.0, 180.0)

    pnames = def_evt_params[8:10]  # ['minmagnitude', 'minmag'],
    pname, pval = pop_param(evt_params_dict, pnames, None)
    if pval is not None:
        newp_name = pnames[0].split('.')[-1]  # remove prefix, if any
        evt_params2[newp_name] = validate_param(pname, pval, float)

    pnames = def_evt_params[10:12]  # ['maxmagnitude', 'maxmag'],
    pname, pval = pop_param(evt_params_dict, pnames, None)
    if pval is not None:
        newp_name = pnames[0].split('.')[-1]  # remove prefix, if any
        evt_params2[newp_name] = validate_param(pname, pval, float)

    pnames = def_evt_params[12:13]  # ['mindepth'],
    pname, pval = pop_param(evt_params_dict, pnames, None)
    if pval is not None:
        newp_name = pnames[0].split('.')[-1]  # remove prefix, if any
        evt_params2[newp_name] = validate_param(pname, pval, float)

    pnames = def_evt_params[13:14]  # ['maxdepth'],
    pname, pval = pop_param(evt_params_dict, pnames, None)
    if pval is not None:
        newp_name = pnames[0].split('.')[-1]  # remove prefix, if any
        evt_params2[newp_name] = validate_param(pname, pval, float)

    return evt_params2


def _validate_download_advanced_settings(config, adv_settings_key):
    """FIXME: write doc, validates keys in place"""
    adv_settings_dict = config[adv_settings_key]
    prefix = adv_settings_key + '.'  # 'advanced_settings'

    pname = 'download_blocksize'
    pval = validate_param(prefix + pname, pop_param(config, prefix + pname)[1],
                          valid_type, int)
    adv_settings_dict[pname] = pval if pval > 0 else -1

    pname = 'db_buf_size'
    adv_settings_dict[pname] = validate_param(prefix + pname,
                                              pop_param(config, prefix + pname)[1],
                                              valid_between, 1, None,
                                              pass_if_none=False)

    pnames = [prefix + _ for _ in ('max_concurrent_downloads', 'max_thread_workers')]
    pname, pval = pop_param(config, pnames)
    if pname == pnames[1]:
        # When max_thread_workers<0, it defaulted to None:
        pval = validate_param(pname, pval, valid_type, int, None)
        if pval <= 0:
            pval = None
    else:
        pval = validate_param(pname, pval, valid_between, 1, None, pass_if_none=True)
    adv_settings_dict[pnames[0].split('.')[-1]] = pval

    return adv_settings_dict
    # get the whole advanced_settings dict, and update with the validated params:
    # adv_settings = pop_param(config, prefix[:-1])[1]
    # adv_settings.update(validated_params)
    # return adv_settings


# def _pop_download_advanced_settings(config, adv_settings_key):
#     validated_params = {}
#     prefix = adv_settings_key + '.'  # 'advanced_settings'
#
#     # Call get_param with no default just for raising if param is missing:
#     get_param(config, adv_settings_key)
#
#     pname = 'download_blocksize'
#     pval = validate_param(prefix + pname, pop_param(config, prefix + pname)[1],
#                           valid_type, int)
#     validated_params[pname] = pval if pval > 0 else -1
#
#     pname = 'db_buf_size'
#     validated_params[pname] = validate_param(prefix + pname,
#                                              pop_param(config, prefix + pname)[1],
#                                              valid_between, 1, None,
#                                              pass_if_none=False)
#
#     pnames = [prefix + _ for _ in ('max_concurrent_downloads', 'max_thread_workers')]
#     pname, pval = get_param(config, pnames)
#     if pname == pnames[1]:
#         # When max_thread_workers<0, it defaulted to None:
#         pval = validate_param(pname, pval, valid_type, int, None)
#         if pval <= 0:
#             pval = None
#     else:
#         pval = validate_param(pname, pval, valid_between, 1, None, pass_if_none=True)
#     validated_params[pnames[0].split('.')[-1]] = pval
#
#     # get the whole advanced_settings dict, and update with the validated params:
#     adv_settings = pop_param(config, prefix[:-1])[1]
#     adv_settings.update(validated_params)
#     return adv_settings

    # pname = 'max_concurrent_downloads'
    # try:
    #     pval = pop_param(config, prefix + pname)
    # except MissingValue as mval:
    #     try:
    #         pval = pop_param(config, prefix + 'max_thread_workers')  # try old name
    #         # When max_thread_workers<=0, it defaulted to None:
    #         if pval <= 0:
    #             pval = None
    #     except MissingValue:
    #         raise mval
    # # Check that pval is either none or int >=1:
    # ret[pname] = validate_param(prefix + pname, pval,
    #                             valid_between, 1, None, pass_if_none=True)
    #
    # pname = 'db_buf_size'
    # ret[pname] = validate_param(prefix + pname, pop_param(config, prefix + pname),
    #                             valid_between, 1, None, pass_if_none=False)
    #
    # # get the whole advanced_settings dict, and update with the validated params:
    # adv_settings = pop_param(config, prefix[:-1], {})
    # adv_settings.update(ret)
    # return adv_settings

    #
    # advanced_settings[pname] = validate_param(pval)
    #
    #
    # paramname = 'download_blocksize'
    # try:
    #     if advanced_settings[paramname] <= 0:
    #         advanced_settings[paramname] = -1
    #
    #     paramname = 'max_concurrent_downloads'
    #     if paramname not in advanced_settings:
    #         # try to search old parameter "max_thread_workers"
    #         # (maybe an old download config)
    #         old_paramname = 'max_thread_workers'
    #         if old_paramname not in advanced_settings:
    #             raise KeyError()  # (will raise paramname error)
    #         # When old_paramname<=0, it defaulted to None (= max thread workers
    #         # automatically set by threadPool):
    #         if advanced_settings[old_paramname] <= 0:
    #             advanced_settings[old_paramname] = None
    #         advanced_settings[paramname] = advanced_settings.pop(old_paramname)
    #
    #     if advanced_settings[paramname] is not None:
    #         advanced_settings[paramname] = int(advanced_settings[paramname])
    #
    #     paramname = 'db_buf_size'
    #     advanced_settings[paramname] = max(advanced_settings[paramname], 1)
    # except Exception as exc:
    #     raise InvalidValue(exc, paramname)
    # return advanced_settings


def load_config_for_process(dburl, pyfile, funcname=None, config=None,
                            outfile=None, **param_overrides):
    """Check process arguments. Returns the tuple session, pyfunc, config_dict,
    where session is the dql alchemy session from `dburl`, `funcname` is the
    Python function loaded from `pyfile`, and config_dict is the dict loaded
    from `config` which must denote a path to a yaml file, or None (config_dict
    will be empty in this latter case)
    """
    session = validate_param("dburl", dburl, valid_session, for_process=True)
    funcname = validate_param("funcname", funcname, valid_funcname)
    pyfunc = validate_param("pyfile", pyfile, valid_pyfunc, funcname)
    config = validate_param("config", config or {}, yaml_load, **param_overrides)
    if outfile is not None:
        validate_param('outfile', outfile, valid_filewritable)
        # (ignore return value of filewritable: it's outfile, we already have it)
    seg_sel = _extract_segments_selection(config)

    multi_process, chunksize = _get_process_advanced_settings(config,
                                                              'advanced_settings')

    # pop and validate advanced settings:
    # try:
    #     adv_settings = config.get('advanced_settings', {})  # dict
    #     multi_process = adv_settings.get('multi_process', False)
    #     # the number of Pool processes can now be set directly to multi_process (which
    #     # accepts bool or int). Previously, there was a separate parameter for that,
    #     # num_processes. Let' implement backward compatibility here:
    #     if multi_process is True and 'num_processes' in adv_settings:
    #         multi_process = adv_settings['num_processes']
    #
    #     # check parameters:
    #     if multi_process not in (True, False):
    #         try:
    #             int(multi_process)
    #             assert multi_process >= 0
    #         except:
    #             raise ValueError('advanced_settings.multi_process must be '
    #                              'boolean or non negative int')
    #     chunksize = adv_settings.get('segments_chunksize', None)
    #     if chunksize is not None:
    #         try:
    #             int(chunksize)
    #             assert chunksize > 0
    #         except:
    #             raise ValueError('advanced_settings.chunksize must be '
    #                              'null or positive int')
    #
    # except Exception as exc:
    #     raise InvalidValue(exc, 'config.advanced_settings')

    return session, pyfunc, funcname, config, seg_sel, multi_process, chunksize


def _get_process_advanced_settings(config, adv_settings_key):
    prefix = adv_settings_key + '.'  # 'advanced_settings.'

    pname, multi_process = get_param(config, prefix + 'multi_process', default=False)
    if multi_process is True:
        # Backward compatibility: if multi_process is True, there
        # was a separate parameter to set the Pool processes: num_processes.
        # (now just set multi_process as int)
        num_processes = get_param(config, prefix + 'num_processes',
                                  default=multi_process)[1]
        # the line below is no-op if num_process was not found (new config):
        multi_process = num_processes
    if multi_process not in (True, False):
        multi_process = validate_param(pname, multi_process,
                                       valid_between, 1, None)

    pname, chunksize = get_param(config, prefix + 'segments_chunksize', None)
    if chunksize is not None:
        chunksize = validate_param(pname, chunksize, valid_between, 1, None)

    return multi_process, chunksize


def load_config_for_visualization(dburl, pyfile=None, config=None):
    """Check visualization arguments and return a tuple of well formed args.
    Raise BadParam
    """
    session = validate_param('dburl', dburl, valid_session, for_process=True, scoped=True)
    pymodule = None if not pyfile else validate_param('pyfile', pyfile, load_source)
    config_dict = {} if not config else validate_param('configfile', config, yaml_load)
    seg_sel = _extract_segments_selection(config_dict)

    return session, pymodule, config_dict, seg_sel


def _extract_segments_selection(config):
    # FIXME write doc: here we implement in ONE place the segment selection keys
    return pop_param(config, ['segments_selection', 'segment_select'], {})[1]


#####################################################################################
# Low level validation functions.
# IMPORTANT: By convention, these functions should start with "valid_" and
# NOT BE CALLED DIRECTLY but wrapped within :func:`validate_param`
######################################################################################


def valid_type(value, *other_values_or_types):
    """Returns value if it is of the same type (same class, or subclass) of *any*
    other value type (at least one). Raises TypeError otherwise.

    :param value: a python object
    :param other_values_or_types: Python objects or classes. In the first case,
        it will compare the value class vs. the object class.

    :return: value
    """
    value_type = value.__class__
    other_types = tuple(_ if isinstance(_, type) else _.__class__
                        for _ in other_values_or_types)

    if issubclass(value_type, other_types):
        return value

    raise TypeError("%s expected, found %s" %
                    (" or ".join(str(_) for _ in other_types), str(value_type)))


def valid_nslc(value):
    """Return a nslc (network/station/location/channel) parameter value
    converted as list. This method cleans-up and checks `value` splitting each
    of its string elements with the comma "," and aggregating all the string
    chunks into a single list, after performing some sanity check. The
    resulting list is also sorted alphabetically (for unit testing and
    readability). Raises ValueError in case some sanity checks fail (e.g.,
    conflicts, syntax errors)

    Examples:

    Func. arguments      Result (with comment)
    =================== =================================================
    (['A','D','C','B'])  ['A', 'B', 'C', 'D']  # note result is sorted
    ('B,C,D,A')          ['A', 'B', 'C', 'D']  # same as above
    ('A*, B??, C*')      ['A*', 'B??', 'C*']  # FDSN wildcards accepted
    ('!A*, B??, C*')     ['!A*', 'B??', 'C*']  # in s2s, !A* means "not A*"
    (' A, B ')           ['A', 'B']  # leading and trailing spaces ignored
    ('*')                []  # [] means "match all"
    ([])                 []  # same as above
    ('  ')               ['']  # string is stripped: match the empty string
    ("")                 [""]  # match the empty string
    ("!")                ['!']  # match any non empty string
    ("!*")               this raises (you cannot specify "discard all")
    ("!H*, H*")          this raises (it's a paradox)
    (" A B,  CD")        this raises ('A B' invalid: only leading and trailing
                                      spaces allowed)

    :param value: string or iterable of strings: (iterable in this context
        means Python iterable EXCEPT strings). If string, the argument will be
        converted to the list [value] to make it iterable before processing it
    """
    try:
        strings = set()

        # we assume, when parsearg is not list, that parsearg is str in both
        # py2 and py3, i.e. it is NOT bytes in python2. The line below checks
        # if is an iterable first:
        # in python2, it is sufficient to say it's not a string
        # in python3, we need to check that is no str also
        if not hasattr(value, "__iter__") or isinstance(value, str):
            # it's an iterable not a string
            value = [value]

        for string in value:
            splitted = string.split(",")
            for chunk in splitted:
                chunk = chunk.strip()
                if ' ' in chunk:
                    raise Exception("invalid space char(s): '%s'" % chunk)
                # if i == 3 (location) convert '--' to '':
                strings.add(chunk)

        # some checks:
        if "!*" in strings:  # discard everything is not valid
            raise ValueError("'!*' (=discard all) invalid")
        elif "*" in strings:  # accept everything or X => X is redundant
            strings = set(_ for _ in strings if _[0:1] == '!')
        else:
            for string in strings:  # accept A end discard A is not valid
                opposite = "!%s" % string
                if opposite in strings:
                    raise Exception("conflicting values: '%s' and '%s'" %
                                    (string, opposite))

        return sorted(strings)

    except Exception as exc:
        raise ValueError(str(exc))


def valid_dburl_or_download_yamlpath(value, param_name='dburl'):
    """Return the database path from 'value': 'value' can be a file (in that
    case is assumed to be a yaml file with the `param_name` key in it, which
    must denote a db path) or the database path otherwise
    """
    if not isinstance(value, string_types) or not value:
        raise TypeError('please specify a string denoting either a path to a '
                        'yaml file with the `dburl` parameter defined, or a '
                        'valid db path')
    return yaml_load(value)[param_name] if (os.path.isfile(value)) else value


# def keyval_list_to_dict(value):
#     """Parse optional event query args (when the 'd' command is issued) into
#     a dict"""
#     # use iter to make a dict from a list whose even indices = keys, odd
#     # ones = values (https://stackoverflow.com/a/4576128)
#     itr = iter(value)
#     return dict(zip(itr, itr))


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
    if for_process:
        # important, rename otherwise conflicts with this function name:
        from stream2segment.process.db import get_session as sess_func
    else:
        # important, rename otherwise conflicts with this function name:
        from stream2segment.io.db import get_session as sess_func

    sess = sess_func(dburl, scoped=scoped, **engine_kwargs)

    # Check if database exist, which should not always be done (e.g.
    # for_processing=True). Among other methods
    # (https://stackoverflow.com/a/3670000
    # https://stackoverflow.com/a/59736414) this seems to do what we need
    # (we might also not check if the tables length > 0 sometime):
    sess.bind.engine.table_names()
    return sess


def valid_authorizer(restricted_data, dataws, configfile=None):
    """Create an :class:`stream2segment.download.utils.Authorizer`
    (handling authentication/authorization) from the given restricted_data

    :param restricted_data: either file path, to token, token data in bytes, or
        tuple (user, password). If None, or the empty string, None is returned
    """
    if restricted_data in ('', None, b''):
        restricted_data = None
    elif isinstance(restricted_data, string_types) and configfile is not None:
        restricted_data = normalizedpath(restricted_data, os.path.dirname(configfile))
    ret = Authorizer(restricted_data)
    # check dataws is single element list:
    if len(dataws) != 1:
        raise ValueError('downloading restricted data requires '
                         'a single URL in `dataws`')
    dataws = dataws[0]
    # Here we have 4 cases:
    # 1 'eida' + token: OK
    # 2. Any other fdsn + username & password: OK
    # 3. eida + username & password: BAD. raise ValueError
    # 4. Any other fdsn + token: OK (we might have provided a single eida
    #                                datacenter in which case it's fine)
    if dataws.lower() == 'eida' and ret.userpass:
        raise ValueError('downloading from EIDA requires a token, '
                         'not username and password')
    return ret


def valid_updatemetadata_param(value):
    """Parse parse_update_metadata returning True, False or 'only'"""
    val_str = str(value).lower()
    if val_str == 'true':
        return True
    if val_str == 'false':
        return False
    if val_str == 'only':
        return val_str
    raise ValueError('value can be true, false or only, %s provided' % val_str)


def valid_tt_table(file_or_name):
    """Load the given TTTable object from the given file path or name. If name
    (string) it must match any of the builtin TTTable .npz files defined in
    this package. Raises TypeError or any Exception that TTTable might raise
    (including when the file is not found)
    """
    if not isinstance(file_or_name, string_types):
        raise TypeError('string required, not %s' % str(type(file_or_name)))
    filepath = get_ttable_fpath(file_or_name)
    if not os.path.isfile(filepath):
        filepath = file_or_name
    if not os.path.isfile(filepath):
        raise Exception('file or builtin model name not found')
    return TTTable(filepath)


def valid_date(obj):
    try:
        return strptime(obj)  # if obj is datetime, returns obj
    except (TypeError, ValueError) as _:
        try:
            days = int(obj)
            now = datetime.utcnow()
            endt = datetime(now.year, now.month, now.day, 0, 0, 0, 0)
            return endt - timedelta(days=days)
        except Exception:
            pass
        if isinstance(_, TypeError):
            raise TypeError(("iso-formatted datetime string, datetime "
                             "object or int required, found %s") %
                            str(type(obj)))
        else:
            raise _


def valid_fdsn(url, is_eventws, configfile=None):
    """Return url if it matches a FDSN service (valid strings are 'eida' and
    'iris'), raises ValueError or TypeError otherwise
    """
    if not isinstance(url, string_types):
        raise TypeError('string required')

    if (is_eventws and url.lower() in EVENTWS_MAPPING) or \
            (not is_eventws and url.lower() in ('eida', 'iris')):
        return url.lower()

    if is_eventws:
        fpath = url if configfile is None else \
            normalizedpath(url, os.path.dirname(configfile))
        if os.path.isfile(fpath):
            return fpath
        try:
            return Fdsnws(url).url()
        except Exception:
            raise ValueError('Invalid FDSN url or file path, check typos')

    return Fdsnws(url).url()


# def valid_dict_or_none(value):
#     """Check that value is a dict and returns `value`.
#     Returns {} if the value is None. In any other cases, raise ValueError
#     """
#     if value is None:
#         value = {}
#     if isinstance(value, dict):
#         return value
#     raise ValueError('dict/None required, found: %s' % str(type(value)))


def valid_between(val, min, max, include_min=True, include_max=True, pass_if_none=True):
    if val is None:
        if pass_if_none:
            return val
        raise ValueError('value is None/null')

    # if min is not None:
    #     try:
    #         valid_type(val, min)
    #     except Exception:
    #         raise ValueError('Types mismatch, cannot compare %s and %s' %
    #                          (str(val), str(min)))
    #     if not (val > min or (include_min and val >= min)):
    #         raise ValueError('%s must be %s %s' %
    #                          (str(val), '>=' if include_min else '>', str(min)))
    #
    # if max is not None:
    #     try:
    #         valid_type(val, max)
    #     except Exception:
    #         raise ValueError('Types mismatch, cannot compare %s and %s' %
    #                          (str(val), str(max)))
    #     if not (val < max or (include_max and val <= max)):
    #         raise ValueError('%s must be %s %s' %
    #                          (str(val), '<=' if include_max else '<', str(max)))

    is_ok = min is None or val > min or (include_min and val >= min)
    if not is_ok:
        raise ValueError('%s must be %s %s' %
                         (str(val), '>=' if include_min else '>', str(min)))

    is_ok = max is None or val < max or (include_max and val <= max)
    if not is_ok:
        raise ValueError('%s must be %s %s' %
                         (str(val), '<=' if include_max else '<', str(max)))
    return val


def valid_search_radius(search_radius):
    """Check the validity of the 'search_radius' argument (dict)"""
    args = [
        search_radius.get('minmag'),
        search_radius.get('maxmag'),
        search_radius.get('minmag_radius'),
        search_radius.get('maxmag_radius'),
        search_radius.get('min'),
        search_radius.get('max')
    ]
    magdep_args = args[:4]
    magindep_args = args[4:]
    magdep_argscount = sum(_ is not None for _ in magdep_args)
    magindep_argscount = sum(_ is not None for _ in magindep_args)
    is_mag_dep = magdep_argscount == len(magdep_args) and not magindep_argscount
    is_mag_indep = magindep_argscount == len(magindep_args) and not magdep_argscount

    if is_mag_dep == is_mag_indep:
        raise ValueError("provide either 'min', 'max' or 'minmag', 'maxmag', "
                         "'minmag_radius', 'maxmag_radius'")

    # check errors:
    nofloaterr = ValueError('numeric values expected')
    if is_mag_dep:
        if not all(isinstance(_, (int, float)) for _ in magdep_args):
            raise nofloaterr
        if args[0] > args[1]:  # minmag > maxmag
            raise ValueError('minmag should not be greater than maxmag')
        if args[2] <= 0 or args[3] <= 0:  # minmag_radius or maxmag_radius <=0
            raise ValueError('minmag_radius and maxmag_radius should be '
                             'greater than 0')
        if args[0] == args[1] and args[2] == args[3]:
            # minmag == maxmag, minmag_radius == maxmag_radius => error
            raise ValueError('To supply a constant radius, set "min: 0" and '
                             'specify the radius with the "max" argument')
    else:
        if not all(isinstance(_, (int, float)) for _ in magindep_args):
            raise nofloaterr
        if args[-2] < 0:
            raise ValueError('min should not be lower than 0')
        if args[-1] <= 0:
            raise ValueError('max should be greater than 0')
        if args[-2] >= args[-1]:
            raise ValueError('min should be lower than max')

    return search_radius


def valid_pyfunc(pyfile, funcname):
    """Return the Python module from the given python file"""
    if not isinstance(pyfile, string_types):
        raise TypeError('string required, not %s' % str(type(pyfile)))

    if not os.path.isfile(pyfile):
        raise Exception('file does not exist')

    pymoduledict = load_source(pyfile).__dict__

    # check for new style module: SkipSegment instead of ValueError
    if 'SkipSegment' not in pymoduledict:
        raise ValueError('The provided Python module looks outdated.\nYou first need to '
                         'import SkipSegment ("from stream2segment.process import '
                         'SkipSegment") to suppress this warning, and\n'
                         'check your code: to skip a segment, please type '
                         '"raise SkipSegment(.." instead of "raise ValueError(..."')

    if funcname not in pymoduledict:
        raise Exception('function "%s" not found in %s' %
                        (str(funcname), pyfile))
    return pymoduledict[funcname]


def valid_funcname(funcname=None):
    """Return the Python module from the given python file"""
    if funcname is None:
        funcname = valid_default_processing_funcname()

    if not isinstance(funcname, string_types):
        raise TypeError('string required, not %s' % str(type(funcname)))

    return funcname


def valid_default_processing_funcname():
    """Return 'main', the default function name for processing, when such
    a name is not given"""
    return 'main'


def valid_filewritable(filepath):
    """Check that the file is writable, i.e. that is a string and its
    directory exists"""
    if not isinstance(filepath, string_types):
        raise TypeError('string required, found %s' % str(type(filepath)))

    if not os.path.isdir(os.path.dirname(filepath)):
        raise ValueError('cannot write file: parent directory does not exist')

    return filepath


