'''
Module returning all project specific resources (files).

Created on Feb 20, 2017

@author: riccardo
'''
from os import listdir
from os.path import join, dirname, abspath, normpath, isfile, isabs, splitext
import re
from collections import defaultdict
import yaml


def _get_main_path():
    """Returns the main root path of the project 'stream2segment', the **parent** folder of the
    `_get_package_path()`"""
    # we are in ./stream2segment/utils.resources.py, so we need to get up 3 times
    return normpath(abspath(dirname(dirname(dirname(__file__)))))


def _get_package_path():
    """Returns the main root path of the package 'stream2segment', the **child** folder of the
    `_get_main_path()`"""
    # we are in ./stream2segment/utils.resources.py, so we need to get up 3 times
    return join(_get_main_path(), "stream2segment")


def get_resources_fpath(filename):
    """Returns the resource file with given filename inside the package `resource` folder
    :param filename: a filename relative to the resource folder
    """
    resfolder = join(_get_package_path(), "resources")
    return join(resfolder, filename)


def get_templates_dirpath():
    """Returns the templates directory path (located inside the package `resource` folder)
    """
    return get_resources_fpath("templates")


def get_traveltimes_dirpath():
    """Returns the travel time table directory path (located inside the package `resource` folder)
    """
    return get_resources_fpath("traveltimes")


def get_templates_fpaths(*filenames):
    """Returns the template file paths with given filename(s) inside the package `templates` of the
    `resource` folder. If filenames is empty (no arguments), returns all files (no dir) in the
    `templates` folder
    :param filenames: a list of file names relative to the templates folder. With no argument,
    returns all valid files inside that directory
    """
    templates_path = get_templates_dirpath()
    if not len(filenames):
        filenames = listdir(templates_path)

    ret = []
    for _name in filenames:
        fpath = join(templates_path, _name)
        if isfile(fpath):
            ret.append(fpath)
    return ret


def get_templates_fpath(filename):
    """Returns the template file path with given filename inside the package `templates` of the
    `resource` folder
    :param filename: a filename relative to the templates folder
    """
    return get_templates_fpaths(filename)[0]


def version(onerr=""):
    """Returns the program version saved in the main root dir 'version' file.
    :param onerr (string, "" when missing): what to return in case of IOError.
    If 'raise', then the exception is raised
    """
    try:
        with open(join(_get_main_path(), "version")) as _:
            return _.read().strip()
    except IOError as exc:
        if onerr == 'raise':
            raise exc
        return onerr


def get_ws_fpath():
    """Returns the web-service config file (yaml)"""
    return get_resources_fpath(filename='ws.yaml')


def yaml_load(filepath, raw=False, **defaults):
    """Loads default config from yaml file, normalizing relative sqlite file paths if any
    assuming they are relative to `filepath`, and setting the given defaults (if any)
    for arguments missing in the config
    (if raw is True)"""
    with open(filepath, 'r') as stream:
        ret = yaml.safe_load(stream) if not raw else stream.read()
    if not raw:
        configfilepath = abspath(dirname(filepath))
        # convert relative sqlite path to absolute, assuming they are relative to the config:
        sqlite_prefix = 'sqlite:///'
        # we cannot modify a dict while in iteration, thus create a new dict of possibly
        # modified sqlite paths and use later dict.update
        newdict = {}
        for k, v in ret.iteritems():
            try:
                if v.startswith(sqlite_prefix) and ":memory:" not in v:
                    dbpath = v[len(sqlite_prefix):]
                    if not isabs(dbpath):
                        newdict[k] = sqlite_prefix + normpath(join(configfilepath, dbpath))
            except AttributeError:
                pass
        if newdict:
            ret.update(newdict)

        for key, val in defaults.iteritems():
            if key not in ret:
                ret[key] = val
    return ret


def yaml_load_doc(filepath, varname=None):
    """Loads the doc from a yaml. The doc is intended to be all *consecutive* commented lines
    (with *no* leading spaces) before each top-level variable (nested variables are not considered).
    If `varname` is None (the default), the returned dict is a defaultdict which returns as
    string values (**unicode** strings in python 2) or an empty string for non-found documented
    variables.
    If `varname` is not None, as soon as the doc for `varname` is found, this function
    returns that doc string, and not the whole dict, or the empty string if nothing is found
    :param filepath: The yaml file to read the doc from
    :param varname: if None, returns a `defaultdict` with all docs (consecutive
    commented lines before) the yaml top-level variables. Otherwise, return the doc for the
    given variable name (string)
    """
    comments = []
    reg_yaml_var = re.compile("^([^:]+):.*")
    reg_comment = re.compile("^#+(.*)")
    ret = defaultdict(str) if varname is None else ''
    isbytes = None
    with open(filepath, 'r') as stream:
        while True:
            line = stream.readline()  # last char of line is a newline
            if isbytes is None:
                isbytes = isinstance(line, bytes)
            if not line:
                break
            m = reg_comment.match(line)
            if m and m.groups():  # set comment
                # note that our group does not include last newline
                comments.append(m.groups()[0].strip())  # del leading and trailing spaces, if any
            else:  # try to see if it's a variable, and in case set the doc (if any)
                if comments:  # parse variable only if we have comments
                    # otherwise each nested variable is added to the dict with empty comment
                    m = reg_yaml_var.match(line)
                    if m and m.groups():
                        var_name = m.groups()[0]
                        comment = " ".join(comments)
                        docstring = comment.decode('utf8') if isbytes else comment
                        if varname is None:
                            ret[var_name] = docstring
                        elif varname == var_name:
                            ret = docstring
                            break
                # in any case, if not comment, reset comments:
                comments = []
    return ret


def get_ttable_fpath(basename):
    '''Returns the file for the given traveltimes table
    :param basename: the file name (with or without extension) located under
    `get_traveltimestables_dirpath()`
    '''
    if not splitext(basename)[1]:
        basename += ".npz"
    return join(get_traveltimes_dirpath(), basename)
