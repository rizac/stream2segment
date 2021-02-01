"""
Module for easily accessing all project specific resources.

:date: Feb 20, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
from os import listdir
from os.path import join, dirname, abspath, normpath, isfile, isabs, splitext
import yaml
from collections import defaultdict

# python2-3 compatibility for items and viewitems:
from future.utils import viewitems, viewkeys, string_types


def _get_main_path():
    """Return the main root path of the project 'stream2segment',
    the **parent** folder of the `_get_package_path()`"""
    # we are in ./stream2segment/utils.resources.py, so we need to get up 3
    # times
    return normpath(abspath(dirname(dirname(dirname(__file__)))))


def _get_package_path():
    """Return the main root path of the package 'stream2segment', the CHILD
    directory of the `_get_main_path()`
    """
    # we are in ./stream2segment/utils.resources.py, so we need to get up 3
    # times
    return join(_get_main_path(), "stream2segment")


def get_resources_fpath(filename):
    """Return the resource file with given filename inside the package
    `resource` directory

    :param filename: a filename relative to the resource directory
    """
    resfolder = join(_get_package_path(), "resources")
    return join(resfolder, filename)


def get_templates_dirpath():
    """Return the templates directory path (located inside the package
    `resource` folder)
    """
    return get_resources_fpath("templates")


def get_traveltimes_dirpath():
    """Return the travel time tables directory path (located inside the
    package `resource` folder)
    """
    return get_resources_fpath("traveltimes")


def get_ttable_fpath(basename):
    """Return the file for the given travel times table

    :param basename: the file name (with or without extension) located under
        `get_traveltimestables_dirpath()`
    """
    if not splitext(basename)[1]:
        basename += ".npz"
    return join(get_traveltimes_dirpath(), basename)


def get_templates_fpaths(*filenames):
    """Return the template file paths with given filename(s) inside the package
    `templates` of the `resource` directory. If filenames is empty (no
    arguments), returns all files (no dir) in the `templates` directory

    :param filenames: a list of file names relative to the templates directory.
        With no argument,returns all valid files inside that directory
    """
    templates_path = get_templates_dirpath()
    if not filenames:
        filenames = listdir(templates_path)

    return list(join(templates_path, _name) for _name in filenames)


def get_templates_fpath(filename):
    """Return the template file path with given filename inside the package
    `templates` of the `resource` directory

    :param filename: a filename relative to the templates directory
    """
    return get_templates_fpaths(filename)[0]


def version(onerr=""):
    """Return the program version saved in the main root dir 'version' file.

    :param onerr: (str, default ""): what to return in case of IOError.
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
    """Return the web-service config file (yaml)"""
    return get_resources_fpath(filename='ws.yaml')


def yaml_load(filepath, **updates):
    """Load a yaml file into a dict (if `filepath` is a `dict`, skips loading).
    Then:
    1. If `filepath` denotes a file path (and not a dict),
       normalizes non-absolute sqlite path values relative to `filepath`, if any
    2. updates the dict values with `updates` and returns the yaml dict. The
       update is recursive, meaning that nested dict values will be updated
       recursively and not completely overwritten. Example:
       param. a = {'b': 1, 'c' :2}, updates a = {'b': 2, 'd' :3} =>
       result = {'b': 2, 'c': 2, 'd': 3}

    :param filepath: str, dict or file-like object. If str, it must denote a
        path to an existing .yaml file
    :param updates: arguments which will updates the yaml dict before it is
        returned
    """
    if isinstance(filepath, string_types):
        with open(filepath, 'r') as stream:
            ret = yaml.safe_load(stream)
    elif isinstance(filepath, dict):
        ret = filepath
    elif hasattr(filepath, 'read'):
        ret = yaml.safe_load(filepath)
    else:
        raise TypeError('required file path (string), file object or dict, '
                        '%s found' % str(type(filepath)))

    # update recursively (which means sub-dicts are updated as well and not
    # overwritten):
    def update(dic1, dic2):
        """update dic1 with dic2 recursively"""
        # Terminology: If the same key exists in both dicts and is mapped to
        # two dictionaries (not necessarily equal), the latter are called
        # "shared dicts"

        # 1. Move shared dicts from `dic2` in a temporary dictionary 'dickeys':
        dickeys = {k: dic2.pop(k) for k in viewkeys(dic1)
                   if isinstance(dic1[k], dict) and
                   isinstance(dic2.get(k, None), dict)}
        # 2. `dic1` and `dic2` have no shared dicts, update dicts "normally":
        dic1.update(dic2)
        # 3. Update shared dicts recursively:
        for k in dickeys:
            update(dic1[k], dickeys[k])

    update(ret, updates)

    if isinstance(filepath, string_types):
        # convert sqlite into absolute paths, if any. This does not convert
        # nested sub-dict strings
        configfilepath = abspath(dirname(filepath))
        # convert relative sqlite path to absolute, assuming they are relative
        # to the config:
        sqlite_prefix = 'sqlite:///'
        # we cannot modify a dict while in iteration, thus create a new dict of
        # possibly modified sqlite paths and use later dict.update
        newdict = {}
        for key, val in viewitems(ret):
            try:
                if val.startswith(sqlite_prefix) and ":memory:" not in val:
                    dbpath = val[len(sqlite_prefix):]
                    npath = normalizedpath(dbpath, configfilepath)
                    if npath != dbpath:
                        newdict[key] = sqlite_prefix + npath
            except AttributeError:
                pass

        ret.update(newdict)
    return ret


def normalizedpath(path, basedir):
    """Normalize `path` if it's not absolute, making it relative to `basedir`.
    If path is already absolute, returns it as it is

    :param path: the path
    :param basedir: the base directory path
    """
    if isabs(path):
        return path
    return abspath(normpath(join(basedir, path)))


def yaml_load_doc(filepath, varname=None, preserve_newlines=False):
    """Return the documentation from a YAML file. The returned object is
    a the documentation (str) of the given variable name (if `varname` is set),
    or a dict[str, str] (defaultdict("")) of all variables found, mapped to
    their documentation (a variable documentation is made up of all
    consecutive commented lines -  with *no* leading spaces - placed immediately
    before the variable). Only top-level variables can be parsed, nested ones
    are skipped.

    :param filepath: The YAML file to read the doc from
    :param varname: str or None (the default). Return the doc for this specific
        YAML variable. if None, returns a `defaultdict` with all top-level
        variables found.
    :param preserve_newlines: boolean. Whether to preserve newlines in comment
        or not. If False (the default), each variable comment is returned as a
        single line, concatenating parsed lines with a space
    """
    comments = []
    # reg_yaml_var = re.compile("^([^:]+):\\s.*")
    # reg_comment = re.compile("^#+(.*)")
    ret = defaultdict(str) if varname is None else ''
    isbytes = None
    with open(filepath, 'r') as stream:
        while True:
            line = stream.readline()
            # from the docs (https://docs.python.org/3/tutorial/inputoutput.html): if
            # f.readline() returns an empty string, the end of the file has been reached,
            # while a blank line is represented by '\n'
            if not line:
                break
            if isbytes is None:
                isbytes = isinstance(line, bytes)
            # is line a comment? do not use regexp, it's slower
            # m = reg_comment.match(line)
            if line.startswith('#'):
                # the line is a comment, add the comment text.
                # Note that the line does not include last newline, if present
                comments.append(line[1:].strip())
            else:
                # the line is not a comment line. Do we have parsed comments?
                if comments:
                    # use string search and not regexp because faster:
                    idx = line.find(': ')
                    if idx == -1:
                        idx = line.find(':\n')
                    var_name = None if idx < 1 else line[:idx]
                    # We have parsed comments. Is the line a YAML parameter?
                    # m = reg_yaml_var.match(line)
                    # if m and m.groups():
                    if var_name:
                        # the line is a yaml variable, it's name is
                        # m.groups()[0]. Map the variable to its comment
                        # var_name = m.groups()[0]
                        join_char = "\n" if preserve_newlines else " "
                        comment = join_char.join(comments)
                        docstring = comment
                        if isbytes:
                            docstring = comment.decode('utf8')
                        if varname is None:
                            ret[var_name] = docstring
                        elif varname == var_name:
                            return docstring
                # In any case, if not comment, reset comments:
                comments = []
    return ret
