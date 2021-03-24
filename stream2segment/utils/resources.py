"""
Module for easily accessing all project specific resources.

:date: Feb 20, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
from os import listdir
from os.path import join, dirname, abspath, normpath, isabs, splitext
from collections import defaultdict

# python2-3 compatibility for items and viewitems:


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


def normalizedpath(path, basedir):
    """Normalize `path` if it's not absolute, making it relative to `basedir`.
    If path is already absolute, returns it as it is

    :param path: the path
    :param basedir: the base directory path
    """
    if isabs(path):
        return path
    return abspath(normpath(join(basedir, path)))
