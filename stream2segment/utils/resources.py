'''
Module returning all project specific resources (files).

Created on Feb 20, 2017

@author: riccardo
'''
from os import listdir
from os.path import join, dirname, abspath, normpath, isfile


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
