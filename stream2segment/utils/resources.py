'''
Module implementing all project specific resources (data such as files)

Created on Feb 20, 2017

@author: riccardo
'''
from os.path import join, dirname, abspath, normpath


def _get_main_path():
    """Returns the main rott path of the project, the one hosting the 'stream2segment' package"""
    # we are in ./stream2segment/utils.resources.py, so we need to get up 3 times
    return normpath(abspath(dirname(dirname(dirname(__file__)))))


def get_proc_template_files():
    """Returns the tuple (pyton file, yaml config file) to be used for a processing template"""
    _dir = join(_get_main_path(), "stream2segment", "process", "templates")
    return join(_dir, "template1.py"), join(_dir, "template1.conf.yaml")


def get_default_cfg_filepath(filename='config.example.yaml'):
    """Returns the configuration file path named `filename` in the main project dir
    (the root project hosting the stream2segment package)
    :param filename: The file name (no path). if missing, defaults to 'config.yaml'
    """
    return join(_get_main_path(), filename)


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
    return get_default_cfg_filepath(filename='ws.yaml')
