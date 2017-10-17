'''
Functions for launching the web app

:date: Jun 20, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
from __future__ import print_function

import sys
from os.path import realpath, abspath
import random
import threading
import webbrowser

from stream2segment.utils import load_source
from stream2segment.utils.resources import yaml_load
from stream2segment.gui.webapp import create_app
from stream2segment.gui.webapp.processing.plots.core import PlotManager


def create_p_app(dbpath, pyfile, configfile):
    """
        Creates a new app for processing. Note that config_py_file is the stream2segment gui
        config, not the config passed to Flask `app.config.from_pyfile`. For Flask config, please
        provide a valid object in `config_object`
    """
    pymodule = load_source(pyfile)
    configdict = yaml_load(configfile)
    with create_app(dbpath) as app:
        app.config['PLOTMANAGER'] = PlotManager(pymodule, configdict)
        app.config['CONFIG.YAML'] = configdict
        from stream2segment.gui.webapp.processing.views import main_page
        app.register_blueprint(main_page)

    return app


def create_drep_app(dbpath):
    """
        Creates a new app for the download report. Note that config_py_file is the stream2segment
        gui config, not the config passed to Flask `app.config.from_pyfile`. For Flask config,
        please provide a valid object in `config_object`
    """
    with create_app(dbpath) as app:
        from stream2segment.gui.webapp.dreport.views import main_page
        app.register_blueprint(main_page)
    return app


def run_in_browser(app, port=None, debug=False):
    if port is None:
        port = 5000 + random.randint(0, 999)
    url = "http://127.0.0.1:{0}".format(port)
    if not debug:
        threading.Timer(1.25, lambda: webbrowser.open(url)).start()
    app.run(port=port, debug=debug)


# def run_p_in_browser(db_uri, pyfile, configfile, port=None, debug=False):
#     if port is None:
#         port = 5000 + random.randint(0, 999)
#     url = "http://127.0.0.1:{0}".format(port)
#     if not debug:
#         threading.Timer(1.25, lambda: webbrowser.open(url)).start()
#     pyfile = None if not pyfile else abspath(realpath(pyfile))
#     configfile = None if not configfile else abspath(realpath(configfile))
#     app = create_app(db_uri, pyfile, configfile)
#     app.run(port=port, debug=debug)


