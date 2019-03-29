'''
Functions for launching the web app

:date: Jun 20, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
from __future__ import print_function

import os
from os.path import realpath, abspath
import random
import threading
from webbrowser import open as open_in_browser

from stream2segment.utils import load_source
from stream2segment.utils.resources import yaml_load
from stream2segment.gui.webapp import create_app
from stream2segment.gui.webapp.mainapp.plots.core import PlotManager


def create_main_app(dbpath, pyfile=None, configfile=None):
    """
        Creates a new app for processing. Note that config_py_file is the stream2segment gui
        config, not the config passed to Flask `app.config.from_pyfile`.
    """
    pymodule = None if pyfile is None else load_source(pyfile)
    configdict = {} if configfile is None else yaml_load(configfile)
    with create_app(dbpath) as app:
        app.config['PLOTMANAGER'] = PlotManager(pymodule, configdict)
        app.config['CONFIG.YAML'] = configdict
        from stream2segment.gui.webapp.mainapp.views import main_app
        app.register_blueprint(main_app)

    return app


def run_in_browser(app, port=None, debug=False):
    # https://stackoverflow.com/a/53919435
    os.environ['FLASK_ENV']='development'
    if port is None:
        port = 5000 + random.randint(0, 999)
    url = "http://127.0.0.1:{0}".format(port)
    if not debug:
        threading.Timer(1.25, lambda: open_in_browser(url)).start()
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


