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


def run(db_uri, pyfile, configfile, port, debug):
    pymodule = load_source(pyfile)
    configdict = yaml_load(configfile)
    app = create_app(db_uri, pymodule, configdict)
    app.run(port=port, debug=debug)


def run_in_browser(db_uri, pyfile, configfile, port=None, debug=False):
    if port is None:
        port = 5000 + random.randint(0, 999)
    url = "http://127.0.0.1:{0}".format(port)
    if not debug:
        threading.Timer(1.25, lambda: webbrowser.open(url)).start()
    pyfile = None if not pyfile else abspath(realpath(pyfile))
    configfile = None if not configfile else abspath(realpath(configfile))
    run(db_uri, pyfile, configfile, port=port, debug=debug)
