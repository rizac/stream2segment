"""
Functions for launching the web app
"""
# :date: Jun 20, 2016
import uuid
from pathlib import Path
from webbrowser import open as open_in_browser
import random
import threading

import yaml
from flask import Flask


from stream2segment.process import get_default_segments_selection
from stream2segment.process.gui.introspection import load_source
from stream2segment.resources import get_templates_fpath


# from stream2segment.process.db import get_session



def show_gui(
    db_url: str,
    py_file: Path | None,
    config_file: Path | None
):
    """Show downloaded data plots in a system browser dynamic web page"""
    seg_sel = get_default_segments_selection() | {'gap_score_percent': '[-50, 50]'}
    # Add constraints on traces with gaps. This is not only to avoid plotting traces
    # with gaps, but to help users showing an example of segment selection expr.
    run_in_browser(
        create_s2s_show_app(db_url, py_file, config_file, seg_sel)
    )
    return 0


def create_s2s_show_app(
    db_url: str,
    py_file: Path | None = None,
    config_file: Path | None =None,
    segments_selection=None
):
    """Create a new app for processing. Note that config_py_file is the
    stream2segment GUI config, not the config passed to Flask
    `app.config.from_pyfile`.
    """
    if py_file is None:
        py_file = get_templates_fpath('gui.py')
        if config_file is None:
            config_file = get_templates_fpath('gui.yaml')
    py_module = load_source(py_file)
    config = {}
    if config_file is not None:
        config = yaml.safe_load(config_file.read_text())

    from stream2segment.process.gui import webapp
    # http://flask.pocoo.org/docs/0.12/patterns/appfactories/#basic-factories
    app = Flask(webapp.__name__)

    from stream2segment.process.gui.webapp.mainapp import core
    seg_count = core.init(app, db_url, py_module, config, segments_selection)
    if seg_count < 1:
        raise ValueError('No plottable waveform found on the database')
    core.reset_segment_ids_array(seg_count)

    # Note that the template_folder of the Blueprint and the static paths in
    # the HTML are relative to the path of THIS MODULE, so execute the lines
    # below HERE or good luck changing all static paths in the html:
    from stream2segment.process.gui.webapp.mainapp.views import main_app
    app.register_blueprint(main_app)

    return app


def run_in_browser(app, port=None, debug=False):
    app.config.update(
        ENV='development',  # https://stackoverflow.com/a/53919435,
        # DEBUG = True,
        # although we do not use sessions (which write cookies client side),
        # we set a secret key neverthless:
        # https://www.tutorialspoint.com/flask/flask_sessions.htm
        SECRET_KEY=str(uuid.uuid4())
    )
    if port is None:
        port = 5000 + random.randint(0, 999)
    url = "http://127.0.0.1:{0}".format(port)
    if not debug:
        threading.Timer(1.25, lambda: open_in_browser(url)).start()
    app.run(port=port, debug=debug)
