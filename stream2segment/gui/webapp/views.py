'''
Created on Jun 20, 2016

@author: riccardo
'''

from stream2segment.gui.webapp import app
from flask import render_template


@app.route("/")
def main():
    return "asd"
    # return render_template('index.html', title=app.config['DB_URI'])

# @app.route("/get_elements")
# def view_page2():
    