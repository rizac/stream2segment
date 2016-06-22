'''
Created on Jun 20, 2016

@author: riccardo
'''
from stream2segment.gui.webapp import app
import sys
# from stream2segment.s2sio.db import ClassAnnotator


class Config(object):
    DEBUG = False
    TESTING = False
    DATABASE_URI = 'sqlite://:memory:'

    def __init__(self, db_uri):
        Config.DATABASE_URI = db_uri


def main(db_uri):
    app.config.update(
                      DATABASE_URI=db_uri
                      )
    app.run()


if __name__ == '__main__':
    # global files
    if len(sys.argv) < 2:
        print "please specify a valid directory of mseed files"
        sys.exit(1)
    path_ = sys.argv[1]
    main(path_)
