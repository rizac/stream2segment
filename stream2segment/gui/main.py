'''
Created on Jun 20, 2016

@author: riccardo
'''
from stream2segment.gui.webapp import app
import sys
import random
import threading
import webbrowser
from stream2segment.main import cfg_dict


# from stream2segment.s2sio.db import ClassAnnotator
def main(db_uri, port, debug):
    app.config.update(
                      DATABASE_URI=db_uri
                      )
    app.run(port=port, debug=debug)


def run_in_browser(db_uri):
    port = 5000 + random.randint(0, 999)
    url = "http://127.0.0.1:{0}".format(port)

    threading.Timer(1.25, lambda: webbrowser.open(url)).start()

    main(db_uri, port=port, debug=False)


if __name__ == '__main__':
    db_uri = cfg_dict['dburi']
    print "Using config.py dburi: %s" % db_uri

    # global files
#     if len(sys.argv) < 2:
#         dburi = cfg_dict['dburi']
#         print "Using config.py dburi: %s" % dburi
#     else:
#         db_uri = sys.argv[1]

    run_in_browser(db_uri)