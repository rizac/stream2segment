'''
Created on Feb 25, 2016

@author: riccardo
'''
import os
import re
import sys
from os import listdir
from os.path import isfile, join, basename, isdir
# import numpy as np
from obspy import read
# from obspy.core.utcdatetime import UTCDateTime
import matplotlib.pyplot as plt

from matplotlib.backend_bases import NavigationToolbar2
NavigationToolbar2.home = lambda self, *args, **kwargs: plot_other(self, None)
NavigationToolbar2.back = lambda self, *args, **kwargs: plot_other(self, -1)
NavigationToolbar2.forward = lambda self, *args, **kwargs: plot_other(self, 1)

# NavigationToolbar2.toolitems[0][1] = 'Plot first mseed'  # home tooltip
# NavigationToolbar2.toolitems[0][2] = 'Plot next mseed'  # back tooltip
# NavigationToolbar2.toolitems[0][3] = 'Plot previous mseed'  # forward tooltip

origtime_re = re.compile("origtime_(\\d*(?:\\.\\d+)?)")  # match a float following origtime

curr_pos = 0
fig = plt.figure()
files = []


def plot_other(self, key=0):  # key = None: home (print first plot), +1: print next, -1: print prev.
    global curr_pos
    old_curr_pos = curr_pos
    curr_pos = 0 if key is None else (curr_pos + key) % len(files)
    if old_curr_pos != curr_pos:
        plot(self.canvas, curr_pos)


def plot(canvas, index):
    data = None
    canvas.figure.clear()
    try:
        data = read(files[index])
    except (IOError, TypeError) as ioerr:
        canvas.figure.suptitle(str(ioerr))
        canvas.draw()
        return

    data.plot(fig=fig)  # , block=True)
    canvas.set_window_title(basename(files[index] + " (%d of %d)" % (index+1, len(files))))

    mobj = origtime_re.search(files[index])
    if mobj and len(mobj.groups()) == 1:
        try:
            datenum = float(mobj.group(1))
            for axes in fig.get_axes():
                ylim = axes.get_ylim()
                # axes.vlines(datenum, ylim[0], ylim[1], color='#00ee00', label='origTime')
                axes.plot([datenum, datenum], ylim, color='#00ee00', label='origTime')
                # print "origtime: " + str(datenum)
        except (TypeError, ValueError) as exc:
            print str(exc)

    canvas.draw()


if __name__ == '__main__':
    # global files
    if len(sys.argv) < 2:
        print "please specify a valid directory of mseed files"
        sys.exit(1)

    dir_ = sys.argv[1]
    if not isdir(dir_):
        print "'%s' is not a valid directopry" % dir_
        sys.exit(1)

    dir_ = sys.argv[1]
    files = [join(dir_, f) for f in listdir(dir_) if isfile(join(dir_, f))]
    plot(fig.canvas, 0)
    plt.show(block=True)

# SOME INFOS FOUND BROWSING INTERNET (SEOM OF THEM USED ABOVE):

# A) THIS HIDES THE TOOLBAR:
# import matplotlib as mpl
# mpl.rcParams['toolbar'] = 'None'

# B) THIS OVERRIDES THE DEFAULT BEHAVIOR:
# See http://stackoverflow.com/questions/14896580/matplotlib-hooking-in-to-home-back-forward-button-events
# from matplotlib.backend_bases import NavigationToolbar2
# home = NavigationToolbar2.home
# def new_home(self, *args, **kwargs):
#     print 'new home'
#     home(self, *args, **kwargs)
# NavigationToolbar2.home = new_home

# C) THIS OVERRIDES THE MOUSE / KEY EVENTS ON THE PLOT (NOT THE NAVIGATION TOOLBAR):
# See http://matplotlib.org/users/event_handling.html
# def onclick(event):
#     print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
#         event.button, event.x, event.y, event.xdata, event.ydata)
# cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # To set the title of the axes withion a figure:
    # self.canvas.figure.suptitle('ah ah ' + str(key))

    # to set the window title
    # self.canvas.set_window_title()

#    To open a file dialog box (within matplotlib) do (NOTE: NOT TESTED!):
#     import tkFileDialog as fd
#     fname = fd.askopenfilename(initialfile=files[0])
#     fname
