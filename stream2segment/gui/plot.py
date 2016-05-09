'''
Program to annotate classes of mseeds previously downloaded with this program
Created on Feb 25, 2016

@author: riccardo
'''
# import matplotlib
# matplotlib.use('Qt4Agg')
import sys
from stream2segment.io.db import Reader
import re
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons

# Overriding default buttons behaviour:
from matplotlib.backend_bases import NavigationToolbar2

# set here left or right:
plot_position = 'right'

# Rewrite tooltiptexts for back and forward buttons (we copy the whole tuple defined in
# NavigatorToolbar2 although it's quite inefficient because it's easier than modifying a tuple of
# tuples
NavigationToolbar2.toolitems = toolitems = (
        ('Home', 'Reset original view', 'home', 'home'),
        ('Back', 'Back to  previous plot', 'back', 'back'),
        ('Forward', 'Forward to next plot', 'forward', 'forward'),
        (None, None, None, None),
        ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
        ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
        (None, None, None, None),
        ('Subplots', 'Configure subplots', 'subplots', 'configure_subplots'),
        ('Save', 'Save the figure', 'filesave', 'save_figure'),
      )
# titems_lst = list(NavigationToolbar2.toolitems)
# titems_lst.insert(4, ('Home', 'Reset original view', 'stock_refresh', 'home'))
# NavigationToolbar2.toolitems = tuple(titems_lst)

# NavigationToolbar2.home = new_home
NavigationToolbar2.back = lambda self, *args, **kwargs: plot_other(self, -1)
NavigationToolbar2.forward = lambda self, *args, **kwargs: plot_other(self, 1)

# global vars:
curr_pos = 0
fig = plt.figure(figsize=(16, 9), dpi=80)
dbreader = None
# keep a reference to the figure title so that we do not need to create one every time
# (avoiding checking for titles etcetera)
infotext = fig.suptitle("", multialignment='left', fontsize=11, family='monospace',
                        horizontalalignment='left', verticalalignment='top')
# http://matplotlib.org/users/text_props.html:
# horizontalalignment controls whether the x positional argument for the text indicates the left,
# center or right side of the text bounding box. verticalalignment controls whether the y
# positional argument for the text indicates the bottom, center or top side of the
# text bounding box. multialignment, for newline separated strings only, controls whether
# the different lines are left, center or right justified

# some global variables for axes and controls dimensions:
# padding around figure. This is the padding of the shorter axes, usually the y one
# (the other one will be adjusted accordingly)
fig_padding = 0.025
legend_width = 0.35

# these variables are global so that we can make them interactive
# the axes housing the radiobuttons:
# Note: the position (0.95, 0.3) will be RESET later, here only elements 3 and 4 (width and height)
# are set!
rax = plt.axes([0.95, 0.3, legend_width, legend_width*0.75],  # axisbg='lightgoldenrodyellow',
               title=Reader.annotated_class_id_colname,
               aspect='equal')  # the last one makes radio buttons circles and not ellipses
# the radiobuttons widget:
radiobuttons = None

_pass_set_flag = False  # this flag is set in updateradiobuttons and used in setclass


def setclass(label):
    if _pass_set_flag:
        return
    classes_df = dbreader.get_classes()
    label_ = re.sub("\\s+\\(\\s*\\d+\\s*\\)\\s*$", "", label)
    class_id = classes_df[classes_df['Label'] == label_].iloc[0]['Id']
    dbreader.set_class(curr_pos, class_id)
    update_radio_buttons()


def plot_other(self, key=0):  # key = None: home (print first plot), +1: print next, -1: print prev.
    global curr_pos
    old_curr_pos = curr_pos
    curr_pos = 0 if key is None else (curr_pos + key) % dbreader.seg_count()
    if old_curr_pos != curr_pos:
        plot(self.canvas, curr_pos)


def mseed_axes_iterator(fig):
    """returns all axes within the figure which are mseed plots (assuming they are all BUT rax)"""
    axes_ = fig.get_axes()
    for a in axes_:
        if a != rax:
            yield a


def getinfotext(metadata):
    """Returns a nicely formatted string from the mseed metadata read from db"""
    first_col_chars = max(len(str(key)) for key in metadata.keys())
    max_second_col_chars = 42

    # custom str function replacement for the dict values:
    def ztr(data):
        val = str(data)
        if len(val) <= max_second_col_chars:
            return val
        splits = [(i * max_second_col_chars, (i+1) * max_second_col_chars)
                  for i in xrange(len(val)/(max_second_col_chars))]
        if len(val) % max_second_col_chars > 0:
            splits.append((splits[-1][1], None))
        return ("\n " + (" " * first_col_chars)).join(val[s[0]:s[1]] for s in splits)

    # print the metadata on the figure title. Set the format string:
    frmt_str = "{0:" + str(first_col_chars) + "} {1}"
    # first remove the annotated class id cause we will use the editor for that:
    metadata.pop(Reader.annotated_class_id_colname, None)
    # set the string:
    sorted_keys = sorted(metadata)
    title_str = "\n".join(frmt_str.format(str(k), ztr(metadata[k])) for k in sorted_keys)
    return title_str


def plot(canvas, index):

    canvas.set_window_title("%s: FILE %d OF %d" % (dbreader.db_uri, index+1, dbreader.seg_count()))
    data = None
    # canvas.figure.clear() this is BAD cause the radiobuttons do not work anymore. Then
    # clear only axes of interest:
    for a in mseed_axes_iterator(fig):
        if a != rax:
            fig.delaxes(a)

    try:
        data = dbreader.get_segment(index)
    except (IOError, TypeError) as ioerr:
        # canvas.figure.suptitle(str(ioerr))
        errmsg = "Unable to show data plot(s):\n%s: %s" % (str(ioerr.__class__.__name__), str(ioerr))
        infotext.set_text(errmsg)
        # canvas.draw()
        return
    data.plot(fig=fig, draw=False)  # , block=True)

    axez = sorted(mseed_axes_iterator(fig), key=lambda ax: ax.get_position().y0)

    # calculate fig padding:
    # NOTE: fig_padding does not include axis ticks. For the vertical ones is ok, as they are
    # a single line height, for the horizontal one we add a bit more space:
    fig_padding_h, fig_padding_w = fig_padding, fig_padding
    sizez = fig.get_size_inches()
    if sizez[0] > sizez[1]:
        fig_padding_w *= sizez[1] / sizez[0]
    elif sizez[1] > sizez[0]:
        fig_padding_h *= sizez[0] / sizez[1]

    ypos = fig_padding_h
    additional_left_margin = 0.03
    height = (1.0 - 2*(fig_padding_h)) / len(axez)
    width = (1.0 - 3*(fig_padding_w) - additional_left_margin - legend_width)
    axez_x = fig_padding_w + additional_left_margin if plot_position == 'left' else \
        legend_width + 2*fig_padding_w + additional_left_margin
    for axs in axez:
        # testing: do we really set the ypos on the right axes?
        # print str(axs.get_position().y0) + " " + str(ypos)
        axs.set_position([axez_x, ypos, width, height])
        ypos += height

    # Set info text on the figure title (NOTE: it is placed on the right)
    infotext.set_text(getinfotext(dbreader.get_metadata(index)))

    # adjust dimensions:
    xxx = 1 - legend_width - fig_padding_w if plot_position == 'left' else \
        fig_padding_w
    # infotext:
    infotext.set_position((xxx, 1-fig_padding_h))
    # set radiobuttons position:
    rax_pos = rax.get_position()
    rax.set_position([xxx, fig_padding_h, legend_width, rax_pos.height])
    # update the selected radio button
    update_radio_buttons(update_texts=False)


def update_radio_buttons(update_texts=True):
    """
        Updates the radio buttons
        :param update_texts: updates the label texts (with the counts for each class label)
        :param update_active: if True, sets the active button according to the selected mseed class
        SET TO FALSE IF CALLING THIS FROM WITHIN A MOUSE CLICK ON ONE RADIO BUTTON TO AVOID
        INFINITE LOOPS
    """
    global radiobuttons, dbreader
    classes_df = dbreader.get_classes()

    if update_texts or radiobuttons is None:
        clbls = classes_df['Label'].tolist()
        counts = classes_df['Count'].tolist()
        radiolabels = ["%s (%d)" % (s, v) for s, v in zip(clbls, counts)]
        if radiobuttons is None:
            radiobuttons = RadioButtons(rax, radiolabels)
            # Resize all radio buttons in `r` collection by fractions `f`"
            for circle in radiobuttons.circles:
                circle.set_radius(circle.get_radius() * .75)

            radiobuttons.on_clicked(setclass)  # set this after 'update_radio_buttons' above
        else:
            for text, label in zip(radiobuttons.labels, radiolabels):
                text.set_text(label)

    global _pass_set_flag
    _pass_set_flag = True
    class_id = dbreader.get_class(curr_pos)
    radiobuttonindex = classes_df[classes_df['Id'] == class_id].index[0]
    radiobuttons.set_active(radiobuttonindex)
    _pass_set_flag = False


def main(db_uri):
    global dbreader
    dbreader = Reader(db_uri)

    plot(fig.canvas, 0)

    plt.show(True)


if __name__ == '__main__':
    # global files
    if len(sys.argv) < 2:
        print "please specify a valid directory of mseed files"
        sys.exit(1)

    path_ = sys.argv[1]
    # files = [join(dir_, f) for f in listdir(dir_) if isfile(join(dir_, f))]
        # plt.show(block=True)
    
    main(path_)


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
