'''
Created on Jul 29, 2016

@author: riccardo

This module is an example module to query the *local* db after running
the main program stream2segments. Follow the instruction to analyze your data
'''
# this import is needed to have the sql alchemy session for IO db operations (see below)
from stream2segment.utils import get_session

def example():
    """
        This is an example function to get data from the local db.

        You can implement *any* function (like this one) in *any* module as long as you
        first type in the module:
        ```
            from stream2segment.workwithdb import get_session
        ```
        Read the comments below before proceeding
    """


    # =============================
    # 1) Introduction
    # =============================


    # We use slqAlchemy library to handle database IO
    # The library allow us to *write in python code* our tables (database schema)
    # and cares about the rest (creating a db, writing and reading to it, handle different
    # types of sql databases if in the future we will change, handle database migrations etc..)

    # The way sql alchemy works is writing python classes, called MODELS.
    # IMPORTANT: A MODEL is a python class and represents a database table.
    # A model instance (or instance only) is a python object and represents a table row

    # Let's import our models (PYTHON classes reflecting the database tables)
    # You should import them at module level, we do it here for code readability
    from stream2segment.io.db.models import Event, Station, Segment, Processing, Channel,\
        Run, DataCenter

    # have a look in stream2segment.io.db.models for inspecting the TYPE of the value
    # of each column. For instance
    #
    # class Event(FDSNBase):
    # """Events"""
    #
    # __tablename__ = "events"
    #
    # id = Column(String, primary_key=True, autoincrement=False)
    # time = Column(DateTime, nullable=False)
    # latitude = Column(Float, nullable=False)
    # ...
    #
    # Then, you see that we have an 'id' column primary key (you don't care), a 'time' column
    # of type 'DateTime' (python datetime object), a column 'latitude' of type float, and so on...

    # then, you instantiate the session. The session is the tool to communicate with the database
    session = get_session()
    # the line above loads db from config.yaml. To supply a custom sqlite path, use e.g.:
    # session = get_session("sqlite:///path/to/my/db.sqlite")
    


    # ================================
    # 2) Now we can do simple queries:
    # ================================

    # query *all* downloaded segments:
    segments = session.query(Segment).all()

    # The returned type is a python list of model instances (python objects).
    # how many of them? normal python way with lists:
    seg_num = len(segments)

    # Each instance attributes represents the table columns.
    # Again, if you forget which columns a table has, just look at the relative model in stream2segment.io.db.models
    # in this case it would be the 'Segment' model
    # So, if we are interested in the distance to the seismic event:
    first_seg = segments[0]
    distance_to_event = first_seg.event_distance_deg  # a python float

    # and so on. Quite simple. The only difference is the segments 'data' column, where
    # we store the mseed. That is 'Binary' data. We have implemented our custom function
    # to load binary data:
    from stream2segment.analysis.mseeds import loads
    mseed = loads(first_seg.data)  # obspy Stream object

    # Same for binary data in the processings table
    # First we get one instance (reflecting a table row, remember) the usual way:
    processings = session.query(Processing).all()
    first_pro = processings[0]

    # And then we read data, as we did above:
    # Note that we implemented our API in such a way that All these variables are Stream objects,
    # so  no the logic, methods and functions of these objects are the same!
    mseed_rem_resp_savewindow = loads(first_pro.mseed_rem_resp_savewindow)   # obspy Stream object
    wood_anderson_savewindow = loads(first_pro.wood_anderson_savewindow)   # obspy Stream object
    cum_rem_resp = loads(first_pro.cum_rem_resp)   # obspy Stream object

    # You want to save as pbspy Stream? As usual:
    # cum_rem_resp.write("/path/to/myfilename.mseed")
    # mseed.write("/path/to/myfilename.mseed")

    # small exceptions for ffts: you read them with loads (As usual)
    fft_rem_resp_t05_t95 = loads(first_pro.fft_rem_resp_t05_t95)
    fft_rem_resp_until_atime = loads(first_pro.fft_rem_resp_until_atime)
    # but the objects are not obspy Stream objects because they are on freq scale
    # So you won't have all the methods of the Stream objects but
    # accessing their data is THE SAME:
    data = fft_rem_resp_t05_t95.data  # as for obspy Stream.data, except data is numpy COMPLEX array
    df = fft_rem_resp_t05_t95.stats.delta  # as for obspy Stream.stats.delta, except that the unit is in Herz
    f0 = fft_rem_resp_t05_t95.stats.startfreq  # as for obspy Stream.stats.starttime, except that the unit is in Herz

    # So, do some (very stupid) computation:
    fft_rem_resp_times_two = 2 * fft_rem_resp_t05_t95.data

    # You could in principle save an fft-like array, but it's up to you to decide how
    # As said, this kind of objects cannot be converted back to obspy Stream(s)


    # =============================
    # 3) Working with relationships
    # =============================

    # Last thing: if you want more complex queries, there are a lot of methods
    # see e.g. here:
    # http://docs.sqlalchemy.org/en/latest/orm/tutorial.html#querying
    # but you can also use "our" relationships implemented via some sqlalchemy facilities
    # This might be less performant but it's easier to work with at the beginning:

    evt = first_seg.event  # an instance of the Event class representing the seismic event originating the segment
    cha = first_seg.channel  # an instance of the Channel class representing the segment channel
    dcn = first_seg.datacenter  # an instance of the Channel class representing the segment datacenter
    run = first_seg.run  # an instance of the Channel class representing the segment run

    # Examples:

    # give me all segments whose event's magnitude is between 3 and 4
    segments2 = []
    for seg in segments:
        if seg.event.magnitude > 3 and seg.event.magnitude < 4:
            segments2.append(seg)
    # now work with your filtered segments (sub) list..

    # give me all segments whose event is greater than a datetime
    from datetime import datetime
    time = datetime(2016, 1, 5)
    segments2 = []
    for seg in segments:
        if seg.event.time > time:
            segments2.append(seg)
    # now work with your filtered segments (sub) list..

    # In the same manner, a processings instance has an attribute 'segment':
    seg = first_pro.segment
    # give me all processed data from the run with id = 3 (knowing e.g., that the run was a specific one with interesting segments)
    processings2 = []
    for pro in processings:
        if pro.segment.run.id == 3:
            processings2.append(pro)
    # now work with your filtered segments (sub) list..

    # Note that a segment instance has an attribute 'processings' WHICH IS A LIST
    # why? there might be more processing rows for one segment, although this is NOT
    # currently implemented. Thus you should always get a zero- or one- element list
    procs = first_seg.processings
    if len(procs) > 0:
        first_pro = procs[0]


    # =====================
    # 4) Notes:
    # =====================

    # remember that, for working with numpy data, we already have implemented some functions in
    # stream2segment.analysis.__init__.py
    # Have a look there if you need it. Unfortunately, few of them are not tested.
    # The module is importable as:
    # from stream2segment.analysis import [whatever function is there]


    # FYI: the line:
    #
    # mseed = loads(first_seg.data)  # obspy Stream object
    #
    # ONLY FOR segment.data, is equivalent to:
    #
    # from obspy.core.stream import read
    # mseed = read(StringIO(first_seg.data))

    # ==================
    # That's all. Enjoy!


# this line is just to run this script and assure that has no errors (kind of test):
if __name__ == "__main__":
    example()