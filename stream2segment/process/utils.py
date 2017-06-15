'''
Created on Feb 24, 2017

@author: riccardo
'''
# from obspy.core import read, Stream, Trace
# from cStringIO import StringIO
# from stream2segment.io.db import models
# from sqlalchemy.sql.expression import and_
# from sqlalchemy.orm.session import object_session
# from urlparse import urlparse


# def dcname(datacenter):
#     """Returns the datacenter name. Uses urlparse"""
#     return urlparse(datacenter.station_url).netloc
# 
# 
# def segstr(segment):
#     """Utility to print a segment identifier uniformely across sub-programs"""
#     return "{} [{}, {}] (id: {})".format(segment.channel_id,
#                                          segment.start_time,
#                                          segment.end_time,
#                                          segment.id)



# # FIXME: not implemented! remove?!!
# def has_data(segment, session):
#     pass


# def linfunc(object):
# 
#     def __init__(self, dict):
#         self.intervals = [parsechunk(h) for h in dict.iterkeys()]
#         self.vals = [parsechunk(v) for h in dict.itervalues()]
#         
# 
#     @staticmethod
#     def parsechunk(chunk):
#         
#         
#         
#     
#     def __call__(self, value):
#         pass
#         # index using binary search of values, return nan if not found or the value if found
# 
# def chunk(object):
# 
#     def __init__(self, chunkstr):
#         chunkstr = chunkstr.strip()
#         if chunkstr[0] in ('[', ']'):
#             assert len(chunkstr) > 1 and chunkstr[-1] in ('[', ']')
#         self.l = chunkstr[0]
#         self.r = chunkstr[-1]
#         chunkstrs = chunkstr[1:-1].split(",")
#         assert len(chunkstrs) == 2
#         self.lval = float(chunkstrs[0])
#         self.rval = float(chunkstrs[1])
#         assert self.lval <= self.rval

#     def isin(self, value):
#         return (value > self.lval and value < self.rval) or \
#             (self.lval == '[' and value == self.lval) or (self.rval == ']' and value == self.rval)


