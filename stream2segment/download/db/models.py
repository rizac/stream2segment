"""
s2s Download database ORM

:date: Jul 15, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""

from sqlalchemy import event
# import declarative_base from io.db.models to be sqlalchemy 1.x vs 2.x compliant:
from stream2segment.io.db import models, declarative_base


Base = declarative_base(cls=models.Base)


class Download(Base, models.Download):  # noqa
    """Model representing the executed downloads"""
    pass


class Event(Base, models.Event):  # noqa
    """Model representing a seismic Event"""
    pass


class WebService(Base, models.WebService):
    """Model representing a web service (e.g., event web service)"""
    pass


class DataCenter(Base, models.DataCenter):
    """Model representing a Data center (data provider, e.g. EIDA Node)"""
    pass


# listen for insertion and updates and check Datacenter URLS (the call below
# is the same as decorating check_datacenter_urls_fdsn with '@event.listens_for'):
event.listens_for(DataCenter, 'before_insert')(models.check_datacenter_urls_fdsn)
event.listens_for(DataCenter, 'before_update')(models.check_datacenter_urls_fdsn)


class Station(Base, models.Station):
    """Model representing a Station"""
    pass


class Channel(Base, models.Channel):
    """Model representing a Channel"""
    pass


class Segment(Base, models.Segment):
    """Model representing a Waveform segment"""
    pass


class Class(Base, models.Class):
    """Model representing a segment class label"""
    pass


class ClassLabelling(Base, models.ClassLabelling):
    """Model representing a class labelling (or segment annotation), i.e. a
    pair (segment, class label)"""
    pass

