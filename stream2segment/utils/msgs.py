'''
Created on Nov 30, 2016

Handles the messy amount of log messages trying to harmonize them a bit.
Usage is rather simple
```
import msgs

msgs.format(ValueError('arg'))
>>> 'ValueError: arg'

msgs.format(ValueError('arg'), 'http://mydomain')
>>> 'ValueError: arg. URL: http://mydomain'

msgs.format('arg', 'http://mydomain')
>>> 'arg. URL: http://mydomain'

# event has an attribute 'id' and event.id = 'my_id'
msgs.calc.dropped_evt(event)
>>> 'event 'my_id' discarded'

msgs.calc.dropped_evt(event, 'during my hard calculation'))
>>> 'event 'my_id' discarded during my hard calculation'

msgs.calc.dropped_evt(event,  None, ValueError('unformatted data (NaN)))
>>> 'event 'my_id' discarded. ValueError: unformatted data (NaN)'

msgs.calc.dropped_evt(event, 'during my hard calculation', ValueError('unformatted data (NaN)'))
>>> 'event 'my_id' discarded during my hard calculation. ValueError: unformatted data (NaN)'

msgs.calc.dropped_evt(event, 'during my hard calculation', 'unformatted data (NaN)'))
>>> 'event 'my_id' discarded during my hard calculation. unformatted data (NaN)'

msgs.query.dropped_evt(1)
>>> '1 event(s) discarded'

msgs.query.dropped_evt(1, 'http://mydomain')
>>> '1 event(s) discarded. URL: http://mydomain'

msgs.query.dropped_evt(1, 'http://mydomain', 'invalid data')
>>> '1 event(s) discarded. invalid data. URL: http://mydomain'

msgs.query.dropped_evt(1, 'http://mydomain', ValueError('invalid data'))
>>> '1 event(s) discarded. ValueError: invalid data. URL: http://mydomain'

msgs.db.dropped_evt(1)
>>> '1 event(s) discarded (not saved to db). Unknown db error'

msgs.db.dropped_evt(1, 'http://mydomain')
>>> '1 event(s) discarded (not saved to db). Unknown db error. URL: http://mydomain'

msgs.db.dropped_evt(1, None, ValueError("(NaN)"))
>>> '1 event(s) discarded (not saved to db). ValueError: (NaN)'

msgs.db.dropped_evt(1, 'http://mydomain', ValueError("(NaN)"))
>>> '1 event(s) discarded (not saved to db). ValueError: (NaN). URL: http://mydomain'

msgs.db.dropped_evt(1, 'http://mydomain', "(NaN)")
>>> '1 event(s) discarded (not saved to db). (NaN). URL: http://mydomain'
```


@author: riccardo
'''


def format(exc_or_msg, url=None):  # @ReservedAssignment # pylint:disable=redefined-builtin
    '''
    formats the given exception or message (string) appending an optional url, if given
    ```
    format(ValueError('arg'))
    >>> 'ValueError: arg'
    format(ValueError('arg'), 'http://mydomain')
    >>> 'ValueError: arg. URL: http://mydomain'
    format('arg', 'http://mydomain')
    >>> 'arg. URL: http://mydomain'
    :param exc_or_msg: (Exception or message) and exception or a string
    :param url: (string or None, default: None) an optional url originating the error
    :return: string
    '''
    if isinstance(exc_or_msg, Exception):
        exc_or_msg = _colonjoin(exc_or_msg.__class__.__name__, str(exc_or_msg))
    url = "" if not url else _colonjoin("URL", url)
    return _dotjoin(exc_or_msg, url)


# utilities for "smart" string joins:

def _dotjoin(str1, str2):
    return ". ".join(x for x in [str1, str2] if x)


def _colonjoin(str1, str2):
    return ": ".join(x for x in [str1, str2] if x)


def _join(str1, str2):
    return " ".join(x for x in [str1, str2] if x)


def _parjoin(str1, str2):
    if str2:
        str2 = "(%s)" % str2
    return " ".join(x for x in [str1, str2] if x)


def _discarded(msg_header, url=None, msg_or_exc=None):
    return format(_dotjoin(msg_header, format(msg_or_exc)), url)


def _id(obj):
    return str(obj.id) if hasattr(obj, "id") else str(obj)


def _(num, singular, plural):
    return singular if num == 1 else plural


class calc(object):  # pylint:disable=invalid-name
    """ any error/message related to calculating/processing data"""

    @staticmethod
    def dropped_evt(event, where=None, msg_or_exc=None):
        '''
        Formats an error or message related to an event discarded during calculation
        ```
        # event has an attribute 'id' and event.id = 'my_id'
        calc.dropped_evt(event)
        >>> 'event 'my_id' discarded'
        calc.dropped_evt(event, 'during my hard calculation'))
        >>> 'event 'my_id' discarded during my hard calculation'
        calc.dropped_evt(event,  None, ValueError('unformatted data (NaN)))
        >>> 'event 'my_id' discarded. ValueError: unformatted data (NaN)'
        calc.dropped_evt(event, 'during my hard calculation', ValueError('unformatted data (NaN)'))
        >>> 'event 'my_id' discarded during my hard calculation. ValueError: unformatted data (NaN)'
        calc.dropped_evt(event, 'during my hard calculation', 'unformatted data (NaN)'))
        >>> 'event 'my_id' discarded during my hard calculation. unformatted data (NaN)'
        ```
        :param event: an object identifying an event. If has the attribute 'idi' (usually the case
        if the argument is a database model instance), it's id is printed, otherwise its `str` value
        :param where: a string
        :param msg_or_exc: a message string or an exception
        '''
        return _discarded(_join("event '%s' discarded" % _id(event), where), None, msg_or_exc)

    @staticmethod
    def dropped_sta(station, where=None, msg_or_exc=None):
        '''
        Formats the given error or message related to a station discarded during calculation
        Refer to the doc of :ref:`msgs.calc.dropped_evt` with 'event' replaced by 'station'
        '''
        return _discarded(_join("station '%s' discarded" % _id(station), where), None, msg_or_exc)

    @staticmethod
    def dropped_seg(segment, where=None, msg_or_exc=None):
        '''
        Formats the given error or message related to a segment discarded during calculation
        Refer to the doc of :ref:`msgs.calc.dropped_evt` with 'event' replaced by 'segment'
        '''
        return _discarded(_join("segment '%s' discarded" % _id(segment), where), None, msg_or_exc)


class query(object):  # pylint:disable=invalid-name
    """ any error/message related to getting data via url query"""

    @staticmethod
    def empty(url=None):
        '''
        Formats an error or message related to empty data from a given (optional) url
        ```
        query.empty()
        >>> 'empty data'
        query.empty('http://mydomain')
        >>> 'empty data. URL: http://mydomain'
        ```
        :param url: an optional string or None (default: None) denoting a source url
        '''
        return format("empty data", url)

    @staticmethod
    def dropped_evt(num, url=None, msg_or_exc=None):
        '''
        Formats an error or message related to a number of events discarded from a given (optional)
        url
        ```
        query.dropped_evt(1)
        >>> '1 event(s) discarded'
        query.dropped_evt(1, 'http://mydomain')
        >>> '1 event(s) discarded. URL: http://mydomain'
        query.dropped_evt(1, 'http://mydomain', 'invalid data')
        >>> '1 event(s) discarded. invalid data. URL: http://mydomain'
        query.dropped_evt(1, 'http://mydomain', ValueError('invalid data'))
        >>> '1 event(s) discarded. ValueError: invalid data. URL: http://mydomain'
        ```
        :param num: (integer) the number of events discarded
        :param url: (string or None, default: None) an optional url where event was read from
        :param msg_or_exc: (string or Exception or None. Default: None) an optional message or
        exception describing the reason of this message
        '''
        return _discarded("%d event(s) discarded" % num, url, msg_or_exc)

    @staticmethod
    def dropped_sta(num, url=None, msg_or_exc=None):
        '''
        Formats an error or message related to a number of events discarded from a given (optional)
        url. Refer to the doc of :ref:`msgs.query.dropped_evt` replacing 'event' with 'station'
        '''
        return _discarded("%d station(s) discarded" % num, url, msg_or_exc)

    @staticmethod
    def dropped_seg(num, url=None, msg_or_exc=None):
        '''
        Formats an error or message related to a number of events discarded from a given (optional)
        url. Refer to the doc of :ref:`msgs.query.dropped_evt` replacing 'event' with 'segment'
        '''
        return _discarded("%d segment(s) discarded" % num, url, msg_or_exc)


class db(object):  # pylint:disable=invalid-name
    """any error/message related to writing downloaded data to a storage (local db... etc)"""

    @staticmethod
    def dropped_evt(num, url=None, msg_or_exc=None):
        '''
        Formats an error or message related to a number of events not saved to db
        ```
        db.dropped_evt(1)
        >>> '1 event(s) discarded (not saved to db). Unknown db error'
        db.dropped_evt(1, 'http://mydomain')
        >>> '1 event(s) discarded (not saved to db). Unknown db error. URL: http://mydomain'
        db.dropped_evt(1, None, ValueError("(NaN)"))
        >>> '1 event(s) discarded (not saved to db). ValueError: (NaN)'
        db.dropped_evt(1, 'http://mydomain', ValueError("(NaN)"))
        >>> '1 event(s) discarded (not saved to db). ValueError: (NaN). URL: http://mydomain'
        db.dropped_evt(1, 'http://mydomain', "(NaN)")
        >>> '1 event(s) discarded (not saved to db). (NaN). URL: http://mydomain'
        ```
        :param num:
        :param url:
        :param msg_or_exc:
        '''
        return _discarded("%d event(s) discarded (not saved to db)" % num, url,
                          msg_or_exc or "unknown db error")

    @staticmethod
    def dropped_dc(num, url=None, msg_or_exc=None):
        '''
        Formats an error or message related to a number of data centers not saved to db.
        Refer to the doc of :ref:`msgs.db.dropped_evt` replacing 'event' with 'data center'
        '''
        return _discarded("%d data center(s) discarded (not saved to db)" % num, url,
                          msg_or_exc or "unknown db error")

    @staticmethod
    def dropped_sta(num, url=None, msg_or_exc=None):
        '''
        Formats an error or message related to a number of data centers not saved to db.
        Refer to the doc of :ref:`msgs.db.dropped_evt` replacing 'event' with 'station'
        '''
        return _discarded("%d station(s) discarded (not saved to db)" % num, url,
                          msg_or_exc or "unknown db error")

    @staticmethod
    def dropped_cha(num, url=None, msg_or_exc=None):
        '''
        Formats an error or message related to a number of data centers not saved to db.
        Refer to the doc of :ref:`msgs.db.dropped_evt` replacing 'event' with 'station'
        '''
        return _discarded("%d channel(s) discarded (not saved to db)" % num, url,
                          msg_or_exc or "unknown db error")

    @staticmethod
    def dropped_seg(num, url=None, msg_or_exc=None):
        '''
        Formats an error or message related to a number of data centers not saved to db.
        Refer to the doc of :ref:`msgs.db.dropped_evt` replacing 'event' with 'segment'
        '''
        return _discarded("%d segment(s) discarded (not saved to db)" % num, url,
                          msg_or_exc or "unknown db error")


if __name__ == "__main__":
    from itertools import product
    exc = ValueError("unformatted data (invalid or NaN)")
    url_ = "http://mydomain/very/nice?query=something=cool&something=less_cool"
    class mock_db_inst(object):
        id='my_id'
    inst = mock_db_inst()
    print format(exc)
    print format(exc, url_)
    print calc.dropped_evt(inst, "during my hard calculation", exc)
    print calc.dropped_sta(inst, "during my hard calculation", exc)
    print calc.dropped_seg(inst, "during my hard calculation", exc)
    print calc.dropped_evt(inst, "during my hard calculation", 'wat?')
    print calc.dropped_sta(inst, "during my hard calculation", 'wat?')
    print calc.dropped_seg(inst, "during my hard calculation", 'wat?')
    print calc.dropped_evt(inst, "", exc)
    print calc.dropped_sta(inst, "", exc)
    print calc.dropped_seg(inst, "", exc)
    print calc.dropped_evt(inst, "during my hard calculation")
    print calc.dropped_sta(inst, "during my hard calculation")
    print calc.dropped_seg(inst, "during my hard calculation")
    print calc.dropped_evt(inst)
    print calc.dropped_sta(inst)
    print calc.dropped_seg(inst)

    print query.empty()
    print query.empty(url_)
    print query.dropped_evt(1, url_, exc)
    print query.dropped_seg(1, url_, exc)
    print query.dropped_sta(1, url_, exc)
    print query.dropped_evt(1, "", exc)
    print query.dropped_seg(1, "", exc)
    print query.dropped_sta(1, "", exc)
    print query.dropped_evt(1, url_)
    print query.dropped_seg(1, url_)
    print query.dropped_sta(1, url_)
    print query.dropped_evt(1)
    print query.dropped_seg(1)
    print query.dropped_sta(1)

    print db.dropped_evt(1, url_, exc)
    print db.dropped_seg(1, url_, exc)
    print db.dropped_sta(1, url_, exc)
    print db.dropped_evt(1, "", exc)
    print db.dropped_seg(1, "", exc)
    print db.dropped_sta(1, "", exc)
    print db.dropped_evt(1, url_)
    print db.dropped_seg(1, url_)
    print db.dropped_sta(1, url_)
    print db.dropped_evt(1)
    print db.dropped_seg(1)
    print db.dropped_sta(1)
    print db.dropped_dc(1, url_, exc)
    print db.dropped_dc(1, "", exc)
    print db.dropped_dc(1, url_)
    print db.dropped_dc(1)
