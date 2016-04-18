'''
Created on Feb 19, 2016

@author: riccardo
'''
# import yaml
import inspect
import datetime as dt
import time
import bisect
# import dateutil.parser as dparser

# # good: dateutil.parser parses strings into datetime
# # bad: dateutil.parser checks the "Z" at the end as UTC timezone, and prints it in isoformat
# # bad: dateutil.parser returns everything as datetime, maybe sometimes we want a date
# def str2isodate(string):
#     dtm = dparser.parse(string, yearfirst=True, dayfirst=False, ignoretz=True)
#     dtm_str = dtm.isoformat()
#     if dtm_str[-1] == 'Z':
#         dtm_str = dtm_str[:-1]


def prepare_datestr(string, ignore_z=True, allow_space=True):
    """
        "Prepares" string trying to make it datetime iso standard. This method basically gives the
        opportunity to remove the 'Z' at the end (denoting the zulu timezone) and replaces spaces
        with 'T'. NOTE: this methods returns the same string argument if any TypeError, IndexError
        or AttributeError is found.
        :param ignore_z: if True (the default), removes any 'Z' at the end of string, as 'Z' denotes
            the "zulu" timezone
        :param allow_spaces: if True (the default) all spaces of string will be replaced with 'T'.
        :return a new string according to the arguments or the same string object
    """
    # kind of redundant but allows unit testing
    try:
        if ignore_z and string[-1] == 'Z':
            string = string[:-1]

        if allow_space:
            string = string.replace(' ', 'T')
    except (TypeError, IndexError, AttributeError):
        pass

    return string


def to_datetime(string, ignore_z=True, allow_space=True):
    """
        Converts a date in string format (as returned by a fdnsws query) into
        a datetime python object. The inverse can be obtained by calling
        dt.isoformat() (which returns 'T' as date time separator, and optionally microseconds
        if they are not zero)
        Example:
        to_datetime("2016-06-01T09:04:00.5600Z")
        to_datetime("2016-06-01T09:04:00.5600")
        to_datetime("2016-06-01 09:04:00.5600Z")
        to_datetime("2016-06-01 09:04:00.5600Z")
        to_datetime("2016-06-01")
    """
    dtm = None
    string = prepare_datestr(string, ignore_z, allow_space)

    array = ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d', '%Y-%m-%dT%H:%M:%S.%f']

    for dtformat in array:
        try:
            dtm = dt.datetime.strptime(string, dtformat)
            break
        except ValueError:  # as exce:
            pass
        except TypeError:  # as terr:
            return None

    return dtm


# def getfargs():
#     """
#         Returns a dict of arguments and values of the function calling this function
#     """
#     thisframe = inspect.currentframe()
#     frame = inspect.getouterframes(thisframe)[1][0]
#     args, _, _, values = inspect.getargvalues(frame)
# #     print 'function name "%s"' % inspect.getframeinfo(frame)[2]
# #     for i in args:
# #         print "    %s = %s" % (i, values[i])
# #    return [(i, values[i]) for i in args]
#     return {i: values[i] for i in args}

class EstRemTimer():
    """
        An object used to calculate the estimated remaining time in loops. For a simple usage
        (print estimated remaining time, ert) just print this object, example:

            etr = EstRemTimer(N)
            for i in xrange(N):
                etr.get()   # returns the estimated remaining time (first time returns None)
                            # and increments the internal counter. Call get(False) not to
                            # increment the counter (returns the last ert)
                etr.done    # returns the numbers of iterations done (first time 0)
                etr.total   # returns N
                ... code here ...
    """
    def __init__(self, total_iterations, approx_to_seconds=True, use="median"):
        """
            Initializes an EstRemTimer for calculating the estimated remaining time (ert)
            :param: total_iterations the total iterations this object is assumed to use for 
            calculate the ert
            :type: total_iterations integer
            :param: approx_to_seconds: when True (the default) approximate the ert
            (timedelta object) to seconds
            :type: approx_to_seconds: boolean
            :param: use: if 'median' (case insensitive) calculates the estimated remaining time
            using the median of all durations of the iterations done. For any other string,
            use the mean. The default is "median" because it is less sensitive to skewed
            distributions, so basically iterations which take far more (or less) time than the
            average weight less in the computation of the ert.
        """
        self.total = total_iterations
        self.done = 0
        self._start_time = None  # time.time()
        self.ert = None
        self.approx_to_seconds = approx_to_seconds
        self._times = [] if use.lower() == "median" else None

    def get(self, increment=True, approx_to_seconds=None):
        """
            Gets the estimated remaing time etr. If increment is True, the returned object is None
            the first time this method is called, at subsequent calls it will be a timedelta object.
            If increment is False, returns the last calculated ert (which might be None)
            :param: increment: (True by default) returns the ert and increments the internal counter
            :type: increment: boolean
            :param: approx_to_seconds: sets whether the ert is approximated to seconds. If None (the
            default) the value of the argument approx_to_seconds passed in the constructor (True
            by default) is used
            :type: approx_to_seconds: boolean, or None
            :return: the estimated remaining time, or None
            :rtype: timedelta object, or None
        """
        if self._start_time is None:
            self._start_time = time.time()  # start now timing
            # first iteration, leave done to zero so that user can query the 'done' attribute
            # and it correctly displays the done iteration
        elif increment:
            self.done += 1
            if approx_to_seconds is None:
                approx_to_seconds = self.approx_to_seconds
            if self.done >= self.total:
                ret = dt.timedelta()
            else:
                elapsed_time = time.time() - self._start_time
                if self._times is not None:  # use median
                    # Find rightmost value less than or equal to ret:
                    i = bisect.bisect_right(self._times, elapsed_time)
                    self._times.insert(i, elapsed_time)
                    idx = len(self._times) / 2
                    ret = self._times[idx] if len(self._times) % 2 == 1 else \
                        (self._times[idx] + self._times[idx-1]) / 2
                    ret *= (self.total - self.done)
                    ret = dt.timedelta(seconds=int(ret + 0.5) if approx_to_seconds else ret)
                    self._start_time = time.time()  # re-start timer (for next iteration)
                else:
                    ret = estremttime(elapsed_time, self.done, self.total, approx_to_seconds)
            self.ert = ret
        return self.ert


def estremttime(elapsed_time, iteration_number, total_iterations, approx_to_seconds=True):
    """Called within a set of N=total_iterations "operations" (usually in a for loop) started since
    elapsed_time, this method returns a timedelta object representing the ESTIMATED remaining time
    when the iteration_number-th operation has been finished.
    Estimated means that the remaining time is calculated as if each of the remaining operations
    will take in average the average time taken for the operations done, which might not always be
    the case
    :Example:
    import time
    start_time = time.time()
    for i, elm in enumerate(events):  # events being e.g., a list / tuple or whatever
        elapsed = time.time() - start_time
        est_rt = str(estremttime(elapsed, i, len(events))) if i > 0 else "unwknown"
        ... your code here ...

    :param: elapsed_time: the time elapsed since the first operation (operation 0) started
    :type: elapsed_time a timedelta object, or any type castable to float (int, floats, numeric
        strings)
    :param: iteration_number: the number of operations done
    :type: iteration_number: a positive int
    :param: total_iterations: self-explanatory, specifies the total number of operations expected
    :type: total_iterations: a positive int greater or equal than iteration_number
    :param: approx_to_seconds: True by default if missing, returns the remaining time aproximated
        to seconds, which is sufficient for the typical use case of a process remaining time
        which must be shown to the user
    :type: approx_to_seconds: boolean
    :return: the estimated remaining time according to elapsed_time, which is the time taken to
        process iteration_number operations of a total number of total_iterations operations
    :rtype: timedelta object. Note that it's string value (str function) can be called to display
    the text of the estimated remaining time
    """
    if isinstance(elapsed_time, dt.timedelta):
        elapsed_time = elapsed_time.total_seconds()  # is a float
    else:
        elapsed_time = float(elapsed_time)  # to avoid rounding below (FIXME: use true division?)
    remaining_seconds = (total_iterations - iteration_number) * (elapsed_time / iteration_number)
    dttd = dt.timedelta(seconds=int(remaining_seconds+0.5)
                        if approx_to_seconds else remaining_seconds)
    return dttd


# # Original function
# def to_datetime(date_str):
#     """
#         Converts a date in string format (as returned by a fdnsws query) into
#         a datetime python object
#         Example:
#         to_datetime("2016-06-01T09:04:00.5600Z")
#         to_datetime("2016-06-01T09:04:00.5600")
#         to_datetime("2016-06-01 09:04:00.5600Z")
#         to_datetime("2016-06-01 09:04:00.5600Z")
#         to_datetime("2016-06-01")
#     """
#     # Note: dateutil.parser.parse(string, yearfirst=True, dayfirst=False, ignoretz=True)
#     # does ALMOST the same except that:
#     # ignoretz ignores all timezones, we want to ignore only Z
#     # '00-09-03 20:56:35.450686Z' is converted to datetime.datetime(2000, 9, 3, 20, 56, 35, 450686)
#     # whereas:
#     # datetime.datetime(0, 9, 3, 20, 56, 35, 450686) raises a ValueError which we want to have
#     # Thus, this function
#     try:
#         date_str = date_str.replace('-', ' ').replace('T', ' ')\
#             .replace(':', ' ').replace('.', ' ').replace('Z', '').split()
#         return dt.datetime(*(int(value) for value in date_str))
#     except (AttributeError, IndexError, ValueError, TypeError):
#         return None
# 
# # Original function stricter. Tries to be like a python parser BUT as fast as to_datetime above
# def to_datetime2(date_str, ignore_z=True, allow_space=True):
#     """
#         Converts a date in string format (as returned by a fdnsws query) into
#         a datetime python object
#         Example:
#         to_datetime("2016-06-01T09:04:00.5600Z")
#         to_datetime("2016-06-01T09:04:00.5600")
#         to_datetime("2016-06-01 09:04:00.5600Z")
#         to_datetime("2016-06-01 09:04:00.5600Z")
#         to_datetime("2016-06-01")
#     """
#     # Note: dateutil.parser.parse(string, yearfirst=True, dayfirst=False, ignoretz=True)
#     # does ALMOST the same except that:
#     # ignoretz ignores all timezones, we want to ignore only Z
#     # '00-09-03 20:56:35.450686Z' is converted to datetime.datetime(2000, 9, 3, 20, 56, 35, 450686)
#     # whereas:
#     # datetime.datetime(0, 9, 3, 20, 56, 35, 450686) raises a ValueError which we want to have
#     # Thus, this function
#     if ignore_z and date_str[-1] == 'Z':
#         date_str = date_str[:-1]
# 
#     if allow_space:
#         date_str = date_str.replace(' ', 'T')
#     dsplit = date_str.split('T')
# 
#     try:
#         assert len(dsplit) in (1, 2)
#         split1 = dsplit[0].split("-")
#         if len(dsplit) == 2:
#             split2 = dsplit[1].split(":")
#             split1.extend(split2[:-1])
#             split1.extend(split2[-1].split('.'))
#         return dt.datetime(*(int(value) for value in split1))
#     except (AssertionError, AttributeError, IndexError, ValueError, TypeError):
#         return None


# import time
# 
# if __name__ == "__main__":
#     import os
#     print os.path.abspath("seed")
#     
#     date_ = "2006-01-05" # "2006-01-05 12:34:56Z"
#     N = 10000
#     clock_ = time.clock()
#     for i in xrange(N):
#         str2isodate(date_)
# 
#     c1 = str(time.clock() - clock_)
#     print "str2isodate " + str(c1)
# 
#     clock_ = time.clock()
#     for i in xrange(N):
#         str2isodate_(date_)
# 
#     c1 = str(time.clock() - clock_)
#     print "str2isodate_ " + str(c1)
    
#     clock_ = time.clock()
#     for i in xrange(N):
#         to_datetime(date_)
# 
#     c1 = str(time.clock() - clock_)
#     print "to_datetime: " + str(c1)
    
#     clock_ = time.clock()
#     for i in xrange(N):
#         to_datetime2(date_)
# 
#     c1 = str(time.clock() - clock_)
#     print "to_datetime2: " + str(c1)
#  