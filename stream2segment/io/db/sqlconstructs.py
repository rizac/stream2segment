"""
Module for registering non-standard SQL constructs

The following SQL functions work with both SQLite and Postgres. If you
add support for new databases, you should modify the code below. For info:
http://docs.sqlalchemy.org/en/latest/core/compiler.html#further-examples)
"""

from sqlalchemy import Integer, String, Float
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import FunctionElement


class strpos(FunctionElement):
    name = 'strpos'
    type = Integer()


@compiles(strpos)
def standard_strpos(element, compiler, **kw):
    """delegates strpos to the strpos db function"""
    return compiler.visit_function(element)


@compiles(strpos, 'sqlite')
def sqlite_strpos(element, compiler, **kw):
    return "instr(%s)" % compiler.process(element.clauses)
    # return func.instr(compiler.process(element.clauses))


# function `concat`

class concat(FunctionElement):
    name = 'concat'
    type = String()


@compiles(concat)
def standard_concat(element, compiler, **kw):
    return compiler.visit_function(element)


@compiles(concat, 'sqlite')
def sqlite_concat(element, compiler, **kw):
    return " || ".join(compiler.process(c) for c in element.clauses)


# two utility functions to return the timestamp from a datetime
def _duration_sqlite(start, end):
    """Return the time in seconds since 1970 as floating point for of the
    specified argument (a datetime in sqlite format)
    """
    # note: sqlite time format is bizarre. They have %s: timestamp in SECONDS
    # since 1970, %f seconds only (with 3 decimal digits WTF?) and %S: seconds
    # part (integer). Thus to have a floating point value with 3 decimal digits
    # we should return:
    # ```
    # round(strftime('%s',{}) + strftime('%f',{}) - strftime('%S',{}), 3)".\
    #   format(dtime)
    # ```
    # However, for performance reasons we think it's sufficient to return the
    # seconds, thus we keep it more simple with the use round at the end to
    # coerce to float with 3 decimal digits, for safety (yes, round in sqlite
    # returns a float) and avoid integer divisions when needed but proper
    # floating point arithmentic
    return ("round(strftime('%s',{1})+strftime('%f',{1})-strftime('%S',{1}) - "
            "(strftime('%s',{0})+strftime('%f',{0})-strftime('%S',{0})), 3)").\
        format(start, end)


def _duration_postgres(start, end):
    """Return the time in seconds since 1970 as floating point for of the
    specified argument (a datetime in postgres format)
    """
    # Note: we use round at the end to coerce to float with 3 decimal digits,
    # for safety and avoid integer divisions when needed but proper floating
    # point arithmentic
    return "round(EXTRACT(EPOCH FROM ({1}-{0}))::numeric, 3)".format(start,
                                                                     end)


# function `duration_sec`

class duration_sec(FunctionElement):
    name = 'duration_sec'
    type = Float()


@compiles(duration_sec)
def standard_duration_sec(element, compiler, **kw):
    starttime, endtime = [compiler.process(c) for c in element.clauses]
    return _duration_postgres(starttime, endtime)


@compiles(duration_sec, 'sqlite')
def sqlite_duration_sec(element, compiler, **kw):
    starttime, endtime = [compiler.process(c) for c in element.clauses]
    return _duration_sqlite(starttime, endtime)


# function `missing_data_sec`

class missing_data_sec(FunctionElement):
    name = 'missing_data_sec'
    type = Float()


@compiles(missing_data_sec)
def standard_missing_data_sec(element, compiler, **kw):
    start, end, request_start, request_end = [compiler.process(c)
                                              for c in element.clauses]
    return "({1}) - ({0})".format(_duration_postgres(start, end),
                                  _duration_postgres(request_start, request_end))


@compiles(missing_data_sec, 'sqlite')
def sqlite_missing_data_sec(element, compiler, **kw):
    start, end, request_start, request_end = [compiler.process(c)
                                              for c in element.clauses]
    return "({1}) - ({0})".format(_duration_sqlite(start, end),
                                  _duration_sqlite(request_start, request_end))


# function `missing_data_ratio`

class missing_data_ratio(FunctionElement):
    name = 'missing_data_ratio'
    type = Float()


@compiles(missing_data_ratio)
def standard_missing_data_ratio(element, compiler, **kw):
    start, end, request_start, request_end = [compiler.process(c)
                                              for c in element.clauses]
    return "1.0 - (({0}) / ({1}))".format(_duration_postgres(start, end),
                                          _duration_postgres(request_start, request_end))


@compiles(missing_data_ratio, 'sqlite')
def sqlite_missing_data_ratio(element, compiler, **kw):
    start, end, request_start, request_end = [compiler.process(c)
                                              for c in element.clauses]
    return "1.0 - (({0}) / ({1}))".format(_duration_sqlite(start, end),
                                          _duration_sqlite(request_start, request_end))


# function `deg2km`

class deg2km(FunctionElement):
    name = 'deg2km'
    type = Float()


@compiles(deg2km)
def standard_deg2km(element, compiler, **kw):
    deg = compiler.process(list(element.clauses)[0])
    return "%s * (2.0 * 6371 * 3.14159265359 / 360.0)" % deg


# function `substr`

class substr(FunctionElement):
    name = 'substr'
    type = String()


@compiles(substr)
def standard_substr(element, compiler, **kw):
    clauses = list(element.clauses)
    column = compiler.process(clauses[0])
    start = compiler.process(clauses[1])
    leng = compiler.process(clauses[2])
    return "substr(%s, %s, %s)" % (column, start, leng)

