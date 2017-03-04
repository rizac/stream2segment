import sys
import numpy as np
import math
import json
from datetime import datetime
from datetime import MAXYEAR, MINYEAR
import re
from __builtin__ import eval as builtin_eval
from itertools import izip

if sys.version_info[0] == 2:
    _strtypes = (bytes, str, unicode)
    _numtypes = (int, float, long)
else:
    _strtypes = (str)
    _numtypes = (int, float)


def _isnumpy(value):
    # first option (from internet):
    # assert type(value).__module__ == 'numpy'

    # but we use this one:
    return hasattr(value, 'dtype') and hasattr(value.dtype, 'kind')


def _isbool(val):
    """return true if val is a boolean type.
    Works in python 2 and 3 and also with if `val` is a `numpy` array (does not check if `val`
    shape or length, just `val`s dtype)
    """
    return type(val) == bool or (_isnumpy(val) and val.dtype.kind == 'b')


def _isstr(val):
    """return true if val is a datetime type.
    Works in python 2 and 3. For instance, in python 2: _isstr('abc') == _isstr(u'abc').
    Works and also with if `val` is a `numpy` array (does not check if `val` shape or length,
    just `val`s dtype)
    """
    return type(val) in _strtypes or (_isnumpy(val) and val.dtype.kind in 'SU')


def _isdtime(val):
    """return true if val is a datetime type. Works in python 2 and 3,
    and also with if `val` is a `numpy` array (does not check if `val` shape or length,
    just `val`s dtype)
    """
    return isinstance(val, datetime) or (_isnumpy(val) and val.dtype.kind == 'M')


def _isnum(val):
    """return true if val is a numeric type: _isnum(5) == _isnum(5.6). Works in python 2 and 3,
    and also with if `val` is a `numpy` array (does not check if `val` shape or length,
    just `val`s dtype)"""
    # important: DO NOT SET AS _isnum types WHICH ARE NOT COMPARABLE with other isnum!
    # IT MUST hold:
    # _types_comparable(v1, v2) for any v1, v2 in any of the type
    # implemented here (int, float, ...)

    # no need to check isnan or isinf cause np.nan, np.inf, float('nan') and float('inf')
    # have all type == float
    # on the other hand we want to include numpy numeric types
    return type(val) in _numtypes or (_isnumpy(val) and val.dtype.kind in 'iuf')


def _types_comparable(val1, val2):
    """Returns true if `val1` and `val2` are meningful comparable. The algorithm checks
    (stopping at the first true result):
    - if they are both numeric, i.e. both in (int, float, long if python2, numpy array with one of
      the numeric dtypes)
    - if they are both strings, i.e. both in (str, unicode if python2, numpy array with one of the
      strings dtypes)
    - if they are both datetimes, i.e. both in (datetime, numpy array with the datetime dtype)
    - if they are both boolean, i.e. both in (bool, numpy array with the bool dtype)
    - if they are both numpy arrays with the same dtype
    - if type(val1) == type(val2)
    If any of the previous conditions matches, returns False
    """
    # the first three are functions which take into account types compatibility for order
    # relations. The last two conditions checks the type equality IF val1 and val2 are not a numpy
    # array, because if they are (4th condition) a further check on dtype.kind must be done

    return (_isnum(val1) and _isnum(val2)) \
        or (_isdtime(val1) and _isdtime(val2)) \
        or (_isstr(val1) and _isstr(val2)) \
        or (_isbool(val1) and _isbool(val2)) or (not _isnumpy(val1) and type(val1) == type(val2)) \
        or (_isnumpy(val1) and _isnumpy(val2) and val1.dtype.kind == val2.dtype.kind)


def _get_domain_bounds(obj):
    """Returns the bounds (ranges) of minima nd maxima values the type associated to
    `obj` can have. The type of obj is given by the `_is*` functions (e.g. `_isnum`, `_isstr`, ...)
    thus is in a wider sense than the python type. If the object has no bounds known, return the
    tuple None, None
    """
    if _isnum(obj):
        return -float('inf'), +float('inf')
    elif _isdtime(obj):  # note: comparison of dtimes must be done before _isstr
        return datetime(MINYEAR, 1, 1), datetime(MAXYEAR, 12, 31, 23, 59, 59, 999999)
    elif _isstr(obj):
        return '', None
    elif type(obj) == bool:
        return False, True
    else:
        return None, None


class Piecewisefunc():

    def __init__(self, **kwargs):
        self._pieces = [(interval(k), v) for k, v in kwargs.iteritems()]
        self._pieces.sort(key=lambda x: x[0])
        prev_piece = None
        for piece in self._pieces:
            if prev_piece is not None:
                assert piece >= prev_piece, ("Intervals of the pieceweise function intersect: "
                                             "%s and %s" % (prev_piece, piece))
            prev_piece = piece

    def __call__(self, value, funcarg_name='x'):
        np_value = np.asarray(value, dtype=float)
        assert len(np_value.shape) < 1, ('Pieceweise functions can calculate result for scalar or '
                                         'one-dimensional arrays only')
        ret = np.zeros(shape=np_value.shape, dtype=float)
        ret.fill(np.NaN)
        for interval, func_str in self._pieces:
            condition = interval(np_value)
            ret[condition] = eval(func_str, {funcarg_name: np_value})
        return ret


_eval_imports = {'np': np, 'inf': float('inf'), 'nan': float('nan'), 'e': math.e,  # for safety
                 'pi': math.pi,  # for safety
                 # by disabling builtins, boolean are not recognized (but str are.Why?)
                 # so set boolean values (being javascript compatible and adding ucase compatiblity)
                 "False": False, "false": False,
                 "FALSE": False, "True": True, "true": True,
                 "TRUE": True}

# import math functions, so we can refer to them without "math." in string expressions
for name in math.__dict__:
    elm = math.__dict__[name]
    if hasattr(elm, '__call__') or _isnum(elm):
        _eval_imports[name] = elm


def _eval(expr, vars=None):  # @ReservedAssignment
    """eval is evil, eval is evil. But we are not paranoid: we studied the risks, this software
    will not be online, this function is already provided with some security (see below), so if
    somebody wants to kill his/her computer via this function, wow please let us know we will be
    honored.
    By the way, this function calls python `eval` on `expr`. The globals variables can be set in
    `vars`, but for security reasons `__builtins__`, if provided as key in `vars`, will be
    overridden and cannot be set.

    This function has access to all functions and constant defined in the `math` package (by typing
    as they are, so `pow`, not `math.pow`) and is intended - although not mandatory - to produce a
    scalar. In any case, it is not optimized for speed or to produce complex numpy operations
    on arrays (`numpy` is used in the `interval` object for performances, and therein is
    recommended to use them). However, `numpy` can be accessed via `numpy` or `np` in `expr`,
    if needed.
    """
    # safety1: any dot must be preceeded by numpy, np, a space or a number, and must be
    # followed by a space or a number
#     if re.match(r'(?<!numpy)\.|(?<!np)\.|(?<!\s)\.|(?<!\d)\.|\.(?![\d\s])', expr):
#         raise ValueError("Invalid string expression: '%s'" % expr)
    if not vars:
        vars = {}  # @ReservedAssignment
    vars.update(_eval_imports)
    # add some safety:
    # (https://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html)
    vars['__builtins__'] = {}

    return builtin_eval(expr, vars)


class interval(object):
    """Class extending a mathematical interval, i.e an set of real numbers with the property
    that any number that lies between two numbers in the set is also included in the set.
    This object can represent an interval of any python type which has
    an order relation set between elements (e.g. bool, datetime's, strings)

    The interval of numbers, strings, datetime's or booleans between a and b, including a and b,
    is often denoted [a, b]. If the interval does not include a and b, is indicated as
    (a, b) or ]a, b[. Thus (a, b] or ]a, b] denote an interval including b and not a,
    and so on. The points a and b are called left end-point and right end-point,
    respectively.

    This class supports the `in` operator, is callable, can parse python-like syntax to produce
    an interval (for instance, if the string is given from a form request or a config file),
    and supports order relations with other intervals

    As callable
    ===========

    When called with an argument 'arg', this object will return a numpy array of
    booleans indicating whether each of the points of 'arg' belong to it.
    If 'arg' is not iterable, then returns a single element boolean array.

    The `in` operator basically calls this object and returns a boolean (specifically, the
    `array.all()` on the returned `numpy` boolean array).

    As parser
    =========

    Any python source code string can be parsed to produce an interval (e.g. "<=5.35 * pi").
    In addition, syntax like "['a', 'b')" is also valid, and `datetime`s are recognized if input
    as iso-format (e.g. "2006-01-01", ""2006-01-01T13:21:45" or "2006-01-01T12:45.45.5000"). Note:
    `datetime`s should not be quoted otherwise they will be recognized as strings. Any python
    expression will be evaluated (e.g. "['a'+'b', 'c']) and the user can supply all `math` package
    functions and constants (without typing `math.` before it). Datetimes, as they are not
    standard python, do not supports expressions. The expression in a string interval is not
    intended to be optimized for performances. However, numpy is available in expressions if
    strings are pre-pended with 'np.'

    Relation orders
    ===============

    Intervals support also order relations. Note only that, given two intervals `I1` and `I2`,
    `I1 <= I2` does not mean that all points of `I1` are lower or equal than all points of `I2`.
    `I1 <= I2` means that all points of `I1` are lower than all points of `I2` *except* one
    point which belongs to both intervals I1 right end-point and I2 left end-point.

    Thus:
    [1,4] < [5,6]
    [1,4] != [5,6]
    [1,4] <= [4,6]
    [1,4] < (4,6] (left-open, it does not include the point)
    [1,4] == [1,4]
    [1,4] != [2,5]

    Equality between two intervals I1, I2 means that for any object holds:
    `obj in I1 == obj inI2`. Note
    a suble difference in open/closed intervals due to domain bounds. This is True:
    `interval('[', 'a', None, ']') == interval('[', 'a', None, ')')`
    because any possible string that belongs to the first interval belongs to the second, and
    vice-versa. On the other hand
    `interval('[', 1, None, ']') != interval('[', 1, None, ')')`
    because numbers have -inf and inf as bounds and thus float('inf') belongs to the first
    interval, but not to the second
    """

    _OPERATORS1 = set(('<', '>', '='))
    _OPERATORS2 = set(('==', '>=', '<='))
    _L_BRACKETS = set(['(', ']', '['])
    _R_BRACKETS = set([')', ']', '['])
    _L_OPENBRACKETS = set(['(', ']'])
    _R_OPENBRACKETS = set([')', '['])
    single_equal_is_operator = True
    _nan = float('nan')
    _funcarg_varname = "__x__"

    def __init__(self, *args):
        """Creates a new interval. The __init__ function can be called with these arguments:

        * interval(list)
        * interval(numpy_array)
        * interval(tuple)
        * interval(l_bracket, l_pt, r_pt, r_bracket)

        If the argument is a single list/tuple/array, then it must have
        2 elements specifying the interval end-points (left, right).
        Reflecting the mathematical notation:
        "[a,b]" for closed intervals (i.e., which include their end-points),
        "(a, b)" or "]a, b[" for open intervals (i.e., which do not include their end-points),
        Then a list (or numpy array) will create a closed interval, a tuple an open one.

        An interval that is nor closed neither open, i.e. is left-closed or right-closed only,
        can be instantiated via the last constructor, where

        * `l_bracket` is a string in '[', '(' or ']'
        * `l_pt` and `r_pt` are the end-points of the interval (left, right), and
        * `r_bracket` is a string in ']', ')' or '['

        The end-points of the interval can be None. In that case they will be replaced with the
        domain minimum and maximum, i.e. the python constant value which is lower / greater than
        all other python values of the same type (if they exist): the interval
        can be still closed or open, but if such a python value does not exist,
        being closed or open is obviously meaningless and ignored in calculations or interval
        comparisons.
        For instance, for strings there is no python constant value denoting the maximum and the
        following are the same:
        ```
        interval('[', 'a', None, "]")
        interval('[', 'a', None, ")")
        ```
        because any string belonging to the first interval belongs to the second, and vice-versa.
        On the other hand, given that numbers have defined bounds `-float('inf')` and
        `float('inf')`, the following are different intervals:
        ```
        interval('[', 1, None, "]")
        interval('[', 1, None, ")")
        ```
        as `float('inf')` belongs to the first interval only.
        Finally, the end-points cannot be both None as otherwise
        this object cannot infer the type of its elements, which is necessary for its
        functionality
        """

        if args[0] in self._L_BRACKETS:
            l_isopen = args[0] in self._L_OPENBRACKETS
            l_bound = args[1]
            u_bound = args[2]
            u_isopen = args[3] in self._R_OPENBRACKETS
            assert args[0] in self._L_BRACKETS and args[3] in self._R_BRACKETS
        else:
            argz = args[0]
            if _isnumpy(argz):
                argz = argz.tolist()
            assert type(argz) in (tuple, list) and len(args[0]) == 2, \
                'Specify a 2- element tuple / list or numpy array as interval single argument'

            l_bound, u_bound = argz[0], argz[1]
            _open = type(argz) == tuple
            l_isopen, u_isopen = _open, _open

        if hasattr(l_bound, "__iter__") and not isinstance(l_bound, str):
            raise SyntaxError("Lower bound interval iterable not string: invalid value")

        if hasattr(u_bound, "__iter__") and not isinstance(u_bound, str):
            raise SyntaxError("Upper bound interval iterable not string: invalid value")

        if l_bound is None and u_bound is None:
            raise ValueError('lower and upper bounds both None')

        # coerce to datetime if any is can be coerced. If only one of the two is datetime
        # and the other not, we will check the error later
        is_l_dtime = l_bound is not None and _isdtime(l_bound)
        is_u_dtime = u_bound is not None and _isdtime(u_bound)
        # set a flag cause if datetime we need to convert strings into datetime(s) in __call__
        # comparison via __eq__, __ne__ on the other hand is fine because bounds are stored as
        # python datetime(s)
        self._dtype = 'datetime64[us]' if any([is_l_dtime, is_u_dtime]) else None

        global_min, global_max = _get_domain_bounds(u_bound if l_bound is None else l_bound)
        if l_bound is None:
            self._refval = u_bound
        elif u_bound is None:
            self._refval = l_bound
        else:
            self._refval = l_bound  # maybe pick float if int, float?
            if not _types_comparable(l_bound, u_bound):
                raise ValueError('bound types not compatible: %s and %s' %
                                 (str(type(l_bound)), str(type(u_bound))))

        l_bound_defined = l_bound is not None
        u_bound_defined = u_bound is not None
        if l_bound_defined and u_bound_defined:
            if not l_bound < u_bound:
                if not (l_bound == u_bound and l_isopen == u_isopen):
                    raise ValueError('Malformed interval, check bounds')

        # if both bounds are not None, check if there is one which equals the type min/max
        # and that is closed. Then set it to None to speed up calculations
        if l_bound_defined and l_bound == global_min and not l_isopen and l_bound != u_bound:
            l_bound = None
        if u_bound_defined and u_bound == global_max and not u_isopen and l_bound != u_bound:
            u_bound = None

        self._endpoints = l_bound, u_bound
        self._lopen, self._ropen = l_isopen, u_isopen
        self._globalbounds = global_min, global_max

    @property
    def leftopen(self):
        return self._lopen

    @property
    def rightopen(self):
        return self._ropen

    @property
    def leftclosed(self):
        return not self.leftopen

    @property
    def rightclosed(self):
        return not self.rightopen

    @property
    def closed(self):
        return self.leftclosed and self.rightclosed

    @property
    def open(self):
        return self.leftopen and self.rightopen

    @property
    def endpoints(self):
        """Returns the end points of this interval, possibly converting None's passed in the
        constructor with relative boundary values
        :return : a list of two elements indicating the interval end points. They might be None
        to indicate unbound interval
        """
        return [self._globalbounds[0] if self._endpoints[0] is None else self._endpoints[0],
                self._globalbounds[1] if self._endpoints[1] is None else self._endpoints[1]]

    @property
    def unbounded(self):
        return self.leftunbounded and self.rightunbounded

    @property
    def leftunbounded(self):
        return self._endpoints[0] is None and (self.leftclosed or self._globalbounds[0] is None)

    @property
    def rightunbounded(self):
        return self._endpoints[1] is None and (self.rightclosed or self._globalbounds[1] is None)

    @property
    def empty(self):
        if self.closed:
            return False
        epts = self.endpoints
        return epts[0] == epts[1] and epts[0] is not None

    @property
    def degenerate(self):
        """Returns if this interval is degenerate, i.e. consisting of a single element"""
        # note: for numbers, [inf, inf] IS degenerate (inf is a point as it's a float)
        if not self.closed:
            return False
        epts = self.endpoints
        return epts[0] == epts[1] and epts[0] is not None

    def __call__(self, val):
        """Calls the interval in order to return a numpy boolean array of values, where each value
        returns if the corresponding element of val belongs to this interval. If val is a scalar,
        return a numpy array of 1 value.

        """
        # problems in numpy==1.11.3:
        #
        # Operation               Result and type(result)                         Problems
        # ======================  ======================================== ======================
        # 5 <= 5                  True <type 'bool'>
        # np.array(5) <= 5        True <type 'numpy.bool_'>
        #
        # [4,5,6] <= 5            False <type 'bool'>
        # np.array([4,5,6]) <= 5  [True True False] <type 'numpy.ndarray'>
        #
        # "5" <= 5                False <type 'bool'>
        # np.array("5") <= 5      False <type 'bool'>                      WTF?!!! no a np.bool_?
        # np.array(5) <= 4        False <type 'numpy.bool_'>               WTF?!! type is now ok!
        #
        # WE CHOOSE TO RETURN ALWAYS np objects. The latter case issues from dtypes uncomparable
        # so we will use the module function _types_comparable

        cond = False
        shape = None
        np_val = np.asarray(val)
        try:
            if self.empty:
                raise TypeError()

            # if datetime and val is datetime (or a list of datetime's),
            # numpy has created an array of objects. This lets convert them:
            if self._dtype == 'datetime64[us]' and np_val.dtype.kind == 'O':
                np_val = np_val.astype(self._dtype)

            # this should avoid tht cond is False (see above)
            if not _types_comparable(np_val, self._refval):
                raise TypeError()

            if self.degenerate:  # only a single pt:
                cond = np_val == self._refval
            else:
                l_bound, u_bound = self.endpoints
                l_cond = None if self.leftunbounded else \
                    np_val > l_bound if self.leftopen else np_val >= l_bound
                u_cond = None if self.rightunbounded else \
                    np_val < u_bound if self.rightopen else np_val <= u_bound

                if l_cond is None and u_cond is None:
                    return np.ones(shape=np_val.shape, dtype=bool)
                elif l_cond is None:
                    cond = u_cond
                elif u_cond is None:
                    cond = l_cond
                else:
                    cond = l_cond & u_cond
        except TypeError:
            return np.zeros(shape=shape, dtype=bool)

        return cond

    def __contains__(self, value):
        return self(value).all()

    @classmethod
    def parse(cls, jsonlike_string):
        """
            Parses the given string to produce an interval.
            :param jsonlike_string: A string denoting an interval
            as a mathematical expression either with open/closed brackets ("['a', 'b'+'c'[")
            or relational operators ("<= e*2". Note that "==" - e.g. "==45.6" - is valid and
            indicates an interval of a single point). It is "json-like" with the
            following additions / exceptions:

            - Brackets symbols indicate interval bounds. Thus "['a', 'd']" indicates the interval
              of all strings greater or equal than 'a', and lower or equal than 'd'.
            - Brackets can be set as "open", as commonly used in mathematics.
              Thus "]-12.5, 5]" is the interval of all numbers greater than -12.5, and lower or
              equal than 5
            - The string can start with any relational operator like '<', '>=', '==' etcetera,
              followed by a valid expression
            - Interval bounds **can be any kind of expression that will be evaluated**.
              This module uses the `eval` function python
              function (for safety using `eval`, see module doc) and thus recognizes any valid
              python syntax. The expression recognizes also by default 'inf', 'nan', 'e', booleans,
              plus 'numpy' and 'math' modules. Thus "<inf+np.e*4/2" or "]-inf, e/2]" are valid
              (numpy can be references via 'np', too)
            - For inputting datetimes, type them as iso format WITHOUT QUOTES (2016-01-01, or
              2016-02-15T01:45:23.450), otherwise they
              will be interpreted as strings. Unlike numbers, strings and boolean, `datetimes`
              thus cannot contain cannot thus be evaluated as expressions. `datetime.datetime`
              could be imported, but would make the syntax hard for non-python users, and this
              function is intended to create a python object from e.g. json request strings like
              "<=2014-0313T01:02:59"
        """
        chunk = jsonlike_string
        opr, bound = (chunk[:2], chunk[2:]) if chunk[:2] in cls._OPERATORS2 else \
            (chunk[:1], chunk[1:]) if chunk[:1] in cls._OPERATORS1 else (None, None)
        if opr == '=' and not cls.single_equal_is_operator:
            opr, bound = None, None

        if opr is not None:
            bound = cls._evalchunk(bound)
            if opr == '==':
                return interval('[', bound, bound, ']')
            elif opr == '<':
                return interval('[', None, bound, ')')
            elif opr == '<=':
                return interval('[', None, bound, ']')
            elif opr == '>':
                return interval('(', bound, None, ']')
            else:  # opr == '>=':
                return interval('[', bound, None, ']')

        elif chunk[0] in cls._L_BRACKETS and chunk[-1] in cls._R_BRACKETS:
            bounds = chunk[1:-1].split(',')
            assert len(bounds) == 2, "Invalid syntax (no comma): %s" % chunk
            l_b, r_b = cls._evalchunk(bounds[0]), cls._evalchunk(bounds[1])
            return interval(chunk[0], l_b, r_b, chunk[-1])

        else:
            val = cls._evalchunk(chunk)
            return interval('[', val, val, ']')

    # datetime regexp, used to eval otherwise unrecognized strings to datetime's
    # See interval.eval_chunk
    _dtyme_re = re.compile(r'''(?<![\w\d_'"])\s*(\d\d\d\d-\d\d-\d\d(?:[T ]\d\d:\d\d:\d\d(?:.\d{0,6})?)?)\s*(?![\w\d_'"])''')

    @classmethod
    def _evalchunk(cls, chunk):
        """evaluates a chunk of json-like string into a python value. The chunk must be
        the result of an interval parsable string"""
        chunk = cls._dtyme_re.sub(r"""np.array("\1", dtype='datetime64[us]').item()""", chunk)
        try:
            return _eval(chunk, {'np': np})  # np is included in _eval, but for safety ...
        except Exception:
            raise SyntaxError('Invalid syntax: "%s"' % chunk)

    def __cmp__(self, other):
        """Compares this interval with another one. Returns 2 if this interval is greater
        than the other (=all points of interval are greater), 1 if greater or equal (=all points
        of this interval are greater, ONE is shared), zero if intervals are equal (each element
        of this interval belongs to the other, and viceversa), -1 and -2 accordingly (other
        greater or equal, other greater.
        NOTE: An interval is greater or equal than another if it is "contiguous": i.e., the
        biggest point of the other interval is the smallest point of the interval.
        Thus, i0 >= i1 DOES NOT MEAN: all points of i0 >= i1, but:
        max(i1) == min(i0), all other points of i0 are > than all other points of i1
        """
        try:
            if isinstance(other, interval) and \
                _types_comparable(self._refval, other._refval) and \
                    not self.empty and not other.empty:
                my_min, my_max = self.endpoints
                its_min, its_max = other.endpoints
                if my_min == its_min and my_max == its_max:
                    if (my_min is None or self.leftopen == other.leftopen) and\
                            (my_max is None or self.rightopen == other.rightopen):
                        return 0
                elif my_min >= its_max and not any(x is None for x in [my_min, its_max]):
                    if my_min == its_max:
                        if self.leftclosed and other.rightclosed:
                            return 1
                    return 2
                elif my_max <= its_min and not any(x is None for x in [my_max, its_min]):
                    if my_max == its_min:
                        if self.rightclosed and other.leftclosed:
                            return -1
                    return -2
        except:
            pass
        return NotImplemented

    def _cmp_result(self, other):
        # used below. Just return something that evaluates to False if NotImplemented
        # (the latter is not comparable, e.g. NotImplemented > 1 is True!
        res = self.__cmp__(other)
        return self._nan if res is NotImplemented else res

    def __lt__(self, other):
        return self._cmp_result(other) < -1

    def __le__(self, other):
        return self._cmp_result(other) == -1

    def __eq__(self, other):
        return self._cmp_result(other) == 0

    def __ne__(self, other):
        return self._cmp_result(other) != 0

    def __gt__(self, other):
        return self._cmp_result(other) > 1

    def __ge__(self, other):
        return self._cmp_result(other) == 1

    def __str__(self):
        """The string representation of the interval"""
        # try to get a nicer representation
        # remember: if self.l_bound is None it means we passed None in the constructor
        # and the domain (str, numeric) has no given min. So we can avoid to display open-closed
        # interval. Same for self.u_bound

        def jsondumps(val):
            try:
                return json.dumps(val)
            except TypeError:
                try:
                    return val.isoformat()
                except AttributeError:
                    return str(val)
        ret = None
        endpts = self.endpoints  # remember: they cannot be both None
        if self.degenerate:
            ret = "==%s" % jsondumps(endpts[0])
        elif endpts[0] is None:
            ret = "%s%s" % ("<" if self.rightopen else "<=", jsondumps(endpts[1]))
        elif endpts[1] is None:
            ret = "%s%s" % (">" if self.leftopen else ">=", jsondumps(endpts[0]))

        if ret is None:
            br1 = "]" if self.leftopen else '['
            br2 = "[" if self.rightopen else ']'
            ret = "%s%s, %s%s" % (br1, jsondumps(endpts[0]), jsondumps(endpts[1]), br2)

        if self.empty:
            ret += " (empty set)"
        return ret


def match(condition_expr, values, on_type_mismatch='raise'):
    """Returns a **numpy array** of booleans indicating element-wise if each element of values
    matches the `condition_expr`. The argument `values` can be also a scalar (e.g. python value)

    :param condition_expr: a conditional expression that will be evaluated. E.g. "[-inf, 45.6*e[",
    ">='abc", "[" ..
    :param values: a single python value or an iterables of values. The returned numpy array
    of booleans will have size 1 in the former case, or size equal to len(values) in the latter
    :param on_type_mismatch: string (default 'raise'): If 'ignore', then when comparing different
    types, a numpy array of False's is returned. If 'raise' or any other value, a TypeError is
    raised. This happens for instance in this call: `match("<4", ['a', 45, 6])` due to that 'a'
    in values (string)

    :return: a **numpy array** of booleans indicating element-wise which element matches
    `condition_expr`
    """
    a = np.asanyarray(values)
    if condition_expr is None:
        return np.zeros(shape=a.shape, dtype=bool)
    elif condition_expr.startswith("!="):
        match_val = _eval(condition_expr[2:])
        cmd = np.ones(shape=a.shape, dtype=type(match_val))
        if not _types_comparable(a, cmd):
            if on_type_mismatch == 'ignore':
                return np.zeros(shape=a.shape, dtype=bool)
            else:
                raise TypeError("Uncomparable types: %s and %s" % (str(cmd.dtype), str(a.dtype)))
        cmd.fill(match_val)
        return a != cmd
    else:
        return interval.parse(condition_expr)(values)


def where(condition_expr, values, on_type_mismatch='raise'):  # @ReservedAssignment
    """Filters `values` removing those elements not matching
    `condition_expr`. This method returns a `numpy array`:
    if the argument `values` is a scalar (e.g. python value), then
    a numpy array of 1 element or empty (if the element did not match) is returned

    :param condition_expr: a conditional expression that will be evaluated. E.g. "[-inf, 45.6*e[",
    ">='abc", "[" ..
    :param values: a single python value or an iterables of values. The returned numpy array
    of booleans will have size 1 in the former case, or size equal to len(values) in the latter
    :param on_type_mismatch: string (default 'raise'): If 'ignore', then when comparing different
    types, a numpy array of False's is returned. If 'raise' or any other value, a TypeError is
    raised. This happens for instance in this call: `match("<4", ['a', 45, 6])` due to that 'a'
    in values (string)

    :return: a **numpy array** with the element of `values` which matched `condition_expr`. If
    `values` is a python scalar (e.g., str, float, int) an array of at most 1 element is returned,
    or an empty array (with `size=0`) if `values` did not match `condition_expr`
    """
    vals = np.asarray(values)
    ret = vals[match(condition_expr, values, on_type_mismatch)]
    return ret  # if vals is values else ret.tolist()
