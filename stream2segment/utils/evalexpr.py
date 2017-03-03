import numpy as np
import math
import json
from datetime import datetime
import re
from __builtin__ import eval as builtin_eval

import sys
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
    """see doc for _isnum applied to booleans"""
    return type(val) == bool or (_isnumpy(val) and val.dtype.kind == 'b')


def _isstr(val):
    """see doc for _isnum applied to booleans"""
    return type(val) in _strtypes or (_isnumpy(val) and val.dtype.kind in 'SU')


# datetime regexp, to do a preliminary check and see if it's a datetime.
# In any case, we will parse it with numpy to be sure. The latter accepts leading and trailing
# spaces so we do it as well
_dtyme_re = re.compile(r'^\s*\d\d\d\d-\d\d-\d\d(?:[T ]\d\d:\d\d:\d\d(?:.\d{1,6})?)?\s*$')


def _isdtime(val):
    """see doc for _isnum applied to booleans. This evaluates to True also
    if val is a scalar string (e.g., not array of strings) and has the iso format of a datetime
    """
    ret = isinstance(val, datetime) or (_isnumpy(val) and val.dtype.kind == 'M')
    if not ret and type(val) in _strtypes and _dtyme_re.match(val):
        # ok, we have a format like \d\d\d\d-\d\d-\d\d... but are months correct??
        # delegate numpy:
        try:
            np.array(val, dtype='datetime64[us]')
            ret = True
        except ValueError:
            ret = False
    return ret


def _isnum(val):
    """return true if val is a numeric type. _isnum(5) == _isnum(5.6). Takes into
    account the case where val is a numpy array (but does not check if it's scalar or not,
    just that the dtype is numeric)"""
    # important: DO NOT SET AS _isnum types WHICH ARE NOT COMPARABLE with other isnum!
    # IT MUST hold:
    # _types_comparable(v1, v2) for any v1, v2 in any of the type
    # implemented here (int, float, ...)

    # no need to check isnan or isinf cause np.nan, np.inf, float('nan') and float('inf')
    # have all type == float
    # on the other hand we want to include numpy numeric types
    return type(val) in _numtypes or (_isnumpy(val) and val.dtype.kind in 'iuf')


def _types_comparable(val1, val2):
    # the first three are functions which take into account types compatibility for order
    # relations. The last two conditions checks the type equality IF val1 and val2 are not a numpy
    # array, because if they are (4th condition) a further check on dtype.kind must be done

    # note: comparison of dtimes must be done before _isstr
    return (_isnum(val1) and _isnum(val2)) \
         or (_isdtime(val1) and _isdtime(val2)) \
         or (_isstr(val1) and _isstr(val2)) \
         or (_isbool(val1) and _isbool(val2)) or (not _isnumpy(val1) and type(val1) == type(val2)) \
         or (_isnumpy(val1) and _isnumpy(val2) and val1.dtype.kind == val2.dtype.kind)


def _get_domain_bounds(bound):
    if _isnum(bound):
        return -float('inf'), +float('inf')
    elif _isdtime(bound):  # note: comparison of dtimes must be done before _isstr
        return None, None
    elif _isstr(bound):
        return '', None
    elif type(bound) == bool:
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


_eval_imports = {'np': np, 'math': math, 'inf': np.inf, 'nan': np.nan, 'e': np.e, 'pi': np.pi,
                 'sqrt': np.sqrt, "False": False, "false": False,
                 "FALSE": False, "True": True, "true": True,
                 "TRUE": True}


def eval(expr, vars=None):
    if _isdtime(expr):
        return expr
    if not vars:
        vars = {}
    vars.update(_eval_imports)
    # add some safety:
    # (https://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html)
    vars['__builtins__'] = {}

    return builtin_eval(expr, vars)


class interval(object):
    _OPERATORS1 = set(('<', '>', '='))
    _OPERATORS2 = set(('==', '>=', '<='))
    single_equal_is_operator = True
    # _inf = float('inf')
    _nan = float('nan')
    _funcarg_varname = "__x__"

    def __init__(self, l_isopen, l_bound, u_bound, u_isopen):
        """
            :param l_isopen: boolean indicating if the lower bound is 'open'. An interval
            with an open bound X does NOT include X, e.g.: the expression `x>6`, if seen as
            an interval from `6` to `inf`, has an open lower upper bound (6). On the other
            hand, `x>=6` has a closed upper bound.
            :param l_bound: the lower bound. None has the specific meaning of inferring it from
            the domain (string, numeric). If that min does not exist (e.g., minimum of datetimes -
            although there is currently not support for datetimes), the interval upper bound is to
            be considered "endless", and the argument 'u_isopen' is irrelevant.
            On the other hand, if that maximum exists (e.g. inf for numbers, '' for strings), note
            that in some (rare) edge cases `u_isopen` might be set equal to `l_isopen` to create
            an empty interval instead of raising an exception
            :param u_bound: the upper bound. None has the specific meaning of inferring it from
            the domain (string, numeric). If that max does not exist (e.g., maximum of strings),
            the interval upper bound is to be considered "endless", and the argument 'u_isopen'
            is irrelevant.
            On the other hand, if that maximum exists (e.g. inf for numbers), note that in some
            (rare) edge cases `u_isopen` might be set equal to `l_isopen` to create an empty
            interval instead of raising an exception
            :param u_isopen: boolean indicating if the upper bound is 'open'. An interval
            with an open bound X does NOT include X, e.g.: the expression `x>6`, if seen as
            an interval from `-inf` to 6, has an open upper bound (6). On the other
            hand, `x<=6` has a closed upper bound.
        """
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
        self._dtype = 'datetime64[us]' if any([is_l_dtime, is_u_dtime]) else None
        l_bound = np.array(l_bound, dtype='datetime64[us]').item() if is_l_dtime else l_bound
        u_bound = np.array(u_bound, dtype='datetime64[us]').item() if is_u_dtime else u_bound
        # set a flag cause if datetime we need to convert strings into datetime(s) in __call__
        # comparison via __eq__, __ne__ on the other hand is fine because bounds are stored as numpy
        # datetime(s)
        self.l_bound, self.u_bound = l_bound, u_bound
        self.l_isopen, self.u_isopen = l_isopen, u_isopen

        if l_bound is None:
            self.l_bound = _get_domain_bounds(u_bound)[0]
            # we just set the minimum as the domain default. If the domain default is not None
            # (e.g. '' for strings) and the upper interval
            # is the same (''), then we might be now in a situation like ['', ''[.
            # As the lower bound was None, it was asked to infer it, thus change the left interval
            # to get ]'', ''[  which is empty instead of rising exceptions in the check at the
            # end

            # so, if we changed the value of l_bound and now is equal to u_bound:
            if self.l_bound is not None and self.u_bound == self.l_bound:
                self.l_isopen = self.u_isopen
            self._refval = u_bound
        elif u_bound is None:
            self.u_bound = _get_domain_bounds(l_bound)[1]
            # we just set the maximum as the domain default. If the domain default is not None
            # (e.g. inf for numbers) and the lower interval
            # is the same (inf), then we might be now in a situation like [inf, inf[.
            # As the upper bound was None, it was asked to infer it, thus change the right interval
            # to get ]inf, inf[  which is empty instead of rising exceptions in the check at the
            # end

            # so, if we changed the value of u_bound and now is equal to l_bound:
            if self.u_bound is not None and self.u_bound == self.l_bound:
                self.u_isopen = self.l_isopen
            self._refval = l_bound
        else:
            if not _types_comparable(l_bound, u_bound):
                raise ValueError('bound types not compatible (e.g., str and int)')
            self._refval = l_bound  # maybe pick float if int, float?

        l_bound_defined = self.l_bound is not None
        u_bound_defined = self.u_bound is not None
        if l_bound_defined and u_bound_defined:
            if not self.l_bound < self.u_bound:
                if not (self.l_bound == self.u_bound and self.l_isopen == self.u_isopen):
                    raise ValueError('Malformed interval, check bounds')

    @property
    def empty(self):
        none_isnone = self.l_bound is not None and self.u_bound is not None
        return none_isnone and self.l_bound == self.u_bound and self.l_isopen and self.u_isopen

    @property
    def isscalar(self):
        # note: need to check what happens for [inf]. Is it scalar? If yes, this should not
        # be accounted for here if we specified None as u_bound for instance
        none_isnone = self.l_bound is not None and self.u_bound is not None
        return none_isnone and self.l_bound == self.u_bound and not self.l_isopen and not self.u_isopen

    @classmethod
    def _get_condition(cls, direction, bound, isopen, val):
        cond = None
        if bound is None:
            return cond  # skip any comparison on this side of the interval
        elif direction == -1:  # left
            cond = val > bound if isopen else val >= bound
        else:  # right
            cond = val < bound if isopen else val <= bound
        return cond

    def __call__(self, val):
        """
        Calls the interval in order to return a numpy boolean array of values, where each value
        returns if the corresponding element of val belongs to this interval. If val is a scalar,
        return a numpy array of 1 value.

        """
        # problems in numpy==1.11.3:
        #
        # Operation               Result and type(result)                         Problems
        # ======================  ======================================== ====================================
        # 5 <= 5                  True <type 'bool'>
        # np.array(5) <= 5        True <type 'numpy.bool_'>
        #
        # [4,5,6] <= 5            False <type 'bool'>                      should be same output as np (row below)
        # np.array([4,5,6]) <= 5  [True True False] <type 'numpy.ndarray'>
        #
        # "5" <= 5                False <type 'bool'>
        # np.array("5") <= 5      False <type 'bool'>                      WTF?!!! should return a np.bool_!!!!
        # np.array(5) <= 4        False <type 'numpy.bool_'>               WTF?!! type is now ok!
        #
        # WE CHOOSE TO RETURN ALWAYS np objects. We need to correct the case where the result is False
        # (i.e., python False)

        # Another issue when calling np.asarray: self._dtype is usually None, *except* for
        # datetimes. In the latter
        # case, np.asarray will coerce strings or datetimes into an appropriate 
        # (and hopefully, performance efifcient) numpy
        # datetime. BUT in this case we might have ValueError's, which we do not have for other
        # types where numpy infers it from the data.

        # Thus, we decide to set cond=False in case of dtype coerce errors, or interval empty,
        # which is the same as the case outlined in the table above where no errors are thrown
        # but numpy compares apples with oranges
        cond = False
        shape = None
        if not self.empty:
            try:
                np_val = np.asarray(val, dtype=self._dtype)  # does not copy if already numpy array
                # check if isscalar first (for speed): (e.g. [5], closed interval of just one
                # element)
                shape = np_val.shape
                if self.isscalar:
                    cond = np_val == self._refval
                else:
                    l_cond = self._get_condition(-1, self.l_bound, self.l_isopen, np_val)
                    u_cond = self._get_condition(1, self.u_bound, self.u_isopen, np_val)

                    if l_cond is None and u_cond is None:
                        return np.ones(shape=np_val.shape, dtype=bool)
                    elif l_cond is None:
                        cond = u_cond
                    elif u_cond is None:
                        cond = l_cond
                    else:
                        if l_cond is False or u_cond is False:  # see above commented text
                            cond = False
                        else:
                            cond = l_cond & u_cond
            except ValueError:
                pass

        # see above commented text
        if cond is False:
            if shape is None:  # we didn't convert val to numpy array, we need a shape.
                # Try to infer it without allocating a whole array:
                if hasattr(val, "__len__") and not isinstance(val, str):
                    shape = (len(val),)
                else:  # ok, try to get the shape by allocating a numpy array:
                    shape = np.asarray(val).shape
            cond = np.zeros(shape=shape, dtype=bool)
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
        """
        chunk = jsonlike_string
        opr, bound = (chunk[:2], chunk[2:]) if chunk[:2] in cls._OPERATORS2 else \
            (chunk[:1], chunk[1:]) if chunk[:1] in cls._OPERATORS1 else (None, None)
        if opr == '=' and not cls.single_equal_is_operator:
            opr, bound = None, None

        if opr is not None:
            bound = cls._evalchunk(bound)
            if opr == '==':
                return interval(l_isopen=False, l_bound=bound, u_bound=bound, u_isopen=False)
            elif opr == '<':
                return interval(l_isopen=False, l_bound=None, u_bound=bound, u_isopen=True)
            elif opr == '<=':
                return interval(l_isopen=False, l_bound=None, u_bound=bound, u_isopen=False)
            elif opr == '>':
                return interval(l_isopen=True, l_bound=bound, u_bound=None, u_isopen=False)
            else:  # opr == '<=':
                return interval(l_isopen=False, l_bound=bound, u_bound=None, u_isopen=False)
        elif chunk[0] in ('[', ']') and chunk[-1] in ('[', ']'):
            bounds = chunk[1:-1].split(',')
            assert len(bounds) == 2, "Invalid syntax (no comma): %s" % chunk
            l_b, u_b = cls._evalchunk(bounds[0]), cls._evalchunk(bounds[1])
            l_open, u_open = chunk[0] == ']', chunk[-1] == '['
            return interval(l_isopen=l_open, l_bound=l_b, u_bound=u_b, u_isopen=u_open)
        else:
            val = cls._evalchunk(chunk)
            return interval(l_isopen=False, l_bound=val, u_bound=val, u_isopen=False)

    @classmethod
    def _evalchunk(cls, chunk):
        try:
            return eval(chunk)
        except Exception:
            raise SyntaxError('Invalid syntax: "%s"' % chunk)

    def _cmp_(self, other):
        try:
            if isinstance(other, interval) and \
                _types_comparable(self._refval, other._refval) and \
                    not self.empty and not other.empty:
                my_min, my_max = self.l_bound, self.u_bound
                its_min, its_max = other.l_bound, other.u_bound
                if my_min == its_min and my_max == its_max:
                    if (self.l_isopen == other.l_isopen) and\
                            (self.u_isopen == other.u_isopen):
                        return 0
                elif my_min >= its_max and not any(x is None for x in [my_min, its_max]):
                    if my_min == its_max:
                        if not self.l_isopen and not other.u_isopen:
                            return 1
                    return 2
                elif my_max <= its_min and not any(x is None for x in [my_max, its_min]):
                    if my_max == its_min:
                        if not self.u_isopen and not other.l_isopen:
                            return -1
                    return -2
        except:
            pass
        return NotImplemented

    def _cmp_result(self, other):
        # used below. Just return something that evaluates to False if NotImplemented
        # (the latter is not comparable, e.g. NotImplemented > 1 is True!
        res = self._cmp_(other)
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
        min_, max_ = _get_domain_bounds(self._refval)
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
        equal = self.l_bound == self.u_bound  # remember: they cannot be both None
        if equal:
            if not self.l_isopen and not self.u_isopen:
                ret = "==%s" % jsondumps(self.l_bound)
        elif self.l_bound is None or (self.l_bound == min_ and not equal and not self.l_isopen):
            ret = "%s%s" % ("<" if self.u_isopen else "<=", jsondumps(self.u_bound))
        elif self.u_bound is None or (self.u_bound == max_ and not equal and not self.u_isopen):
            ret = "%s%s" % (">" if self.l_isopen else ">=", jsondumps(self.l_bound))

        if ret is None:
            br1 = "]" if self.l_isopen else '['
            br2 = "[" if self.u_isopen else ']'
            ret = "%s%s, %s%s" % (br1, jsondumps(self.l_bound), jsondumps(self.u_bound), br2)

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
        match_val = eval(condition_expr[2:])
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
