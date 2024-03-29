"""
Decorators for the making function displayable on the
Graphical User Interface (GUI)

:date: Sep 19, 2017

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""


def preprocess(func):
    """Decorator to add the attribute `func._s2s_att = "gui.preprocess"`"""
    func._s2s_att = "gui.preprocess"  # pylint: disable=protected-access
    return func


def customplot(func):  # DEPRECATED: backward compatibility
    """Decorator to add the attribute `func._s2s_att = "gui.customplot"`"""
    func._s2s_att = "gui.customplot"  # pylint: disable=protected-access
    return plot('b')(func)


def sideplot(func):  # DEPRECATED: backward compatibility
    """Decorator to add the attribute `func._s2s_att = "gui.sideplot"`"""
    return plot('r', xaxis={'type': 'log'}, yaxis={'type': 'log'})(func)


def plot(*args, **kwargs):
    """Decorator to add the attribute `func._s2s_att` = "gui.plot" and the given
    properties

    :param kwargs: `position` ('b' for bottom, the default, or 'r' for right),
        `xaxis`, `yaxis` (both dict of plotly axis properties, default: None, i.e.
        empty dict. For info on axis, see: https://plot.ly/python/axes/)
    """
    position = kwargs.get('position', 'b')
    xaxis = kwargs.get('xaxis', None)
    yaxis = kwargs.get('yaxis', None)

    # Here we want to allow @decorator, @decorator() and @decorator(position='b',...)
    # Solution hint here: https://stackoverflow.com/q/3931627

    # First define decorator wrapper:
    def decorator(func):
        """Set the attributes on the function in order to make it recognizable as
        gui func
        """
        func._s2s_att = 'gui.plot'  # pylint: disable=protected-access
        func._s2s_position = position  # pylint: disable=protected-access
        func._s2s_xaxis = xaxis or {}  # pylint: disable=protected-access
        func._s2s_yaxis = yaxis or {}  # pylint: disable=protected-access
        return func

    if len(args) == 1 and hasattr(args[0], '__call__') and not kwargs:
        # we called @gui.plot (with no arguments nor brackets)
        return decorator(args[0])

    # now we pay back: we have to parse args, as we might have called the
    # decorator with positional arguments...
    if len(args) > 3:
        raise SyntaxError('@gui.plot: 0 to 3 positional arguments expected, '
                          '%d received' % len(args))

    if len(args) >= 1:
        position = args[0]
    if len(args) >= 2:
        xaxis = args[1]
    if len(args) == 3:
        yaxis = args[2]

    return decorator


def get_func_attrs(func):
    """Return the function attributes for a function decorated with the decorators of
    this class: `attname, position, xaxis, yaxis`. Check for attname first: if empty
    string, the function is not a gui decorated function
    """
    return getattr(func, '_s2s_att', ''), \
        getattr(func, '_s2s_position', 'b'), \
        getattr(func, '_s2s_xaxis', {}), \
        getattr(func, '_s2s_yaxis', {})
