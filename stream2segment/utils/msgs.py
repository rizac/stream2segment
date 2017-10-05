'''
Module handling messages to be displayed throughout the program by normalizing messages
format

:date: Nov 30, 2016

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''


def MSG(action=None, errmsg=None, url=None):
    """Function which formats a message in order to have normalized message
    types across the program (e.g., in logging utilities). The argument can contain new
    (e.g., "{}") but also old-style format keywords (such as '%s', '%d') for usage within the
    logging functions, e.g.: `logging.warning(MSG('%d segments discarded', 'no response'), 3)`.
    The resulting string message will be in any of the following formats (according to
    how many arguments are non-empty):
    ```
        "{action} ({errmsg}). url: {url}"
        "{action} ({errmsg})"
        "{action}"
        "{errmsg}. url: {url}"
        "{errmsg}"
        "{url}"
        ""
    ```
    :param action: string or None: what has been done (e.g. "discarded 3 events")
    :param errmsg: string or Exception: the Exception or error message which caused the action
    :param url: the url (string) or `urllib2.Request` object: the url originating the message, if
    the latter was issued from a web request
    """
    msg = action.strip()
    if errmsg:
        # sometimes exceptions have no message, append their name
        # (e.g. socket.timeout would now print at least 'timeout')
        strerr = (str(errmsg) or str(errmsg.__class__.__name__)).strip()
        msg = "{} ({})".format(msg, strerr) if msg else strerr
    if url:
        urlmsg = url2str(url).strip()
        msg = "{}. url: {}".format(msg, urlmsg) if msg else urlmsg
    return msg


def url2str(obj):
    """converts an url or `urllib2.Request` object to string. In the latter case, the format is:
    "{obj.get_full_url()}" if `obj.data` is falsy
    "{obj.get_full_url()}, data: '{obj.get_data()}'" if `obj.data` has no newlines, or
    "{obj.get_full_url()}, data: '{obj.get_data()[:I]}'" otherwise (I=obj.get_data().find('\n')`)
    """
    try:
        url = obj.get_full_url()
        data = obj.data
        if data is not None:
            N = 200
            str_data = ("%s\n...(showing first %d characters only)" % (data[:N], N)) \
                if len(data) > N else data
        url = "%s, POST data:\n%s" % (url, str_data)
    except AttributeError:
        url = obj
    return url
