'''
Created on Nov 30, 2016

Module handling messages to be displayed throughout the program
@author: riccardo
'''


def MSG(topic=None, action=None, errmsg=None, url=None):
    """Utility function which formats a message in order to have normalized message
    types across the program (e.g., in logging utilities). The argument can contain new
    (e.g., "{}") but also old-style format keywords (such as '%s', '%d') for usage within the
    logging functions (e.g. `logging.warning(MSG('dataselect ws', '%d segments discarded'), 3)`).
    The resulting string message will be in any of the following formats (according to
    how many arguments are truthy):
    ```
        "{topic}: {action} ({errmsg}). url: {url}"
        "{topic}: {action} ({errmsg})"
        "{topic}: {action}"
        "{topic}"
        "{action} ({errmsg}). url: {url}"
        "{action} ({errmsg})"
        "{action}"
        "{errmsg}. url: {url}"
        "{errmsg}"
        "{url}"
        ""
    ```
    :param topic: string or None: the topic of the message (e.g. "downloading channels")
    :param action: string or None: what has been done (e.g. "discarded 3 events")
    :param errmsg: string or Exception: the Exception or error message which caused the action
    :param url: the url (string) or urllib2.Request object: the url originating the message, if
    the latter was issued from a web request
    """
    msg = ""
    if topic:
        msg = topic
    if action:
        msg = "{}: {}".format(msg, action) if msg else action
    if errmsg:
        # sometimes exceptions have no message, append their name
        # (e.g. socket.timeout would now print at least 'timeout')
        strerr = str(errmsg) or str(errmsg.__class__.__name__)
        msg = "{} ({})".format(msg, strerr) if msg else strerr
    if url:
        _ = url2str(url)
        msg = "{}. url: {}".format(msg, _) if msg else _
    return msg


def url2str(obj):
    """converts an url or `urllib2.Request` object to string. In the latter case, the format is:
    "{obj.get_full_url()}" if `obj.data` is falsy
    "{obj.get_full_url()}, data: '{obj.get_data()}'" if `obj.data` has no newlines, or
    "{obj.get_full_url()}, data: '{obj.get_data()[:I]}'" otherwise (I=obj.get_data().find('\n')`)
    """
    full_url = getattr(obj, 'get_full_url', None)
    if full_url is not None:
        data = getattr(obj, 'get_data', None)
        if data is not None:
            idx = data.find("\n")
            url = "%s, data%s: '%s'" % (full_url, " (showing first line only)" if idx > -1 else '',
                                        data)
        else:
            url = full_url
    else:
        url = obj
    return url
