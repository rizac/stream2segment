"""
Created on Dec 12, 2016

@author: riccardo
"""
from io import StringIO, BytesIO
from urllib.request import Request
from unittest.mock import Mock, patch
import pytest
from click.termui import progressbar

from stream2segment.download.url import (urlread, URLError, socket, HTTPError)
from stream2segment.io.cli import Nop, get_progressbar
from stream2segment.io.db import secure_dburl
from stream2segment.download.modules.utils import formatmsg


DEFAULT_TIMEOUT = socket._GLOBAL_DEFAULT_TIMEOUT  # noqa


@patch('stream2segment.download.url.urlopen')
def test_utils_url_read(mock_urlopen):

    def side_effect(argss):
        return StringIO(argss)

    mockread = Mock()
    class mybytesio:

        def __init__(self, url, **kwargs):
            mockread.reset_mock()
            if isinstance(url, Exception):
                self.a = url
            else:
                self.code = 200
                self.msg = 'Ok'
                self.a = BytesIO(url)

        def read(self, *a, **kw):
            if isinstance(self.a, Exception):
                raise self.a
            mockread(*a, **kw)
            return self.a.read(*a, **kw)

        def close(self, *a, **kw):
            if not isinstance(self.a, Exception):
                self.a.close()

        def __enter__(self,*a,**v):
            return self

        def __exit__(self, *a, **kw):
            return self.close(*a, **kw)


    mock_urlopen.side_effect = lambda url, **kw: mybytesio(url, **kw)
    with pytest.raises(TypeError):
        urlread('', "name")

    val = b'url'
    blockSize = 1024 * 1024
    assert urlread(val, blockSize)[0] == val
    mock_urlopen.assert_called_with(val)  # , timeout=DEFAULT_TIMEOUT)
    assert mockread.call_count == 2
    mockread.assert_called_with(blockSize)

    mock_urlopen.side_effect = lambda url, **kw: mybytesio(url, **kw)

    assert urlread(val, arg_to_read=56)[0] == val
    mock_urlopen.assert_called_with(val, arg_to_read=56)
    assert mockread.call_count == 1  # because blocksize is -1

    mock_urlopen.side_effect = lambda url, **kw: mybytesio(URLError('wat?'))
    d, e, c = urlread(val)
    assert isinstance(e, URLError)

    mock_urlopen.side_effect = lambda url, **kw: mybytesio(socket.timeout())
    d, e, c = urlread(val)
    assert isinstance(e, socket.error)

    mock_urlopen.side_effect = lambda url, **kw: mybytesio(HTTPError('url', 500, '?', None, None))
    d, e, c = urlread(val)
    assert isinstance(e, HTTPError)

    err = HTTPError('url', 500, '?', None, None)
    mock_urlopen.side_effect = lambda url, **kw: mybytesio(err)
    assert urlread(val) == (None, err, 500)


@pytest.mark.parametrize('input, expected_result, ',
                         [
                          ("postgresql://scott:@localhost/mydatabase",
                           "postgresql://scott:***@localhost/mydatabase"),
                          ("postgresql://scott:tiger@localhost/mydatabase",
                           "postgresql://scott:***@localhost/mydatabase"),
                          ('postgresql+psycopg2://scott:tiger@localhost/mydatabase',
                           'postgresql+psycopg2://scott:***@localhost/mydatabase'),
                          ('postgresql+pg8000://scott:tiger@localhost/mydatabase',
                           'postgresql+pg8000://scott:***@localhost/mydatabase'),
                          ('mysql://scott:tiger@localhost/foo',
                           'mysql://scott:***@localhost/foo'),
                          ('mysql+mysqldb://scott:tiger@localhost/foo',
                           'mysql+mysqldb://scott:***@localhost/foo'),
                          ('sqlite:////absolute/path/to/foo.db',
                           'sqlite:////absolute/path/to/foo.db')
                          ],
                        )
def test_secure_dburl(input, expected_result):
    assert secure_dburl(input) == expected_result


# IF RUNNING WITH ECLIPSE, UNCOMMENT THE LINES BELOW:
@pytest.mark.skip(reason="fails if run from within eclipse "
                         "because of cryptic bytes vs string propblem")
@patch("stream2segment.io.cli.Nop", side_effect=lambda *a, **v: Nop(*a, **v))
@patch("stream2segment.io.cli.click_progressbar", side_effect=lambda *a, **v: progressbar(*a, **v))
def test_progressbar(mock_pbar, mock_nop):
    """this test has problems with eclipse"""
    N = 5
    with get_progressbar(0) as bar:  # no-op
        for i in range(N):
            bar.update(i)
    assert mock_nop.call_count == 1
    assert mock_pbar.call_count == 0

    with get_progressbar(1) as bar:  # normal progressbar
        for i in range(N):
            bar.update(i)
    assert mock_nop.call_count == 1
    assert mock_pbar.call_count == 1

    with get_progressbar(1) as bar:  # normal progressbar
        for i in range(N):
            bar.update(i)
    assert mock_nop.call_count == 1
    assert mock_pbar.call_count == 2


# IF RUNNING WITH ECLIPSE, UNCOMMENT THE LINES BELOW:
# @pytest.mark.skip(reason="fails if run from within eclipse "
#                          "because of cryptic bytes vs string propblem")
def test_progressbar_functional():
    """this test has problems with eclipse"""
    N = 5
    with get_progressbar(0) as bar:  # no-op
        for i in range(N):
            bar.update(i)

    with get_progressbar(10) as bar:  # normal progressbar
        for i in range(N):
            bar.update(i)


def test_formatmsg():
    req = Request('http://mysite/query', data='a'*1000)
    msg = formatmsg("action", "errmsg", req)
    expected = ("action (errmsg). url: http://mysite/query, POST data:\n%s\n"
                "...(showing first 200 characters only)") % ('a' * 200)
    assert msg == expected

    req = Request('http://mysite/query', data='a\n'*5)
    msg = formatmsg("action", "errmsg", req)
    expected = ("action (errmsg). url: http://mysite/query, POST data:\n%s") % ('a\n' * 5)
    assert msg == expected.strip()

    req = Request('http://mysite/query', data=b'a\n'*5)
    msg = formatmsg("action", "errmsg", req)
    expected = ("action (errmsg). url: http://mysite/query, POST data:\n"
                "b'a\\na\\na\\na\\na\\n'")
    assert msg == expected.strip()
