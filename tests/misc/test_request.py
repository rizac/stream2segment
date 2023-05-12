"""
Created on Sep 18, 2018

@author: rizac
"""

import pytest
from urllib.request import Request

from stream2segment.download.url import read_async, urlread


def no_connection():
    from stream2segment.download.url import HTTPError
    try:
        data, err, code = urlread("https://geofon.gfz-potsdam.de/")
        return err is None or isinstance(err, HTTPError)
    except Exception:  # noqa
        return True


@pytest.mark.skipif(no_connection(),
                    reason="no internet connection")
def test_request():
    '''This tests `read_async` in case of a REAL connection. Ignored if the computer
    is not online'''

    post_data_str = """* * * HH?,HL?,HN? 2017-01-01T00:00:00 2017-06-01T00:00:00
format=text
level=channel"""
    urls = ["http://geofon.gfz-potsdam.de/fdsnws/station/1/query",
            "http://geofon.gfz-potsdam.de/fdsnws/station/1/query2"]
    ids = [1]
    iterable = ((id_, Request(url,
                              data=('format=text\nlevel=channel\n'+post_data_str).encode('utf8')))
                for url, id_ in zip(urls, ids))

    for obj, url, result, exc, code in read_async(iterable, urlkey=lambda obj: obj[-1],
                                                       blocksize=1048576,
                                                       max_workers=None,
                                                       decode='utf8', timeout=120):

        pass
#     r = Request("http://geofon.gfz-potsdam.de/fdsnws/station/1/query",
#                 data="""* * * HH?,HL?,HN? 2017-01-01T00:00:00 2017-06-01T00:00:00
# format=text
# level=channel""".encode('utf8'))
#     
#     urlread(r)
#     h = 9


from stream2segment.cli import cli

@pytest.mark.skipif(no_connection(),
                    reason="no internet connection")
def test_no_eventtype_column_db(clirunner, pytestdir, data):
    result = clirunner.invoke(cli, ['download',
                                    '-c', data.path('db.no_event_type_column.yaml'),
                                    ])
    assert ("No row saved to table 'events' (error: table events "
            "has no column named event_type)") in result.output
    assert result.exit_code != 0




