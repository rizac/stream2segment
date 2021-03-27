'''
Created on Sep 18, 2018

@author: rizac
'''

import pytest

from stream2segment.download.url import read_async, Request, urlread


def no_connection():
    try:
        urlread("https://geofon.gfz-potsdam.de/")
        return False
    except:
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

    for obj, result, exc, url in read_async(iterable, urlkey=lambda obj: obj[-1],
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