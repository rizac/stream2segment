# -*- coding: utf-8 -*-
'''
Created on Mar 27, 2021

@author: riccardo
'''
from itertools import product

import pytest

from stream2segment.io import Fdsnws



def test_models_fdsn_url_1():
    for url in ["mock/fdsnws/station/1/query",
                "mock/fdsnws/station/1/query?",
                "https://mock/fdsnws/station/1/query",
                "http://mock/fdsnws/station/1/query?",
                "https://mock/fdsnws/station/1/",
                "https://mock/fdsnws/station/1",
                "http://mock/fdsnws/station/1/query?h=8&b=76",
                "https://mock/fdsnws/station/1/auth?h=8&b=76",
                # "mock/station/fdsnws/station/1/"  # invalid (see test_resif below)
                ]:
        fdsn = Fdsnws(url)
        expected_scheme = 'https' if url.startswith('https://') else 'http'
        assert fdsn.site == '%s://mock' % expected_scheme
        assert fdsn.service == Fdsnws.STATION
        assert str(fdsn.majorversion) == str(1)
        normalizedurl = fdsn.url()
        assert normalizedurl == '%s://mock/fdsnws/station/1/query' % expected_scheme
        for service in list(Fdsnws.SERVICES) + ['abc']:
            assert fdsn.url(service) == normalizedurl.replace('station', service)

        assert fdsn.url(majorversion=55) == normalizedurl.replace('1', '55')
        assert fdsn.url(majorversion='1.1') == normalizedurl.replace('1', '1.1')

        for method in list(Fdsnws.METHODS) + ['abcdefg']:
            assert fdsn.url(method=method) == normalizedurl.replace('query', method)

    for url in ["fdsnws/station/1/query",
                "/fdsnws/station/1/query",
                "http:mysite.org/fdsnws/dataselect/1",  # Note: this has invalid scheme
                "http:mysite.org/and/another/path/fdsnws/dataselect/1",
                "http://mysite.org/and/another/path/fdsnws/dataselect/1",
                "http://www.google.com",
                "https://mock/fdsnws/station/abc/1/whatever/abcde?h=8&b=76",
                "https://mock/fdsnws/station/", "https://mock/fdsnws/station",
                "https://mock/fdsnws/station/1/abcde?h=8&b=76",
                "https://mock/fdsnws/station/1/whatever/abcde?h=8&b=76",
                "mock/station/fdsnws/station/1/",
                "http://ws.resif.fr/ph5/fdsnws/dataselect/1/query"]:
        with pytest.raises(ValueError):
            Fdsnws(url)


def test_resif_url():
    with pytest.raises(ValueError):
        url1 = Fdsnws("http://ws.resif.fr/ph5/fdsnws/dataselect/1/query").url()

    url1 = Fdsnws("http://ws.resif.fr/ph5/fdsnws/dataselect/1/query",
                  strict_path=False).url()
    url2 = Fdsnws("http://ws.resif.fr/fdsnws/dataselect/1/query").url()
    assert url1 != url2
    assert url1.replace("/ph5", "") == url2


def test_models_fdsn_url():
    url_ = 'abc.org/fdsnws/station/1'
    for (pre, post, slash) in product(['', 'http://', 'https://'],
                                      ['' ] + list(Fdsnws.METHODS),
                                      ['', '/', '?']
                                      ):
        if not post and slash == '?':
            continue  # do not test "abc.org/fdsnws/station/1?" it's invalid
        elif slash == '?':
            asd = 6
        url = pre + url_ + ('/' if post else '') + post + slash
        fdsn = Fdsnws(url)
        if url.startswith('https'):
            assert fdsn.site == 'https://abc.org'
        else:
            assert fdsn.site == 'http://abc.org'
        assert fdsn.service == Fdsnws.STATION
        assert fdsn.majorversion == '1'

        normalizedurl = fdsn.url()
        for service in list(Fdsnws.SERVICES) + ['abc']:
            assert fdsn.url(service) == normalizedurl.replace('station', service)

        assert fdsn.url(majorversion=55) == normalizedurl.replace('1', '55')

        for method in list(Fdsnws.METHODS) + ['abcdefg']:
            assert fdsn.url(method=method) == normalizedurl.replace('query', method)


@pytest.mark.parametrize(['url_'],
                         [
                          ('',),
                          ('/fdsnws/station/1',),
                          ('fdsnws/station/1/',),
                          ('fdsnws/station/1/query',),
                          ('fdsnws/station/1/query/',),
                          ('abc.org',),
                          ('abc.org/',),
                          ('abc.org/fdsnws',),
                          ('abc.org/fdsnws/',),
                          ('abc.org/fdsnws/bla',),
                          ('abc.org/fdsnws/bla/',),
                          ('abc.org/fdsnws/bla/1',),
                          ('abc.org/fdsnws/bla/1r',),
                          ('abc.org/fdsnws/station/a',),
                          ('abc.org/fdsnws/station/b/',),
                          ('abc.org//fdsnws/station/1.1/',),
                          # ('abc.org/fdsnws/station/1?',),
                          ('abc.org/fdsnws/station/1.1//',),
                          ('abc.org/fdsnws/station/1.1/bla',),
                          ('abc.org/fdsnws/station/1.1/bla/',),])
def test_models_bad_fdsn_url(url_):
    for url in [url_, 'http://' + url_, 'https://'+url_]:
        with pytest.raises(ValueError):
            Fdsnws(url)
