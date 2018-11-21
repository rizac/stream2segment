'''
Created on 14 Mar 2018

@author: riccardo
'''
import os

from stream2segment.utils.resources import yaml_load, get_templates_fpath, get_templates_fpaths,\
    get_templates_dirpath


def test_yaml_load():
    # NB: all dic keys must be strings
    dic1 = {'a': 7, '5': 'h'}
    dic2 = {'a': 7, '7': 'h'}
    d = yaml_load(dic1, **dic2)
    assert d['a'] == 7
    assert d['5'] == 'h'
    assert d['7'] == 'h'
    assert sorted(d.keys()) == sorted(['a', '5', '7'])

    dic1 = {'a': 7, '5': 'h', 'v': {1: 2, 3: 4}}
    dic2 = {'a': 7, '7': 'h', 'v': {1: 2, 3: 5}}
    d = yaml_load(dic1, **dic2)
    assert d['a'] == 7
    assert d['5'] == 'h'
    assert d['7'] == 'h'
    assert d['v'][1] == 2
    assert d['v'][3] == 5
    assert sorted(d.keys()) == sorted(['a', '5', '7', 'v'])

    dic1 = yaml_load(get_templates_fpath('download.yaml'))
    key2test = 'minlat'
    # This will also asserts minlat is a valid key. Otherwise, change to a valid key:
    val2test = dic1['eventws_query_args'][key2test]
    dic2 = yaml_load(get_templates_fpath('download.yaml'),
                     eventws_query_args={key2test: val2test - 1.1, 'wawa': 45.5})
    assert dic2['eventws_query_args'][key2test] == val2test - 1.1
    assert dic2['eventws_query_args']['wawa'] == 45.5

    keys1 = set(dic1['eventws_query_args'])
    keys2 = set(dic2['eventws_query_args'])

    assert keys1 - keys2 == set()
    assert keys2 - keys1 == set(['wawa'])


def test_templates_fpath():
    basedir = get_templates_dirpath()

    res = get_templates_fpaths()
    assert sorted(res) == sorted(os.path.join(basedir, n) for n in os.listdir(basedir))

    filenames = ['a', 'b']
    res = get_templates_fpaths(*filenames)
    assert sorted(res) == sorted(os.path.join(get_templates_dirpath(), n) for n in filenames)

