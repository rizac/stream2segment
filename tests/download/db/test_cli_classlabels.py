'''
Created on Feb 14, 2017

@author: riccardo
'''
from __future__ import print_function, division

import os
from click.testing import CliRunner
from mock import patch
import pandas as pd
import pytest

from stream2segment.cli import cli
from stream2segment.resources import get_templates_fpath
from stream2segment.io.db.models import get_classlabels
from stream2segment.process.db.models import (get_inventory, get_stream,
                                              Event, Station, Segment,
                                              Channel, Download, DataCenter, ClassLabelling,
                                              Class)
from stream2segment.process.main import query4process
from stream2segment.process.log import configlog4processing as o_configlog4processing


@pytest.fixture
def yamlfile(pytestdir):
    '''global fixture wrapping pytestdir.yamlfile'''
    def func(**overridden_pars):
        return pytestdir.yamlfile(get_templates_fpath('paramtable.yaml'), **overridden_pars)

    return func


def readcsv(filename, header=True):
    return pd.read_csv(filename, header=None) if not header else pd.read_csv(filename)


class Test(object):

    pyfile = get_templates_fpath("paramtable.py")

    @property
    def logfilecontent(self):
        assert os.path.isfile(self._logfilename)
        with open(self._logfilename) as opn:
            return opn.read()

    # The class-level `init` fixture is marked with autouse=true which implies that all test
    # methods in the class will use this fixture without a need to state it in the test
    # function signature or with a class-level usefixtures decorator. For info see:
    # https://docs.pytest.org/en/latest/fixture.html#autouse-fixtures-xunit-setup-on-steroids
    @pytest.fixture(autouse=True)
    def init(self, request, pytestdir, db4process):
        db4process.create(to_file=True)
        session = db4process.session

        class patches(object):
            # paths container for class-level patchers used below. Hopefully
            # will mek easier debug when refactoring/move functions
            valid_session = 'stream2segment.process.main.valid_session'
            close_session = 'stream2segment.process.main.close_session'
            configlog4processing = 'stream2segment.process.main.configlog4processing'

        # sets up the mocked functions: db session handling (using the already created
        # session) and log file handling:
        with patch(patches.valid_session, return_value=session):
            with patch(patches.close_session, side_effect=lambda *a, **v: None):
                with patch(patches.configlog4processing) as mock2:

                    def clogd(logger, logfilebasepath, verbose):
                        # config logger as usual, but redirects to a temp file
                        # that will be deleted by pytest, instead of polluting the program
                        # package:
                        o_configlog4processing(logger,
                                               pytestdir.newfile('.log') \
                                               if logfilebasepath else None,
                                               verbose)

                        self._logfilename = logger.handlers[0].baseFilename

                    mock2.side_effect = clogd

                    yield

    def inlogtext(self, string):
        '''Checks that `string` is in log text.
        The assertion `string in self.logfilecontent` fails in py3.5, although the differences
        between characters is the same position is zero. We did not find any better way than
        fixing it via this cumbersome function'''
        logtext = self.logfilecontent
        i = 0
        while len(logtext[i:i+len(string)]) == len(string):
            if (sum(ord(a)-ord(b) for a, b in zip(string, logtext[i:i+len(string)]))) == 0:
                return True
            i += 1
        return False

# ## ======== ACTUAL TESTS: ================================

    @patch('stream2segment.cli.input', side_effect=lambda *a, **kw: 'y')
    def test_classlabel_cmd(self, mock_input,
                            # fixtures:
                            db4process):

        # legacy code: get_classlabels was get_classes, feel lazy:
        get_classes = lambda session, *v: get_classlabels(session, Class)

        classes = get_classes(db4process.session)
        assert not classes
        runner = CliRunner()
        # test add a class from the command line argument
        result = runner.invoke(cli, ['db', 'classlabel',
                                     '-d', db4process.dburl,
                                     '--add', 'label', 'description'])
        assert not result.exception
        assert 'label (description)' in result.output
        classes = get_classes(db4process.session)
        assert classes[0]['label'] == 'label'
        assert classes[0]['description'] == 'description'
        # store id to be sure we will have from now on the same id:
        id_ = classes[0]['id']

        # test rename a class from the command line argument
        # only label, no description
        result = runner.invoke(cli, ['db', 'classlabel',
                                     '-d', db4process.dburl,
                                     '--rename', 'label', 'label2', ''])
        assert not result.exception
        assert 'label2 (description)' in result.output
        classes = get_classes(db4process.session)
        assert classes[0]['label'] == 'label2'
        assert classes[0]['description'] == 'description'
        assert classes[0]['id'] == id_

        # test rename a class and the description from the command line argument
        # only label, no description
        result = runner.invoke(cli, ['db', 'classlabel',
                                     '-d', db4process.dburl,
                                     '--rename', 'label2', 'label2',
                                     'description2'])
        assert not result.exception
        assert 'label2 (description2)' in result.output
        classes = get_classes(db4process.session)
        assert classes[0]['label'] == 'label2'
        assert classes[0]['description'] == 'description2'
        assert classes[0]['id'] == id_

        # add a class labelling
        assert len(db4process.session.query(ClassLabelling).all()) == 0
        segments = db4process.segments(False, False, False).all()
        cl = ClassLabelling(class_id=classes[0]['id'], segment_id=segments[0].id)
        db4process.session.add(cl)
        db4process.session.commit()
        assert len(db4process.session.query(ClassLabelling).all()) == 1

        # test delete a class from the command line argument
        # (non existing label)
        ccount =  mock_input.call_count
        assert ccount > 0
        result = runner.invoke(cli, ['db', 'classlabel',
                                     '--no-prompt'
                                     '-d', db4process.dburl,
                                     '--delete', 'label'])
        assert mock_input.call_count == ccount
        # The method assert result.exception
        # still same class:
        classes = get_classes(db4process.session)
        assert classes[0]['label'] == 'label2'
        assert classes[0]['description'] == 'description2'
        assert classes[0]['id'] == id_

        # test delete a class from the command line argument
        result = runner.invoke(cli, ['db', 'classlabel',
                                     '-d', db4process.dburl,
                                     '--delete', 'label2'])
        assert not result.exception
        assert 'None' in result.output
        assert mock_input.call_count == ccount + 1
        classes = get_classes(db4process.session)
        assert not classes
        assert len(db4process.session.query(ClassLabelling).all()) == 0
