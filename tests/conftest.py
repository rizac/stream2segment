'''
Basic conftest.py defining fixtures to be accessed during tests

Created on 3 May 2018

@author: riccardo
'''

import tempfile
import os
from io import BytesIO

import pytest
from sqlalchemy.engine import create_engine
from sqlalchemy.orm.session import sessionmaker
from obspy.core.stream import read as read_stream
from obspy.core.inventory.inventory import read_inventory

from stream2segment.io.db.models import Base
from stream2segment.traveltimes.ttloader import TTTable


# https://docs.pytest.org/en/3.0.0/parametrize.html#basic-pytest-generate-tests-example
# add option --dburl to the command line
# FIXME: how do we invoke the help command? what is the library used under the hood (OptionParse)?
def pytest_addoption(parser):
    '''Adds the dburl option to pytest command line arguments. The option can be input multiple
    times and will parametrize all tests with the 'db' fixture with all defined databases
    (plus a default SQLite database)'''
    parser.addoption("--dburl", action="append", default=[],
                     help=("list of database url(s) to be used for testing *in addition* "
                           "to the default SQLite database"))


# def _sqlite(string):
#     return string[:10] == 'sqlite:///'


# make all test functions having 'db' in their argument use the passed databases
def pytest_generate_tests(metafunc):
    '''This function is called before generating all tests and parametrizes all tests with the
    argument 'db' (which is a fixture defined below)'''
    if 'db' in metafunc.fixturenames:
        dburls = ["sqlite:///:memory:",
                  os.getenv("DB_URL", None)] + metafunc.config.option.dburl  # command line (list)
        dburls = [_ for _ in dburls if _]
        ids = [_[:_.find('/')] for _ in dburls]
        # metafunc.parametrize("db", dburls)
        metafunc.parametrize('db', dburls,
                             ids=ids,
                             indirect=True, scope='module')


@pytest.fixture
def db(request):  # pylint: disable=invalid-name
    '''Fixture handling all db reoutine stuff. Pass it as argument to a test function
    ```
        def test_bla(..., db,...)
    ```
    and just use `db.session` inside the code
    '''
    class DB(object):
        '''class handling a database in testing functions'''
        def __init__(self, dburl):
            self.dburl = dburl
            self._base = Base
            self._reinit_vars()

        def _reinit_vars(self):
            self._session = None
            self.engine = None

        def create(self, to_file=False, base=None):
            '''creates the database, deleting it if already existing (i.e., if this method
            has already been called and self.delete has not been called).
            :param to_file: boolean (False by default) tells whether, if the url denotes sqlite,
            the database should be created on the filesystem. Creating in-memory sqlite databases
            is handy in most cases but once the session is closed it seems that the database is
            closed too.
            :param base: the Base class whereby creating the db schema. If None (the
            default) it defaults to streams2segment.io.db.models.Base
            '''
            self.delete()
            self.dburl = 'sqlite:///' + tempfile.NamedTemporaryFile(delete=False).name \
                if self.is_sqlite and to_file else self.dburl

            self.engine = create_engine(self.dburl)
            # Base.metadata.drop_all(self.engine)  # @UndefinedVariable
            if base is not None:
                self._base = base
            self._base.metadata.create_all(self.engine)  # @UndefinedVariable

        @property
        def session(self):
            '''creates a session if not already created and returns it'''
            if self.engine is None:
                raise TypeError('Database not created. Call `create` first')
            if self._session is None:
                session_maker = sessionmaker(bind=self.engine)
                # create a Session
                self._session = session_maker()
            return self._session

        @property
        def is_sqlite(self):
            '''returns True if this db is sqlite, False otherwise'''
            return (self.dburl or '').startswith("sqlite:///")

        @property
        def is_postgres(self):
            '''returns True if this db is postgres, False otherwise'''
            return (self.dburl or '').startswith("postgresql://")

        def delete(self):
            '''Deletes this da tabase, i.e. all tables and the file referring to it, if any'''
            if self.engine:
                if self._session is not None:
                    try:
                        self.session.rollback()
                        self.session.close()
                    except:  # @IgnorePep8 pylint: disable=bare-except
                        pass
                try:
                    self._base.metadata.drop_all(self.engine)  # @UndefinedVariable
                except:  # @IgnorePep8 pylint: disable=bare-except
                    pass
            # clear file if sqlite:
            sqlite = "sqlite:///"
            if self.dburl.startswith(sqlite):
                filename = self.dburl[len(sqlite):]
                if filename != ':memory:' and os.path.isfile(filename):
                    try:
                        os.remove(filename)
                    except:  # @IgnorePep8 pylint: disable=bare-except
                        pass
            self._reinit_vars()

    ret = DB(request.param)
    request.addfinalizer(ret.delete)
    return ret


@pytest.fixture(scope="session")
def data(request):  # pylint: disable=unused-argument
    '''Fixture handling all data to be used for testing. It points to the testing 'data' folder
    allowing to just get files / read file contents by file name.
    Pass it as argument to a test function
    ```
        def test_bla(..., data,...)
    ```
    and just use its methods inside the code, e.g,:
    ```
        data.path('myfilename')
        data.read('myfilename')
        data.read_stream('myfilename')
        data.read_inv('myfilename')
    ```
    '''
    class Data():
        '''class handling common read operations on files in the 'data' folder'''

        def __init__(self, root=None):
            self.root = root or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
            self._cache = {}

        def path(self, filename):
            '''returns the full path (string) of the given data file name
            :param filename: a string denoting the file name inside the test data directory'''
            filepath = os.path.join(self.root, filename)
            assert os.path.isfile(filepath)
            return filepath

        def read(self, filename):
            '''reads the data (byte mode) and returns it
            :param filename: a string denoting the file name inside the test data directory
            '''
            key = ('raw', filename)
            bytesdata = self._cache.get(key, None)
            if bytesdata is None:
                with open(self.path(filename), 'rb') as opn:
                    bytesdata = self._cache[key] = opn.read()
            return bytesdata

        def _read_stream(self, filename):
            key = ('stream', filename, None, None)
            stream = self._cache.get(key, None)
            if stream is None:
                stream = self._cache[key] = read_stream(self.path(filename))  # , format='MSEED')
            return stream.copy()

        def read_stream(self, filename, inventory_filename=None, output=None):
            '''Returns the obpsy stream object of the given filename
            :param filename: a string denoting the file name inside the test data directory'''
            assert (inventory_filename is None) == (output is None)
            # read the non-processed stream
            stream = self._read_stream(filename)  # returns a copy
            if inventory_filename is None:
                return stream
            key = ('stream', filename, inventory_filename, output)
            ret_stream = self._cache.get(key, None)
            if ret_stream is None:
                inv_obj = self.read_inv(inventory_filename)
                stream.remove_response(inv_obj, output=output)
                ret_stream = self._cache[key] = stream
            return ret_stream.copy()

        def read_inv(self, filename):
            '''Returns the obpsy Inventory object of the given filename
            :param filename: a string denoting the file name inside the test data directory'''
            key = ('inv', filename)
            inv = self._cache.get(key, None)
            if inv is None:
                inv = self._cache[key] = read_inventory(self.path(filename))
            # seems that inventories do not have a copy method as they cannot be modified
            # https://docs.obspy.org/packages/autogen/obspy.core.inventory.inventory.Inventory.html
            return inv

        def read_tttable(self, filename):
            '''Returns the stream2segment TTTable object of the given filename
            :param filename: a string denoting the file name inside the test data directory'''
            # the cache assumes we do not modify ttable (which should be always the case)
            key = ('tttable', filename)
            tttable = self._cache.get(key, None)
            if tttable is None:
                tttable = self._cache[key] = TTTable(self.path(filename))
            return tttable

        def to_segment_dict(self, filename):
            '''returns a dict to be passed as argument for creating new Segment(s), by reading
            an existing miniseed. The arrival time is set one third of the miniseed time span'''
            bdata = self.read(filename)
            stream = read_stream(BytesIO(bdata))
            start_time = stream[0].stats.starttime
            end_time = stream[0].stats.endtime
            # set arrival time to one third duration
            return dict(data=bdata,
                        arrival_time=(start_time + (end_time - start_time) / 3).datetime,
                        request_start=start_time.datetime,
                        request_end=end_time.datetime,
                        sample_rate=stream[0].stats.sampling_rate)

    return Data()
