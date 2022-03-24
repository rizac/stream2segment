"""
Basic conftest.py defining fixtures to be accessed during tests

Created on 3 May 2018

@author: riccardo
"""

import os
from io import BytesIO
import traceback
import uuid
from datetime import datetime
import yaml

import pytest
import pandas as pd
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import close_all_sessions
from sqlalchemy.orm.session import sessionmaker
from obspy.core.stream import read as read_stream
from obspy.core.inventory.inventory import read_inventory

from click.testing import CliRunner

import stream2segment.download.db.models as dbd
import stream2segment.process.db.models as dbp
from stream2segment.traveltimes.ttloader import TTTable
from stream2segment.io import yaml_load
from stream2segment.download.modules.stations import compress


# https://docs.pytest.org/en/3.0.0/parametrize.html#basic-pytest-generate-tests-example
# add option --dburl to the command line
def pytest_addoption(parser):
    """Adds the dburl option to pytest command line arguments. The option can be input
    multiple times and will parametrize all tests with the 'db' fixture with all defined
    databases (plus a default SQLite database)
    """
    parser.addoption("--dburl", action="append", default=[],
                     help=("list of database url(s) to be used for testing "
                           "*in addition* to the default SQLite database"))


# make all test functions having 'db' in their argument use the passed databases
def pytest_generate_tests(metafunc):
    """This function is called before generating all tests and parametrizes all tests
    with the argument 'db' (which is a fixture defined below)
    """
    if 'db' in metafunc.fixturenames:
        dburls = ["sqlite:///:memory:",
                  os.getenv("DB_URL", None)] + metafunc.config.option.dburl  # command line (list)
        dburls = [_ for _ in dburls if _]
        ids = [_[:_.find('://')] for _ in dburls]
        # metafunc.parametrize("db", dburls)
        metafunc.parametrize('db', dburls,
                             ids=ids,
                             indirect=True, scope='module')


_PDOPT = {_: pd.get_option(_) for _ in ['display.max_colwidth']}


def pytest_sessionstart(session):  # pylint: disable=unused-argument
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """

    # https://stackoverflow.com/a/35394239
    pd.set_option('display.max_colwidth', 500)  # do not set to -1: it messes
    # alignement. Also, pandas 1.0+ wants None


def pytest_sessionfinish(session,  # pylint: disable=unused-argument
                         exitstatus):  # pylint: disable=unused-argument
    """
    Called after whole test run finished, right before
    returning the exit status to the system.
    """
    # we shouldn;t care about restoring pandas stuff because pytest exit to
    # the system. However:
    for k, v in _PDOPT.items():
        pd.set_option(k, v)


@pytest.fixture(scope="session")
def clirunner(request):  # pylint: disable=unused-argument
    """Shorthand for
        runner = CliRunner()
    with an additional method `assertok(result)` which asserts the returned value of
    the cli (command line interface) is ok, and prints relevant information to the
    standard output and error:

        result = clirunner.invoke(...)
        assert clirunner.ok(result)
    """
    class Clirunner(CliRunner):

        @classmethod
        def ok(cls, result):
            """Return True if result's exit_code is 0. If nonzero, prints
            relevant info to the standard output and error for debugging"""
            if result.exit_code != 0:
                print(result.output)
                if result.exception:
                    if result.exc_info[0] != SystemExit:
                        traceback.print_exception(*result.exc_info)
            return result.exit_code == 0

    return Clirunner()


@pytest.fixture(scope="session")
def data(request):  # noqa
    """Fixture handling all data to be used for testing. It points to the testing 'data'
    folder allowing to just get files / read file contents by file name.
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
    """
    class Data():
        """class handling common read operations on files in the 'data' folder"""

        def __init__(self, root=None):
            self.root = root or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'data')
            self._cache = {}

        def path(self, filename):
            """Return the full path (string) of the given data file name

            :param filename: a string denoting the file name inside the test data
                directory
            """
            filepath = os.path.join(self.root, filename)
            assert os.path.isfile(filepath)
            return filepath

        def read(self, filename):
            """Read the data (byte mode) and returns it

            :param filename: a string denoting the file name inside the test data
                directory
            """
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
            """Return the ObpPy stream object of the given filename

            :param filename: a string denoting the file name inside the test data
                directory"""
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
            """Return the ObpPy Inventory object of the given filename

            :param filename: a string denoting the file name inside the test data
                directory
            """
            key = ('inv', filename)
            inv = self._cache.get(key, None)
            if inv is None:
                inv = self._cache[key] = read_inventory(self.path(filename))
            # seems that inventories do not have a copy method as they cannot be modified
            # https://docs.obspy.org/packages/autogen/obspy.core.inventory.inventory.Inventory.html
            return inv

        def read_tttable(self, filename):
            """Return the stream2segment TTTable object of the given filename

            :param filename: a string denoting the file name inside the test data
            directory
            """
            # the cache assumes we do not modify ttable (which should be always the case)
            key = ('tttable', filename)
            tttable = self._cache.get(key, None)
            if tttable is None:
                tttable = self._cache[key] = TTTable(self.path(filename))
            return tttable

        def to_segment_dict(self, filename):
            """Return a dict to be passed as argument for creating new Segment(s), by
            reading an existing miniseed. The arrival time is set one third of the
            miniSeed time span
            """
            bdata = self.read(filename)
            stream = read_stream(BytesIO(bdata))
            start_time = stream[0].stats.starttime
            end_time = stream[0].stats.endtime
            # set arrival time to one third duration
            return dict(data=bdata,
                        arrival_time=(start_time + (end_time - start_time) / 3).datetime,
                        request_start=start_time.datetime,
                        request_end=end_time.datetime,
                        start_time=start_time.datetime,
                        end_time=end_time.datetime,
                        sample_rate=stream[0].stats.sampling_rate)

    return Data()


@pytest.fixture
def pytestdir(tmpdir):
    """This fixture "inherits" from tmpdir fixture in that it allows to create
    non-existing (unique) file paths and directories inside `tmpdir`, which is a
    LocalPath object
    (http://py.readthedocs.io/en/latest/_modules/py/_path/local.html#LocalPath)
    representing a temporary directory unique to each test function invocation.
    `str(tmpdir)` represents in turn a path inside the base temporary directory of the
    test session (`tmpdir_factory.mktemp`).
    Another option is to use `newyaml(file, **overrides)` or `newyaml(dict, **overrides)`
    which returns a file path to a new yaml with optional overrides

    Curiously, there is no feature in pytest to create random unique files and directory
    names.

    Usage:

    def my_test(pytestdir):
        pytestdir.path()           # Return a STRING denoting the path of this dir (i.e.
                                   # the string of the underlying tmpdir)
        pytestdir.newfile()        # Returns a STRING denoting an UNIQUE path in
                                   # `pytestdir.path`
        pytestdir.newfile('.py')   # Same as above, but with the given extension
        pytestdir.makedir()        # Returns a STRING denoting an UNIQUE (sub)directory
                                   # of `pytestdir.path
        pytestdir.join('out.csv')  # Return a STRING os.path.join(`self.path`, "out.csv")
        pytestdir.join('out.csv', True)  # same as above but file will be created if it
                                         # does not exists
    """

    class Pytestdir(object):
        """Pytestdir object"""

        @classmethod
        def yamlfile(cls, src, removals=None, **overrides):
            """Create a YAML file from src (a dict or path to yaml file) and optional
            overrides. Returns the newly created yaml file path
            """
            newyamlfile = cls.newfile('.yaml', create=False)
            data_ = yaml_load(src)
            data_.update(**overrides)
            for rem in [] if not removals else removals:
                data_.pop(rem, None)
            with open(newyamlfile, 'w') as outfile:
                yaml.dump(data_, outfile, default_flow_style=False)
            return newyamlfile

        @staticmethod
        def join(filename):
            """Returns a file under `self.path`. Implemented for compatibility with
            `tmpdir.join`
            """
            return str(tmpdir.join(filename))

        @staticmethod
        def path():
            """Return the string of the realpath of the underlying tmpdir"""
            return str(tmpdir.realpath)

        @staticmethod
        def newfile(name_or_extension=None, create=False):
            """Return an file path inside `self.path`. If 'name_or_extension' starts with
            the dot or is falsy (empty or None), the file is assured to be unique (no
            file with that name exists on `self.path`). If 'name_or_extension'  starts
            with a dot, then the (unique) file name has the given extension.
            If `create` is True (False by default), the file will be also created if it
            does not exist.
            Summary table:
            "unique" below denotes a random sequence of 16 alpha numeric digits which
            assures the file name does NOT denote an existing file name prior to this
            call.
            +---------------------------------+--------------------+--------------+
            |                                 | Returned           | Returned     |
            | call:                           | file name is:      | file exists: |
            +---------------------------------+--------------------+--------------+
            | newfile()                       | unique             | False        |
            | newfile(create=True)            |                    | True         |
            +---------------------------------+--------------------+--------------+
            | newfile('.py')                  | unique with suffix | False        |
            | newfile('.py', create=True)     | (extension) '.py'  | True         |
            +---------------------------------+--------------------+--------------+
            | newfile('abc.txt')              | 'abc.txt'          | it may exist |
            | newfile('abc.txt', create=True) |                    | True         |
            +---------------------------------+--------------------+--------------+

            :param name_or_extension: string. if truthy (not empty and not None), and
                does not start with a dot, it's the file name. If it starts with a dot,
                an unique file name is created with suffix `name_or_extension`. If falsy
                (empty or None), an unique file name is created
            :param create: boolean (default False) whether to create the given file
                before returning its path (string) if it does not exist.
            """
            if not name_or_extension:
                extension = ''
                filename = None
            elif name_or_extension[:1] == '.':
                extension = name_or_extension
                filename = None
            else:
                extension = ''
                filename = name_or_extension
            while filename is None:
                filename = str(uuid.uuid4()) + extension
                _path = str(tmpdir.join(filename).realpath())
                if os.path.exists(_path) or os.path.isfile(_path):
                    filename = None
            path = str(tmpdir.join(filename).realpath())
            if create:
                open(path, 'a').close()
            return path

        @staticmethod
        def makedir():
            """creates and returns a unique (sub)directory of `self.path`"""
            name = None
            while True:
                name = str(uuid.uuid4())
                ret = str(tmpdir.join(name).realpath())
                if not os.path.exists(ret) and not os.path.isdir(ret):
                    break
            return str(tmpdir.mkdir(name))

    return Pytestdir


@pytest.fixture
def db(request, tmpdir_factory):  # pylint: disable=invalid-name
    """Fixture handling all db reoutine stuff. Pass it as argument to a test function
    ```
        def test_bla(..., db,...)
    ```
    and just use `db.session` inside the code
    """
    class DB(object):
        """class handling a database in testing functions. You should call self.create
        inside the tests"""
        def __init__(self, dburl):
            self.dburl = dburl
            self._session = None
            self.engine = None
            self.session_maker = None

        def create(self, to_file=False, process=False, custom_base=None):
            """Create the database, deleting it if already existing (i.e., if this
            method has already been called and self.delete has not been called)

            :param to_file: boolean (False by default) tells whether, if the url denotes
                sqlite, the database should be created on the filesystem. Creating
                in-memory sqlite databases is handy in most cases but once the session
                is closed it seems that the database is closed too.
            :param process: ignored if base is supplied, if True attaches obspy methods
                to the ORM classes (streams2segment.process.db)
            :param custom_base: the Base class whereby creating the db schema. If None
                (the default) it defaults to the download ot process Base defined in s2s,
                depending on the value of the `process` argument
            """
            self.delete()
            self.dburl = 'sqlite:///' + \
                str(tmpdir_factory.mktemp('db', numbered=True).join('db.sqlite')) \
                if self.is_sqlite and to_file else self.dburl

            self.engine = create_engine(self.dburl)
            self._base = custom_base
            if self._base is None:
                self._base = dbp.Base if process else dbd.Base
            self._base.metadata.create_all(self.engine)  # @UndefinedVariable

        @property
        def session(self):
            """Create a session if not already created and returns it"""
            if self.engine is None:
                raise TypeError('Database not created. Call `create` first')
            if self._session is None:
                session_maker = self.session_maker = sessionmaker(bind=self.engine)
                # create a Session
                self._session = session_maker()
            return self._session

        @property
        def is_sqlite(self):
            """Return True if this db is sqlite, False otherwise"""
            return (self.dburl or '').startswith("sqlite:///")

        @property
        def is_postgres(self):
            """Return True if this db is postgres, False otherwise"""
            return (self.dburl or '').startswith("postgresql://")

        def delete(self):
            """Delete this da tabase, i.e. all tables and the file referring to it,
            if any
            """
            if self._session is not None:
                try:
                    self.session.rollback()
                    self.session.close()
                except:  # @IgnorePep8 pylint: disable=bare-except
                    pass
                self._session = None

            # 'drop_all' below hangs sometimes. The fix is to type beforehand a
            # `self.session_maker.close_all()` (https://stackoverflow.com/a/44437760)
            # which is now deprecated in favour of:
            close_all_sessions()  # https://stackoverflow.com/a/62884420

            if self.engine:
                try:
                    self._base.metadata.drop_all(self.engine)  # @UndefinedVariable
                except:  # @IgnorePep8 pylint: disable=bare-except
                    pass
                self.engine.dispose()

            # clear file if sqlite:
            sqlite = "sqlite:///"
            if self.dburl.startswith(sqlite):
                filename = self.dburl[len(sqlite):]
                if filename != ':memory:' and os.path.isfile(filename):
                    try:
                        os.remove(filename)
                    except:  # @IgnorePep8 pylint: disable=bare-except
                        pass

    ret = DB(request.param)
    request.addfinalizer(ret.delete)
    return ret


@pytest.fixture
def db4process(db, data):
    """This fixture basically extends the `db` fixture and returns an object with all the
    method of the `db` object (db.dburl, db.session) plus:
    db4process.segments(self, with_inventory, with_data, with_gap)
    """
    class _ProcessDB(object):
        """So, no easy way to override easily with pytest from the object returned by the
        `db` fixture. The best way is to pass `db` as argument above, which also assures
        that functions/methods having `db4process` as arguments will behave as those
        having `db` (i.e., they will be called iteratively for any database url passed in
        the command line). Drawback: we cannot override the class returned by `db`, so we
        provide a different class which mimics inheritance by forwarding to db each
        attribute not found (__getatttr__). Whether there might be a better way to
        achieve this, it wasn't clear from pytest docs
        """
        def __getattr__(self, name):
            """Lets the user call db4process.dburl, db4process.session,..."""
            return getattr(db, name)

        def segments(self, with_inventory, with_data, with_gap):
            """Return the segment ids matching the given criteria"""
            data_seed_id = 'ok.' if with_inventory else 'no.'
            data_seed_id += 'ok' if with_data else ('gap' if with_gap else 'no')
            return self.session.query(dbp.Segment).\
                filter(dbp.Segment.data_seed_id == data_seed_id)

        def create(self, to_file=False):
            """Call `db.create` and then populates the database with the data for
            processing tests
            """
            # re-init a sqlite database (no-op if the db is not sqlite):
            db.create(to_file, True)
            # init db:
            session = db.session

            # Populate the database:
            dwl = dbp.Download()
            session.add(dwl)
            session.commit()

            wsv = dbp.WebService(id=1, url='eventws')
            session.add(wsv)
            session.commit()

            # setup an event:
            ev1 = dbp.Event(id=1, webservice_id=wsv.id, event_id='abc1', latitude=8, longitude=9,
                        magnitude=5, depth_km=4, time=datetime.utcnow())
            ev2 = dbp.Event(id=2, webservice_id=wsv.id, event_id='abc2', latitude=8, longitude=9,
                        magnitude=5, depth_km=4, time=datetime.utcnow())
            ev3 = dbp.Event(id=3, webservice_id=wsv.id, event_id='abc3', latitude=8, longitude=9,
                        magnitude=5, depth_km=4, time=datetime.utcnow())

            session.add_all([ev1, ev2, ev3])
            session.commit()

            dtc = dbp.DataCenter(station_url='asd', dataselect_url='sdft')
            session.add(dtc)
            session.commit()

            # s_ok stations have lat and lon > 11, other stations do not
            inv_xml = data.read("inventory_GE.APE.xml")
            s_ok = dbp.Station(datacenter_id=dtc.id, latitude=11, longitude=12, network='ok',
                           station='ok', start_time=datetime.utcnow(),
                           inventory_xml=compress(inv_xml))
            session.add(s_ok)
            session.commit()

            s_none = dbp.Station(datacenter_id=dtc.id, latitude=-31, longitude=-32, network='no',
                             station='no', start_time=datetime.utcnow())
            session.add(s_none)
            session.commit()

            c_ok = dbp.Channel(station_id=s_ok.id, location='ok', channel="ok", sample_rate=56.7)
            session.add(c_ok)
            session.commit()

            c_none = dbp.Channel(station_id=s_none.id, location='no', channel="no", sample_rate=56.7)
            session.add(c_none)
            session.commit()

            atts_ok = dict(data.to_segment_dict('trace_GE.APE.mseed'))
            atts_gap = data.to_segment_dict('IA.BAKI..BHZ.D.2016.004.head')
            atts_none = dict(atts_ok, data=b'')

            for ch_ in (c_ok, c_none):

                # ch_.location  below reflects if the station has inv
                atts = dict(atts_ok, data_seed_id='%s.ok' % ch_.location, download_code=200)
                sg1 = dbp.Segment(channel_id=ch_.id, datacenter_id=dtc.id, event_id=ev1.id,
                              download_id=dwl.id, event_distance_deg=35, **atts)
                atts = dict(atts_gap, data_seed_id='%s.gap' % ch_.location, download_code=200)
                sg2 = dbp.Segment(channel_id=ch_.id, datacenter_id=dtc.id, event_id=ev2.id,
                              download_id=dwl.id, event_distance_deg=35, **atts)
                atts = dict(atts_none, data_seed_id='%s.no' % ch_.location, download_code=204)
                sg3 = dbp.Segment(channel_id=ch_.id, datacenter_id=dtc.id, event_id=ev3.id,
                              download_id=dwl.id, event_distance_deg=35, **atts)
                session.add_all([sg1, sg2, sg3])
                session.commit()

    return _ProcessDB()
