'''
Created on Apr 11, 2017

@author: riccardo
'''
import unittest
from sqlalchemy.ext.declarative.api import declarative_base
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.orm.scoping import scoped_session
import pandas as pd
from sqlalchemy import Column, Integer, String,  create_engine, Binary, DateTime
from stream2segment.io.db.pd_sql_utils import fetchsetpkeys, insertdf, _get_max, syncdf,\
    mergeupdate, updatedf
from datetime import datetime
import numpy as np
from itertools import izip
from sqlalchemy.sql.expression import bindparam

Base = declarative_base()

class Customer(Base):
    __tablename__ = "customer"
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, default='a')
    data = Column(Binary)
    time = Column(DateTime)


class Test(unittest.TestCase):


    def setUp(self):
        self.dburl = "sqlite:///:memory:"
        
        
        DBSession = scoped_session(sessionmaker())
        self.engine = create_engine(self.dburl, echo=False)
        DBSession.remove()
        DBSession.configure(bind=self.engine, autoflush=False, expire_on_commit=False)
        # Base.metadata.drop_all(engine)
        Base.metadata.create_all(self.engine)
        
        self.session = DBSession

    def tearDown(self):
        pass


    def testName(self):
        pass

    def init_db(self, list_of_dicts):
        for d in list_of_dicts:
            customer = Customer(**d)
            self.session.add(customer)
        self.session.commit()

    
    def test_insertdf_double_single_matching_col(self):
        self.init_db([{'id':1}, {'id':2}, {'id':3}])
        
        
        # add the same ids as we have on the table. Assert nothing is added, d is returned as it is:
        d = pd.DataFrame([{'id':1, 'name': 'a'}, {'id':2, 'name': 'b'}, {'id':3, 'name': 'c'}])
        dlen = len(d)
         
        newd, new = insertdf(d, self.session, [Customer.id])
        assert dlen - len(newd) == 0
        assert new == 0
        assert dlen == len(newd)

        # add the same ids as we have on the table, with dupes id.
        d = pd.DataFrame([{'id': 3, 'name': 'a'}, {'id':2, 'name': 'b'}, {'id':3, 'name': 'c'}])
        dlen = len(d)
        newd, new = insertdf(d, self.session, [Customer.id], drop_duplicates=False)
        # discarded is still 0, as we consider first and third instance the same
        assert dlen - len(newd) == 0
        assert new == 0
        assert len(newd) == dlen
        
        # As above, but with default drop_dupes=True discarded is now 1 (the second duplicate: {'id':3, 'name': 'c'})
        d = pd.DataFrame([{'id': 3, 'name': 'a'}, {'id':2, 'name': 'b'}, {'id':3, 'name': 'c'}])
        dlen = len(d)
        newd, new = insertdf(d, self.session, [Customer.id])
        # discarded is still 0, as we consider first and third instance the same
        assert dlen - len(newd) == 1
        assert new == 0
        assert len(newd) == dlen - 1
        
        # add the a new id (4). Assert one is added, d is returned as it is:
        d = pd.DataFrame([{'id':4, 'name': 'a'}, {'id':2, 'name': 'b'}, {'id':3, 'name': 'c'}])
        dlen = len(d)
        newd, new = insertdf(d, self.session, [Customer.id])
        assert dlen - len(newd) == 0
        assert new == 1
        assert dlen == len(newd)
        
        # raise an error: duplicated ids:
        d = pd.DataFrame([{'id':45, 'name': 'a'}, {'id':45, 'name': 'b'}, {'id':3, 'name': 'c'}])
        
        dlen = len(d)
        newd, new = insertdf(d, self.session, [Customer.id])
        discarded = dlen - len(newd)
        assert discarded == 1
        assert new == 1
        assert dlen == len(newd) + discarded



    def test_insertdf_double_matching_cols(self):

        d2006 = datetime(2006, 1, 1)
        d2007 = datetime(2007, 12, 31)
        d2008 = datetime(2008, 3, 14)
        d2009 = datetime(2009, 9, 25)

        self.init_db([{'id': 1, 'time': d2006}, {'id': 2, 'time': d2007}, {'id': 3, 'time': d2008}])

        # add the same ids as we have on the table. Assert nothing is added, d is returned as it is:
        d = pd.DataFrame([{'id': 1, 'time': d2006}, {'id': 2, 'time': d2007}, {'id': 3, 'time': d2008}])
        dlen = len(d)
        newd, new = insertdf(d, self.session, [Customer.id, Customer.time])
        assert dlen - len(newd) == 0
        assert new == 0
        assert dlen == len(newd)


        # add a wrong entry {'id': 1, 'time': d2009}: the tuple (id, time) is not present on the
        # db BUT it violates id unique constraint: 
        d = pd.DataFrame([{'id': 1, 'time': d2009}, {'id': 2, 'time': d2007}, {'id': 3, 'time': d2008}])
        dlen = len(d)
        newd, new = insertdf(d, self.session, [Customer.id, Customer.time])
        assert dlen - len(newd) == 1
        assert new == 0
        assert len(newd) == dlen - 1

        # add a wrong entry {'id': 1, 'time': d2009} and a good one {'id': 22, 'time': d2009},
        # but as the wrong entry
        # is before the good one and in the same buffer chunk, the second is not 
        # added too
        d = pd.DataFrame([{'id': 1, 'time': d2009}, {'id': 22, 'time': d2009}, {'id': 3, 'time': d2008}])
        dlen = len(d)
        newd, new = insertdf(d, self.session, [Customer.id, Customer.time])
        assert dlen - len(newd) == 2
        assert new == 0
        assert len(newd) == dlen - 2

        # add a wrong entry {'id': 1, 'time': d2009} and a good one {'id': 22, 'time': d2009},
        # with buf_size=1 the second is added (they are not in the same buf chunk)
        d = pd.DataFrame([{'id': 1, 'time': d2009}, {'id': 22, 'time': d2009}, {'id': 3, 'time': d2008}])
        dlen = len(d)
        newd, new = insertdf(d, self.session, [Customer.id, Customer.time], buf_size=1)
        assert dlen - len(newd) == 1
        assert new == 1
        assert len(newd) == dlen - 1

        # add a good entry {'id': 23, 'time': d2009} and a wrong one {'id': 1, 'time': d2009},
        # now at least the good entry is added because it's BEFORE the wrong one
        d = pd.DataFrame([{'id': 23, 'time': d2009}, {'id': 1, 'time': d2009}, {'id': 3, 'time': d2008}])
        dlen = len(d)
        newd, new = insertdf(d, self.session, [Customer.id, Customer.time])
        assert dlen - len(newd) == 1
        assert new == 1
        assert len(newd) == dlen - 1

        # add a new good entry {'id': 411, 'time': d2009}, the other three already added
        # again, note that second and last entry are all added cause dupes are not checked for
        # we should issue a drop_duplicates first
        d = pd.DataFrame([{'id': 411, 'time': d2009}, {'id': 22, 'time': d2009},
                          {'id': 3, 'time': d2008}, {'id': 22, 'time': d2009}])
        
        dlen = len(d)
        newd, new = insertdf(d, self.session, [Customer.id, Customer.time], drop_duplicates=False)
        assert dlen - len(newd) == 0
        assert new == 1
        assert len(newd) == dlen
        
        # Same as above, but now we drop duplicates (the default):
        d = pd.DataFrame([{'id': 411, 'time': d2009}, {'id': 22, 'time': d2009},
                          {'id': 3, 'time': d2008}, {'id': 22, 'time': d2009}])
        
        dlen = len(d)
        newd, new = insertdf(d, self.session, [Customer.id, Customer.time], drop_duplicates=True)
        assert dlen - len(newd) == 1  # the second {'id': 22, 'time': d2009} is discarded
        assert new == 0  # nothing is added (as new)
        assert len(newd) == dlen -1

    def test_fetchsetpkeys_(self):
        d2006 = datetime(2006, 1, 1)
        d2007 = datetime(2007, 12, 31)
        d2008 = datetime(2008, 3, 14)
        d2009 = datetime(2009, 9, 25)
        self.init_db([{'id':1, 'name':'a'}, {'id':2, 'name':'b'}, {'id':3, 'name': "c"}])
        d = pd.DataFrame([{'name': 'a'}, {'name': 'b'}, {'name': 'd'}])
        d2 = fetchsetpkeys(d, self.session, [Customer.name], Customer.id)
        expected_ids = [1, 1, np.nan]
        assert array_equal(d2['id'], expected_ids)
        
    def test_fetchsetpkeys_2(self):
        d2006 = datetime(2006, 1, 1)
        d2007 = datetime(2007, 12, 31)
        d2008 = datetime(2008, 3, 14)
        d2009 = datetime(2009, 9, 25)
        self.init_db([{'id':1, 'name':'a'}, {'id':2, 'name':'b'}])
        d = pd.DataFrame([{'name': 'a'}, {'name': 'b'}, {'name': 'd'}])
        d2 = fetchsetpkeys(d, self.session, [Customer.name], Customer.id)
        expected_ids = [1,2,np.nan]
        assert array_equal(d2['id'], expected_ids)
        
    def test_fetchsetpkeys_3(self):
        d2006 = datetime(2006, 1, 1)
        d2007 = datetime(2007, 12, 31)
        d2008 = datetime(2008, 3, 14)
        d2009 = datetime(2009, 9, 25)
        self.init_db([{'id':1, 'name':'a', 'time': d2006}, {'id':2, 'name':'a', 'time': d2008},
                      {'id':3, 'name':'a', 'time': None}])
        d = pd.DataFrame([{'name': 'a', 'time': None}, {'name': 'b', 'time': d2006},
                          {'name': 'c', 'time': d2009}, {'name': 'a', 'time': None},
                          {'name': 'a', 'time': d2006}])
        d2 = fetchsetpkeys(d, self.session, [Customer.name, Customer.time], Customer.id)
        expected_ids = [3,np.nan,np.nan,3,1]
        assert array_equal(d2['id'], expected_ids)
        # assert pd.isnull(d2.loc[d2['name'] == 'd']['id']).all()
        
    def test_fetchsetpkeys_4(self):
        d2006 = datetime(2006, 1, 1)
        d2007 = datetime(2007, 12, 31)
        d2008 = datetime(2008, 3, 14)
        d2009 = datetime(2009, 9, 25)
        self.init_db([{'id':1, 'name':'a', 'time': d2006}, {'id':2, 'name':'a', 'time': d2008},
                      {'id':3, 'name':'a', 'time': None}])
        d = pd.DataFrame([{'id':45, 'name': 'a', 'time': None}, {'id':45, 'name': 'b', 'time': d2006},
                          {'id':45, 'name': 'c', 'time': d2009}, {'id':45, 'name': 'a', 'time': None},
                          {'id':45, 'name': 'a', 'time': d2006}])
        d2 = fetchsetpkeys(d, self.session, [Customer.name, Customer.time], Customer.id)
        expected_ids = [3,45, 45, 3, 1]
        assert array_equal(d2['id'], expected_ids)
        
    
    def test_syncdf(self):
        d2006 = datetime(2006, 1, 1)
        d2007 = datetime(2007, 12, 31)
        d2008 = datetime(2008, 3, 14)
        d2009 = datetime(2009, 9, 25)
        self.init_db([{'id':1, 'name':'a', 'time': d2006}, {'id':2, 'name':'a', 'time': d2008},
                      {'id':3, 'name':'a', 'time': None}])
        mxx = _get_max(self.session, Customer.id)
        # expected_ids should be [3, mxx+1, mxx+2, 3, 1]
        # but dataframe 1st item and 4th item are the same, the second will be dropped, thus:
        expected_ids = [3, mxx+1, mxx+2, 1]
        d = pd.DataFrame([{'name': 'a', 'time': None}, {'name': 'b', 'time': d2006},
                          {'name': 'c', 'time': d2009}, { 'name': 'a', 'time': None},
                          {'name': 'a', 'time': d2006}])
        d2, new = syncdf(d, self.session, [Customer.name, Customer.time], Customer.id)
        
        assert array_equal(d2['id'], expected_ids)
    
    def test_syncdf_2(self):
        """Same as above but the second non-existing item is conflicting with the first added"""
        d2006 = datetime(2006, 1, 1)
        d2007 = datetime(2007, 12, 31)
        d2008 = datetime(2008, 3, 14)
        d2009 = datetime(2009, 9, 25)
        mxx = _get_max(self.session, Customer.id)
        self.init_db([{'id':1, 'name':'a', 'time': d2006}, {'id':2, 'name':'a', 'time': d2008},
                      {'id':3, 'name':'a', 'time': None}])
        mxx = _get_max(self.session, Customer.id)
        d = pd.DataFrame([{'name': 'a', 'time': None}, {'name': 'b', 'time': d2006},
                          {'name': 'b', 'time': d2006}, { 'name': 'a', 'time': None},
                          {'name': 'a', 'time': d2006}])
# d is:
#           name       time   id
#         0    a        NaT  3.0
#         1    b 2006-01-01  NaN
#         2    b 2006-01-01  NaN
#         3    a        NaT  3.0
#         4    a 2006-01-01  1.0
        d2, new = syncdf(d, self.session, [Customer.name, Customer.time], Customer.id)
# d2 should be:
#           name       time   id
#         0    a        NaT  3.0
#         1    b 2006-01-01  4.0
#         4    a 2006-01-01  1.0
        assert array_equal(d2['id'], [3, mxx+1, 1])

    def test_syncdf_2_no_drop_dupes(self):
        d2006 = datetime(2006, 1, 1)
        d2007 = datetime(2007, 12, 31)
        d2008 = datetime(2008, 3, 14)
        d2009 = datetime(2009, 9, 25)
        mxx = _get_max(self.session, Customer.id)
        self.init_db([{'id':1, 'name':'a', 'time': d2006}, {'id':2, 'name':'a', 'time': d2008},
                      {'id':3, 'name':'a', 'time': None}])
        mxx = _get_max(self.session, Customer.id)
        d = pd.DataFrame([{'name': 'a', 'time': None}, {'name': 'b', 'time': d2006},
                          {'name': 'b', 'time': d2006}, { 'name': 'a', 'time': None},
                          {'name': 'a', 'time': d2006}])
# d is:
#           name       time  (id)
#         0    a        NaT  3.0
#         1    b 2006-01-01  NaN
#         2    b 2006-01-01  NaN
#         3    a        NaT  3.0
#         4    a 2006-01-01  1.0
        d2, new = syncdf(d, self.session, [Customer.name, Customer.time], Customer.id,
                                  drop_duplicates=False)
# d2 should be:
#           name       time   id
#         0    a        NaT  3.0
#         1    b 2006-01-01  4.0
#         2    b 2006-01-01  5.0
#         3    a        NaT  3.0
#         4    a 2006-01-01  1.0
        assert array_equal(d2['id'], [3, mxx+1, mxx+2, 3, 1])
        
    def test_syncdf_3(self):
        d2006 = datetime(2006, 1, 1)
        d2007 = datetime(2007, 12, 31)
        d2008 = datetime(2008, 3, 14)
        d2009 = datetime(2009, 9, 25)
        self.init_db([{'id':1, 'name':'a', 'time': d2006}, {'id':2, 'name':'a', 'time': d2008},
                      {'id':3, 'name':'a', 'time': None}])
        d = pd.DataFrame([{'id':45, 'name': 'a', 'time': None}, {'id':45, 'name': 'b', 'time': d2006},
                          {'id':45, 'name': 'c', 'time': d2009}, {'id':45, 'name': 'a', 'time': None},
                          {'id':45, 'name': 'a', 'time': d2006}])
        d2, new = syncdf(d, self.session, [Customer.name, Customer.time], Customer.id)
        # Note that in this case {'id':45, 'name': 'b', 'time': d2006}, and
        # {'id':45, 'name': 'c', 'time': d2009} are still valid and in d2 cause they don't have
        # matching on the table => their id is not updated and they don't have null id =>
        # they id is not set
        expected_ids = [3, 45, 45, 1]
        assert array_equal(d2['id'], expected_ids)
    

    def test_mergeupdate1(self):
        d2006 = datetime(2006, 1, 1)
        d2007 = datetime(2007, 12, 31)
        d2008 = datetime(2008, 3, 14)
        d2009 = datetime(2009, 9, 25)
        d = pd.DataFrame([{'id':45, 'name': 'a', 'time': None}, {'id':45, 'name': 'b', 'time': d2006},
                          {'id':45, 'name': 'c', 'time': d2009}, {'id':45, 'name': 'd', 'time': d2008},
                          {'id':45, 'name': 'e', 'time': d2006}])
        
        dnew = pd.DataFrame([{'id':45, 'name': 'a', 'time': None, 'a':5}, {'id':45, 'name': 'b', 'time': d2006, 'a':5},
                          {'id':45, 'name': 'c', 'time': d2009, 'a':5}, {'id':45, 'name': 'd', 'time': None, 'a':5},
                          {'id':45, 'name': 'e', 'time': d2006, 'a':5}])
        
        d2 = mergeupdate(d, dnew, ['id', 'name'], ['time'])
        
        assert len(d2) == len(d)
        # assert new column of dnew is not added to d2:
        assert 'a' not in d2.columns
        # assert time columns of d2 are time columns of dnew
        assert array_equal(d2['time'].dropna(), dnew['time'].dropna()) and len(d2) == len(dnew)
        # for times, as numpy is weird about that and mergeupdate sets the values of a column (np array)
        # check also that types are the same
        assert d2['time'].dtype == dnew['time'].dtype
        g = 9
        
    def test_mergeupdate2(self):
        d2006 = datetime(2006, 1, 1)
        d2007 = datetime(2007, 12, 31)
        d2008 = datetime(2008, 3, 14)
        d2009 = datetime(2009, 9, 25)
        d = pd.DataFrame([{'id':45, 'name': 'a', 'time': None}, {'id':45, 'name': 'b', 'time': d2006},
                          {'id':45, 'name': 'c', 'time': d2009}, {'id':45, 'name': 'd', 'time': d2008},
                          {'id':45, 'name': 'e', 'time': d2006}])
        
        # dnew has a dupe (last two elements) we will remove them in mergeupdate
        dnew = pd.DataFrame([{'id':45, 'name': 'a', 'time': None, 'a':5}, {'id':45, 'name': 'b', 'time': d2006, 'a':5},
                          {'id':45, 'name': 'c', 'time': d2009, 'a':5}, {'id':45, 'name': 'd', 'time': None, 'a':5},
                          {'id':45, 'name': 'e', 'time': d2006, 'a':5}, {'id':45, 'name': 'e', 'time': d2006, 'a':5}])
        
        d2 = mergeupdate(d, dnew, ['id', 'name'], ['time'])
        
        assert len(d2) == len(d)
        
        # assert new column of dnew is not added to d2:
        assert 'a' not in d2.columns
        # for times, as numpy is weird about that and mergeupdate sets the values of a column (np array)
        # check also that types are the same
        assert d2['time'].dtype == dnew['time'].dtype
        g = 9
        
    def test_mergeupdate3(self):
        d2006 = datetime(2006, 1, 1)
        d2007 = datetime(2007, 12, 31)
        d2008 = datetime(2008, 3, 14)
        d2009 = datetime(2009, 9, 25)
        d = pd.DataFrame([{'id':45, 'name': 'a', 'time': None}, {'id':45, 'name': 'b', 'time': d2006},
                          {'id':45, 'name': 'c', 'time': d2009}, {'id':45, 'name': 'd', 'time': d2008},
                          {'id':45, 'name': 'e', 'time': d2006}])
        
        # dnew has a dupe (last two elements) we will remove them in mergeupdate
        dnew = pd.DataFrame([{'id':45, 'name': 'a', 'time': None, 'a':5}, {'id':45, 'name': 'b', 'time': d2006, 'a':5},
                          {'id':45, 'name': 'c', 'time': d2009, 'a':5}, {'id':45, 'name': 'd', 'time': None, 'a':5},
                          {'id':45, 'name': 'e', 'time': d2006, 'a':5}, {'id':45, 'name': 'e', 'time': d2007, 'a':5}])
        
        d2 = mergeupdate(d, dnew, ['id', 'name'], ['time'])
        
        assert len(d2) == len(d)
        
        # assert last element of dnew is not in d2, as we have removed the duplicate:
        assert (~(d2['time']==d2007)).all()
        # assert new column of dnew is not added to d2:
        assert 'a' not in d2.columns
        # for times, as numpy is weird about that and mergeupdate sets the values of a column (np array)
        # check also that types are the same
        assert d2['time'].dtype == dnew['time'].dtype
        g = 9
        
    def test_mergeupdate4(self):
        d2006 = datetime(2006, 1, 1)
        d2007 = datetime(2007, 12, 31)
        d2008 = datetime(2008, 3, 14)
        d2009 = datetime(2009, 9, 25)
        d = pd.DataFrame([{'id':45, 'name': 'a', 'time': None}, {'id':45, 'name': 'b', 'time': d2006},
                          {'id':45, 'name': 'c', 'time': d2009}, {'id':45, 'name': 'd', 'time': d2008},
                          {'id':45, 'name': 'e', 'time': d2006}])
        
        # dnew has a dupe (last two elements) we will remove them in mergeupdate
        dnew = pd.DataFrame([{'id':45, 'name': 'a', 'time': None, 'a':5}, {'id':45, 'name': 'b', 'time': d2006, 'a':5},
                          {'id':45, 'name': 'c', 'time': d2009, 'a':5}, {'id':45, 'name': 'd', 'time': None, 'a':5},
                          {'id':45, 'name': 'e', 'time': d2006, 'a':5}, {'id':45, 'name': 'x', 'time': d2007, 'a':5}])
        
        d2 = mergeupdate(d, dnew, ['id', 'name'], ['time'])
        
        assert len(d2) == len(d)
        
        # assert last element of dnew is not in d2, as it has no matches ['id', 'name']
        assert (~(d2['time']==d2007)).all()
        # assert new column of dnew is not added to d2:
        assert 'a' not in d2.columns
        # for times, as numpy is weird about that and mergeupdate sets the values of a column (np array)
        # check also that types are the same
        assert d2['time'].dtype == dnew['time'].dtype
        g = 9


    def test_updatedf_(self):
        d2006 = datetime(2006, 1, 1)
        d2007 = datetime(2007, 12, 31)
        d2008 = datetime(2008, 3, 14)
        d2009 = datetime(2009, 9, 25)
        self.init_db([{'id':1, 'name':'a'}, {'id':2, 'name':'b'}, {'id':3, 'name': "c"}])
        d = pd.DataFrame([{'id':1, 'name': 'x'}, {'id':2, 'name': 'x'}, {'id':3, 'name': 'x'}])
        d2 = updatedf(d, self.session, Customer.id, [Customer.name])
        assert array_equal(d2['id'], [1,2,3])
        assert ([x[0] for x in self.session.query(Customer.name).all()] == d2['name']).all()
        assert len(d2) == 3

    def test_updatedf_nullcontraint_violation(self):
        d2006 = datetime(2006, 1, 1)
        d2007 = datetime(2007, 12, 31)
        d2008 = datetime(2008, 3, 14)
        d2009 = datetime(2009, 9, 25)
        self.init_db([{'id':1, 'name':'a'}, {'id':2, 'name':'b'}, {'id':3, 'name': "c"}])
        # first item violates a non-null constraint
        d = pd.DataFrame([{'id':1, 'name': None}, {'id':2, 'name': 'x'}, {'id':3, 'name': 'x'}])
        d2 = updatedf(d, self.session, Customer.id, [Customer.name])
        assert d2.empty
        assert not self.session.query(Customer.name).filter(Customer.id.in_(d2['id'].values)).all()
       

    def test_updatedf_nullcontraint_violation2(self):
        d2006 = datetime(2006, 1, 1)
        d2007 = datetime(2007, 12, 31)
        d2008 = datetime(2008, 3, 14)
        d2009 = datetime(2009, 9, 25)
        self.init_db([{'id':1, 'name':'a'}, {'id':2, 'name':'b'}, {'id':3, 'name': "c"}])
        # third item violates a non-null constraint
        d = pd.DataFrame([{'id':1, 'name': 'x'}, {'id':2, 'name': 'x'}, {'id':3, 'name': None}])
        d2 = updatedf(d, self.session, Customer.id, [Customer.name])
        assert array_equal(d2['id'], [1,2])
        assert sorted([x[0] for x in self.session.query(Customer.name).all()]) == ['c', 'x', 'x']
#         assert lst == expected_lst
#         assert [x[0] for x in self.session.query(Customer.name).all()] == ['x', 'x', 'c']


def array_equal(a1, a2):
    """test array equality by assuming nan == nan. Probably already implemented somewhere in numpy, no time for browsing now"""
    return len(a1) == len(a2) and all([c ==d or (np.isnan(c) == np.isnan(d)) for c, d in izip(a1, a2)])

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()