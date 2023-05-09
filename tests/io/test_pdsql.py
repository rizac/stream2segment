"""
Created on Apr 11, 2017

@author: riccardo
"""
import math
from datetime import datetime

from unittest.mock import patch
import numpy as np
import pytest
import pandas as pd


from sqlalchemy import Column, Integer, String, LargeBinary, DateTime
from sqlalchemy.exc import SQLAlchemyError

# import declarative_base from io.db.models to be sqlalchemy 1.x vs 2.x compliant:
from stream2segment.io.db import declarative_base
from stream2segment.io.db.pdsql import (syncdfcol, insertdf, _get_max, syncdf, syncdfseq,
                                        mergeupdate, updatedf, dbquery2df, DbManager)

Base = declarative_base()


class Customer(Base):
    __tablename__ = "customer"
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, default='a')
    data = Column(LargeBinary)
    time = Column(DateTime)

class CustomerAC(Base):  # this is used in test_insertdf_nopkeys
    __tablename__ = "customer_with_explicit_autoincrement"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, default='a')
    data = Column(LargeBinary)
    time = Column(DateTime)


class Test:

    # execute this fixture always even if not provided as argument:
    # https://docs.pytest.org/en/documentation-restructure/how-to/fixture.html#autouse-fixtures-xunit-setup-on-steroids
    @pytest.fixture(autouse=True)
    def init(self, request, db, data):
        # re-init a sqlite database (no-op if the db is not sqlite):
        db.create(to_file=False, custom_base=Base)

    def init_db(self, session, list_of_dicts):
        # use insertdf which takes care of type casting:
        dataframe = pd.DataFrame(list_of_dicts)
        syncdfseq(dataframe, session, Customer.id, overwrite=False, pkeycol_maxval=None)
        insertdf(dataframe, session, Customer, buf_size=len(list_of_dicts),
                 return_df=False)

#         for d in list_of_dicts:
#             customer = Customer(**d)
#             session.add(customer)
#         session.commit()


    @pytest.mark.parametrize('model', [Customer, CustomerAC])
    def test_insertdf_nopkeys(self, model,
                              # fixtures:
                              db):
        NAME = 'CU23oo'
        new, df = insertdf(pd.DataFrame([{'id': 1, 'name': NAME}]),
                           db.session, model, buf_size=1,
                           return_df=True)
        # re-insert
        new, df = insertdf(pd.DataFrame([{'id': 1, 'name': NAME}]),
                           db.session, model, buf_size=1,
                           return_df=True)
        assert len(df) == 0

        # without id, it inserts the row if the primary key is autoincrement
        # otherwise not ...
        new, df = insertdf(pd.DataFrame([{'name': NAME}]),
                           db.session, model, buf_size=1,
                           return_df=True)
 
        if db.is_sqlite:  # sqlite seems to handle nextval in sequences
            # with core inserts (no ORM):
            assert len(df) == 1
            assert sorted([_[0] for _ in
                           db.session.query(model.id).
                           filter(model.name == NAME)]) == [1, 2]
        else:
            # postgres seems to fail, even when we set autoincrement=True explicitly,
            # probably because the latter is used with the ORM, not for core inserts:
            assert df.empty
            assert sorted([_[0] for _ in
                           db.session.query(model.id).
                           filter(model.name == NAME)]) == [1]

        asd = 9


    def test_syncdfseq(self, db):
        self.init_db(db.session, [{'id': 1, 'name': 'a'}, {'id': 2, 'name': 'b'},
                                  {'id': 3, 'name': "c"}])
        d = pd.DataFrame([{'name': 'a', 'id': 1}, {'name': 'b', 'id': 3}, {'name': 'a'},
                          {'name': 'x'}])
        assert d['id'].dtype not in (np.int64, np.int32, np.int16, np.int8)
        max_id = 3  # the max id
        for ov in [True, False]:
            for max_ in [None, 113]:
                maxid = max_id if max_ is None else max_
                if ov:
                    expected_ids = np.arange(maxid+1, maxid + 1 + len(d), dtype=int).tolist()
                else:
                    numnan = pd.isnull(d['id']).sum()
                    alreadyset_ids = d[~pd.isnull(d['id'])]['id'].values  # pylint: disable=invalid-unary-operand-type
                    new_ids = (1 + maxid + np.arange(numnan))
                    expected_ids = new_ids.tolist() + alreadyset_ids.tolist()

                df = syncdfseq(d, db.session, Customer.id, overwrite=ov,
                               pkeycol_maxval=max_)
                ids = df['id'].values.tolist()
                # assert stuff:
                assert sorted(ids) == sorted(expected_ids)
                assert df['id'].dtype in (np.int64, np.int32, np.int16, np.int8)

        # let's see two cases when id is provided:
        for ov in [True, False]:
            d = pd.DataFrame([{'name': 'a', 'id': 11}, {'name': 'b', 'id': 1},
                              {'name': 'a', 'id': 34}, {'name': 'x', 'id': 4}])
            max_ = 200
            expected_ids = np.array(d['id'].values, copy=True) if not ov else \
                np.arange(max_ + 1, max_ + 1 + len(d), dtype=int)
            df = syncdfseq(d, db.session, Customer.id, overwrite=ov,
                           pkeycol_maxval=max_)
            assert (df['id'] == expected_ids).all()

        for ov in [True, False]:
            d = pd.DataFrame([{'name': 'a', 'id': np.nan}, {'name': 'b', 'id': 1},
                              {'name': 'a', 'id': 34}, {'name': 'x', 'id': 4}])
            max_ = 200
            expected_ids = np.array([201, 1, 34, 4]) if not ov else \
                np.arange(max_ + 1, max_ + 1 + len(d), dtype=int)
            df = syncdfseq(d, db.session, Customer.id, overwrite=ov,
                           pkeycol_maxval=max_)
            assert (df['id'] == expected_ids).all()

        for ov in [True, False]:
            d = pd.DataFrame([{'name': 'a', 'id': np.nan}, {'name': 'b', 'id': np.nan},
                              {'name': 'a', 'id': np.nan}, {'name': 'x', 'id': np.nan}])
            max_ = 200
            expected_ids = np.arange(max_ + 1, max_ + 1 + len(d), dtype=int)
            df = syncdfseq(d, db.session, Customer.id, overwrite=ov,
                           pkeycol_maxval=max_)
            assert (df['id'] == expected_ids).all()


    def test_syncdfcol_(self, db):
        d2006 = datetime(2006, 1, 1)
        d2007 = datetime(2007, 12, 31)
        d2008 = datetime(2008, 3, 14)
        d2009 = datetime(2009, 9, 25)
        self.init_db(db.session, [{'id': 1, 'name': 'a'}, {'id': 2, 'name': 'b'},
                                  {'id': 3, 'name': "c"}])
        d = pd.DataFrame([{'name': 'a'}, {'name': 'b'}, {'name': 'd'}])
        d2 = syncdfcol(d, db.session, [Customer.name], Customer.id)
        expected_ids = [1, 1, np.nan]
        assert array_equal(d2['id'], expected_ids)
        # note: dtype has changed to accomodate nans:
        assert d2['id'].dtype == np.float64

    def test_syncdfcol_2(self, db):
        d2006 = datetime(2006, 1, 1)
        d2007 = datetime(2007, 12, 31)
        d2008 = datetime(2008, 3, 14)
        d2009 = datetime(2009, 9, 25)
        self.init_db(db.session, [{'id': 1, 'name': 'a'}, {'id': 2, 'name': 'b'}])
        d = pd.DataFrame([{'name': 'a'}, {'name': 'b'}, {'name': 'd'}])
        d2 = syncdfcol(d, db.session, [Customer.name], Customer.id)
        expected_ids = [1, 2, np.nan]
        assert array_equal(d2['id'], expected_ids)
        # note: dtype has changed to accomodate nans:
        assert d2['id'].dtype == np.float64

    def test_syncdfcol_3(self, db):
        d2006 = datetime(2006, 1, 1)
        d2007 = datetime(2007, 12, 31)
        d2008 = datetime(2008, 3, 14)
        d2009 = datetime(2009, 9, 25)
        self.init_db(db.session, [{'id': 1, 'name': 'a', 'time': d2006},
                                  {'id': 2, 'name': 'a', 'time': d2008},
                                  {'id': 3, 'name': 'a', 'time': None}])
        d = pd.DataFrame([{'name': 'a', 'time': None}, {'name': 'b', 'time': d2006},
                          {'name': 'c', 'time': d2009}, {'name': 'a', 'time': None},
                          {'name': 'a', 'time': d2006}])
        d2 = syncdfcol(d, db.session, [Customer.name, Customer.time], Customer.id)
        expected_ids = [3, np.nan, np.nan, 3, 1]
        assert array_equal(d2['id'], expected_ids)
        # note: dtype has changed to accomodate nans:
        assert d2['id'].dtype == np.float64

    def test_syncdfcol_4(self, db):
        d2006 = datetime(2006, 1, 1)
        d2007 = datetime(2007, 12, 31)
        d2008 = datetime(2008, 3, 14)
        d2009 = datetime(2009, 9, 25)
        self.init_db(db.session, [{'id': 1, 'name': 'a', 'time': d2006},
                                  {'id': 2, 'name': 'a', 'time': d2008},
                                  {'id': 3, 'name': 'a', 'time': None}])
        d = pd.DataFrame([{'id': 45, 'name': 'a', 'time': None},
                          {'id': 45, 'name': 'b', 'time': d2006},
                          {'id': 45, 'name': 'c', 'time': d2009},
                          {'id': 45, 'name': 'a', 'time': None},
                          {'id': 45, 'name': 'a', 'time': d2006}])
        d2 = syncdfcol(d, db.session, [Customer.name, Customer.time], Customer.id)
        expected_ids = [3, 45, 45, 3, 1]
        assert array_equal(d2['id'], expected_ids)
        # note: dtype has changed even if ids are not nans:
        assert d2['id'].dtype == np.float64

    def test_syncdfcol_dtypes(self, db):
        '''tests when fetchsetpkeys changes dtype to float for an id of type INTEGER
        Basically: when any row instance has no match on the db (at least one row)
        '''
        d2006 = datetime(2006, 1, 1)
        d2007 = datetime(2007, 12, 31)
        d2008 = datetime(2008, 3, 14)
        d2009 = datetime(2009, 9, 25)
        self.init_db(db.session, [{'id': 1, 'name': 'a', 'time': d2006},
                                  {'id': 2, 'name': 'a', 'time': d2008},
                                  {'id': 3, 'name': 'a', 'time': None}])

        # 3 CASES WHEN d['id'] IS GIVEN:
        # ------------------------------
        # d has ALL instance mapped to db rows:
        d = pd.DataFrame([{'id': 1, 'name': 'a', 'time': None}])
        d2 = syncdfcol(d, db.session, [Customer.name, Customer.time], Customer.id)
        assert d2['id'].dtype == np.int64
        # d has NO instances mapped to db rows:
        d = pd.DataFrame([{'id': 1, 'name': 'axz', 'time': None}])
        d2 = syncdfcol(d, db.session, [Customer.name, Customer.time], Customer.id)
        assert d2['id'].dtype == np.float64
        # d has SOME instance mapped to db ros, SOME not:
        d = pd.DataFrame([{'id': 1, 'name': 'a', 'time': None},
                          {'id': 1, 'name': 'axz', 'time': None}])
        d2 = syncdfcol(d, db.session, [Customer.name, Customer.time], Customer.id)
        assert d2['id'].dtype == np.float64

        # 3 CASES WHEN d['id'] IS NOT GIVEN:
        # ------------------------------
        # d has ALL instance mapped to db rows:
        d = pd.DataFrame([{'name': 'a', 'time': None}])
        d2 = syncdfcol(d, db.session, [Customer.name, Customer.time], Customer.id)
        assert d2['id'].dtype == np.int64
        # d has NO instances mapped to db rows:
        d = pd.DataFrame([{'name': 'axz', 'time': None}])
        d2 = syncdfcol(d, db.session, [Customer.name, Customer.time], Customer.id)
        assert d2['id'].dtype == np.float64
        # d has SOME instance mapped to db ros, SOME not:
        d = pd.DataFrame([{'name': 'a', 'time': None},
                          {'name': 'axz', 'time': None}])
        d2 = syncdfcol(d, db.session, [Customer.name, Customer.time], Customer.id)
        assert d2['id'].dtype == np.float64

    @patch('stream2segment.io.db.pdsql.insertdf', side_effect=insertdf)
    @patch('stream2segment.io.db.pdsql.updatedf', side_effect=updatedf)
    def test_dbmanager_callcounts(self, mock_updatedf, mock_insertdf, db):
        d2006 = datetime(2006, 1, 1)
        d2007 = datetime(2007, 12, 31)
        d2008 = datetime(2008, 3, 14)
        d2009 = datetime(2009, 9, 25)
        self.init_db(db.session,
                     [{'id': 1, 'name': 'a', 'time': d2006}, {'id': 2, 'name': 'a', 'time': d2008},
                      {'id': 3, 'name': 'a', 'time': None}])

        d = pd.DataFrame([{'name': 'a', 'time': None}, {'name': 'b', 'time': d2006},
                          {'name': 'c', 'time': d2009}, {'name': 'a', 'time': None},
                          {'name': 'a', 'time': d2006}])
        dbm = DbManager(db.session, Customer.id, update=False, buf_size=10)
        d[Customer.id.key] = np.nan  # must be set when using dbmanager
        # contrarily to syncdf, all instances will be added
        # we do not have constraints thus a new id will be set for all of them
        for i in range(len(d)):
            dbm.add(d.iloc[i:i+1,:])
        _, inserted, not_inserted, updated, not_updated = dbm.close()
        assert (inserted, not_inserted, updated, not_updated) == (len(d), 0, 0, 0)
        assert mock_insertdf.call_count == 1  # != inserted
        assert mock_updatedf.call_count == 0

        mock_insertdf.reset_mock()
        mock_updatedf.reset_mock()
        d = pd.DataFrame([{'name': 'a', 'time': None}, {'name': 'b', 'time': d2006},
                          {'name': 'c', 'time': d2009}, {'name': 'a', 'time': None},
                          {'name': 'a', 'time': d2006}])
        dbm = DbManager(db.session, Customer.id, update=False, buf_size=1)
        d[Customer.id.key] = np.nan  # must be set when using dbmanager
        # contrarily to syncdf, all instances will be added
        for i in range(len(d)):
            dbm.add(d.iloc[i: i+1, :])
        _, inserted, not_inserted, updated, not_updated = dbm.close()
        assert (inserted, not_inserted, updated, not_updated) == (len(d), 0, 0, 0)
        assert mock_insertdf.call_count == len(d)  # == inserted (buf_size=1)
        assert mock_updatedf.call_count == 0

        # now update:
        mock_insertdf.reset_mock()
        mock_updatedf.reset_mock()
        d = dbquery2df(db.session.query(Customer.id, Customer.name, Customer.time))
        # {'name': 'a', 'time': None} will be dropped (drop duplicates is True by default)
        # from the other three elements, the other three will be updated
        dbm = DbManager(db.session, Customer.id, update=True, buf_size=10)
        for i in range(len(d)):
            dbm.add(d.iloc[i:i+1,:])
        _, inserted, not_inserted, updated, not_updated = dbm.close()
        assert (inserted, not_inserted, updated, not_updated) == (0, 0, len(d), 0)
        assert mock_insertdf.call_count == 0
        assert mock_updatedf.call_count == int(math.ceil(len(d) / 10.0))  # != inserted

        # now update with buf_size=1
        mock_insertdf.reset_mock()
        mock_updatedf.reset_mock()
        d = dbquery2df(db.session.query(Customer.id, Customer.name, Customer.time))
        # {'name': 'a', 'time': None} will be dropped (drop duplicates is True by default)
        # from the other three elements, the other three will be updated
        dbm = DbManager(db.session, Customer.id, update=True, buf_size=1)
        for i in range(len(d)):
            dbm.add(d.iloc[i:i+1,:])
        _, inserted, not_inserted, updated, not_updated = dbm.close()
        assert (inserted, not_inserted, updated, not_updated) == (0, 0, len(d), 0)
        assert mock_insertdf.call_count == 0
        assert mock_updatedf.call_count == len(d)  # == inserted (buf_size=1)

    @patch('stream2segment.io.db.pdsql.insertdf', side_effect=insertdf)
    @patch('stream2segment.io.db.pdsql.updatedf', side_effect=updatedf)
    def test_syncdf_several_combinations_constraints(self, mock_updatedf, mock_insertdf, db):
        dt = datetime(2006, 1, 1, 12, 31, 7, 456789)
        EXISTING_ID = 1
        db_df = [{'id': EXISTING_ID, 'name': 'a', 'time': None}]
        dfs = [
                # non existing, violates constraint (name NULL):
                pd.DataFrame([{'name': None, 'time': None}]),
                # existing, violates constraint (name NULL):
                pd.DataFrame([{'id': 1, 'name': None, 'time': None}]),
                # non existing, violates constraint (name NULL) and non existing (OK)
                pd.DataFrame([{'name': None, 'time': None}, {'name': 'x', 'time': None}]),
                # existing, violates constraint (name NULL)  and non existing (OK)
                pd.DataFrame([{'id': 1, 'name': None, 'time': None}, {'name': 'x', 'time': None}]),
                # non existing, violates constraint (name NULL) and existing (OK)
                pd.DataFrame([{'name': None, 'time': None}, {'name': 'a', 'time': None}]),
                # existing, violates constraint (name NULL)  and existing (OK)
                pd.DataFrame([{'id': 1, 'name': None, 'time': None}, {'name': 'a', 'time': None}]),
            ]

        # stupid hack to access these vars from within functions below
        # Python 3 has better ways, we still need python2 compatibility:
        inserterr_callcount = [0]
        updateerr_callcount = [0]

        def onerri(dataframe, exc):
            assert isinstance(dataframe, pd.DataFrame)
            assert isinstance(exc, SQLAlchemyError)
            inserterr_callcount[0] += 1

        def onerru(dataframe, exc):
            assert isinstance(dataframe, pd.DataFrame)
            assert isinstance(exc, SQLAlchemyError)
            updateerr_callcount[0] += 1

        for update in [True]:
            for buf_size in [1]:
                for i, d in enumerate(dfs):
                    # re-initialize db every time:
                    inserterr_callcount[0] = 0
                    updateerr_callcount[0] = 0
                    db.session.query(Customer).delete()
                    self.init_db(db.session, db_df)
                    assert len(db.session.query(Customer.id).all()) == len(db_df)

                    inserted, not_inserted, updated, not_updated, d2 = \
                        syncdf(d.copy(), db.session, [Customer.name, Customer.time], Customer.id,
                               buf_size=buf_size, update=update, oninsert_err_callback=onerri,
                               onupdate_err_callback=onerru)

                    if not_inserted:
                        assert inserterr_callcount[0] > 0
                    else:
                        assert inserterr_callcount[0] == 0
                    if not_updated:
                        assert updateerr_callcount[0] > 0
                    else:
                        assert updateerr_callcount[0] == 0

                    # SYNC 1 ROW WHICH DOES NOT EXIST AND VIOLATES CONSTRAINTS
                    if i == 0:
                        assert (inserted, not_inserted, updated, not_updated) == (0, 1, 0, 0)
                        assert len(d2) == 0

                    # SYNC 1 ROW WHICH EXISTS AND VIOLATES CONSTRAINTS
                    elif i == 1:
                        if update:
                            assert (inserted, not_inserted, updated, not_updated) == (0, 0, 0, 1)
                            assert len(d2) == 0
                        else:
                            assert (inserted, not_inserted, updated, not_updated) == (0, 0, 0, 0)
                            assert len(d2) == 1 and d2['id'].iloc[0] == EXISTING_ID

                    # SYNC 2 ROWS: ONE DOES NOT EXIST (AND VIOLATES COSNTRAINTS),
                    # THE OTHER DOES NOT EXIST (AND IS OK)
                    elif i == 2:
                        if buf_size != 1:
                            # bad instances fall in the same chunk and thus are not inserted
                            assert (inserted, not_inserted, updated, not_updated) == (0, 2, 0, 0)
                            assert len(d2) == 0
                        else:
                            # bad instances fall NOT in the same chunk and thus are not inserted
                            assert (inserted, not_inserted, updated, not_updated) == (1, 1, 0, 0)
                            # the only instance has not EXISTING_ID+1 (that was discarded) but
                            # EXISTING_ID+2:
                            assert len(d2) == 1 and d2['id'].iloc[0] == EXISTING_ID+2

                    # SYNC 2 ROWS: ONE EXISTS (AND VIOLATES COSNTRAINTS),
                    # THE OTHER DOES NOT EXIST (AND IS OK)
                    elif i == 3:
                        if update:
                            assert (inserted, not_inserted, updated, not_updated) == (1, 0, 0, 1)
                            assert len(d2) == 1 and d2['id'].iloc[0] == EXISTING_ID + 1
                        else:
                            assert (inserted, not_inserted, updated, not_updated) == (1, 0, 0, 0)
                            assert len(d2) == 2 and sorted(d2['id'].values.tolist()) == \
                                [EXISTING_ID, EXISTING_ID+1]

                    # SYNC 2 ROWS: ONE DOES NOT EXIST (AND VIOLATES COSNTRAINTS),
                    # THE OTHER EXISTS (AND IS OK)
                    elif i == 4:
                        if update:
                            assert (inserted, not_inserted, updated, not_updated) == (0, 1, 1, 0)
                        else:
                            assert (inserted, not_inserted, updated, not_updated) == (0, 1, 0, 0)
                        assert len(d2) == 1 and d2['id'].iloc[0] == EXISTING_ID

                    # SYNC 2 ROWS: ONE EXISTS (AND VIOLATES COSNTRAINTS),
                    # THE OTHER EXISTS (AND IS OK)
                    elif i == 5:
                        if update:
                            if buf_size != 1:
                                # both not updated (same update chunk, which fails):
                                assert (inserted, not_inserted, updated, not_updated) == \
                                    (0, 0, 0, 2)
                                assert len(d2) == 0
                            else:
                                # the first updated, the second not
                                assert (inserted, not_inserted, updated, not_updated) == \
                                    (0, 0, 1, 1)
                                assert len(d2) == 1 and d2['id'].iloc[0] == EXISTING_ID
                        else:
                            assert (inserted, not_inserted, updated, not_updated) == \
                                (0, 0, 0, 0)
                            assert len(d2) == 2 and sorted(d2['id'].values.tolist()) == \
                                [EXISTING_ID, EXISTING_ID]


    @patch('stream2segment.io.db.pdsql.insertdf', side_effect=insertdf)
    @patch('stream2segment.io.db.pdsql.updatedf', side_effect=updatedf)
    def test_syncdf_several_combinations(self, mock_updatedf, mock_insertdf, db):
        dt = datetime(2006, 1, 1, 12, 31, 7, 456789)
        EXISTING_ID = 1
        db_df = [{'id': EXISTING_ID, 'name': 'a', 'time': None}]
        dfs = [
                # all rows existing:
                pd.DataFrame([{'name': 'a', 'time': None}]),
                # no row existing:
                pd.DataFrame([{'name': 'b', 'time': None}]),
                # some exist, some not
                pd.DataFrame([{'name': 'a', 'time': None}, {'name': 'b', 'time': dt}]),
                # some exist (inserted twice), some not:
                pd.DataFrame([{'name': 'a', 'time': None}, {'name': 'a', 'time': None},
                              {'name': 'b', 'time': dt}]),
                # some exust, some note (inserted twice):
                pd.DataFrame([{'name': 'a', 'time': None}, {'name': 'b', 'time': dt},
                              {'name': 'b', 'time': dt}]),
            ]

        for keep_dup in [False, True]:
            for update in [True, False]:
                for i, d in enumerate(dfs):
                    # re-initialize db every time:
                    db.session.query(Customer).delete()
                    self.init_db(db.session, db_df)
                    assert len(db.session.query(Customer.id).all()) == len(db_df)

                    inserted, not_inserted, updated, not_updated, d2 = \
                        syncdf(d.copy(), db.session, [Customer.name, Customer.time], Customer.id,
                               keep_duplicates=keep_dup, update=update)

                    # SYNC 1 ROW WHICH EXISTS ON THE DB
                    if i == 0:
                        if update:
                            assert (inserted, not_inserted, updated, not_updated) == (0, 0, 1, 0)
                        else:
                            assert (inserted, not_inserted, updated, not_updated) == (0, 0, 0, 0)
                        assert len(d2) == 1 and d2['id'].iloc[0] == EXISTING_ID

                    # SYNC 1 ROW WHICH DOES NOT EXIST ON THE DB
                    elif i == 1:
                        assert (inserted, not_inserted, updated, not_updated) == (1, 0, 0, 0)
                        assert len(d2) == 1 and d2['id'].iloc[0] == EXISTING_ID + 1

                    # SYNC 2 ROWS: ONE EXISTS ON THE DB THE OTHER NOT
                    elif i == 2:
                        if update:
                            assert (inserted, not_inserted, updated, not_updated) == (1, 0, 1, 0)
                        else:
                            assert (inserted, not_inserted, updated, not_updated) == (1, 0, 0, 0)
                        assert len(d2) == 2 and sorted(d2['id'].values.tolist()) == \
                            [EXISTING_ID, EXISTING_ID+1]

                    # SYNC 3 ROWS: TWO EXISTS AND ARE THE SAME (DUPLICATED),
                    # THE OTHER DOES NOT EXIST ON THE DB
                    elif i == 3:
                        if not keep_dup:  # drop duplicates (all)
                            # rows duplicates  will be dropped,
                            # it remains only the already existing row, thus this is the
                            # same as no row existing (i==1)
                            assert (inserted, not_inserted, updated, not_updated) == (1, 0, 0, 0)
                            assert len(d2) == 1 and d2['id'].iloc[0] == EXISTING_ID + 1
                        else:
                            if update:
                                # rows duplicates will NOT be dropped, They exist, hence,
                                # their id will be fetched and
                                # they will NOT be inserted, thus they will be updated.
                                # The other row will be inserted, thus will be updated
                                assert (inserted, not_inserted, updated, not_updated) == \
                                    (1, 0, 2, 0)
                            else:
                                # rows duplicates will NOT be dropped, Their id will be fetched and
                                # they will NOT be inserted. The other row will be inserted
                                assert (inserted, not_inserted, updated, not_updated) == \
                                    (1, 0, 0, 0)

                            assert len(d2) == 3 and sorted(d2['id'].values.tolist()) == \
                                [EXISTING_ID, EXISTING_ID, EXISTING_ID+1]

                    # SYNC 3 ROWS: ONE EXIST ON THE DB, TWO DO NOT EXISTS AND ARE THE SAME (DUPLICATED)
                    elif i == 4:
                        if not keep_dup:  # drop duplicates (all)
                            if update:
                                # rows duplicates will be dropped,  it remains only the already
                                # existing row, thus this is the same as all row existing (i==0)
                                assert (inserted, not_inserted, updated, not_updated) == \
                                    (0, 0, 1, 0)
                            else:
                                # rows duplicates will be dropped,
                                # it remains only the already existing row,
                                # thus this is the same as all row existing (i==0)
                                assert (inserted, not_inserted, updated, not_updated) == \
                                    (0, 0, 0, 0)
                            assert len(d2) == 1 and d2['id'].iloc[0] == EXISTING_ID
                        else:
                            if update:
                                # rows duplicates will NOT be dropped. They do not exist,
                                # hence their id will not be fetched thus it will be assigned
                                # incrementally. The other row will be updated (it exists)
                                assert (inserted, not_inserted, updated, not_updated) == \
                                    (2, 0, 1, 0)
                            else:
                                # rows duplicates will NOT be dropped, Their id will not be
                                # fetched thus it will be assigned incrementally
                                assert (inserted, not_inserted, updated, not_updated) == \
                                    (2, 0, 0, 0)
                            assert len(d2) == 3 and sorted(d2['id'].values.tolist()) == \
                                [EXISTING_ID, EXISTING_ID+1, EXISTING_ID+2]

    # syncdf TESTS BELOW MIGHT BE REDUNDANT (THEY WERE OLD TESTS ADAPTED BUT I GUESS WE TEST
    # SEVERAL TIMES EITHER NON IMPORTANT STUFF OR STUFF ALREADY TESTED ABOVE)

    @patch('stream2segment.io.db.pdsql.insertdf', side_effect=insertdf)
    @patch('stream2segment.io.db.pdsql.updatedf', side_effect=updatedf)
    def test_syncdf(self, mock_updatedf, mock_insertdf, db):
        d2006 = datetime(2006, 1, 1)
        d2007 = datetime(2007, 12, 31)
        d2008 = datetime(2008, 3, 14)
        d2009 = datetime(2009, 9, 25)
        self.init_db(db.session,
                     [{'id': 1, 'name': 'a', 'time': d2006}, {'id': 2, 'name': 'a', 'time': d2008},
                      {'id': 3, 'name': 'a', 'time': None}])
        mxx = _get_max(db.session, Customer.id)

        d = pd.DataFrame([{'name': 'a', 'time': None}, {'name': 'b', 'time': d2006},
                          {'name': 'c', 'time': d2009}, {'name': 'a', 'time': None},
                          {'name': 'a', 'time': d2006}])
        # {'name': 'a', 'time': None} will be dropped (drop duplicates is True by default)
        # from the other three elements, the last is already existing and will not be added
        # the other two will have an autoincremented id:
        expected_ids = [1, mxx+1, mxx+2]
        inserted, not_inserted, updated, not_updated, d2 = \
            syncdf(d, db.session, [Customer.name, Customer.time], Customer.id)

        assert array_equal(d2['id'], expected_ids)
        assert inserted == len(d2)-1  # assert one is already existing
        assert updated == not_updated == 0
        assert mock_insertdf.call_count == 1

        # not try with update=True. Pass the dataframe d and change the name to 'x' for all rows
        d.loc[:, ['name']] = 'x'
        # Drop duplicates is still true, so duplicates under [Customer.name, Customer.time]
        # will be dropped: This means only {'name': 'x', 'time': d2009}
        # (previously {'name': 'c', 'time': d2009}) will be inserted
        inserted, not_inserted, updated, not_updated, d2 = \
            syncdf(d, db.session, [Customer.name, Customer.time], Customer.id, update=['name'])
        assert array_equal(d2['id'], [6])
        assert inserted == 1  # assert one is already existing
        assert updated == not_updated == 0

        # same as above, but fetch data from the db:
        d = dbquery2df(db.session.query(Customer.id, Customer.name, Customer.time))
        # assert we do not have instances with 'x':
        dbase = dbquery2df(db.session.query(Customer.id,
                                            Customer.name,
                                            Customer.time).filter(Customer.name == 'x'))
        d.loc[:, ['name']] = 'x'
        inserted, not_inserted, updated, not_updated, d2 = \
            syncdf(d, db.session, [Customer.name, Customer.time], Customer.id, update=['name'])
        # assert we returned all excpected instances
        assert d.drop_duplicates(subset=['name', 'time'], keep=False).equals(d2)
        # assert returned values
        assert (inserted, not_inserted, updated, not_updated) == (0, 0, 2, 0)
        # assert on the db we have the previously marked instances with 'x', PLUS the
        # added now
        dbase2 = dbquery2df(db.session.query(Customer.id,
                                             Customer.name,
                                             Customer.time).filter(Customer.name == 'x'))
        # and now assert it:
        oldids_with_x_as_name = dbase['id'].values.tolist()
        newids_with_x_as_name = d2['id'].values.tolist()
        currentids_with_x_as_name = dbase2['id'].values.tolist()
        assert sorted(oldids_with_x_as_name+newids_with_x_as_name) == \
            sorted(currentids_with_x_as_name)

        # same as above, but with drop_duplicates=False
        d = dbquery2df(db.session.query(Customer.id, Customer.name, Customer.time))
        # assert we do not have instances with 'x':
        dbase = dbquery2df(db.session.query(Customer.id,
                                            Customer.name,
                                            Customer.time).filter(Customer.name == 'w'))
        d.loc[:, ['name']] = 'w'
        inserted, not_inserted, updated, not_updated, d2 = \
            syncdf(d, db.session, [Customer.name, Customer.time], Customer.id, update=['name'],
                   keep_duplicates=True)
        # we should have the same result of the database, as we updated ALL instances:
        # BUT: fetchsetpkeys changed to float the dtype of id, so:
        assert not d.equals(d2)
        # this should work
        d['id'] = d['id'].astype(int)
        assert d.equals(d2)
        # assert returned values
        assert (inserted, not_inserted, updated, not_updated) == (0, 0, len(d), 0)


    @patch('stream2segment.io.db.pdsql.insertdf', side_effect=insertdf)
    @patch('stream2segment.io.db.pdsql.updatedf', side_effect=updatedf)
    def test_syncdf_dtypes_and_index(self, mock_updatedf, mock_insertdf, db):
        '''Tests that syncdf (calling DbManager) nor DbManager preserve the index'''
        d2006 = datetime(2006, 1, 1)
        d2007 = datetime(2007, 12, 31)
        d2008 = datetime(2008, 3, 14)
        d2009 = datetime(2009, 9, 25)
        df_data = [{'id': 1, 'name': 'a', 'time': d2006}, {'id': 2, 'name': 'a', 'time': d2008},
                   {'id': 3, 'name': 'a', 'time': None}]
        self.init_db(db.session, df_data)

        # Sync a dataframe with a new item N= [{'name': 'b', 'time': d2006}]
        # Index is preserved but only because the rows
        # to insert and to update are contiguous, e.g.
        # [row2beInserted1, ..., row2beInsertedN, row2beUpdated1, .., row2beUpdatedM]
        d = pd.DataFrame(df_data + [{'name': 'b', 'time': d2006}])
        inserted, not_inserted, updated, not_updated, d2 = \
            syncdf(d, db.session, [Customer.name, Customer.time], Customer.id,
                   buf_size=1)
        # indices are preserved:
        assert (d.index == d2.index).all()
        # column dtypes not, as d2 is casted to the SQL type:
        assert d2['id'].dtype == np.int64
        assert d['id'].dtype == np.float64

        # Now we insert a new item in between.
        # Index is preserved because the rows
        # to insert and to update are NOT contiguous, i.e.
        # [row2beUpdated1, row2beInserted, row2beUpdated2]
        df_data = [{'id': 1, 'name': 'a', 'time': d2006},
                   {'name': 'xyz', 'time': None},
                   {'id': 2, 'name': 'a', 'time': d2008},
                   ]
        self.init_db(db.session, df_data)

        d = pd.DataFrame(df_data)
        # {'name': 'a', 'time': None} will be dropped (drop duplicates is True by default)
        # from the other three elements, the last is already existing and will not be added
        # the other two will have an autoincremented id:
        inserted, not_inserted, updated, not_updated, d2 = \
            syncdf(d, db.session, [Customer.name, Customer.time], Customer.id,
                   buf_size=1)

        # indices are preserved:
        assert not (d.index == d2.index).all()
        # column dtypes not, as d2 is casted to the SQL type:
        assert d2['id'].dtype == np.int64
        assert d['id'].dtype == np.float64

        # Now try with DbManager (used in the underlying syncdf) and show that
        # indices are not preserved if we add several dataframes sequentially
        # (because we use a pd.concat):
        dbm = DbManager(db.session, Customer.id, update=True, buf_size=100,
                        return_df=True)
        for _ in range(len(d)):
            dbm.add(d[_:_+1])
        dbm.close()
        idx1 = d.index
        idx2 = dbm.dataframe.index
        assert len(idx1) == len(idx2)
        assert not (idx1.values == idx2.values).all()

    def test_syncdf_2(self, db):
        """Same as `test_syncdf` but the second non-existing item is conflicting with
        the first added"""
        d2006 = datetime(2006, 1, 1)
        d2007 = datetime(2007, 12, 31)
        d2008 = datetime(2008, 3, 14)
        d2009 = datetime(2009, 9, 25)
        mxx = _get_max(db.session, Customer.id)
        self.init_db(db.session, 
                     [{'id': 1, 'name': 'a', 'time': d2006}, {'id': 2, 'name': 'a', 'time': d2008},
                      {'id': 3, 'name': 'a', 'time': None}])
        mxx = _get_max(db.session, Customer.id)
        d = pd.DataFrame([{'name': 'a', 'time': None}, {'name': 'b', 'time': d2006},
                          {'name': 'b', 'time': d2006}, {'name': 'a', 'time': None},
                          {'name': 'a', 'time': d2006}])
        # {'name': 'a', 'time': None} will be dropped (drop duplicates is True by default)
        # from the other three elements, the last is already existing and will not be added
        # the other two have a conflict and will be dropped as duplicates as well:
        expected_ids = [1]
        inserted, not_inserted, updated, not_updated, d2 = \
            syncdf(d, db.session, [Customer.name, Customer.time], Customer.id)
# d2 should be:
#           name       time   id
#         0    a        NaT  3.0
#         1    b 2006-01-01  4.0
#         4    a 2006-01-01  1.0
        assert array_equal(d2['id'], expected_ids)
        assert inserted == len(d2)-1  # assert one is already existing

    def test_syncdf_2_no_drop_dupes(self, db):
        """Same as `test_syncdf_2` but with drop duplicates False"""
        d2006 = datetime(2006, 1, 1)
        d2007 = datetime(2007, 12, 31)
        d2008 = datetime(2008, 3, 14)
        d2009 = datetime(2009, 9, 25)
        mxx = _get_max(db.session, Customer.id)
        self.init_db(db.session, 
                     [{'id': 1, 'name': 'a', 'time': d2006}, {'id': 2, 'name': 'a', 'time': d2008},
                      {'id': 3, 'name': 'a', 'time': None}])
        mxx = _get_max(db.session, Customer.id)
        d = pd.DataFrame([{'name': 'a', 'time': None}, {'name': 'b', 'time': d2006},
                          {'name': 'b', 'time': d2006}, {'name': 'a', 'time': None},
                          {'name': 'a', 'time': d2006}])
        # {'name': 'a', 'time': None} will NOT be dropped (drop duplicates is False here)
        # thus when fetching the pkey 'id' they will both have 3 (see above)
        #  {'name': 'b', 'time': d2006} will NOT be dropped as well and they will have two
        # different ids cause they do not have counterparts saved on the db
        # {'name': 'a', 'time': d2006} will have the 'id' 1 (see above):
        inserted, not_inserted, updated, not_updated, d2 = \
            syncdf(d, db.session, [Customer.name, Customer.time], Customer.id,
                   keep_duplicates=True)
        # the returned dataframe, as it does NOT update, has the instances with already set id
        # first (these instances are those who should be updated) and then the rest, so it's
        # like this: 
        assert array_equal(d2['id'], [3, 3, 1, mxx+1, mxx+2])

    def test_syncdf_3(self, db):
        d2006 = datetime(2006, 1, 1)
        d2007 = datetime(2007, 12, 31)
        d2008 = datetime(2008, 3, 14)
        d2009 = datetime(2009, 9, 25)
        self.init_db(db.session,
                     [{'id': 1, 'name': 'a', 'time': d2006}, {'id': 2, 'name': 'a', 'time': d2008},
                      {'id': 3, 'name': 'a', 'time': None}])
        d = pd.DataFrame([{'id': 45, 'name': 'a', 'time': None},
                          {'id': 45, 'name': 'b', 'time': d2006},
                          {'id': 45, 'name': 'c', 'time': d2009},
                          {'id': 45, 'name': 'a', 'time': None},
                          {'id': 45, 'name': 'a', 'time': d2006}])
        # we have these already existing instances: 1st, fourth, last instances
        # first and fourth are duplicates, so as drop duplicates is True, so they will be REMOVED
        # We remain with the other three: first, 2nd and third. AS update is false, they will
        # always be returned:
        inserted, not_inserted, updated, not_updated, d2 = \
            syncdf(d, db.session, [Customer.name, Customer.time], Customer.id)
        # Note that in this case {'id':45, 'name': 'b', 'time': d2006}, and
        # {'id':45, 'name': 'c', 'time': d2009} are still valid and in d2 cause they don't have
        # matching on the table => their id is not updated and they don't have null id =>
        # they id is not set
        expected_ids = [45, 45, 1]
        assert array_equal(d2['id'], expected_ids)

    def test_mergeupdate1(self):
        d2006 = datetime(2006, 1, 1)
        d2007 = datetime(2007, 12, 31)
        d2008 = datetime(2008, 3, 14)
        d2009 = datetime(2009, 9, 25)
        d = pd.DataFrame([{'id': 45, 'name': 'a', 'time': None},
                          {'id': 45, 'name': 'b', 'time': d2006},
                          {'id': 45, 'name': 'c', 'time': d2009},
                          {'id': 45, 'name': 'd', 'time': d2008},
                          {'id': 45, 'name': 'e', 'time': d2006}])

        dnew = pd.DataFrame([{'id': 45, 'name': 'a', 'time': None, 'a': 5},
                             {'id': 45, 'name': 'b', 'time': d2006, 'a': 5},
                             {'id': 45, 'name': 'c', 'time': d2009, 'a': 5},
                             {'id': 45, 'name': 'd', 'time': None, 'a': 5},
                             {'id': 45, 'name': 'e', 'time': d2006, 'a': 5}])

        d2 = mergeupdate(d, dnew, ['id', 'name'], ['time'])

        assert len(d2) == len(d)
        # assert new column of dnew is not added to d2:
        assert 'a' not in d2.columns
        # assert time columns of d2 are time columns of dnew
        assert array_equal(d2['time'].dropna(), dnew['time'].dropna()) and len(d2) == len(dnew)
        # for times, as numpy is weird about that and mergeupdate sets the values of a column
        # (np array) check also that types are the same
        assert d2['time'].dtype == dnew['time'].dtype

    def test_mergeupdate2(self):
        d2006 = datetime(2006, 1, 1)
        d2007 = datetime(2007, 12, 31)
        d2008 = datetime(2008, 3, 14)
        d2009 = datetime(2009, 9, 25)
        d = pd.DataFrame([{'id': 45, 'name': 'a', 'time': None},
                          {'id': 45, 'name': 'b', 'time': d2006},
                          {'id': 45, 'name': 'c', 'time': d2009},
                          {'id': 45, 'name': 'd', 'time': d2008},
                          {'id': 45, 'name': 'e', 'time': d2006}])

        # dnew has a dupe (last two elements) we will remove them in mergeupdate
        dnew = pd.DataFrame([{'id': 45, 'name': 'a', 'time': None, 'a': 5},
                             {'id': 45, 'name': 'b', 'time': d2006, 'a': 5},
                             {'id': 45, 'name': 'c', 'time': d2009, 'a': 5},
                             {'id': 45, 'name': 'd', 'time': None, 'a': 5},
                             {'id': 45, 'name': 'e', 'time': d2006, 'a': 5},
                             {'id': 45, 'name': 'e', 'time': d2006, 'a': 5}])

        d2 = mergeupdate(d, dnew, ['id', 'name'], ['time'])

        assert len(d2) == len(d)

        # assert new column of dnew is not added to d2:
        assert 'a' not in d2.columns
        # for times, as numpy is weird about that and mergeupdate sets the values of a column
        # (np array) check also that types are the same
        assert d2['time'].dtype == dnew['time'].dtype

    def test_mergeupdate3(self):
        d2006 = datetime(2006, 1, 1)
        d2007 = datetime(2007, 12, 31)
        d2008 = datetime(2008, 3, 14)
        d2009 = datetime(2009, 9, 25)
        d = pd.DataFrame([{'id': 45, 'name': 'a', 'time': None},
                          {'id': 45, 'name': 'b', 'time': d2006},
                          {'id': 45, 'name': 'c', 'time': d2009},
                          {'id': 45, 'name': 'd', 'time': d2008},
                          {'id': 45, 'name': 'e', 'time': d2006}])

        # dnew has a dupe (last two elements) we will remove them in mergeupdate
        dnew = pd.DataFrame([{'id': 45, 'name': 'a', 'time': None, 'a': 5},
                             {'id': 45, 'name': 'b', 'time': d2006, 'a': 5},
                             {'id': 45, 'name': 'c', 'time': d2009, 'a': 5},
                             {'id': 45, 'name': 'd', 'time': None, 'a': 5},
                             {'id': 45, 'name': 'e', 'time': d2006, 'a': 5},
                             {'id': 45, 'name': 'e', 'time': d2007, 'a': 5}])

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
        d = pd.DataFrame([{'id': 45, 'name': 'a', 'time': None},
                          {'id': 45, 'name': 'b', 'time': d2006},
                          {'id': 45, 'name': 'c', 'time': d2009},
                          {'id': 45, 'name': 'd', 'time': d2008},
                          {'id': 45, 'name': 'e', 'time': d2006}])

        # dnew has a dupe (last two elements) we will remove them in mergeupdate
        dnew = pd.DataFrame([{'id': 45, 'name': 'a', 'time': None, 'a': 5},
                             {'id': 45, 'name': 'b', 'time': d2006, 'a': 5},
                             {'id': 45, 'name': 'c', 'time': d2009, 'a': 5},
                             {'id': 45, 'name': 'd', 'time': None, 'a': 5},
                             {'id': 45, 'name': 'e', 'time': d2006, 'a': 5},
                             {'id': 45, 'name': 'x', 'time': d2007, 'a': 5}])

        d2 = mergeupdate(d, dnew, ['id', 'name'], ['time'])

        assert len(d2) == len(d)

        # assert last element of dnew is not in d2, as it has no matches ['id', 'name']
        assert (~(d2['time'] == d2007)).all()
        # assert new column of dnew is not added to d2:
        assert 'a' not in d2.columns
        # for times, as numpy is weird about that and mergeupdate sets the values of a column
        # (np array) check also that types are the same
        assert d2['time'].dtype == dnew['time'].dtype

    def test_mergeupdate5(self):
        '''test typical case and check resulting data frame'''
        d2006 = datetime(2006, 1, 1)
        d2007 = datetime(2007, 12, 31)
        d2008 = datetime(2008, 3, 14)
        d2009 = datetime(2009, 9, 25)
        d = pd.DataFrame([{'id': 45, 'name': 'a', 'time': None, 'data': 4},
                          {'id': 45, 'name': 'b', 'time': d2006, 'data': 5}])

        dnew = pd.DataFrame([{'id': 45, 'name': 'a', 'time': d2007, 'count': 5, 'value': np.nan},
                             {'id': 45, 'name': 'c', 'time': d2009, 'count': 5, 'value': 4.5}])

        d2 = mergeupdate(d, dnew, ['id', 'name'], ['time', 'value'])

        expected_df = pd.DataFrame([{'id': 45, 'name': 'a', 'time': d2007, 'data': 4,
                                     'value': np.nan},
                                    {'id': 45, 'name': 'b', 'time': d2006, 'data': 5,
                                     'value': np.nan}])
        assert len(d2) == len(d)
        pd.testing.assert_frame_equal(d2, expected_df)


def array_equal(a1, a2):
    """test array equality by assuming nan == nan. Probably already implemented
    somewhere in numpy, no time for browsing now"""
    return len(a1) == len(a2) and all([c ==d or (np.isnan(c) == np.isnan(d)) for c, d in zip(a1, a2)])
