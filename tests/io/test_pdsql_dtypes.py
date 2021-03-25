'''
Created on Apr 11, 2017

@author: riccardo
'''
from builtins import zip
from datetime import datetime
import os
import sys

import numpy as np
import pytest
import pandas as pd
from sqlalchemy.ext.declarative.api import declarative_base
from sqlalchemy import Column, Integer, String, LargeBinary, DateTime, Float, \
    Boolean

from stream2segment.io.db.pdsql import insertdf, dbquery2df, harmonize_columns

Base = declarative_base()


class Customer(Base):
    __tablename__ = "customer"
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, default='a')
    data = Column(LargeBinary)
    time = Column(DateTime)
    count = Column(Integer)
    price = Column(Float)
    validated = Column(Boolean)


class Test(object):

    # execute this fixture always even if not provided as argument:
    # https://docs.pytest.org/en/documentation-restructure/how-to/fixture.html#autouse-fixtures-xunit-setup-on-steroids
    @pytest.fixture(autouse=True)
    def init(self, request, db, data):
        # re-init a sqlite database (no-op if the db is not sqlite):
        db.create(to_file=False, custom_base=Base)

    def init_db(self, session, dframe):
        # use insertdf which takes care of type casting:
        _, df = insertdf(dframe, session, Customer, buf_size=len(dframe),
                         return_df=True)
        return df


    def test_types(self, db):

        # the dtypes of an inserted dataframe are inferred. How?
        # If all are the same type, ok, but what about Nones? then it's object
        dfr = pd.DataFrame([{'id': None, 'name': None, 'time': None, 'data': None,
                             'count': None, 'price': None, 'validated': None}])
        assert all(dfr.dtypes[c] == object for c in dfr.dtypes.keys())

        # if we provide a non-None value then the type is correctly inferred. How? let's see:
        dfr = pd.DataFrame([{'id': None, 'name': None, 'time': None, 'data': None,
                             'count': None, 'price': None, 'validated': None},
                            {'id': 1, 'name': 'a', 'time': pd.NaT, 'data': b'',
                             'count': 2, 'price': 1.1, 'validated': True}])
        # ints are converted to float if Nones are present, see columns id and count):
        assert dfr.dtypes['id'] == dfr.dtypes['count'] == dfr.dtypes['price'] == \
            np.dtype('float64')
        # strings and bytes to objects, as pandas (or numpy) does not support them natively.
        # Also, note that booleans with none are not converted to booleans (column 'validated'):
        assert dfr.dtypes['name'] == dfr.dtypes['data'] == dfr.dtypes['validated'] == \
            object
        # Contrarily to boolean, datetimes are converted:
        assert dfr.dtypes['time'] == np.dtype('datetime64[ns]')

        # What if we supply some missing values? then pandas assumes floats!
        dic = {'id': None, 'name': None, 'time': None, 'data': None,
               'count': None, 'price': None, 'validated': None}
        for col in ['time', 'id', 'name', 'time', 'count', 'validated', 'price']:
            dic1 = dict(dic)
            dic1.pop(col)
            dfr = pd.DataFrame([dic, dic1])
            assert dfr.dtypes[col] == np.dtype('float64')
            # all others are object, as above:
            assert all([dfr.dtypes[c] == object for c in dfr.dtypes.keys() if c != col])

    def test_dbquery2df_types(self, db):
        # now let's save to the db adataframe with nulls:
        dfr = pd.DataFrame([{'id': 1, 'name': 'a', 'time': None, 'data': None,
                             'count': None, 'price': None, 'validated': None},
                            {'id': 2, 'name': 'b', 'time': datetime.utcnow(), 'data': b'',
                             'count': 2, 'price': 1.1, 'validated': True}])
        # as seen above, the dataframe has weird types due to the way pandas infers them:
        # id                    int64
        # name                 object
        # time         datetime64[ns]
        # count               float64
        # data                 object
        # price               float64
        # validated            object

        # if we write to the db and we query back
        dfr_pre = self.init_db(db.session, dfr)
        # and we get beck the result:
        dfr_post = dbquery2df(db.session.query(Customer.id, Customer.name, Customer.data,
                                               Customer.time, Customer.count, Customer.price,
                                               Customer.validated))
        # THen:
        # id                    int64
        # name                 object -> pandas stores strings as object
        # data                 object -> see above (bytes behaves as strings)
        # time         datetime64[ns] -> ok, inferred from the non-None value (the other is NaT)
        # count               float64 -> pandas cannot handle int+None, defaults to float + nan
        # price               float64 -> ok, inferred from the non-None value (the other is nan)
        # validated            object -> pandas cannot handle bool+None, defaults to obj
        dtypes_pre = dfr_post.dtypes
        dfr_post2 = harmonize_columns(Customer, dfr_post.copy())
        dtypes_post = dfr_post2.dtypes
        # The conversion did converted only boolean because None values had to be handled
        # id                    int64
        # name                 object
        # data                 object
        # time         datetime64[ns]
        # count               float64
        # price               float64
        # validated           float64 -> boolean

        # let's check that:
        assert all(dfr_post[c].dtype == dfr_post2[c].dtype for c in dfr_post.columns
                   if c != 'validated')
        assert dfr_post['validated'].dtype == object
        assert dfr_post2['validated'].dtype == np.dtype('float64')

#         src_dtypes = dfr.dtypes
#         dfr2 = self.init_db(db.session, dfr)
#         assert all([dfr.dtypes[c] == dfr2.dtypes[c] for c in dfr.dtypes.keys()])
# 
#         dfr2 = dbquery2df(db.session.query(Customer.id, Customer.name, Customer.data,
#                                            Customer.time, Customer.count, Customer.price,
#                                            Customer.validated))
#         assert dfr2.dtypes['id'] == np.int64
#         for col in dfr2.dtypes.keys():
#             assert dfr2.dtypes[col] == (np.int64 if col == 'id' else object)




def array_equal(a1, a2):
    """test array equality by assuming nan == nan. Probably already implemented
    somewhere in numpy, no time for browsing now"""
    return len(a1) == len(a2) and all([c ==d or (np.isnan(c) == np.isnan(d)) for c, d in zip(a1, a2)])
