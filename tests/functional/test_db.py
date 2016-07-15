'''
Created on Jul 15, 2016

@author: riccardo
'''
import unittest
import datetime
import os
from stream2segment.s2sio.db import models
from stream2segment.s2sio.db.models import Base  # This is your declarative base class
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd


class Test(unittest.TestCase):


    def setUp(self):
        file = os.path.dirname(__file__)
        filedata = os.path.join(file,"..","data")
        url = os.path.join(filedata, "_test.sqlite")
        # an Engine, which the Session will use for connection
        # resources
        # some_engine = create_engine('postgresql://scott:tiger@localhost/')
        self.engine = create_engine('sqlite:///'+url)

        # create a configured "Session" class
        Session = sessionmaker(bind=self.engine)

        # create a Session
        self.session = Session()

        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)
    
    def tearDown(self):
        self.session.close()
        # self.DB.drop_all()


    def testAddRunId(self):
        #run_cols = models.Run.__table__.columns.keys()
        #run_cols.remove('id')  # remove id (auto set in the model)
        #d = pd.DataFrame(columns=run_cols, data=[[None for _ in run_cols]])
        #records = d.to_dict('records') # records(index=False)
        # record = records[0]

        # pass a run_id without id and see if it's updated as utcnow:
        run_row = models.Run()
        assert run_row.id is None

        run_row = models.Run(id=None)
        assert run_row.id is None

        # test that methods of the base class work:
        assert len(run_row.get_cols()) > 0
        assert len(run_row.get_col_names()) > 0

        # test id is auto added:
        self.session.add_all([run_row])
        # self.session.flush()
        self.session.commit()
        assert run_row.id is not None

        # now pass a utcdatetime and see if we keep that value:
        utcnow = datetime.datetime.utcnow()
        run_row = models.Run(id=utcnow)
        assert run_row.id == utcnow
        self.session.add_all([run_row])
        # self.session.flush()
        self.session.commit()
        assert run_row.id == utcnow

        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()