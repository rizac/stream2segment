'''
Created on Feb 4, 2016

@author: riccardo
'''
# from event2waveform import getWaveforms
# from utils import date
# assert sys.path[0] == os.path.realpath(myPath + '/../../')
from mock import patch
import pytest
from mock import Mock
from datetime import datetime, timedelta
from stream2segment.utils import datetime as dtime
from StringIO import StringIO
from obspy.taup.taup import getTravelTimes
import unittest, os
from sqlalchemy.engine import create_engine
from stream2segment.s2sio.db.models import Base
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.exc import IntegrityError
from stream2segment.main import main
from click.testing import CliRunner
from stream2segment.s2sio.db import models
from stream2segment.s2sio.db.pd_sql_utils import df_to_table_iterrows
import pandas as pd

class Test(unittest.TestCase):
    
    engine = None

#     @classmethod
#     def setUpClass(cls):
#         file = os.path.dirname(__file__)
#         filedata = os.path.join(file,"..","data")
#         url = os.path.join(filedata, "_test.sqlite")
#         # an Engine, which the Session will use for connection
#         # resources
#         # some_engine = create_engine('postgresql://scott:tiger@localhost/')
#         cls.engine = create_engine('sqlite:///'+url)
#         Base.metadata.drop_all(cls.engine)
#         Base.metadata.create_all(cls.engine)
# 
# 
#     def setUp(self):
#         # create a configured "Session" class
#         Session = sessionmaker(bind=self.engine)
#         # create a Session
#         self.session = Session()
# 
#     def tearDown(self):
#         try:
#             self.session.flush()
#             self.session.commit()
#         except IntegrityError:
#             self.session.rollback()
#         self.session.close()

    def test_download(self):
        runner = CliRunner()
        result = runner.invoke(main , ['--start', '2016-05-08T00:00:00', '--end', '2016-05-08T09:00:00'])
        if result.exception:
            raise result.exception
            
        # assert result.exit_code == 1

    def test_df_to_iterrows(self):
        from stream2segment.classification import class_labels_df
            
        # change column names to some names not present in models.Class,
        # and assert the length of returned model
        # instances is zero
        dframe = class_labels_df.copy()
        dframe.columns =['a', 'b', 'c']
        d = df_to_table_iterrows(models.Class,
                                        dframe,
                                        harmonize_columns_first=False,
                                        harmonize_rows=False)
        assert len(list(d)) == 0

        # test the "normal" case
        dframe = class_labels_df
        for c,r  in [(True, False), (True, True), (False, True), (False, False)]:
            d = df_to_table_iterrows(models.Class,
                                            dframe,
                                            harmonize_columns_first=c,
                                            harmonize_rows=r)
            assert len(list(d)) == len(dframe)
        
        # change a type which should be numeric and check that the returned dataframe
        # has one row less
        dframe.loc[1, models.Class.id.key] = 'a'
        
        # but wait... set harmonize to false, we should still have the same number
        # of rows
        d = df_to_table_iterrows(models.Class,
                                        dframe,
                                        harmonize_columns_first=False,
                                        harmonize_rows=False)
        assert len(list(d)) == len(dframe)
        
        # now set harmonize_rows to True and.. one row less? NO! because
        # harmonize columns is False, so there will not be a type conversion
        # which makes that 'a' = None
        d = df_to_table_iterrows(models.Class,
                                        dframe,
                                        harmonize_columns_first=False,
                                        harmonize_rows=True)
        assert len(list(d)) == len(dframe)

        # now set harmonize_columns also to True and.. one row less?
        d = df_to_table_iterrows(models.Class,
                                        dframe,
                                        harmonize_columns_first=True,
                                        harmonize_rows=True)
        assert len(list(d)) == len(dframe)-1
        
        
        