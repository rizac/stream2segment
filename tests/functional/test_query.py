'''
Created on Feb 4, 2016

@author: riccardo
'''
# from event2waveform import getWaveforms
# from utils import date
# assert sys.path[0] == os.path.realpath(myPath + '/../../')
import numpy as np
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
from stream2segment.s2sio.db.pd_sql_utils import df2dbiter, get_col_names
import pandas as pd
from stream2segment.query import get_datacenters as gdc_orig
from obspy.core.stream import Stream


class Test(unittest.TestCase):
    
    engine = None
    dburi = ""

#     @classmethod
#     def setUpClass(cls):
#         file = os.path.dirname(__file__)
#         filedata = os.path.join(file,"..","data")
#         url = os.path.join(filedata, "_test.sqlite")
#         Test.dburi = 'sqlite:///' + url
#         # an Engine, which the Session will use for connection
#         # resources
#         # some_engine = create_engine('postgresql://scott:tiger@localhost/')
#         Test.engine = create_engine(Test.dburi)
#         # Base.metadata.drop_all(cls.engine)
#         Base.metadata.create_all(cls.engine)
# 
#     @classmethod
#     def tearDownClass(cls):
#         Base.metadata.drop_all(cls.engine)

    def setUp(self):
        file = os.path.dirname(__file__)
        filedata = os.path.join(file,"..","data")
        url = os.path.join(filedata, "_test.sqlite")
        Test.dburi = 'sqlite:///' + url
        # an Engine, which the Session will use for connection
        # resources
        # some_engine = create_engine('postgresql://scott:tiger@localhost/')
        Test.engine = create_engine(Test.dburi)
        # Base.metadata.drop_all(cls.engine)
        Base.metadata.create_all(self.engine)

    def tearDown(self):
        Base.metadata.drop_all(self.engine)

    @property
    def session(self):
        # create a configured "Session" class
        Session = sessionmaker(bind=self.engine)
        # create a Session
        session = Session()
        return session

    
    def get_events_df(self, *a, **k):
        pddf = pd.DataFrame(columns = get_col_names(models.Event),
                            data = [[
                                     "20160508_0000129",
                                     "2016-05-08 05:17:11.500000",
                                     "40.57",
                                     "52.23",
                                     "60.0",
                                     "AZER",
                                     "EMSC-RTS",
                                     "AZER",
                                     "505483",
                                     "ml",
                                     "3.1",
                                     "AZER",
                                     "CASPIAN SEA, OFFSHR TURKMENISTAN"],
                                    ["20160508_0000004",
                                     "2016-05-08 01:45:30.300000",
                                     "44.96",
                                     "15.35",
                                     "2.0",
                                     "EMSC",
                                     "EMSC-RTS",
                                     "EMSC",
                                     "505183",
                                     "ml",
                                     "3.6",
                                     "EMSC",
                                     "CROATIA"],
                                    ["20160508_0000113",
                                     "2016-05-08 22:37:20.100000",
                                     "45.68",
                                     "26.64",
                                     "163.0",
                                     "BUC",
                                     "EMSC-RTS",
                                     "BUC",
                                     "505351",
                                     "ml",
                                     "3.4",
                                     "BUC",
                                     "ROMANIA"]])
        return pddf
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

#     def test_download(self):
#         runner = CliRunner()
#         result = runner.invoke(main , ['-a', 'p', '--start', '2016-05-08T00:00:00', '--end', '2016-05-08T09:00:00'])
#         if result.exception:
#             raise result.exception
    
    def get_geofon_dc_only(self, *a, **k):
        dcs = gdc_orig(*a, **k)
        return [d for d in dcs if "geofon" in d.station_query_url]
        
    
    @patch('stream2segment.query.get_datacenters')
    def test_download_too_little_timespan(self, getdc):
        getdc.side_effect = self.get_geofon_dc_only
         
        prevlen = len(self.session.query(models.Segment).all())
 
        runner = CliRunner()
        result = runner.invoke(main , ['--dburi', self.dburi, '-a', 'd',
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T9:00:00'])
        if result.exception:
            raise result.exception
         
        assert len(self.session.query(models.Segment).all()) == prevlen
        # assert result.exit_code == 1
 
     
    @patch('stream2segment.query.get_datacenters')
    def test_download_no_sample_rate_matching(self, getdc):
        getdc.side_effect = self.get_geofon_dc_only
         
        prevlen = len(self.session.query(models.Segment).all())
 
        runner = CliRunner()
        result = runner.invoke(main , ['--dburi', self.dburi, '--min_sample_rate', 60, '-a', 'd',
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-08T18:00:00'])
        if result.exception:
            raise result.exception
         
        assert len(self.session.query(models.Segment).all()) == prevlen


    @patch('stream2segment.query.get_datacenters')
    @patch('stream2segment.query.get_events_df')
    def test_download_process(self, getedf, getdc):
        getedf.side_effect = self.get_events_df
        getdc.side_effect = self.get_geofon_dc_only
        runner = CliRunner()
        result = runner.invoke(main , ['--dburi', self.dburi, '--min_sample_rate', 60, '-a', 'dp',
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-09T00:00:00'])
        if result.exception:
            raise result.exception
        
        segs = self.session.query(models.Segment).all()
        procs = self.session.query(models.Processing).all()
        
        assert len(segs) == len(procs) 
        assert len(segs) > 0
        
        self.txst_process()
        
    
    def txst_process(self):
        
        runner = CliRunner()
        result = runner.invoke(main , ['--dburi', self.dburi, '--min_sample_rate', 60, '-a', 'P',
                                       '--start', '2016-05-08T00:00:00',
                                       '--end', '2016-05-09T00:00:00'])
        if result.exception:
            raise result.exception
        
        segs = self.session.query(models.Segment).all()
        procs = self.session.query(models.Processing).all()
        
        assert len(segs) == len(procs)
        assert len(segs) > 0

        self.txst_read_obj()

    def txst_read_obj(self):
        pro = self.session.query(models.Processing).first()
        from stream2segment.s2sio.dataseries import loads

        array = loads(pro.mseed_rem_resp_savewindow)
        assert isinstance(array, Stream)
        assert len(array) == 1

        array = loads(pro.wood_anderson_savewindow)
        assert isinstance(array, Stream)
        assert len(array) == 1

        array = loads(pro.cum_rem_resp)
        assert isinstance(array, Stream)
        assert len(array) == 1

        array = loads(pro.fft_rem_resp_t05_t95)
        assert isinstance(array, Stream)
        assert array[0].stats._format == 'PICKLE'
        assert len(array[0].data) > 1
#         assert array.stats.startfreq == 0
#         assert array.stats.delta > 0

        array = loads(pro.fft_rem_resp_until_atime)
        assert isinstance(array, Stream)
        assert array[0].stats._format == 'PICKLE'
        assert len(array[0].data) > 1



    def test_df2dbiter(self):
        from stream2segment.classification import class_labels_df
            
        # define a list function which behaves like a list(iterator) but skips None values,
        # which might be returned by df2dbiter
        
        def list_(iterator):
            return [x for x in iterator if x is not None]
        # change column names to some names not present in models.Class,
        # and assert the length of returned model
        # instances is zero
        dframe = class_labels_df.copy()
        dframe.columns =['a', 'b', 'c']
        d = df2dbiter(dframe,
                                        models.Class,
                                        harmonize_columns_first=False,
                                        harmonize_rows=False)
        assert len(list_(d)) == 0

        # test the "normal" case
        dframe = class_labels_df
        for c,r  in [(True, False), (True, True), (False, True), (False, False)]:
            d = df2dbiter(dframe,
                                            models.Class,
                                            harmonize_columns_first=c,
                                            harmonize_rows=r)
            assert len(list_(d)) == len(dframe)
        
        # change a type which should be numeric and check that the returned dataframe
        # has one row less
        dframe.loc[1, models.Class.id.key] = 'a'
        
        # but wait... set harmonize to false, we should still have the same number
        # of rows
        d = df2dbiter(dframe,
                                        models.Class,
                                        harmonize_columns_first=False,
                                        harmonize_rows=False)
        assert len(list_(d)) == len(dframe)
        
        # now set harmonize_rows to True and.. one row less? NO! because
        # harmonize columns is False, so there will not be a type conversion
        # which makes that 'a' = None
        d = df2dbiter(dframe,
                                        models.Class,
                                        harmonize_columns_first=False,
                                        harmonize_rows=True)
        assert len(list_(d)) == len(dframe)

        # now set harmonize_columns also to True and.. one row less?
        d = df2dbiter(dframe,
                                        models.Class,
                                        harmonize_columns_first=True,
                                        harmonize_rows=True)
        assert len(list_(d)) == len(dframe)-1
        
        
        # FIXME: Check this warning WARNING:
        
        # WARNING (norm_resp): computed and reported sensitivities differ by more than 5 percent.
        # Execution continuing.
        
        # common query for 1 event found and all datacenters (takes more or less 10 to 20 minutes):
        # stream2segment -f 2016-05-08T22:45:00 -t 2016-05-08T23:00:00
        