'''
Created on Feb 4, 2016

@author: riccardo
'''
from __future__ import print_function

from builtins import str
from datetime import datetime
# this can apparently not be avoided neither with the future package:
# The problem is io.StringIO accepts unicodes in python2 and strings in python3:
from io import BytesIO
try:
    from cStringIO import StringIO  # python2.x
except ImportError:
    from io import StringIO

from itertools import cycle
from logging import StreamHandler
from mock import patch
from mock import Mock

import pytest

from stream2segment.cli import cli
from stream2segment.download.main import get_events_df, get_datacenters_df, \
    save_inventories, get_channels_df, download_save_segments, configlog4download
from stream2segment.download.db import Segment, Download, Station
from stream2segment.io.db.pdsql import insertdf, updatedf
from stream2segment.download.modules.utils import s2scodes
from stream2segment.download.modules.mseedlite import unpack
from stream2segment.download.url import HTTPError, responses
from stream2segment.resources import get_templates_fpath
from stream2segment.io import yaml_load


class Test(object):

    # execute this fixture always even if not provided as argument:
    # https://docs.pytest.org/en/documentation-restructure/how-to/fixture.html#autouse-fixtures-xunit-setup-on-steroids
    @pytest.fixture(autouse=True)
    def init(self, request, db, data, pytestdir):
        # re-init a sqlite database (no-op if the db is not sqlite):
        db.create(to_file=False)


        self.logout = StringIO()
        self.handler = StreamHandler(stream=self.logout)
        # THIS IS A HACK:
        # s2s_download_logger.setLevel(logging.INFO)  # necessary to forward to handlers
        # if we called closing (we are testing the whole chain) the level will be reset
        # (to level.INFO)
        # otherwise it stays what we set two lines above. Problems might arise if closing
        # sets a different level, but for the moment who cares
        # s2s_download_logger.addHandler(self.handler)

        # setup a run_id:
        r = Download()
        db.session.add(r)
        db.session.commit()
        self.run = r

        # side effects:
        # THESE ARE COPIED VIA DEBUGGING FROM A CASE WHERE 
        # WE HAD TIME BOUNDS ERRORS

        self._evt_urlread_sideeffect =  """#EventID | Time | Latitude | Longitude | Depth/km | Author | Catalog | Contributor | ContributorID |  MagType | Magnitude | MagAuthor | EventLocationName
20160605_0000085|2016-06-05T21:06:04.7Z|45.51|25.91|49.0|BUC|EMSC-RTS|BUC|510656|ml|3.9|BUC|ROMANIA
"""
        eidafile = data.path("eida_routing_service_response.txt")
        with open(eidafile, 'r') as opn:
            self._dc_urlread_sideeffect = opn.read()


# Note: by default we set sta_urlsideeffect to return such a channels which result in 12
# segments (see lat and lon of channels vs lat and lon of events above)
        self._sta_urlread_sideeffect  = ["""#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|StartTime|EndTime
BS|BLKB||BHE|43.6227|22.675|650.0|0.0|90.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|BLKB||BHN|43.6227|22.675|650.0|0.0|0.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|BLKB||BHZ|43.6227|22.675|650.0|0.0|0.0|-90.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|BLKB||HHE|43.6227|22.675|650.0|0.0|90.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|BLKB||HHN|43.6227|22.675|650.0|0.0|0.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|BLKB||HHZ|43.6227|22.675|650.0|0.0|0.0|-90.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|BLKB||HNE|43.6227|22.675|650.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|BLKB||HNN|43.6227|22.675|650.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|BLKB||HNZ|43.6227|22.675|650.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|DOBAM||HNE|43.5811|27.831|246.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|DOBAM||HNN|43.5811|27.831|246.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|DOBAM||HNZ|43.5811|27.831|246.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|ELND||BHE|42.9287|25.8751|334.0|0.0|90.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|ELND||BHN|42.9287|25.8751|334.0|0.0|0.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|ELND||BHZ|42.9287|25.8751|334.0|0.0|0.0|-90.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|ELND||HHE|42.9287|25.8751|334.0|0.0|90.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|ELND||HHN|42.9287|25.8751|334.0|0.0|0.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|ELND||HHZ|42.9287|25.8751|334.0|0.0|0.0|-90.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|ELND||HNE|42.9287|25.8751|334.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|ELND||HNN|42.9287|25.8751|334.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|ELND||HNZ|42.9287|25.8751|334.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|JMB||BHE|42.466999|26.583|216.0|0.0|270.0|0.0|CMG40|120720000.0|0.02|M/S|20.0|2004-04-01T00:00:00|
BS|JMB||BHN|42.466999|26.583|216.0|0.0|180.0|0.0|CMG40|120720000.0|0.02|M/S|20.0|2004-04-01T00:00:00|
BS|JMB||BHZ|42.466999|26.583|216.0|0.0|180.0|90.0|CMG40|120720000.0|0.02|M/S|20.0|2004-04-01T00:00:00|
BS|KALB||BHE|43.4059|28.4162|121.0|0.0|90.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|KALB||BHN|43.4059|28.4162|121.0|0.0|0.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|KALB||BHZ|43.4059|28.4162|121.0|0.0|0.0|-90.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|KALB||HHE|43.4059|28.4162|121.0|0.0|90.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|KALB||HHN|43.4059|28.4162|121.0|0.0|0.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|KALB||HHZ|43.4059|28.4162|121.0|0.0|0.0|-90.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|KALB||HNE|43.4059|28.4162|121.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|KALB||HNN|43.4059|28.4162|121.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|KALB||HNZ|43.4059|28.4162|121.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|KOZAM||HNE|43.8274|23.2364|760.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|KOZAM||HNN|43.8274|23.2364|760.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|KOZAM||HNZ|43.8274|23.2364|760.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|KUBB||BHE|43.8024|26.4941|261.0|0.0|90.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|KUBB||BHN|43.8024|26.4941|261.0|0.0|0.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|KUBB||BHZ|43.8024|26.4941|261.0|0.0|0.0|-90.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|KUBB||HHE|43.8024|26.4941|261.0|0.0|90.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|KUBB||HHN|43.8024|26.4941|261.0|0.0|0.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|KUBB||HHZ|43.8024|26.4941|261.0|0.0|0.0|-90.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|KUBB||HNE|43.8024|26.4941|261.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|KUBB||HNN|43.8024|26.4941|261.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|KUBB||HNZ|43.8024|26.4941|261.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|LOZB||BHE|43.3701|26.593|342.0|0.0|90.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|LOZB||BHN|43.3701|26.593|342.0|0.0|0.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|LOZB||BHZ|43.3701|26.593|342.0|0.0|0.0|-90.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|LOZB||HHE|43.3701|26.593|342.0|0.0|90.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|LOZB||HHN|43.3701|26.593|342.0|0.0|0.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|LOZB||HHZ|43.3701|26.593|342.0|0.0|0.0|-90.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|LOZB||HNE|43.3701|26.593|342.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|LOZB||HNN|43.3701|26.593|342.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|LOZB||HNZ|43.3701|26.593|342.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|MALO||HNE|43.3559|23.7402|370.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2013-01-11T00:00:00|
BS|MALO||HNN|43.3559|23.7402|370.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2013-01-11T00:00:00|
BS|MALO||HNZ|43.3559|23.7402|370.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2013-01-11T00:00:00|
BS|MNNAM||HNE|43.4111|23.2262|0.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|MNNAM||HNN|43.4111|23.2262|0.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|MNNAM||HNZ|43.4111|23.2262|0.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|PLVAM||HNE|43.9523|22.8502|880.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|PLVAM||HNN|43.9523|22.8502|880.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|PLVAM||HNZ|43.9523|22.8502|880.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|PLVB||BHE|43.387|24.6207|199.0|0.0|90.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|PLVB||BHN|43.387|24.6207|199.0|0.0|0.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|PLVB||BHZ|43.387|24.6207|199.0|0.0|0.0|-90.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|PLVB||HHE|43.387|24.6207|199.0|0.0|90.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|PLVB||HHN|43.387|24.6207|199.0|0.0|0.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|PLVB||HHZ|43.387|24.6207|199.0|0.0|0.0|-90.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|PLVB||HNE|43.387|24.6207|199.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|PLVB||HNN|43.387|24.6207|199.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|PLVB||HNZ|43.387|24.6207|199.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|RAZAM||HNE|43.6451|25.1243|820.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|RAZAM||HNN|43.6451|25.1243|820.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|RAZAM||HNZ|43.6451|25.1243|820.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|RAZG||BHE|43.5662|26.5079|383.0|0.0|90.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|RAZG||BHN|43.5662|26.5079|383.0|0.0|0.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|RAZG||BHZ|43.5662|26.5079|383.0|0.0|0.0|-90.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|RAZG||HHE|43.5662|26.5079|383.0|0.0|90.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|RAZG||HHN|43.5662|26.5079|383.0|0.0|0.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|RAZG||HHZ|43.5662|26.5079|383.0|0.0|0.0|-90.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|RAZG||HNE|43.5662|26.5079|383.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|RAZG||HNN|43.5662|26.5079|383.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|RAZG||HNZ|43.5662|26.5079|383.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|RUSAM||HNE|43.8462|25.9612|0.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|RUSAM||HNN|43.8462|25.9612|0.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|RUSAM||HNZ|43.8462|25.9612|0.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|SHAB||BHE|43.5389|28.6057|430.0|0.0|90.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|SHAB||BHN|43.5389|28.6057|430.0|0.0|0.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|SHAB||BHZ|43.5389|28.6057|430.0|0.0|0.0|-90.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
BS|SHAB||HHE|43.5389|28.6057|430.0|0.0|90.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|SHAB||HHN|43.5389|28.6057|430.0|0.0|0.0|0.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|SHAB||HHZ|43.5389|28.6057|430.0|0.0|0.0|-90.0|Seismometer/Basalt|926790000.0|0.4|M/S|100.0|2012-11-20T00:00:00|
BS|SHAB||HNE|43.5389|28.6057|430.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|SHAB||HNN|43.5389|28.6057|430.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|SHAB||HNZ|43.5389|28.6057|430.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|SILAM||HNE|44.1046|27.2665|840.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|SILAM||HNN|44.1046|27.2665|840.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|SILAM||HNZ|44.1046|27.2665|840.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|VETAM||HNE|43.0805|25.6367|224.0|0.0|90.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|VETAM||HNN|43.0805|25.6367|224.0|0.0|0.0|0.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|
BS|VETAM||HNZ|43.0805|25.6367|224.0|0.0|0.0|-90.0|200|427475.0|0.02|M/S**2|100.0|2012-11-20T00:00:00|"""]
        # self._sta_urlread_sideeffect = cycle([partial_valid, '', invalid, '', '', URLError('wat'), socket.timeout()])

        # the segments downloads returns ALWAYS the same miniseed, which is the BS network
        # in a specified
        self._seg_data = data.read("BS.*.*.*.2016-06-05.21:05-09.47.mseed")

        self._seg_urlread_sideeffect = [self._seg_data]

        self._inv_data = data.read("inventory_GE.APE.xml")

        #add cleanup (in case tearDown is not called due to exceptions):
        # self.addCleanup(Test.cleanup, self)
                        #self.patcher3)

        self.configfile = get_templates_fpath("download.yaml")
        # self._logout_cache = ""

                # class-level patchers:
        with patch('stream2segment.utils.url.urlopen') as mock_urlopen:
            self.mock_urlopen = mock_urlopen
            with patch('stream2segment.utils.inputvalidation.valid_session', return_value=db.session):
                # this mocks yaml_load and sets inventory to False, as tests rely on that
                with patch('stream2segment.main.closesession'):  # no-op (do not close session)

                    def yload(*a, **v):
                        dic = yaml_load(*a, **v)
                        if 'inventory' not in v:
                            dic['inventory'] = False
                        else:
                            sdf = 0
                        return dic
                    with patch('stream2segment.utils.inputvalidation.yaml_load',
                               side_effect=yload) as mock_yaml_load:
                        self.mock_yaml_load = mock_yaml_load

                        # mock ThreadPool (tp) to run one instance at a time, so we
                        # get deterministic results:
                        class MockThreadPool(object):

                            def __init__(self, *a, **kw):
                                pass

                            def imap(self, func, iterable, *args):
                                # make imap deterministic: same as standard python map:
                                # everything is executed in a single thread the right input order
                                return map(func, iterable)

                            def imap_unordered(self, func, iterable, *args):
                                # make imap_unordered deterministic: same as standard python map:
                                # everything is executed in a single thread in the right input
                                # order
                                return map(func, iterable)

                            def close(self, *a, **kw):
                                pass
                        # assign patches and mocks:
                        with patch('stream2segment.utils.url.ThreadPool',
                                   side_effect=MockThreadPool) as mock_thread_pool:

                            def c4d(logger, logfilebasepath, verbose):
                                # config logger as usual, but redirects to a temp file
                                # that will be deleted by pytest, instead of polluting the program
                                # package:
                                if logfilebasepath is not None:
                                    logfilebasepath = pytestdir.newfile('.log')
                                ret = configlog4download(logger, logfilebasepath, verbose)
                                logger.addHandler(self.handler)
                                return ret
                            with patch('stream2segment.main.configlog4download',
                                       side_effect=c4d) as mock_config4download:
                                self.mock_config4download = mock_config4download

                                yield

    def log_msg(self):
        return self.logout.getvalue()
#         idx = len(self._logout_cache)
#         self._logout_cache = self.logout.getvalue()
#         if len(self._logout_cache) == idx:
#             idx = None # do not slice
#         return self._logout_cache[idx:]

    def setup_urlopen(self, urlread_side_effect):
        """setup urlopen return value. 
        :param urlread_side_effect: a LIST of strings or exceptions returned by urlopen.read, that will be converted
        to an itertools.cycle(side_effect) REMEMBER that any element of urlread_side_effect which is a nonempty
        string must be followed by an EMPTY
        STRINGS TO STOP reading otherwise we fall into an infinite loop if the argument
        blocksize of url read is not negative !"""

        self.mock_urlopen.reset_mock()
        # convert returned values to the given urlread return value (tuple data, code, msg)
        # if k is an int, convert to an HTTPError
        retvals = []
        # Check if we have an iterable (where strings are considered not iterables):
        if not hasattr(urlread_side_effect, "__iter__") or \
            isinstance(urlread_side_effect, (bytes, str)):
            # it's not an iterable (wheere str/bytes/unicode are considered NOT
            # iterable in both py2 and 3)
            urlread_side_effect = [urlread_side_effect]

        for k in urlread_side_effect:
            a = Mock()
            if type(k) == int:
                a.read.side_effect = HTTPError('url', int(k),  responses[k], None, None)
            elif type(k) in (bytes, str):
                def func(k):
                    b = BytesIO(k.encode('utf8') if type(k) == str else k)  # py2to3 compatible
                    def rse(*a, **v):
                        rewind = not a and not v
                        if not rewind:
                            currpos = b.tell()
                        ret = b.read(*a, **v)
                        # hacky workaround to support cycle below: if reached the end,
                        # go back to start
                        if not rewind:
                            cp = b.tell()
                            rewind = cp == currpos
                        if rewind:
                            b.seek(0, 0)
                        return ret
                    return rse
                a.read.side_effect = func(k)
                a.code = 200
                a.msg = responses[a.code]
            else:
                a.read.side_effect = k
            retvals.append(a)

        self.mock_urlopen.side_effect = cycle(retvals)

    def get_events_df(self, url_read_side_effect, *a, **v):
        self.setup_urlopen(self._evt_urlread_sideeffect if url_read_side_effect is None
                           else url_read_side_effect)
        return get_events_df(*a, **v)

    def get_datacenters_df(self, url_read_side_effect, *a, **v):
        self.setup_urlopen(self._dc_urlread_sideeffect if url_read_side_effect is None
                           else url_read_side_effect)
        return get_datacenters_df(*a, **v)

    def get_channels_df(self, url_read_side_effect, *a, **kw):
        self.setup_urlopen(self._sta_urlread_sideeffect if url_read_side_effect is None
                           else url_read_side_effect)
        return get_channels_df(*a, **kw)

# # ================================================================================================= 

    def download_save_segments(self, url_read_side_effect, *a, **kw):
        self.setup_urlopen(self._seg_urlread_sideeffect if url_read_side_effect is None
                           else url_read_side_effect)
        return download_save_segments(*a, **kw)

    def save_inventories(self, url_read_side_effect, *a, **v):
        self.setup_urlopen(self._inv_data if url_read_side_effect is None
                           else url_read_side_effect)
        return save_inventories(*a, **v)


    @patch('stream2segment.download.main.get_events_df')
    @patch('stream2segment.download.main.get_datacenters_df')
    @patch('stream2segment.download.main.get_channels_df')
    @patch('stream2segment.download.main.save_inventories')
    @patch('stream2segment.download.main.download_save_segments')
    @patch('stream2segment.download.modules.segments.mseedunpack')
    @patch('stream2segment.io.db.pdsql.insertdf')
    @patch('stream2segment.io.db.pdsql.updatedf')
    def test_cmdline_outofbounds(self, mock_updatedf, mock_insertdf, mock_mseed_unpack,
                                 mock_download_save_segments, mock_save_inventories,
                                 mock_get_channels_df, mock_get_datacenters_df, mock_get_events_df,
                                 # fixtures:
                                 db, clirunner):

        mock_get_events_df.side_effect = lambda *a, **v: self.get_events_df(None, *a, **v)
        mock_get_datacenters_df.side_effect = \
            lambda *a, **v: self.get_datacenters_df(None, *a, **v) 
        mock_get_channels_df.side_effect = lambda *a, **v: self.get_channels_df(None, *a, **v)
        mock_save_inventories.side_effect = lambda *a, **v: self.save_inventories(None, *a, **v)
        mock_download_save_segments.side_effect = \
            lambda *a, **v: self.download_save_segments(None, *a, **v)
        mock_mseed_unpack.side_effect = lambda *a, **v: unpack(*a, **v)
        mock_insertdf.side_effect = lambda *a, **v: insertdf(*a, **v)
        mock_updatedf.side_effect = lambda *a, **v: updatedf(*a, **v)
        # prevlen = len(db.session.query(Segment).all())

        # The run table is populated with a run_id in the constructor of this class
        # for checking run_ids, store here the number of runs we have in the table:
        runs = len(db.session.query(Download.id).all())

        result = clirunner.invoke(cli, ['download',
                                        '-c', self.configfile,
                                        '--dburl', db.dburl,
                                        '--start', '2016-05-08T00:00:00',
                                        '--end', '2016-05-08T9:00:00'])
        assert clirunner.ok(result)
        
        # check datetimes are correctly written (pd > 1.0 and DatetimeArrays):
        # BS|BLKB||BHE|43.6227|22.675|650.0|0.0|90.0|0.0|Seismometer/Basalt|926787000.0|0.2|M/S|20.0|2012-11-20T00:00:00|
        qry = db.session.query(Station).\
            filter((Station.network == 'BS') &
                   (Station.station == 'BLKB') &
                   (Station.start_time == datetime(2012, 11, 20))).all()
        assert len(qry) == 1

        assert len(db.session.query(Download.id).all()) == runs + 1
        runs += 1

        timespan_err, timespan_warn = s2scodes.timespan_err, s2scodes.timespan_warn

        # assert when we have a timespan error we do not have data:
        assert db.session.query(Segment).filter(Segment.has_data &
                                                (Segment.download_code==timespan_err)).count() == 0

        # assert when we have a timespan warn or 200 response we have data:
        seg_with_data = db.session.query(Segment).filter(Segment.has_data).count()
        assert db.session.query(Segment).\
            filter(Segment.has_data &
                   ((Segment.download_code == 200) |
                    (Segment.download_code == timespan_warn))).count() == seg_with_data

        # pickup some examples and test them (the examples where found by debugging
        # download_segments):
        data = {timespan_warn: ['BS.BLKB..HHE', 'BS.BLKB..HHN', 'BS.BLKB..HHZ', 'BS.BLKB..HNZ',
                                'BS.DOBAM..HNE', 'BS.DOBAM..HNN', 'BS.DOBAM..HNZ'],
                timespan_err: ['BS.KALB..HHE', 'BS.KALB..HHN', 'BS.KALB..HHZ'],
                200: ['BS.BLKB..HNE', 'BS.BLKB..HNN']}

        for downloadcode, mseedids in data.items():
            for mseedid in mseedids:
                seg = db.session.query(Segment).filter(Segment.data_seed_id == mseedid).first()
                assert seg.download_code == downloadcode
                if (downloadcode == timespan_err):
                    assert not seg.has_data
                else:
                    assert seg.has_data
