'''
Created on Oct 24, 2016

@author: riccardo
'''
import os
import numpy as np
from obspy import read as obspy_read
from stream2segment.analysis.mseeds import remove_response, dumps__, dumps, np_dumps_compressed,\
loads__, np_loads_compressed, dumps_test, loads_test, fft
from stream2segment.analysis import mseeds
import tempfile
import pickle
from obspy.core.utcdatetime import UTCDateTime
import gzip
import cPickle
import time

import warnings
from collections import OrderedDict
from obspy.core.stream import Stream
from obspy.core.trace import Trace
from obspy.io.segy.header import DATA_SAMPLE_FORMAT_CODE_DTYPE
from stream2segment.s2sio.dataseries import LightTrace
warnings.filterwarnings("ignore")


def size(filepath):
    stats = os.stat(filepath)
    size_ = stats.st_size
    return "%.3f Kb" % (size_ / 1000.0)


def test_io():
    # see here:
    # http://stackoverflow.com/questions/10075661/how-to-save-dictionaries-and-arrays-in-the-same-archive-with-numpy-savez
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    mseedfile = os.path.join(folder, 'trace_GE.APE.mseed')
    print "Loading %s, size: %s " % (mseedfile, size(mseedfile))
    print "Simulating typical calculation, removing response (please wait)"

    mseed = obspy_read(mseedfile)
    inv_path = os.path.join(folder, 'inventory_GE.APE.xml')
    mseed_disp = remove_response(mseed, inv_path, output='DISP')

    fft_test = fft(mseed_disp, mseed_disp.traces[0].stats.starttime + 0.2, 60,
                                              taper_max_percentage=0.05)

    fft_test = LightTrace(fft_test, startfreq=0, delta=1.0/mseed_disp.traces[0].stats.delta)

    objects = {'stream': mseed, 'stream-rem-resp': mseed_disp, 'stream-rem-resp-fft': fft_test}
    # light_obj = LightTrace(obj)

    print "done"

    print "Dumping new mseed (response removed) with several scenarios. Compare sizes:"
    vals = OrderedDict()
    vals['pickle (old)'] = lambda obj: dumps__(obj, data_type=mseeds._IO_FORMAT_STREAM)
    vals['pickle + bz2'] = lambda obj: dumps_test(obj, pickleprotocol=0,
                                                  compresslevel=9, compression='bz2')
    vals['pickle + zlib'] = lambda obj: dumps_test(obj, pickleprotocol=0,
                                                   compresslevel=9, compression='zlib')
    vals['pickle + zip'] = lambda obj: dumps_test(obj, pickleprotocol=0,
                                                   compresslevel=9, compression='zip')
    vals['pickle + gzip'] = lambda obj: dumps_test(obj, pickleprotocol=0,
                                                   compresslevel=9, compression='gzip')
    vals['pickle high protocol + bz2'] = lambda obj: dumps_test(obj,
                                                                pickleprotocol=pickle.HIGHEST_PROTOCOL,
                                                                compresslevel=9, compression='bz2')
    vals['pickle h.p. + zlib'] = lambda obj: dumps_test(obj, pickleprotocol=pickle.HIGHEST_PROTOCOL,
                                                                 compresslevel=9, compression='zlib')
    vals['pickle h.p. + zip'] = lambda obj: dumps_test(obj, pickleprotocol=pickle.HIGHEST_PROTOCOL,
                                                                 compresslevel=9, compression='zip')
    vals['pickle h.p. + gzip'] = lambda obj: dumps_test(obj, pickleprotocol=pickle.HIGHEST_PROTOCOL,
                                                                 compresslevel=9, compression='gzip')
#     vals['pickle h.p. + gzip + less data'] = lambda obj: dumps_test(light_obj,
#                                                                              pickleprotocol=pickle.HIGHEST_PROTOCOL,
#                                                                              compresslevel=9, compression='gzip')

    # ---------------------------------
#     vals['pickle_new_hihprotocol'] = dumps_test(obj, protocol=pickle.HIGHEST_PROTOCOL)
#     vals['pickle_new_0protocol_compressed_gzip'] = dumps_test(obj,
#                                                            protocol=0,
#                                                            compression=9)
#     vals['pickle_new_highprotocol_compressed_gzip_lessdata'] = dumps_test(dict(data=mseed_disp.traces[0].data,
#                                     starttime=float(mseed_disp.traces[0].stats.starttime),
#                                     delta=float(mseed_disp.traces[0].stats.delta)),
#                                                            protocol=pickle.HIGHEST_PROTOCOL,
#                                                            compression=9)
#     vals['pickle_new_0rotocol_compressed_bz2'] = dumps_test(obj,
#                                                            protocol=0,
#                                                            compression=9, compressionname='bz2')
#     vals['numpy_savez'] = np_dumps_compressed(data=mseed_disp.traces[0].data,
#                                     starttime=float(mseed_disp.traces[0].stats.starttime),
#                                     delta=float(mseed_disp.traces[0].stats.delta),
#                                     id='asdasdasdasdasdasdasdasdasdasdasd')
    u = UTCDateTime()
    assert UTCDateTime(float(u)) == u

    names = []
    try:
        for obj in objects:
            print " Test with '%s'" % obj
            obj = objects[obj]
            for key in vals:
                start = time.time()
                try:
                    bytestr = vals[key](obj)
                except ValueError as error:
                    print "'%30s' ERROR: %s" % (key, str(error))
                    continue
                # print "key: %s" % key
                try:
                    _ = loads_test(bytestr)
                except Exception as exc:
                    # _ = loads_test(bytestr)
                    print "'%30s' ERROR: %s" % (key, str(exc))
                    continue
                end = time.time()
                ttt = end - start
                with tempfile.NamedTemporaryFile(suffix='.zip') as t:
                    names.append(t.name)
                    assert type(bytestr) == str
                    t.write(bytestr)
                    print "'%30s' string lenght: %10d, file size: %14s. Time to dump and load back: %f" % \
                        (key, len(bytestr), size(t.name), ttt)
    finally:
        # foe safety:
        for n in names:
            if os.path.isfile(n):
                try:
                    os.remove(n)
                except:
                    pass
if __name__ == "__main__":
    test_io()