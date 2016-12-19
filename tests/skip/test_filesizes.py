'''
Created on Oct 24, 2016

@author: riccardo
'''
import os
import numpy as np
from obspy import read as obspy_read
from stream2segment.analysis.mseeds import remove_response, fft
# , dumps__, dumps, np_dumps_compressed,\
# loads__, np_loads_compressed, fft  # , dumps_new, loads_test, fft
from stream2segment.io.dataseries import dumps, loads # as dumps_new, loads as loads_new
from stream2segment.analysis import mseeds
import tempfile
import pickle
from obspy.core.utcdatetime import UTCDateTime
import gzip
import cPickle
import time
from datetime import timedelta
import warnings
from collections import OrderedDict as odict
from obspy.core.stream import Stream
from obspy.core.trace import Trace
from obspy.io.segy.header import DATA_SAMPLE_FORMAT_CODE_DTYPE
#from stream2segment.s2sio.dataseries import LightTrace
from obspy.core.inventory.inventory import read_inventory
warnings.filterwarnings("ignore")


def filesize_kb(filepath):
    stats = os.stat(filepath)
    size_ = stats.st_size
    return (size_ / 1000.0)


def size_kb(bytestr):
    return (len(bytestr) / 1000.0)


def test_io():
    # see here:
    # http://stackoverflow.com/questions/10075661/how-to-save-dictionaries-and-arrays-in-the-same-archive-with-numpy-savez
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    mseedfile = os.path.join(folder, 'Package_GE.APE.mseed')  # 'trace_GE.APE.mseed')
    print "Loading %s, size: %s Kb" % (mseedfile, filesize_kb(mseedfile))
    mseeds = obspy_read(mseedfile)
    # add custom mseed:
    mseedsingle = obspy_read(os.path.join(folder, 'trace_GE.APE.mseed'))
    mseeds.traces.insert(0, mseedsingle.traces[0])
    

    mseedstimes = sorted(mseeds, key=lambda t:t.stats.endtime-t.stats.starttime)
    mseedsnpts = sorted(mseeds, key=lambda t:len(t.data))

    max_traces_tries = 20
    n = int(round(max_traces_tries / 4.0))
    mseeds = Stream(mseedstimes[:n] + mseedstimes[-n:]+mseedsnpts[:n] + mseedsnpts[-n:])
    max_traces_tries = len(mseeds)
    M = 5
    
    mseeds = Stream([t.trim(None, t.stats.starttime + timedelta(minutes=M)) for t in mseeds])

    print "Taken %d mseeds " % max_traces_tries
    print "mseed npts (max, min, average): %d %d %d" % (max(len(t) for t in mseeds),
                                                  min(len(t) for t in mseeds),
                                                  sum(len(t) for t in mseeds)/float(max_traces_tries))
    print "mseeds duration: %d minutes" % M
    print "Loading inventory object (please wait)"
    inv_path = os.path.join(folder, 'inventory_GE.APE.xml')
    inv_obj = read_inventory(inv_path)
# 
#     mseed_disp = remove_response(mseed, inv_path, output='DISP')
# 
#     fft_test = fft(mseed_disp, mseed_disp.traces[0].stats.starttime + 0.2, 60,
#                                               taper_max_percentage=0.05)
# 
#     fft_test = LightTrace(fft_test, startfreq=0, delta=1.0/mseed_disp.traces[0].stats.delta)

    objects = odict((('stream', {}), ('stream-rem-resp', {}), ('stream-rem-resp-fft', {})))
    # light_obj = LightTrace(obj)

    print "done"

    print "IO operations with with several scenarios:"
    # this flag is set to False as pickle+compression (where pickle is WITHOUT HIGH_PROTOCOL)
    # ALWAYS has lower performances (in term of file size) than pickle HP + compression
    # Set to True if you want to see the results

    # dummy test (remove later?):
    u = UTCDateTime()
    assert UTCDateTime(float(u)) == u

    benchmark_filenum = 5000
    
    names = []
    print "Benchmark db number of items: %d" % benchmark_filenum
    print "Processing %d traces, taking the mean and multiplying for %d" % (max_traces_tries,
                                                                              benchmark_filenum)

    try:
        for count_, mseed in enumerate(mseeds):
            print "processing mseed %d of %d (removing response, calculating fft, dumping and loading)" % (count_, max_traces_tries)
            mseed = Stream(mseed)
            mseed_disp = remove_response(mseed, inv_obj, output='DISP')
            fft_test = fft(mseed_disp, mseed_disp.traces[0].stats.starttime + 0.2, 60,
                            taper_max_percentage=0.05)
            # fft_test = LightTrace(fft_test, startfreq=0, delta=1.0/mseed_disp.traces[0].stats.delta)
            for dataname, performances in objects.iteritems():
                obj = mseed if dataname == 'stream' else mseed_disp if dataname == 'stream-rem-resp' else \
                    fft_test
    
                methods = odict()
                # methods['format=pickle'] = lambda obj: dumps(obj, format=0, compression=None, 
                #                                      compresslevel=9)
                
                if obj == fft_test:
                    frmt = "pickle"
                else:
                    frmt = "mseed"
    
                methods['format=None(%s)NO-COMP.' % frmt] = lambda obj: dumps(obj, format=None, compression=None,
                                                                 compresslevel=9)
                methods['format=None(%s)+zip' % frmt] = lambda obj: dumps(obj, format=None, compression='zip',
                                                                 compresslevel=9)
                methods['format=None(%s)+bz2' % frmt] = lambda obj: dumps(obj, format=None, compression='bz2',
                                                                            compresslevel=9)
                methods['format=None(%s)+zlib' % frmt] = lambda obj: dumps(obj, format=None, compression='zlib',
                                                                             compresslevel=9)
                methods['format=None(%s)+gzip' % frmt] = lambda obj: dumps(obj, format=None, compression='gzip',
                                                                             compresslevel=9)

                methods['format=None(%s, gzip)' % frmt] = lambda obj: dumps(obj), lambda obj: loads(obj)

                for methodname in methods:
                    dumpsfunc = methods[methodname]
                    if not hasattr(dumpsfunc, "__len__"):
                        loadsfunc = loads
                    else:
                        loadsfunc = dumpsfunc[1]
                        dumpsfunc = dumpsfunc[0]
                    try:
                        start_d = time.time()
                        bytestr = dumpsfunc(obj)
                        end_d = time.time()
                        start_l = time.time()
                        _ = loadsfunc(bytestr)
                        end_l = time.time()
                        # assert type(_) == type(obj)
                    except Exception as exc:
                        # _ = loads_test(bytestr)
                        print "%20s %40s ERROR: %s" % ("", methodname, str(exc))
                        return
                    ttt_d = end_d - start_d
                    ttt_l = end_l - start_l
                    ttt = ttt_d + ttt_l

                    method_perfs = np.array([len(bytestr), ttt_d, ttt_l, ttt])
                    if methodname not in performances:
                        performances[methodname] = method_perfs
                    else:
                        performances[methodname] += method_perfs
#                     with tempfile.NamedTemporaryFile(suffix='.zip') as t:
#                         names.append(t.name)
#                         assert type(bytestr) == str
#                         t.write(bytestr)
#                         filesize_kb_ = filesize_kb(t.name)
#                         space_saved = (filesize_kb_) * files / 1000000
#                         tdelta = timedelta(seconds=round(files*ttt))
#                         data.append(("", methodname, size_kb(bytestr), filesize_kb_, ttt, space_saved, tdelta))
    
            
        print "%30s %30s %30s %30s %30s %30s" % ("Data name",
                                             "Method",
                                             "size(Gb)",
                                             "dumps_time(HH:mm:ss)",
                                             "loads_time(HH:mm:ss)",
                                             "d+l_time(HH:mm:ss)")
        for dataname, performances in objects.iteritems():
            for method in performances:
                performances[method] = benchmark_filenum * performances[method] / max_traces_tries
                
            perf_list_of_tuples = performances.items()
            data = sorted(perf_list_of_tuples, key=lambda x: (int(x[1][0]/1000000)/1000.0, round(x[1][-1]))) # sort by size saved and then, if equal, by time taken (less time is better)
            
            print "%30s " % dataname.upper()
            for i, d in enumerate(data):
                # convert bytes to guga, and seconds to timedelta so that we have the correct format
                # (and round them to seconds)
                methodname = d[0]
                method_perfs = d[1]
                method_perfs = [int(method_perfs[0]/1000000)/1000.0, timedelta(seconds=round(method_perfs[1])),
                             timedelta(seconds=round(method_perfs[2])), timedelta(seconds=round(method_perfs[3]))]
                print "{:>30d} {:>30} {:>30.3f} {:>30} {:>30} {:>30}".format(i+1, d[0], *method_perfs)
                                                                               
#                 \
#                         ("", key, size_kb(bytestr), filesize_kb_, space_saved, ttt)
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
