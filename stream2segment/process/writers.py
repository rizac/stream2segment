'''
Module handling the Writers, i.e. classes handling the IO operation from the
processing function into a file

Created on 22 May 2018

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
import os
import csv
from future.utils import viewkeys
import pandas as pd

from stream2segment.utils import open2writetext

HDF_FILE_EXTENSIONS = ['.hdf', '.h5', '.hdf5']
SEGMENT_ID_COLNAME = 'Segment.db.id'
HDF_DEFAULT_CHUNKSIZE = 100000


def get_writer(outputfile=None, append=False, options_dict=None):
    '''Returns the writer from the given outputfile (string denoting a file path, or None)
    and append flag (boolean)'''
    if outputfile is None:
        return BaseWriter(outputfile, append)
    if os.path.splitext(os.path.basename(outputfile))[1].lower() in \
            HDF_FILE_EXTENSIONS:
        return HDFWriter(outputfile, append, options_dict)
    return CsvWriter(outputfile, append)


class BaseWriter(object):
    '''Base class, basically no-op: it can be used in a with statement but it's basically no-op
    **IMPORTANT**: subclasses need to call super.__init__ !!!
    '''

    def __init__(self, outputfile=None, append=False):
        self.append = append
        self.outputfile = outputfile
        self.outputfilehandle = None
        self._isbasewriter = self.__class__ is BaseWriter

    @property
    def isbasewriter(self):
        '''property returning if this object is a base writer, i.e., no-op'''
        return self._isbasewriter

    @property
    def outputfileexists(self):
        '''Returns True if the output file given in the constructor exists. Returns False in
        any other case'''
        return self.outputfile and os.path.isfile(self.outputfile)

    @property
    def outputfileempty(self):
        '''Returns True if the output file exists and is empty'''
        return self.outputfileexists and os.stat(self.outputfile).st_size == 0

    def already_processed_segments(self, aslist=False):
        '''returns an iterable of integers denoting the already processed segments.
        It must return a *list of Python integers* if `aslist=True` (False by default).
        This method will probably open the internal file handle for reading,
        consider that it might be already opened for writing (we omit doing this
        check and raising for the moment)

        :param aslist: if True, this method must return a list of **Python integers**
            (no other types, e.g. numpy.int64s, and so on)
        '''
        return []

    def write(self, segment_id, result):  # result is surely not None
        '''Core function to write a processed segment result to the specified output'''
        pass

    def __enter__(self):
        '''method executed at the beginning of a `with` clause.
        this method MUST be overridden and MUST set `self.outputfilehandle`
        != None
        '''
        self.outputfilehandle = True  # subclasses might set a more meaningful value

    def __exit__(self, exc_type, exc_val, exc_tb):  # @UnusedVariable
        '''method executed at the end of a `with` clause. Calls `self.close()` by default'''
        self.close()

    def close(self):
        '''closes `self.outputfilehandle`, which is the file object of this class, most likely
        set in `self.__enter__`. If `self.outputfilehandle` has not been set, or has no attribute
        `close()`, this method is no-op'''
        try:
            self.outputfilehandle.close()
            return True
        except:  # @IgnorePep8 pylint: disable=bare-except
            return False
        finally:
            self.outputfilehandle = None


class CsvWriter(BaseWriter):
    '''Class that can be used in a with statement writing each processed segments results
    into a csv file'''

    def __init__(self, outputfile, append):
        '''calls super.__init__ (**mandatory**) and sets up class specific stuff'''
        super(CsvWriter, self).__init__(outputfile, append)
        self.csvwriterkwargs = dict(delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        self.csvwriter = None  # bad hack: in python3, we might use 'nonlocal' @UnusedVariable
        self.csvwriterisdict = False

    def already_processed_segments(self, aslist=False):
        ret = self._already_processed_segments_iter()
        return list(ret) if aslist else ret

    def _already_processed_segments_iter(self):
        with open(self.outputfile) as filep:
            reader = csv.reader(filep)
            for firstrow in reader:
                try:
                    yield int(firstrow[0])
                except ValueError as _:
                    # be relaxed about first row, it might be the header made of string columns:
                    pass
                # now read all other first-col values, they must be integers this time:
                for row in reader:
                    yield int(row[0])

    def __enter__(self):
        # py2 compatibility of csv library: open in 'wb'. If py3, open in 'w' mode:
        # See utils module. buffering=1 flushes each line
        self.outputfilehandle = open2writetext(self.outputfile, buffering=1, encoding='utf8',
                                               errors='replace', newline='', append=self.append)

    def write(self, segment_id, result):  # result is surely not None
        csvwriter, isdict, seg_id_colname = \
            self.csvwriter, self.csvwriterisdict, SEGMENT_ID_COLNAME
        if csvwriter is None:  # instantiate writer according to first input
            isdict = self.csvwriterisdict = isinstance(result, dict)
            # write first column(s):
            if isdict:
                # we need to pass a list and not an iterable cause the iterable needs
                # to be consumed twice (the doc states differently, however...):
                fieldnames = [seg_id_colname]
                fieldnames.extend(viewkeys(result))
                csvwriter = self.csvwriter = csv.DictWriter(self.outputfilehandle,
                                                            fieldnames=fieldnames,
                                                            **self.csvwriterkwargs)
                # write header if we need it (file does not exists, append is False, or
                # file exist, append=True but file has no row):
                if not self.append or self.outputfileempty:
                    csvwriter.writeheader()
            else:
                csvwriter = self.csvwriter = csv.writer(self.outputfilehandle,
                                                        **self.csvwriterkwargs)

        if isdict:
            result[seg_id_colname] = segment_id
        else:
            # we might have numpy arrays, we should support variable types (numeric, strings,..)
            res = [segment_id]
            res.extend(result)
            result = res

        csvwriter.writerow(result)


class HDFWriter(BaseWriter):
    '''Class that can be used in a with statement writing each processed segments results
    into a HDF file'''

    def __init__(self, outputfile, append, options_dict=None):
        '''calls super.__init__ (**mandatory**) and sets up class specific stuff'''
        super(HDFWriter, self).__init__(outputfile, append)
        self._dframeslist = []
        self.options = options_dict or {}
        # remove 'value' from options, it must be set by the user-defined python file:
        self.options.pop('value', None)
        # pop the chunksize from options, if any, because we handle here
        # the chunksize:
        self.chunksize = self.options.pop('chunksize', HDF_DEFAULT_CHUNKSIZE)
        # needs to overwrite append: this append is True because we write
        # in chunks, it is not the append above (which means open the storage
        # in write or append mode)
        self.options['append'] = True
        # set options defaults, if not given:
        self.options.setdefault('key', 's2s_table')
        self.options.setdefault('format', 'table')
        # data_columns set the indexed columns: set also the
        # SEGMENT_ID_COLNAME column which we insert automatically:
        data_columns = self.options.get('data_columns', [])
        if SEGMENT_ID_COLNAME not in data_columns:
            data_columns += [SEGMENT_ID_COLNAME]
        self.options['data_columns'] = data_columns

    def already_processed_segments(self, aslist=False):
        '''Returns the segment ids which should not be processed because already in the
        output file'''
        outputfile = self.outputfile
        ret = pd.read_hdf(outputfile, columns=[SEGMENT_ID_COLNAME])[SEGMENT_ID_COLNAME]
        if aslist:
            return ret.tolist()
        return ret

    def __enter__(self):
        # py2 compatibility of csv library: open in 'wb'. If py3, open in 'w' mode:
        # See utils module. buffering=1 flushes each line
        self.outputfilehandle = pd.HDFStore(self.outputfile,
                                            mode='a' if self.append else 'w')

    def write(self, segment_id, result):  # result is surely not None
        dframelist = self._dframeslist
        if isinstance(result, list):
            # convert to dict with integer keys, emulating pandas.
            # Maybe inefficient, but we need to add the segment id later
            result = {i: k for i, k in enumerate(result)}

        if isinstance(result, (dict, pd.Series)):
            result = pd.DataFrame([result])

        result[SEGMENT_ID_COLNAME] = segment_id
        dframelist.append(result)

        self._write()

    def _write(self, force=False):
        dframelist = self._dframeslist
        if dframelist and (len(dframelist) >= self.chunksize or force):
            dfr = pd.concat(dframelist, axis=0, sort=False, ignore_index=True,
                            verify_integrity=False)
            self.outputfilehandle.append(value=dfr,
                                         chunksize=len(dfr),
                                         **self.options)
            self._dframeslist = []

    def close(self):
        '''closes `self.outputfilehandle`, which is the file object of this class, most likely
        set in `self.__enter__`. If `self.outputfilehandle` has not been set, or has no attribute
        `close()`, this method is no-op'''
        try:
            self._write(True)
        except:
            pass
        self._dframeslist = []  # clear data, if any, and also help gc)
        return BaseWriter.close(self)
