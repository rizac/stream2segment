"""
Module handling the Writers, i.e. classes handling the IO operation from the
processing function into a file

Created on 22 May 2018

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
import os
import csv
from future.utils import viewkeys
import pandas as pd

from stream2segment.io import open2writetext

HDF_FILE_EXTENSIONS = ['.hdf', '.h5', '.hdf5']
SEGMENT_ID_COLNAME = 'segment_db_id'
# previous versions had different SEGMENT_ID_COLNAMEs:
_SEGMENT_ID_COLNAMES = (SEGMENT_ID_COLNAME, 'Segment.db.id')  # order matters
HDF_DEFAULT_CHUNKSIZE = 10000


def get_writer(outputfile=None, append=False, options_dict=None):
    """Return the writer from the given outputfile (string denoting a file
    path, or None) and append flag (boolean)
    """
    if outputfile is None:
        return BaseWriter(outputfile, append)
    if os.path.splitext(os.path.basename(outputfile))[1].lower() in \
            HDF_FILE_EXTENSIONS:
        return HDFWriter(outputfile, append, options_dict)
    return CsvWriter(outputfile, append)


class BaseWriter(object):
    """Base class, basically no-op: it can be used in a with statement but it's
    basically no-op **IMPORTANT**: subclasses need to call super.__init__ !!!
    """

    _SEGID_NOTFOUND_ERR = TypeError('Cannot append to file, segment_id column '
                                    'name not found (was the file created with '
                                    'this program?)')

    def __init__(self, outputfile=None, append=False):
        self.append = append
        self.outputfile = os.path.abspath(outputfile)
        self.outputfilehandle = None
        # self._isbasewriter = self.__class__ is BaseWriter
        self._segment_id_colname = SEGMENT_ID_COLNAME
        # self._segment_id_colname could be retrieved from outputfile
        # but append must be True, outputfile must exist and be non empty:
        if append and self.outputfileexists and not self.outputfileempty:
            try:
                self._segment_id_colname = \
                    self.get_segment_id_colname(outputfile)
            except:
                self._segment_id_colname = None

    def get_segment_id_colname(self, outputfile):
        """Retrieve the segment id column name from `outputfile` (which must
        exist). Return None if such a column could not be found
        :param outputfile: a valid EXISTING file
        """
        return None

    @property
    def isbasewriter(self):
        """Property returning if this object is a base writer, i.e., no-op"""
        # return self._isbasewriter
        return self.__class__ is BaseWriter

    @property
    def outputfileexists(self):
        """Return True if the output file given in the constructor exists.
        Returns False in any other case
        """
        return self.outputfile and os.path.isfile(self.outputfile)

    @property
    def outputfileempty(self):
        """Return True if the output file exists and is empty"""
        return self.outputfileexists and os.stat(self.outputfile).st_size == 0

    def already_processed_segments(self):
        """Return a numpy array of UNIQUE integers denoting the the already
        processed segments
        """
        return []

    def write(self, segment_id, result):  # result is surely not None
        """Core function to write a processed segment result to the specified
        output
        """
        pass

    def __enter__(self):
        """Method executed at the beginning of a `with` clause.
        this method MUST be overridden and MUST set `self.outputfilehandle`
        != None
        """
        # subclasses might set a more meaningful value
        self.outputfilehandle = True

    def __exit__(self, exc_type, exc_val, exc_tb):  # @UnusedVariable
        """Method executed at the end of a `with` clause. Calls `self.close()`
        by default
        """
        self.close()

    def close(self):
        """Close `self.outputfilehandle`, which is the file object of this
        class, most likely set in `self.__enter__`. If `self.outputfilehandle`
        has not been set, or has no attribute `close()`, this method is no-op
        """
        try:
            self.outputfilehandle.close()
            return True
        except:  # @IgnorePep8 pylint: disable=bare-except
            return False
        finally:
            self.outputfilehandle = None

    def __str__(self):
        return "%s (output: %s)" % (self.__class__.__name__ ,
                                    os.path.abspath(self.outputfile) if self.outputfile
                                    else 'n/a')


class CsvWriter(BaseWriter):
    """Class that can be used in a with statement writing each processed
    segments results into a csv file
    """

    _SEGID_NOTFOUND_ERR = TypeError(str(BaseWriter._SEGID_NOTFOUND_ERR) +
                                    '. You can only append lists, not dicts')

    def __init__(self, outputfile, append):
        """Call super.__init__ (**mandatory**) and sets up class specific
        stuff
        """
        # call super as first call (mandatory):
        super(CsvWriter, self).__init__(outputfile, append)
        # Do not raise now if the segment id column name is None (i.e., if
        # append is True and we did not find the column in the file) because we
        # need to wait if dicts are appended (raise), or lists (goon)
        self.csvwriterkwargs = dict(delimiter=',', quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
        self.csvwriter = None
        self.csvwriterisdict = False

    def get_segment_id_colname(self, outputfile):
        """Retrieve the segment id column name from `outputfile` (which must
        exist). Return None if such a column could not be found
        :param outputfile: a valid EXISTING file
        """
        # read header automatically
        with open(outputfile, "r") as f:
            reader = csv.reader(f)
            for header in reader:
                segidcolname = str(header[0])
                if segidcolname in _SEGMENT_ID_COLNAMES:
                    return segidcolname
                break
        return None

    def already_processed_segments(self):
        """Return a numpy array of UNIQUE integers denoting the the already
        processed segments
        """
        return pd.unique(list(self._already_processed_segments_iter()))

    def _already_processed_segments_iter(self):
        with open(self.outputfile) as filep:
            reader = csv.reader(filep)
            for firstrow in reader:
                try:
                    yield int(firstrow[0])
                except ValueError as _:
                    # be relaxed about first row, it might be the header made
                    # of string columns:
                    pass
                # now read all other first-col values, they must be integers
                # this time:
                for row in reader:
                    yield int(row[0])

    def __enter__(self):
        # py2 compatibility of csv library: open in 'wb'. If py3, open in 'w'
        # mode. See utils module. buffering=1 flushes each line
        self.outputfilehandle = open2writetext(self.outputfile, buffering=1,
                                               encoding='utf8',
                                               errors='replace', newline='',
                                               append=self.append)

    def write(self, segment_id, result):  # result is surely not None
        csvwriter, isdict, seg_id_colname = \
            self.csvwriter, self.csvwriterisdict, self._segment_id_colname
        if csvwriter is None:  # instantiate writer according to first input
            isdict = self.csvwriterisdict = isinstance(result, dict)
            # write first column(s):
            if isdict:
                if seg_id_colname is None:
                    raise self._SEGID_NOTFOUND_ERR

                # we need to pass a list and not an iterable cause the iterable
                # needs to be consumed twice (the doc states differently,
                # however...):
                fieldnames = [seg_id_colname]
                fieldnames.extend(viewkeys(result))
                csvwriter = csv.DictWriter(self.outputfilehandle,
                                           fieldnames=fieldnames,
                                           **self.csvwriterkwargs)
                self.csvwriter = csvwriter
                # write header if we need it (file does not exists, append is
                # False, or file exist, append=True but file has no row):
                if not self.append or self.outputfileempty:
                    csvwriter.writeheader()
            else:
                csvwriter = self.csvwriter = csv.writer(self.outputfilehandle,
                                                        **self.csvwriterkwargs)

        if isdict:
            result[seg_id_colname] = segment_id
        else:
            # we might have numpy arrays, we should support variable types
            # (numeric, strings,..)
            res = [segment_id]
            res.extend(result)
            result = res

        csvwriter.writerow(result)


class HDFWriter(BaseWriter):
    """Class that can be used in a with statement writing each processed
    segments results into a HDF file
    """

    def __init__(self, outputfile, append, options_dict=None):
        """Call super.__init__ (**mandatory**) and sets up class specific
        stuff
        """
        # call super as first call (mandatory):
        super(HDFWriter, self).__init__(outputfile, append)
        # Raise now if the segment id column name is None (i.e., if append is
        # True and we did not find the column in the file):
        if self._segment_id_colname is None:
            raise self._SEGID_NOTFOUND_ERR
        self._dframeslist = []
        self.options = options_dict or {}
        # remove 'value' from options, it must be set by the user-defined
        # Python file:
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
        if self._segment_id_colname not in data_columns:
            data_columns += [self._segment_id_colname]
        self.options['data_columns'] = data_columns

    def get_segment_id_colname(self, outputfile):
        """Retrieve the segment id column name from `outputfile` (which must
        exist). Return None if such a column could not be found
        :param outputfile: a valid EXISTING file
        """
        columns = set(pd.read_hdf(outputfile, start=0, stop=0).columns)
        for c in _SEGMENT_ID_COLNAMES:
            if c in columns:
                return c
        return None

    def already_processed_segments(self):
        """Return a numpy array of UNIQUE integers denoting the the already
        processed segments
        """
        col = self._segment_id_colname
        ids = pd.read_hdf(self.outputfile, columns=[col])[col]
        return pd.unique(ids)

    def __enter__(self):
        # py2 compatibility of csv library: open in 'wb'. If py3, open in 'w'
        # mode. See utils module. buffering=1 flushes each line
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

        result[self._segment_id_colname] = segment_id
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
        """Close `self.outputfilehandle`, which is the file object of this
        class, most likely set in `self.__enter__`. If `self.outputfilehandle`
        has not been set, or has no attribute `close()`, this method is no-op
        """
        try:
            self._write(True)
        finally:
            self._dframeslist = []  # clear data, if any, and also help gc)
            BaseWriter.close(self)
