'''
Module handling the Writers, i.e. classes handling the IO operation from the
processing function into a file

Created on 22 May 2018

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
'''
import os
import csv
from stream2segment.io.db.models import Segment
from stream2segment.utils import open2writetext
from future.utils import viewkeys


def get_writer(outputfile=None, append=False):
    '''Returns the writer from the given outputfile (string denoting a file path, or None)
    and append flag (boolean)'''
    return BaseWriter(outputfile, append) if outputfile is None else CsvWriter(outputfile, append)


class BaseWriter(object):
    '''Base class, basically no-op: it can be used in a with statement but it's basically no-op
    **IMPORTANT**: subclasses need to call super.__init__ !!!
    '''

    def __init__(self, outputfile=None, append=False):
        self.append = append
        self.outputfile = outputfile
        self.seg_id_attname = Segment.id.key
        self.seg_id_colname = "Segment.db.%s" % self.seg_id_attname
        self.outputfilep = None
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

    @property
    def already_processed_segments(self):
        '''returns an iterable of integers denoting the already processed segments. The iterable
        is empty if `append` (passed in the constructor) is False.
        THIS METHOD SHOULD NOT BE OVERRIDEN, use `self.already_processed_segments_iter`
        instead
        '''
        return [] if not self.append or not self.outputfileexists else \
            self.already_processed_segments_iter(self.outputfile)

    def already_processed_segments_iter(self, outputfile):
        '''Returns an iterator of integers denoting already processed files which have to be skipped
        :param outputfile: the output file passed in the constructor, IT DENOTES AN EXISTING FILE'''
        return []

    def __call__(self, segment_id, result):  # result is surely not None
        '''Core function to write a processed segment result to the specified output'''
        pass

    def __enter__(self):
        '''method executed at the beginning of a `with` clause. If an output file has to be opened,
        the resulting file object should be set as `self.outputfilep`, so that
        `self.outputfilep.close()` will be invoked in `__exit__`'''
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):  # @UnusedVariable
        '''method executed at the end of a `with` clause. Calls `self.close()` by default'''
        self.close()

    def close(self):
        '''closes `self.outputfilep`, which is the file object of this class, most likely
        set in `self.__enter__`. If `self.outputfilep` has not been set, or has no attribute
        `close()`, this method is no-op'''
        try:
            self.outputfilep.close()
            return True
        except:
            pass
        return False


class CsvWriter(BaseWriter):
    '''Class that can be used in a with statement writing each processed segments results
    into a csv file'''

    def __init__(self, outputfile, append):
        '''calls super.__init__ (**mandatory**) and sets up class specific stuff'''
        super(CsvWriter, self).__init__(outputfile, append)
        self.csvwriterkwargs = dict(delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        self.csvwriter = None  # bad hack: in python3, we might use 'nonlocal' @UnusedVariable
        self.csvwriterisdict = False

    def already_processed_segments_iter(self, outputfile):
        '''Returns the segment ids which should not be processed because already in the
        output file'''
        with open(outputfile) as filep:
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
        self.outputfilep = open2writetext(self.outputfile, buffering=1, encoding='utf8',
                                          errors='replace', newline='', append=self.append)

    def __call__(self, segment_id, result):  # result is surely not None
        csvwriter, isdict, seg_id_colname = \
            self.csvwriter, self.csvwriterisdict, self.seg_id_colname
        if csvwriter is None:  # instantiate writer according to first input
            isdict = self.csvwriterisdict = isinstance(result, dict)
            # write first column(s):
            if isdict:
                # we need to pass a list and not an iterable cause the iterable needs
                # to be consumed twice (the doc states differently, however...):
                fieldnames = [seg_id_colname]
                fieldnames.extend(viewkeys(result))
                csvwriter = self.csvwriter = csv.DictWriter(self.outputfilep, fieldnames=fieldnames,
                                                            **self.csvwriterkwargs)
                # write header if we need it (file does not exists, append is False, or
                # file exist, append=True but file has no row):
                if not self.append or self.outputfileempty:
                    csvwriter.writeheader()
            else:
                csvwriter = self.csvwriter = csv.writer(self.outputfilep, **self.csvwriterkwargs)

        if isdict:
            result[seg_id_colname] = segment_id
        else:
            # we might have numpy arrays, we should support variable types (numeric, strings,..)
            res = [segment_id]
            res.extend(result)
            result = res

        csvwriter.writerow(result)
