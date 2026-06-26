"""
Module handling the Writers, i.e. classes handling the IO operation from the
processing function into a file
"""
from collections.abc import Iterable
# 22 May 2018
from pathlib import Path

import pandas as pd

hdf_file_extensions = ['.hdf', '.h5', '.hdf5']


def get_writer(
    output_file:str | Path | None, append=False, options: dict | None = None
):
    """
    Return the writer from the given outputfile (string denoting a file
    path, or None) and append flag (boolean)
    """
    if output_file is None:
        return BaseWriter(output_file, append)
    file_ext = Path(output_file).suffix.lower()
    if file_ext in hdf_file_extensions:
        return HDFWriter(output_file, append, options)
    return CsvWriter(output_file, append, options)


class BaseWriter:
    """Base Writer, No-op"""

    def __init__(self, output_file: str | Path | None=None, append=False, options=None):
        self.append = append
        self.output_file = Path(output_file).resolve() if output_file else None
        self.file_handle = None  # must have a close method
        self.options = {} if options is None else options

    def convert_to_df(
        self, result: dict | pd.Series | pd.DataFrame | list[dict | pd.Series]
    ):
        """Core function to write a processed segment result to the specified
        output
        """
        invalid = None
        if isinstance(result, pd.DataFrame):
            yield result
        elif isinstance(result, (dict, pd.Series)):
            yield pd.DataFrame([result])
        elif isinstance(result, (list, tuple, set)):
            for res in result:
                if isinstance(res, pd.DataFrame):
                    yield res
                elif isinstance(res, (dict, pd.Series)):
                    yield pd.DataFrame([res])
                else:
                    invalid = res
                    break
        else:
            invalid = result

        if invalid is not None:
            raise ValueError(f'Cannot write objects of type {type(invalid)}')

    def write(self, result: dict | pd.Series | pd.DataFrame | list[dict | pd.Series]):
        pass  # no-op

    def __enter__(self):
        """Opens file handler"""
        # subclasses might set a more meaningful value
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # @UnusedVariable
        """Close file handler"""
        self.close()

    def close(self):
        """"""
        if self.file_handle is None:
            return True
        try:
            self.file_handle.flush()
        except: # noqa
            pass
        try:
            self.file_handle.close()
        except: # noqa
            return False
        finally:
            self.file_handle = None
        return True

    def __str__(self):
        return f"{self.__class__.__name__}({self.output_file or '<no file>'})"


class CsvWriter(BaseWriter):
    """
    Class that can be used in a with statement writing each processed
    segments results into a csv file
    """

    def __init__(self, output_file, append, options=None):
        """
        Initialize a CSVWriter

        :param options: dict of optional keyword arguments to be passed
            to the writer (some are set in this class unless overwritten). For details,
            see <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html>
        """
        # call super as first call (mandatory):
        super(CsvWriter, self).__init__(output_file, append, options)
        # self.options.setdefault('sep', ',')
        # self.options.setdefault('quotechar', '"')
        # self.options.setdefault('quoting', csv.QUOTE_MINIMAL)
        self.options.pop('header', None)
        self.options.pop('index', None)
        self.writer = None

    def __enter__(self):
        self.file_handle = open(
            str(self.output_file),
            'a' if self.append else 'w',
            # buffering=10, # buffering=1: flush 10 lines
            encoding='utf-8',
            errors='replace',
            newline=''
        )
        return self

    def write(self, result: dict | pd.Series | pd.DataFrame | list[dict | pd.Series]):

        for res in self.convert_to_df(result):
            res.to_csv(
                self.file_handle,
                header=self.file_handle.tell() == 0,
                index=False,
                **self.options
            )


class HDFWriter(BaseWriter):
    """HDF Writer"""

    def __init__(self, output_file, append, options=None):
        """Initialize a HDFWriter

        :param options: dict of optional keyword arguments to be passed to:
            <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.HDFStore.append.html>
            (some are set in this class unless overwritten)
        """
        super(HDFWriter, self).__init__(output_file, append, options)
        # remove 'value' from options, it must be set by the user-defined
        # Python file:
        self.options.pop('value', None)
        # needs to overwrite append: this append is True because we write
        # in chunks, it is not the append above (which means open the storage
        # in write or append mode)
        self.options['append'] = True
        # set options defaults, if not given:
        self.options.setdefault('key', 's2s_table')
        self.options.setdefault('format', 'table')


    def __enter__(self):
        self.file_handle = pd.HDFStore(
            str(self.output_file), mode='a' if self.append else 'w'
        )
        return self

    def write(self, result: dict | pd.Series | pd.DataFrame | list[dict | pd.Series]):
        for res in self.convert_to_df(result):
            res.to_df(self.file_handle, **self.options)

