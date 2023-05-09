try:
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
except ImportError:
    nbformat = None
    # import pytest
    # pytest.mark.skip("Jupyter not installed, not testing notebooks correctness")

import pytest
import os
from stream2segment.process import get_segment_help
from stream2segment.resources import get_resource_abspath


# pytest.skip(allow_module_level=True)

def test_segment_help():
    """This test check that we did not add any new method or attribute to the Segment
    object without considering it in the doc (either make it hidden or visible)
    """
    get_segment_help()


@pytest.mark.skipif(nbformat is None,
                    reason="Jupyter not installed, not testing notebooks correctness")
def test_notebook(data):

    # cwd = os.getcwd()
    for fle_ in ['Using-Stream2segment-in-your-Python-code.ipynb',
                 'The-Segment-object.ipynb']:
        fle = get_resource_abspath('templates', fle_)
        with open(fle) as f:
            nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600)  # , kernel_name='python3')
        cwd = os.path.dirname(fle)
        ep.preprocess(nb, {'metadata': {'path': cwd}})

def test_imap(capsys):
    def my_processing_function(segment, config):
        """simple processing function. Take the segment stream and remove its instrumental response"""
        # Get ObsPy Trace object. If the waveform has no gapos/overlaps, the trace is the only element
        # of the segment stream object (otherwise the stream will have several traces):
        trace = segment.stream()[0]
        # remove the instrumental response of the Trace:
        # get ObsPy Inventory object:
        inventory = segment.inventory()
        # remove the response:
        trace_remresp = trace.remove_response(inventory)  # see caveat below
        # return the segment.id, the event magnitude, the original trace and the trace with response removed
        return segment.id, segment.event.magnitude, segment.stream()[0], trace_remresp

    # create the selection dict. This dict select a single segment (id=2) for illustrative purposes:
    segments_selection = {
        'has_data': 'true',
        'maxgap_numsamples': '[-0.5, 0.5]',
        'event_distance_deg': '[70, 80]'
        # other optional attributes (see cheatsheet below for details):
        # missing_data_sec: '<120'
        # missing_data_ratio: '<0.5'
        # id: '<300'
        # event.time: "(2014-01-01T00:00:00, 2014-12-31T23:59:59)"
        # event.latitude: "[24, 70]"
        # event.longitude: "[-11, 24]"
    }

    from stream2segment.process import SkipSegment
    def my_processing_function_raising(segment, config):
        if segment.sample_rate < 30:
            raise SkipSegment("segment sample rate too low")
        # ... implement your code here

    import os
    dbpath = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        'stream2segment', 'resources', 'templates', 'example.db.sqlite')
    dburl = 'sqlite:///' + dbpath

    from stream2segment.process import imap

    for (segment_id, mag, trace, trace_remresp) in imap(my_processing_function, dburl,
                                                        segments_selection):
        print()
        print('Segment Id: %d (event magnitude: %.1f)' % (segment_id, mag))
        print('Segment trace (first three points):')
        print('  - Counts units (no response removed):    %s' % trace.data[:3])
        print('  - Physical units (response removed):     %s' % trace_remresp.data[:3])