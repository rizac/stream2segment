"""This module holds doc strings to be injected via jinja2 into the templates when
running `s2s init`.
Any NON-PRIVATE variable name (i.e., without leading underscore '_') of this module
can be injected in a template file in the usual way, e.g.:
{{ PROCESS_PY_BANDPASSFUNC }}

.. moduleauthor:: Riccardo Zaccarelli <rizac@gfz-potsdam.de>
"""
from stream2segment.download.modules.utils import EVENTWS_MAPPING
# DO NOT REMOVE IMPORT BELOW, IT IS USED IN TEMPLATES:
from stream2segment.process.writers import SEGMENT_ID_COLNAME

_WIKI_BASE_URL = 'https://github.com/rizac/stream2segment/wiki'

USING_S2S_IN_YOUR_PYTHON_CODE_WIKI_URL = \
    _WIKI_BASE_URL + '/using-stream2segment-in-your-python-code'

THE_SEGMENT_OBJECT_WIKI_URL = _WIKI_BASE_URL + '/the-segment-object'

THE_SEGMENT_OBJECT_ATTRS_AND_METHS = \
  THE_SEGMENT_OBJECT_WIKI_URL + '#attributes-and-methods'

THE_SEGMENT_OBJECT_WIKI_URL_SEGMENT_SELECTION = \
    THE_SEGMENT_OBJECT_WIKI_URL + '#segments-selection'


PROCESS_PY_BANDPASSFUNC = """
Apply a pre-process on the given segment waveform by filtering the signal and
removing the instrumental response. 

This function is used for processing (see `main` function) and visualization
(see the `@gui.preprocess` decorator and its documentation above)

The steps performed are:
1. Sets the max frequency to 0.9 of the Nyquist frequency (sampling rate /2)
   (slightly less than Nyquist seems to avoid artifacts)
2. Offset removal (subtract the mean from the signal)
3. Tapering
4. Pad data with zeros at the END in order to accommodate the filter transient
5. Apply bandpass filter, where the lower frequency is magnitude dependent
6. Remove padded elements
7. Remove the instrumental response

IMPORTANT: This function modifies the segment stream in-place: further calls to 
`segment.stream()` will return the pre-processed stream. During visualization, this
is not an issue because Stream2segment always caches a copy of the raw trace.
During processing (see `main` function) you need to be more careful: if needed, you
can store the raw stream beforehand (`raw_trace=segment.stream().copy()`) or reload
the segment stream afterwards with `segment.stream(reload=True)`.

:return: a Trace object (a Stream is also valid value for functions decorated with
    `@gui.preprocess`)
"""


DOWNLOAD_EVENTWS_LIST = '\n'.join('%s"%s": %s' % ('# ' if i > 0 else '', str(k), str(v))
                                  for i, (k, v) in enumerate(EVENTWS_MAPPING.items()))

# setting up DOCVARS:
DOCVARS = {k: v.strip() for k, v in globals().items()
           if hasattr(v, 'strip') and not k.startswith('_')}
