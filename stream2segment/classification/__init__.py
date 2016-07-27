# from __future__ import unicode_literals
import pandas as pd
from stream2segment.s2sio.db import models


UNKNOWN_CLASS_ID = -1
"""the standard integer for unknown class id (as a result from the classifier or because
not annotated)"""
OTHER_CLASS_ID = -2
"""the global integer denoting instances not to be used for training (e.g., unknown artifacts)"""

# class labels data frame: note: ID is the primary key so it must be unique
class_labels_df = pd.DataFrame(
                               data=[
                                       [OTHER_CLASS_ID, 'Discarded',
                                        ('Segment which does not fall in any other cathegory '
                                         '(e.g., unknown artifacts, bad formats etcetera)')],
                                       [UNKNOWN_CLASS_ID, 'Unknown',
                                        ('Segment which is either: unlabeled (not annotated) '
                                         'or unclassified')],
                                       [0, 'Ok', 'Segment with no artifact'],
                                       [1, "LowS2N", "Segment has a low signal-to-noise ratio"],
                                       [2, "MultiEvent", ("Segment with overlapping "
                                                          "multi-events recorded")],
                                       [3, "BadCoda", ("Segment has a bad coda (bad decay)")],
                                       [4, 'Clipped', "Segment is clipped"],
                                       [5, 'InstrProblem', 'Segment has instrumental problems'],
                                       [6, "NoSigAll", "Signal is not present in all channels"],
                                       [7, "Gaps", "Signal with gaps"]
                                       ],
                               columns=[models.Class.id.key,
                                        models.Class.label.key,
                                        models.Class.description.key],
                              )
