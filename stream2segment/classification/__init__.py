import pandas as pd


UNKNOWN_CLASS_ID = -1
"""the standard integer for unknown class id (as a result from the classifier or because
not annotated)"""
DISCARDED_CLASS_ID = -2
"""the global integer denoting instances not to be used for training (e.g., unknown artifacts)"""

# class labels data frame: note: ID is the primary key so it must be unique
class_labels_df = pd.DataFrame(
                               data=[
                                       [DISCARDED_CLASS_ID, 'Discarded',
                                        ('Segment to be discarded from any processing '
                                         '(e.g., unknown artifacts, bad formats etcetera)')],
                                       [UNKNOWN_CLASS_ID, 'Unknown',
                                        ('Segment which is either: not annotated or unclassified')],
                                       [0, 'Ok', 'Segment with no artifact ("standard" or "normal" in plain english)'],
                                       [1, 'Clipped', "Segments is clipped"],
                                       [2, 'InstrProblem', 'Segments has instrumental problems'],
                                       [3, "LowS2N", "Segment has a low signal-to-noise ratio"],
                                       [4, "MultiEvent", "Overlapping multi-events recorded"],
                                       [5, "NoSigAll", "Signal is not present in all channels"],
                                       [6, "Gaps", "Signal with gaps"]
                                       ],
                               columns=["Id", "Label", "Description"],
                              )
