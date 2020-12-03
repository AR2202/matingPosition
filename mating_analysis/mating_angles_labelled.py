import pandas as pd
import numpy as np
import mating_angles_model2
from mating_angles_model2 import filtered_outputs, unfiltered_outputs, load_csv_file, tilting_index, tilting_index_all_frames


def mating_angles_labelled(path_to_csv, path_to_labels):
    """prepares training dataset"""
    labeltable = pd.read_csv(path_to_labels, header=0)
    nums_neg = []
    nums_pos = []
    angles_b, _, _, _, _ = unfiltered_outputs(path_to_csv)
    for i in range(0, len(labeltable[labeltable.keys()[1]]), 2):
        nums_neg = nums_neg+list(range(labeltable[labeltable.keys()[1]][i],
                                 labeltable[labeltable.keys()[1]][i+1]))
    for i in range(0, len(labeltable[labeltable.keys()[2]]), 2):
        nums_pos = nums_pos+list(range(labeltable[labeltable.keys()[2]][i],
                                 labeltable[labeltable.keys()[2]][i+1]))
    angles_b_pos = angles_b[nums_pos]
    angles_b_neg = angles_b[nums_neg]
    return angles_b_pos, angles_b_neg
