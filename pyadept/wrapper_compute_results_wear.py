"""Computes information on wear time.

Wear time is defined as corresponding to time of stride start for each element of
`segment_pattern` function output.

The wear time information is NOT used for walking strides/bouts identificaton,
as I (@MK) assumed the waling bouts identification algorithm works in a way
it will return info there is no walking for the non-wear time (b/c of, i.e.,
vector magnitude filetering params which set some minimal threshold that
non-wear data wound not pass).

Instead, this script is used to compute wear data for the vizualization purposes,
that is, for computing heatmap matrices used to show per-subject walking
distribution across days of acc collection for Dublin study.
(Mea culpa for params hard-coded on the top of the file.)
"""
import os
import sys
from typing import List

import numpy as np
from actigrapy.algorithms.wear_detection.wear_data.wear_data import WearData
from numpy import typing as npt

# This script contains function which computes information on wear time
#
# PARAMS

project_dir_path = "/home/karasma6/PROJECTS/walking_segmentation"
data_CP_dir_path = "/CHBS/data/digital_rwe/data_CP"
# Directory of pattern segmentation results, per-subject and per-dad files
results_partial_dir = os.path.join(
    project_dir_path, "data_results/20190805_DDV001A2302/results_partial"
)
# Desired directory of device wear information that corresponds to
# pattern segmentation results
results_wear_dir = os.path.join(
    project_dir_path, "data_results/20190805_DDV001A2302/results_wear"
)
# Path to directory with Dublin wear data HDF5 files
dublin_hdf5_wear_dir = os.path.join(
    data_CP_dir_path,
    "AnalysisOutput/DDV001/A2302/ggir_wearDetection/51ac5b63f48302fc0ca02e99066ed740",
)


def wrapper_compute_results_wear(subj_fname_prefix):
    """Compute information on wear time."""
    # List of parial results (pattern segmentation results) for the patient
    res_fname_list = [
        val for val in os.listdir(results_partial_dir) if subj_fname_prefix in val
    ]

    for res_fname in res_fname_list:
        print("res_fname: " + res_fname)

        try:
            res_f = np.load(os.path.join(results_partial_dir, res_fname))
            res_f_dad = int((res_fname.split(".")[0]).split("_")[3])

            # Path to file with HDF5 wearDetection data
            hdf5_wear_fname = (
                "DDV001A2302" + "_" + res_fname.split("_")[1] + ".wearDetection"
            )
            hdf5_wear_fpath = os.path.join(dublin_hdf5_wear_dir, hdf5_wear_fname)

            # Pull wear data
            n_samples = int(np.max(res_f[:, 0])) + 1
            wear_data = WearData(hdf5_wear_fpath)
            [daily_wear, time] = wear_data.get_data(
                what="wear",
                when={"type": "dad24", "number": [res_f_dad]},
                output_frequency=30,
                n_samples=n_samples,
            )
            # Subset wear data information to what is needed
            is_worn: List[int] = [int(daily_wear[int(val)]) for val in res_f[:, 0]]
            is_worn_arr: npt.NDArray[np.int_] = np.array(is_worn)
            print("len(is_worn_arr) = " + str(len(is_worn_arr)))

            # Save array to file
            np.save(os.path.join(results_wear_dir, res_fname), is_worn_arr)

        except Exception as e:
            print("Error: " + str(e))
            continue


if __name__ == "__main__":
    wrapper_compute_results_wear(sys.argv[1])
