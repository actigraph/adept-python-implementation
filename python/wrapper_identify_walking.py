import os
import sys

import numpy as np
from actigrapy.actigraphy_data.activity_data import RawActivityData

# The commented below I (@MK) used when ran the script from console level.
# exec(open(os.path.join('sliding_functions.py')).read())
# exec(open(os.path.join('segment_pattern_utils.py')).read())
# exec(open(os.path.join('segment_pattern.py')).read())
# exec(open(os.path.join('describe_segments.py')).read())

## Hard-coded (thought to be always fixed for it) segmentation params,
## likely to be read from some external params file later.
x_slice_vl = 6000
x_sim_smooth_w = 0.2
ftune = True
x_ftune_nbh_w = 0.6
x_ftune_smooth_w = np.nan
pattern_dur_grid = np.linspace(0.5, 4, 30, endpoint=True)

def wrapper_identify_walking(
        hdf5_fpath,
        templ_fpath,
        results_partial_dir):
    '''
    Function to run the first, most consuming part of walking identification algorithm:
    `segment_pattern` and `describe_segments` parts (`segment_walking_bouts` is not done).

    It process the whole HDF5 data for one patient.
    It saves the results of this part of walking identification algorithm as
    "partial results" that is, subject- and dad24-specific numpy.array files.

    Identifying walking bouts based on output of this file takes ~1s per
    subject- and dad24-specific numpy.array file. There is potential
    value (use case) in storing the pattern segmentation results first,
    hence this is what this wrapper does.

    :param hdf5_fpath: Path to subject-specific HDF5 file.
    :param templ_fpath: Path to CSV file containing precmputed stride pattern
    templates. The current location on the cluster is:
    "/home/karasma6/PROJECTS/walking_segmentation/data/stride_templates.csv"
    :param results_partial_dir: Path to directory where "partial results" are
    saved as subject- and dad24-specific numpy.array files.
    :return: none
    '''
    ## Get wrist stride pattern template
    # templ_fpath = os.path.join(project_dir, 'data', 'stride_templates.csv')
    templ_array = stride_templ_df_to_array(
        templ_fpath,
        sensor_location_id_val="left_wrist",
        collection_id_val="size_3")
    print("shape templ_array: " + str(templ_array.shape))

    ## Define prefix of partial results file names
    results_partial_fname_prefix = os.path.basename(hdf5_fpath).split(".")[0]

    ## Read HDSF5 file
    raw_activity_data = RawActivityData(hdf5_fpath)
    print("raw_activity_data DONE")
    ## Get number of dad24
    dad24_cnt = raw_activity_data.get_dad_count()
    ## Get sampling frequency
    data_fs = int(raw_activity_data.get_sampling_frequency())

    for dad24_i in range(0, dad24_cnt):
        try:
            ## Pull raw data
            [data_xyzptr, data_time] = raw_activity_data.get_data(
                do_correct=False,
                what="xyzptr",
                when={"type": "dad24", "number": [dad24_i]})

            ## Run pattern segmentation algorithm
            sp_out = segment_pattern_wrap(
                x_slice_vl=x_slice_vl,
                x=copy(data_xyzptr[:, 5]),
                x_fs=data_fs,
                templ_array=templ_array,
                pattern_dur_grid=pattern_dur_grid,
                x_sim_smooth_w=x_sim_smooth_w,
                ftune=ftune,
                x_ftune_nbh_w=x_ftune_nbh_w,
                x_ftune_smooth_w=x_ftune_smooth_w
            )

            ## Describe segments
            spdesc_out = describe_segments(sp_out, data_xyzptr, data_fs)

            ## Save results to file
            results_partial_fname = results_partial_fname_prefix + "_dad24_" + str(dad24_i) + '.npy'
            results_partial_fpath = os.path.join(results_partial_dir, results_partial_fname)
            np.save(results_partial_fpath, np.array(spdesc_out))
            print("Saved results to: " + results_partial_fpath)

        except Exception as e:
            print('Error:' + str(e))
            continue


if __name__ == "__main__":
    wrapper_identify_walking(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
