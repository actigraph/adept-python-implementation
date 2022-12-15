"""Determine whether a segment is part of a walking bout."""
from copy import copy

import numpy as np
from numpy import typing as npt

from pyadept.sliding_functions import rolling_mean, rolling_diff1
from pyadept.segment_pattern import rank_chunks_of_ones


def segment_walking_bouts(
    spdesc_df,
    sim_MIN_stride=0.85,
    sim_MIM_bridge=0.75,
    dur_MIN=0.85,
    dur_MAX=1.4,
    ptp_r_MIN=0.4,
    ptp_r_MAX=3.0,
    vmc_r_MIN=0.1,
    vmc_r_MAX=0.8,
    mean_abs_diff_med_p_MAX=0.2,
    mean_abs_diff_med_t_MAX=0.2,
    mean_abs_diff_dur_MAX=0.25,
    keep_valid_walking_bout_only=True,
):
    """
    Determine whether a segment is part of a walking bout.

    Takes pandas.DataFrame with stride pattern
    segmentation results (output of `describe_segments` function after conversion
    to pandas.DataFrame) and for each stride identified,
    determines whether it is a part of walking instance / walking bout.

    Conceptual description of the procedure:
    Step 1: For each stride in pattern segmentaton results, determine if
    individual stride is a "VALID" STRIDE (takes filtering of its duration,
    vector magnitude values etx.)
    Step 2: Identify WALKING INSTANCES, that is, sequences of >=3 "valid"
    subsequent walking strides. It checking whether or not those subsequent
    walking strides are "similar" to each other within each walking instance.
    Step 3: Identify WALKING BOUTS, that is, sequences subsequent walking strides
    which are consisting og >= 1 WALKING INSTANCES. This step is to merge
    identified walking instances which are i.e. very close to each other and
    it is plausible to think they come from one walking action, but something
    in a middle of that walking action has missed the "valid" stride criteria.
    3 cases of conditions upon which the merging happens:
    (1) no break between two neighbouring walking instances,
    or 1-element break (bridge) between the two neighbouring walking instances
    (2) 2-element break (bridge) between the two neighbouring walking instances,
    where at least 1 element of the bridge is close to passing valid stride condition
    (3) 3-element break (bridge) between the two neighbouring walking instances,
    where at least 2 elements of the bridge are close to passing valid stride condition.

    Each identified walking instance receives `'walking_instance_id'`,
    and each identified walking bout receives `'walking_bout_id'`.

    The primary conceptual difference is that while I assume strides from same
    WALKING BOUT are coming from one walking bout i.e. for bout size computation,
    I only want to use strides
    from WALKING INSTANCE (that is, strides, which have `'walking_instance_id'`
    assigned) for any analytics purposes as those are which we determined as
    "VALID" STRIDES.

    :param spdesc_df: (pandas.DataFrame) Data frame with stride pattern
    segmentation results. Output of `describe_segments` function
    (after conversion to pandas.DataFrame). Data frame columns:
    "tau_idx",
    "dur_vl",
    "sim",
    "med_x",
    "med_y",
    "med_z",
    "med_p",
    "med_t",
    "med_r",
    "std_x",
    "std_y",
    "std_z",
    "std_p",
    "std_t",
    "std_r",
    "ptp_x",
    "ptp_y",
    "ptp_z",
    "ptp_p",
    "ptp_t",
    "ptp_r",
    "vmc_r",
    "dur".
    :param sim_MIN_stride: (float) Scalar. Step 1: valid strides filtering parameter.
    Determines minimum similarity value between predefined stride pattern and
    identified stride occurrence needed to consider it as an individual "valid"
    stride.
    :param sim_MIM_bridge: (float) Scalar. Step 3: merging into walking bout parameter.
    Determines minimum similarity value between predefined stride pattern and
    identified stride occurrence needed to consider a stride as a one which lies
    in-between two walking instances.
    :param dur_MIN: (float) Scalar. Step 1: valid strides filtering parameter.
    Determines minimum stride duration time needed to consider it as an individual
    "valid" stride. Expressed in seconds.
    :param dur_MAX: (float) Scalar. Step 1: valid strides filtering parameter.
    Determines maximum stride duration time needed to
    consider corresponding segmented stride as an individual "valid" stride.
    Expressed in seconds.
    :param ptp_r_MIN: (float) Scalar. Step 1: valid strides filtering parameter.
    Determines minimum peak-to-peak value for vector magnitude [r] needed to
    consider corresponding segmented stride as an individual "valid" stride.
    :param ptp_r_MAX: (float) Scalar. Step 1: valid strides filtering parameter.
    Determines maximum peak-to-peak value for vector magnitude [r] needed to
    consider corresponding segmented stride as an individual "valid" stride.
    :param vmc_r_MIN: (float) Scalar. Step 1: valid strides filtering parameter.
    Determines minimum vector magnitude count value needed to
    consider corresponding segmented stride as an individual "valid" stride.
    :param vmc_r_MAX: (float) Scalar. Step 1: valid strides filtering parameter.
    Determines maximum vector magnitude count value needed to
    consider corresponding segmented stride as an individual "valid" stride.
    :param mean_abs_diff_med_p_MAX: (float) Scalar. Step 2: merging into walking instance parameter.
    Maximum value of mean of absolute differences in Azimuth median for 3 subsequent
    valid strides needed to consider them as valid walking instance.
    :param mean_abs_diff_med_t_MAX: (float) Scalar. Step 2: merging into walking instance parameter.
    Maximum value of mean of absolute differences in Elevation median for 3 subsequent
    valid strides needed to consider them as valid walking instance.
    :param mean_abs_diff_dur_MAX: (float) Scalar. Step 2: merging into walking instance parameter.
    Maximum value of mean of absolute differences in duration time for 3 subsequent
    valid strides needed to consider them as valid walking instance.
    :param keep_valid_walking_bout_only: (boolean) Whether or not to return
    data frame with strids which are a part of any valid walking bout, or
    return the whole imput table of all stride pattern segmentation results
    (with additional labels of whether or not a segmented stride is a part of any
    walking instance and walking bout).
    :return: (pandas.DataFrame) Data frame with stride pattern
    segmentation results (output of `describe_segments` function) extended
    by two additional columns: `'walking_instance_id'` and `'walking_bout_id'`.
    """
    # Define condition for a segmented pattern to be a potential WALKING STRIDE
    cond_stride_0 = (
        (spdesc_df["ptp_r"] >= ptp_r_MIN)
        & (spdesc_df["ptp_r"] <= ptp_r_MAX)
        & (spdesc_df["vmc_r"] >= vmc_r_MIN)
        & (spdesc_df["vmc_r"] <= vmc_r_MAX)
        & (spdesc_df["dur"] >= dur_MIN)
        & (spdesc_df["dur"] <= dur_MAX)
    )
    cond_stride: npt.NDArray[np.int_] = np.array(
        cond_stride_0 & (spdesc_df["sim"] >= sim_MIN_stride)
    ).astype("int")
    cond_stride_bridge: npt.NDArray[np.int_] = np.array(
        cond_stride_0 & (spdesc_df["sim"] >= sim_MIM_bridge)
    ).astype("int")

    # Define conditions for a potential WALKING STRIDE to be a part of a WALKING INSTANCE
    cond_stride_mean3 = rolling_mean(np.array(cond_stride), w_vl=3, undef_val=0) > 0.9
    med_p_diff1 = np.abs(rolling_diff1(np.array(spdesc_df["med_p"]), undef_val=0))
    med_t_diff1 = np.abs(rolling_diff1(np.array(spdesc_df["med_t"]), undef_val=0))
    dur_diff1 = np.abs(rolling_diff1(np.array(spdesc_df["dur"]), undef_val=0))
    cond_med_p_diff1_mean2 = (
        rolling_mean(med_p_diff1, w_vl=2, undef_val=mean_abs_diff_med_p_MAX)
        < mean_abs_diff_med_p_MAX
    )
    cond_med_t_diff1_mean2 = (
        rolling_mean(med_t_diff1, w_vl=2, undef_val=mean_abs_diff_med_t_MAX)
        < mean_abs_diff_med_t_MAX
    )
    cond_dur_diff1_mean2 = (
        rolling_mean(dur_diff1, w_vl=2, undef_val=mean_abs_diff_dur_MAX)
        < mean_abs_diff_dur_MAX
    )
    cond_wi = (
        cond_stride_mean3
        & cond_med_p_diff1_mean2
        & cond_med_t_diff1_mean2
        & cond_dur_diff1_mean2
    )

    # Define walking instance (wi) ID
    wi_id0 = rank_chunks_of_ones(cond_wi)
    wi_id_concat: npt.NDArray[np.float_] = np.column_stack(
        (wi_id0, np.roll(wi_id0, 1), np.roll(wi_id0, 2))
    )
    wi_id_concat[np.isnan(wi_id_concat)] = -1
    wi_id = np.nanmax(wi_id_concat, axis=1)
    wi_id[wi_id < 0] = np.nan
    # Define walking bout (wb) ID
    wi_id_CPY = copy(wi_id)
    wb_id = copy(wi_id)
    wi_id_unq = np.sort(np.unique(wi_id[~np.isnan(wi_id)]))
    for i in range(len(wi_id_unq) - 1):
        # Current walking instance
        id_unq = wi_id_unq[i]
        id_unq_idxMAX = int(np.max(np.where(wi_id_CPY == id_unq)[0]))
        # Next walking instance
        id_unq_next = wi_id_unq[i + 1]
        id_unq_next_idx = np.where(wi_id_CPY == id_unq_next)[0]
        id_unq_next_idxMIN = int(np.min(id_unq_next_idx))
        id_unq_next_idxMAX = int(np.max(id_unq_next_idx))
        # no break or 1-element break between the two neighbouring walking instances
        bridge_len = id_unq_next_idxMIN - id_unq_idxMAX - 1
        cond_join_1 = bridge_len <= 1
        # 2-element break break between the two neighbouring walking instances
        cond_join_2 = (bridge_len == 2) & (
                np.mean(cond_stride_bridge[(id_unq_idxMAX + 1): id_unq_next_idxMIN]) > 0
        )
        # 3-element break break between the two neighbouring walking instances
        cond_join_3 = (bridge_len == 3) & (
            np.mean(cond_stride_bridge[(id_unq_idxMAX + 1): id_unq_next_idxMIN])
            > 0.5
        )
        # If any of the conditions is matched, join the two neighbouring walking
        # instances onto one walking bout
        if cond_join_1 | cond_join_2 | cond_join_3:
            # Update walking bouts
            wb_id[id_unq_idxMAX : (id_unq_next_idxMAX + 1)] = id_unq
            # Update walking instances
            wi_id_CPY[id_unq_next_idxMIN : (id_unq_next_idxMAX + 1)] = id_unq
            wi_id_unq[i + 1] = wi_id_unq[i]

    # Assign columns to pandas data frame
    spdesc_df["walking_instance_id"] = wi_id
    spdesc_df["walking_bout_id"] = wb_id
    # Keep only those for which walking bout is not np.nan
    if keep_valid_walking_bout_only:
        spdesc_df = spdesc_df[~np.isnan(spdesc_df["walking_bout_id"])]
    # Return Pandas dataframe
    return spdesc_df
