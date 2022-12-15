"""Core algorithm for stride pattern segmentation."""
from copy import copy
from typing import List

import numpy as np
import pandas as pd
from numpy import typing as npt

from pyadept.sliding_functions import rolling_corr, rolling_smooth


def dstack_product(x, y):
    """Compute a 2D array whose rows consists of combination of all pairs of elements.

    :param x: (numpy.array) An array.
    :param y: (numpy.array) An array.
    :return: A 2D array.
    """
    return np.dstack(np.meshgrid(x, y)).reshape(-1, 2)


def interp_vec(vec, vl_out):
    """Interpolate array linearly to have certain length.

    :param vec: (numpy.ndarray) An array.
    :param vl_out: (int) A scalar. Number of elements in resulted array.
    :return: (numpy.ndarray) An array.
    """
    x = np.linspace(0, 1, len(vec))
    y = vec
    xvals = np.linspace(0, 1, vl_out)
    yinterp = np.interp(xvals, x, y)
    return yinterp


def interp_scale_vec(vec, vl_out):
    """Interpolate array to have certain length.

    The values are then normalized to have mean 0, variance 1.

    A multiplier sqrt((vl_out-1) / vl_out) is used in the scaling
    function so as the resulted array of values has variance equal 1
    according to R convention.

    :param vec: (numpy.ndarray) An array.
    :param vl_out: (int) A numeric scalar. Number of elements in resulted array.
    :return: (numpy.ndarray) An array.
    """
    x = np.linspace(0, 1, len(vec))
    y = vec
    xvals = np.linspace(0, 1, vl_out)
    yinterp = np.interp(xvals, x, y)
    yinterps = (
        (yinterp - np.mean(yinterp))
        * (1 / np.std(yinterp))
        * np.sqrt((vl_out - 1) / vl_out)
    )
    return yinterps


def stride_templ_df_to_array(
    df_path, sensor_location_id_val="left_wrist", collection_id_val="size_3"
):
    """
    Read precomputed stride pattern templates from CSV file to 2D numpy array.

    For detailed description of origin and content of the precomputed stride
    pattern templates, see `stride_template` description in R adeptdata package
    documentation:
    https://cran.r-project.org/web/packages/adeptdata/adeptdata.pdf.

    :param templ_df_path: (str) String scalar. Absolute path to data frame with
    precomputed stride pattern templates.
    :param sensor_location_id_val: (str) String scalar. Sensor location for
    which the pattern templates are to be returned. One of:
    'left_wrist', 'left_hip', 'left_ankle', 'right_ankle'.

    :param collection_id_val: (str) String scalar. Denotes number of unique
    location-specific pattern templates which are to be returned.  One of:
    'size_1', 'size_2', 'size_3', 'size_4', 'size_5'.
    :return: (numpy.ndarray) 2D array with stride pattern templates. Each row
    of the array corresponds to one distinct pattern template.
    """
    # Read CSV
    templ_df = pd.read_csv(df_path)
    templ_df_sub = templ_df[
        (templ_df["sensor_location_id"] == sensor_location_id_val)
        & (templ_df["collection_id"] == collection_id_val)
    ]
    templ_df_cols_sub = [col for col in templ_df_sub.columns if "ph" in col]
    templ_df_sub = templ_df_sub[templ_df_cols_sub]
    if templ_df_sub.empty:
        raise ValueError(
            "Templates data frame is empty. "
            "Select correct `sensor_location_id_val` and/or `collection_id_val` values. "
            "Correct `sensor_location_id_val` values: 'left_wrist', 'left_hip', "
            "'left_ankle', 'right_ankle'. Correct `collection_id` values: 'size_1', "
            "'size_2', 'size_3', 'size_4', 'size_5'. "
        )
    templ_array = templ_df_sub.values
    return templ_array


def rank_chunks_of_ones(arr):
    """Label individual chunks of subsequent `1`'s in numpy.array.

    Reference:
    https://stackoverflow.com/questions/57573350/fast-python-ish-way-of-ranking-chunks-of-1s-in-numpy-array

    :param arr: (numpy.array) An array.
    :return: (numpy.array) An array which contains distinct labels (int scalars)
    for distinct individual chunks of subsequent `1`'s. The elements corresponding
    to places where `arr` does not have `1` value are set to numpy.nan.
    """
    m = np.r_[False, arr.astype(bool), False]
    idx = np.flatnonzero(m[:-1] != m[1:])
    ll = idx[1::2] - idx[::2]
    out: npt.NDArray[np.float_] = np.full(len(arr), np.nan, dtype=float)
    out[arr != 0] = np.repeat(np.arange(len(ll)), ll)
    return out


def do_ftune(tau_i, T_i, x_ftune, nbhwing_vl, interp_vl_min, interp_vl_max):
    """Finetune."""
    tau_1 = tau_i
    tau_2 = tau_1 + T_i - 1
    tau_1_nbh: npt.NDArray[np.float_] = np.array(
        range(max(0, tau_1 - nbhwing_vl), min(tau_2, tau_1 + nbhwing_vl + 1))
    )
    tau_2_nbh: npt.NDArray[np.float_] = np.array(
        range(max(0, tau_2 - nbhwing_vl), min(len(x_ftune), tau_2 + nbhwing_vl + 1))
    )
    # Mask to assure the potential updated pattern location corresponds to a
    # pattern which duration is within allowed pattern duration range
    tau_prod: npt.NDArray[np.float_] = dstack_product(tau_2_nbh, tau_1_nbh)
    tau_mask: npt.NDArray[np.int_] = tau_prod[:, 0] - tau_prod[:, 1] + 1
    tau_mask = (tau_mask >= interp_vl_min) & (tau_mask <= interp_vl_max)
    # Identify fine-tuned location which corresponds to signal magnitude peaks
    nbh_x_prod = dstack_product(x_ftune[tau_2_nbh], x_ftune[tau_1_nbh])
    # @MK: changed 2019-07-11
    # nbh_x_multip = (nbh_x_prod[:, 0] * nbh_x_prod[:, 1]) * tau_mask
    nbh_x_multip = nbh_x_prod[:, 0] * nbh_x_prod[:, 1]
    nbh_x_multip[~tau_mask] = np.nan
    amax = np.nanargmax(nbh_x_multip)
    tau_2_new, tau_1_new = tau_prod[amax, :]
    # Update x_ftune to mark location of newly identified pattern
    x_ftune[(tau_1_new + 1) : tau_2_new] = np.nan
    return [tau_1_new, tau_2_new - tau_1_new + 1, x_ftune]


def segment_pattern(
    x,
    x_fs,
    templ_array,
    pattern_dur_grid,
    x_sim_smooth_w=np.nan,
    ftune=False,
    x_ftune_nbh_w=np.nan,
    x_ftune_smooth_w=np.nan,
):
    """
    Segment stride patterns from a 1-dimensional data array `x`.

    Algorithm uses an array of pattern templates to iteratively identify in
    the array `x` occurrences of the pattern. Fine-tuning algorithm is available
    to exploit the fact that beginning and end of a walking stride is (in many
    use cases) characterized by peak of magnitude of vector magnitude array `x`,
    and hence local maxima detection allows to identify those beginning/end
    precisely.

    :param x: (numpy.ndarray) 1-dimensional array. For typical application of
    strides identification from accelerometry data, this would be [r], that is,
    vector magnitude of raw [x,y,z] data.
    :param x_fs: (float) A numeric scalar. Sampling frequency of `x`, expressed
    in Hz.
    :param templ_array: (numpy.ndarray) 2D array with stride pattern templates,
    where each distinct pattern takes 1 row of the 2D array.
    :param pattern_dur_grid: (list of float scalars) Grid of potential
    pattern durations (durations the algorithm searches within), expressed
    in seconds.
    :param x_sim_smooth_w: (float) Scalar. Width (expressed in seconds) of
    smoothing window used to smooth accelerometry data VM for
    similarity computation between VM and pattern template(s).
    :param ftune: (boolean) Scalar. Whether or not to use fine-tune procedure
    (see the description above).
    :param x_ftune_nbh_w: (float) Scalar. Width (expressed in seconds) of data
    window ("neighbourhood") centered at preliminarily identified location
    of stride start/end, where the search of local maxima happends in fine-tune
    procedure (see: param `ftune`).
    :param x_ftune_smooth_w: (float) Scalar. Width (expressed in seconds) of
    smoothing window used to smooth accelerometry data VM for fine-tune procedure.
    :return: List of lists that describes the pattern segmentation results.
    Each "inner" list is a 3-element-long list
    describing one individual segmented stride:
    0. index of `x` VM time-series, where is the start of identified stride,
    1. duration (expressed of number of indices of `x`) of the identified stride,
    2. similarity (correlation) between the identified stride and stride pattern
    used in segmentation.
    """
    # Compute potential pattern vector length values
    pattern_vl_grid0 = np.sort(
        np.unique((np.array(pattern_dur_grid) * x_fs).astype(int))
    )
    vl_grid_min = np.min(np.ceil(np.array(pattern_dur_grid) * x_fs).astype(int))
    vl_grid_max = np.max(np.floor(np.array(pattern_dur_grid) * x_fs).astype(int))
    pattern_vl_grid = pattern_vl_grid0[
        (pattern_vl_grid0 >= vl_grid_min) & (pattern_vl_grid0 <= vl_grid_max)
    ]
    vl_grid_n = len(pattern_vl_grid)
    x_n = len(x)
    # Sanity checks
    if max(pattern_vl_grid) > x_n:
        raise ValueError(
            "Vector length of maximum potential pattern duration is longer than `x` vector length. "
            "Cannot identify a pattern that long. Decrease max value in `pattern_dur_grid`."
        )
    if min(pattern_vl_grid) < 3:
        raise ValueError(
            "Vector length of minimum potential pattern duration is less than 3. "
            "Cannot identify a pattern that short. Increase min value in `pattern_dur_grid`."
        )
    # Compute list of 2D arrays of rescaled pattern template(s)
    # Each 2D array corresponds to particular potential pattern vector length
    templ_rescaled = []
    for vl_i in pattern_vl_grid:
        item_tmp: npt.NDArray[np.float_] = np.stack(
            [interp_scale_vec(templ_i, vl_i) for templ_i in templ_array]
        )
        templ_rescaled.append(item_tmp)
    # Smooth x for computation of similarity matrix between x and template(s)
    x_sim = (
        copy(x) if np.isnan(x_sim_smooth_w) else rolling_smooth(x, x_sim_smooth_w, x_fs)
    )
    # Compute similarity matrix between x and template(s)
    simmat: npt.NDArray[np.float_] = np.stack(
        [
            np.amax(
                np.stack(
                    [rolling_corr(x_sim, templ_j) for templ_j in templ_rescaled_i]
                ),
                axis=0,
            )
            for templ_rescaled_i in templ_rescaled
        ]
    )
    simmat[np.isnan(simmat)] = -3
    # Define objects needed for fine-tuning of pattern location, if finetuning is selected
    x_ftune, nbhwing_vl, interp_vl_min, interp_vl_max = [None] * 4
    if ftune:
        # Smooth x for pattern location
        x_ftune = (
            copy(x)
            if np.isnan(x_ftune_smooth_w)
            else rolling_smooth(x, x_ftune_smooth_w, x_fs)
        )
        # Define neighbourhood within which we fine-tune pattern locations
        x_ftune_nbh_vl = int(x_ftune_nbh_w * x_fs)
        interp_vl_min = min(pattern_vl_grid)
        interp_vl_max = max(pattern_vl_grid)
        nbhwing_vl = min(int(np.floor(x_ftune_nbh_vl / 2)), interp_vl_min - 1)
    # Identify pattern occurrences iteratively
    out = []
    while True:
        # Break loop if all nan in similarity matrix
        # (:= no room is left for any more pattern in x signal)
        if (simmat < -2).all():
            break
        # Identify pattern location via finding similarity matrix maximum
        simmat_maxrow, simmat_maxcol = np.unravel_index(simmat.argmax(), simmat.shape)
        tau_i = simmat_maxcol
        T_i = pattern_vl_grid[simmat_maxrow]
        sim_i = simmat[simmat_maxrow][simmat_maxcol]
        # Fine-tune pattern location to match maxima of x data vector magnitude
        if ftune:
            tau_i, T_i, x_ftune = do_ftune(
                tau_i, T_i, x_ftune, nbhwing_vl, interp_vl_min, interp_vl_max
            )
        out_tmp = [tau_i, T_i, sim_i]
        out.append(out_tmp)
        # Update similarity matrix to mark location of newly identified pattern
        for i in range(0, vl_grid_n):
            col_idx_min = max(0, tau_i - pattern_vl_grid[i] + 2)
            col_idx_max = min(
                tau_i + T_i - 2, x_n - 2
            )  # x_n-2 (instead x_n-1)to prevent from escaping outside
            simmat[i : i + 1, col_idx_min : col_idx_max + 1] = -3
    return out


def segment_pattern_wrap(x_slice_vl, x, **kwargs):
    """Wrap segment_pattern function.

    Its role is to cut the (potentially very long one) vector magnitude accelerometry
    signal into shorter pieces, apply `segment_pattern` algorithm at each of those
    pieces, and merge the results.

    :param x_slice_vl: (int) Scalar. Length of a short arrays into which
    the `x` vector magnitude time-series is cut into.
    :param x:
    :param kwargs: The following arguments of `segment_pattern` function:
    x_fs, templ_array, pattern_dur_grid, x_sim_smooth_w, ftune, x_ftune_nbh_w, x_ftune_smooth_w
    :return: List of lists that describes the pattern segmentation results.
    Each "inner" list is a 3-element-long list
    describing one individual segmented stride:
    0. index of `x` VM time-series, where is the start of identified stride,
    1. duration (expressed of number of indices of `x`) of the identified stride,
    2. similarity (correlation) between the identified stride and stride pattern
    used in segmentation.
    """
    # Segment pattern from signal at once
    if np.isnan(x_slice_vl):
        return segment_pattern(x, **kwargs)
    # Segment pattern from signal at once over
    # partitions of signal x onto slices
    x_nn = len(x)
    out: List[List[float]] = []
    x_slice_idx0 = list(np.arange(0, x_nn, x_slice_vl).astype(int))
    x_slice_idx0 = x_slice_idx0 if x_slice_idx0[-1] == x_nn else x_slice_idx0 + [x_nn]
    for i in range(0, len(x_slice_idx0) - 1):
        idx_1 = x_slice_idx0[i]
        idx_n = x_slice_idx0[i + 1]
        x_slice = x[idx_1:idx_n]
        out_x_slice0 = segment_pattern(x_slice, **kwargs)
        out_x_slice = [[row[0] + idx_1] + row[1:] for row in out_x_slice0]
        out = out + out_x_slice
    return out
