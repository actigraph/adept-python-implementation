import numpy as np
import pandas as pd


def dstack_product(x, y):
    '''
    Compute a 2D array whose rows consists of combination of all
    pairs of element fro `x` and `y`.

    :param x: (numpy.array) An arry.
    :param y: (numpy.array) An arry.
    :return: A 2D array.
    '''
    return np.dstack(np.meshgrid(x, y)).reshape(-1, 2)


def interp_vec(vec, vl_out):
    """
    Interpolate array to have certain length (number of elements in the array)
    via linear interpolation of values.

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
    """
    Interpolate array to have certain length (number of elements in the array)
    via linear interpolation of values, and then standardize to have
    mean 0, variance 1.

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
    yinterps = (yinterp - np.mean(yinterp)) * (1 / np.std(yinterp)) * np.sqrt((vl_out - 1) / vl_out)
    return yinterps


def stride_templ_df_to_array(df_path,
                             sensor_location_id_val="left_wrist",
                             collection_id_val="size_3"):
    '''
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
    '''
    ## Read CSV
    templ_df = pd.read_csv(df_path)
    templ_df_sub = templ_df[(templ_df['sensor_location_id'] == sensor_location_id_val) &
                            (templ_df['collection_id'] == collection_id_val)]
    templ_df_cols_sub = [col for col in templ_df_sub.columns if 'ph' in col]
    templ_df_sub = templ_df_sub[templ_df_cols_sub]
    if templ_df_sub.empty:
        raise ValueError(
            "Templates data frame is empty. "
            "Select correct `sensor_location_id_val` and/or `collection_id_val` values. "
            "Correct `sensor_location_id_val` values: 'left_wrist', 'left_hip', 'left_ankle', 'right_ankle'. "
            "Correct `collection_id` values: 'size_1', 'size_2', 'size_3', 'size_4', 'size_5'. ")
    templ_array = templ_df_sub.values
    return templ_array


def rank_chunks_of_ones(arr):
    '''
    Labels individual chunks of subsequent `1`'s in numpy.array.
    Reference:
    https://stackoverflow.com/questions/57573350/fast-python-ish-way-of-ranking-chunks-of-1s-in-numpy-array

    :param arr: (numpy.array) An array.
    :return: (numpy.array) An array which contains distinct labels (int scalars)
    for distinct individual chunks of subsequent `1`'s. The elements corresponding
    to places where `arr` does not have `1` value are set to numpy.nan.
    '''
    m = np.r_[False, arr.astype(bool), False]
    idx = np.flatnonzero(m[:-1] != m[1:])
    l = idx[1::2] - idx[::2]
    out = np.full(len(arr), np.nan, dtype=float)
    out[arr != 0] = np.repeat(np.arange(len(l)), l)
    return out
