"""Provide statistics about segments."""
import numpy as np


def vmc(vec):
    """
    Compute Vector Magnitude Count of a time-series.

    :param vec:
    :return:
    """
    return np.mean(np.abs(vec - np.mean(vec)))


def describe_segments(sp_out, data_xyzptr, data_fs):
    """
    Provide statistics (median, std, peak-to-peak, vector magnitude count).

    Statistics are provided for each individual pattern occurrence segmented with
    `segment_pattern` function.

    Uses the output of `segment_pattern` function and
    raw `xyzptr` data from which the data values corresponding to each identified
    pattern occurrence are pulled from.

    :param sp_out: (list of lists) Output of `segment_pattern` function. Each
    "inner list" constsis of 3 elements:
    0. pattern occurrence start location index,
    1. pattern occurrence length expressed in array length,
    2. correlation value between the accelerometry data and predefined pattern template.
    :param data_xyzptr: (numpy.array) "xyzptr" raw accelerometry data array.
    :param data_fs: (float) Scalar. Accelerometry data collection frequency,
    expressed in Hz.
    :return: List of lists. Each "inner" list describes one segmented pattern
    occurrence, and contains 23 elements - numerical summaries of a segmented pattern
    occurrence. Those 23 summaries are:
    0. pattern occurrence start location index,
    1. pattern occurrence length expressed in array length,
    2. correlation value between the accelerometry data and predefined pattern template,
    3. median of `x` Cartesian coordinate of corresponding accelerometry data segment,
    4. median of `y` Cartesian coordinate of corresponding accelerometry data segment,
    5. median of `z` Cartesian coordinate of corresponding accelerometry data segment,
    6. median of `p` spherical coordinate of corresponding accelerometry data segment,
    7. median of `t` spherical coordinate of corresponding accelerometry data segment,
    8. median of `r` spherical coordinate of corresponding accelerometry data segment,
    9. standard deviation of `x` Cartesian coordinate of corresponding accelerometry data segment,
    10. standard deviation of `y` Cartesian coordinate of corresponding accelerometry data segment,
    11. standard deviation of `z` Cartesian coordinate of corresponding accelerometry data segment,
    12. standard deviation of `p` spherical coordinate of corresponding accelerometry data segment,
    13. standard deviation of `t` spherical coordinate of corresponding accelerometry data segment,
    14. standard deviation of `r` spherical coordinate of corresponding accelerometry data segment,
    15. peak-to-peak of `x` Cartesian coordinate of corresponding accelerometry data segment,
    16. peak-to-peak of `y` Cartesian coordinate of corresponding accelerometry data segment,
    17. peak-to-peak of `z` Cartesian coordinate of corresponding accelerometry data segment,
    18. peak-to-peak of `p` spherical coordinate of corresponding accelerometry data segment,
    19. peak-to-peak of `t` spherical coordinate of corresponding accelerometry data segment,
    20. peak-to-peak of `r` spherical coordinate of corresponding accelerometry data segment,
    21. Vector Magnitude Count,
    22. duration (expressed in seconds).
    """
    desc_out = [
        out_i
        # median
        + [
            np.median(data_xyzptr[(out_i[0]) : (out_i[0] + out_i[1]), i])
            for i in range(6)
        ]
        # std
        + [np.std(data_xyzptr[(out_i[0]) : (out_i[0] + out_i[1]), i]) for i in range(6)]
        # ptp
        + [np.ptp(data_xyzptr[(out_i[0]) : (out_i[0] + out_i[1]), i]) for i in range(6)]
        # VMC
        + [vmc(data_xyzptr[(out_i[0]) : (out_i[0] + out_i[1]), 5])]
        # stride duration
        + [out_i[1] / data_fs]
        for out_i in sp_out
    ]
    return desc_out
