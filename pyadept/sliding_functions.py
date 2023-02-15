"""Functions for calculations on rolling windows."""
import math

import numpy as np
from numpy import typing as npt


def rolling_mean(x, w_vl, undef_val=np.nan):
    """
    Compute rolling mean of an array `x` over a window of length `w_vl`.

    The convention is that in the returned array of values, a value is
    shifted to the left with respect to a data window for which it (the value)
    is computed, i.e. the first element of the returned array corresponds
    to sample mean of `x[:w_vl]`.
    Indices of the returned array for which a rolling statistic is undefined are
    hence located at the array's tail; their values are defined by parameter
    `undef_val` and by default set to numpy.nan.

    :param x: (numpy.ndarray) 1-dimensional array
    :param w_vl: (int) A numeric scalar. Length of `x` window over which a
    rolling statistics is computed.
    :param undef_val: Scalar. Value inserted into a tail of returned array,
    for which a rolling statistic is undefined. Default is numpy.nan.
    :return: (numpy.ndarray) An array of rolling statistic values. The length of
        output array equals the length of `x` array.
    """
    out = np.convolve(x, np.ones(w_vl) / w_vl, mode="valid")
    return np.concatenate([out, [undef_val] * (w_vl - 1)])


def rolling_diff1(x, undef_val=np.nan):
    """
    Compute rolling difference between the subsequent elements of the array `x`.

    :param x: (numpy.ndarray) 1-dimensional array
    :param undef_val: Scalar. Value inserted into a tail of returned array,
    for which a rolling statistic is undefined. Default is numpy.nan.
    :return: (numpy.ndarray) An array of rolling difference values. The length of
        output array equals the length of `x` array.
    """
    # Note for defining the filter vec:
    # It must be [1, -1] b/c numpy.convolve implementation uses
    # *reversed* filtered signal in contrary to classical notation of
    # convolution.
    # Hence by using [1,-1] we obtain the desired effect:
    # "A positive derivative means that the function is increasing".
    filter_vec: npt.NDArray[np.int_] = np.array([1, -1])
    out = np.convolve(x, filter_vec, mode="valid")
    return np.concatenate([out, [undef_val] * 1])


def rolling_corr(x, y, undef_val=np.nan):
    """
    Compute rolling correlation between an array `x` and a short array `y`.

    The used convention is that in the returned array of values, a value is
    shifted to the left with repect to a data window for which it (the value)
    is computed, i.e. the first element of the returned array corresponds
    to sample correlation between `x[:len(y)]` and `y`.
    Indices of the returned array for which a rolling statistic is undefined are
    hence located at the vector's tail; their values are defined by parameter
    `undef_val` and by default set to numpy.nan.

    :param x: (numpy.ndarray) 1-dimensional array
    :param y: (numpy.ndarray) 1-dimensional array, shorter than `x`.
    :param undef_val: Scalar. Value inserted into a tail of returned array,
    for which a rolling statistic is undefined. Default is numpy.nan.
    :return: (numpy.ndarray) An array of rolling statistic values. The length of
        output array equals the length of `x` array.
    """
    w_vl = len(y)
    # Fix for the fact that numpy reverses the shorter array. Reference:
    # https://stackoverflow.com/questions/47096025/the-result-of-numpy-convolve-is-not-as-expected?noredirect=1&lq=1
    y_REV = y[::-1]
    # Mean of x (rolling mean), mean of y
    win = np.ones(w_vl)
    meanx = np.convolve(x, win / w_vl, mode="valid")
    meany = np.mean(y_REV)
    # Unbiased estimator of sample covariance
    covxy = (np.convolve(x, y_REV, mode="valid") - w_vl * meanx * meany) / (w_vl - 1)
    # Unbiased estimator of sample variance
    # S^2 = \frac{\sum X^2 - \frac{(\sum X)^2}{N}}{N-1}
    sigmax2: npt.NDArray[np.float_] = (  # type: ignore
        np.convolve(x ** 2, win, mode="valid")
        - ((np.convolve(x, win, mode="valid")) ** 2) / w_vl
    ) / (w_vl - 1)

    # Correct numerical errors if any
    sigmax2[sigmax2 < 0] = 0
    # Unbiased estimator of sample standard deviation
    # Note "w_vl / (w_vl-1)" gives version of vector variance consistent i.e. with R
    sigmax = np.sqrt(sigmax2)
    sigmay = np.sqrt(np.var(y) * w_vl / (w_vl - 1))
    # Compute unbiased estimator of sample covariance
    # * Surpress the warning which may occur if sigmax or sigmay being numerical 0
    # * Surpress the warning which occurs b/c of tail of vectors being filled with np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        corxy = covxy / (sigmax * sigmay)
    # Correct for infinite values which may occur if
    # sigmax or sigmay being numerical 0
    corxy[~np.isfinite(corxy)] = 0

    return np.concatenate([corxy, [undef_val] * (w_vl - 1)])


def rolling_smooth(x, w, x_fs=1, replace_undef_with_closest_val=True):
    """
    Compute moving window average of a time-series `x`.

    The used convention is that in the returned array of values, a value is
    centered with respect to a data window for which it (the value) is computed.
    Indices of the returned array for which a rolling smooth is undefined are
    hence located at the vector's head and tail; if
    `replace_undef_with_closest_val = False`,
    their values are set to  NAs (numpy.nan) whereas if
    `replace_undef_with_closest_val = True`, their values are with first and last
    non-NA value, respectively.

    Time-series frequency `x_fs` and a length of a moving window (expressed in
    seconds) `w` together determine `w_vl = round(w * x_fs)`, a length of a
    moving window expressed in a vector length (i.e., number of vector indices).

    Note:
    - `w_vl` must be equal or greater than 1. Otherwise, an error is risen.
    - `w_vl` must not be longer than length of `x` vector.  Otherwise, an error
    is risen.
    - If `w_vl` is an even number then `(w_vl - 1)` value is silently used
    instead.

    :param x: (numpy.ndarray) 1-dimensional array. A time-series for which a
    moving window average is computed.
    :param w: (float) A numeric scalar. A length of a moving window, expressed in
    time (seconds).
    :param x_fs: (float) A numeric scalar. Sampling frequency of `x`, expressed
    in Hz.
    :returns: (numpy.ndarray): An array of rolling smooth values. The length of
    output array equals the length of `x` vector.
    """
    n = len(x)
    # Check validity of smoothing window width `w` given frequency `x_fs`
    # of `x` data collection
    w_vl = int(round(w * x_fs))
    if w_vl < 1:
        raise ValueError(
            "Averaging window vector length `w_vl`"
            "(refer to function's details description) must not be "
            "less than 3. Define wider `w` averaging "
            "window length"
        )
    if w_vl > n:
        raise ValueError(
            "Averaging window vector length `w_vl`"
            "(refer to function's details description) must be less "
            "than `x` vector length. Define narrower `w` averaging window "
            "length"
        )
    # Replace `w_vl` with closest odd integer no larger than `w_vl`
    w_vl = w_vl + (w_vl % 2) - 1
    # Compute length of a moving window's "wing"
    w_vl_wing = int(math.floor(w_vl / 2))
    # Compute rolling mean (convention: result values values are shifted to the left
    # with repect to a data window over which mean is computed; hence undefined
    # rolling mean values are put as NA's at the end of the resulted vector)
    vec_out0 = rolling_mean(x, w_vl)
    # Shift rolling mean values (new convention: result values values are centered
    # with repect to a data window over which mean is computed; hence undefined
    # rolling mean values are put as NA's at both head and tail of the resulted vector)
    vec_out0 = vec_out0[: (n - 2 * w_vl_wing)]
    vec_out = [np.nan] * n
    vec_out[w_vl_wing : (w_vl_wing + len(vec_out0))] = vec_out0
    # If specified, replace NA values in the returned vector in head and tail with
    # first and last non-NA value
    if replace_undef_with_closest_val:
        # vec_out[:w_vl_wing] = [vec_out[w_vl_wing]] * w_vl_wing
        # vec_out[(n - w_vl_wing):] = [vec_out[(n - w_vl_wing - 1)]] * w_vl_wing
        vec_out[:w_vl_wing] = x[:w_vl_wing]
        vec_out[(n - w_vl_wing) :] = x[(n - w_vl_wing) :]

    return np.array(vec_out)
