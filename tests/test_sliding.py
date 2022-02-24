import math

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from pyadept.sliding_functions import rolling_mean, rolling_corr, rolling_smooth


def conventional_mean(x, W):
    """
    Compute rolling mean of a vector `x` over a window of a vector length `W`
    in a 'conventional' way, that is, with a loop.
    Comparator for `rolling_mean`.
    """
    out = [np.nan] * len(x)
    for i in np.arange(len(x) - W + 1):
        val = np.mean(x[i : (i + W)])
        out[i] = val
    return np.array(out)


def conventional_corr(x, y):
    """
    Compute rolling correlation between a vector `x` and a short vector `y`.
    in a 'conventional' way, that is, with a loop.
    Comparator for `rolling_corr`.
    """
    win = len(y)
    out = [np.nan] * len(x)
    for i in np.arange(len(x) - win + 1):
        val = np.corrcoef(x[i : (i + win)], y)[0, 1]
        out[i] = val
    return np.array(out)


def test_mean_result_is_unchanged():
    # Define objects used in tests
    np.random.seed(1)
    x = np.random.normal(loc=1, scale=1, size=1000)
    # Compute result
    out = rolling_mean(x, 100)
    # Check output length unchanged
    assert len(out) == 1000
    # Check output mean value unchanged
    assert np.nanmean(out) == pytest.approx(1.045207145486756)
    # Check output tail NA's unchanged
    assert pd.Series(out[901:]).isnull().values.all()
    # Check output head NA's unchanged
    assert not (pd.Series(out[:900]).isnull().values.any())


def test_mean_result_agrees_with_conventional():
    # Define objects used in tests
    np.random.seed(1)
    x = np.random.normal(loc=1, scale=1, size=1000)
    # Compute result
    out1 = rolling_mean(x, 100)
    out2 = conventional_mean(x, 100)
    # Check output same as when computed with conventional function
    npt.assert_array_almost_equal(out1, out2)


def test_corr_result_is_unchanged():
    # Define objects used in tests
    y = np.sin(np.linspace(0, 2 * math.pi, 101))
    x = np.concatenate([np.tile(y[:-1], 10), y])
    # Compute result
    out = rolling_corr(x, y)
    # Check output length unchanged
    assert len(out) == 1101
    # Check output mean value unchanged
    assert np.nanmean(out) == pytest.approx(0.0009990009990010007)
    # Check output tail NA's unchanged
    assert pd.Series(out[1001:]).isnull().values.all()
    # Check output head NA's unchanged
    assert not (pd.Series(out[:1000]).isnull().values.any())


def test_corr_result_is_unchanged2():
    # Define objects used in tests
    N = 1000
    n = 100
    np.random.seed(1)
    x = np.random.normal(0, 1, N)
    y = np.random.normal(0, 1, n)
    # Compute result
    result = np.nanmean(rolling_corr(x, y))
    assert result == pytest.approx(-0.0001137391883578133)


def test_corr_result_agrees_with_conventional():
    # Define objects used in tests
    y = np.sin(np.linspace(0, 2 * math.pi, 101))
    x = np.concatenate([np.tile(y[:-1], 10), y])
    # Compute result
    out1 = rolling_corr(x, y)
    out2 = conventional_corr(x, y)
    # Check output same as when computed with conventional function
    npt.assert_array_almost_equal(out1, out2)


def test_corr_result_agrees_with_conventional2():
    # Define objects used in tests
    N = 1000
    n = 100
    np.random.seed(1)
    x = np.random.normal(0, 1, N)
    y = np.random.normal(0, 1, n)
    # Compute result
    out1 = rolling_corr(x, y)
    out2 = conventional_corr(x, y)
    # Check output same as when computed with conventional function
    npt.assert_array_almost_equal(out1, out2)


def test_smooth_result_is_unchanged_short_vector1():
    # Define objects used in tests
    N = 10
    x = np.linspace(1, N, num=N)
    # Compute result and test against expected result
    result = rolling_smooth(x, w=5)
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    npt.assert_array_almost_equal(result, expected)


def test_smooth_result_is_unchanged_short_vector2():
    # Define objects used in tests
    N = 10
    x = np.linspace(1, N, num=N)
    # Compute result and test against expected result
    result = rolling_smooth(x, w=N)
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    npt.assert_array_almost_equal(result, expected)


def test_smooth_result_is_unchanged_short_vector3():
    # Define objects used in tests
    N = 9
    x = np.linspace(1, N, num=N)
    # Compute result and test against expected result
    result = rolling_smooth(x, w=N)
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    npt.assert_array_almost_equal(result, expected)


def test_smooth_result_is_unchanged_long_vector1():
    # Define objects used in tests
    N = 10000
    x = np.linspace(1, N, num=N)
    # Compute result and test against expected result
    result = rolling_smooth(x, w=10)
    assert np.mean(result) == 5000.5
    assert np.var(result), 8333333.25


def test_smooth_result_is_unchanged_long_vector2():
    # Define objects used in tests
    N = 10000
    x = np.linspace(1, N, num=N)
    # Compute result and test against expected result
    # Change frequency and adjust window length accordingly
    result = rolling_smooth(x, w=1, x_fs=10)
    assert np.mean(result) == 5000.5
    assert np.var(result) == 8333333.25


def test_smooth_result_is_unchanged_long_vector3():
    # Define objects used in tests
    N = 10000
    x = np.linspace(1, N, num=N)
    # Compute result and test againsts expected result
    result = rolling_smooth(x, w=1)
    assert np.mean(result) == 5000.5
    assert np.var(result) == 8333333.25


def test_smooth_error_is_thrown():
    # Define objects used in tests
    N = 10000
    x = np.linspace(1, N, num=N)
    # Compute result and test against expected result
    with pytest.raises(Exception):
        _ = rolling_smooth(x, w=0.5)


def test_smooth_error_is_thrown2():
    # Define objects used in tests
    N = 10000
    x = np.linspace(1, N, num=N)
    # Compute result and test against expected result
    with pytest.raises(Exception):
        _ = rolling_smooth(x, w=N + 1)
