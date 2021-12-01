import unittest
import math
import numpy as np
import numpy.testing as npt
import pandas as pd

## When ran from a notebook, read file with code being tested
## import os
## project_dir = "C:\\Users\\kARASMA6\\OneDrive - Novartis Pharma AG\\PROJECTS\\walking_segmentation"
## exec(open(os.path.join(project_dir, 'python', 'sliding_functions.py')).read())

def conventional_mean(x, W):
    '''
    Compute rolling mean of a vector `x` over a window of a vector length `W`
    in a 'conventional' way, that is, with a loop.
    Comparator for `rolling_mean`.
    '''
    out = [np.nan] * len(x)
    for i in np.arange(len(x) - W + 1):
        val = np.mean(x[i:(i + W)])
        out[i] = val
    return np.array(out)


def conventional_corr(x, y):
    '''
    Compute rolling correlation between a vector `x` and a short vector `y`.
    in a 'conventional' way, that is, with a loop.
    Comparator for `rolling_corr`.
    '''
    win = len(y)
    out = [np.nan] * len(x)
    for i in np.arange(len(x) - win + 1):
        val = np.corrcoef(x[i:(i + win)], y)[0, 1]
        out[i] = val
    return np.array(out)


class TestRollingMean(unittest.TestCase):

    def test_mean_result_is_unchanged(self):
        ## Define objects used in tests
        np.random.seed(1)
        x = np.random.normal(loc=1, scale=1, size=1000)
        ## Compute result
        out = rolling_mean(x, 100)
        ## Check output length unchanged
        self.assertTrue(len(out) == 1000)
        ## Check output mean value unchanged
        self.assertAlmostEqual(np.nanmean(out), 1.045207145486756, places=7)
        ## Check output tail NA's unchanged
        self.assertTrue(pd.Series(out[901:]).isnull().values.all())
        ## Check output head NA's unchanged
        self.assertFalse(pd.Series(out[:900]).isnull().values.any())

    def test_mean_result_agrees_with_conventional(self):
        ## Define objects used in tests
        np.random.seed(1)
        x = np.random.normal(loc=1, scale=1, size=1000)
        ## Compute result
        out1 = rolling_mean(x, 100)
        out2 = conventional_mean(x, 100)
        ## Check output same as when computed with conventional function
        npt.assert_array_almost_equal(out1, out2)


class TestRollingCorr(unittest.TestCase):

    def test_corr_result_is_unchanged(self):
        ## Define objects used in tests
        y = np.sin(np.linspace(0, 2 * math.pi, 101))
        x = np.concatenate([np.tile(y[:-1], 10), y])
        ## Compute result
        out = rolling_corr(x, y)
        ## Check output length unchanged
        self.assertTrue(len(out) == 1101)
        ## Check output mean value unchanged
        self.assertAlmostEqual(np.nanmean(out), 0.0009990009990010007, places=7)
        ## Check output tail NA's unchanged
        self.assertTrue(pd.Series(out[1001:]).isnull().values.all())
        ## Check output head NA's unchanged
        self.assertFalse(pd.Series(out[:1000]).isnull().values.any())

    def test_corr_result_is_unchanged2(self):
        ## Define objects used in tests
        N = 1000
        n = 100
        np.random.seed(1)
        x = np.random.normal(0, 1, N)
        y = np.random.normal(0, 1, n)
        ## Compute result
        result = np.nanmean(rolling_corr(x, y))
        self.assertAlmostEqual(result, -0.0001137391883578133, places=7)

    def test_corr_result_agrees_with_conventional(self):
        ## Define objects used in tests
        y = np.sin(np.linspace(0, 2 * math.pi, 101))
        x = np.concatenate([np.tile(y[:-1], 10), y])
        ## Compute result
        out1 = rolling_corr(x, y)
        out2 = conventional_corr(x, y)
        ## Check output same as when computed with conventional function
        npt.assert_array_almost_equal(out1, out2)

    def test_corr_result_agrees_with_conventional2(self):
        ## Define objects used in tests
        N = 1000
        n = 100
        np.random.seed(1)
        x = np.random.normal(0, 1, N)
        y = np.random.normal(0, 1, n)
        ## Compute result
        out1 = rolling_corr(x, y)
        out2 = conventional_corr(x, y)
        ## Check output same as when computed with conventional function
        npt.assert_array_almost_equal(out1, out2)


class TestRollingSmooth(unittest.TestCase):

    def test_smooth_result_is_unchanged_short_vector1(self):
        ## Define objects used in tests
        N = 10
        x = np.linspace(1, N, num=N)
        ## Compute result and test aganist expected result
        result = rolling_smooth(x, w=5)
        expected = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
        npt.assert_array_almost_equal(result, expected)

    def test_smooth_result_is_unchanged_short_vector2(self):
        ## Define objects used in tests
        N = 10
        x = np.linspace(1, N, num=N)
        ## Compute result and test aganist expected result
        result = rolling_smooth(x, w=N)
        expected = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
        npt.assert_array_almost_equal(result, expected)

    def test_smooth_result_is_unchanged_short_vector3(self):
        ## Define objects used in tests
        N = 9
        x = np.linspace(1, N, num=N)
        ## Compute result and test aganist expected result
        result = rolling_smooth(x, w=N)
        expected = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.])
        npt.assert_array_almost_equal(result, expected)

    def test_smooth_result_is_unchanged_long_vector1(self):
        ## Define objects used in tests
        N = 10000
        x = np.linspace(1, N, num=N)
        ## Compute result and test aganist expected result
        result = rolling_smooth(x, w=10)
        self.assertEqual(np.mean(result), 5000.5)
        self.assertEqual(np.var(result), 8333333.25)

    def test_smooth_result_is_unchanged_long_vector2(self):
        ## Define objects used in tests
        N = 10000
        x = np.linspace(1, N, num=N)
        ## Compute result and test aganist expected result
        ## Change frequency and adjust window length accordingly
        result = rolling_smooth(x, w=1, x_fs=10)
        self.assertEqual(np.mean(result), 5000.5)
        self.assertEqual(np.var(result), 8333333.25)

    def test_smooth_result_is_unchanged_long_vector3(self):
        ## Define objects used in tests
        N = 10000
        x = np.linspace(1, N, num=N)
        ## Compute result and test aganist expected result
        result = rolling_smooth(x, w=1)
        self.assertEqual(np.mean(result), 5000.5)
        self.assertEqual(np.var(result), 8333333.25)

    def test_smooth_error_is_thrown(self):
        ## Define objects used in tests
        N = 10000
        x = np.linspace(1, N, num=N)
        ## Compute result and test aganist expected result
        with self.assertRaises(Exception) as context:
            result = rolling_smooth(x, w=0.5)

    def test_smooth_error_is_thrown2(self):
        ## Define objects used in tests
        N = 10000
        x = np.linspace(1, N, num=N)
        ## Compute result and test aganist expected result
        with self.assertRaises(Exception) as context:
            result = rolling_smooth(x, w=N + 1)


if __name__ == '__main__':
    ## When used in notebook, use:
    ## https://medium.com/@vladbezden/using-python-unittest-in-ipython-or-jupyter-732448724e31
    ## unittest.main(argv=['first-arg-is-ignored'], exit=False)
    unittest.main()