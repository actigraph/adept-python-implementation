import numpy as np
import numpy.testing as npt

from src.segment_pattern import segment_pattern, interp_vec


# [If ran from the notebook]
# Read script with code being tested
# import os
# project_dir = "C:\\Users\\kARASMA6\\OneDrive - Novartis Pharma AG\\PROJECTS\\walking_segmentation"
# exec(open(os.path.join(project_dir, 'python', 'sliding_functions.py')).read())
# exec(open(os.path.join(project_dir, 'python', 'segment_pattern.py')).read())
# exec(open(os.path.join(project_dir, 'python', 'segment_pattern_utils.py')).read())


def test_suite_1_example_a():
    """
    CASE:
    - no noise in signal
    - all pattern occurences of the same length (vector length: 101)
    - true vector length of pattern occurences *not* inclued in
      grid of potential pattern durations considered
    """
    # Generate signal and template
    t_vec = np.linspace(0, 2 * np.pi * 10, 1001, endpoint=True)
    x = np.cos(t_vec)
    templ_array = np.array([np.array(x[:101])])
    x_fs = 1
    pattern_dur_grid = [90, 100, 110]
    # Compute result
    out = segment_pattern(
        x=x, x_fs=x_fs, templ_array=templ_array, pattern_dur_grid=pattern_dur_grid
    )
    out.sort(key=lambda x: int(x[0]))
    # Check pattern start index
    res = [row[0] for row in out]
    exp = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    npt.assert_equal(res, exp)
    # Check pattern duration
    res = [row[1] for row in out]
    exp = [100] * 10
    npt.assert_equal(res, exp)
    # Check similarity
    res = [row[2] for row in out]
    exp = [0.9994793417318214] * 10
    npt.assert_almost_equal(res, exp)


def test_suite_1_example_b():
    """
    CASE:
    - no noise in signal
    - all pattern occurences of the same length (vector length: 101)
    - true vector length of pattern occurences inclued in
      grid of potential pattern durations considered
    """
    # Generate signal and template
    t_vec = np.linspace(0, 2 * np.pi * 10, 1001, endpoint=True)
    x = np.cos(t_vec)
    templ_array = np.array([np.array(x[:101])])
    x_fs = 1
    pattern_dur_grid = [90, 101, 110]
    # Compute result
    out = segment_pattern(
        x=x, x_fs=x_fs, templ_array=templ_array, pattern_dur_grid=pattern_dur_grid
    )
    out.sort(key=lambda x: int(x[0]))
    # Check pattern start index
    res = [row[0] for row in out]
    exp = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    npt.assert_equal(res, exp)
    # Check pattern duration
    res = [row[1] for row in out]
    exp = [101] * 10
    npt.assert_equal(res, exp)
    # Check similarity
    res = [row[2] for row in out]
    exp = [1] * 10
    npt.assert_almost_equal(res, exp)


def test_suite_1_example_c():
    """
    CASE:
    - no noise in signal
    - all pattern occurences of the same length (vector length: 101)
    - true vector length of pattern occurences inclued in grid considered
    - modify assumed signal frequency (and adjust grid of potential pattern
      durations accordingly)
    """
    # Generate signal and template
    t_vec = np.linspace(0, 2 * np.pi * 10, 1001, endpoint=True)
    x = np.cos(t_vec)
    templ_array = np.array([np.array(x[:101])])
    x_fs = 100
    pattern_dur_grid = np.array([90, 101, 110]) * 0.01
    # Compute result
    out = segment_pattern(
        x=x, x_fs=x_fs, templ_array=templ_array, pattern_dur_grid=pattern_dur_grid
    )
    out.sort(key=lambda x: int(x[0]))
    # Check pattern start index
    res = [row[0] for row in out]
    exp = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    npt.assert_equal(res, exp)
    # Check pattern duration
    res = [row[1] for row in out]
    exp = [101] * 10
    npt.assert_equal(res, exp)
    # Check similarity
    res = [row[2] for row in out]
    exp = [1] * 10
    npt.assert_almost_equal(res, exp)


def test_suite_2_example_a():
    """
    CASE:
    - no noise in signal
    - pattern occurences of different length
    - true vector length of pattern occurences inclued in
      grid of potential pattern durations considered
    """
    # Generate template
    t_vec = np.linspace(0, 2 * np.pi, 200, endpoint=True)
    templ = np.cos(t_vec)
    # Generate signal
    interp_vl_grid = [76, 82, 93, 112, 71, 110, 111, 95, 118, 63]
    x = interp_vec(templ, interp_vl_grid[0]).tolist()
    for interp_vl in interp_vl_grid[1:]:
        x.extend((interp_vec(templ, interp_vl))[1:])
    x = np.array(x)
    x_fs = 1
    templ_array = np.array([templ])
    pattern_dur_grid = np.array(range(60, 120))
    # Compute result
    out = segment_pattern(
        x=x, x_fs=x_fs, templ_array=templ_array, pattern_dur_grid=pattern_dur_grid
    )
    out.sort(key=lambda x: int(x[0]))
    # Check pattern start index
    res = [row[0] for row in out]
    exp = [0, 75, 156, 248, 359, 429, 538, 648, 742, 859]
    npt.assert_equal(res, exp)
    # Check pattern duration
    res = [row[1] for row in out]
    exp = interp_vl_grid
    npt.assert_equal(res, exp)
    # Check similarity
    res = [row[2] for row in out]
    exp = [1] * 10
    npt.assert_almost_equal(res, exp)


def test_suite_2_example_b():
    """
    CASE:
    - no noise in signal
    - pattern occurences of different length
    - use "poor" grid of potential pattern durations considered
    """
    # Generate template
    t_vec = np.linspace(0, 2 * np.pi, 200, endpoint=True)
    templ = np.cos(t_vec)
    # Generate signal
    interp_vl_grid = [76, 82, 93, 112, 71, 110, 111, 95, 118, 63]
    x = interp_vec(templ, interp_vl_grid[0]).tolist()
    for interp_vl in interp_vl_grid[1:]:
        x.extend((interp_vec(templ, interp_vl))[1:])
    x = np.array(x)
    x_fs = 1
    templ_array = np.array([templ])
    pattern_dur_grid = np.array([60, 80, 100, 120])
    # Compute result
    out = segment_pattern(
        x=x, x_fs=x_fs, templ_array=templ_array, pattern_dur_grid=pattern_dur_grid
    )
    out.sort(key=lambda x: int(x[0]))
    # Check pattern start index
    res = [row[0] for row in out]
    exp = [8, 76, 163, 244, 364, 434, 533, 655, 741, 860]
    npt.assert_equal(res, exp)
    # Check pattern duration
    res = [row[1] for row in out]
    exp = [60, 80, 80, 120, 60, 100, 120, 80, 120, 60]
    npt.assert_equal(res, exp)
    # Check similarity
    res = np.mean([row[2] for row in out])
    exp = [0.9964646292042814]
    npt.assert_almost_equal(res, exp)


def test_suite_2_example_c():
    """
    CASE:
    - no noise in signal
    - 2 distinct patterns occurring in the signal
    - pattern occurences of different length
    - true vector length of pattern occurences inclued in
      grid of potential pattern durations considered
    """
    # Generate templates
    t_vec = np.linspace(0, 2 * np.pi, 300, endpoint=True)
    templ1 = np.cos(t_vec)
    templ2_a = np.linspace(1, 0, 100, endpoint=True)
    templ2_b = np.array([0] * 100)
    templ2_c = np.linspace(0, 1, 100, endpoint=True)
    templ2 = np.concatenate((templ2_a, templ2_b, templ2_c), axis=None)
    # Generate signal
    interp_vl_grid = [76, 82, 93, 112, 71, 110, 111, 95, 118, 63]
    x = interp_vec(templ1, interp_vl_grid[0])
    x = np.concatenate((x, (interp_vec(templ2, interp_vl_grid[0]))[1:]), axis=None)
    for interp_vl in interp_vl_grid[1:]:
        x = np.concatenate((x, (interp_vec(templ1, interp_vl))[1:]), axis=None)
        x = np.concatenate((x, (interp_vec(templ2, interp_vl))[1:]), axis=None)
    # Compute result
    x_fs = 1
    templ_array = np.array([templ1, templ2])
    pattern_dur_grid = np.array(range(60, 120))
    out = segment_pattern(
        x=x, x_fs=x_fs, templ_array=templ_array, pattern_dur_grid=pattern_dur_grid
    )
    out.sort(key=lambda x: int(x[0]))
    # Check pattern start index
    res = np.mean([row[0] for row in out])
    exp = [856.85]
    npt.assert_almost_equal(res, exp)
    res = np.var([row[0] for row in out])
    exp = [307093.1275]
    npt.assert_almost_equal(res, exp)
    # Check pattern duration
    res = [row[1] for row in out]
    exp = sum([[val, val] for val in interp_vl_grid], [])
    npt.assert_equal(res, exp)
    # Check similarity
    res = [row[2] for row in out]
    exp = [1] * 20
    npt.assert_almost_equal(res, exp)
