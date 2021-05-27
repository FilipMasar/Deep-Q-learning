import numpy as np

from src.utils import calculate_training_stats


def test_calculate_training_stats():
    data = [1, 1, 1, 1, 1, 1, 1]
    x, mean, std = calculate_training_stats(data)

    np.testing.assert_array_equal(x, np.arange(len(data)))
    np.testing.assert_array_equal(mean, np.ones(len(data)))
    np.testing.assert_array_equal(std, np.zeros(len(data)))
