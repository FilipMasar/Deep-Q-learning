import numpy as np


# decorator to convert input arguments and return value of a function to np.ndarray with dtype np.float32
def convert_to_numpy(func):
    def converted(*args):
        ndarray_args = []
        for a in list(args):
            try:
                ndarray_args.append(np.array(a, dtype=np.float32))
            except TypeError:
                ndarray_args.append(a)

        return np.array(func(*ndarray_args), np.float32)

    return converted


def calculate_training_stats(returns, average_over=50):
    n = len(returns)

    x = np.arange(n)
    mean = np.array([np.average(returns[max(0, i - average_over + 1):i + 1]) for i in range(n)])
    std = np.array([np.std(returns[max(0, i - average_over + 1):i + 1]) for i in range(n)])

    return x, mean, std
