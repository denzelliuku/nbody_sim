"""
File for miscellaneous functions used in the simulations
"""

import numpy as np

# For typing
numeric = int | float


def sq_len(v: np.ndarray) -> numeric:
    """
    Squared length of a vector
    :param v:
    :return:
    """
    return sum(np.power(v, 2))


def vector_len(v: np.ndarray) -> numeric:
    """
    :param v:
    :return:
    """
    return np.sqrt(sum(np.power(v, 2)))
