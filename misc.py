"""
File for miscellaneous functions used in the simulations
"""

import numpy as np


def sq_len(v: np.ndarray) -> int | float:
    """
    Squared length of a vector
    :param v:
    :return:
    """
    return sum(np.power(v, 2))


def vector_len(v: np.ndarray) -> int | float:
    """
    :param v:
    :return:
    """
    return np.sqrt(sum(np.power(v, 2)))

