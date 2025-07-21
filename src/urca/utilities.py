"""
Utilities
=========

These are the common functions needed across all ciphers.
"""
import math
import numpy as np


def get_dtype(word_size: int) -> np.dtype:
    """Return the minimum size dtype.

    This function returns the minimum size dtype object that can contain the
    word size. This is useful for those ciphers having a non-power-of-2 word
    size (e.g. Speck 48/96).

    Parameters
    ----------
    word_size : int
        the size of the word in bits

    Returns
    -------
    np.dtype
        the numpy dtype object of the minimum size
    """
    power_of_2 = 2 ** math.ceil(math.log2(word_size))
    if power_of_2 < 8:
        numpy_dtype = np.dtype(f"uint8")
    else:
        numpy_dtype = np.dtype(f"uint{power_of_2}")

    return numpy_dtype
