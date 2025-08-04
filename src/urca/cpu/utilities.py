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

    Examples
    --------
    >>> from urca.utilities import get_dtype
    >>> get_dtype(24)
    dtype('uint32')

    """
    power_of_2 = 2 ** math.ceil(math.log2(word_size))
    if power_of_2 < 8:
        numpy_dtype = np.dtype(f"uint8")
    else:
        numpy_dtype = np.dtype(f"uint{power_of_2}")

    return numpy_dtype


def get_bits(values: tuple[int], value_size: int) -> tuple[tuple[int], ...]:
    """Return the bit representation of the values.

    Parameters
    ----------
    values : tuple[int]
        integers whose bit representation is needed
    value_size : int
        the bit size for each integer

    Returns
    -------
    tuple[tuple[int], ...]
        a tuple whose size is (len(values), value_size)

    Examples
    --------
    >>> from urca.utilities import get_bits
    >>> get_bits((0x6, 0xA, 0x1, 0xA), 4)
    ((0, 1, 1, 0), (1, 0, 1, 0), (0, 0, 0, 1), (1, 0, 1, 0))

    """
    return tuple(tuple(map(int, f"{value:0{value_size}b}")) for value in values)
