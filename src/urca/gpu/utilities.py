import math
import cupy as cp


def get_dtype(word_size: int) -> cp.dtype:
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
    cp.dtype
        the cupy dtype object of the minimum size

    Examples
    --------
    >>> from urca.utilities import get_dtype
    >>> get_dtype(24)
    dtype('uint32')

    """
    power_of_2 = 2 ** math.ceil(math.log2(word_size))
    if power_of_2 < 8:
        cupy_dtype = cp.dtype("uint8")
    else:
        cupy_dtype = cp.dtype(f"uint{power_of_2}")

    return cupy_dtype
