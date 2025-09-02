import math
import numpy as np


def gen_bits(values: tuple[int, ...], value_size: int) -> tuple[tuple[int, ...], ...]:
    """The bit representation of the values.

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
    >>> from urca.common import gen_bits
    >>> gen_bits((0x6, 0xA, 0x1, 0xA), 4)
    ((0, 1, 1, 0), (1, 0, 1, 0), (0, 0, 0, 1), (1, 0, 1, 0))
    """
    return tuple(tuple(map(int, f"{value:0{value_size}b}")) for value in values)


def gen_bytesbox(sbox: tuple[int, ...]) -> tuple[int, ...]:
    """The byte sbox from the nibble one.

    Parameters
    ----------
    sbox : tuple[int]
        the nibble s-box whose byte version is required

    Returns
    -------
    tuple[int]
        the byte version of the sbox

    Examples
    --------
    >>> from urca.common import gen_bytesbox
    >>> gift_sbox = (0x1, 0xA, 0x4, 0xC, 0x6, 0xF, 0x3, 0x9, 0x2, 0xD, 0xB, 0x7, 0x5, 0x0, 0x8, 0xE)
    >>> gift_bytesbox = gen_bytesbox(gift_sbox)
    >>> gift_bytesbox[0x15] == 0xAF
    True
    """
    return tuple(i << 4 ^ j for i in sbox for j in sbox)


def get_dtype(word_size: int) -> np.dtype[np.uint8]:
    """Return the minimum size dtype.

    This function returns the minimum size dtype object that can contain the
    word size. This is useful for those primitives having a non-power-of-2 word
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
    >>> from urca.common import get_dtype
    >>> get_dtype(24)
    dtype('uint32')
    """
    power_of_2 = 2 ** math.ceil(math.log2(word_size))
    if power_of_2 < 8:
        numpy_dtype = np.dtype("uint8")
    else:
        numpy_dtype = np.dtype(f"uint{power_of_2}")

    return numpy_dtype


def invert_sbox(sbox: tuple[int, ...]) -> tuple[int, ...]:
    """The inverted S-box.

    Parameters
    ----------
    sbox : tuple[int]
        the S-box to be inverted

    Returns
    -------
    tuple[int]
        the inverted S-box

    Examples
    --------
    >>> from urca.common import invert_sbox
    >>> gift_sbox = (0x1, 0xA, 0x4, 0xC, 0x6, 0xF, 0x3, 0x9, 0x2, 0xD, 0xB, 0x7, 0x5, 0x0, 0x8, 0xE)
    >>> invert_sbox(gift_sbox)
    (13, 0, 8, 6, 2, 12, 4, 11, 14, 7, 1, 10, 3, 9, 15, 5)
    """
    return tuple(sbox.index(value) for value in range(len(sbox)))
