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
