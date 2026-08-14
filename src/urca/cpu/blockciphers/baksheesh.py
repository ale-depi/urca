import numpy as np

from urca import common
from urca.cpu.block import Block


class Baksheesh(Block):
    """The Baksheesh block cipher.

    Parameters
    ----------
    text_size : int, optional, default = 128
        the bit size of the block
    key_size : int, optional, default = 128
        the bit size of the key
    sbox : tuple[int, ...], optional, default = `original`
        the s-box for the cipher
    permutation : numpy.array, optional, default = `original`
        the permutation for the cipher
    """

    # fmt: off
    permutation = np.array(
        (
            96, 1, 34, 67, 64, 97, 2, 35, 32, 65, 98, 3, 0, 33, 66, 99, 100, 5, 38, 71, 68, 101, 6,
            39, 36, 69, 102, 7, 4, 37, 70, 103, 104, 9, 42, 75, 72, 105, 10, 43, 40, 73, 106, 11,
            8, 41, 74, 107, 108, 13, 46, 79, 76, 109, 14, 47, 44, 77, 110, 15, 12, 45, 78, 111, 112,
            17, 50, 83, 80, 113, 18, 51, 48, 81, 114, 19, 16, 49, 82, 115, 116, 21, 54, 87, 84, 117,
            22, 55, 52, 85, 118, 23, 20, 53, 86, 119, 120, 25, 58, 91, 88, 121, 26, 59, 56, 89, 122,
            27, 24, 57, 90, 123, 124, 29, 62, 95, 92, 125, 30, 63, 60, 93, 126, 31, 28, 61, 94, 127,
        ),
        dtype=np.uint8,
    )
    # fmt: on
    constants = np.array(
        (
            (0, 0, 0, 0, 1, 0),
            (1, 0, 0, 0, 0, 1),
            (0, 1, 0, 0, 0, 0),
            (0, 0, 1, 0, 0, 1),
            (1, 0, 0, 1, 0, 0),
            (0, 1, 0, 0, 1, 1),
            (1, 0, 1, 0, 0, 0),
            (1, 1, 0, 1, 0, 1),
            (0, 1, 1, 0, 1, 0),
            (0, 0, 1, 1, 0, 1),
            (1, 0, 0, 1, 1, 0),
            (1, 1, 0, 0, 1, 1),
            (1, 1, 1, 0, 0, 0),
            (1, 1, 1, 1, 0, 1),
            (1, 1, 1, 1, 1, 0),
            (0, 1, 1, 1, 1, 1),
            (0, 0, 1, 1, 1, 0),
            (0, 0, 0, 1, 1, 1),
            (1, 0, 0, 0, 1, 0),
            (1, 1, 0, 0, 0, 1),
            (0, 1, 1, 0, 0, 0),
            (1, 0, 1, 1, 0, 1),
            (1, 1, 0, 1, 1, 0),
            (1, 1, 1, 0, 1, 1),
            (0, 1, 1, 1, 0, 0),
            (1, 0, 1, 1, 1, 1),
            (0, 1, 0, 1, 1, 0),
            (1, 0, 1, 0, 1, 1),
            (0, 1, 0, 1, 0, 0),
            (0, 0, 1, 0, 1, 1),
            (0, 0, 0, 1, 0, 0),
            (0, 0, 0, 0, 1, 1),
            (1, 0, 0, 0, 0, 0),
            (0, 1, 0, 0, 0, 1),
            (0, 0, 1, 0, 0, 0),
        ),
        dtype=np.uint8,
    )
    constant_positions = np.array((21, 60, 92, 108, 114, 119), dtype=np.uint8)
    sbox = (0x3, 0x0, 0x6, 0xD, 0xB, 0x5, 0x8, 0xE, 0xC, 0xF, 0x9, 0x2, 0x4, 0xA, 0x7, 0x1)

    def __init__(
        self,
        text_size: int = 128,
        key_size: int = 128,
        sbox: tuple[int, ...] = sbox,
        permutation: np.ndarray = permutation,
    ) -> None:
        super().__init__(text_size, key_size)
        # required
        self.word_size = 1
        self.word_type = np.dtype("uint8")
        self.n_text_words = text_size
        self.n_key_words = key_size
        # cipher specific
        self.n_rounds = 35
        self.sbox = sbox
        self.inverse_sbox = common.invert_sbox(sbox)
        self.permutation = permutation
        # numpy internals
        self.np_sbox = np.array(common.gen_bytesbox(sbox), dtype=self.word_type)
        self.np_inversesbox = np.array(common.gen_bytesbox(self.inverse_sbox), dtype=self.word_type)

    def encrypt(self, texts: np.ndarray, keys: np.ndarray, state_index: int, n_rounds: int) -> None:
        """Encrypt in-place.

        Parameters
        ----------
        texts : np.ndarray
            plaintexts
        keys : np.ndarray
            keys
        state_index : int
            index of the current state
        n_rounds : int
            number of encryption rounds
        """
        if state_index == 0:
            texts ^= keys
        for round_number in range(state_index, state_index + n_rounds):
            # update keys
            keys[:, :] = np.roll(keys, 1, axis=1)
            # SubCells
            texts[:, :] = np.unpackbits(self.np_sbox[np.packbits(texts, axis=1)], axis=1)
            # PermBits
            texts[:, self.permutation] = texts[:, np.arange(self.text_size)]
            # AddConstants
            texts[:, self.constant_positions] ^= self.constants[round_number]
            # AddRoundKey
            texts ^= keys

    def decrypt(self, texts: np.ndarray, keys: np.ndarray, state_index: int, n_rounds: int) -> None:
        """Dencrypt in-place.

        Parameters
        ----------
        texts : np.ndarray
            ciphertexts
        keys : np.ndarray
            keys
        state_index : int
            index of the current state
        n_rounds : int
            number of decryption rounds
        """
        for round_number in reversed(range(state_index - n_rounds, state_index)):
            # AddRoundKey
            texts ^= keys
            # AddConstants
            texts[:, self.constant_positions] ^= self.constants[round_number]
            # PermBits
            texts[:, np.arange(self.text_size)] = texts[:, self.permutation]
            # SubCells
            texts[:, :] = np.unpackbits(self.np_inversesbox[np.packbits(texts, axis=1)], axis=1)
            # revert keys
            keys[:, :] = np.roll(keys, -1, axis=1)
        if state_index - n_rounds == 0:
            texts ^= keys
