import numpy as np

from urca import common
from urca import constants
from urca.cpu.block import Block


class Simeck(Block):
    """
    The Simeck block cipher.

    Parameters
    ----------
    text_size : int, optional, default = 32
        the bit size of the block
    key_size : int, optional, default = 64
        the bit size of the key
    rot : tuple, optional, default = (5, 1)
        the rotation amounts in round schedule
    z_sequence : int, optional, default = constants.SIMECK_Z0
        the bit sequence for key schedule

    """

    def __init__(
        self,
        text_size: int,
        key_size: int,
        rot: tuple = (5, 1),
        z_sequence: int = constants.SIMECK_Z0,
    ) -> None:
        super().__init__(text_size, key_size)
        # required
        self.word_size = text_size // 2
        self.word_type = common.get_dtype(self.word_size)
        self.n_text_words = text_size // self.word_size
        self.n_key_words = key_size // self.word_size
        # cipher specific
        self.constant = 2**self.word_size - 4
        self.rot = rot
        self.z_sequence = z_sequence
        # numpy internals
        self.mask = np.sum(2 ** np.arange(self.word_size), dtype=self.word_type)
        self.np_rot = np.array(rot, dtype=np.uint8)
        self.np_rotc = self.word_size - self.np_rot

    def feistel(self, texts: np.ndarray, keys: np.ndarray) -> None:
        """Apply the Simeck Feistel function to texts.

        Parameters
        ----------
        texts : np.ndarray
            plaintexts or texts
        keys : np.ndarray
            keys
        """
        output = (texts[:, 0] << self.np_rot[0] | texts[:, 0] >> self.np_rotc[0]) & self.mask
        output &= texts[:, 0]
        output ^= (texts[:, 0] << self.np_rot[1] | texts[:, 0] >> self.np_rotc[1]) & self.mask
        texts[:, 1] ^= output ^ keys

    def encrypt(self, texts: np.ndarray, keys: np.ndarray, state_index: int, n_rounds: int) -> None:
        """Encrypt using Simeck.

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
        for round_number in range(state_index, state_index + n_rounds):
            self.feistel(texts, keys[:, 3])
            texts[:, :] = np.roll(texts, 1, axis=1)
            self.feistel(keys[:, 2:4], self.constant ^ ((self.z_sequence >> round_number) & 1))
            keys[:, :] = np.roll(keys, 1, axis=1)

    def decrypt(self, texts: np.ndarray, keys: np.ndarray, state_index: int, n_rounds: int) -> None:
        """Decrypt in-place.

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
            keys[:, :] = np.roll(keys, -1, axis=1)
            self.feistel(keys[:, 2:4], self.constant ^ ((self.z_sequence >> round_number) & 1))
            texts[:, :] = np.roll(texts, 1, axis=1)
            self.feistel(texts, keys[:, 3])
