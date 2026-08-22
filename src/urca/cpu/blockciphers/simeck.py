import numpy as np

from urca import common, constants
from urca.cpu.blockcipher import BlockCipher


class Simeck(BlockCipher):
    """
    The Simeck block cipher.

    Parameters
    ----------
    block_size : int, optional, default = 32
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
        block_size: int = 32,
        key_size: int = 64,
        rot: tuple = (5, 1),
        z_sequence: int = constants.SIMECK_Z0,
    ) -> None:
        super().__init__(block_size, key_size)
        # required
        self.word_size = block_size // 2
        self.word_type = common.get_dtype(self.word_size)
        self.n_block_words = block_size // self.word_size
        self.n_key_words = key_size // self.word_size
        # cipher specific
        self.constant = 2**self.word_size - 4
        self.rot = rot
        self.z_sequence = z_sequence
        # numpy internals
        self.mask = np.sum(2 ** np.arange(self.word_size), dtype=self.word_type)
        self.np_rot = np.array(rot, dtype=np.uint8)
        self.np_rotc = self.word_size - self.np_rot

    def feistel(self, blocks: np.ndarray, keys: np.ndarray) -> None:
        """Apply the Simeck Feistel function to blocks.

        Parameters
        ----------
        blocks : np.ndarray
            blocks
        keys : np.ndarray
            keys
        """
        output = (blocks[:, 0] << self.np_rot[0] | blocks[:, 0] >> self.np_rotc[0]) & self.mask
        output &= blocks[:, 0]
        output ^= (blocks[:, 0] << self.np_rot[1] | blocks[:, 0] >> self.np_rotc[1]) & self.mask
        blocks[:, 1] ^= output ^ keys

    def encrypt(self, blocks: np.ndarray, keys: np.ndarray, state_index: int, n_rounds: int) -> None:
        """Encrypt using Simeck.

        Parameters
        ----------
        blocks : np.ndarray
            blocks
        keys : np.ndarray
            keys
        state_index : int
            index of the current state
        n_rounds : int
            number of encryption rounds
        """
        for round_number in range(state_index, state_index + n_rounds):
            self.feistel(blocks, keys[:, 3])
            blocks[:, :] = np.concatenate((blocks[:, 1:], blocks[:, :1]), axis=1)
            self.feistel(keys[:, 2:4], self.constant ^ ((self.z_sequence >> round_number) & 1))
            keys[:, :] = np.concatenate((keys[:, -1:], keys[:, :-1]), axis=1)

    def decrypt(self, blocks: np.ndarray, keys: np.ndarray, state_index: int, n_rounds: int) -> None:
        """Decrypt in-place.

        Parameters
        ----------
        blocks : np.ndarray
            blocks
        keys : np.ndarray
            keys
        state_index : int
            index of the current state
        n_rounds : int
            number of decryption rounds
        """
        for round_number in reversed(range(state_index - n_rounds, state_index)):
            keys[:, :] = np.concatenate((keys[:, 1:], keys[:, :1]), axis=1)
            self.feistel(keys[:, 2:4], self.constant ^ ((self.z_sequence >> round_number) & 1))
            blocks[:, :] = np.concatenate((blocks[:, 1:], blocks[:, :1]), axis=1)
            self.feistel(blocks, keys[:, 3])
