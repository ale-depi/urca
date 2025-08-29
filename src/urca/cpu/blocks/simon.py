import numpy as np

from urca import common
from urca import constants
from urca.cpu.block import Block


class Simon(Block):
    """The Simon block cipher.

    Parameters
    ----------
    text_size : int, optional, default = 32
        the bit size of the block
    key_size : int, optional, default = 64
        the bit size of the key
    key_rot : tuple, optional, default = (3, 1)
        the rotation amounts in key schedule
    rot : tuple, optional, default = (1, 8, 2)
        the rotation amounts in round schedule
    z_period : int, optional, default = 62
        the period for the z sequence
    z_sequence : int, optional, default = constants.SIMON_Z0
        the bit sequence for key schedule

    """

    def __init__(
        self,
        text_size: int = 32,
        key_size: int = 64,
        key_rot: tuple = (3, 1),
        rot: tuple = (1, 8, 2),
        z_period: int = 62,
        z_sequence: int = constants.SIMON_Z0,
    ) -> None:
        super().__init__(text_size, key_size)
        # required
        self.word_size = text_size // 2
        self.word_type = common.get_dtype(self.word_size)
        self.n_text_words = text_size // self.word_size
        self.n_key_words = key_size // self.word_size
        # cipher specific
        self.constant = 2**self.word_size - 4
        self.key_rot = key_rot
        self.rot = rot
        self.z_period = z_period
        self.z_sequence = z_sequence
        # numpy internals
        self.mask = np.sum(2 ** np.arange(self.word_size), dtype=self.word_type)
        self.np_keyrot = np.array(key_rot, dtype=np.uint8)
        self.np_keyrotc = self.word_size - self.np_keyrot
        self.np_rot = np.array(rot, dtype=np.uint8)
        self.np_rotc = self.word_size - self.np_rot

    def feistel(self, texts: np.ndarray, keys: np.ndarray) -> None:
        """Apply the Simon Feistel function to texts.

        Parameters
        ----------
        texts : np.ndarray
            plaintexts or ciphertexts
        keys : np.ndarray
            keys
        """
        output = (texts[:, 0] << self.np_rot[0] | texts[:, 0] >> self.np_rotc[0]) & self.mask
        output &= (texts[:, 0] << self.np_rot[1] | texts[:, 0] >> self.np_rotc[1]) & self.mask
        output ^= (texts[:, 0] << self.np_rot[2] | texts[:, 0] >> self.np_rotc[2]) & self.mask
        texts[:, 1] ^= output ^ keys

    def update_keys(self, keys: np.ndarray, round_number: int) -> None:
        """Update the keys in-place.

        Parameters
        ----------
        keys : np.ndarray
            keys
        round_number : int
            current round
        """
        temp = keys[:, -2].copy()
        new_words = (0, 0, temp)[self.n_key_words - 2]
        new_words ^= keys[:, 0] << self.np_keyrotc[0] | keys[:, 0] >> self.np_keyrot[0]
        new_words &= self.mask
        new_words ^= new_words << self.np_keyrotc[1] | new_words >> self.np_keyrot[1]
        new_words &= self.mask
        new_words ^= keys[:, -1]
        new_words ^= self.constant ^ ((self.z_sequence >> (round_number % self.z_period)) & 1)
        keys[:, :] = np.roll(keys, 1, axis=1)
        keys[:, 0] = new_words

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
        number_of_rounds : int
            number of encryption rounds
        """
        for round_number in range(state_index, state_index + n_rounds):
            self.feistel(texts, keys[:, -1])
            texts[:, :] = np.roll(texts, 1, axis=1)
            self.update_keys(keys, round_number)

    def revert_keys(self, keys: np.ndarray, round_number: int) -> None:
        """Revert the keys in-place.

        Parameters
        ----------
        keys : np.ndarray
            keys
        round_number : int
            current round
        """
        temp = keys[:, -1].copy()
        new_words = (0, 0, temp)[self.n_key_words - 2]
        new_words ^= keys[:, 1] << self.np_keyrotc[0] | keys[:, 1] >> self.np_keyrot[0]
        new_words &= self.mask
        new_words ^= new_words << self.np_keyrotc[1] | new_words >> self.np_keyrot[1]
        new_words &= self.mask
        new_words ^= self.constant ^ ((self.z_sequence >> (round_number % self.z_period)) & 1)
        keys[:, 0] ^= new_words
        keys[:, :] = np.roll(keys, -1, axis=1)

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
            self.revert_keys(keys, round_number)
            texts[:, :] = np.roll(texts, 1, axis=1)
            self.feistel(texts, keys[:, -1])
