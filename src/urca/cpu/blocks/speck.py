import numpy as np

from urca import common
from urca.cpu.block import Block


class Speck(Block):
    """The Speck block cipher.

    Parameters
    ----------
    text_size : int, optional, default = 32
        the bit size of the block
    key_size : int, optional, default = 64
        the bit size of the key
    alpha : int, optional, default = 7
        the first rotation parameter
    beta : int, optional, default = 2
        the second rotation parameter
    """

    def __init__(
        self, text_size: int = 32, key_size: int = 64, alpha: int = 7, beta: int = 2
    ) -> None:
        super().__init__(text_size, key_size)
        # required
        self.word_size = text_size // 2
        self.word_type = common.get_dtype(self.word_size)
        self.n_text_words = text_size // self.word_size
        self.n_key_words = key_size // self.word_size
        # cipher specific
        self.alpha = alpha
        self.alphac = self.word_size - alpha
        self.beta = beta
        self.betac = self.word_size - beta
        # numpy internals
        self.mask = np.sum(2 ** np.arange(self.word_size), dtype=self.word_type)

    def encrypt_function(self, texts: np.ndarray, keys: np.ndarray) -> None:
        """Encrypt one round in-place.

        Parameters
        ----------
        texts : np.ndarray
            plaintexts
        keys : np.ndarray
            keys
        """
        texts[:, 0] = texts[:, 0] << self.alphac | texts[:, 0] >> self.alpha
        texts[:, 0] += texts[:, 1]
        texts[:, 0] ^= keys
        texts[:, 0] &= self.mask
        texts[:, 1] = texts[:, 1] << self.beta | texts[:, 1] >> self.betac
        texts[:, 1] ^= texts[:, 0]
        texts[:, 1] &= self.mask

    def update_keys(self, keys: np.ndarray, round_number: int) -> None:
        """Update the keys in-place.

        Parameters
        ----------
        keys : np.ndarray
            keys
        round_number : int
            current round
        """
        round_num_array = np.array([round_number], dtype=self.word_type)
        self.encrypt_function(keys[:, (self.n_key_words - 2) : self.n_key_words], round_num_array)
        keys[:, : (self.n_key_words - 1)] = np.roll(keys[:, : (self.n_key_words - 1)], 1, axis=1)

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
        for round_number in range(state_index, state_index + n_rounds):
            self.encrypt_function(texts, keys[:, -1])
            self.update_keys(keys, round_number)

    def decrypt_function(self, texts: np.ndarray, keys: np.ndarray) -> None:
        """Decrypt one round in-place.

        Parameters
        ----------
        texts : np.ndarray
            ciphertexts
        keys : np.ndarray
            keys
        """
        texts[:, 1] ^= texts[:, 0]
        texts[:, 1] = texts[:, 1] << self.betac | texts[:, 1] >> self.beta
        texts[:, 1] &= self.mask
        texts[:, 0] ^= keys
        texts[:, 0] = (texts[:, 0] - texts[:, 1]) & self.mask
        texts[:, 0] = texts[:, 0] << self.alpha | texts[:, 0] >> self.alphac
        texts[:, 0] &= self.mask

    def revert_keys(self, keys: np.ndarray, round_number: int) -> None:
        """Revert the keys in-place.

        Parameters
        ----------
        keys : np.ndarray
            keys
        round_number : int
            current round
        """
        keys[:, : (self.n_key_words - 1)] = np.roll(keys[:, : (self.n_key_words - 1)], -1, axis=1)
        round_num_array = np.array([round_number], dtype=self.word_type)
        self.decrypt_function(keys[:, (self.n_key_words - 2) : self.n_key_words], round_num_array)

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
            self.revert_keys(keys, round_number)
            self.decrypt_function(texts, keys[:, -1])
