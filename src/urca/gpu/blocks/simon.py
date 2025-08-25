import cupy as cp

from urca import common
from urca import constants
from urca.cpu.block import Block


class Simon(Block):
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
        self.word_size = self.text_size // 2
        self.word_type = common.get_dtype(self.word_size)
        self.n_text_words = self.text_size // self.word_size
        self.n_key_words = self.key_size // self.word_size
        # cipher specific
        self.constant = 2**self.word_size - 4
        self.key_rot = key_rot
        self.rot = rot
        self.z_period = z_period
        self.z_sequence = z_sequence
        # cupy internals
        self.mask = cp.sum(2 ** cp.arange(self.word_size), dtype=self.word_type)
        self.cp_keyrot = cp.array(self.key_rot, dtype=cp.uint8)
        self.cp_keyrotc = self.word_size - self.cp_keyrot
        self.cp_rot = cp.array(self.rot, dtype=cp.uint8)
        self.cp_rotc = self.word_size - self.cp_rot

    def feistel(self, texts: cp.ndarray, keys: cp.ndarray) -> None:
        """Apply the Simon Feistel function to texts.

        Parameters
        ----------
        texts : cp.ndarray
            plaintexts or ciphertexts
        keys : cp.ndarray
            keys
        """
        output = (texts[:, 0] << self.cp_rot[0] | texts[:, 0] >> self.cp_rotc[0]) & self.mask
        output &= (texts[:, 0] << self.cp_rot[1] | texts[:, 0] >> self.cp_rotc[1]) & self.mask
        output ^= (texts[:, 0] << self.cp_rot[2] | texts[:, 0] >> self.cp_rotc[2]) & self.mask
        texts[:, 1] ^= output ^ keys

    def update_keys(self, keys: cp.ndarray, round_number: int) -> None:
        """Update the keys in-place.

        Parameters
        ----------
        keys : cp.ndarray
            keys
        round_number : int
            current round
        """
        temp = keys[:, -2].copy()
        new_words = (0, 0, temp)[self.n_key_words - 2]
        new_words ^= keys[:, 0] << self.cp_keyrotc[0] | keys[:, 0] >> self.cp_keyrot[0]
        new_words &= self.mask
        new_words ^= new_words << self.cp_keyrotc[1] | new_words >> self.cp_keyrot[1]
        new_words &= self.mask
        new_words ^= keys[:, -1]
        new_words ^= self.constant ^ ((self.z_sequence >> (round_number % self.z_period)) & 1)
        keys[:, :] = cp.roll(keys, 1, axis=1)
        keys[:, 0] = new_words

    def encrypt(self, texts: cp.ndarray, keys: cp.ndarray, state_index: int, n_rounds: int) -> None:
        """Encrypt in-place.

        Parameters
        ----------
        texts : cp.ndarray
            plaintexts
        keys : cp.ndarray
            keys
        state_index : int
            index of the current state
        number_of_rounds : int
            number of encryption rounds
        """
        for round_number in range(state_index, state_index + n_rounds):
            self.feistel(texts, keys[:, -1])
            texts[:, :] = cp.roll(texts, 1, axis=1)
            self.update_keys(keys, round_number)

    def revert_keys(self, keys: cp.ndarray, round_number: int) -> None:
        """Revert the keys in-place.

        Parameters
        ----------
        keys : cp.ndarray
            keys
        round_number : int
            current round
        """
        temp = keys[:, -1].copy()
        new_words = (0, 0, temp)[self.n_key_words - 2]
        new_words ^= keys[:, 1] << self.cp_keyrotc[0] | keys[:, 1] >> self.cp_keyrot[0]
        new_words &= self.mask
        new_words ^= new_words << self.cp_keyrotc[1] | new_words >> self.cp_keyrot[1]
        new_words &= self.mask
        new_words ^= self.constant ^ ((self.z_sequence >> (round_number % self.z_period)) & 1)
        keys[:, 0] ^= new_words
        keys[:, :] = cp.roll(keys, -1, axis=1)

    def decrypt(self, texts: cp.ndarray, keys: cp.ndarray, state_index: int, n_rounds: int) -> None:
        """Decrypt in-place.

        Parameters
        ----------
        texts : cp.ndarray
            ciphertexts
        keys : cp.ndarray
            keys
        state_index : int
            index of the current state
        n_rounds : int
            number of decryption rounds
        """
        for round_number in reversed(range(state_index - n_rounds, state_index)):
            self.revert_keys(keys, round_number)
            texts[:, :] = cp.roll(texts, 1, axis=1)
            self.feistel(texts, keys[:, -1])
