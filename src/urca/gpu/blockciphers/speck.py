import cupy as cp

from urca import common
from urca.gpu.block import Block


class Speck(Block):
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
        # cupy internals
        self.mask = cp.sum(2 ** cp.arange(self.word_size), dtype=self.word_type)

    def encrypt_function(self, texts: cp.ndarray, keys: cp.ndarray) -> None:
        texts[:, 0] = texts[:, 0] << self.alphac | texts[:, 0] >> self.alpha
        texts[:, 0] += texts[:, 1]
        texts[:, 0] ^= keys
        texts[:, 0] &= self.mask
        texts[:, 1] = texts[:, 1] << self.beta | texts[:, 1] >> self.betac
        texts[:, 1] ^= texts[:, 0]
        texts[:, 1] &= self.mask

    def update_keys(self, keys: cp.ndarray, round_number: int) -> None:
        round_num_array = cp.array([round_number], dtype=self.word_type)
        self.encrypt_function(keys[:, (self.n_key_words - 2) : self.n_key_words], round_num_array)
        keys[:, : (self.n_key_words - 1)] = cp.roll(keys[:, : (self.n_key_words - 1)], 1, axis=1)

    def encrypt(self, texts: cp.ndarray, keys: cp.ndarray, state_index: int, n_rounds: int) -> None:
        for round_number in range(state_index, state_index + n_rounds):
            self.encrypt_function(texts, keys[:, -1])
            self.update_keys(keys, round_number)

    def decrypt_function(self, texts: cp.ndarray, keys: cp.ndarray) -> None:
        texts[:, 1] ^= texts[:, 0]
        texts[:, 1] = texts[:, 1] << self.betac | texts[:, 1] >> self.beta
        texts[:, 1] &= self.mask
        texts[:, 0] ^= keys
        texts[:, 0] = (texts[:, 0] - texts[:, 1]) & self.mask
        texts[:, 0] = texts[:, 0] << self.alpha | texts[:, 0] >> self.alphac
        texts[:, 0] &= self.mask

    def revert_keys(self, keys: cp.ndarray, round_number: int) -> None:
        keys[:, : (self.n_key_words - 1)] = cp.roll(keys[:, : (self.n_key_words - 1)], -1, axis=1)
        round_num_array = cp.array([round_number], dtype=self.word_type)
        self.decrypt_function(keys[:, (self.n_key_words - 2) : self.n_key_words], round_num_array)

    def decrypt(self, texts: cp.ndarray, keys: cp.ndarray, state_index: int, n_rounds: int) -> None:
        for round_number in reversed(range(state_index - n_rounds, state_index)):
            self.revert_keys(keys, round_number)
            self.decrypt_function(texts, keys[:, -1])
