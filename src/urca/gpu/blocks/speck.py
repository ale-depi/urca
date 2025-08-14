import cupy as cp

from urca.gpu import utilities
from urca.gpu.block import Block


class Speck(Block):
    def __init__(
        self,
        text_size: int = 32,
        key_size: int = 64,
        alpha: int = 7,
        beta: int = 2,
    ) -> None:
        super().__init__(text_size, key_size)
        self.alpha = alpha
        self.alphac = self.word_size - alpha
        self.beta = beta
        self.betac = self.word_size - beta
        self.mask = cp.sum(2 ** cp.arange(self.word_size), dtype=self.word_type)

    @property
    def word_size(self) -> int:
        return self.text_size // 2

    @property
    def word_type(self) -> cp.dtype:
        return utilities.get_dtype(self.word_size)

    @property
    def n_text_words(self) -> int:
        return self.text_size // self.word_size

    @property
    def n_key_words(self) -> int:
        return self.key_size // self.word_size

    def __eq__(self, other) -> bool:
        if not isinstance(other, Speck):
            return False

        text_size_eq = self.text_size == other.text_size
        key_size_eq = self.key_size == other.key_size
        alpha_eq = self.alpha == other.alpha
        beta_eq = self.beta == other.beta

        return text_size_eq and key_size_eq and alpha_eq and beta_eq

    def __hash__(self) -> None:
        return hash((self.text_size, self.key_size, self.alpha, self.beta))

    def encrypt_function(self, texts: cp.ndarray, keys: cp.ndarray) -> None:
        """Encrypt one round in-place.

        Parameters
        ----------
        texts : cp.ndarray
            plaintexts
        keys : cp.ndarray
            keys
        """
        texts[:, 0] = texts[:, 0] << self.alphac | texts[:, 0] >> self.alpha
        texts[:, 0] += texts[:, 1]
        texts[:, 0] ^= keys
        texts[:, 0] &= self.mask
        texts[:, 1] = texts[:, 1] << self.beta | texts[:, 1] >> self.betac
        texts[:, 1] ^= texts[:, 0]
        texts[:, 1] &= self.mask

    def update_keys(self, keys: cp.ndarray, round_number: int) -> None:
        """Update the keys in-place.

        Parameters
        ----------
        keys : cp.ndarray
            keys
        round_number : int
            current round
        """
        self.encrypt_function(keys[:, (self.n_key_words - 2) : self.n_key_words], round_number)
        keys[:, : (self.n_key_words - 1)] = cp.roll(keys[:, : (self.n_key_words - 1)], 1, axis=1)

    def encrypt(self, texts: cp.ndarray, keys: cp.ndarray, state_index: int, n_rounds: int) -> None:
        """Encrypt in-place.

        Parameters
        ----------
        texts : cp.ndarray
            plaintexts
        keys : cp.ndarray
            keys
        current_round : int
            current round
        n_rounds : int
            number of encryption rounds
        """
        for round_number in range(state_index, state_index + n_rounds):
            self.encrypt_function(texts, keys[:, -1])
            self.update_keys(keys, round_number)

    def decrypt_function(self, texts: cp.ndarray, keys: cp.ndarray) -> None:
        """Decrypt one round in-place.

        Parameters
        ----------
        texts : cp.ndarray
            ciphertexts
        keys : cp.ndarray
            keys
        """
        texts[:, 1] ^= texts[:, 0]
        texts[:, 1] = texts[:, 1] << self.betac | texts[:, 1] >> self.beta
        texts[:, 1] &= self.mask
        texts[:, 0] ^= keys
        texts[:, 0] = (texts[:, 0] - texts[:, 1]) & self.mask
        texts[:, 0] = texts[:, 0] << self.alpha | texts[:, 0] >> self.alphac
        texts[:, 0] &= self.mask

    def revert_keys(self, keys: cp.ndarray, round_number: int) -> None:
        """Revert the keys in-place.

        Parameters
        ----------
        keys : cp.ndarray
            keys
        round_number : int
            current round
        """
        keys[:, : (self.n_key_words - 1)] = cp.roll(keys[:, : (self.n_key_words - 1)], -1, axis=1)
        self.decrypt_function(keys[:, (self.n_key_words - 2) : self.n_key_words], round_number)

    def decrypt(self, texts: cp.ndarray, keys: cp.ndarray, state_index: int, n_rounds: int) -> None:
        """Dencrypt in-place.

        Parameters
        ----------
        texts : cp.ndarray
            ciphertexts
        keys : cp.ndarray
            keys
        current_round : int
            current round
        n_rounds : int
            number of decryption rounds
        """
        for round_number in reversed(range(state_index - n_rounds, state_index)):
            self.revert_keys(keys, round_number)
            self.decrypt_function(texts, keys[:, -1])
