from abc import ABC, abstractmethod

import numpy as np


class BlockCipher(ABC):
    def __init__(self, block_size: int, key_size: int) -> None:
        self.block_size = block_size
        self.key_size = key_size
        self.word_size: int = 0
        self.word_type: np.dtype = np.dtype("uint8")
        self.n_block_words: int = 0
        self.n_key_words: int = 0

    @abstractmethod
    def encrypt(
        self, blocks: np.ndarray, keys: np.ndarray, state_index: int, n_rounds: int
    ) -> None: ...

    @abstractmethod
    def decrypt(
        self, blocks: np.ndarray, keys: np.ndarray, state_index: int, n_rounds: int
    ) -> None: ...
