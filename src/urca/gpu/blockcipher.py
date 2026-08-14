from abc import ABC, abstractmethod

import cupy as cp


class BlockCipher(ABC):
    def __init__(self, block_size: int, key_size: int) -> None:
        self.block_size = block_size
        self.key_size = key_size
        self.word_size: int = 0
        self.word_type: cp.dtype = cp.dtype("uint8")
        self.n_block_words: int = 0
        self.n_key_words: int = 0

    @abstractmethod
    def encrypt(
        self, blocks: cp.ndarray, keys: cp.ndarray, state_index: int, n_rounds: int
    ) -> None: ...

    @abstractmethod
    def decrypt(
        self, blocks: cp.ndarray, keys: cp.ndarray, state_index: int, n_rounds: int
    ) -> None: ...
