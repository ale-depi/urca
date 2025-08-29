from abc import ABC, abstractmethod

import numpy as np


class Block(ABC):
    def __init__(self, text_size: int, key_size: int) -> None:
        self.text_size = text_size
        self.key_size = key_size
        self.word_size: int = 0
        self.word_type: np.dtype = np.dtype("uint8")
        self.n_text_words: int = 0
        self.n_key_words: int = 0

    @abstractmethod
    def encrypt(
        self, texts: np.ndarray, keys: np.ndarray, state_index: int, n_rounds: int
    ) -> None: ...

    @abstractmethod
    def decrypt(
        self, texts: np.ndarray, keys: np.ndarray, state_index: int, n_rounds: int
    ) -> None: ...
