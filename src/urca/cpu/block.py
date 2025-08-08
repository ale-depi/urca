import numpy as np

from abc import ABC, abstractmethod


class Block(ABC):
    def __init__(self, text_size: int, key_size: int) -> None:
        super().__init__()
        self.text_size = text_size
        self.key_size = key_size

    @property
    @abstractmethod
    def word_size(self) -> int: ...

    @property
    @abstractmethod
    def word_type(self) -> np.dtype: ...

    @property
    @abstractmethod
    def n_text_words(self) -> int: ...

    @property
    @abstractmethod
    def n_key_words(self) -> int: ...

    @abstractmethod
    def encrypt(
        self, texts: np.ndarray, keys: np.ndarray, state_index: int, n_rounds: int
    ) -> None: ...

    @abstractmethod
    def decrypt(
        self, texts: np.ndarray, keys: np.ndarray, state_index: int, n_rounds: int
    ) -> None: ...
