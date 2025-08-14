import numpy as np

from abc import ABC, abstractmethod
from functools import cached_property


class Block(ABC):
    @cached_property
    @abstractmethod
    def word_size(self) -> int: ...

    @cached_property
    @abstractmethod
    def word_type(self) -> np.dtype: ...

    @cached_property
    @abstractmethod
    def n_text_words(self) -> int: ...

    @cached_property
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
