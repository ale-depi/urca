import numpy as np

from abc import ABC, abstractmethod


class Block(ABC):
    def __init__(self, text_size: int, key_size: int) -> None:
        super().__init__()
        self.text_size = text_size
        self.key_size = key_size

    @abstractmethod
    def encrypt(self, texts: np.ndarray, keys: np.ndarray, state_index: int, n_rounds: int) -> None:
        pass

    @abstractmethod
    def decrypt(self, texts: np.ndarray, keys: np.ndarray, state_index: int, n_rounds: int) -> None:
        pass
