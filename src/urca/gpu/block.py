import cupy as cp

from abc import ABC, abstractmethod


class Block(ABC):
    def __init__(self, text_size: int, key_size: int) -> None:
        super().__init__()
        self.text_size = text_size
        self.key_size = key_size

    @abstractmethod
    def encrypt(self, texts: cp.ndarray, keys: cp.ndarray, state_index: int, n_rounds: int) -> None:
        pass

    @abstractmethod
    def decrypt(self, texts: cp.ndarray, keys: cp.ndarray, state_index: int, n_rounds: int) -> None:
        pass
