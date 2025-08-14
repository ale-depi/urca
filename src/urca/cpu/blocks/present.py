from dataclasses import dataclass
from functools import cached_property

import numpy as np

from urca import common
from urca import constants
from urca.cpu.block import Block


@dataclass(frozen=True)
class Present(Block):
    """The Present block cipher.

    Parameters
    ----------
    text_size : int, optional, default = 64
        the bit size of the block
    key_size : int, optional, default = 80
        the bit size of the key
    sbox : tuple[int], optional, default = :py:data:`urca.constants.PRESENT_SBOX`
        the s-box for the cipher
    """

    text_size: int = 64
    key_size: int = 80
    sbox: tuple[int] = constants.PRESENT_SBOX

    @cached_property
    def npsbox(self):
        byte_sbox = common.gen_bytesbox(self.sbox)
        return np.array(byte_sbox, dtype=self.word_type)

    @cached_property
    def inverse_npsbox(self):
        byte_inverse_sbox = common.gen_bytesbox(common.invert_sbox(self.sbox))
        return np.array(byte_inverse_sbox, dtype=self.word_type)

    @cached_property
    def permutation(self):
        return tuple((i // 4) + (self.text_size // 4) * (i % 4) for i in range(self.text_size))

    @cached_property
    def key_factor(self):
        return self.key_size // (self.text_size // 8)

    @cached_property
    def key_rotation(self):
        return self.key_factor * 6 + 1

    @cached_property
    def key_sbox_size(self):
        keyfactor_to_keysboxsize = {10: 4, 16: 8}
        return keyfactor_to_keysboxsize[self.key_factor]

    @cached_property
    def counter_low(self):
        keyfactor_to_offset = {10: 0, 16: 1}
        return self.key_factor * 6 + keyfactor_to_offset[self.key_factor]

    @cached_property
    def counter_high(self):
        return self.counter_low + 5

    @cached_property
    def word_size(self) -> int:
        return 1

    @cached_property
    def word_type(self) -> np.dtype:
        return np.uint8

    @cached_property
    def n_text_words(self) -> int:
        return self.text_size

    @cached_property
    def n_key_words(self) -> int:
        return self.key_size

    def update_keys(self, keys: np.ndarray, round_number: int) -> None:
        """Update the keys in-place.

        Parameters
        ----------
        keys : np.ndarray
            keys
        round_number : int
            current round
        """
        keys[:, :] = np.roll(keys, -self.key_rotation, axis=1)
        sbox_output = np.unpackbits(self.npsbox[np.packbits(keys[:, :8], axis=1)], axis=1)
        keys[:, : self.key_sbox_size] = sbox_output[:, : self.key_sbox_size]
        round_counter = np.array(tuple(map(int, f"{round_number + 1:05b}")), dtype=self.word_type)
        keys[:, self.counter_low : self.counter_high] ^= round_counter

    def encrypt(self, texts: np.ndarray, keys: np.ndarray, state_index: int, n_rounds: int) -> None:
        """Encrypt in-place.

        Parameters
        ----------
        texts : np.ndarray
            plaintexts
        keys : np.ndarray
            keys
        state_index : int
            index of the current state
        n_rounds : int
            number of encryption rounds
        """
        for round_number in range(state_index, state_index + n_rounds):
            # addRoundKey(STATE, K_i)
            texts ^= keys[:, : self.text_size]
            # sBoxLayer(STATE)
            texts[:, :] = np.unpackbits(self.npsbox[np.packbits(texts, axis=1)], axis=1)
            # pLayer(STATE)
            texts[:, self.permutation] = texts[:, np.arange(self.text_size)]
            # update Key
            self.update_keys(keys, round_number)
        texts ^= keys[:, : self.text_size]

    def revert_keys(self, keys: np.ndarray, round_number: int) -> None:
        """Revert the keys in-place.

        Parameters
        ----------
        keys : np.ndarray
            keys
        round_number : int
            current round
        """
        round_counter = np.array(tuple(map(int, f"{round_number + 1:05b}")), dtype=self.word_type)
        keys[:, self.counter_low : self.counter_high] ^= round_counter
        sbox_output = np.unpackbits(self.inverse_npsbox[np.packbits(keys[:, :8], axis=1)], axis=1)
        keys[:, : self.key_sbox_size] = sbox_output[:, : self.key_sbox_size]
        keys[:, :] = np.roll(keys, self.key_rotation, axis=1)

    def decrypt(self, texts: np.ndarray, keys: np.ndarray, state_index: int, n_rounds: int) -> None:
        """Dencrypt in-place.

        Parameters
        ----------
        texts : np.ndarray
            ciphertexts
        keys : np.ndarray
            keys
        state_index : int
            index of the current state
        n_rounds : int
            number of decryption rounds
        """
        texts ^= keys[:, : self.text_size]
        for round_number in reversed(range(state_index - n_rounds, state_index)):
            self.revert_keys(keys, round_number)
            texts[:, np.arange(self.text_size)] = texts[:, self.permutation]
            texts[:, :] = np.unpackbits(self.inverse_npsbox[np.packbits(texts, axis=1)], axis=1)
            texts ^= keys[:, : self.text_size]
