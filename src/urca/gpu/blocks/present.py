from dataclasses import dataclass
from functools import cached_property

import cupy as cp

from urca import common
from urca import constants
from urca.gpu.block import Block


@dataclass
class Present(Block):
    text_size: int = 64
    key_size: int = 80
    sbox: tuple[int, ...] = constants.PRESENT_SBOX
    _n_rounds: int = 31

    @property
    def n_rounds(self) -> int:
        return self._n_rounds

    @n_rounds.setter
    def n_rounds(self, n_rounds: int) -> None:
        self._n_rounds = n_rounds

    @cached_property
    def npsbox(self):
        byte_sbox = common.gen_bytesbox(self.sbox)
        return cp.array(byte_sbox, dtype=self.word_type)

    @cached_property
    def inverse_cpsbox(self):
        byte_inverse_sbox = common.gen_bytesbox(common.invert_sbox(self.sbox))
        return cp.array(byte_inverse_sbox, dtype=self.word_type)

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
    def word_type(self) -> cp.dtype:
        return cp.dtype("uint8")

    @cached_property
    def n_text_words(self) -> int:
        return self.text_size

    @cached_property
    def n_key_words(self) -> int:
        return self.key_size

    def update_keys(self, keys: cp.ndarray, round_number: int) -> None:
        keys[:, :] = cp.roll(keys, -self.key_rotation, axis=1)
        sbox_output = cp.unpackbits(self.sbox[cp.packbits(keys[:, :8])]).reshape(-1, 8)
        keys[:, : self.key_sbox_size] = sbox_output[:, : self.key_sbox_size]
        round_counter = cp.array(tuple(map(int, f"{round_number + 1:05b}")), dtype=self.word_type)
        keys[:, self.counter_low : self.counter_high] ^= round_counter

    def encrypt(self, texts: cp.ndarray, keys: cp.ndarray, state_index: int, n_rounds: int) -> None:
        for round_number in range(state_index, state_index + n_rounds):
            # addRoundKey(STATE, K_i)
            texts ^= keys[:, : self.text_size]
            # sBoxLayer(STATE)
            sbox_output = cp.unpackbits(self.sbox[cp.packbits(texts)])
            texts[:, :] = sbox_output.reshape(-1, self.text_size)
            # pLayer(STATE)
            texts[:, self.permutation] = texts[:, cp.arange(self.text_size)]
            # update Key
            self.update_keys(keys, round_number)
        if state_index + n_rounds == self._n_rounds:
            texts ^= keys[:, : self.text_size]

    def revert_keys(self, keys: cp.ndarray, round_number: int) -> None:
        round_counter = cp.array(tuple(map(int, f"{round_number + 1:05b}")), dtype=self.word_type)
        keys[:, self.counter_low : self.counter_high] ^= round_counter
        sbox_output = cp.unpackbits(self.inverse_cpsbox[cp.packbits(keys[:, :8])]).reshape(-1, 8)
        keys[:, : self.key_sbox_size] = sbox_output[:, : self.key_sbox_size]
        keys[:, :] = cp.roll(keys, self.key_rotation, axis=1)

    def decrypt(self, texts: cp.ndarray, keys: cp.ndarray, state_index: int, n_rounds: int) -> None:
        if state_index == self._n_rounds:
            texts ^= keys[:, : self.text_size]
        for round_number in reversed(range(state_index - n_rounds, state_index)):
            self.revert_keys(keys, round_number)
            texts[:, cp.arange(self.text_size)] = texts[:, self.permutation]
            sbox_output = cp.unpackbits(self.reversed_sbox[cp.packbits(texts)])
            texts[:, :] = sbox_output.reshape(-1, self.text_size)
            texts ^= keys[:, : self.text_size]
