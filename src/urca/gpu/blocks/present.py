import cupy as cp

from urca import common
from urca import constants
from urca.gpu.block import Block


class Present(Block):
    keyfactor_to_keysboxsize = {10: 4, 16: 8}
    keyfactor_to_offset = {10: 0, 16: 1}

    def __init__(self, text_size: int, key_size: int, sbox: tuple = constants.PRESENT_SBOX) -> None:
        super().__init__(text_size, key_size)
        # required
        self.word_size = 1
        self.word_type = cp.dtype("uint8")
        self.n_text_words = text_size
        self.n_key_words = key_size
        # cipher specific
        self.n_rounds = 31
        self.sbox = sbox
        self.inverse_sbox = common.invert_sbox(sbox)
        self.permutation = tuple(
            (i // 4) + (self.text_size // 4) * (i % 4) for i in range(self.text_size)
        )
        self.key_factor = key_size // (text_size // 8)
        self.key_rotation = self.key_factor * 6 + 1
        self.key_sbox_size = self.keyfactor_to_keysboxsize[self.key_factor]
        self.counter_low = self.key_factor * 6 + self.keyfactor_to_offset[self.key_factor]
        self.counter_high = self.counter_low + 5
        # cupy internals
        self.cp_sbox = cp.array(common.gen_bytesbox(sbox), dtype=self.word_type)
        self.cp_inversesbox = cp.array(common.gen_bytesbox(self.inverse_sbox), dtype=self.word_type)

    def update_keys(self, keys: cp.ndarray, round_number: int) -> None:
        keys[:, :] = cp.roll(keys, -self.key_rotation, axis=1)
        sbox_output = cp.unpackbits(self.cp_sbox[cp.packbits(keys[:, :8])]).reshape(-1, 8)
        keys[:, : self.key_sbox_size] = sbox_output[:, : self.key_sbox_size]
        round_counter = cp.array(tuple(map(int, f"{round_number + 1:05b}")), dtype=self.word_type)
        keys[:, self.counter_low : self.counter_high] ^= round_counter

    def encrypt(self, texts: cp.ndarray, keys: cp.ndarray, state_index: int, n_rounds: int) -> None:
        for round_number in range(state_index, state_index + n_rounds):
            # addRoundKey(STATE, K_i)
            texts ^= keys[:, : self.text_size]
            # sBoxLayer(STATE)
            sbox_output = cp.unpackbits(self.cp_sbox[cp.packbits(texts)])
            texts[:, :] = sbox_output.reshape(-1, self.text_size)
            # pLayer(STATE)
            texts[:, self.permutation] = texts[:, cp.arange(self.text_size)]
            # update Key
            self.update_keys(keys, round_number)
        if state_index + n_rounds == self.n_rounds:
            texts ^= keys[:, : self.text_size]

    def revert_keys(self, keys: cp.ndarray, round_number: int) -> None:
        round_counter = cp.array(tuple(map(int, f"{round_number + 1:05b}")), dtype=self.word_type)
        keys[:, self.counter_low : self.counter_high] ^= round_counter
        sbox_output = cp.unpackbits(self.cp_inversesbox[cp.packbits(keys[:, :8])]).reshape(-1, 8)
        keys[:, : self.key_sbox_size] = sbox_output[:, : self.key_sbox_size]
        keys[:, :] = cp.roll(keys, self.key_rotation, axis=1)

    def decrypt(self, texts: cp.ndarray, keys: cp.ndarray, state_index: int, n_rounds: int) -> None:
        if state_index == self.n_rounds:
            texts ^= keys[:, : self.text_size]
        for round_number in reversed(range(state_index - n_rounds, state_index)):
            self.revert_keys(keys, round_number)
            texts[:, cp.arange(self.text_size)] = texts[:, self.permutation]
            sbox_output = cp.unpackbits(self.cp_inversesbox[cp.packbits(texts)])
            texts[:, :] = sbox_output.reshape(-1, self.text_size)
            texts ^= keys[:, : self.text_size]
