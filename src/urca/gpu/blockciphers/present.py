import cupy as cp

from urca import common, constants
from urca.gpu.blockcipher import BlockCipher


class Present(BlockCipher):
    keysize_to_masks = {80: (0xF0, 0x0F), 128: (0xFF, 0x00)}
    keysize_to_rightshift = {80: 1, 128: 2}
    keysize_to_countmask = {80: 0b01, 128: 0b11}
    keysize_to_leftshift = {80: 7, 128: 6}

    def __init__(
        self, block_size: int = 64, key_size: int = 80, sbox: tuple = constants.PRESENT_SBOX
    ) -> None:
        super().__init__(block_size, key_size)
        # required
        self.word_size = 8
        self.word_type = cp.dtype("uint8")
        self.n_block_words = block_size // self.word_size
        self.n_key_words = key_size // self.word_size
        # cipher specific
        self.n_rounds = 31
        self.sbox = sbox
        self.inverse_sbox = common.invert_sbox(sbox)
        self.leftmask, self.rightmask = self.keysize_to_masks[key_size]
        self.rightshift = self.keysize_to_rightshift[key_size]
        self.countmask = self.keysize_to_countmask[key_size]
        self.leftshift = self.keysize_to_leftshift[key_size]
        # numpy internals
        self.np_sbox = cp.array(common.gen_bytesbox(sbox), dtype=self.word_type)
        self.np_inversesbox = cp.array(common.gen_bytesbox(self.inverse_sbox), dtype=self.word_type)

    def update_keys(self, keys: cp.ndarray, round_number: int) -> None:
        rotate_left = cp.concatenate((keys[:, 7:], keys[:, :7]), axis=1)
        rotate_right = cp.concatenate((keys[:, 8:], keys[:, :8]), axis=1)
        keys[:, :] = (rotate_left << 5) | (rotate_right >> 3)
        keys[:, 0] = (self.np_sbox[keys[:, 0]] & self.leftmask) | (keys[:, 0] & self.rightmask)
        keys[:, 7] ^= (round_number + 1) >> self.rightshift
        keys[:, 8] ^= ((round_number + 1) & self.countmask) << self.leftshift

    def encrypt(self, blocks: cp.ndarray, keys: cp.ndarray, state_index: int, n_rounds: int) -> None:
        for round_number in range(state_index, state_index + n_rounds):
            # addRoundKey(STATE, K_i)
            blocks ^= keys[:, : self.n_block_words]
            # sBoxLayer(STATE)
            blocks[:, :] = self.np_sbox[blocks]
            # pLayer(STATE)
            blocks = blocks.view(cp.uint64)
            sliced = ((blocks >> 3) ^ blocks) & 0x0A0A0A0A0A0A0A0A
            blocks ^= sliced ^ (sliced << 3)
            sliced = ((blocks >> 10) ^ blocks) & 0x0033003300330033
            blocks ^= sliced ^ (sliced << 10)
            sliced = ((blocks >> 20) ^ blocks) & 0x00000F0F00000F0F
            blocks ^= sliced ^ (sliced << 20)
            sliced = ((blocks >> 24) ^ blocks) & 0x00000000FF00FF00
            blocks ^= sliced ^ (sliced << 24)
            blocks = blocks.view(cp.uint8)
            # update Key
            self.update_keys(keys, round_number)
        if state_index + n_rounds == self.n_rounds:
            blocks ^= keys[:, : self.n_block_words]

    def revert_keys(self, keys: cp.ndarray, round_number: int) -> None:
        keys[:, 8] ^= ((round_number + 1) & self.countmask) << self.leftshift
        keys[:, 7] ^= (round_number + 1) >> self.rightshift
        sbox_output = self.np_inversesbox[keys[:, 0]]
        keys[:, 0] = (sbox_output & self.leftmask) | (keys[:, 0] & self.rightmask)
        rotate_left = cp.concatenate((keys[:, -8:], keys[:, :-8]), axis=1)
        rotate_right = cp.concatenate((keys[:, -7:], keys[:, :-7]), axis=1)
        keys[:, :] = (rotate_left << 3) | (rotate_right >> 5)


    def decrypt(self, blocks: cp.ndarray, keys: cp.ndarray, state_index: int, n_rounds: int) -> None:
        if state_index == self.n_rounds:
            blocks ^= keys[:, : self.n_block_words]
        for round_number in reversed(range(state_index - n_rounds, state_index)):
            self.revert_keys(keys, round_number)
            blocks = blocks.view(cp.uint64)
            sliced = ((blocks >> 24) ^ blocks) & 0x00000000FF00FF00
            blocks ^= sliced ^ (sliced << 24)
            sliced = ((blocks >> 20) ^ blocks) & 0x00000F0F00000F0F
            blocks ^= sliced ^ (sliced << 20)
            sliced = ((blocks >> 10) ^ blocks) & 0x0033003300330033
            blocks ^= sliced ^ (sliced << 10)
            sliced = ((blocks >> 3) ^ blocks) & 0x0A0A0A0A0A0A0A0A
            blocks ^= sliced ^ (sliced << 3)
            blocks = blocks.view(cp.uint8)
            blocks[:, :] = self.np_inversesbox[blocks]
            blocks ^= keys[:, : self.n_block_words]
