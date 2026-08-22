import numpy as np

from urca import common, constants
from urca.cpu.blockcipher import BlockCipher


class Present(BlockCipher):
    """The Present block cipher.

    This implementation differs from the general one since it is optimized to save as much as
    possible memory.

    Parameters
    ----------
    block_size : int, optional, default = 64
        the bit size of the block
    key_size : int, optional, default = 80
        the bit size of the key
    sbox : tuple[int], optional, default = :py:data:`urca.constants.PRESENT_SBOX`
        the s-box for the cipher
    """
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
        self.word_type = np.dtype("uint8")
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
        self.np_sbox = np.array(common.gen_bytesbox(sbox), dtype=self.word_type)
        self.np_inversesbox = np.array(common.gen_bytesbox(self.inverse_sbox), dtype=self.word_type)

    def update_keys(self, keys: np.ndarray, round_number: int) -> None:
        """Update the keys in-place.

        Parameters
        ----------
        keys : np.ndarray
            keys
        round_number : int
            current round
        """
        rotate_left = np.concatenate((keys[:, 7:], keys[:, :7]), axis=1)
        rotate_right = np.concatenate((keys[:, 8:], keys[:, :8]), axis=1)
        keys[:, :] = (rotate_left << 5) | (rotate_right >> 3)
        keys[:, 0] = (self.np_sbox[keys[:, 0]] & self.leftmask) | (keys[:, 0] & self.rightmask)
        keys[:, 7] ^= (round_number + 1) >> self.rightshift
        keys[:, 8] ^= ((round_number + 1) & self.countmask) << self.leftshift

    def encrypt(self, blocks: np.ndarray, keys: np.ndarray, state_index: int, n_rounds: int) -> None:
        """Encrypt in-place.

        Parameters
        ----------
        blocks : np.ndarray
            blocks
        keys : np.ndarray
            keys
        state_index : int
            index of the current state
        n_rounds : int
            number of encryption rounds
        """
        for round_number in range(state_index, state_index + n_rounds):
            # addRoundKey(STATE, K_i)
            blocks ^= keys[:, : self.n_block_words]
            # sBoxLayer(STATE)
            blocks[:, :] = self.np_sbox[blocks]
            # pLayer(STATE)
            blocks = blocks.view('>u8')
            sliced = ((blocks >> 3) ^ blocks) & 0x0A0A0A0A0A0A0A0A
            blocks ^= sliced ^ (sliced << 3)
            sliced = ((blocks >> 6) ^ blocks) & 0x00CC00CC00CC00CC
            blocks ^= sliced ^ (sliced << 6)
            sliced = ((blocks >> 12) ^ blocks) & 0x0000F0F00000F0F0
            blocks ^= sliced ^ (sliced << 12)
            sliced = ((blocks >> 24) ^ blocks) & 0x00000000FF00FF00
            blocks ^= sliced ^ (sliced << 24)
            blocks = blocks.view('>u1')
            # update Key
            self.update_keys(keys, round_number)
        if state_index + n_rounds == self.n_rounds:
            blocks ^= keys[:, : self.n_block_words]

    def revert_keys(self, keys: np.ndarray, round_number: int) -> None:
        """Revert the keys in-place.

        Parameters
        ----------
        keys : np.ndarray
            keys
        round_number : int
            current round
        """
        keys[:, 8] ^= ((round_number + 1) & self.countmask) << self.leftshift
        keys[:, 7] ^= (round_number + 1) >> self.rightshift
        sbox_output = self.np_inversesbox[keys[:, 0]]
        keys[:, 0] = (sbox_output & self.leftmask) | (keys[:, 0] & self.rightmask)
        rotate_left = np.concatenate((keys[:, -8:], keys[:, :-8]), axis=1)
        rotate_right = np.concatenate((keys[:, -7:], keys[:, :-7]), axis=1)
        keys[:, :] = (rotate_left << 3) | (rotate_right >> 5)


    def decrypt(self, blocks: np.ndarray, keys: np.ndarray, state_index: int, n_rounds: int) -> None:
        """Dencrypt in-place.

        Parameters
        ----------
        blocks : np.ndarray
            blocks
        keys : np.ndarray
            keys
        state_index : int
            index of the current state
        n_rounds : int
            number of decryption rounds
        """
        if state_index == self.n_rounds:
            blocks ^= keys[:, : self.n_block_words]
        for round_number in reversed(range(state_index - n_rounds, state_index)):
            self.revert_keys(keys, round_number)
            blocks = blocks.view('>u8')
            sliced = ((blocks >> 24) ^ blocks) & 0x00000000FF00FF00
            blocks ^= sliced ^ (sliced << 24)
            sliced = ((blocks >> 12) ^ blocks) & 0x0000F0F00000F0F0
            blocks ^= sliced ^ (sliced << 12)
            sliced = ((blocks >> 6) ^ blocks) & 0x00CC00CC00CC00CC
            blocks ^= sliced ^ (sliced << 6)
            sliced = ((blocks >> 3) ^ blocks) & 0x0A0A0A0A0A0A0A0A
            blocks ^= sliced ^ (sliced << 3)
            blocks = blocks.view('>u1')
            blocks[:, :] = self.np_inversesbox[blocks]
            blocks ^= keys[:, : self.n_block_words]
