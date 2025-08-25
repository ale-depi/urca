import numpy as np

from urca import common
from urca import constants
from urca.cpu.block import Block


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

    keyfactor_to_keysboxsize = {10: 4, 16: 8}
    keyfactor_to_offset = {10: 0, 16: 1}

    def __init__(self, text_size: int, key_size: int, sbox: tuple = constants.PRESENT_SBOX) -> None:
        super().__init__(text_size, key_size)
        # required
        self.word_size = 1
        self.word_type = np.dtype("uint8")
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
        keys[:, :] = np.roll(keys, -self.key_rotation, axis=1)
        sbox_output = np.unpackbits(self.np_sbox[np.packbits(keys[:, :8], axis=1)], axis=1)
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
            texts[:, :] = np.unpackbits(self.np_sbox[np.packbits(texts, axis=1)], axis=1)
            # pLayer(STATE)
            texts[:, self.permutation] = texts[:, np.arange(self.text_size)]
            # update Key
            self.update_keys(keys, round_number)
        if state_index + n_rounds == self.n_rounds:
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
        sbox_output = np.unpackbits(self.np_inversesbox[np.packbits(keys[:, :8], axis=1)], axis=1)
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
        if state_index == self.n_rounds:
            texts ^= keys[:, : self.text_size]
        for round_number in reversed(range(state_index - n_rounds, state_index)):
            self.revert_keys(keys, round_number)
            texts[:, np.arange(self.text_size)] = texts[:, self.permutation]
            texts[:, :] = np.unpackbits(self.np_inversesbox[np.packbits(texts, axis=1)], axis=1)
            texts ^= keys[:, : self.text_size]
