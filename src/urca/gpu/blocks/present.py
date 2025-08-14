import cupy as cp

from urca import common
from urca.gpu.block import Block


class Present(Block):
    keyfactor_to_keysboxsize = {10: 4, 16: 8}
    keyfactor_to_offset = {10: 0, 16: 1}

    def __init__(
        self, text_size: int = 64, key_size: int = 80, sbox: tuple[int] = common.PRESENT_SBOX
    ) -> None:
        super().__init__(text_size, key_size)
        self.sbox = cp.array(sbox, dtype=self.word_type)
        self.inverse_sbox = cp.array(common.invert_sbox(sbox), dtype=self.word_type)
        self.permutation = tuple((i // 4) + (text_size // 4) * (i % 4) for i in range(text_size))
        self.key_factor = key_size // (text_size // 8)
        self.key_rotation = self.key_factor * 6 + 1
        self.key_sbox_size = self.keyfactor_to_keysboxsize[self.key_factor]
        self.counter_low = self.key_factor * 6 + self.keyfactor_to_offset[self.key_factor]
        self.counter_high = self.counter_low + 5

    @property
    def word_size(self) -> int:
        return 1

    @property
    def word_type(self) -> cp.dtype:
        return cp.uint8

    @property
    def n_text_words(self) -> int:
        return self.text_size

    @property
    def n_key_words(self) -> int:
        return self.key_size

    def __eq__(self, other):
        if not isinstance(other, Present):
            return False

        text_size_eq = self.text_size == other.text_size
        key_size_eq = self.key_size == other.key_size
        sbox_eq = self.sbox == other.sbox

        return text_size_eq and key_size_eq and sbox_eq

    def __hash__(self) -> None:
        return hash((self.text_size, self.key_size, self.sbox))

    def update_keys(self, keys: cp.ndarray, round_number: int) -> None:
        """Update the keys in-place.

        Parameters
        ----------
        keys : cp.ndarray
            keys
        round_number : int
            current round
        """
        keys[:, :] = cp.roll(keys, -self.key_rotation, axis=1)
        sbox_output = cp.unpackbits(self.sbox[cp.packbits(keys[:, :8])]).reshape(-1, 8)
        keys[:, : self.key_sbox_size] = sbox_output[:, : self.key_sbox_size]
        round_counter = cp.array(tuple(map(int, f"{round_number + 1:05b}")), dtype=self.word_type)
        keys[:, self.counter_low : self.counter_high] ^= round_counter

    def encrypt(self, texts: cp.ndarray, keys: cp.ndarray, state_index: int, n_rounds: int) -> None:
        """Encrypt in-place.

        Parameters
        ----------
        texts : cp.ndarray
            plaintexts
        keys : cp.ndarray
            keys
        state_index : int
            current round
        n_rounds : int
            number of encryption rounds
        """
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
        texts ^= keys[:, : self.text_size]

    def revert_keys(self, keys: cp.ndarray, round_number: int) -> None:
        """Revert the keys in-place.

        Parameters
        ----------
        keys : cp.ndarray
            keys
        round_number : int
            current round
        """
        round_counter = cp.array(tuple(map(int, f"{round_number + 1:05b}")), dtype=self.word_type)
        keys[:, self.counter_low : self.counter_high] ^= round_counter
        sbox_output = cp.unpackbits(self.inverse_sbox[cp.packbits(keys[:, :8])]).reshape(-1, 8)
        keys[:, : self.key_sbox_size] = sbox_output[:, : self.key_sbox_size]
        keys[:, :] = cp.roll(keys, self.key_rotation, axis=1)

    def decrypt(self, texts: cp.ndarray, keys: cp.ndarray, state_index: int, n_rounds: int) -> None:
        """Dencrypt in-place.

        Parameters
        ----------
        texts : cp.ndarray
            ciphertexts
        keys : cp.ndarray
            keys
        current_round : int
            current round
        n_rounds : int
            number of decryption rounds
        """
        texts ^= keys[:, : self.text_size]
        for round_number in reversed(range(state_index - n_rounds, state_index)):
            self.revert_keys(keys, round_number)
            texts[:, cp.arange(self.text_size)] = texts[:, self.permutation]
            sbox_output = cp.unpackbits(self.reversed_sbox[cp.packbits(texts)])
            texts[:, :] = sbox_output.reshape(-1, self.text_size)
            texts ^= keys[:, : self.text_size]
