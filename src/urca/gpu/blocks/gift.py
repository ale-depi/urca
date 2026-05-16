import cupy as cp

from urca import common
from urca.cpu.block import Block


class Gift(Block):
    # fmt: off
    key_permutation = (
        110, 111, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 116, 117, 118,
        119, 120, 121, 122, 123, 124, 125, 126, 127, 112, 113, 114, 115, 0, 1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
        55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
        78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
    )
    # fmt: on

    constants = cp.array(
        (
            (1, 0, 0, 0, 0, 0, 1),
            (1, 0, 0, 0, 0, 1, 1),
            (1, 0, 0, 0, 1, 1, 1),
            (1, 0, 0, 1, 1, 1, 1),
            (1, 0, 1, 1, 1, 1, 1),
            (1, 1, 1, 1, 1, 1, 0),
            (1, 1, 1, 1, 1, 0, 1),
            (1, 1, 1, 1, 0, 1, 1),
            (1, 1, 1, 0, 1, 1, 1),
            (1, 1, 0, 1, 1, 1, 1),
            (1, 0, 1, 1, 1, 1, 0),
            (1, 1, 1, 1, 1, 0, 0),
            (1, 1, 1, 1, 0, 0, 1),
            (1, 1, 1, 0, 0, 1, 1),
            (1, 1, 0, 0, 1, 1, 1),
            (1, 0, 0, 1, 1, 1, 0),
            (1, 0, 1, 1, 1, 0, 1),
            (1, 1, 1, 1, 0, 1, 0),
            (1, 1, 1, 0, 1, 0, 1),
            (1, 1, 0, 1, 0, 1, 1),
            (1, 0, 1, 0, 1, 1, 0),
            (1, 1, 0, 1, 1, 0, 0),
            (1, 0, 1, 1, 0, 0, 0),
            (1, 1, 1, 0, 0, 0, 0),
            (1, 1, 0, 0, 0, 0, 1),
            (1, 0, 0, 0, 0, 1, 0),
            (1, 0, 0, 0, 1, 0, 1),
            (1, 0, 0, 1, 0, 1, 1),
            (1, 0, 1, 0, 1, 1, 1),
            (1, 1, 0, 1, 1, 1, 0),
            (1, 0, 1, 1, 1, 0, 0),
            (1, 1, 1, 1, 0, 0, 0),
            (1, 1, 1, 0, 0, 0, 1),
            (1, 1, 0, 0, 0, 1, 1),
            (1, 0, 0, 0, 1, 1, 0),
            (1, 0, 0, 1, 1, 0, 1),
            (1, 0, 1, 1, 0, 1, 1),
            (1, 1, 1, 0, 1, 1, 0),
            (1, 1, 0, 1, 1, 0, 1),
            (1, 0, 1, 1, 0, 1, 0),
        ),
        dtype=cp.uint8,
    )
    constant_positions = (0, -24, -20, -16, -12, -8, -4)
    textsize_to_positions = {
        64: tuple(range(2, 64, 4)) + tuple(range(3, 64, 4)),
        128: tuple(range(1, 128, 4)) + tuple(range(2, 128, 4)),
    }
    textsize_to_rkpositions = {
        64: tuple(range(96, 112)) + tuple(range(112, 128)),
        128: tuple(range(32, 64)) + tuple(range(96, 128)),
    }
    sbox = (0x1, 0xA, 0x4, 0xC, 0x6, 0xF, 0x3, 0x9, 0x2, 0xD, 0xB, 0x7, 0x5, 0x0, 0x8, 0xE)

    def __init__(
        self,
        text_size: int = 64,
        key_size: int = 128,
        sbox: tuple[int, ...] = sbox,
    ) -> None:
        super().__init__(text_size, key_size)
        # required
        self.word_size = 1
        self.word_type = cp.dtype("uint8")
        self.n_text_words = text_size
        self.n_key_words = key_size
        # cipher specific
        self.sbox = sbox
        self.inverse_sbox = common.invert_sbox(sbox)
        self.permutation = tuple(
            text_size - 1 - p for p in reversed(common.gen_gift_permutation(text_size))
        )
        self.positions = self.textsize_to_positions[text_size]
        self.round_key_positions = self.textsize_to_rkpositions[text_size]
        # numpy internals
        self.cp_sbox = cp.array(common.gen_bytesbox(sbox), dtype=self.word_type)
        self.cp_inversesbox = cp.array(common.gen_bytesbox(self.inverse_sbox), dtype=self.word_type)

    def encrypt(self, texts: cp.ndarray, keys: cp.ndarray, state_index: int, n_rounds: int) -> None:
        for round_number in range(state_index, state_index + n_rounds):
            # SubCells
            sbox_output = cp.unpackbits(self.cp_sbox[cp.packbits(texts)])
            texts[:, :] = sbox_output.reshape(-1, self.text_size)
            # PermBits
            texts[:, self.permutation] = texts[:, :]
            # AddRoundKey
            texts[:, self.positions] ^= keys[:, self.round_key_positions]
            texts[:, self.constant_positions] ^= self.constants[round_number]
            # update Key
            keys[:, :] = keys[:, self.key_permutation]

    def decrypt(self, texts: cp.ndarray, keys: cp.ndarray, state_index: int, n_rounds: int) -> None:
        for round_number in reversed(range(state_index - n_rounds, state_index)):
            keys[:, self.key_permutation] = keys[:, :]
            texts[:, self.positions] ^= keys[:, self.round_key_positions]
            texts[:, self.constant_positions] ^= self.constants[round_number]
            texts[:, :] = texts[:, self.permutation]
            sbox_output = cp.unpackbits(self.cp_inversesbox[cp.packbits(texts)])
            texts[:, :] = sbox_output.reshape(-1, self.text_size)
