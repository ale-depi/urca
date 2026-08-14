import cupy as cp

from urca import common, constants
from urca.gpu.blockcipher import BlockCipher


class Simeck(BlockCipher):
    def __init__(
        self,
        block_size: int = 32,
        key_size: int = 64,
        rot: tuple = (5, 1),
        z_sequence: int = constants.SIMECK_Z0,
    ) -> None:
        super().__init__(block_size, key_size)
        # required
        self.word_size = block_size // 2
        self.word_type = common.get_dtype(self.word_size)
        self.n_block_words = block_size // self.word_size
        self.n_key_words = key_size // self.word_size
        # cipher specific
        self.constant = 2**self.word_size - 4
        self.rot = rot
        self.z_sequence = z_sequence
        # cupy internals
        self.mask = cp.sum(2 ** cp.arange(self.word_size), dtype=self.word_type)
        self.cp_rot = cp.array(rot, dtype=cp.uint8)
        self.cp_rotc = self.word_size - self.cp_rot

    def feistel(self, blocks: cp.ndarray, keys: cp.ndarray) -> None:
        output = (blocks[:, 0] << self.cp_rot[0] | blocks[:, 0] >> self.cp_rotc[0]) & self.mask
        output &= blocks[:, 0]
        output ^= (blocks[:, 0] << self.cp_rot[1] | blocks[:, 0] >> self.cp_rotc[1]) & self.mask
        blocks[:, 1] ^= output ^ keys

    def encrypt(self, blocks: cp.ndarray, keys: cp.ndarray, state_index: int, n_rounds: int) -> None:
        for round_number in range(state_index, state_index + n_rounds):
            self.feistel(blocks, keys[:, 3])
            blocks[:, :] = cp.roll(blocks, 1, axis=1)
            self.feistel(keys[:, 2:4], self.constant ^ ((self.z_sequence >> round_number) & 1))
            keys[:, :] = cp.roll(keys, 1, axis=1)

    def decrypt(self, blocks: cp.ndarray, keys: cp.ndarray, state_index: int, n_rounds: int) -> None:
        for round_number in reversed(range(state_index - n_rounds, state_index)):
            keys[:, :] = cp.roll(keys, -1, axis=1)
            self.feistel(keys[:, 2:4], self.constant ^ ((self.z_sequence >> round_number) & 1))
            blocks[:, :] = cp.roll(blocks, 1, axis=1)
            self.feistel(blocks, keys[:, 3])
