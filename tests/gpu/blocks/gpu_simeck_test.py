# https://doi.org/10.1007/978-3-662-48324-4_16

import cupy as cp
import numpy as np
import pytest
from urca import constants
from urca.cpu.blocks.simeck import Simeck


@pytest.mark.parametrize(
    "text_size, key_size, n_rounds, z_sequence, plaintexts, keys, ciphertexts",
    (
        (
            32,
            64,
            32,
            constants.SIMECK_Z0,
            ((0x6565, 0x6877),),
            ((0x1918, 0x1110, 0x0908, 0x0100),),
            ((0x770D, 0x2C76),),
        ),
        (
            48,
            96,
            36,
            constants.SIMECK_Z0,
            ((0x726963, 0x20646E),),
            ((0x1A1918, 0x121110, 0x0A0908, 0x020100),),
            ((0xF3CF25, 0xE33B36),),
        ),
        (
            64,
            128,
            44,
            constants.SIMECK_Z1,
            ((0x656B696C, 0x20646E75),),
            ((0x1B1A1918, 0x13121110, 0x0B0A0908, 0x03020100),),
            ((0x45CE6902, 0x5F7AB7ED),),
        ),
    ),
)
def test_simeck(text_size, key_size, n_rounds, z_sequence, plaintexts, keys, ciphertexts):
    simeck = Simeck(text_size, key_size, z_sequence=z_sequence)
    texts = cp.array(plaintexts, simeck.word_type)
    keys = cp.array(keys, simeck.word_type)
    # encryption test
    simeck.encrypt(texts, keys, 0, n_rounds)
    assert np.all(cp.asnumpy(texts) == ciphertexts)
    # decryption test
    simeck.decrypt(texts, keys, n_rounds, n_rounds)
    assert np.all(cp.asnumpy(texts) == plaintexts)
