# https://eprint.iacr.org/2013/404.pdf

import cupy as cp
import numpy as np
import pytest
from urca import constants
from urca.cpu.blocks.simon import Simon


@pytest.mark.parametrize(
    "text_size, key_size, n_rounds, z_sequence, plaintexts, keys, ciphertexts",
    (
        (
            32,
            64,
            32,
            constants.SIMON_Z0,
            ((0x6565, 0x6877),),
            ((0x1918, 0x1110, 0x0908, 0x0100),),
            ((0xC69B, 0xE9BB),),
        ),
        (
            48,
            72,
            36,
            constants.SIMON_Z0,
            ((0x612067, 0x6E696C),),
            ((0x121110, 0x0A0908, 0x020100),),
            ((0xDAE5AC, 0x292CAC),),
        ),
        (
            48,
            96,
            36,
            constants.SIMON_Z1,
            ((0x726963, 0x20646E),),
            ((0x1A1918, 0x121110, 0x0A0908, 0x020100),),
            ((0x6E06A5, 0xACF156),),
        ),
        (
            64,
            96,
            42,
            constants.SIMON_Z2,
            ((0x6F722067, 0x6E696C63),),
            ((0x13121110, 0x0B0A0908, 0x03020100),),
            ((0x5CA2E27F, 0x111A8FC8),),
        ),
        (
            64,
            128,
            44,
            constants.SIMON_Z3,
            ((0x656B696C, 0x20646E75),),
            ((0x1B1A1918, 0x13121110, 0x0B0A0908, 0x03020100),),
            ((0x44C8FC20, 0xB9DFA07A),),
        ),
        (
            96,
            96,
            52,
            constants.SIMON_Z2,
            ((0x2072616C6C69, 0x702065687420),),
            ((0x0D0C0B0A0908, 0x050403020100),),
            ((0x602807A462B4, 0x69063D8FF082),),
        ),
        (
            96,
            144,
            54,
            constants.SIMON_Z3,
            ((0x746168742074, 0x73756420666F),),
            ((0x151413121110, 0x0D0C0B0A0908, 0x050403020100),),
            ((0xECAD1C6C451E, 0x3F59C5DB1AE9),),
        ),
        (
            128,
            128,
            68,
            constants.SIMON_Z2,
            ((0x6373656420737265, 0x6C6C657661727420),),
            ((0x0F0E0D0C0B0A0908, 0x0706050403020100),),
            ((0x49681B1E1E54FE3F, 0x65AA832AF84E0BBC),),
        ),
        (
            128,
            192,
            69,
            constants.SIMON_Z3,
            ((0x206572656874206E, 0x6568772065626972),),
            ((0x1716151413121110, 0x0F0E0D0C0B0A0908, 0x0706050403020100),),
            ((0xC4AC61EFFCDC0D4F, 0x6C9C8D6E2597B85B),),
        ),
        (
            128,
            256,
            72,
            constants.SIMON_Z4,
            ((0x74206E69206D6F6F, 0x6D69732061207369),),
            ((0x1F1E1D1C1B1A1918, 0x1716151413121110, 0x0F0E0D0C0B0A0908, 0x0706050403020100),),
            ((0x8D2B5579AFC8A3A0, 0x3BF72A87EFE7B868),),
        ),
    ),
)
def test_simon(text_size, key_size, n_rounds, z_sequence, plaintexts, keys, ciphertexts):
    simon = Simon(text_size, key_size, z_sequence=z_sequence)
    texts = cp.array(plaintexts, simon.word_type)
    keys = cp.array(keys, simon.word_type)
    # encryption test
    simon.encrypt(texts, keys, 0, n_rounds)
    assert np.all(cp.asnumpy(texts) == ciphertexts)
    # decryption test
    simon.decrypt(texts, keys, n_rounds, n_rounds)
    assert np.all(cp.asnumpy(texts) == plaintexts)
