# https://eprint.iacr.org/2013/404.pdf

import cupy as cp
import numpy as np
import pytest
from urca.cpu.blocks.speck import Speck


@pytest.mark.parametrize(
    "text_size, key_size, n_rounds, alpha, beta, plaintexts, keys, ciphertexts",
    (
        (
            32,
            64,
            22,
            7,
            2,
            ((0x6574, 0x694C),),
            ((0x1918, 0x1110, 0x0908, 0x0100),),
            ((0xA868, 0x42F2),),
        ),
        (
            48,
            72,
            22,
            8,
            3,
            ((0x20796C, 0x6C6172),),
            ((0x121110, 0x0A0908, 0x020100),),
            ((0xC049A5, 0x385ADC),),
        ),
        (
            48,
            96,
            23,
            8,
            3,
            ((0x6D2073, 0x696874),),
            ((0x1A1918, 0x121110, 0x0A0908, 0x020100),),
            ((0x735E10, 0xB6445D),),
        ),
        (
            64,
            96,
            26,
            8,
            3,
            ((0x74614620, 0x736E6165),),
            ((0x13121110, 0x0B0A0908, 0x03020100),),
            ((0x9F7952EC, 0x4175946C),),
        ),
        (
            64,
            128,
            27,
            8,
            3,
            ((0x3B726574, 0x7475432D),),
            ((0x1B1A1918, 0x13121110, 0x0B0A0908, 0x03020100),),
            ((0x8C6FA548, 0x454E028B),),
        ),
        (
            96,
            96,
            28,
            8,
            3,
            ((0x65776F68202C, 0x656761737520),),
            ((0x0D0C0B0A0908, 0x050403020100),),
            ((0x9E4D09AB7178, 0x62BDDE8F79AA),),
        ),
        (
            96,
            144,
            29,
            8,
            3,
            ((0x656D6974206E, 0x69202C726576),),
            ((0x151413121110, 0x0D0C0B0A0908, 0x050403020100),),
            ((0x2BF31072228A, 0x7AE440252EE6),),
        ),
        (
            128,
            128,
            32,
            8,
            3,
            ((0x6C61766975716520, 0x7469206564616D20),),
            ((0x0F0E0D0C0B0A0908, 0x0706050403020100),),
            ((0xA65D985179783265, 0x7860FEDF5C570D18),),
        ),
        (
            128,
            192,
            33,
            8,
            3,
            ((0x7261482066656968, 0x43206F7420746E65),),
            ((0x1716151413121110, 0x0F0E0D0C0B0A0908, 0x0706050403020100),),
            ((0x1BE4CF3A13135566, 0xF9BC185DE03C1886),),
        ),
        (
            128,
            256,
            34,
            8,
            3,
            ((0x65736F6874206E49, 0x202E72656E6F6F70),),
            ((0x1F1E1D1C1B1A1918, 0x1716151413121110, 0x0F0E0D0C0B0A0908, 0x0706050403020100),),
            ((0x4109010405C0F53E, 0x4EEEB48D9C188F43),),
        ),
    ),
)
def test_speck(text_size, key_size, n_rounds, alpha, beta, plaintexts, keys, ciphertexts):
    speck = Speck(text_size, key_size, alpha, beta)
    texts = cp.array(plaintexts, speck.word_type)
    keys = cp.array(keys, speck.word_type)
    # encryption test
    speck.encrypt(texts, keys, 0, n_rounds)
    assert np.all(cp.asnumpy(texts) == ciphertexts)
    # decryption test
    speck.decrypt(texts, keys, n_rounds, n_rounds)
    assert np.all(cp.asnumpy(texts) == plaintexts)
