# https://link.springer.com/content/pdf/10.1007/978-3-540-74735-2_31.pdf

import numpy as np
import pytest

from urca.cpu.blocks.present import Present
from urca.common import gen_bits


@pytest.mark.parametrize(
    "text_size, key_size, n_rounds, plaintexts, keys, ciphertexts",
    (
        (
            64,
            80,
            31,
            (
                0x0000000000000000,
                0x0000000000000000,
                0xFFFFFFFFFFFFFFFF,
                0xFFFFFFFFFFFFFFFF,
            ),
            (
                0x00000000000000000000,
                0xFFFFFFFFFFFFFFFFFFFF,
                0x00000000000000000000,
                0xFFFFFFFFFFFFFFFFFFFF,
            ),
            (
                0x5579C1387B228445,
                0xE72C46C0F5945049,
                0xA112FFC72F68417B,
                0x3333DCD3213210D2,
            ),
        ),
        (
            64,
            128,
            31,
            (
                0x0000000000000000,
                0x0000000000000000,
                0xFFFFFFFFFFFFFFFF,
                0xFFFFFFFFFFFFFFFF,
            ),
            (
                0x00000000000000000000000000000000,
                0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,
                0x00000000000000000000000000000000,
                0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,
            ),
            (
                0x02F3ABF3A99796F2,
                0x0C1F091A1D2DD0B9,
                0xA28E002983461D15,
                0x1EA30E2ADC7236D6,
            ),
        ),
    ),
)
def test_present(text_size, key_size, n_rounds, plaintexts, keys, ciphertexts):
    present = Present(text_size, key_size)
    texts = np.array(gen_bits(plaintexts, text_size), present.word_type)
    keys = np.array(gen_bits(keys, key_size), present.word_type)
    # encryption test
    present.encrypt(texts, keys, 0, n_rounds)
    assert np.all(texts == gen_bits(ciphertexts, text_size))
    # decryption test
    present.decrypt(texts, keys, n_rounds, n_rounds)
    assert np.all(texts == gen_bits(plaintexts, text_size))
