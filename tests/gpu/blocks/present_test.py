# https://link.springer.com/content/pdf/10.1007/978-3-540-74735-2_31.pdf

import cupy as cp
import numpy as np
import pytest

from urca.gpu.blocks.present import Present
from urca.common import get_bits


@pytest.mark.parametrize(
    "text_size, key_size, n_rounds, plaintexts, keys, ciphertexts",
    [
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
    ],
)
def test_present(text_size, key_size, n_rounds, plaintexts, keys, ciphertexts):
    present = Present(text_size, key_size)
    texts = cp.array(get_bits(plaintexts, text_size), present.word_type)
    keys = cp.array(get_bits(keys, key_size), present.word_type)
    # encryption test
    present.encrypt(texts, keys, 0, n_rounds)
    assert np.all(cp.asnumpy(texts) == get_bits(ciphertexts, text_size))
    # decryption test
    present.decrypt(texts, keys, n_rounds, n_rounds)
    assert np.all(cp.asnumpy(texts) == get_bits(plaintexts, text_size))
