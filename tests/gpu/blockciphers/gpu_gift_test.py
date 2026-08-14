# https://github.com/giftcipher/gift/tree/master/implementations/test%20vectors

import cupy as cp
import numpy as np
import pytest

from urca.common import gen_bits
from urca.gpu.blocks.gift import Gift


@pytest.mark.parametrize(
    "text_size, key_size, n_rounds, plaintexts, keys, ciphertexts",
    (
        (
            64,
            128,
            28,
            (0x0000000000000000, 0xFEDCBA9876543210, 0xC450C7727A9B8A7D),
            (
                0x00000000000000000000000000000000,
                0xFEDCBA9876543210FEDCBA9876543210,
                0xBD91731EB6BC2713A1F9F6FFC75044E7,
            ),
            (0xF62BC3EF34F775AC, 0xC1B71F66160FF587, 0xE3272885FA94BA8B),
        ),
        (
            128,
            128,
            40,
            (
                0x00000000000000000000000000000000,
                0xFEDCBA9876543210FEDCBA9876543210,
                0xE39C141FA57DBA43F08A85B6A91F86C1,
            ),
            (
                0x00000000000000000000000000000000,
                0xFEDCBA9876543210FEDCBA9876543210,
                0xD0F5C59A7700D3E799028FA9F90AD837,
            ),
            (
                0xCD0BD738388AD3F668B15A36CEB6FF92,
                0x8422241A6DBF5A9346AF468409EE0152,
                0x13EDE67CBDCC3DBF400A62D6977265EA,
            ),
        ),
    ),
)
def test_gift(text_size, key_size, n_rounds, plaintexts, keys, ciphertexts):
    gift = Gift(text_size, key_size)
    texts = cp.array(gen_bits(plaintexts, text_size), gift.word_type)
    keys = cp.array(gen_bits(keys, key_size), gift.word_type)
    # encryption test
    gift.encrypt(texts, keys, 0, n_rounds)
    assert np.all(cp.asnumpy(texts) == gen_bits(ciphertexts, text_size))
    # decryption test
    gift.decrypt(texts, keys, n_rounds, n_rounds)
    assert np.all(cp.asnumpy(texts) == gen_bits(plaintexts, text_size))
