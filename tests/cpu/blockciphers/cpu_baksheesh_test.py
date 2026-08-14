# https://link.springer.com/content/pdf/10.1007/978-3-540-74735-2_31.pdf

import numpy as np
import pytest

from urca.common import gen_bits
from urca.cpu.blocks.baksheesh import Baksheesh


@pytest.mark.parametrize(
    "text_size, key_size, n_rounds, plaintexts, keys, ciphertexts",
    (
        (
            128,
            128,
            35,
            (
                0x00000000000000000000000000000000,
                0x00000000000000000000000000000007,
                0x70000000000000000000000000000000,
                0x44444444444444444444444444444444,
                0x11111111111111111111111111111111,
                0x789A789A789A789A789A789A789A789A,
                0xB6E4789AB6E4789AB6E4789AB6E4789A,
                0xE6517531ABF63F3D7805E126943A081C,
            ),
            (
                0x00000000000000000000000000000000,
                0x00000000000000000000000000000000,
                0x00000000000000000000000000000000,
                0x00000000000000000000000000000000,
                0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,
                0x76543210032032032032032032032032,
                0x23023023023023023023023001234567,
                0x5920EFFB52BC61E33A98425321E76915,
            ),
            (
                0xC002BE5E64C78A72AB9A3439518352AA,
                0x6F7D7746EAF0D97A154079F6BD846438,
                0x1BA3363734C09A29F67C23BBB2CCCC05,
                0x7AD3303667B2AF6DEEF434DD110D7FB8,
                0x806F0CF45B94F0370206975FE78AC10F,
                0xAE654B5333B876584F8E8DD54F4E490A,
                0x3DBBDF7FE254CC0BE396A753442DCCAD,
                0xFC7E61FEE3D587308CA7BC594EBF3244,
            ),
        ),
    ),
)
def test_baksheesh(text_size, key_size, n_rounds, plaintexts, keys, ciphertexts):
    baksheesh = Baksheesh(text_size, key_size)
    texts = np.array(gen_bits(plaintexts, text_size), baksheesh.word_type)
    keys = np.array(gen_bits(keys, key_size), baksheesh.word_type)
    # encryption test
    baksheesh.encrypt(texts, keys, 0, n_rounds)
    assert np.all(texts == gen_bits(ciphertexts, text_size))
    # decryption test
    baksheesh.decrypt(texts, keys, n_rounds, n_rounds)
    assert np.all(texts == gen_bits(plaintexts, text_size))
