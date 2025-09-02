# URCA

**U**nified **R**esource for **C**ryptographic **A**rrays

<img src="docs/source/_static/logo.svg" width="128" alt="Official logo">

The URCA project aims to give the vectorized implementations of many
cryptographic primitives in order to investigate the statistical properties.
Not only, since sometimes it could be useful to study a reduced version of the
primitives, the URCA project tries to give the generalised version of the
primitives. The project implements vectorised implementations both for CPUs and
GPUs, using respectvely [NumPy](https://numpy.org/) and
[CuPy](https://cupy.dev/).

## Pages

[Documentation](https://ale-depi.github.io/urca/)

[Guide for the user](https://ale-depi.github.io/urca/guide/user.html)

[Guide for the developer](https://ale-depi.github.io/urca/guide/developer.html)

## Examples

Multiple plaintext can be encrypted at once.

```python
>>> import numpy as np
>>> from urca.cpu.blocks.speck import Speck
>>> speck = Speck(32, 64)
>>> word_type = speck.word_type
>>> texts = np.array([[0x6574, 0x694C], [0x0000, 0x0000]], dtype=word_type)
>>> keys = np.array([[0x1918, 0x1110, 0x0908, 0x0100], [0x0000, 0x0000, 0x0000, 0x0000]], dtype=word_type)
>>> speck.encrypt(texts, keys, 0, 22)
>>> np.vectorize(hex)(texts)
array([['0xa868', '0x42f2'],
       ['0x2bb9', '0xc642']], dtype='<U6')
```

URCA is designed to be as general as possible. The following workflow,
encrypting a bunch of texts, can be applied to any primitive.

```python
>>> import random
>>> import numpy as np
>>> from urca.cpu.blocks.speck import Speck
>>> primitive = Speck(32, 64)
>>> word_size = primitive.word_size
>>> word_type = primitive.word_type
>>> n_text_words = primitive.n_text_words
>>> n_key_words = primitive.n_key_words
>>> n_instances = 4
>>> texts = [[random.getrandbits(word_size) for _ in range(n_text_words)] for _ in range(n_instances)]
>>> texts = np.array(texts, dtype=word_type)
>>> keys = [[random.getrandbits(word_size) for _ in range(n_key_words)] for _ in range(n_instances)]
>>> keys = np.array(keys, dtype=word_type)
>>> primitive.encrypt(texts, keys, 0, 22)
>>> np.vectorize(hex)(texts)
# array([['0x3068', '0xc0bf'],
#        ['0xb30b', '0xbed8'],
#        ['0xbb16', '0xece6'],
#        ['0x921a', '0x6f0a']], dtype='<U6')
# Example of output
```
