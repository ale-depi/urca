Block ciphers
=============

Block ciphers are implemented as classes handling encryption and decryption of
data blocks. Moreover, other helpers methods are implemented and can be used as
well even if they are thought to reduce code.

Design Evolution
----------------

After careful consideration and multiple iterations, the chosen implementation
of block ciphers is a plain Python class rather than as dataclass, even though
they resemble dataclasses in structure. This choice was deliberate:

* Avoids constraints of dataclasses: Dataclasses (especially frozen ones)
  impose rigidity on attribute definitions and dunder methods. In contrast,
  plain classes offer more flexibility for custom behavior.
* Granular control: With a plain class, you maintain fine-grained control over
  method implementations (e.g., ``__eq__``, ``__hash__``, custom
  initializations, validations, cached properties).

This structure ensures that each block cipher is clear, customizable, and
maintainable.

Uniform implementation
----------------------

Every block cipher class must:

* Subclass from the shared abstract base class (Block).
* Explicitly implement the ``__init__`` method, including initialization of
  (strongly) recommended attributes such as ``text_size`` and ``key_size``.

Even though these attributes are not enforced by the base class, their explicit
initialization promotes:

* consistency across implementations;
* predictability during instantiation;
* alignment with the project’s design philosophy.

Each block cipher must provide two methods:

* ``encrypt(texts, keys, state_index, n_rounds) -> None``
* ``decrypt(texts, keys, state_index, n_rounds) -> None``

These methods must share the same signature in all implementations. This
guarantees that every cipher can be used interchangeably.

Encryption and decryption are performed in place. This is a deliberate design
choice, made to reduce memory usage and obtain faster implementations. It
remains subject to future refactoring if a more flexible or suitable approach
is needed.

Example
-------

The implementation is designed to be as general as possible. The following
workflow, encrypting a bunch of texts, can be applied to any block cipher.

.. code:: python

   >>> import random
   >>> import numpy as np
   >>> from urca.cpu.blocks.speck import Speck
   >>> cipher = Speck(32, 64)
   >>> word_size = cipher.word_size
   >>> word_type = cipher.word_type
   >>> n_text_words = cipher.n_text_words
   >>> n_key_words = cipher.n_key_words
   >>> n_instances = 4
   >>> texts = [[random.getrandbits(word_size) for _ in range(n_text_words)] for _ in range(n_instances)]
   >>> texts = np.array(texts, dtype=word_type)
   >>> keys = [[random.getrandbits(word_size) for _ in range(n_key_words)] for _ in range(n_instances)]
   >>> keys = np.array(keys, dtype=word_type)
   >>> cipher.encrypt(texts, keys, 0, 22)
   >>> np.vectorize(hex)(texts)
   # array([['0x3068', '0xc0bf'],
   #        ['0xb30b', '0xbed8'],
   #        ['0xbb16', '0xece6'],
   #        ['0x921a', '0x6f0a']], dtype='<U6')
   # Example of output

Implemented ciphers
-------------------

.. toctree::
   :maxdepth: 1

   present
   simeck
   simon
   speck
