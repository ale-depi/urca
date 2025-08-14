Block ciphers
=============

Since the nature of block ciphers is to encrypt blocks of data, every
implementation has to have ``text_size`` and ``key_size`` among its attributes.
The design choice is to not enforce them, nevertheless they are strictly
recommended.

Note that every cipher is implemented as a frozen dataclass. The design choice
is to have an automatic method to identify the instance. This is achieved
caching in the best effort way all properties that can be derived by the
attributes which are mandatory.

.. toctree::
   :maxdepth: 1

   present
   speck
