URCA documentation
==================

Pages for the Unified Resource for Cryptographic Arrays project.

.. figure:: _static/logo.svg
   :figwidth: 128px

   (Official logo.)

Motivation
----------

When talking about simmetric ciphers, it could be handy having a resource that
efficiently encrypt and decrypt batch of texts. Moreover, if a change of sizes
is needed, there are many constraints to take into account in order to have the
standard implementation working correctly.

The URCA project tries to provide a standard for the above tasks. Moreover,
URCA tries to provide a tool that evaluates a generalised version of the cipher
keeping the reference configurations.

Note that all over the documentation, :math:`\mathbb{N}` is the set of natural
numbers excluding :math:`0`, unless otherwise specified.

First and last rounds
^^^^^^^^^^^^^^^^^^^^^

One of the most common scheme for a simmetric cipher is to have a number of
rounds repeating the same structure. This is the most important skeleton of the
cipher.

However, it is frequent to have some changes in the first round or even in the
last one. When this occurs in some design, it is important to say what is and
what is not a round. Think about Speck [BSS2015]_, PRESENT [BKL2007]_ and AES
[DR2003]_.

They represent three important cases.

#. Speck is word oriented and has not corner cases. Its implementation is
   suitable for any word size, unless constraints on parameters are met.
#. PRESENT is a bit oriented cipher. Note that there is a final XOR with the
   key, meaning that the last round is different from the other ones. Moreover,
   in order to be compliant with the reference implementation, some constraints
   on the parameters should be met (e.g. the key must be 5/4 or 8/4 of the
   text).
#. AES has a declared different last round. It lacks of the mixcolumn
   operation.

Numbering rounds
^^^^^^^^^^^^^^^^

Numbering rounds is another task that should be accomplished carefully and
consistently. Think again about Speck. The key scheduling can be inserted
starting from the second round, but this could be misleading since the state of
the cipher is not preserved as per round. The suggestion from this project is
to consider ciphers as follows.::

   | ---- (registers == state) ----------------------------------- |
   | ---- text[0] (plaintext) ---- | ---- key[0] (master-key) ---- |
            |                                                         |
            | (schedule)                                              |
            |                                                         |
            V                                                         |
   | ---- text[1] ---------------- | ---- key[0] ----------------- |  | round[0]
                                           |                          |
                                           | (key-schedule)           |
                                           |                          |
                                           V                          V
   | ---- text[1] ---------------- | ---- key[1] ----------------- |

Note that the schedule and the key-schedule can be swapped.

Moreover, the proposed standard from this project is to use 0 as index for the
plaintext and the master-key and considering both the starting round and the
number of rounds for encryption/decryption.

.. toctree::
   :maxdepth: 1
   :caption: Guide
   :hidden:

   guide/user
   guide/developer

.. toctree::
   :maxdepth: 2
   :caption: Notes
   :hidden:

   notes/blocks/index
   notes/references

.. toctree::
   :maxdepth: 2
   :caption: API
   :hidden:

   api/blocks/index
   api/utilities
