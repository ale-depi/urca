URCA documentation
==================

Unified Resource for Cryptographic Arrays

When talking about simmetric ciphers, it could be handy having a resource that
computes efficiently plaintext and ciphertext. Moreover, if a change of sizes
is needed, there are many constraints to take into account in order to have the
standard implementation working correctly.

The URCA project tries to provide a standard for the above tasks.

Motivation
----------

First and last rounds
^^^^^^^^^^^^^^^^^^^^^

One of the most common scheme for a simmetric cipher is to have a number of
rounds repeating the same structure. This is the most important skeleton of the
cipher.

However, it is frequent to have some changes in the first round or even in the
last one. When this occurs in some design, it is important to say what is and
is not a round. Think about Speck [BSS2015]_, PRESENT [BKL2007]_ and AES
[DR2003]_.

They represent three important cases.

#. Speck is word oriented and has not corner cases. Its implementation is
   suitable for any word size, unless constraints on parameters are met.
#. PRESENT is a bit oriented cipher. Note that there is a first XOR with the
   key, meaning that the first round is different from the other ones.
   Moreover, in order to be compliant with the reference implementation, some
   constraints on the parameters should be met (e.g. the key must be 5/4 or 8/4
   of the text).
#. AES has a declared different last round. It lacks of the mixcolumn
   operation.

Number of rounds
^^^^^^^^^^^^^^^^

Numbering rounds is another task that should be accomplished carefully and
consistently. Think again about Speck. The key scheduling can be inserted in
the second round, but this could be misleading since the state of the cipher
is not preserved. The suggestion from this work is to consider ciphers as
follows.::

   | -- (registers == state) --------------------------------- |
   | -- plaintext[0] --------- | -- key[0] (master-key) ------ |
           |                                                      |
           | (schedule)                                           |
           |                                                      |
           V                                                      |
   | -- plaintext[1] --------- | -- key[0] ------------------- |  | round[0]
                                       |                          |
                                       | (key-schedule)           |
                                       |                          |
                                       V                          V
   | -- plaintext[1] --------- | -- key[1] ------------------- |

Note that the schedule and the key-schedule can be swapped.

.. toctree::
   :maxdepth: 1
   :caption: API

   blocks/speck
   utilities

Bibliography
------------

* :ref:`Papers <papers>`
