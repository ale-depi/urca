Present
=======

The PRESENT [present-2007]_ block cipher has reference configurations suitable
essentially for one text size. Based on that, two key sizes are possible. The
implementation is given for both configurations.

.. math::

   \begin{array}{l}
      \texttt{text_size} = 8t, \quad t \in \mathbb{N} \\
      \texttt{key_size} = kt, \quad k \in \{10, 16\} \\
      \texttt{sbox} = (x_0, \dots, x_{15}), \quad x_i \in [0..15]\ \text{and}\ x_i \neq x_j\ \text{when}\ i \neq j
   \end{array}
