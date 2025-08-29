Simon
=====

The Simon block cipher [simon-speck-2015]_ has reference configurations
suitable for many sizes. Even if the implementation could actually work in the
most general scenario, the suggested size is any multiple of 4
bits, i.e. :math:`t = 2\tau,\ \tau \in \mathbb{N}`.

.. math::

   \begin{array}{l}
      \texttt{text_size} = 2t, \quad t \in \mathbb{N} \\
      \texttt{key_size} = 2kt, \quad k \in \{2, 3, 4\} \\
      \texttt{keyrot} = (3, 1) \\
      \texttt{rot} = (1, 8, 2)
   \end{array}
