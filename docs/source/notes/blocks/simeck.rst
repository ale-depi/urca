Simeck
======

The Simeck block cipher [simeck-2015]_ has reference configurations
suitable for many sizes. Even if the implementation could actually work in the
most general scenario, the suggested size is any multiple of 8
bits, i.e. :math:`t = 4\tau,\ \tau \in \mathbb{N}`.

.. math::

   \begin{array}{l}
      \texttt{text_size} = 2t, \quad t \in \mathbb{N} \\
      \texttt{key_size} = 4t \\
      \texttt{rot} = (5, 1)
   \end{array}

