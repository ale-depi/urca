Speck
=====

The Speck block cipher [simon-speck-2015]_ has reference configurations
suitable for many sizes. Based on those, its generalization can be done as
follows:

.. math::

   \begin{array}{l}
      \texttt{text_size} = 2t, \quad t \in \mathbb{N} \\
      \texttt{key_size} = 2kt, \quad k \in \{2, 3, 4\} \\
      \texttt{n_rounds} = r, \quad r \in \mathbb{N} \\
      \alpha < t \\
      \beta < t
   \end{array}

The implementation is ready to be used. The suggested size is any multiple of 4
bits, i.e. :math:`t = 4\tau,\ \tau \in \mathbb{N}`.