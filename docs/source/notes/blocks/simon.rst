Simon
=====

:authors: Ray Beaulieu, Douglas Shors, Jason Smith, Stefan Treatman-Clark,
  Bryan Weeks, Louis Wingers
:title: SIMON and SPECK: Block Ciphers for the Internet of Things
:year: 2015
:link: https://eprint.iacr.org/2015/585.pdf

The Simon block cipher :cite:labelpar:`beaulieu2015simon` has reference configurations
suitable for many sizes. Even if the implementation could actually work in the
most general scenario, the suggested size is any multiple of 4
bits, i.e. :math:`t = 2\tau,\ \tau \in \mathbb{N}`.

.. math::

   \begin{array}{l}
      \texttt{text\_size} = 2t, \quad t \in \mathbb{N} \\
      \texttt{key\_size} = 2kt, \quad k \in \{2, 3, 4\} \\
      \texttt{keyrot} = (3, 1) \\
      \texttt{rot} = (1, 8, 2)
   \end{array}
