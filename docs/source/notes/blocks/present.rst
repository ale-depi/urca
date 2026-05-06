Present
=======

:authors: Andrey Bogdanov, Lars R Knudsen, Gregor Leander, Christof Paar, Axel
  Poschmann, Matthew JB Robshaw, Yannick Seurin, Charlotte Vikkelsoe
:title: PRESENT: An Ultra-Lightweight Block Cipher
:year: 2007
:link: https://link.springer.com/content/pdf/10.1007/978-3-540-74735-2_31.pdf

The PRESENT :cite:labelpar:`bogdanov2007present` block cipher has reference configurations suitable
essentially for one text size. Based on that, two key sizes are possible. The
implementation is given for both configurations.

.. math::

   \begin{array}{l}
      \texttt{text\_size} = 8t, \quad t \in \mathbb{N} \\
      \texttt{key\_size} = kt, \quad k \in \{10, 16\} \\
      \texttt{sbox} = (x_0, \dots, x_{15}), \quad x_i \in [0..15]\ \text{and}\ x_i \neq x_j\ \text{when}\ i \neq j
   \end{array}
