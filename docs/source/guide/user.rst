User
====

Getting started
---------------

To have fun with URCA, it is enough the following.

.. code-block:: bash

   $ python3 -m venv venv_name
   $ source venv_name/bin/activate
   (venv_name) $ pip install urca

Latest commit
^^^^^^^^^^^^^

Note that in order to have the last changes or fixes because of a commit which
is not a release, you can do the following.

.. code-block:: bash

   $ python3 -m venv venv_name
   $ source venv_name/bin/activate
   (venv_name) $ git clone https://github.com/ale-depi/urca
   (venv_name) $ cd urca
   (venv_name) $ pip install .

From now on, in the virtual environment named ``venv_name`` you can call
``import urca``.

Implementations
---------------

The package will be installed with support for both `NumPy
<https://numpy.org>`_ (CPU) and `CuPy <https://cupy.dev>`_ (GPU). This dual
installation does not cause any compatibility issues on systems without a GPU,
since the two implementations are completely isolated. The isolation ensures
coherent behavior regardless of the underlying hardware environment, while also
enabling efficient utilization of GPU resources when available.

The API documentation is `NumPy <https://numpy.org>`_ oriented, but the
interfaces, methods, variables, and overall structure remain exactly the same
when using the `CuPy <https://cupy.dev>`_ version. The only differences are
found in the implementation details, since not all NumPy interfaces are
directly available in CuPy.

Selecting one implementation or the other is simply a matter of the import
path: ``urca.cpu`` for the CPU-based implementation or ``urca.gpu`` for the
GPU-based one.

