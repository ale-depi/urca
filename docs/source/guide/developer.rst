Developer
=========

.. contents:: Table of contents

Installation
------------

Since the developer must access more tools the installation is slightly
different, namely, it has a little parameter added.

.. code-block:: bash

   $ python3 -m venv venv
   $ source venv/bin/activate
   (venv) $ pip install .[dev]

The above commands will install additional packages such as `pytest
<https://docs.pytest.org/en/stable/>`_ and `Sphinx
<https://www.sphinx-doc.org/en/master/index.html>`_. Both serve for different
purposes.

Automatic testing
^^^^^^^^^^^^^^^^^

The implementation of the ciphers is designed to achieve the highest possible
level of generalization, while remaining compliant with the reference
specifications. To ensure this compliance, automated tests must be implemented
to verify the correctness of the provided test vectors. These tests are
executed using pytest, and all test cases must pass successfully (i.e., pytest
should return with a green status).

Offline Documentation
^^^^^^^^^^^^^^^^^^^^^

The offline documentation for the project can be built locally using Sphinx.
To build the documentation, run the command

.. code-block:: bash

   (venv) $ sphinx-build -b html docs/source/ docs/build/

from the root of the project. Performing a local build is strongly recommended
before committing or pushing changes, as it helps identify syntax errors,
formatting issues, or broken references in the documentation early in the
development process. This ensures that the published documentation remains
clear, accurate, and free of rendering problems.

Contributing
------------

The following guidelines are meant to give uniformity to the project logs.

Format
^^^^^^

In order to have a shared code format the predefined tool is
`Ruff <https://docs.astral.sh/ruff/>`_.

Commits
^^^^^^^

* When declaring a meaningful change (e.g. important features or critical
  fixes), please use a
  `Conventional Commit <https://www.conventionalcommits.org/en/v1.0.0/>`_.
* If an helper is needed, after pip installation,
  `Commitizen <https://commitizen-tools.github.io/commitizen/>`_ can be used.
* Recording changes without conventional commits is allowed for meaningless
  ones, but, please, follow the well established commit etiquette as follows.

Etiquette
^^^^^^^^^

#. Capitalize the subject line and each paragraph
#. Keep the subject line 50 characters long at most 
#. Do not end the subject line with a period
#. Separate subject from body with a blank line
#. Use the imperative mood in the subject line
#. Wrap lines of the body at 72 characters
#. Use the body to explain what and why you have done something.
   In most cases, you can leave out details about how a change has been made.

**Attention**: pull requests could be rejected at any moment if the commit
history will not follow the above etiquette.
