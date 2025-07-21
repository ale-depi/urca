# URCA

**U**nified **R**esource for **C**ryptographic **A**rrays

## Getting started

To have fun with URCA locally, try the following steps.

```bash
$ python3 -m venv venv
$ source venv/bin/activate
(venv) $ pip install .
```

## Format

In order to have a common view the predefined formatter is
[Ruff](https://docs.astral.sh/ruff/).

## Commits

* When declaring a meaningful change (e.g. important features or critical
fixes), please use a
[conventional commit](https://www.conventionalcommits.org/en/v1.0.0/).
* If an helper is needed, after pip installation,
[commitizen](https://commitizen-tools.github.io/commitizen/)
can be used.
* Recording changes without conventional commits is allowed for meaningless,
but, please, follow the well established commit etiquette.

### Etiquette

* Capitalize the subject line and each paragraph
* Keep the subject line 50 characters long at most 
* Do not end the subject line with a period
* Separate subject from body with a blank line
* Use the imperative mood in the subject line
* Wrap lines of the body at 72 characters
* Use the body to explain what and why you have done something.
In most cases, you can leave out details about how a change has been made.

***Attention***: pull requests could be rejected at any moment if the commit
history will not follow the above etiquette.
