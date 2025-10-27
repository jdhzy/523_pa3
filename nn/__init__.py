"""Shim package to expose autograd.nn as top-level `nn`.

This allows external tests to `import nn...` while keeping source under `autograd/nn`.
"""

from importlib import import_module as _imp
import sys as _sys

# Map selected modules directly
for _name in (
    'module', 'param', 'lf', 'optimizer',
):
    _sys.modules[__name__ + '.' + _name] = _imp('autograd.nn.' + _name)

# Map subpackages wholesale
for _pkg in (
    'layers', 'losses', 'models', 'optimizers',
):
    _sys.modules[__name__ + '.' + _pkg] = _imp('autograd.nn.' + _pkg)

del _imp, _sys, _name, _pkg
