# Lightweight shim package to expose autograd.nn as top-level `nn`
from importlib import import_module as _imp
import sys as _sys

# Ensure autograd.nn and its subpackages are available under the `nn` namespace
for _name in (
    'module', 'param', 'lf', 'optimizer',
):
    _sys.modules[__name__ + '.' + _name] = _imp('autograd.nn.' + _name)

for _pkg in (
    'layers', 'losses', 'models', 'optimizers',
):
    _sys.modules[__name__ + '.' + _pkg] = _imp('autograd.nn.' + _pkg)

del _imp, _sys, _name, _pkg

