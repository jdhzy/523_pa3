# SYSTEM IMPORTS
from __future__ import annotations
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..param import Parameter


# TYPES DECLARED IN THIS MODULE

class TimeDistributed(Module):

    def __init__(self: TimeDistributed,
                 module: Module) -> None:
        super().__init__()
        self.module: Module = module

    def forward(self: TimeDistributed,
                X: np.ndarray) -> np.ndarray:
        # X shape: (batch_size, sequence_size, ...)
        sequence_size = X.shape[1]
        outputs: List[np.ndarray] = []
        for t in range(sequence_size):
            outputs.append(self.module.forward(X[:, t, ...]))
        return np.stack(outputs, axis=1)
    
    def backward(self: TimeDistributed,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:
        sequence_size = X.shape[1]
        grads_x: List[np.ndarray] = []
        for t in range(sequence_size):
            grads_x.append(self.module.backward(X[:, t, ...], dLoss_dModule[:, t, ...]))
        return np.stack(grads_x, axis=1)
    
    def parameters(self: TimeDistributed) -> List[Parameter]:
        return self.module.parameters()
    
