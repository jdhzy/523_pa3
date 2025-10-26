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
        # X shape: (batch_size, sequence size, input_dim)
        batch_size, sequence_size = X.shape[0], X.shape[1]

        # Initialize output array
        Y_hat: np.ndarray = np.zeros((batch_size, sequence_size,
                                     self.module.forward(X[0, 0:1, :]).shape[-1]))

        # Apply the module to each sequence
        for t in range(sequence_size):
            Y_hat[:, t, :] = self.module.forward(X[:, t, :])
        return Y_hat
    
    def backward(self: TimeDistributed,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray) -> np.ndarray:

        batch_size, sequence_size = X.shape[0], X.shape[1]
        dLoss_dX: np.ndarray = np.zeros_like(X)

        for t in range(sequence_size):
            dLoss_dX[:, t, :] = self.module.backward(X[:, t, :], dLoss_dModule[:, t, :])

        return dLoss_dX
    
    def parameters(self: TimeDistributed) -> List[Parameter]:
        return self.module.parameters()
    