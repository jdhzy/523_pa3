# SYSTEM IMPORTS
from __future__ import annotations
from abc import abstractmethod, ABC
from typing import List, Optional, Tuple
import numpy as np

from .rnn_cell import RNNCell
from ...layers.dense import Dense
from ...layers.tanh import Tanh
from ...layers.sigmoid import Sigmoid


# PYTHON PROJECT IMPORTS
from ...module import Module

class VanillaRNNCell(RNNCell):
    def __init__(self: VanillaRNNCell,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 hidden_activation: Module = None,
                 output_activation: Module = None) -> None:
        super().__init__(in_dim, hidden_dim)

        self.in_dim: int = in_dim
        self.hidden_dim: int = hidden_dim
        self.out_dim: int = out_dim

        # Layers
        self.hidden_dense: Dense = Dense(in_dim + hidden_dim, hidden_dim)
        self.out_dense: Dense = Dense(hidden_dim, out_dim)

        # Activations with defaults
        self.hidden_activation: Module = hidden_activation if hidden_activation is not None else Tanh()
        self.output_activation: Module = output_activation if output_activation is not None else Sigmoid()

    def init_states(self: VanillaRNNCell, batch_size: int) -> np.ndarray:
            return np.zeros((batch_size, self.hidden_dim))

    def forward(self: VanillaRNNCell,
                H_t_minus_1: np.ndarray,
                X_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        # Concatenate as [X_t, H_{t-1}] per requirement
        concat: np.ndarray = np.concatenate([X_t, H_t_minus_1], axis=1)
        Z_t: np.ndarray = self.hidden_dense.forward(concat)                    # f1
        H_t: np.ndarray = self.hidden_activation.forward(Z_t)                  # g
        R_t: np.ndarray = self.out_dense.forward(H_t)                          # f2
        A_t: np.ndarray = self.output_activation.forward(R_t)                  # o

        return H_t, A_t

    def backward(self: VanillaRNNCell,
                 H_t_minus_1: np.ndarray,
                 X_t: np.ndarray,
                 dLoss_dModule_t,
                 dLoss_dStates_t) -> Tuple[np.ndarray, np.ndarray]:
        # Recompute intermediates for gradient flow
        # Use the same concatenation order as in forward: [X_t, H_{t-1}]
        concat: np.ndarray = np.concatenate([X_t, H_t_minus_1], axis=1)
        Z_t: np.ndarray = self.hidden_dense.forward(concat)
        H_t: np.ndarray = (
            self.hidden_activation.forward(Z_t) if self.hidden_activation is not None else Z_t
        )
        R_t: np.ndarray = self.out_dense.forward(H_t)

        # Path 1: gradients from output A_t
        dLoss_dH_from_output: Optional[np.ndarray] = None
        if dLoss_dModule_t is not None:
            dLoss_dR_t: np.ndarray = (
                self.output_activation.backward(R_t, dLoss_dModule_t)
                if self.output_activation is not None else dLoss_dModule_t
            )
            dLoss_dH_from_output = self.out_dense.backward(H_t, dLoss_dR_t)

        # Path 2: gradients from future states
        dLoss_dH_total: np.ndarray
        if dLoss_dStates_t is None and dLoss_dH_from_output is None:
            # No gradients provided; return zeros of correct shape
            dLoss_dH_total = np.zeros_like(H_t)
        elif dLoss_dStates_t is None:
            dLoss_dH_total = dLoss_dH_from_output  # type: ignore
        elif dLoss_dH_from_output is None:
            dLoss_dH_total = dLoss_dStates_t
        else:
            dLoss_dH_total = dLoss_dH_from_output + dLoss_dStates_t

        # Backprop through hidden activation and dense
        dLoss_dZ_t: np.ndarray = (
            self.hidden_activation.backward(Z_t, dLoss_dH_total)
            if self.hidden_activation is not None else dLoss_dH_total
        )
        dLoss_dConcat: np.ndarray = self.hidden_dense.backward(concat, dLoss_dZ_t)

        # Split gradients back into input and state components to match [X_t, H_{t-1}]
        dLoss_dX_t: np.ndarray = dLoss_dConcat[:, :self.in_dim]
        dLoss_dH_t_minus_1: np.ndarray = dLoss_dConcat[:, self.in_dim:]
        return dLoss_dH_t_minus_1, dLoss_dX_t

    def parameters(self: VanillaRNNCell):
        # Parameters from all submodules in the specified order
        params = []
        params += self.hidden_dense.parameters()
        if self.hidden_activation is not None:
            params += self.hidden_activation.parameters()
        params += self.out_dense.parameters()
        if self.output_activation is not None:
            params += self.output_activation.parameters()
        return params
