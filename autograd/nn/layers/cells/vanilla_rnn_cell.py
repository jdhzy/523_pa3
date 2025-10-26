# SYSTEM IMPORTS
from __future__ import annotations
from abc import abstractmethod, ABC
from typing import List, Optional, Tuple
import numpy as np

from nn.layers.cells.rnn_cell import RNNCell
from nn.layers.dense import Dense
from nn.layers.tanh import Tanh
from nn.layers.sigmoid import Sigmoid


# PYTHON PROJECT IMPORTS
from ...module import Module

class VanillaRNNCell(RNNCell):
    def __init__(self: VanillaRNNCell,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 hidden_activation: Module = None,
                 output_activation: Module = None) -> None:
        # Initialize base RNNCell with input and hidden dimensions
        super().__init__(in_dim, hidden_dim)

        # Store constructor args
        self.in_dim: int = in_dim
        self.hidden_dim: int = hidden_dim
        self.out_dim: int = out_dim

        # Layers
        # f1: projects [H_{t-1}, X_t] -> hidden_dim
        self.hidden_dense: Dense = Dense(in_dim + hidden_dim, hidden_dim)
        # f2: projects H_t -> out_dim
        self.out_dense: Dense = Dense(hidden_dim, out_dim)

        # Activations with defaults
        self.hidden_activation: Module = hidden_activation if hidden_activation is not None else Tanh()
        self.output_activation: Module = output_activation if output_activation is not None else Sigmoid()

    def init_states(self: VanillaRNNCell,
                    batch_size: int) -> np.ndarray:
        # Zero-initialized hidden state
        return np.zeros((batch_size, self.hidden_dim))

    def forward(self: VanillaRNNCell,
                H_t_minus_1: np.ndarray,
                X_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Concatenate state and input along feature axis
        concat: np.ndarray = np.concatenate([H_t_minus_1, X_t], axis=1)
        # Hidden projection + activation 
        Z_t: np.ndarray = self.hidden_dense.forward(concat)
        H_t: np.ndarray = self.hidden_activation.forward(Z_t)
        # Output projection + activation 
        R_t: np.ndarray = self.out_dense.forward(H_t)
        A_t: np.ndarray = self.output_activation.forward(R_t)
        return A_t, H_t

    def backward(self: VanillaRNNCell,
                 H_t_minus_1: np.ndarray,
                 X_t: np.ndarray,
                 dLoss_dModule_t: Optional[np.ndarray],
                 dLoss_dStates_t: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        # Recompute intermediates for gradient flow
        concat: np.ndarray = np.concatenate([H_t_minus_1, X_t], axis=1)
        Z_t: np.ndarray = self.hidden_dense.forward(concat)
        H_t: np.ndarray = self.hidden_activation.forward(Z_t)
        R_t: np.ndarray = self.out_dense.forward(H_t)

        # Path 1: gradients from output A_t
        dLoss_dH_from_output: Optional[np.ndarray] = None
        if dLoss_dModule_t is not None:
            dLoss_dR_t: np.ndarray = self.output_activation.backward(R_t, dLoss_dModule_t)
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
        dLoss_dZ_t: np.ndarray = self.hidden_activation.backward(Z_t, dLoss_dH_total)
        dLoss_dConcat: np.ndarray = self.hidden_dense.backward(concat, dLoss_dZ_t)

        # Split gradients back into state and input components
        dLoss_dH_t_minus_1: np.ndarray = dLoss_dConcat[:, :self.hidden_dim]
        dLoss_dX_t: np.ndarray = dLoss_dConcat[:, self.hidden_dim:]
        return dLoss_dH_t_minus_1, dLoss_dX_t

    def parameters(self: VanillaRNNCell):
        # Parameters from all submodules in the specified order
        return (
            self.hidden_dense.parameters()
            + self.hidden_activation.parameters()
            + self.out_dense.parameters()
            + self.output_activation.parameters()
        )
