# SYSTEM IMPORTS
from __future__ import annotations
from typing import List, Tuple, Union, Optional
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..param import Parameter
from .cells.rnn_cell import RNNCell


class RNN(Module):
    def __init__(self,
                 cell,
                 return_sequences: bool = False,
                 return_states: bool = False,
                 backprop_through_time_limit: Optional[int] = None) -> None:
        super().__init__()

        self.cell = cell
        self.return_sequences = return_sequences
        self.return_states = return_states
        self.backprop_through_time_limit = (
            np.inf if backprop_through_time_limit is None else backprop_through_time_limit
        )

    def init_states(self,
                    batch_size: int) -> np.ndarray:
        return self.cell.init_states(batch_size)

    def forward(self,
            X: np.ndarray, 
            states_init: np.ndarray = None):
        batch_size, seq_size = X.shape[0], X.shape[1]
        
        # Initialize hidden state
        H_prev: np.ndarray = self.init_states(batch_size) if states_init is None else states_init

        outputs: List[np.ndarray] = []
        states: List[np.ndarray] = [H_prev]  

        for t in range(seq_size):
            X_t = X[:, t, :]
            H_t, A_t = self.cell.forward(H_prev, X_t)
            outputs.append(A_t)
            states.append(H_t)
            H_prev = H_t

        if self.return_sequences:
            Y_hat = np.stack(outputs, axis=1)
            H_all = np.stack(states, axis=1)
            if self.return_states:
                # Return (states, predictions)
                return H_all, Y_hat
            return Y_hat
        else:
            Y_hat_last = outputs[-1]
            H_last = states[-1]
            if self.return_states:
                # Return (state, prediction)
                return H_last, Y_hat_last
            return Y_hat_last

    def backward(self,
                 X: np.ndarray,
                 dLoss_dModule: np.ndarray,
                 states_init:  np.ndarray = None) -> np.ndarray:
        # Forward pass to cache intermediates
        batch_size, seq_size = X.shape[0], X.shape[1]
        H_prev: np.ndarray = self.init_states(batch_size) if states_init is None else states_init

        H_prev_list: List[np.ndarray] = []  # H_{t-1}
        X_list: List[np.ndarray] = []
        outputs: List[np.ndarray] = []
        states: List[np.ndarray] = []

        for t in range(seq_size):
            X_t = X[:, t, ...]
            H_prev_list.append(H_prev)
            X_list.append(X_t)
            H_t, A_t = self.cell.forward(H_prev, X_t)
            outputs.append(A_t)
            states.append(H_t)
            H_prev = H_t

        # Prepare gradient w.r.t inputs
        dLoss_dX = np.zeros_like(X)

        # Helper to fetch output gradient at a timestep
        def grad_at_t(t):
            if self.return_sequences:
                return dLoss_dModule[:, t, ...]
            else:
                if t == seq_size - 1:
                    return dLoss_dModule
                return None

        # Backpropagation Through Time with optional truncation
        limit = self.backprop_through_time_limit
        for t_out in range(seq_size - 1, -1, -1):
            dLoss_dA_t = grad_at_t(t_out)
            if dLoss_dA_t is None:
                continue

            d_h_carry: Optional[np.ndarray] = None
            if np.isinf(limit):
                k_lower = 0
            else:
                # ensure integer limit, propagate back 'limit' steps including t_out
                k_lower = max(0, t_out - int(limit) + 1)

            for k in range(t_out, k_lower - 1, -1):
                H_k_minus_1 = H_prev_list[k]
                X_k = X_list[k]
                # Only the output time contributes dLoss_dModule at t_out
                dLoss_dModule_k = dLoss_dA_t if k == t_out else None
                dLoss_dStates_k = d_h_carry

                dH_k_minus_1, dX_k = self.cell.backward(
                    H_k_minus_1, X_k, dLoss_dModule_k, dLoss_dStates_k
                )

                dLoss_dX[:, k, ...] += dX_k
                d_h_carry = dH_k_minus_1

        return dLoss_dX

    def parameters(self) -> List[Parameter]:
        return self.cell.parameters()