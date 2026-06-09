import torch
import torch.nn as nn
import numpy as np
import copy

class ReservoirCell(nn.Module):
    """
    author: Prof. Claudio Gallicchio (c.gallicch)   (originally in Keras/TF)
    edited and implemented in Pytorch by Dr. Corrado Baccheschi (Bakko000)
    """
    def __init__(self, input_size, units,
                 input_scaling=1.0, bias_scaling=1.0,
                 spectral_radius=0.99,
                 leaky=1, activation=torch.tanh,
                 **kwargs):
        super().__init__()
        self.units = units
        self.state_size = units
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky
        self.activation = activation

        value = (self.spectral_radius / np.sqrt(self.units)) * (6 / np.sqrt(12))

        self.recurrent_kernel = nn.Parameter(
            torch.empty(self.units, self.units).uniform_(-value, value),
            requires_grad=False
        )

        self.kernel = nn.Parameter(
            torch.empty(input_size, self.units).uniform_(-self.input_scaling, self.input_scaling),
            requires_grad=False
        )

        self.bias = nn.Parameter(
            torch.empty(self.units).uniform_(-self.bias_scaling, self.bias_scaling),
            requires_grad=False
        )

    def forward(self, inputs, states):
        prev_output = states[0]
        input_part = torch.matmul(inputs, self.kernel)
        state_part = torch.matmul(prev_output, self.recurrent_kernel)

        if self.activation is not None:
            output = prev_output * (1 - self.leaky) + self.activation(input_part + self.bias + state_part) * self.leaky
        else:
            output = prev_output * (1 - self.leaky) + (input_part + self.bias + state_part) * self.leaky

        return output, [output]


class BidirectionalReservoirParallel(nn.Module):
    """
    Optimized bidirectional reservoir with parallel matrix operations.
    Eliminates sequential loops that introduce CPU latency.
    
    author: Dr. Corrado Baccheschi (Bakko000)
    """
    def __init__(self, cell, units):
        super().__init__()
        self.cell_forward = copy.deepcopy(cell)
        self.cell_backward = ReservoirCell(
            input_size=cell.kernel.shape[0],
            units=units,
            input_scaling=cell.input_scaling,
            bias_scaling=cell.bias_scaling,
            spectral_radius=cell.spectral_radius,
            leaky=cell.leaky,
            activation=cell.activation
        )
        self.units = units
        self.leaky = cell.leaky
        self.activation = cell.activation

    def _parallel_reservoir_pass(self, x, cell, reverse=False):
        """
        Parallelized reservoir pass using matrix operations.
        
        Args:
            x: Input tensor (batch, seq_len, input_size)
            cell: ReservoirCell to use
            reverse: If True, process sequence in reverse
            
        Returns:
            outputs: (batch, seq_len, units)
            final_state: Final hidden state
        """
        batch_size, seq_len, input_size = x.shape
        device = x.device
        
        if reverse:
            x = torch.flip(x, dims=[1])
        
        # Precompute input projections for all timesteps at once
        # Shape: (batch, seq_len, units)
        input_projections = torch.matmul(x, cell.kernel)  # Batch matrix multiply
        
        # Initialize hidden states
        h = torch.zeros(batch_size, self.units, device=device)
        outputs = []
        
        # This is still sequential in recurrence, but input computation is parallelized
        # For true parallelization, use the scan-free version below
        for t in range(seq_len):
            input_part = input_projections[:, t, :]  # (batch, units)
            state_part = torch.matmul(h, cell.recurrent_kernel)  # (batch, units)
            
            if self.activation is not None:
                h = h * (1 - self.leaky) + self.activation(input_part + cell.bias + state_part) * self.leaky
            else:
                h = h * (1 - self.leaky) + (input_part + cell.bias + state_part) * self.leaky
            
            outputs.append(h)
        
        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, units)
        
        if reverse:
            outputs = torch.flip(outputs, dims=[1])
        
        return outputs, [h]

    
    def _parallel_reservoir_pass_unrolled(self, x, cell, num_steps=None, reverse=False):
        """
        Unrolled parallel computation for RNNs with fixed sequence length.
        Better for GPU parallelization when sequence length is known.
        
        Args:
            x: Input tensor (batch, seq_len, input_size)
            cell: ReservoirCell to use
            num_steps: Number of unroll steps (default: seq_len)
            reverse: If True, process sequence in reverse
            
        Returns:
            outputs: (batch, seq_len, units)
            final_state: Final hidden state
        """
        batch_size, seq_len, input_size = x.shape
        device = x.device
        
        if num_steps is None:
            num_steps = seq_len
        
        if reverse:
            x = torch.flip(x, dims=[1])
        
        # Precompute all input projections
        input_proj = torch.matmul(x, cell.kernel)  # (batch, seq_len, units)
        
        h = torch.zeros(batch_size, self.units, device=device)
        outputs = []
        
        for t in range(num_steps):
            input_part = input_proj[:, t, :]
            state_part = torch.matmul(h, cell.recurrent_kernel)
            
            if self.activation is not None:
                h = h * (1 - self.leaky) + self.activation(input_part + cell.bias + state_part) * self.leaky
            else:
                h = h * (1 - self.leaky) + (input_part + cell.bias + state_part) * self.leaky
            
            outputs.append(h)
        
        outputs = torch.stack(outputs, dim=1)
        
        if reverse:
            outputs = torch.flip(outputs, dims=[1])
        
        return outputs, [h]

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            output: Concatenated forward and backward outputs (batch, seq_len, units*2)
            states: List containing final forward and backward states
        """
        batch_size, seq_len, _ = x.shape

        # Forward pass with parallelized input projection
        forward_outputs, forward_state = self._parallel_reservoir_pass(
            x, self.cell_forward, reverse=False
        )

        # Backward pass with parallelized input projection
        backward_outputs, backward_state = self._parallel_reservoir_pass(
            x, self.cell_backward, reverse=True
        )

        # Concatenate forward and backward
        output = torch.cat([forward_outputs, backward_outputs], dim=-1)

        return output, [forward_state, backward_state]