import torch
import torch.nn as nn
import numpy as np
from reservoir import *
from readout import *
from typing import Tuple
import torch.nn.functional as F
from sklearn.metrics import (
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score
)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

import time

# Custom class for timing
class Timer:
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed_time = time.time() - self.start_time

        

class DeepESN(nn.Module):
    """
    Optimized Deep Echo State Network with parallelized forward pass.
    
    author: Prof. Claudio Gallicchio (c.gallicch)   (originally in Keras/TF)
    edited and implemented in Pytorch by Dr. Corrado Baccheschi (Bakko000)
    
    Implements a Deep Echo State Network (DeepESN) model for time-series classification or regression problems.
    This model consists of:
    - From 1 to multiple recurrent layers, each utilizing a `ReservoirCell` or bidirectional one for nonlinear feature extraction.
    - A trainable ridge readout layer for big data for producing the final predictions. 

    Reference:
    Gallicchio, Claudio, Alessio Micheli, and Luca Pedrelli.
    "Deep reservoir computing: A critical experimental analysis." Neurocomputing 268 (2017): 87-99.
    """

    def __init__(self, input_size, units,
                 num_layers=3,
                 input_scaling=1., bias_scaling=1.0, spectral_radius=0.9,
                 leaky=1,
                 bias_scaling_hidden=1.0,
                 spectral_radius_hidden=0.9,
                 input_scaling_hidden=1.,
                 leaky_hidden=1,
                 readout_regularizer=[0],
                 type="nobi",   # bi or nobi
                 score="accuracy",
                 last_layer=False,
                 sequences=False,
                 mean=False,
                 activation=torch.tanh,
                 use_parallel=True,  # NEW: enable parallelization
                 **kwargs):

        super().__init__()
        self.type = type
        self.sequences = sequences
        self.score = score
        self.last = last_layer
        self.mean = mean
        self.units = units
        self.num_layers = num_layers
        self.readout_regularizer = readout_regularizer
        self.readout = None
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky
        self.input_scaling_hidden = input_scaling_hidden
        self.bias_scaling_hidden = bias_scaling_hidden
        self.spectral_radius_hidden = spectral_radius_hidden
        self.leaky_hidden = leaky_hidden
        self.activation = activation
        self.use_parallel = use_parallel  # NEW: parallelization flag

        self.reservoir_layers = nn.ModuleList()

        # ===== CASE 1: single-layer ESN =====
        if num_layers == 1:
            cell = ReservoirCell(
                input_size=input_size,
                units=units,
                input_scaling=input_scaling,
                bias_scaling=bias_scaling,
                spectral_radius=spectral_radius,
                leaky=leaky,
                activation=activation
            )

            if type == "bi":
                self.reservoir_layers.append(BidirectionalReservoirParallel(cell, units))
            else:
                self.reservoir_layers.append(cell)
        else:
            # First layer
            first_cell = ReservoirCell(input_size=input_size,
                                       units=units,
                                       input_scaling=input_scaling,
                                       bias_scaling=bias_scaling,
                                       spectral_radius=spectral_radius,
                                       leaky=leaky,
                                       activation=activation)
    
            if type == "bi":
                self.reservoir_layers.append(BidirectionalReservoirParallel(first_cell, units))
                current_input_size = units * 2
            else:
                self.reservoir_layers.append(first_cell)
                current_input_size = units
    
            # Add the remaining reservoir layers (middle layers)
            for _ in range(num_layers - 2):
                cell = ReservoirCell(input_size=current_input_size,
                                    units=units,
                                    input_scaling=input_scaling_hidden,
                                    bias_scaling=bias_scaling_hidden,
                                    spectral_radius=spectral_radius_hidden,
                                    leaky=leaky_hidden,
                                    activation=activation)
    
                if type == "bi":
                    self.reservoir_layers.append(BidirectionalReservoirParallel(cell, units))
                    current_input_size = units * 2
                else:
                    self.reservoir_layers.append(cell)
                    current_input_size = units
    
            # Last layer
            last_cell = ReservoirCell(input_size=current_input_size,
                                     units=units,
                                     input_scaling=input_scaling_hidden,
                                     bias_scaling=bias_scaling_hidden,
                                     spectral_radius=spectral_radius_hidden,
                                     leaky=leaky_hidden,
                                     activation=activation)
    
            if type == "bi":
                self.reservoir_layers.append(BidirectionalReservoirParallel(last_cell, units))
            else:
                self.reservoir_layers.append(last_cell)

    def forward(self, x, states=None):
        """
        Parallelized forward pass with precomputed input projections.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            states: Optional initial states for each layer

        Returns:
            output: Tensor of shape depends on sequences and mean flags
            states: List of final states for each layer
        """
        batch_size, seq_len, _ = x.shape

        all_outputs = []
        all_states = []

        current_input = x

        for i, layer in enumerate(self.reservoir_layers):
            if isinstance(layer, BidirectionalReservoirParallel):
                # Bidirectional layer (already has optimized forward)
                outputs, layer_state = layer(current_input)
            else:
                # Unidirectional layer with parallelization
                if self.use_parallel:
                    outputs, layer_state = self._process_layer_parallel_fused(layer, current_input)
                else:
                    outputs, layer_state = self._process_layer(layer, current_input, batch_size, seq_len)
            
            if not self.sequences:
                layer_output = outputs[:, -1, :]  # take only the last timestep
            else:
                layer_output = outputs  # 3D sequence
            
            all_outputs.append(layer_output) 
            all_states.append(layer_state)
            current_input = outputs

        # Handle output based on flags
        if self.last:
            final_output = all_outputs[-1]
        else:
            if self.sequences:
                # Concatenate all layer outputs along feature dimension
                final_output = torch.cat(all_outputs, dim=2)
            else:
                final_output = torch.cat(all_outputs, dim=1)
        
        if self.mean:
            # Average over time dimension
            final_output = final_output.mean(dim=1)

        return final_output, all_states

    def _process_layer_parallel(self, cell, input_seq):
        """
        Parallelized processing of a single unidirectional reservoir layer.
        
        Key optimizations:
        1. Batch matrix multiplication for input projection (precomputed for all timesteps)
        2. Recurrence is still sequential but with GPU-friendly operations
        3. Eliminates CPU-GPU transfer latency
        
        Args:
            cell: ReservoirCell to process
            input_seq: Input sequence (batch, seq_len, input_size)
            
        Returns:
            outputs: (batch, seq_len, units)
            state: Final state
        """
        batch_size, seq_len, input_size = input_seq.shape
        device = input_seq.device
        
        # OPTIMIZATION 1: Precompute all input projections at once
        # Instead of: for each timestep: input_proj = input @ kernel
        # Do: all_input_proj = input_seq @ kernel  (batch matrix multiply)
        # This is parallelizable on GPU and reduces CPU-GPU transfers
        all_input_projections = torch.matmul(input_seq, cell.kernel)  # (batch, seq_len, units)
        
        # Initialize hidden state
        state = torch.zeros(batch_size, cell.units, device=device)
        outputs = []
        
        # OPTIMIZATION 2: Recurrence computation with precomputed inputs
        # Recurrence is still sequential (inherent to RNNs), but:
        # - Input part is already computed (no per-timestep CPU work)
        # - Matrix operations are GPU-parallelized
        for t in range(seq_len):
            # Get precomputed input projection
            input_part = all_input_projections[:, t, :]  # (batch, units)
            
            # Compute state contribution (GPU-parallelized)
            state_part = torch.matmul(state, cell.recurrent_kernel)  # (batch, units)
            
            # Update state (GPU operations)
            if cell.activation is not None:
                new_state = state * (1 - cell.leaky) + \
                           cell.activation(input_part + cell.bias + state_part) * cell.leaky
            else:
                new_state = state * (1 - cell.leaky) + \
                           (input_part + cell.bias + state_part) * cell.leaky
            
            state = new_state
            outputs.append(state)
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, units)
        
        return outputs, [state]

    def _process_layer_parallel_fused(self, cell, input_seq):
        """
        Fused parallelized processing with minimal overhead.
        Uses torch.nn.functional operations for better performance.
        
        Args:
            cell: ReservoirCell to process
            input_seq: Input sequence (batch, seq_len, input_size)
            
        Returns:
            outputs: (batch, seq_len, units)
            state: Final state
        """
        batch_size, seq_len, input_size = input_seq.shape
        device = input_seq.device
        
        # Precompute input projections (vectorized, GPU-parallelized)
        input_proj = F.linear(input_seq, cell.kernel.t(), bias=None)  # (batch, seq_len, units)
        
        # Initialize state
        state = torch.zeros(batch_size, cell.units, device=device, dtype=input_seq.dtype)
        outputs = []
        
        # Sequential recurrence with GPU acceleration
        for t in range(seq_len):
            # Fused computation
            input_part = input_proj[:, t, :]
            state_part = F.linear(state, cell.recurrent_kernel.t(), bias=None)
            
            combined = input_part + state_part + cell.bias
            
            if cell.activation is not None:
                activated = cell.activation(combined)
            else:
                activated = combined
            
            state = state * (1 - cell.leaky) + activated * cell.leaky
            outputs.append(state)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs, [state]

    def _process_layer(self, cell, input_seq, batch_size, seq_len):
        """
        Original sequential processing (for reference/fallback).
        This has CPU latency from per-timestep operations.
        """
        outputs = []
        state = [torch.zeros(batch_size, cell.units, device=input_seq.device)]

        for t in range(seq_len):
            output, state = cell(input_seq[:, t, :], state)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)
        return outputs, state

    def predict(self, x):
        """Make predictions on input sequence"""
        states, _ = self(x)
        return self.readout(states)
        

    def fit(self, train, labels, num_targets, validation_data: Tuple[torch.Tensor, torch.Tensor]=None, batches=None, verbose: bool = False, device=device, **kwargs):
        """
        Fits the Readout layer to the given training data, with optional validation.

        This function initializes and trains a custom Readout layer for big data using the provided training data. 
        It supports both full dataset training and batch-based training. If validation data is provided, 
        the model evaluates performance using either accuracy or F1-score (in this context, macro).
        It is possible to add other metrics, just edit the validate function.

        :param train: The training data tensor of shape (num_samples, num_features).
        :param labels: The corresponding labels tensor of shape (num_samples, dim).
        :param num_targets: The number of target classes or outputs for prediction.
        :param validation_data: (Optional) A tuple (x_val, y_val) containing validation data and labels.
                                If provided, the model evaluates validation performance on the DeepESNs lambda coefficients.
        :param batches: (Optional) An iterator yielding batches of (x_batch, y_batch) for batch training.
        :param verbose: (Optional) If True, prints additional training details.
        :param device: GPU or CPU for pytorch.
        :param kwargs: Additional parameters for customization.

        :return: 
            - If validation is provided: A tuple (validation_error, fitting_time, fitting_time_ms),
            where validation_error is the validation loss, and fitting_time is the training duration in seconds and milliseconds.
            - If no validation is provided: A tuple (fitting_time, fitting_time_ms).
        """
        # Initialize the Readout layer with the correct number of targets 
        self.readout = Readout(num_features=train.shape[1], num_targets=num_targets).to(device)
        
        if validation_data is not None:

            x_val, y_val = validation_data
          
            def validate(readout_params: Tuple[torch.Tensor, torch.Tensor]) -> float:
                  W, b = readout_params
                  y_pred = F.linear(x_val, W, b)  # readout fw pass
                  
                  if self.score == "accuracy":
                    # Convert predictions to class indices
                    y_pred_classes = torch.argmax(y_pred, dim=-1)  # (batch_size,)
                
                    # Handle one-hot encoded labels by converting to indices
                    if len(y_val.shape) > 1 and y_val.shape[-1] > 1:
                        y_true_classes = torch.argmax(y_val, dim=-1)
                        accuracy = torch.mean((y_pred_classes == y_true_classes).float())

                    else:
                        y_pred_sum = torch.sum(y_pred, dim=1)
                        y_pred_bin = torch.sign(y_pred_sum)
                        # accuracy 
                        accuracy = (y_pred_bin == y_val.squeeze()).float().mean()

                    score_metric = accuracy
                  else:
                    # True classes: 
                    true_classes = torch.argmax(y_val, dim=-1)
                    predicted_classes = torch.argmax(y_pred, dim=-1)
                        # Calcola il F1 score
                    f1_macro = f1_score(true_classes.cpu().numpy(), predicted_classes.cpu().numpy(), average='macro')
                    score_metric = f1_macro

                  return 1.0 - score_metric

              # Fit the readout layer with validation
            try:
                    start_time = time.time()
                    if not batches:
                        training = (train,labels)
                        batch_mode=False
                        print("Fitting the readout layer with validation")
                    else:
                        print("Fitting the readout layer on batch mode with validation")
                        training=batches
                        batch_mode=True
                    self.readout.fit(training, regularization=self.readout_regularizer, validate=validate, batch_mode= batch_mode, verbose=verbose)
                    end_time = time.time()
                    fitting_time = end_time - start_time
                    fitting_time_ms = fitting_time * 1000  # Convert to milliseconds

                    print(f"Readout fitting completed in {fitting_time:.4f} seconds ({fitting_time_ms:.2f} milliseconds)")
                    # Calculate validation predictions
                    W, b = self.readout.weight, self.readout.bias
                    
            except RuntimeError as e:
                      print(f'Errore durante il fit: {e}')
                      return float('inf')
            # Calculate validation error
            print("Calculating validation error")
            validation_error = validate((self.readout.weight, self.readout.bias))
            return validation_error, fitting_time, fitting_time_ms
        else:
              start_time = time.time()
              training = (train, labels)
              print("Fitting the readout layer without validation")
              self.readout.fit(training, regularization=self.readout_regularizer, verbose=verbose)
              end_time = time.time()
              fitting_time = end_time - start_time
              fitting_time_ms = fitting_time * 1000  # Convert to milliseconds
              print(f"Readout fitting completed in {fitting_time:.4f} seconds ({fitting_time_ms:.2f} milliseconds)")
              return fitting_time, fitting_time_ms


