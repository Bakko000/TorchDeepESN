# Deep Reservoir Computing with Linear Readout for Audio Data

A PyTorch implementation of **Deep Echo State Networks (DeepESN)** and related reservoir computing models, designed for efficient sequence representation with a **fixed recurrent backbone** and a **linear ridge readout**.
Ref. accepted paper: **C. Baccheschi and P. Dazzi, “An Analysis of Untrained Deep Reservoir Networks for Audio Surveillance,”** in Proceedings of the IEEE International Conference on Advanced Video and Signal-Based Surveillance (AVSS), Lecce(Italy), 2026, forthcoming.

This repository focuses on the following idea:

- use a **reservoir backbone** as a nonlinear temporal feature extractor;
- keep the recurrent layers **untrained**;
- train only a **linear readout**, making the approach particularly attractive for large-scale settings where full backpropagation through time may be expensive.

The core implementation is provided in [`deepesn.py`](./DeepESN.py).

---

## Overview

This repository implements:

- **single-layer ESN**
- **deep ESN**
- **bidirectional DeepESN**
- **ridge-based linear readout**
- support for **sequence-to-sequence** and **sequence-to-one/sequence-to-vector** tasks
- support for (mean)pooled or last-state outputs
- support for **large datasets** through a readout fitting procedure based on sufficient statistics

---

## Repository Structure

```text
.
├── deepesn.py      # main DeepESN model
├── reservoir.py    # reservoir cell and bidirectional wrapper
├── readout.py      # linear readout + ridge fitting utilities
└── ...
```
---

# Example usage

## Create a DeepESN model

The main constructor is:

```text
model = DeepESN(
    input_size=your_input,
    units=100,
    num_layers=2,
    input_scaling=1.0,
    bias_scaling=1.0,
    spectral_radius=0.9,
    leaky=1.0,
    bias_scaling_hidden=1.0,
    spectral_radius_hidden=0.9,
    input_scaling_hidden=1.0,
    leaky_hidden=1.0,
    readout_regularizer=[0,1e-4,1e-3,1e-2,1,10,100],   # they will be used to regularize the readout !
    type="nobi",
    score="accuracy",
    last_layer=False,
    sequences=False,
    mean=False,
    activation=torch.tanh
)
```

## 1. Extract reservoir features

```text
features, all_states = model(x)
```

## 2. Train the readout 
### Find the best validation performance and the best lambda

For multiclassification, remember to give to the readout the onehot representation. Conversely, for the binary case, just encode in {+1, -1}

```text
val_error, fit_time_s, fit_time_ms = model.fit(
    train=features,
    labels=labels,
    num_targets=num_classes,
    validation_data=(val_features, val_labels),
    verbose=True,
    device=cpu|cuda
)
```


## 3. Predict
```text
predictions = model.predict(test_features)
```
---

# Authors
-Prof. Claudio Gallicchio — original DeepESN formulation and deep reservoir computing research

-Dr. Corrado Baccheschi — PyTorch implementation and extensions


# Cite

If you use this code entirely or partially please cite the following:
```text
Gallicchio, Claudio, Alessio Micheli, and Luca Pedrelli. "Deep reservoir computing: A critical experimental analysis." Neurocomputing 268 (2017): 87-99.
```

and

```text
@inproceedings{baccheschidazzi2026,
  author    = {Corrado Baccheschi and Patrizio Dazzi},
  title     = {An Analysis of Untrained Deep Reservoir Networks for Audio Surveillance},
  booktitle = {Proceedings of the IEEE International Conference on Advanced Visual and Signal-Based Systems (AVSS)},
  year      = {2026},
  note      = {accepted, forthcoming}
}
```
