# RTRL-SnAp Reproduction

A lightweight reproduction project for **Practical Real Time Recurrent Learning with a Sparse Approximation to the Jacobian** by Menick et al.

This project focuses on understanding and reproducing the core idea of **Real-Time Recurrent Learning (RTRL)** and the **Sparse n-Step Approximation (SnAp)**. The initial goal is not to reproduce every experiment in the paper immediately, but to build a clear and verifiable implementation of:

1. A Copy Task benchmark.
2. A Vanilla RNN trained with BPTT.
3. Exact RTRL with gradient verification.
4. SnAp-1 as a sparse approximation to the RTRL influence matrix.
5. Experimental comparison between BPTT, RTRL, and SnAp-1.

Paper link:

* OpenReview: https://openreview.net/forum?id=q3KSThy2GwB
* PDF: https://openreview.net/pdf?id=q3KSThy2GwB

---

## Project Motivation

Backpropagation Through Time, or BPTT, is the standard way to train recurrent neural networks. However, BPTT requires storing past hidden states and usually updates model parameters only after processing a sequence or sequence segment.

Real-Time Recurrent Learning, or RTRL, provides an alternative online learning approach. Instead of backpropagating through a stored sequence history, RTRL maintains an influence matrix:

```text
J_t = ∂h_t / ∂θ
```

where `h_t` is the current hidden state and `θ` represents the model parameters.

The RTRL update can be written as:

```text
J_t = I_t + D_t J_{t-1}
```

where:

```text
J_t = current influence matrix
I_t = direct influence of parameters on current hidden state
D_t = recurrent Jacobian ∂h_t / ∂h_{t-1}
```

The main limitation of exact RTRL is its high computational and memory cost. The SnAp method proposed in the paper approximates the influence matrix by keeping only the entries that are reachable within `n` recurrent steps.

This project starts with the simplest useful setting: **Vanilla RNN + Copy Task + BPTT / RTRL / SnAp-1**.

---

## Current Scope

The first reproduction target is the synthetic **Copy Task**, because it is easier to control and debug than a large-scale language modeling dataset.

The initial implementation will compare:

```text
BPTT baseline
Exact RTRL
SnAp-1
```

The later extension may include:

```text
SnAp-2
Sparse recurrent networks
GRU / LSTM variants
WikiText103 language modeling
```

---

## Project Structure

```text
rtrl-snap-reproduction/
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── copy_bptt.yaml
│   ├── copy_rtrl.yaml
│   ├── copy_snap1.yaml
│   └── copy_debug.yaml
│
├── scripts/
│   ├── train_copy.py
│   ├── eval_copy.py
│   └── plot_results.py
│
├── src/
│   └── rtrl_snap/
│       │
│       ├── __init__.py
│       │
│       ├── tasks/
│       │   ├── __init__.py
│       │   └── copy_task.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── vanilla_rnn.py
│       │   └── readout.py
│       │
│       ├── algorithms/
│       │   ├── __init__.py
│       │   ├── bptt.py
│       │   ├── rtrl.py
│       │   └── snap.py
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── train_state.py
│       │   ├── losses.py
│       │   └── optim.py
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   └── plotting.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── config.py
│           ├── random.py
│           └── logging.py
│
├── tests/
│   ├── test_copy_task.py
│   ├── test_vanilla_rnn_shapes.py
│   ├── test_rtrl_matches_bptt.py
│   └── test_snap_mask.py
│
├── results/
│   ├── logs/
│   ├── checkpoints/
│   └── figures/
│
└── notebooks/
    ├── 01_copy_task_visualization.ipynb
    └── 02_rtrl_gradient_check.ipynb
```

---

## Implementation Plan

### Stage 0: Project Setup

Set up the repository structure and basic configuration files.

Tasks:

```text
Create project folders
Add README.md
Add requirements.txt
Add pyproject.toml
Add initial config files
Add test folder
```

Expected output:

```text
A clean Python project structure ready for implementation.
```

---

### Stage 1: Copy Task and BPTT Baseline

The first technical step is to implement the Copy Task.

Example:

```text
Input:
1 0 1 1 <delimiter> <blank> <blank> <blank> <blank>

Target:
<blank> <blank> <blank> <blank> <blank> 1 0 1 1
```

Tasks:

```text
Implement Copy Task data generator
Implement Vanilla RNN
Implement output readout layer
Implement cross-entropy loss
Train Vanilla RNN with BPTT
Verify that loss decreases
```

Expected output:

```text
A working BPTT baseline on the Copy Task.
```

---

### Stage 2: Exact RTRL

Implement exact RTRL for a small Vanilla RNN.

Tasks:

```text
Maintain the influence matrix J_t = ∂h_t / ∂θ
Implement the RTRL recurrence
Compare RTRL gradients with BPTT gradients
Add gradient check tests
```

The most important test in this stage is:

```text
RTRL gradient ≈ BPTT gradient
```

Expected output:

```text
A verified exact RTRL implementation on a small RNN.
```

---

### Stage 3: SnAp-1

Implement SnAp-1 as a sparse approximation to the RTRL influence matrix.

Initial implementation strategy:

```text
Compute the dense RTRL update
Apply a SnAp-1 mask to the influence matrix
Compare SnAp-1 with exact RTRL and BPTT
```

This implementation is not intended to be maximally efficient at first. The first goal is correctness and interpretability.

Expected output:

```text
A working SnAp-1 implementation with experimental comparison on the Copy Task.
```

---

### Stage 4: Extended Reproduction

After the basic pipeline works, extend the project toward the paper-level experiments.

Possible extensions:

```text
Implement SnAp-2
Implement sparse recurrent networks
Run multiple random seeds
Plot learning curves
Compare online update behavior
Attempt WikiText103 language modeling
```

Expected output:

```text
A more complete reproduction of the paper's reported results.
```

---

## Installation

This project is intended to run with Python 3.10 or later.

Install dependencies:

```bash
pip install -r requirements.txt
```

Suggested initial dependencies:

```text
jax
jaxlib
optax
numpy
matplotlib
pyyaml
pytest
tqdm
```

---

## Running Experiments

Train the BPTT baseline:

```bash
python scripts/train_copy.py --config configs/copy_bptt.yaml
```

Train exact RTRL:

```bash
python scripts/train_copy.py --config configs/copy_rtrl.yaml
```

Train SnAp-1:

```bash
python scripts/train_copy.py --config configs/copy_snap1.yaml
```

Run tests:

```bash
pytest
```

---

## Key Tests

The most important tests are:

```text
test_copy_task.py
```

Checks that Copy Task samples have the correct format and shapes.

```text
test_vanilla_rnn_shapes.py
```

Checks that the Vanilla RNN forward pass returns correctly shaped hidden states and logits.

```text
test_rtrl_matches_bptt.py
```

Checks that exact RTRL gradients match BPTT gradients on a small model.

```text
test_snap_mask.py
```

Checks that the SnAp mask keeps and removes the expected influence matrix entries.

---

## Development Notes

The project should be developed in the following order:

```text
1. Copy Task generator
2. Vanilla RNN forward pass
3. BPTT baseline
4. Exact RTRL update
5. RTRL vs BPTT gradient check
6. SnAp-1 mask
7. Copy Task comparison experiments
8. Plot results
9. Extend to sparse RNN / SnAp-2
```

The first major milestone is not to reproduce all paper results. The first major milestone is:

```text
Exact RTRL gradients match BPTT gradients on a small Vanilla RNN.
```

Once this is verified, the SnAp approximation can be implemented and evaluated.

---

## Status

Work in progress.

Current target:

```text
Copy Task + Vanilla RNN + BPTT / exact RTRL / SnAp-1
```

Future target:

```text
Sparse RNN + SnAp-2 + paper-level Copy Task reproduction
```

---

## Reference

Jacob Menick, Erich Elsen, Utku Evci, Simon Osindero, Karen Simonyan, Alex Graves.

**Practical Real Time Recurrent Learning with a Sparse Approximation to the Jacobian.**

ICLR 2021.

OpenReview: https://openreview.net/forum?id=q3KSThy2GwB
