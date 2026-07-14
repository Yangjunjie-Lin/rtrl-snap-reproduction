# RTRL-SnAp Reproduction

A lightweight Copy Task reproduction of **BPTT**, **exact RTRL**, and **SnAp-1** for the paper:

**Practical Real Time Recurrent Learning with a Sparse Approximation to the Jacobian** (Menick et al., ICLR 2021).

- OpenReview: https://openreview.net/forum?id=q3KSThy2GwB
- PDF: https://openreview.net/pdf?id=q3KSThy2GwB

This is currently a lightweight educational reproduction focused on the Copy Task. It does not yet reproduce the full WikiText103 experiments.

---

## Implemented

| Component | Status |
|-----------|--------|
| Copy Task | Done |
| Vanilla RNN | Done |
| BPTT training | Done |
| Exact RTRL (gradients + training) | Done |
| SnAp-1 (mask + training) | Done |
| RTRL vs BPTT gradient checks | Done |
| CSV logging / simple plots | Done |

Supported algorithms: `bptt`, `rtrl`, `snap1`  
Supported task: Copy Task

---

## Installation

Python 3.10+ recommended.

```bash
pip install -e .
```

---

## Tests

```bash
pytest
```

Key tests:

- `tests/test_copy_task.py` — Copy Task shapes / format
- `tests/test_vanilla_rnn_shapes.py` — RNN forward shapes
- `tests/test_losses.py` — masked loss / accuracy
- `tests/test_rtrl_matches_bptt.py` — exact RTRL matches BPTT on a tiny model
- `tests/test_snap_mask.py` — SnAp-1 structural mask

Manual smoke check (short runs of all three algorithms):

```bash
python scripts/run_smoke_tests.py
```

---

## Training

```bash
python scripts/train_copy.py --config configs/copy_bptt.yaml
python scripts/train_copy.py --config configs/copy_rtrl.yaml
python scripts/train_copy.py --config configs/copy_snap1.yaml
```

RTRL and SnAp-1 configs stay small (`hidden_size: 8`, `copy_length: 3`) because exact / dense-mask influence matrices are expensive.

Optional CSV logging is enabled in the configs:

```yaml
logging:
  save_csv: true
  output_path: results/logs/copy_bptt.csv
```

Plot one or more CSV logs:

```bash
python scripts/plot_results.py --csv results/logs/copy_bptt.csv --metric loss --output results/figures/copy_bptt_loss.png
```

---

## Project layout

```text
rtrl-snap-reproduction/
├── configs/
│   ├── copy_bptt.yaml
│   ├── copy_rtrl.yaml
│   └── copy_snap1.yaml
├── scripts/
│   ├── train_copy.py
│   ├── plot_results.py
│   └── run_smoke_tests.py
├── src/rtrl_snap/
│   ├── algorithms/   # bptt.py, rtrl.py, snap.py
│   ├── models/       # vanilla_rnn.py, readout.py
│   ├── tasks/        # copy_task.py
│   ├── training/     # losses.py
│   ├── evaluation/   # plotting.py
│   └── utils/        # config.py
└── tests/
```

---

## Method sketch

Vanilla RNN:

```text
h_t = tanh(x_t W_xh + h_{t-1} W_hh + b_h)
logits_t = h_t W_hy + b_y
```

Exact RTRL stores the influence matrix `J_t = ∂h_t / ∂θ_recurrent` and updates:

```text
J_t = I_t + D_t J_{t-1}
```

SnAp-1 applies a structural mask so only direct one-step parameter→hidden influences are kept.

---

## Known limitations

- Dense exact RTRL / dense-mask SnAp-1; not the paper’s efficient sparse kernels.
- Copy Task only; no WikiText103, GRU/LSTM, or Hydra/W&B experiment stack.
- Educational priority over wall-clock speed.

---

## Next steps

- SnAp-2 and multi-seed learning curves
- Sparse recurrent connectivity as in the paper
- Optional WikiText103 / language-model experiments
