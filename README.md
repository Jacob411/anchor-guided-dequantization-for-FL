# Anchor-Based Quantization FL

A federated learning framework that uses **anchor clients** (full-precision) to supervise a server-side **CorrectionNet** that debiases quantized model updates before aggregation.

## Method Overview

```
Round t:
  ├── Anchor clients   → train locally → send Δw_fp (full precision)
  ├── Constrained clients → train locally → quantize Δw → send Δw_q (INT2–FP16)
  │
  └── Server:
        1. Train CorrectionNet:  Δw_q → Δw_fp  (supervised by anchors)
        2. Apply correction to all Δw_q
        3. Score-based selective aggregation (cosine sim to anchor mean)
        4. Update global model
```

Compared to:
- **FedShift**: global mean-shift alignment (our method is direction-aware, data-driven)
- **ProWD**: server-side dequantization MLP (our method uses real FP anchors as supervision)

## Project Structure

```
anchor-based-quantization-FL/
├── configs/default.yaml        # All hyperparameters
├── data/cifar10.py             # CIFAR-10 + Dirichlet non-IID partitioning
├── models/
│   ├── cnn.py                  # Shared CNN backbone
│   └── correction_net.py       # Server-side CorrectionNet (residual MLP)
├── quantization/quantizers.py  # INT2/4/8, FP16 quantizers
├── clients/
│   ├── anchor_client.py        # Full-precision updates
│   └── constrained_client.py   # Quantized updates
├── server/
│   ├── correction.py           # CorrectionNet training + application
│   └── aggregator.py           # FedAvg + score-based selective aggregation
├── losses/losses.py            # Reconstruction + distillation losses
├── baselines/
│   ├── fedavg.py               # Vanilla FedAvg
│   ├── fedshift.py             # FedShift baseline
│   └── prowd.py                # ProWD baseline (simplified)
├── metrics/tracker.py          # Per-round metrics + CSV logging
├── simulation/runner.py        # Main simulation loop
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Running Experiments

```bash
# Our method
python -m simulation.runner --strategy ours --rounds 50 --num_clients 10

# Baselines
python -m simulation.runner --strategy fedavg   --rounds 50
python -m simulation.runner --strategy fedshift --rounds 50
python -m simulation.runner --strategy prowd    --rounds 50
```

### CLI Options

| Flag            | Default              | Description                          |
|-----------------|----------------------|--------------------------------------|
| `--strategy`    | `ours`               | `ours`, `fedavg`, `fedshift`, `prowd`|
| `--rounds`      | from config          | Number of FL rounds                  |
| `--num_clients` | from config          | Total number of clients              |
| `--alpha`       | from config          | Dirichlet α (non-IID level)          |
| `--config`      | `configs/default.yaml` | Path to config file                |

Results are saved to `results/results_<strategy>.csv`.

## Key Hyperparameters (`configs/default.yaml`)

| Parameter              | Default | Description                              |
|------------------------|---------|------------------------------------------|
| `anchor_fraction`      | 0.2     | Fraction of clients that are anchors     |
| `quantization_bits`    | [2,4,8,16] | Bit-widths for constrained clients    |
| `correction_epochs`    | 5       | CorrectionNet inner steps per round      |
| `alpha`                | 0.5     | Dirichlet concentration (non-IID)        |
| `distillation_weight`  | 0.5     | Weight on KL distillation loss           |

## Metrics

- **Top-1 Accuracy** — global test accuracy per round
- **Convergence Speed** — round-over-round accuracy delta
- **Uplink Communication Volume** — total MB sent by clients
