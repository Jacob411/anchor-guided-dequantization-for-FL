"""
Main FL simulation runner.

Usage:
    python -m simulation.runner --strategy ours --rounds 50 --num_clients 10

Strategies: ours | fedavg | fedshift | prowd
"""
from __future__ import annotations

import argparse
import copy
import os
import random
from typing import Dict, List

import numpy as np
import torch
import yaml

from data.cifar10 import load_cifar10
from models.cnn import build_model
from models.correction_net import CorrectionNet
from clients.anchor_client import AnchorClient
from clients.constrained_client import ConstrainedClient
from server.correction import ServerCorrection
from server.aggregator import selective_aggregation, fedavg, apply_delta
from baselines.fedavg import FedAvgStrategy
from baselines.fedshift import FedShiftStrategy
from baselines.prowd import ProWDStrategy
from metrics.tracker import MetricsTracker, RoundMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg: dict) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def flatten_model(state_dict: Dict[str, torch.Tensor]) -> int:
    """Return total number of parameters in a state dict."""
    return sum(t.numel() for t in state_dict.values())


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def build_clients(
    cfg: dict,
    client_loaders,
    global_model,
    device: torch.device,
    num_anchors: int,
) -> List:
    """
    Build all client objects.
    Clients 0..num_anchors-1 → AnchorClient
    Clients num_anchors..N-1 → ConstrainedClient with random bit-widths
    """
    clients = []
    bits_choices = cfg["client"]["quantization_bits"]

    for i, loader in enumerate(client_loaders):
        model_copy = copy.deepcopy(global_model)
        kwargs = dict(
            client_id=i,
            model=model_copy,
            dataloader=loader,
            device=device,
            lr=cfg["client"]["local_lr"],
            local_epochs=cfg["client"]["local_epochs"],
        )
        if i < num_anchors:
            clients.append(AnchorClient(**kwargs))
        else:
            bits = random.choice(bits_choices)
            clients.append(ConstrainedClient(bits=bits, **kwargs))

    return clients


# ---------------------------------------------------------------------------
# Our method strategy
# ---------------------------------------------------------------------------

class OurStrategy:
    """Anchor-guided CorrectionNet + score-based selective aggregation."""
    name = "ours"

    def __init__(self, correction: ServerCorrection, similarity_weight: float = 0.5) -> None:
        self.correction = correction
        self.sim_weight = similarity_weight

    def aggregate(
        self,
        all_deltas,
        anchor_deltas,
        quantized_deltas,
        global_state,
        sample_counts=None,
        global_model=None,
        val_loader=None,
    ):
        # 1. Train CorrectionNet
        corr_loss = self.correction.train(
            quantized_deltas, anchor_deltas, global_model, val_loader
        )
        # 2. Apply correction to quantized deltas
        corrected = [self.correction.apply(d) for d in quantized_deltas]
        # 3. Selective aggregation over anchors + corrected
        combined = anchor_deltas + corrected
        n_anchor = len(anchor_deltas)
        n_quant  = len(corrected)
        counts = sample_counts if sample_counts else [1.0] * (n_anchor + n_quant)
        agg_delta, _ = selective_aggregation(
            combined, anchor_deltas, counts, self.sim_weight
        )
        new_state = apply_delta(global_state, agg_delta)
        return new_state, corr_loss


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)

    # Override config with CLI args
    num_clients = args.num_clients or cfg["federation"]["num_clients"]
    num_rounds  = args.rounds     or cfg["federation"]["num_rounds"]
    alpha       = args.alpha      or cfg["data"]["alpha"]
    strategy_name = args.strategy

    set_seed(cfg["experiment"]["seed"])
    device = get_device(cfg)
    print(f"Using device: {device}")

    # ---- Data ----------------------------------------------------------------
    client_loaders, val_loader, test_loader = load_cifar10(
        data_dir    = cfg["data"]["data_dir"],
        num_clients = num_clients,
        alpha       = alpha,
        batch_size  = cfg["client"]["local_batch_size"],
        validation_size = cfg["server"]["validation_size"],
        seed        = cfg["experiment"]["seed"],
    )

    # ---- Global Model --------------------------------------------------------
    global_model = build_model(cfg["model"]["arch"]).to(device)
    global_state = {k: v.cpu() for k, v in global_model.state_dict().items()}

    # ---- Clients -------------------------------------------------------------
    num_anchors = max(1, int(num_clients * cfg["federation"]["anchor_fraction"]))
    clients = build_clients(cfg, client_loaders, global_model, device, num_anchors)
    sample_counts = [len(cl.dataloader.dataset) for cl in clients]  # type: ignore[arg-type]
    print(f"Clients: {num_anchors} anchor, {num_clients - num_anchors} constrained")

    # ---- Strategy ------------------------------------------------------------
    delta_dim = flatten_model(global_state)
    scfg = cfg["server"]
    lcfg = cfg["loss"]

    if strategy_name == "ours":
        correction = ServerCorrection(
            delta_dim    = delta_dim,
            hidden_dim   = scfg["correction_hidden_dim"],
            lr           = scfg["correction_lr"],
            epochs       = scfg["correction_epochs"],
            rec_weight   = lcfg["reconstruction_weight"],
            distill_weight = lcfg["distillation_weight"],
            temperature  = lcfg["distillation_temperature"],
            device       = device,
        )
        strategy = OurStrategy(correction, similarity_weight=0.5)

    elif strategy_name == "fedavg":
        strategy = FedAvgStrategy()  # type: ignore[assignment]

    elif strategy_name == "fedshift":
        strategy = FedShiftStrategy()  # type: ignore[assignment]

    elif strategy_name == "prowd":
        strategy = ProWDStrategy(  # type: ignore[assignment]
            delta_dim  = delta_dim,
            hidden_dim = scfg["correction_hidden_dim"],
            lr         = scfg["correction_lr"],
            epochs     = scfg["correction_epochs"],
            device     = device,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # ---- Metrics -------------------------------------------------------------
    tracker = MetricsTracker(
        output_dir    = cfg["metrics"]["output_dir"],
        strategy_name = strategy.name,
    )

    # ---- FL Loop -------------------------------------------------------------
    for rnd in range(1, num_rounds + 1):
        anchor_deltas    = []
        quantized_deltas = []
        all_deltas       = []
        round_bytes      = 0

        for client in clients:
            delta, comm_bytes = client.train_one_round(global_state)
            round_bytes += comm_bytes
            all_deltas.append(delta)
            if client.is_anchor:
                anchor_deltas.append(delta)
            else:
                quantized_deltas.append(delta)

        # Aggregate
        new_state, corr_loss = strategy.aggregate(
            all_deltas       = all_deltas,
            anchor_deltas    = anchor_deltas,
            quantized_deltas = quantized_deltas,
            global_state     = global_state,
            sample_counts    = sample_counts,
            global_model     = global_model if strategy_name == "ours" else None,
            val_loader       = val_loader   if strategy_name == "ours" else None,
        )
        global_state = new_state
        global_model.load_state_dict({k: v.to(device) for k, v in global_state.items()})

        # Evaluate
        acc, loss = tracker.evaluate(global_model, test_loader, device)
        m = RoundMetrics(
            round_num      = rnd,
            top1_accuracy  = acc,
            loss           = loss,
            uplink_bytes   = round_bytes,
            correction_loss = corr_loss,
        )
        tracker.record(m)
        if rnd % cfg["metrics"]["log_every_n_rounds"] == 0:
            tracker.print_round(m)

    print(f"\nTotal uplink: {tracker.total_uplink_mb():.2f} MB")
    print(f"Final accuracy: {tracker.history[-1].top1_accuracy:.2f}%")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anchor-based Quantization FL Runner")
    parser.add_argument("--config",      type=str, default="configs/default.yaml")
    parser.add_argument("--strategy",   type=str, default="ours",
                        choices=["ours", "fedavg", "fedshift", "prowd"],
                        help="Aggregation strategy to use")
    parser.add_argument("--rounds",     type=int, default=None, help="Override num_rounds")
    parser.add_argument("--num_clients",type=int, default=None, help="Override num_clients")
    parser.add_argument("--alpha",      type=float, default=None, help="Dirichlet alpha")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
