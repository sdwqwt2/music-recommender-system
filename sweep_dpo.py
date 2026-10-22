"""
sweep_dpo.py
------------
Hyperparameter Grid Sweep & Pareto Frontier Exploration for DPO Alignment.

Explores:
1. beta (KL Regularization): [0.05, 0.1, 0.2]
2. sft_weight (Auxiliary Supervised Regularization): [0.0, 0.05, 0.1, 0.2]
3. preference_strategy (Reward Definition): ["binary_skip", "played_ratio", "margin_threshold"]

Computes the trade-off curve between:
- Engagement / Win-Rate (Preference alignment)
- Ranking Accuracy (Val HR@10, Val NDCG@10)
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW

from data_utils import ItemTokenizer, set_seed
from dataset import SequentialRecDataset
from dpo_dataset import create_dpo_dataloader
from model import RecSysGPT2, build_gpt2_recsys_model
from train_base import evaluate, get_device, get_lr_scheduler
from train_dpo import compute_dpo_loss, evaluate_dpo_validation, train_dpo_epoch

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] (%(name)s) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SweepDPO")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def run_dpo_trial(
    trial_id: int,
    beta: float,
    sft_weight: float,
    strategy: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    base_checkpoint_path: Path,
    device: torch.device,
    epochs: int = 2,
    batch_size: int = 64,
    lr: float = 5e-5,
    use_amp: bool = False,
) -> Dict[str, Any]:
    """Runs a single fast DPO trial for given hyperparameters."""
    logger.info("=== Trial #%d: beta=%.2f | sft_weight=%.2f | strategy=%s ===", trial_id, beta, sft_weight, strategy)

    checkpoint = torch.load(base_checkpoint_path, map_location=device)
    cfg = checkpoint["model_config"]

    # 1. Initialize models
    policy_model = build_gpt2_recsys_model(
        num_items=cfg["num_items"],
        max_length=cfg["max_length"],
        n_embd=cfg["n_embd"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
    ).to(device)
    policy_model.load_state_dict(checkpoint["model_state_dict"])

    ref_model = build_gpt2_recsys_model(
        num_items=cfg["num_items"],
        max_length=cfg["max_length"],
        n_embd=cfg["n_embd"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
    ).to(device)
    ref_model.load_state_dict(checkpoint["model_state_dict"])
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # 2. DataLoaders
    train_dpo_loader = create_dpo_dataloader(
        df=train_df,
        max_length=cfg["max_length"],
        batch_size=batch_size,
        shuffle=True,
        preference_strategy=strategy,
        max_pairs_per_user=2,
    )
    val_dpo_loader = create_dpo_dataloader(
        df=val_df,
        history_df=train_df,
        max_length=cfg["max_length"],
        batch_size=batch_size,
        shuffle=False,
        preference_strategy=strategy,
        max_pairs_per_user=2,
    )

    val_seq_dataset = SequentialRecDataset(df=val_df, history_df=train_df, max_length=cfg["max_length"], is_eval=True)
    val_seq_loader = torch.utils.data.DataLoader(
        val_seq_dataset, batch_size=batch_size, shuffle=False, collate_fn=val_seq_dataset.collate_fn
    )

    optimizer = AdamW(policy_model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_dpo_loader) * epochs
    scheduler = get_lr_scheduler(optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    final_val_win = 0.0
    final_val_margin = 0.0
    final_hr = 0.0
    final_ndcg = 0.0

    t0 = time.time()
    for ep in range(1, epochs + 1):
        train_dpo_epoch(
            policy_model=policy_model,
            ref_model=ref_model,
            dataloader=train_dpo_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            beta=beta,
            sft_weight=sft_weight,
            epoch=ep,
        )

        val_dpo = evaluate_dpo_validation(
            policy_model=policy_model,
            ref_model=ref_model,
            dataloader=val_dpo_loader,
            device=device,
            use_amp=use_amp,
            beta=beta,
            sft_weight=sft_weight,
        )
        val_seq = evaluate(policy_model, val_seq_loader, device, use_amp=use_amp, top_k=10)

        final_val_win = val_dpo["val_win_rate"]
        final_val_margin = val_dpo["val_margin"]
        final_hr = val_seq["val_hr@10"]
        final_ndcg = val_seq["val_ndcg@10"]

    elapsed = time.time() - t0
    result = {
        "trial_id": trial_id,
        "beta": beta,
        "sft_weight": sft_weight,
        "strategy": strategy,
        "val_win_rate": final_val_win,
        "val_margin": final_val_margin,
        "val_hr@10": final_hr,
        "val_ndcg@10": final_ndcg,
        "elapsed_sec": round(elapsed, 1),
    }
    logger.info("Trial #%d Complete: WinRate=%.1f%% | Margin=%+.3f | HR@10=%.4f | NDCG@10=%.4f",
                trial_id, final_val_win, final_val_margin, final_hr, final_ndcg)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO Hyperparameter Sweep")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--base_checkpoint", type=str, default="checkpoints/best_gpt2_recsys.pt")
    parser.add_argument("--output_json", type=str, default="reports/dpo_sweep_results.json")
    parser.add_argument("--epochs_per_trial", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    use_amp = args.fp16 and (device.type == "cuda")

    processed_dir = Path(args.processed_dir)
    train_df = pd.read_parquet(processed_dir / "train_interactions.parquet")
    val_df = pd.read_parquet(processed_dir / "val_interactions.parquet")

    # Hyperparameter Grid
    betas = [0.05, 0.1, 0.2]
    sft_weights = [0.0, 0.05, 0.1, 0.2]
    strategies = ["binary_skip", "played_ratio", "margin_threshold"]

    # Representative grid subset
    trials = [
        # Impact of SFT regularization
        {"beta": 0.1, "sft_weight": 0.0, "strategy": "binary_skip"},
        {"beta": 0.1, "sft_weight": 0.05, "strategy": "binary_skip"},
        {"beta": 0.1, "sft_weight": 0.1, "strategy": "binary_skip"},
        {"beta": 0.1, "sft_weight": 0.2, "strategy": "binary_skip"},
        # Impact of beta (KL constraint)
        {"beta": 0.05, "sft_weight": 0.1, "strategy": "binary_skip"},
        {"beta": 0.2, "sft_weight": 0.1, "strategy": "binary_skip"},
        # Impact of Reward formulation
        {"beta": 0.1, "sft_weight": 0.1, "strategy": "played_ratio"},
        {"beta": 0.1, "sft_weight": 0.1, "strategy": "margin_threshold"},
    ]

    results: List[Dict[str, Any]] = []
    print("\n" + "=" * 70)
    print(f"RUNNING DPO HYPERPARAMETER SWEEP: {len(trials)} CONFIGURATIONS")
    print("=" * 70)

    for i, cfg in enumerate(trials, start=1):
        res = run_dpo_trial(
            trial_id=i,
            beta=cfg["beta"],
            sft_weight=cfg["sft_weight"],
            strategy=cfg["strategy"],
            train_df=train_df,
            val_df=val_df,
            base_checkpoint_path=Path(args.base_checkpoint),
            device=device,
            epochs=args.epochs_per_trial,
            batch_size=args.batch_size,
            use_amp=use_amp,
        )
        results.append(res)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    df_res = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("DPO HYPERPARAMETER SWEEP COMPLETE. SUMMARY TABLE:")
    print("=" * 70)
    print(df_res.to_string(index=False))
    print(f"\nSaved sweep results to: {out_path}")


if __name__ == "__main__":
    main()
