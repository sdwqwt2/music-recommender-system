"""
sweep.py
--------
Automated, Resource-Efficient Hyperparameter Sweep for Causal GPT-2 Recommender.

Features:
1. One-Time Data Subsampling & Preprocessing: Cuts data to 20% users once, avoiding redundant preprocessing.
2. Fast Exploration: Runs N random configuration trials for 2-3 epochs each.
3. Leak-Proof Evaluation: Uses identical temporal split and memory-efficient top-k metrics.
4. Auto-Logging & Persistence: Saves tabular results to CSV and the best configuration to JSON.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW

from data_utils import ItemTokenizer, preprocess_pipeline
from dataset import create_dataloader
from model import RecSysGPT2, build_gpt2_recsys_model
from train_base import evaluate, get_device, get_lr_scheduler, train_one_epoch

# Prevent CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# Hyperparameter Search Space
SEARCH_SPACE = {
    "n_embd": [128, 256, 384],
    "n_layer": [3, 4, 6],
    "n_head": [4, 8],
    "lr": [1e-4, 3e-4, 5e-4, 1e-3],
    "warmup_ratio": [0.03, 0.05, 0.10],
    "weight_decay": [0.01, 0.05, 0.1],
    "batch_size": [32, 64],
    "gradient_accumulation_steps": [2, 4],
}


def sample_hyperparameters(rng: random.Random) -> Dict[str, Any]:
    """Samples a valid hyperparameter configuration guaranteeing n_embd % n_head == 0."""
    while True:
        n_embd = rng.choice(SEARCH_SPACE["n_embd"])
        n_head = rng.choice(SEARCH_SPACE["n_head"])
        if n_embd % n_head == 0:
            break

    batch_size = rng.choice(SEARCH_SPACE["batch_size"])
    grad_accum = rng.choice(SEARCH_SPACE["gradient_accumulation_steps"])

    return {
        "n_embd": n_embd,
        "n_layer": rng.choice(SEARCH_SPACE["n_layer"]),
        "n_head": n_head,
        "lr": rng.choice(SEARCH_SPACE["lr"]),
        "warmup_ratio": rng.choice(SEARCH_SPACE["warmup_ratio"]),
        "weight_decay": rng.choice(SEARCH_SPACE["weight_decay"]),
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
    }


def run_trial(
    trial_id: int,
    config: Dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer: ItemTokenizer,
    device: torch.device,
    use_amp: bool,
    epochs: int,
    max_length: int,
    user_col: str,
    num_workers: int,
) -> Dict[str, Any]:
    """Executes a single hyperparameter evaluation trial."""
    print(f"\n{'='*70}")
    print(f"--- Starting Trial {trial_id} ---")
    print(
        f"Config: n_embd={config['n_embd']}, n_layer={config['n_layer']}, "
        f"n_head={config['n_head']}, lr={config['lr']}, warmup={config['warmup_ratio']}, "
        f"wd={config['weight_decay']}, batch_size={config['batch_size']} (accum={config['gradient_accumulation_steps']})"
    )
    print(f"{'='*70}")

    pin_mem = device.type == "cuda"
    train_loader = create_dataloader(
        df=train_df,
        max_length=max_length,
        batch_size=config["batch_size"],
        shuffle=True,
        sliding_window=True,
        user_col=user_col,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )

    val_loader = create_dataloader(
        df=val_df,
        history_df=train_df,
        max_length=max_length,
        batch_size=config["batch_size"],
        shuffle=False,
        sliding_window=False,
        user_col=user_col,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )

    model = build_gpt2_recsys_model(
        num_items=tokenizer.num_items,
        max_length=max_length,
        n_embd=config["n_embd"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    effective_steps_per_epoch = max(
        1,
        len(train_loader) // config["gradient_accumulation_steps"]
        + (1 if len(train_loader) % config["gradient_accumulation_steps"] != 0 else 0),
    )
    total_steps = effective_steps_per_epoch * epochs
    warmup_steps = int(total_steps * config["warmup_ratio"])

    scheduler = get_lr_scheduler(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_hr = -1.0
    best_val_ndcg = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    trial_start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            epoch=epoch,
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
        )

        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            use_amp=use_amp,
            top_k=10,
        )
        ep_elapsed = time.time() - ep_t0

        val_loss = val_metrics["val_loss"]
        val_hr = val_metrics["val_hr@10"]
        val_ndcg = val_metrics["val_ndcg@10"]

        print(
            f"Trial {trial_id} | Epoch {epoch:02d}/{epochs:02d} [{ep_elapsed:.1f}s] | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"HR@10: {val_hr:.4f} | NDCG@10: {val_ndcg:.4f}"
        )

        if val_hr > best_val_hr:
            best_val_hr = val_hr
            best_val_ndcg = val_ndcg
            best_val_loss = val_loss
            best_epoch = epoch

    trial_duration = time.time() - trial_start_time

    # Cleanup GPU memory
    del model, optimizer, scheduler, scaler, train_loader, val_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    result = {
        "trial_id": trial_id,
        "best_epoch": best_epoch,
        "best_val_hr10": round(best_val_hr, 4),
        "best_val_ndcg10": round(best_val_ndcg, 4),
        "best_val_loss": round(best_val_loss, 4),
        "trial_duration_sec": round(trial_duration, 1),
        **config,
    }
    return result


def main() -> None:
    is_kaggle = os.path.exists("/kaggle") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    default_output = "/kaggle/working/sweep_results" if is_kaggle else "sweep_results"
    default_processed = "/kaggle/working/data/processed_sweep" if is_kaggle else "data/processed_sweep"

    parser = argparse.ArgumentParser(description="Hyperparameter Sweep for RecSys GPT-2")
    parser.add_argument("--data_path", type=str, default="data/synthetic_yambda.parquet", help="Path to raw interactions")
    parser.add_argument("--processed_dir", type=str, default=default_processed, help="Directory for sweep processed data")
    parser.add_argument("--output_dir", type=str, default=default_output, help="Directory to save sweep results")
    parser.add_argument("--num_trials", type=int, default=12, help="Number of random hyperparameter trials (default: 12)")
    parser.add_argument("--subsample_users", type=float, default=0.20, help="Fraction of users to sample (default: 0.20 = 20%%)")
    parser.add_argument("--epochs_per_trial", type=int, default=3, help="Training epochs per trial (default: 3)")
    parser.add_argument("--max_length", type=int, default=64, help="Sequence maximum length")
    parser.add_argument("--max_items", type=int, default=100000, help="Vocabulary cap (default: 100000)")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--user_col", type=str, default="uid", help="User column name")
    parser.add_argument("--item_col", type=str, default="item_id", help="Item column name")
    parser.add_argument("--timestamp_col", type=str, default="timestamp", help="Timestamp column name")
    parser.add_argument("--duration_col", type=str, default="listen_duration_sec", help="Duration column name")
    parser.add_argument("--num_workers", type=int, default=2 if is_kaggle else 0, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_amp = args.fp16 and device.type == "cuda"

    print("=" * 70)
    print("HYPERPARAMETER SWEEP FOR RECSYS GPT-2")
    print(f"Trials: {args.num_trials} | Epochs/Trial: {args.epochs_per_trial} | Subsample: {args.subsample_users * 100:.0f}% users")
    print(f"Device: {device} | FP16: {use_amp} | Output: {output_dir}")
    print("=" * 70)

    # 1. Preprocess data once with user subsampling
    processed_dir = Path(args.processed_dir)
    train_parquet = processed_dir / "train_interactions.parquet"
    val_parquet = processed_dir / "val_interactions.parquet"
    tokenizer_path = processed_dir / "item_tokenizer.json"

    if train_parquet.exists() and val_parquet.exists() and tokenizer_path.exists():
        print(f"[Data] Found existing preprocessed sweep data in {processed_dir}. Reusing.")
        train_df = pd.read_parquet(train_parquet)
        val_df = pd.read_parquet(val_parquet)
        tokenizer = ItemTokenizer.load(tokenizer_path)
    else:
        print(f"[Data] Preprocessing dataset with {args.subsample_users * 100:.0f}% user subsampling...")
        train_df, val_df, _, tokenizer, _ = preprocess_pipeline(
            data_path=args.data_path,
            output_dir=processed_dir,
            min_item_frequency=5,
            max_items=args.max_items,
            subsample_users=args.subsample_users,
            seed=args.seed,
            split_mode="global_temporal",
            user_col=args.user_col,
            item_col=args.item_col,
            timestamp_col=args.timestamp_col,
            duration_col=args.duration_col,
        )

    results_csv = output_dir / "sweep_results.csv"
    existing_trials = {}
    if results_csv.exists():
        try:
            prev_df = pd.read_csv(results_csv)
            for _, row in prev_df.iterrows():
                tid = int(row["trial_id"])
                existing_trials[tid] = row.to_dict()
            print(f"[Resume] Found existing sweep results with {len(existing_trials)} completed trials in: {results_csv}")
        except Exception as e:
            print(f"[Resume] Could not read existing results CSV ({e}). Starting fresh.")

    # 2. Run Sweep Trials
    rng = random.Random(args.seed)
    results: List[Dict[str, Any]] = []

    for trial_idx in range(1, args.num_trials + 1):
        config = sample_hyperparameters(rng)
        if trial_idx in existing_trials:
            print(f"\n[Resume] Trial {trial_idx} already finished (best HR@10: {existing_trials[trial_idx].get('best_val_hr10')}). Skipping.")
            results.append(existing_trials[trial_idx])
            continue

        trial_result = run_trial(
            trial_id=trial_idx,
            config=config,
            train_df=train_df,
            val_df=val_df,
            tokenizer=tokenizer,
            device=device,
            use_amp=use_amp,
            epochs=args.epochs_per_trial,
            max_length=args.max_length,
            user_col=args.user_col,
            num_workers=args.num_workers,
        )
        results.append(trial_result)

        # Incrementally save results CSV after every trial
        results_df = pd.DataFrame(results)
        results_df.sort_values(by="best_val_hr10", ascending=False, inplace=True)
        results_df.to_csv(results_csv, index=False)

    # 3. Final Summary & Best Config Persistence
    results_df = pd.DataFrame(results)
    results_df.sort_values(by="best_val_hr10", ascending=False, inplace=True)
    best_row = results_df.iloc[0].to_dict()

    print("\n" + "=" * 70)
    print("SWEEP COMPLETE! TOP 5 CONFIGURATIONS:")
    print("=" * 70)
    display_cols = ["trial_id", "best_val_hr10", "best_val_ndcg10", "best_val_loss", "n_embd", "n_layer", "n_head", "lr", "warmup_ratio", "weight_decay"]
    available_cols = [c for c in display_cols if c in results_df.columns]
    print(results_df[available_cols].head(5).to_string(index=False))

    best_config = {
        "n_embd": int(best_row["n_embd"]),
        "n_layer": int(best_row["n_layer"]),
        "n_head": int(best_row["n_head"]),
        "lr": float(best_row["lr"]),
        "warmup_ratio": float(best_row["warmup_ratio"]),
        "weight_decay": float(best_row["weight_decay"]),
        "batch_size": int(best_row["batch_size"]),
        "gradient_accumulation_steps": int(best_row["gradient_accumulation_steps"]),
        "max_length": args.max_length,
        "max_items": args.max_items,
        "best_val_hr10": float(best_row["best_val_hr10"]),
        "best_val_ndcg10": float(best_row["best_val_ndcg10"]),
    }

    best_config_path = output_dir / "best_config.json"
    with open(best_config_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)

    print(f"\n--> Best config saved to: {best_config_path}")
    print(f"--> Full sweep results saved to: {output_dir / 'sweep_results.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
