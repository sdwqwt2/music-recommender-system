"""
train_base.py
-------------
Production-Grade PyTorch Training Pipeline for Causal GPT-2 Next Item Recommendation.

Features:
1. Multi-Device Detection: CUDA -> Apple Silicon MPS -> CPU.
2. CrossEntropyLoss with ignore_index=0 (<PAD> masked out).
3. RecSys Ranking Evaluation: Computes HitRate@10 (HR@10) and NDCG@10 on validation sequences.
4. Learning Rate Scheduling: Linear warmup with Cosine Annealing decay.
5. Gradient Clipping: Prevents exploding gradients during autoregressive sequence training.
6. Weights & Biases (WandB) Integration: Real-time experiment tracking with offline fallback.
7. Synthetic Data Generator: Instant end-to-end smoke testing on synthetic Yambda-style logs.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import ItemTokenizer, preprocess_pipeline
from dataset import SequentialRecDataset, create_dataloader
from model import RecSysGPT2, build_gpt2_recsys_model

# Optional WandB import with graceful fallback
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# Prevent CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def get_device() -> torch.device:
    """Detects and returns the best available compute device (CUDA -> MPS -> CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple Silicon (MPS)"
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    print(f"[Device] Using compute backend: {device} ({device_name})")
    return device


def compute_recsys_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    top_k: int = 10,
    pad_token_id: int = 0,
) -> Tuple[float, float, int]:
    """
    Computes HitRate@K and NDCG@K for next-item prediction on the final valid position of each sequence.
    Handles both full sequence logits [B, L, vocab_size] and memory-efficient last-token logits [B, 1, vocab_size].

    Args:
        logits: FloatTensor [B, L, vocab_size] or [B, 1, vocab_size]
        labels: LongTensor [B, L]
        attention_mask: LongTensor [B, L]
        top_k: Ranking cutoff (default: 10)
        pad_token_id: Token ID for padding (0)

    Returns:
        (hit_rate_sum, ndcg_sum, valid_samples_count)
    """
    # Identify the last valid label position for each sequence
    seq_lens = attention_mask.sum(dim=-1) - 1  # [B]
    seq_lens = torch.clamp(seq_lens, min=0)
    batch_indices = torch.arange(labels.size(0), device=labels.device)

    # Extract target items at the final position
    target_items = labels[batch_indices, seq_lens]  # [B]

    # Extract candidate logits at the final position
    if logits.dim() == 2:
        last_logits = logits.clone()
    elif logits.size(1) == 1:
        last_logits = logits[:, 0, :].clone()
    else:
        last_logits = logits[batch_indices, seq_lens].clone()

    # Exclude <PAD> from predictions
    last_logits[:, pad_token_id] = -float("inf")

    # Filter out samples where target item is PAD (invalid sequences)
    valid_mask = target_items != pad_token_id
    if not valid_mask.any():
        return 0.0, 0.0, 0

    target_items = target_items[valid_mask]
    last_logits = last_logits[valid_mask]

    # Compute top_k indices
    _, topk_indices = torch.topk(last_logits, k=top_k, dim=-1)  # [valid_B, top_k]

    hits = 0.0
    ndcg = 0.0
    num_valid = target_items.size(0)

    for i in range(num_valid):
        target = target_items[i].item()
        preds = topk_indices[i].tolist()
        if target in preds:
            hits += 1.0
            rank = preds.index(target) + 1  # 1-indexed
            ndcg += 1.0 / math.log2(rank + 1)

    return hits, ndcg, num_valid


def train_one_epoch(
    model: RecSysGPT2,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = False,
    gradient_accumulation_steps: int = 1,
    clip_grad_norm: float = 1.0,
    epoch: int = 1,
) -> float:
    """Executes one training epoch with micro-batching, gradient accumulation, and mixed precision."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
    for step, batch in enumerate(progress_bar, start=1):
        batch = batch.to(device)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            output = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
            )
            # Scale loss for gradient accumulation
            loss = output.loss / gradient_accumulation_steps

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Step optimizer and scheduler every gradient_accumulation_steps
        if step % gradient_accumulation_steps == 0 or step == len(dataloader):
            if use_amp and scaler is not None:
                if clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                optimizer.step()

            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        batch_loss = loss.item() * gradient_accumulation_steps
        total_loss += batch_loss
        num_batches += 1
        progress_bar.set_postfix({"loss": f"{batch_loss:.4f}"})

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate(
    model: RecSysGPT2,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    top_k: int = 10,
) -> Dict[str, float]:
    """Evaluates validation loss, Perplexity, HitRate@10, and NDCG@10 using memory-efficient projection."""
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    total_loss = 0.0
    total_hits = 0.0
    total_ndcg = 0.0
    total_eval_samples = 0
    num_batches = 0

    for batch in dataloader:
        batch = batch.to(device)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            # Crucial: last_token_only=True cuts evaluation memory from 10+ GB to <200 MB!
            output = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
                last_token_only=True,
            )

        total_loss += output.loss.item()
        num_batches += 1

        # Ranking metrics at last position
        hits, ndcg, count = compute_recsys_metrics(
            logits=output.logits,
            labels=batch.labels,
            attention_mask=batch.attention_mask,
            top_k=top_k,
            pad_token_id=model.pad_token_id,
        )
        total_hits += hits
        total_ndcg += ndcg
        total_eval_samples += count

    avg_loss = total_loss / max(1, num_batches)
    perplexity = math.exp(min(avg_loss, 20.0))  # clamp to avoid overflow
    hr = total_hits / max(1, total_eval_samples)
    ndcg = total_ndcg / max(1, total_eval_samples)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "val_loss": avg_loss,
        "perplexity": perplexity,
        f"val_hr@{top_k}": hr,
        f"val_ndcg@{top_k}": ndcg,
    }


def generate_synthetic_yambda_data(output_path: Path, num_interactions: int = 5000) -> Path:
    """Generates synthetic Yambda interaction log matching native schema for smoke-testing."""
    np.random.seed(42)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_users = 150
    num_items = 300

    uids = [f"user_{np.random.randint(1, num_users + 1):04d}" for _ in range(num_interactions)]
    # Use Zipf-like distribution to simulate music popularity long tail
    item_probs = 1.0 / np.arange(1, num_items + 1) ** 0.8
    item_probs /= item_probs.sum()
    items = np.random.choice([f"track_{i:04d}" for i in range(1, num_items + 1)], size=num_interactions, p=item_probs)

    base_time = 1700000000
    timestamps = base_time + np.sort(np.random.randint(0, 86400 * 30, size=num_interactions))
    track_lengths = np.random.uniform(120.0, 300.0, size=num_interactions)  # seconds
    played_ratio_pct = np.random.uniform(10.0, 100.0, size=num_interactions)  # percentage
    is_organic = np.random.choice([True, False], size=num_interactions, p=[0.7, 0.3])

    df = pd.DataFrame({
        "uid": uids,
        "item_id": items,
        "timestamp": timestamps,
        "played_ratio_pct": played_ratio_pct,
        "track_length_seconds": track_lengths,
        "is_organic": is_organic,
    })

    df.to_parquet(output_path, index=False)
    print(f"[SyntheticData] Generated {len(df)} interactions with native Yambda schema at: {output_path}")
    return output_path


def main() -> None:
    # Auto-detect Kaggle environment
    is_kaggle = os.path.exists("/kaggle") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    default_output = "/kaggle/working/checkpoints" if is_kaggle else "checkpoints"
    default_processed = "/kaggle/working/data/processed" if is_kaggle else "data/processed"

    parser = argparse.ArgumentParser(description="Train Base GPT-2 Recommender Model on Yambda")
    parser.add_argument("--data_path", type=str, default="data/synthetic_yambda.parquet", help="Path to interactions file")
    parser.add_argument("--processed_dir", type=str, default=default_processed, help="Directory for processed splits")
    parser.add_argument("--output_dir", type=str, default=default_output, help="Directory for model checkpoints")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic dataset for testing")
    parser.add_argument("--user_col", type=str, default="uid", help="User identifier column (default: uid)")
    parser.add_argument("--item_col", type=str, default="item_id", help="Item identifier column (default: item_id)")
    parser.add_argument("--timestamp_col", type=str, default="timestamp", help="Timestamp column (default: timestamp)")
    parser.add_argument("--duration_col", type=str, default="listen_duration_sec", help="Duration column (default: listen_duration_sec)")
    parser.add_argument("--min_item_freq", type=int, default=5, help="Prune items with fewer interactions")
    parser.add_argument("--max_length", type=int, default=64, help="Sequence maximum length")
    parser.add_argument("--n_embd", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--n_layer", type=int, default=3, help="Transformer layers")
    parser.add_argument("--n_head", type=int, default=4, help="Attention heads")
    parser.add_argument("--batch_size", type=int, default=32, help="Micro-batch size (default: 32)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of steps to accumulate gradients (default: 4, effective batch = batch_size * grad_accum)")
    parser.add_argument("--max_items", type=int, default=100000, help="Cap top-N items by frequency to bound vocabulary (default: 100000)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--split_mode", type=str, default="global_temporal", choices=["global_temporal", "user_temporal"])
    parser.add_argument("--fp16", action="store_true", help="Enable Mixed Precision training (NVIDIA CUDA)")
    parser.add_argument("--num_workers", type=int, default=2 if is_kaggle else 0, help="DataLoader subprocess workers")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="yambda-gpt2-recsys", help="WandB project name")
    args = parser.parse_args()

    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_kaggle:
        print(f"[Kaggle] Detected Kaggle execution environment. Output directory: {output_dir}")

    # Mixed precision setup
    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if args.fp16:
        if device.type == "cuda":
            print("[AMP] FP16 mixed precision enabled on NVIDIA GPU.")
        else:
            print(f"[AMP] Notice: --fp16 requested but current device is {device.type}. Running in full precision.")

    # 1. Prepare data
    data_file = Path(args.data_path)
    if args.synthetic or not data_file.exists():
        data_file = generate_synthetic_yambda_data(data_file)

    train_df, val_df, test_df, tokenizer, stats = preprocess_pipeline(
        data_path=data_file,
        output_dir=args.processed_dir,
        min_item_frequency=args.min_item_freq,
        max_items=args.max_items,
        split_mode=args.split_mode,
        user_col=args.user_col,
        item_col=args.item_col,
        timestamp_col=args.timestamp_col,
        duration_col=args.duration_col,
    )

    # 2. Build PyTorch DataLoaders
    pin_mem = device.type == "cuda"
    train_loader = create_dataloader(
        df=train_df,
        max_length=args.max_length,
        batch_size=args.batch_size,
        shuffle=True,
        sliding_window=True,
        user_col=args.user_col,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )

    val_loader = create_dataloader(
        df=val_df,
        history_df=train_df,
        max_length=args.max_length,
        batch_size=args.batch_size,
        shuffle=False,
        sliding_window=False,
        user_col=args.user_col,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )

    # 3. Initialize Base GPT-2 from scratch
    model = build_gpt2_recsys_model(
        num_items=tokenizer.num_items,
        max_length=args.max_length,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
    ).to(device)

    # 4. Optimization Setup
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    effective_steps_per_epoch = max(1, len(train_loader) // args.gradient_accumulation_steps + (1 if len(train_loader) % args.gradient_accumulation_steps != 0 else 0))
    total_steps = effective_steps_per_epoch * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=1e-5)

    # 5. WandB Logging Setup
    wandb_run = None
    if args.use_wandb and HAS_WANDB:
        try:
            wandb_run = wandb.init(
                project=args.wandb_project,
                config={
                    "vocab_size": tokenizer.vocab_size,
                    "num_items": tokenizer.num_items,
                    "max_length": args.max_length,
                    "n_embd": args.n_embd,
                    "n_layer": args.n_layer,
                    "n_head": args.n_head,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
                    "max_items": args.max_items,
                    "epochs": args.epochs,
                    "split_mode": args.split_mode,
                    "fp16": use_amp,
                },
            )
            print("[WandB] Successfully initialized Weights & Biases tracking.")
        except Exception as e:
            print(f"[WandB] Initialization failed ({e}). Proceeding with local console logging.")

    # 6. Training & Validation Loop
    best_hr = -1.0
    best_checkpoint_path = output_dir / "best_gpt2_recsys.pt"

    effective_batch = args.batch_size * args.gradient_accumulation_steps
    print("\n" + "=" * 60)
    print(f"Starting Training: {args.epochs} Epochs | Micro-Batch: {args.batch_size} (Effective: {effective_batch}) | Device: {device} | FP16: {use_amp}")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            epoch=epoch,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )

        val_metrics = evaluate(model=model, dataloader=val_loader, device=device, use_amp=use_amp, top_k=10)
        elapsed = time.time() - t0

        val_loss = val_metrics["val_loss"]
        val_hr = val_metrics["val_hr@10"]
        val_ndcg = val_metrics["val_ndcg@10"]
        val_ppl = val_metrics["perplexity"]

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} [{elapsed:.1f}s] | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Perplexity: {val_ppl:.2f} | HR@10: {val_hr:.4f} | NDCG@10: {val_ndcg:.4f}"
        )

        if wandb_run:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "perplexity": val_ppl,
                "val_hr@10": val_hr,
                "val_ndcg@10": val_ndcg,
                "learning_rate": optimizer.param_groups[0]["lr"],
            })

        # Save Best Model Checkpoint
        if val_hr > best_hr:
            best_hr = val_hr
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_hr10": val_hr,
                    "val_ndcg10": val_ndcg,
                    "val_loss": val_loss,
                    "model_config": {
                        "num_items": tokenizer.num_items,
                        "max_length": args.max_length,
                        "n_embd": args.n_embd,
                        "n_layer": args.n_layer,
                        "n_head": args.n_head,
                    },
                },
                best_checkpoint_path,
            )
            print(f"  --> Saved new best checkpoint (HR@10: {val_hr:.4f}) to: {best_checkpoint_path}")

    print("\n" + "=" * 60)
    print(f"Training Complete! Best HR@10: {best_hr:.4f}")
    print(f"Model Checkpoint saved at: {best_checkpoint_path}")
    print("=" * 60)

    if wandb_run:
        wandb.finish()


if __name__ == "__main__":
    main()
