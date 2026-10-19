"""
train_dpo.py
------------
Direct Preference Optimization (DPO) Fine-Tuning for Causal GPT-2 RecSys.

Fine-tunes the Phase 1 Base Model directly from offline preference pairs:
- Winning Candidate (y_w): Satisfied, engaged listens (duration >= 30s).
- Losing Candidate (y_l): Skipped, dispreferred tracks (duration < 30s).

Mathematical Objective:
    L_{DPO}(\\theta; \\pi_{ref}) = - E_{(s, y_w, y_l)} [ log \\sigma ( \\beta log ( \\pi_\\theta(y_w|s) / \\pi_{ref}(y_w|s) )
                                                      - \\beta log ( \\pi_\\theta(y_l|s) / \\pi_{ref}(y_l|s) ) ) ]
                                + \\gamma * L_{SFT}(y_w|s)

Key Engineering Features:
1. Memory-Efficient Last-Token Projection: Uses last_token_only=True to evaluate only the active interaction state.
2. Strictly Frozen Reference Policy: pi_ref is in eval() mode with zero gradients.
3. Win-Rate & Margin Tracking: Logs empirical preference accuracy and implicit reward delta at every epoch.
4. Checkpointing: Saves the model with the best validation win-rate and DPO loss to best_dpo_gpt2.pt.
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import ItemTokenizer
from dataset import create_dataloader
from dpo_dataset import DPORecBatch, create_dpo_dataloader
from model import RecSysGPT2, build_gpt2_recsys_model
from train_base import evaluate, get_device, get_lr_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] (%(name)s) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TrainDPO")

# Prevent CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def compute_dpo_loss(
    policy_model: RecSysGPT2,
    ref_model: RecSysGPT2,
    batch: DPORecBatch,
    beta: float = 0.1,
    sft_weight: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes Direct Preference Optimization (DPO) loss for a batch of (context, y_w, y_l) pairs.

    Args:
        policy_model: Trainable RecSysGPT2 policy pi_theta.
        ref_model: Frozen reference RecSysGPT2 policy pi_ref.
        batch: DPORecBatch containing input_ids, attention_mask, winning_ids, losing_ids.
        beta: KL divergence regularization parameter (temperature).
        sft_weight: Weight for auxiliary supervised cross-entropy loss on winning tokens.

    Returns:
        total_loss: Scalar PyTorch tensor for backpropagation.
        metrics: Dictionary containing dpo_loss, win_rate, reward_margin, reward_w, reward_l.
    """
    pad_id = policy_model.pad_token_id
    B = batch.input_ids.size(0)
    batch_idx = torch.arange(B, device=batch.input_ids.device)

    # 1. Forward pass through Policy Model (last_token_only avoids full B x L x V memory)
    policy_output = policy_model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        last_token_only=True,
    )
    policy_logits = policy_output.logits.squeeze(1)  # [B, vocab_size]
    policy_logits[:, pad_id] = -1e9
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)

    # 2. Forward pass through Frozen Reference Model
    with torch.no_grad():
        ref_output = ref_model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            last_token_only=True,
        )
        ref_logits = ref_output.logits.squeeze(1)  # [B, vocab_size]
        ref_logits[:, pad_id] = -1e9
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    # 3. Extract log-probabilities for winning and losing tokens
    pi_w = policy_log_probs[batch_idx, batch.winning_ids]  # [B]
    pi_l = policy_log_probs[batch_idx, batch.losing_ids]   # [B]
    ref_w = ref_log_probs[batch_idx, batch.winning_ids]    # [B]
    ref_l = ref_log_probs[batch_idx, batch.losing_ids]     # [B]

    # 4. Implicit DPO Reward: r(s, y) = beta * (log pi_theta(y|s) - log pi_ref(y|s))
    log_ratio_w = pi_w - ref_w
    log_ratio_l = pi_l - ref_l
    reward_margin = beta * (log_ratio_w - log_ratio_l)     # [B]

    # 5. DPO Objective: -log sigma(beta * (log_ratio_w - log_ratio_l))
    dpo_loss = -F.logsigmoid(reward_margin).mean()

    # 6. Auxiliary SFT Regularization on winning items
    sft_loss = -pi_w.mean()
    total_loss = dpo_loss + sft_weight * sft_loss

    with torch.no_grad():
        win_rate = (reward_margin > 0.0).float().mean().item()
        margin_mean = reward_margin.mean().item()
        reward_w_mean = (beta * log_ratio_w).mean().item()
        reward_l_mean = (beta * log_ratio_l).mean().item()

    metrics = {
        "total_loss": total_loss.item(),
        "dpo_loss": dpo_loss.item(),
        "sft_loss": sft_loss.item(),
        "win_rate": win_rate,
        "reward_margin": margin_mean,
        "reward_w": reward_w_mean,
        "reward_l": reward_l_mean,
    }
    return total_loss, metrics


def train_dpo_epoch(
    policy_model: RecSysGPT2,
    ref_model: RecSysGPT2,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = False,
    beta: float = 0.1,
    sft_weight: float = 0.1,
    gradient_accumulation_steps: int = 1,
    epoch: int = 1,
) -> Dict[str, float]:
    """Runs a single training epoch of DPO policy alignment."""
    policy_model.train()
    ref_model.eval()

    total_loss = 0.0
    total_dpo_loss = 0.0
    total_win_rate = 0.0
    total_margin = 0.0
    num_batches = 0
    optimizer.zero_grad()

    progress = tqdm(dataloader, desc=f"DPO Epoch {epoch:02d}", leave=False, mininterval=5.0)
    for step, batch in enumerate(progress, start=1):
        batch = batch.to(device)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            loss, metrics = compute_dpo_loss(
                policy_model=policy_model,
                ref_model=ref_model,
                batch=batch,
                beta=beta,
                sft_weight=sft_weight,
            )
            loss_to_backprop = loss / gradient_accumulation_steps

        if use_amp and scaler is not None:
            scaler.scale(loss_to_backprop).backward()
            if step % gradient_accumulation_steps == 0 or step == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
        else:
            loss_to_backprop.backward()
            if step % gradient_accumulation_steps == 0 or step == len(dataloader):
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

        total_loss += metrics["total_loss"]
        total_dpo_loss += metrics["dpo_loss"]
        total_win_rate += metrics["win_rate"]
        total_margin += metrics["reward_margin"]
        num_batches += 1

        if step % 25 == 0 or step == len(dataloader):
            progress.set_postfix({
                "loss": f"{metrics['total_loss']:.4f}",
                "win%": f"{metrics['win_rate'] * 100:.1f}%",
                "margin": f"{metrics['reward_margin']:+.3f}",
            })

    return {
        "train_loss": total_loss / max(1, num_batches),
        "train_dpo_loss": total_dpo_loss / max(1, num_batches),
        "train_win_rate": (total_win_rate / max(1, num_batches)) * 100.0,
        "train_margin": total_margin / max(1, num_batches),
    }


@torch.no_grad()
def evaluate_dpo_validation(
    policy_model: RecSysGPT2,
    ref_model: RecSysGPT2,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    beta: float = 0.1,
    sft_weight: float = 0.1,
) -> Dict[str, float]:
    """Evaluates validation DPO loss, implicit reward margin, and preference win rate."""
    policy_model.eval()
    ref_model.eval()

    total_loss = 0.0
    total_dpo_loss = 0.0
    total_win_rate = 0.0
    total_margin = 0.0
    num_batches = 0

    for batch in dataloader:
        batch = batch.to(device)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            _, metrics = compute_dpo_loss(
                policy_model=policy_model,
                ref_model=ref_model,
                batch=batch,
                beta=beta,
                sft_weight=sft_weight,
            )

        total_loss += metrics["total_loss"]
        total_dpo_loss += metrics["dpo_loss"]
        total_win_rate += metrics["win_rate"]
        total_margin += metrics["reward_margin"]
        num_batches += 1

    return {
        "val_loss": round(total_loss / max(1, num_batches), 4),
        "val_dpo_loss": round(total_dpo_loss / max(1, num_batches), 4),
        "val_win_rate": round((total_win_rate / max(1, num_batches)) * 100.0, 2),
        "val_margin": round(total_margin / max(1, num_batches), 4),
    }


def main() -> None:
    is_kaggle = os.path.exists("/kaggle") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    default_processed = "/kaggle/working/data/processed" if is_kaggle else "data/processed"
    default_ckpt = "/kaggle/working/kaggle/working/checkpoints/best_gpt2_recsys.pt" if is_kaggle else "checkpoints/best_gpt2_recsys.pt"
    default_output = "/kaggle/working/checkpoints" if is_kaggle else "checkpoints"

    parser = argparse.ArgumentParser(description="Direct Preference Optimization (DPO) for RecSys GPT-2")
    parser.add_argument("--processed_dir", type=str, default=default_processed, help="Path to processed data splits")
    parser.add_argument("--base_checkpoint", type=str, default=default_ckpt, help="Path to Phase 1 Base GPT-2 checkpoint")
    parser.add_argument("--output_dir", type=str, default=default_output, help="Directory to save DPO checkpoints")
    parser.add_argument("--epochs", type=int, default=5, help="Number of DPO training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=5e-5, help="Fine-tuning learning rate (recommended: 3e-5 to 1e-4)")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature parameter controlling KL constraint")
    parser.add_argument("--sft_weight", type=float, default=0.1, help="Auxiliary SFT loss weight on winning items")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--skip_threshold_sec", type=float, default=30.0, help="Listening threshold separating skip from engaged")
    parser.add_argument("--max_pairs_per_user", type=int, default=2, help="Max preference pairs mined per user (default: 2 for high signal & fast training)")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--num_workers", type=int, default=2 if is_kaggle else 0, help="DataLoader workers")
    args = parser.parse_args()

    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_amp = args.fp16 and (device.type == "cuda")

    # 1. Resilient Processed Data Resolution
    processed_candidates = [
        Path(args.processed_dir),
        Path("/kaggle/working/downloaded_weights/data/processed"),
        Path("/kaggle/working/data/processed"),
        Path("data/processed"),
    ]
    processed_dir = None
    for p in processed_candidates:
        if (p / "train_interactions.parquet").exists():
            processed_dir = p
            break
    if processed_dir is None:
        raise FileNotFoundError(f"Could not locate processed directory with train_interactions.parquet in: {processed_candidates}")

    train_df = pd.read_parquet(processed_dir / "train_interactions.parquet")
    val_df = pd.read_parquet(processed_dir / "val_interactions.parquet")
    tokenizer = ItemTokenizer.load(processed_dir / "item_tokenizer.json")

    print(f"[Data] Loaded from '{processed_dir}': Train={len(train_df):,}, Val={len(val_df):,} | Vocab={tokenizer.num_items:,}")

    # 2. Resilient Base Checkpoint Resolution
    ckpt_candidates = [
        Path(args.base_checkpoint),
        Path("/kaggle/working/kaggle/working/checkpoints/best_gpt2_recsys.pt"),
        Path("/kaggle/working/checkpoints/best_gpt2_recsys.pt"),
        Path("checkpoints/best_gpt2_recsys.pt"),
    ]
    ckpt_path = None
    for c in ckpt_candidates:
        if c.exists():
            ckpt_path = c
            break
    if ckpt_path is None:
        raise FileNotFoundError(f"Base checkpoint not found in any expected location: {ckpt_candidates}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg = checkpoint["model_config"]

    # Initialize Trainable Policy Model pi_theta
    policy_model = build_gpt2_recsys_model(
        num_items=cfg["num_items"],
        max_length=cfg["max_length"],
        n_embd=cfg["n_embd"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
    ).to(device)
    policy_model.load_state_dict(checkpoint["model_state_dict"])

    # Initialize Frozen Reference Model pi_ref
    ref_model = build_gpt2_recsys_model(
        num_items=cfg["num_items"],
        max_length=cfg["max_length"],
        n_embd=cfg["n_embd"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
    ).to(device)
    ref_model.load_state_dict(checkpoint["model_state_dict"])
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    print(f"[Model] Restored Base Policy & Frozen Reference Model from: {ckpt_path}")
    print(f"        Base Model Val HR@10: {checkpoint.get('val_hr10')}")

    # 3. Create DPO Pairwise DataLoaders
    train_dpo_loader = create_dpo_dataloader(
        df=train_df,
        max_length=cfg["max_length"],
        batch_size=args.batch_size,
        shuffle=True,
        skip_threshold_sec=args.skip_threshold_sec,
        max_pairs_per_user=args.max_pairs_per_user,
        user_col="uid",
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_dpo_loader = create_dpo_dataloader(
        df=val_df,
        history_df=train_df,
        max_length=cfg["max_length"],
        batch_size=args.batch_size,
        shuffle=False,
        skip_threshold_sec=args.skip_threshold_sec,
        max_pairs_per_user=args.max_pairs_per_user,
        user_col="uid",
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Also build sequential ranking loader for validation HR@10 tracking
    val_seq_loader = create_dataloader(
        df=val_df,
        history_df=train_df,
        max_length=cfg["max_length"],
        batch_size=args.batch_size,
        shuffle=False,
        sliding_window=False,
        user_col="uid",
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 4. Optimizer and Schedule
    optimizer = AdamW(policy_model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(train_dpo_loader) // args.gradient_accumulation_steps) * args.epochs
    scheduler = get_lr_scheduler(optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_win_rate = 0.0
    best_path = output_dir / "best_dpo_gpt2.pt"

    print("\n" + "=" * 70)
    print(f"TRAINING PHASE 2 DIRECT PREFERENCE OPTIMIZATION (DPO): {args.epochs} EPOCHS")
    print(f"Policy Parameters: {sum(p.numel() for p in policy_model.parameters() if p.requires_grad):,}")
    print(f"DPO Beta (KL Temperature): {args.beta} | SFT Aux Weight: {args.sft_weight} | LR: {args.lr}")
    print(f"Mining Cutoff: Engaged >= {args.skip_threshold_sec}s vs Skip < {args.skip_threshold_sec}s")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats = train_dpo_epoch(
            policy_model=policy_model,
            ref_model=ref_model,
            dataloader=train_dpo_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            beta=args.beta,
            sft_weight=args.sft_weight,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            epoch=epoch,
        )

        val_dpo = evaluate_dpo_validation(
            policy_model=policy_model,
            ref_model=ref_model,
            dataloader=val_dpo_loader,
            device=device,
            use_amp=use_amp,
            beta=args.beta,
            sft_weight=args.sft_weight,
        )

        val_seq_eval = evaluate(
            model=policy_model,
            dataloader=val_seq_loader,
            device=device,
            use_amp=use_amp,
            top_k=10,
        )
        val_hr = val_seq_eval["val_hr@10"]
        val_ndcg = val_seq_eval["val_ndcg@10"]
        elapsed = time.time() - t0

        print(
            f"DPO Epoch {epoch:02d}/{args.epochs:02d} [{elapsed:.1f}s] | "
            f"Train Loss: {train_stats['train_loss']:.4f} (Win%: {train_stats['train_win_rate']:.1f}%, Margin: {train_stats['train_margin']:+.3f}) | "
            f"Val Loss: {val_dpo['val_loss']:.4f} | "
            f"Val Win%: {val_dpo['val_win_rate']:.1f}% | "
            f"Val Margin: {val_dpo['val_margin']:+.3f} | "
            f"Val HR@10: {val_hr:.4f} | "
            f"Val NDCG@10: {val_ndcg:.4f}"
        )

        # Checkpoint based on Validation Win% and HR@10 retention
        if val_dpo["val_win_rate"] > best_win_rate:
            best_win_rate = val_dpo["val_win_rate"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": policy_model.state_dict(),
                    "val_win_rate": val_dpo["val_win_rate"],
                    "val_dpo_loss": val_dpo["val_dpo_loss"],
                    "val_hr10": val_hr,
                    "val_ndcg10": val_ndcg,
                    "beta": args.beta,
                    "sft_weight": args.sft_weight,
                    "model_config": cfg,
                },
                best_path,
            )
            print(f"  --> Saved new best DPO Policy checkpoint to: {best_path}")

    print("\n" + "=" * 70)
    print(f"DPO Alignment Complete! Best Val Preference Win Rate: {best_win_rate:.2f}%")
    print(f"Checkpoint saved to: {best_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
