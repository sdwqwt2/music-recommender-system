"""
reward_model.py
---------------
Phase 2: Reinforcement Learning Value Head & Reward Model (Critic)
for Estimating Long-Term Listening Duration in Sequential Music Recommender Systems.

Architectural Design:
1. Backbone: Uses representations h_t from RecSysGPT2 (frozen or fine-tuned).
2. Value Head (Critic): Deep MLP regressor predicting continuous completion ratio R in [0, 1]
   (or listening duration in seconds).
3. Dual-Mode Evaluation:
   - Taken Action Loss: Regresses against true completion ratio of interacted tracks using Huber/MSE Loss.
   - Catalog-Wide Scoring: Fast linear/bilinear projection to estimate rewards across all 98K candidate tracks.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import ItemTokenizer
from dataset import RecBatch, create_dataloader
from model import RecSysGPT2, build_gpt2_recsys_model
from train_base import get_device, get_lr_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] (%(name)s) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("RewardModel")

# Prevent CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class RecSysRewardModel(nn.Module):
    """
    Reward Model (Critic) predicting expected listening duration / completion ratio.

    Given a user's chronological trajectory s_{1:t} and a candidate track a_t,
    the model predicts the expected completion ratio:
        hat{R}(s_t, a_t) in [0, 1]
    where:
        R approx 1.0 -> Complete, engaged listen (e.g. >= 180s)
        R approx 0.0 -> Instant skip (e.g. <= 15s)
    """

    def __init__(
        self,
        backbone: RecSysGPT2,
        freeze_backbone: bool = True,
        head_hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.n_embd = backbone.config.n_embd
        self.vocab_size = getattr(backbone, "vocab_size", getattr(backbone.config, "vocab_size", backbone.num_items + 1))
        self.pad_token_id = backbone.pad_token_id
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone GPT-2 weights frozen for Value Head training.")

        # Value Head: MLP regressing completion ratio
        self.value_head = nn.Sequential(
            nn.Linear(self.n_embd, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, head_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(head_hidden_dim // 2, 1),
            nn.Sigmoid(),  # Bound output to [0, 1] completion ratio
        )

        # Full-catalog projection head for fast offline candidate re-ranking
        self.catalog_reward_head = nn.Linear(self.n_embd, self.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        durations: Optional[torch.Tensor] = None,
        max_duration_sec: float = 180.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass predicting completion ratio.

        Args:
            input_ids: LongTensor [B, L]
            attention_mask: LongTensor [B, L]
            labels: Optional LongTensor [B, L] containing next-item tokens
            durations: Optional FloatTensor [B, L] containing actual listening seconds
            max_duration_sec: Benchmark duration representing 100% completion (default: 180s = 3 min)

        Returns:
            predicted_ratio: FloatTensor [B, L] in [0, 1]
            loss: Optional Huber / Smooth L1 regression loss
        """
        # Obtain contextual hidden representations from GPT-2
        if self.freeze_backbone:
            with torch.no_grad():
                backbone_outputs = self.backbone.transformer.transformer(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                hidden_states = backbone_outputs.last_hidden_state  # [B, L, n_embd]
        else:
            backbone_outputs = self.backbone.transformer.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            hidden_states = backbone_outputs.last_hidden_state  # [B, L, n_embd]

        # Predict completion ratio across sequence positions: [B, L, 1] -> [B, L]
        predicted_ratio = self.value_head(hidden_states).squeeze(-1)

        loss = None
        if durations is not None:
            # Target completion ratio normalized to [0, 1]
            target_ratio = torch.clamp(durations / max_duration_sec, min=0.0, max=1.0)

            # Mask out padding positions
            active_mask = (attention_mask == 1) & (input_ids != self.pad_token_id)
            if labels is not None:
                active_mask = active_mask & (labels != self.pad_token_id)

            if active_mask.any():
                pred_active = predicted_ratio[active_mask]
                target_active = target_ratio[active_mask]
                # Huber loss (Smooth L1) provides robust regression against extreme duration outliers
                loss = F.smooth_l1_loss(pred_active, target_active, beta=0.1)
            else:
                loss = predicted_ratio.sum() * 0.0

        return predicted_ratio, loss

    @torch.no_grad()
    def predict_candidate_rewards(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimates full-catalog completion scores for all candidate items at the final position.

        Returns:
            reward_scores: FloatTensor [B, vocab_size]
        """
        backbone_outputs = self.backbone.transformer.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = backbone_outputs.last_hidden_state
        # Final position token representation
        seq_lens = torch.clamp(attention_mask.sum(dim=-1) - 1, min=0)
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        last_hidden = hidden_states[batch_idx, seq_lens]  # [B, n_embd]

        # Project to catalog reward scores
        catalog_rewards = torch.sigmoid(self.catalog_reward_head(last_hidden))
        catalog_rewards[:, self.pad_token_id] = 0.0
        return catalog_rewards


def train_reward_epoch(
    model: RecSysRewardModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = False,
    epoch: int = 1,
) -> float:
    """Executes one training epoch for the Reward Model Value Head."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    progress = tqdm(dataloader, desc=f"Reward Epoch {epoch}", leave=False, mininterval=5.0)
    for step, batch in enumerate(progress, start=1):
        batch = batch.to(device)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            _, loss = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
                durations=batch.durations,
            )

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

        batch_loss = loss.item()
        total_loss += batch_loss
        num_batches += 1

        if step % 25 == 0 or step == len(dataloader):
            progress.set_postfix({"loss": f"{batch_loss:.4f}"})

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate_reward_model(
    model: RecSysRewardModel,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    max_duration_sec: float = 180.0,
) -> Dict[str, float]:
    """Evaluates MAE, RMSE, and Loss on validation user durations."""
    model.eval()
    total_loss = 0.0
    total_abs_error = 0.0
    total_sq_error = 0.0
    total_samples = 0
    num_batches = 0

    for batch in dataloader:
        batch = batch.to(device)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            pred_ratio, loss = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
                durations=batch.durations,
                max_duration_sec=max_duration_sec,
            )

        total_loss += loss.item()
        num_batches += 1

        # Evaluate on the final active position of each sequence
        seq_lens = torch.clamp(batch.attention_mask.sum(dim=-1) - 1, min=0)
        batch_idx = torch.arange(batch.labels.size(0), device=device)

        pred_last = pred_ratio[batch_idx, seq_lens] * max_duration_sec
        true_last = batch.durations[batch_idx, seq_lens]

        abs_err = torch.abs(pred_last - true_last)
        sq_err = torch.square(pred_last - true_last)

        total_abs_error += abs_err.sum().item()
        total_sq_error += sq_err.sum().item()
        total_samples += batch.labels.size(0)

    mae = total_abs_error / max(1, total_samples)
    rmse = math.sqrt(total_sq_error / max(1, total_samples))

    return {
        "val_loss": round(total_loss / max(1, num_batches), 4),
        "mae_seconds": round(mae, 2),
        "rmse_seconds": round(rmse, 2),
    }


def main() -> None:
    is_kaggle = os.path.exists("/kaggle") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    default_processed = "/kaggle/working/data/processed" if is_kaggle else "data/processed"
    default_ckpt = "/kaggle/working/checkpoints/best_gpt2_recsys.pt" if is_kaggle else "checkpoints/best_gpt2_recsys.pt"
    default_output = "/kaggle/working/checkpoints" if is_kaggle else "checkpoints"

    parser = argparse.ArgumentParser(description="Train Reward Model (Critic) for Listening Duration")
    parser.add_argument("--processed_dir", type=str, default=default_processed, help="Path to processed data splits")
    parser.add_argument("--base_checkpoint", type=str, default=default_ckpt, help="Path to Phase 1 Base GPT-2 checkpoint")
    parser.add_argument("--output_dir", type=str, default=default_output, help="Directory to save reward model checkpoint")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Value Head")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--unfreeze_backbone", action="store_true", help="Fine-tune GPT-2 backbone end-to-end")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision")
    args = parser.parse_args()

    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_amp = args.fp16 and (device.type == "cuda")

    # 1. Load Preprocessed Data
    processed_dir = Path(args.processed_dir)
    train_df = pd.read_parquet(processed_dir / "train_interactions.parquet")
    val_df = pd.read_parquet(processed_dir / "val_interactions.parquet")
    tokenizer = ItemTokenizer.load(processed_dir / "item_tokenizer.json")

    print(f"[Data] Loaded: Train interactions={len(train_df):,}, Val interactions={len(val_df):,}")

    # 2. Restore Phase 1 Base Model
    ckpt_path = Path(args.base_checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Base checkpoint not found at: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg = checkpoint["model_config"]
    base_model = build_gpt2_recsys_model(**cfg).to(device)
    base_model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[Model] Restored Base GPT-2 from {ckpt_path} (Val HR@10: {checkpoint.get('val_hr10')})")

    # 3. Construct Reward Model
    reward_model = RecSysRewardModel(
        backbone=base_model,
        freeze_backbone=not args.unfreeze_backbone,
        head_hidden_dim=256,
    ).to(device)

    # 4. DataLoaders
    train_loader = create_dataloader(
        df=train_df,
        max_length=cfg["max_length"],
        batch_size=args.batch_size,
        shuffle=True,
        sliding_window=True,
        user_col="uid",
        pin_memory=(device.type == "cuda"),
    )
    val_loader = create_dataloader(
        df=val_df,
        history_df=train_df,
        max_length=cfg["max_length"],
        batch_size=args.batch_size,
        shuffle=False,
        sliding_window=False,
        user_col="uid",
        pin_memory=(device.type == "cuda"),
    )

    # 5. Optimization
    trainable_params = [p for p in reward_model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_lr_scheduler(optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_mae = float("inf")
    best_path = output_dir / "best_reward_model.pt"

    print("\n" + "=" * 60)
    print(f"TRAINING PHASE 2 REWARD MODEL (CRITIC): {args.epochs} EPOCHS")
    print(f"Trainable Parameters: {sum(p.numel() for p in trainable_params):,}")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_reward_epoch(
            model=reward_model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            epoch=epoch,
        )
        val_eval = evaluate_reward_model(
            model=reward_model,
            dataloader=val_loader,
            device=device,
            use_amp=use_amp,
        )
        elapsed = time.time() - t0

        print(
            f"Reward Epoch {epoch:02d}/{args.epochs:02d} [{elapsed:.1f}s] | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_eval['val_loss']:.4f} | "
            f"MAE: {val_eval['mae_seconds']:.1f}s | RMSE: {val_eval['rmse_seconds']:.1f}s"
        )

        if val_eval["mae_seconds"] < best_mae:
            best_mae = val_eval["mae_seconds"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": reward_model.state_dict(),
                    "val_mae": best_mae,
                    "model_config": cfg,
                },
                best_path,
            )
            print(f"  --> Saved new best Reward Model checkpoint to: {best_path}")

    print("\n" + "=" * 60)
    print(f"Reward Model Training Complete! Best Val MAE: {best_mae:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
