"""
reward_model.py
---------------
Phase 2: Reinforcement Learning State-Action Critic (Q-Network) & Value Model
for Predicting User Engagement, Skips, and Long-Term Listening Duration.

Architectural Design:
1. State-Action Conditioning Q(s_t, a_{t+1}):
   - Fuses sequential user state h_t from RecSysGPT2 with candidate track embedding e(a_{t+1}).
   - Unlike unconditional V(s) models which stall at the population mean, Q(s, a) directly
     conditions on candidate track properties (genre, duration, affinity).
2. Dual-Head Multi-Task Formulation:
   - Head 1 (Binary Skip Classifier): Predicts P(skip < 30s | s, a) with BCEWithLogitsLoss.
   - Head 2 (Engaged Duration Regressor): Predicts completion ratio R in [0, 1] for engaged listens (>= 30s).
   - Solves the bimodal distribution of music listening (instant skips vs full completions).
3. Ultra-Fast Online Scoring & Candidate Re-Ranking:
   - Top-K Candidate Re-Ranking: High-fidelity MLP scoring for top candidates proposed by Actor GPT-2.
   - Bilinear Catalog Projection: O(1) matrix multiplication across all 98K items via pre-trained wte weights.
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

from data_utils import ItemTokenizer, set_seed
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
    State-Action Critic Q(s_t, a_{t+1}) predicting user skip probability and listening duration.

    Given a user's chronological trajectory s_{1:t} and a candidate track a_{t+1},
    the model predicts:
        1. Skip Probability: P(duration < 30s | s_t, a_{t+1})
        2. Engaged Completion Ratio: R in [0, 1] for non-skipped tracks (normalized to max_duration_sec).

    Composite Reinforcement Learning Reward:
        R_{RL}(s_t, a) = (1 - P(skip)) * R_{completion} - lambda_skip * P(skip)
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

        # Fused State-Action interaction dimension: [h_t; e_a; h_t * e_a] -> 3 * n_embd
        in_dim = self.n_embd * 3

        # Interaction MLP modeling high-order user-track affinity
        self.interaction_mlp = nn.Sequential(
            nn.Linear(in_dim, head_hidden_dim),
            nn.LayerNorm(head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Head 1: Skip Classifier (duration < 30s)
        self.skip_head = nn.Sequential(
            nn.Linear(head_hidden_dim, head_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(head_hidden_dim // 2, 1),
        )

        # Head 2: Engaged Completion Ratio Regressor in [0, 1]
        self.duration_head = nn.Sequential(
            nn.Linear(head_hidden_dim, head_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(head_hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Fast Bilinear Catalog Scoring Projector: projects h_t to match wte item embedding space
        # Avoids 38M dense layer by sharing backbone item embeddings!
        self.catalog_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def _get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Retrieves item token representations from the backbone embedding table."""
        return self.backbone.transformer.transformer.wte(item_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        durations: Optional[torch.Tensor] = None,
        skip_threshold_sec: float = 30.0,
        max_duration_sec: float = 180.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, float]]:
        """
        Forward pass with State-Action conditioning.

        Args:
            input_ids: LongTensor [B, L] user history sequences.
            attention_mask: LongTensor [B, L] active sequence mask.
            labels: LongTensor [B, L] candidate / target item tokens a_{t+1}.
            durations: FloatTensor [B, L] ground-truth listen durations in seconds.
            skip_threshold_sec: Threshold below which a listen is treated as an instant skip.
            max_duration_sec: Normalization ceiling for 100% completion ratio.

        Returns:
            skip_logits: FloatTensor [B, L] raw logits for P(skip).
            completion_ratio: FloatTensor [B, L] in [0, 1] completion ratio.
            loss: Optional total multi-task loss.
            loss_dict: Dictionary containing component losses and skip classification accuracy.
        """
        # 1. Sequential hidden state from causal transformer
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

        # 2. Action conditioning on target item
        if labels is not None:
            item_embeds = self._get_item_embeddings(labels)  # [B, L, n_embd]
        else:
            # Fallback to zeros if no target item specified
            item_embeds = torch.zeros_like(hidden_states)

        # 3. Fuse User State and Candidate Item
        # Wide & Deep fusion: concatenation + element-wise interaction
        fused = torch.cat([hidden_states, item_embeds, hidden_states * item_embeds], dim=-1)
        h = self.interaction_mlp(fused)

        skip_logits = self.skip_head(h).squeeze(-1)         # [B, L]
        completion_ratio = self.duration_head(h).squeeze(-1)  # [B, L]

        # 4. Multi-Task Loss Calculation
        loss = None
        loss_dict: Dict[str, float] = {}

        if durations is not None:
            active_mask = (attention_mask == 1) & (input_ids != self.pad_token_id)
            if labels is not None:
                active_mask = active_mask & (labels != self.pad_token_id)

            if active_mask.any():
                dur_active = durations[active_mask]
                is_skip = (dur_active < skip_threshold_sec).float()
                active_skip_logits = skip_logits[active_mask]
                active_completion = completion_ratio[active_mask]

                # Component 1: Binary Skip Classification Loss (BCE)
                skip_loss = F.binary_cross_entropy_with_logits(active_skip_logits, is_skip)

                # Component 2: Engaged Duration Huber Regression Loss (for non-skips >= 30s)
                engaged_mask = (dur_active >= skip_threshold_sec)
                if engaged_mask.any():
                    target_ratio = torch.clamp(dur_active[engaged_mask] / max_duration_sec, min=0.0, max=1.0)
                    pred_ratio = active_completion[engaged_mask]
                    duration_loss = F.smooth_l1_loss(pred_ratio, target_ratio, beta=0.1)
                else:
                    duration_loss = skip_loss * 0.0

                # Multi-task objective: joint optimization of skip prevention and duration maximization
                loss = skip_loss + duration_loss

                with torch.no_grad():
                    pred_is_skip = (active_skip_logits > 0.0).float()
                    skip_acc = (pred_is_skip == is_skip).float().mean().item()
                    loss_dict = {
                        "total_loss": loss.item(),
                        "skip_loss": skip_loss.item(),
                        "duration_loss": duration_loss.item(),
                        "skip_acc": skip_acc,
                    }
            else:
                loss = (skip_logits.sum() + completion_ratio.sum()) * 0.0

        return skip_logits, completion_ratio, loss, loss_dict

    @torch.no_grad()
    def score_candidates(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        candidate_item_ids: torch.Tensor,
        lambda_skip: float = 0.5,
    ) -> torch.Tensor:
        """
        Calculates expected composite RL rewards for a matrix of candidate items [B, K].
        Used for top-K re-ranking during inference.

        Args:
            input_ids: LongTensor [B, L]
            attention_mask: LongTensor [B, L]
            candidate_item_ids: LongTensor [B, K] containing K candidate track tokens per user.
            lambda_skip: Penalty factor for predicted skips.

        Returns:
            reward_scores: FloatTensor [B, K] composite reward in [-lambda_skip, 1.0].
        """
        backbone_outputs = self.backbone.transformer.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = backbone_outputs.last_hidden_state
        seq_lens = torch.clamp(attention_mask.sum(dim=-1) - 1, min=0)
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        last_hidden = hidden_states[batch_idx, seq_lens]  # [B, n_embd]

        B, K = candidate_item_ids.shape
        cand_embeds = self._get_item_embeddings(candidate_item_ids)  # [B, K, n_embd]
        user_expanded = last_hidden.unsqueeze(1).expand(-1, K, -1)   # [B, K, n_embd]

        fused = torch.cat([user_expanded, cand_embeds, user_expanded * cand_embeds], dim=-1)
        h = self.interaction_mlp(fused)

        skip_prob = torch.sigmoid(self.skip_head(h).squeeze(-1))           # [B, K]
        completion_ratio = self.duration_head(h).squeeze(-1)               # [B, K]

        # Composite RL Reward: reward high completion while penalizing high skip risk
        reward_scores = (1.0 - skip_prob) * completion_ratio - (lambda_skip * skip_prob)
        return reward_scores

    @torch.no_grad()
    def compute_composite_reward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        action_ids: torch.Tensor,
        reward_type: str = "composite",
        lambda_skip: float = 0.5,
        lambda_repeat: float = 0.3,
        repetition_window: int = 4,
    ) -> torch.Tensor:
        """
        Computes scalar reinforcement learning reward for user history state s and action a.

        Args:
            input_ids: LongTensor [B, L] user interaction history.
            attention_mask: LongTensor [B, L] attention mask.
            action_ids: LongTensor [B] recommended action / candidate track token.
            reward_type: Formulation type ('binary', 'duration', 'composite').
            lambda_skip: Penalty weight for predicted skips.
            lambda_repeat: Penalty weight for repeating recently consumed tracks.
            repetition_window: Number of recent items to inspect for repeat play fatigue.

        Returns:
            rewards: FloatTensor [B] scalar reward per sequence.
        """
        if action_ids.dim() == 1:
            cand_2d = action_ids.unsqueeze(1)  # [B, 1]
        else:
            cand_2d = action_ids

        backbone_outputs = self.backbone.transformer.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = backbone_outputs.last_hidden_state
        seq_lens = torch.clamp(attention_mask.sum(dim=-1) - 1, min=0)
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        last_hidden = hidden_states[batch_idx, seq_lens]  # [B, n_embd]

        cand_embeds = self._get_item_embeddings(cand_2d)   # [B, 1, n_embd]
        user_expanded = last_hidden.unsqueeze(1)            # [B, 1, n_embd]

        fused = torch.cat([user_expanded, cand_embeds, user_expanded * cand_embeds], dim=-1)
        h = self.interaction_mlp(fused)

        skip_prob = torch.sigmoid(self.skip_head(h).squeeze(-1)).squeeze(1)       # [B]
        completion_ratio = self.duration_head(h).squeeze(-1).squeeze(1)           # [B]

        if reward_type == "binary":
            # Binary reward: +1 for predicted engaged (skip_prob < 0.5), -1 for skip
            return torch.where(skip_prob < 0.5, torch.ones_like(skip_prob), -torch.ones_like(skip_prob))

        elif reward_type == "duration":
            # Raw completion ratio in [0, 1]
            return completion_ratio

        else:
            # Multi-objective composite reward:
            # 1. Base engagement reward: (1 - P(skip)) * CompletionRatio
            # 2. Skip penalty: - lambda_skip * P(skip)
            # 3. Repeat-play penalty: - lambda_repeat if item occurred in recent window
            recent_context = input_ids[:, -repetition_window:]  # [B, W]
            is_repeat = (recent_context == action_ids.unsqueeze(1)).any(dim=-1).float()  # [B]

            composite = (1.0 - skip_prob) * completion_ratio - (lambda_skip * skip_prob) - (lambda_repeat * is_repeat)
            return composite

    @torch.no_grad()
    def predict_catalog_rewards(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fast catalog-wide linear scoring across all 98K tracks at the final sequence position.
        Uses bilinear projection against the pre-trained embedding table.

        Returns:
            catalog_scores: FloatTensor [B, vocab_size]
        """
        backbone_outputs = self.backbone.transformer.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = backbone_outputs.last_hidden_state
        seq_lens = torch.clamp(attention_mask.sum(dim=-1) - 1, min=0)
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        last_hidden = hidden_states[batch_idx, seq_lens]  # [B, n_embd]

        # Project user query into embedding space
        user_query = self.catalog_proj(last_hidden)  # [B, n_embd]
        all_item_weights = self.backbone.transformer.transformer.wte.weight  # [vocab_size, n_embd]

        catalog_scores = torch.matmul(user_query, all_item_weights.T) / math.sqrt(self.n_embd)
        catalog_scores[:, self.pad_token_id] = -1e9
        return torch.sigmoid(catalog_scores)


def train_reward_epoch(
    model: RecSysRewardModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = False,
    epoch: int = 1,
) -> Dict[str, float]:
    """Executes one training epoch for the State-Action Critic."""
    model.train()
    total_loss = 0.0
    total_skip_loss = 0.0
    total_dur_loss = 0.0
    total_skip_acc = 0.0
    num_batches = 0
    optimizer.zero_grad()

    progress = tqdm(dataloader, desc=f"Reward Epoch {epoch}", leave=False, mininterval=5.0)
    for step, batch in enumerate(progress, start=1):
        batch = batch.to(device)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            _, _, loss, loss_dict = model(
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
        total_skip_loss += loss_dict.get("skip_loss", 0.0)
        total_dur_loss += loss_dict.get("duration_loss", 0.0)
        total_skip_acc += loss_dict.get("skip_acc", 0.0)
        num_batches += 1

        if step % 25 == 0 or step == len(dataloader):
            progress.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "skip_acc": f"{loss_dict.get('skip_acc', 0.0):.1%}",
            })

    return {
        "train_loss": total_loss / max(1, num_batches),
        "train_skip_loss": total_skip_loss / max(1, num_batches),
        "train_dur_loss": total_dur_loss / max(1, num_batches),
        "train_skip_acc": total_skip_acc / max(1, num_batches),
    }


@torch.no_grad()
def evaluate_reward_model(
    model: RecSysRewardModel,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    skip_threshold_sec: float = 30.0,
    max_duration_sec: float = 180.0,
) -> Dict[str, float]:
    """Evaluates Skip Accuracy, Engaged Duration MAE, and Composite Loss on validation set."""
    model.eval()
    total_loss = 0.0
    total_engaged_abs_error = 0.0
    total_composite_abs_error = 0.0
    total_engaged_samples = 0
    total_samples = 0
    total_correct_skips = 0
    num_batches = 0

    for batch in dataloader:
        batch = batch.to(device)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            skip_logits, completion_ratio, loss, _ = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
                durations=batch.durations,
                skip_threshold_sec=skip_threshold_sec,
                max_duration_sec=max_duration_sec,
            )

        total_loss += loss.item()
        num_batches += 1

        # Evaluate on the final active position of each sequence
        seq_lens = torch.clamp(batch.attention_mask.sum(dim=-1) - 1, min=0)
        batch_idx = torch.arange(batch.labels.size(0), device=device)

        pred_skip_p = torch.sigmoid(skip_logits[batch_idx, seq_lens])
        pred_comp = completion_ratio[batch_idx, seq_lens]
        true_dur = batch.durations[batch_idx, seq_lens]

        # 1. Skip Classification Accuracy
        true_is_skip = (true_dur < skip_threshold_sec).float()
        pred_is_skip = (pred_skip_p >= 0.5).float()
        total_correct_skips += (pred_is_skip == true_is_skip).sum().item()

        # 2. Engaged MAE (strictly on tracks >= 30s)
        engaged_mask = (true_dur >= skip_threshold_sec)
        if engaged_mask.any():
            pred_engaged_dur = pred_comp[engaged_mask] * max_duration_sec
            abs_err = torch.abs(pred_engaged_dur - true_dur[engaged_mask])
            total_engaged_abs_error += abs_err.sum().item()
            total_engaged_samples += engaged_mask.sum().item()

        # 3. Overall Composite Expected Duration MAE
        # E[D] = (1 - P(skip)) * (Ratio * 180s) + P(skip) * 10s
        expected_duration = (1.0 - pred_skip_p) * (pred_comp * max_duration_sec) + (pred_skip_p * 10.0)
        total_composite_abs_error += torch.abs(expected_duration - true_dur).sum().item()
        total_samples += batch.labels.size(0)

    val_loss = total_loss / max(1, num_batches)
    skip_acc = total_correct_skips / max(1, total_samples)
    engaged_mae = total_engaged_abs_error / max(1, total_engaged_samples)
    overall_mae = total_composite_abs_error / max(1, total_samples)

    return {
        "val_loss": round(val_loss, 4),
        "skip_accuracy": round(skip_acc * 100.0, 2),
        "engaged_mae_sec": round(engaged_mae, 1),
        "overall_mae_sec": round(overall_mae, 1),
    }


def main() -> None:
    is_kaggle = os.path.exists("/kaggle") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    default_processed = "/kaggle/working/data/processed" if is_kaggle else "data/processed"
    default_ckpt = "/kaggle/working/checkpoints/best_gpt2_recsys.pt" if is_kaggle else "checkpoints/best_gpt2_recsys.pt"
    default_output = "/kaggle/working/checkpoints" if is_kaggle else "checkpoints"

    parser = argparse.ArgumentParser(description="Train State-Action Reward Model (Critic) for Listening Duration")
    parser.add_argument("--processed_dir", type=str, default=default_processed, help="Path to processed data splits")
    parser.add_argument("--base_checkpoint", type=str, default=default_ckpt, help="Path to Phase 1 Base GPT-2 checkpoint")
    parser.add_argument("--output_dir", type=str, default=default_output, help="Directory to save reward model checkpoint")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Critic Head")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--skip_threshold_sec", type=float, default=30.0, help="Skip cutoff threshold in seconds")
    parser.add_argument("--max_duration_sec", type=float, default=180.0, help="Benchmark duration for 100%% completion")
    parser.add_argument("--unfreeze_backbone", action="store_true", help="Fine-tune GPT-2 backbone end-to-end")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic reproducibility (default: 42)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_amp = args.fp16 and (device.type == "cuda")

    # 1. Resilient Processed Data Resolution
    processed_candidates = [
        Path(args.processed_dir),
        Path("/kaggle/working/Music-Recommender-System/downloaded_weights/data/processed"),
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
        raise FileNotFoundError(f"Could not locate processed directory in: {processed_candidates}")

    train_df = pd.read_parquet(processed_dir / "train_interactions.parquet")
    val_df = pd.read_parquet(processed_dir / "val_interactions.parquet")
    tokenizer = ItemTokenizer.load(processed_dir / "item_tokenizer.json")

    print(f"[Data] Loaded from '{processed_dir}': Train={len(train_df):,}, Val={len(val_df):,}")

    # 2. Resilient Base Checkpoint Resolution
    base_candidates = [
        Path(args.base_checkpoint),
        Path("/kaggle/working/kaggle/working/checkpoints/best_gpt2_recsys.pt"),
        Path("/kaggle/working/checkpoints/best_gpt2_recsys.pt"),
        Path("checkpoints/best_gpt2_recsys.pt"),
    ]
    ckpt_path = None
    for c in base_candidates:
        if c.exists():
            ckpt_path = c
            break
    if ckpt_path is None:
        raise FileNotFoundError(f"Base checkpoint not found in any expected location: {base_candidates}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg = checkpoint["model_config"]
    base_model = build_gpt2_recsys_model(**cfg).to(device)
    weights = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    base_model.load_state_dict(weights)
    print(f"[Model] Restored Base GPT-2 from {ckpt_path} (Val HR@10: {checkpoint.get('val_hr10')})")

    # 3. Construct State-Action Reward Model (Critic)
    reward_model = RecSysRewardModel(
        backbone=base_model,
        freeze_backbone=not args.unfreeze_backbone,
        head_hidden_dim=256,
        dropout=0.1,
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

    best_val_loss = float("inf")
    best_path = output_dir / "best_reward_model.pt"

    print("\n" + "=" * 65)
    print(f"TRAINING PHASE 2 STATE-ACTION CRITIC (REWARD MODEL): {args.epochs} EPOCHS")
    print(f"Trainable Parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"Skip Cutoff: < {args.skip_threshold_sec}s | Max Duration: {args.max_duration_sec}s")
    print("=" * 65)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats = train_reward_epoch(
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
            skip_threshold_sec=args.skip_threshold_sec,
            max_duration_sec=args.max_duration_sec,
        )
        elapsed = time.time() - t0

        print(
            f"Reward Epoch {epoch:02d}/{args.epochs:02d} [{elapsed:.1f}s] | "
            f"Train Loss: {train_stats['train_loss']:.4f} (Skip: {train_stats['train_skip_loss']:.3f}, Dur: {train_stats['train_dur_loss']:.3f}) | "
            f"Val Loss: {val_eval['val_loss']:.4f} | "
            f"Skip Acc: {val_eval['skip_accuracy']:.1f}% | "
            f"Engaged MAE: {val_eval['engaged_mae_sec']:.1f}s | "
            f"Composite MAE: {val_eval['overall_mae_sec']:.1f}s"
        )

        if val_eval["val_loss"] < best_val_loss:
            best_val_loss = val_eval["val_loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": reward_model.state_dict(),
                    "val_loss": best_val_loss,
                    "val_skip_accuracy": val_eval["skip_accuracy"],
                    "val_engaged_mae": val_eval["engaged_mae_sec"],
                    "val_composite_mae": val_eval["overall_mae_sec"],
                    "model_config": cfg,
                },
                best_path,
            )
            print(f"  --> Saved new best Reward Model checkpoint to: {best_path}")

    print("\n" + "=" * 65)
    print(f"Reward Model Training Complete! Best Val Loss: {best_val_loss:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
