"""
train_ppo.py
------------
Proximal Policy Optimization (PPO) & Policy Gradient Alignment for Causal RecSys GPT-2.

Demonstrates the classic Actor-Critic RL approach:
- Actor (Policy pi_theta): Causal GPT-2 initialized from Phase 1 base model.
- Frozen Reference Policy (pi_ref): Prevents excessive policy drift.
- Critic / Reward Model (Q-network): Pre-trained State-Action Critic from reward_model.py.

Mathematical Formulation:
    Action sampling: a ~ pi_theta(. | s)
    Reward: R(s, a) computed via RecSysRewardModel.compute_composite_reward()
    Advantage: A(s, a) = (R(s, a) - mean(R)) / (std(R) + 1e-8)
    PPO Objective:
        L_{PPO}(theta) = - E [ min( r_t(theta) * A_t, clip(r_t(theta), 1-eps, 1+eps) * A_t ) ]
                         + beta_kl * KL( pi_theta || pi_ref )
                         + gamma_sft * L_{SFT}(y_true | s)
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import ItemTokenizer, set_seed
from dataset import SequentialRecDataset, create_dataloader
from model import RecSysGPT2, build_gpt2_recsys_model
from reward_model import RecSysRewardModel
from train_base import evaluate, get_device, get_lr_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] (%(name)s) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TrainPPO")

# Prevent CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def sample_policy_actions(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k_candidates: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Samples actions from policy logits over the top-K candidate pool for numerical stability.

    Args:
        logits: FloatTensor [B, vocab_size] at active decision boundary.
        temperature: Sampling temperature.
        top_k_candidates: Pool size to restrict exploration (prevents OOD random tokens).

    Returns:
        actions: LongTensor [B] sampled item tokens.
        log_probs: FloatTensor [B] log probability of chosen actions under pi_theta.
    """
    scaled_logits = logits / max(temperature, 1e-4)

    # Restrict to top-K candidates to ensure stable exploration on large vocabularies
    top_logits, top_indices = torch.topk(scaled_logits, k=min(top_k_candidates, scaled_logits.size(-1)), dim=-1)
    probs = F.softmax(top_logits, dim=-1)

    dist = torch.distributions.Categorical(probs)
    sampled_idx = dist.sample()  # [B] in range [0, K-1]

    actions = top_indices.gather(1, sampled_idx.unsqueeze(1)).squeeze(1)  # [B]
    log_probs = dist.log_prob(sampled_idx)  # [B]

    return actions, log_probs


def train_ppo_epoch(
    actor_model: RecSysGPT2,
    ref_model: RecSysGPT2,
    reward_model: RecSysRewardModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
    use_amp: bool,
    clip_eps: float = 0.2,
    beta_kl: float = 0.05,
    sft_weight: float = 0.1,
    temperature: float = 1.0,
    top_k_candidates: int = 50,
    reward_type: str = "composite",
    gradient_accumulation_steps: int = 2,
    epoch: int = 1,
) -> Dict[str, float]:
    """Runs one training epoch of PPO alignment against the pre-trained Reward Model."""
    actor_model.train()
    ref_model.eval()
    reward_model.eval()

    total_loss = 0.0
    total_policy_loss = 0.0
    total_kl = 0.0
    total_reward = 0.0
    num_batches = 0

    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc=f"PPO Epoch {epoch:02d}", leave=False)

    for step, batch in enumerate(pbar):
        batch = batch.to(device)
        B = batch.input_ids.size(0)

        # 1. Forward pass through Actor to get current policy logits at decision step
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            actor_output = actor_model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                last_token_only=True,
            )
            actor_logits = actor_output.logits.squeeze(1)  # [B, V]
            actor_logits[:, actor_model.pad_token_id] = -1e9

            # 2. Sample actions from Actor
            actions, sampled_log_probs = sample_policy_actions(
                actor_logits,
                temperature=temperature,
                top_k_candidates=top_k_candidates,
            )

            # 3. Compute reference model log-probs for KL divergence penalty
            with torch.no_grad():
                ref_output = ref_model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    last_token_only=True,
                )
                ref_logits = ref_output.logits.squeeze(1)
                ref_logits[:, ref_model.pad_token_id] = -1e9
                ref_log_probs = F.log_softmax(ref_logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)

                # 4. Compute Reward from Critic Q(s, a)
                rewards = reward_model.compute_composite_reward(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    action_ids=actions,
                    reward_type=reward_type,
                )

                # Standardize advantages across batch
                adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            # 5. PPO Policy Loss
            # In online on-policy step, r_t = exp(log_prob - old_log_prob) = 1.0 initially
            log_prob_curr = F.log_softmax(actor_logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
            ratio = torch.exp(log_prob_curr - sampled_log_probs.detach())
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

            policy_loss = -torch.mean(torch.min(ratio * adv, clipped_ratio * adv))

            # 6. Approximate KL Divergence against reference policy
            kl_div = torch.mean(log_prob_curr - ref_log_probs)

            # 7. Auxiliary SFT loss on factual ground-truth target
            sft_loss = torch.tensor(0.0, device=device)
            if sft_weight > 0.0 and batch.labels is not None:
                # Target at final active sequence position
                seq_lens = torch.clamp(batch.attention_mask.sum(dim=-1) - 1, min=0)
                batch_idx = torch.arange(B, device=device)
                target_tokens = batch.labels[batch_idx, seq_lens]
                sft_loss = F.cross_entropy(actor_logits, target_tokens, ignore_index=actor_model.pad_token_id)

            loss = (policy_loss + beta_kl * kl_div + sft_weight * sft_loss) / gradient_accumulation_steps

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
            if scaler is not None and use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(actor_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(actor_model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * gradient_accumulation_steps
        total_policy_loss += policy_loss.item()
        total_kl += kl_div.item()
        total_reward += rewards.mean().item()
        num_batches += 1

        pbar.set_postfix({
            "Loss": f"{total_loss / num_batches:.4f}",
            "Rew": f"{total_reward / num_batches:.3f}",
            "KL": f"{total_kl / num_batches:.3f}",
        })

    return {
        "train_loss": total_loss / max(num_batches, 1),
        "policy_loss": total_policy_loss / max(num_batches, 1),
        "kl_div": total_kl / max(num_batches, 1),
        "mean_reward": total_reward / max(num_batches, 1),
    }


def main() -> None:
    # Resolve default paths
    is_kaggle = Path("/kaggle/working").exists()
    default_processed = "/kaggle/working/data/processed" if is_kaggle else "data/processed"
    default_base_ckpt = "checkpoints/best_gpt2_recsys.pt"
    default_reward_ckpt = "checkpoints/best_reward_model.pt"
    default_output = "/kaggle/working/checkpoints" if is_kaggle else "checkpoints"

    parser = argparse.ArgumentParser(description="PPO Actor-Critic Fine-Tuning for RecSys GPT-2")
    parser.add_argument("--processed_dir", type=str, default=default_processed, help="Path to processed dataset")
    parser.add_argument("--base_checkpoint", type=str, default=default_base_ckpt, help="Path to Base GPT-2 checkpoint")
    parser.add_argument("--reward_checkpoint", type=str, default=default_reward_ckpt, help="Path to Reward Model checkpoint")
    parser.add_argument("--output_dir", type=str, default=default_output, help="Directory to save PPO checkpoints")
    parser.add_argument("--epochs", type=int, default=3, help="Number of PPO training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate for Actor PPO updates")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clipping parameter epsilon")
    parser.add_argument("--beta_kl", type=float, default=0.05, help="KL divergence regularization weight")
    parser.add_argument("--sft_weight", type=float, default=0.1, help="Auxiliary SFT regularization weight")
    parser.add_argument("--temperature", type=float, default=1.0, help="Action sampling temperature")
    parser.add_argument("--top_k_candidates", type=int, default=50, help="Action candidate pool size")
    parser.add_argument("--reward_type", type=str, default="composite", choices=["binary", "duration", "composite"], help="Reward formulation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_amp = args.fp16 and (device.type == "cuda")

    # 1. Load Data
    processed_dir = Path(args.processed_dir)
    train_df = pd.read_parquet(processed_dir / "train_interactions.parquet")
    val_df = pd.read_parquet(processed_dir / "val_interactions.parquet")
    tokenizer = ItemTokenizer.load(processed_dir / "item_tokenizer.json")

    # 2. Load Base Model and Reference Model
    base_ckpt = torch.load(args.base_checkpoint, map_location=device)
    cfg = base_ckpt["model_config"]

    actor_model = build_gpt2_recsys_model(
        num_items=cfg["num_items"],
        max_length=cfg["max_length"],
        n_embd=cfg["n_embd"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
    ).to(device)
    actor_model.load_state_dict(base_ckpt["model_state_dict"])

    ref_model = build_gpt2_recsys_model(
        num_items=cfg["num_items"],
        max_length=cfg["max_length"],
        n_embd=cfg["n_embd"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
    ).to(device)
    ref_model.load_state_dict(base_ckpt["model_state_dict"])
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # 3. Load Reward Model (Critic)
    reward_ckpt = torch.load(args.reward_checkpoint, map_location=device)
    reward_backbone = build_gpt2_recsys_model(
        num_items=cfg["num_items"],
        max_length=cfg["max_length"],
        n_embd=cfg["n_embd"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
    ).to(device)
    reward_backbone.load_state_dict(reward_ckpt["backbone_state_dict"])

    reward_model = RecSysRewardModel(
        backbone=reward_backbone,
        freeze_backbone=True,
    ).to(device)
    reward_model.load_state_dict(reward_ckpt["model_state_dict"])
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad = False

    print("\n" + "=" * 70)
    print(f"TRAINING PHASE 2 ACTOR-CRITIC (PPO): {args.epochs} EPOCHS")
    print(f"Actor Parameters: {sum(p.numel() for p in actor_model.parameters() if p.requires_grad):,}")
    print(f"Reward Formulation: {args.reward_type} | KL Penalty: {args.beta_kl} | Clip: {args.clip_eps}")
    print("=" * 70)

    # 4. DataLoaders
    train_dataset = SequentialRecDataset(df=train_df, max_length=cfg["max_length"], is_eval=False)
    val_dataset = SequentialRecDataset(df=val_df, history_df=train_df, max_length=cfg["max_length"], is_eval=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)

    optimizer = AdamW(actor_model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_lr_scheduler(optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_ndcg = 0.0
    best_path = output_dir / "best_ppo_gpt2.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats = train_ppo_epoch(
            actor_model=actor_model,
            ref_model=ref_model,
            reward_model=reward_model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            clip_eps=args.clip_eps,
            beta_kl=args.beta_kl,
            sft_weight=args.sft_weight,
            temperature=args.temperature,
            top_k_candidates=args.top_k_candidates,
            reward_type=args.reward_type,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            epoch=epoch,
        )

        val_metrics = evaluate(actor_model, val_loader, device, use_amp=use_amp, top_k=10)
        elapsed = time.time() - t0

        print(
            f"PPO Epoch {epoch:02d}/{args.epochs:02d} [{elapsed:.1f}s] | "
            f"Train Loss: {train_stats['train_loss']:.4f} | "
            f"Policy Loss: {train_stats['policy_loss']:.4f} | "
            f"Mean Reward: {train_stats['mean_reward']:+.3f} | "
            f"KL: {train_stats['kl_div']:.3f} | "
            f"Val HR@10: {val_metrics['val_hr@10']:.4f} | "
            f"Val NDCG@10: {val_metrics['val_ndcg@10']:.4f}"
        )

        if val_metrics["val_ndcg@10"] > best_ndcg:
            best_ndcg = val_metrics["val_ndcg@10"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": actor_model.state_dict(),
                    "val_hr10": val_metrics["val_hr@10"],
                    "val_ndcg10": val_metrics["val_ndcg@10"],
                    "mean_reward": train_stats["mean_reward"],
                    "model_config": cfg,
                },
                best_path,
            )
            print(f"  --> Saved new best PPO checkpoint to: {best_path}")

    print("\n" + "=" * 70)
    print("PPO Actor-Critic Alignment Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
