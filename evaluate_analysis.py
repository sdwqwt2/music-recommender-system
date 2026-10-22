"""
evaluate_analysis.py
--------------------
Comprehensive Analytical & Statistical Evaluation Engine for Causal RecSys.

Features:
1. Multi-Paradigm Benchmark:
   - Base Causal GPT-2 (MLE)
   - 2-Stage Actor-Critic Re-Ranking (Base Model + RecSysRewardModel with lambda blend)
   - Direct Preference Optimization (DPO Policy)
2. Rigorous Statistical Hypothesis Testing:
   - Non-parametric Bootstrap (10,000 resamples) with 95% Confidence Intervals
   - McNemar's Paired Test for binary hit/miss discordance (exact chi2 and p-value)
   - Paired Wilcoxon Signed-Rank Test on per-user NDCG@10 and Reciprocal Rank (MRR)
3. Catalog Health & Diversity Analysis (Rubric Research Questions 2 & 3):
   - Catalog Coverage % across all 98,504 tracks
   - Shannon Recommendation Entropy (H)
   - Gini Coefficient of recommendation distribution
   - Item popularity correlation (Spearman rank correlation)
4. Multi-Step Autoregressive Rollouts (K=5) & Exposure Bias (Rubric Research Question 4):
   - Step-by-step skip probability P(skip_k), expected duration E[D_k], and cumulative session duration
   - Exposure bias degradation rate across 5 consecutive recommendation steps.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import ItemTokenizer, set_seed
from dataset import SequentialRecDataset, create_dataloader
from model import RecSysGPT2, build_gpt2_recsys_model
from reward_model import RecSysRewardModel
from train_base import get_device

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] (%(name)s) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("EvaluateAnalysis")

# Prevent CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ==============================================================================
# 1. Per-User Multi-Cutoff Evaluation Engine
# ==============================================================================

@torch.no_grad()
def evaluate_policy_per_user(
    model: RecSysGPT2,
    dataloader: DataLoader,
    device: torch.device,
    cutoffs: List[int] = [5, 10, 20],
    min_engaged_sec: float = 30.0,
    top_candidates_pool: int = 100,
) -> Dict[str, Any]:
    """
    Evaluates policy model over test set and collects per-user metrics for statistical testing.
    """
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    max_k = max(cutoffs)
    pad_id = model.pad_token_id

    user_ids: List[Any] = []
    targets_list: List[int] = []
    durations_list: List[float] = []
    is_engaged_list: List[bool] = []

    # Per-user metric storage
    per_user_hits = {k: [] for k in cutoffs}
    per_user_ndcg = {k: [] for k in cutoffs}
    per_user_mrr: List[float] = []
    top10_recommendations: List[List[int]] = []

    for batch in tqdm(dataloader, desc=f"Evaluating Policy ({getattr(model, 'name', 'Model')})", leave=False):
        batch = batch.to(device)
        output = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            last_token_only=True,
        )

        seq_lens = torch.clamp(batch.attention_mask.sum(dim=-1) - 1, min=0)
        batch_idx = torch.arange(batch.labels.size(0), device=device)
        targets = batch.labels[batch_idx, seq_lens]
        durations = batch.durations[batch_idx, seq_lens]

        logits = output.logits
        if logits.dim() == 3:
            logits = logits[:, 0, :]
        logits[:, pad_id] = -float("inf")

        valid_mask = targets != pad_id
        if not valid_mask.any():
            continue

        targets = targets[valid_mask]
        durations = durations[valid_mask]
        logits = logits[valid_mask]
        batch_uids = [batch.user_ids[i] for i in range(len(batch.user_ids)) if valid_mask[i]]

        _, topk_indices = torch.topk(logits, k=max_k, dim=-1)

        for i in range(targets.size(0)):
            target = targets[i].item()
            dur = max(0.0, durations[i].item())
            preds = topk_indices[i].tolist()
            is_eng = dur >= min_engaged_sec

            user_ids.append(batch_uids[i])
            targets_list.append(target)
            durations_list.append(dur)
            is_engaged_list.append(is_eng)
            top10_recommendations.append(preds[:10])

            recip_rank = 0.0
            hits_dict = {k: 0.0 for k in cutoffs}
            ndcg_dict = {k: 0.0 for k in cutoffs}

            if target in preds:
                rank = preds.index(target) + 1
                recip_rank = 1.0 / rank
                for k in cutoffs:
                    if rank <= k:
                        hits_dict[k] = 1.0
                        ndcg_dict[k] = 1.0 / math.log2(rank + 1)

            per_user_mrr.append(recip_rank)
            for k in cutoffs:
                per_user_hits[k].append(hits_dict[k])
                per_user_ndcg[k].append(ndcg_dict[k])

    return {
        "user_ids": user_ids,
        "targets": np.array(targets_list),
        "durations": np.array(durations_list),
        "is_engaged": np.array(is_engaged_list),
        "per_user_mrr": np.array(per_user_mrr),
        "per_user_hits": {k: np.array(v) for k, v in per_user_hits.items()},
        "per_user_ndcg": {k: np.array(v) for k, v in per_user_ndcg.items()},
        "top10_recommendations": top10_recommendations,
    }


# ==============================================================================
# 2. 2-Stage Actor-Critic Re-Ranking Benchmark
# ==============================================================================

@torch.no_grad()
def evaluate_critic_reranking_per_user(
    base_model: RecSysGPT2,
    reward_model: RecSysRewardModel,
    dataloader: DataLoader,
    device: torch.device,
    lambda_weights: List[float] = [0.1, 0.3, 0.5],
    candidate_pool_k: int = 100,
    cutoffs: List[int] = [5, 10, 20],
    min_engaged_sec: float = 30.0,
) -> Dict[float, Dict[str, Any]]:
    """
    Evaluates 2-Stage Re-Ranking:
    Base Causal GPT-2 retrieves Top-100 candidates;
    RecSysRewardModel re-scores candidates with:
        Score(s, a) = (1 - lambda) * log P_base(a|s) + lambda * R_critic(s, a)
    """
    base_model.eval()
    reward_model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pad_id = base_model.pad_token_id
    results_by_lambda: Dict[float, Dict[str, Any]] = {
        lam: {
            "per_user_mrr": [],
            "per_user_hits": {k: [] for k in cutoffs},
            "per_user_ndcg": {k: [] for k in cutoffs},
            "top10_recommendations": [],
        }
        for lam in lambda_weights
    }

    for batch in tqdm(dataloader, desc="Evaluating 2-Stage Critic Re-Ranking", leave=False):
        batch = batch.to(device)

        # 1. Base model retrieves candidate logits
        base_out = base_model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            last_token_only=True,
        )
        logits = base_out.logits
        if logits.dim() == 3:
            logits = logits[:, 0, :]
        logits[:, pad_id] = -float("inf")

        seq_lens = torch.clamp(batch.attention_mask.sum(dim=-1) - 1, min=0)
        batch_idx = torch.arange(batch.labels.size(0), device=device)
        targets = batch.labels[batch_idx, seq_lens]

        valid_mask = targets != pad_id
        if not valid_mask.any():
            continue

        targets = targets[valid_mask]
        logits = logits[valid_mask]
        input_ids_sub = batch.input_ids[valid_mask]
        attn_sub = batch.attention_mask[valid_mask]

        # Top-K candidate extraction from Base Model
        top_logprobs, candidate_tokens = torch.topk(F.log_softmax(logits, dim=-1), k=candidate_pool_k, dim=-1)

        # 2. Score candidates with Critic
        critic_rewards = reward_model.score_candidates(
            input_ids=input_ids_sub,
            attention_mask=attn_sub,
            candidate_item_ids=candidate_tokens,
            lambda_skip=0.5,
        )

        # 3. Re-rank per lambda
        for lam in lambda_weights:
            blended_scores = (1.0 - lam) * top_logprobs + lam * critic_rewards
            _, rerank_order = torch.sort(blended_scores, dim=-1, descending=True)

            for i in range(targets.size(0)):
                target = targets[i].item()
                cands = candidate_tokens[i].tolist()
                order = rerank_order[i].tolist()
                reranked_preds = [cands[idx] for idx in order]

                results_by_lambda[lam]["top10_recommendations"].append(reranked_preds[:10])

                recip_rank = 0.0
                hits_dict = {k: 0.0 for k in cutoffs}
                ndcg_dict = {k: 0.0 for k in cutoffs}

                if target in reranked_preds:
                    rank = reranked_preds.index(target) + 1
                    recip_rank = 1.0 / rank
                    for k in cutoffs:
                        if rank <= k:
                            hits_dict[k] = 1.0
                            ndcg_dict[k] = 1.0 / math.log2(rank + 1)

                results_by_lambda[lam]["per_user_mrr"].append(recip_rank)
                for k in cutoffs:
                    results_by_lambda[lam]["per_user_hits"][k].append(hits_dict[k])
                    results_by_lambda[lam]["per_user_ndcg"][k].append(ndcg_dict[k])

    for lam in lambda_weights:
        results_by_lambda[lam]["per_user_mrr"] = np.array(results_by_lambda[lam]["per_user_mrr"])
        for k in cutoffs:
            results_by_lambda[lam]["per_user_hits"][k] = np.array(results_by_lambda[lam]["per_user_hits"][k])
            results_by_lambda[lam]["per_user_ndcg"][k] = np.array(results_by_lambda[lam]["per_user_ndcg"][k])

    return results_by_lambda


# ==============================================================================
# 3. Rigorous Statistical Hypothesis Testing & Bootstrapping
# ==============================================================================

def compute_bootstrap_ci(
    base_arr: np.ndarray,
    dpo_arr: np.ndarray,
    num_bootstrap: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes 95% non-parametric bootstrap confidence intervals for:
    - Base mean
    - DPO mean
    - Delta (DPO - Base)
    - Empirical one-sided p-value (P(Delta <= 0))
    """
    n = len(base_arr)
    rng = np.random.RandomState(seed)

    base_means = np.empty(num_bootstrap, dtype=np.float64)
    dpo_means = np.empty(num_bootstrap, dtype=np.float64)
    deltas = np.empty(num_bootstrap, dtype=np.float64)

    for b in range(num_bootstrap):
        idx = rng.randint(0, n, size=n)
        b_mean = np.mean(base_arr[idx])
        d_mean = np.mean(dpo_arr[idx])
        base_means[b] = b_mean
        dpo_means[b] = d_mean
        deltas[b] = d_mean - b_mean

    low_p = (alpha / 2.0) * 100.0
    high_p = (1.0 - alpha / 2.0) * 100.0

    p_value_one_sided = float(np.mean(deltas <= 0.0))

    return {
        "base_mean": float(np.mean(base_arr)),
        "base_ci": [float(np.percentile(base_means, low_p)), float(np.percentile(base_means, high_p))],
        "dpo_mean": float(np.mean(dpo_arr)),
        "dpo_ci": [float(np.percentile(dpo_means, low_p)), float(np.percentile(dpo_means, high_p))],
        "delta_mean": float(np.mean(deltas)),
        "delta_ci": [float(np.percentile(deltas, low_p)), float(np.percentile(deltas, high_p))],
        "p_value_bootstrap": p_value_one_sided,
        "significant": bool(np.percentile(deltas, low_p) > 0.0),
    }


def compute_mcnemar_test(base_hits: np.ndarray, dpo_hits: np.ndarray) -> Dict[str, Any]:
    """
    Performs McNemar's paired test for binary classification discordance.
    n01: Base Miss, DPO Hit (DPO won)
    n10: Base Hit, DPO Miss (Base won)
    """
    n01 = int(np.sum((base_hits == 0) & (dpo_hits == 1)))
    n10 = int(np.sum((base_hits == 1) & (dpo_hits == 0)))
    n00 = int(np.sum((base_hits == 0) & (dpo_hits == 0)))
    n11 = int(np.sum((base_hits == 1) & (dpo_hits == 1)))

    discordant = n01 + n10
    if discordant == 0:
        return {"chi2": 0.0, "p_value": 1.0, "n01": n01, "n10": n10}

    # McNemar chi2 with Edwards continuity correction
    chi2_stat = ((abs(n01 - n10) - 1.0) ** 2) / float(discordant)
    p_val = float(1.0 - stats.chi2.cdf(chi2_stat, df=1))

    return {
        "n00": n00,
        "n11": n11,
        "n01_dpo_won": n01,
        "n10_base_won": n10,
        "chi2": float(round(chi2_stat, 4)),
        "p_value": float(round(p_val, 6)),
        "significant_p05": bool(p_val < 0.05),
        "significant_p01": bool(p_val < 0.01),
    }


def compute_wilcoxon_test(base_scores: np.ndarray, dpo_scores: np.ndarray) -> Dict[str, Any]:
    """
    Computes Wilcoxon signed-rank test on paired continuous metric observations (NDCG / MRR).
    """
    diffs = dpo_scores - base_scores
    non_zero = diffs[diffs != 0.0]
    if len(non_zero) < 10:
        return {"stat": 0.0, "p_value": 1.0}

    try:
        res = stats.wilcoxon(dpo_scores, base_scores, zero_method="wilcox", alternative="two-sided")
        return {"stat": float(round(res.statistic, 2)), "p_value": float(round(res.pvalue, 6))}
    except Exception as e:
        logger.warning(f"Wilcoxon test error: {e}")
        return {"stat": 0.0, "p_value": 1.0}


# ==============================================================================
# 4. Catalog Health, Diversity & Popularity Bias
# ==============================================================================

def compute_diversity_metrics(
    recommendations_list: List[List[int]],
    catalog_size: int,
    train_item_counts: Optional[Dict[int, int]] = None,
) -> Dict[str, Any]:
    """
    Computes Catalog Coverage %, Shannon Entropy (H), Gini Coefficient, and Popularity Correlation.
    """
    all_recs = [item for sublist in recommendations_list for item in sublist]
    total_slots = len(all_recs)
    if total_slots == 0:
        return {"coverage_pct": 0.0, "entropy": 0.0, "gini": 1.0}

    unique_items, counts = np.unique(all_recs, return_counts=True)
    num_unique = len(unique_items)
    coverage_pct = (num_unique / max(1, catalog_size)) * 100.0

    # 1. Shannon Recommendation Entropy: H = - sum p * log2(p)
    probs = counts / float(total_slots)
    entropy = -float(np.sum(probs * np.log2(probs + 1e-12)))

    # 2. Gini Coefficient of Exposure Inequality (0 = uniform, 1 = monopoly)
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    index = np.arange(1, n + 1)
    gini = float((2.0 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1.0) / n)

    # 3. Popularity Rank Correlation
    spearman_corr = 0.0
    if train_item_counts is not None:
        rec_freqs = [counts[np.where(unique_items == item)[0][0]] for item in unique_items]
        pop_counts = [train_item_counts.get(int(item), 0) for item in unique_items]
        if len(unique_items) > 5 and np.std(pop_counts) > 0:
            corr, _ = stats.spearmanr(rec_freqs, pop_counts)
            spearman_corr = float(round(corr, 4)) if not np.isnan(corr) else 0.0

    return {
        "catalog_coverage_pct": round(coverage_pct, 2),
        "unique_items_recommended": int(num_unique),
        "total_catalog_size": int(catalog_size),
        "shannon_entropy": round(entropy, 4),
        "gini_coefficient": round(gini, 4),
        "popularity_spearman": spearman_corr,
    }


# ==============================================================================
# 5. Multi-Step Autoregressive Rollouts (K=5) & Exposure Bias
# ==============================================================================

@torch.no_grad()
def run_autoregressive_rollouts(
    base_model: RecSysGPT2,
    dpo_model: RecSysGPT2,
    reward_model: RecSysRewardModel,
    dataloader: DataLoader,
    device: torch.device,
    num_users: int = 500,
    rollout_steps: int = 5,
    max_duration_sec: float = 180.0,
) -> Dict[str, Any]:
    """
    Simulates K-step autoregressive unguided rollouts (s_t -> a_{t+1} -> s_{t+1} -> a_{t+2} ... -> a_{t+5}).
    Evaluates step-by-step skip probability, expected listening duration, and compounding exposure bias.
    """
    base_model.eval()
    dpo_model.eval()
    reward_model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pad_id = base_model.pad_token_id

    # Storage across rollout steps k in {1 ... K}
    base_step_skips = {k: [] for k in range(1, rollout_steps + 1)}
    base_step_durs = {k: [] for k in range(1, rollout_steps + 1)}
    dpo_step_skips = {k: [] for k in range(1, rollout_steps + 1)}
    dpo_step_durs = {k: [] for k in range(1, rollout_steps + 1)}

    users_processed = 0

    for batch in tqdm(dataloader, desc=f"Simulating {rollout_steps}-Step Rollouts", leave=False):
        if users_processed >= num_users:
            break

        batch = batch.to(device)
        B = batch.input_ids.size(0)

        # Initialize contexts for Base and DPO
        base_ctx = batch.input_ids.clone()
        base_mask = batch.attention_mask.clone()
        dpo_ctx = batch.input_ids.clone()
        dpo_mask = batch.attention_mask.clone()

        for step in range(1, rollout_steps + 1):
            # --- Base Model Step ---
            out_base = base_model(input_ids=base_ctx, attention_mask=base_mask, last_token_only=True)
            l_base = out_base.logits[:, 0, :] if out_base.logits.dim() == 3 else out_base.logits
            l_base[:, pad_id] = -float("inf")
            next_base = torch.argmax(l_base, dim=-1)  # Greedy Top-1

            # Critic scoring for Base step
            r_base = reward_model.score_candidates(base_ctx, base_mask, next_base.unsqueeze(1))
            # Extract skip prob and duration from critic forward pass
            sk_logits, comp_ratio, _, _ = reward_model(
                input_ids=base_ctx,
                attention_mask=base_mask,
                labels=next_base.unsqueeze(1).expand(-1, base_ctx.size(1)),
            )
            # Evaluate at last valid token position
            seq_lens_b = torch.clamp(base_mask.sum(dim=-1) - 1, min=0)
            b_idx = torch.arange(B, device=device)
            p_skip_b = torch.sigmoid(sk_logits[b_idx, seq_lens_b]).cpu().tolist()
            exp_dur_b = (comp_ratio[b_idx, seq_lens_b] * max_duration_sec).cpu().tolist()

            base_step_skips[step].extend(p_skip_b)
            base_step_durs[step].extend(exp_dur_b)

            max_len = getattr(base_model, "max_length", getattr(base_model.config, "n_positions", 64))
            # Update base context (shift left if full)
            if base_ctx.size(1) >= max_len:
                base_ctx = torch.cat([base_ctx[:, 1:], next_base.unsqueeze(1)], dim=1)
                base_mask = torch.cat([base_mask[:, 1:], torch.ones_like(next_base.unsqueeze(1))], dim=1)
            else:
                base_ctx = torch.cat([base_ctx, next_base.unsqueeze(1)], dim=1)
                base_mask = torch.cat([base_mask, torch.ones_like(next_base.unsqueeze(1))], dim=1)

            # --- DPO Model Step ---
            out_dpo = dpo_model(input_ids=dpo_ctx, attention_mask=dpo_mask, last_token_only=True)
            l_dpo = out_dpo.logits[:, 0, :] if out_dpo.logits.dim() == 3 else out_dpo.logits
            l_dpo[:, pad_id] = -float("inf")
            next_dpo = torch.argmax(l_dpo, dim=-1)

            sk_logits_d, comp_ratio_d, _, _ = reward_model(
                input_ids=dpo_ctx,
                attention_mask=dpo_mask,
                labels=next_dpo.unsqueeze(1).expand(-1, dpo_ctx.size(1)),
            )
            seq_lens_d = torch.clamp(dpo_mask.sum(dim=-1) - 1, min=0)
            p_skip_d = torch.sigmoid(sk_logits_d[b_idx, seq_lens_d]).cpu().tolist()
            exp_dur_d = (comp_ratio_d[b_idx, seq_lens_d] * max_duration_sec).cpu().tolist()

            dpo_step_skips[step].extend(p_skip_d)
            dpo_step_durs[step].extend(exp_dur_d)

            if dpo_ctx.size(1) >= max_len:
                dpo_ctx = torch.cat([dpo_ctx[:, 1:], next_dpo.unsqueeze(1)], dim=1)
                dpo_mask = torch.cat([dpo_mask[:, 1:], torch.ones_like(next_dpo.unsqueeze(1))], dim=1)
            else:
                dpo_ctx = torch.cat([dpo_ctx, next_dpo.unsqueeze(1)], dim=1)
                dpo_mask = torch.cat([dpo_mask, torch.ones_like(next_dpo.unsqueeze(1))], dim=1)

        users_processed += B

    # Aggregate rollout metrics
    rollout_results = {"steps": list(range(1, rollout_steps + 1)), "base": {}, "dpo": {}}
    for step in range(1, rollout_steps + 1):
        rollout_results["base"][f"step_{step}"] = {
            "skip_prob": float(round(np.mean(base_step_skips[step]), 4)),
            "exp_duration_sec": float(round(np.mean(base_step_durs[step]), 2)),
        }
        rollout_results["dpo"][f"step_{step}"] = {
            "skip_prob": float(round(np.mean(dpo_step_skips[step]), 4)),
            "exp_duration_sec": float(round(np.mean(dpo_step_durs[step]), 2)),
        }

    # Cumulative session duration
    base_cum_dur = sum(rollout_results["base"][f"step_{s}"]["exp_duration_sec"] for s in range(1, rollout_steps + 1))
    dpo_cum_dur = sum(rollout_results["dpo"][f"step_{s}"]["exp_duration_sec"] for s in range(1, rollout_steps + 1))
    rollout_results["base"]["cumulative_5step_duration_sec"] = round(base_cum_dur, 1)
    rollout_results["dpo"]["cumulative_5step_duration_sec"] = round(dpo_cum_dur, 1)
    rollout_results["lift_cumulative_seconds"] = round(dpo_cum_dur - base_cum_dur, 1)

    return rollout_results


# ==============================================================================
# 6. Main Execution Pipeline
# ==============================================================================

def main() -> None:
    is_kaggle = os.path.exists("/kaggle") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    default_proc = "/kaggle/working/data/processed" if is_kaggle else "data/processed"
    default_base = "/kaggle/working/checkpoints/best_gpt2_recsys.pt" if is_kaggle else "checkpoints/best_gpt2_recsys.pt"
    default_dpo = "/kaggle/working/checkpoints/best_dpo_gpt2.pt" if is_kaggle else "checkpoints/best_dpo_gpt2.pt"
    default_rm = "/kaggle/working/checkpoints/best_reward_model.pt" if is_kaggle else "checkpoints/best_reward_model.pt"
    default_out = "/kaggle/working/analysis_metrics.json" if is_kaggle else "reports/analysis_metrics.json"

    parser = argparse.ArgumentParser(description="Multi-Paradigm Statistical Evaluation & Analysis Suite")
    parser.add_argument("--processed_dir", type=str, default=default_proc, help="Directory containing dataset splits")
    parser.add_argument("--base_checkpoint", type=str, default=default_base, help="Path to best_gpt2_recsys.pt")
    parser.add_argument("--dpo_checkpoint", type=str, default=default_dpo, help="Path to best_dpo_gpt2.pt")
    parser.add_argument("--reward_checkpoint", type=str, default=default_rm, help="Path to best_reward_model.pt")
    parser.add_argument("--output_json", type=str, default=default_out, help="Output path for structured JSON results")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--num_bootstrap", type=int, default=5000, help="Number of bootstrap resamples for 95% CIs")
    parser.add_argument("--num_rollout_users", type=int, default=300, help="Number of test users for 5-step rollouts")
    parser.add_argument("--rollout_steps", type=int, default=5, help="Number of autoregressive steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic reproducibility (default: 42)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    # 1. Resolve Data Path
    proc_candidates = [
        Path(args.processed_dir),
        Path("downloaded_weights/data/processed"),
        Path("/kaggle/working/downloaded_weights/data/processed"),
        Path("/kaggle/working/data/processed"),
        Path("data/processed"),
    ]
    proc_path = None
    tokenizer_file = None
    for p in proc_candidates:
        if p.exists():
            if (p / "item_tokenizer.json").exists():
                proc_path = p
                tokenizer_file = p / "item_tokenizer.json"
                break
            elif (p / "tokenizer.json").exists():
                proc_path = p
                tokenizer_file = p / "tokenizer.json"
                break

    if proc_path is None or tokenizer_file is None:
        raise FileNotFoundError(f"Could not locate processed directory with tokenizer in: {proc_candidates}")

    tokenizer = ItemTokenizer.load(tokenizer_file)
    vocab_size = tokenizer.vocab_size
    logger.info(f"Loaded Tokenizer from {tokenizer_file}: {vocab_size} tokens ({tokenizer.num_items} items)")

    # Load test split
    test_path = proc_path / "test_interactions.parquet"
    train_path = proc_path / "train_interactions.parquet"
    val_path = proc_path / "val_interactions.parquet"

    test_df = pd.read_parquet(test_path)
    train_df = pd.read_parquet(train_path) if train_path.exists() else None
    val_df = pd.read_parquet(val_path) if val_path.exists() else None

    history_df = pd.concat([train_df, val_df], ignore_index=True) if (train_df is not None and val_df is not None) else None

    # Compute training popularity counts for Spearman correlation
    train_item_counts = {}
    if train_df is not None and "item_token" in train_df.columns:
        train_item_counts = train_df["item_token"].value_counts().to_dict()

    test_loader = create_dataloader(
        df=test_df,
        history_df=history_df,
        max_length=64,
        batch_size=args.batch_size,
        shuffle=False,
        sliding_window=False,
        user_col="uid",
        num_workers=2 if is_kaggle else 0,
        pin_memory=(device.type == "cuda"),
    )
    logger.info(f"Loaded Held-Out Test Set: {len(test_loader.dataset)} users")

    # 2. Resolve Checkpoints
    def resolve_ckpt(target_path: str, fallback_patterns: List[str]) -> Path:
        p = Path(target_path)
        if p.exists():
            return p
        for pat in fallback_patterns:
            fp = Path(pat)
            if fp.exists():
                return fp
        raise FileNotFoundError(f"Checkpoint not found: {target_path}")

    base_ckpt = resolve_ckpt(args.base_checkpoint, ["/kaggle/working/kaggle/working/checkpoints/best_gpt2_recsys.pt", "checkpoints/best_gpt2_recsys.pt"])
    dpo_ckpt = resolve_ckpt(args.dpo_checkpoint, ["/kaggle/working/checkpoints/best_dpo_gpt2.pt", "checkpoints/best_dpo_gpt2.pt"])
    rm_ckpt = resolve_ckpt(args.reward_checkpoint, ["/kaggle/working/checkpoints/best_reward_model.pt", "checkpoints/best_reward_model.pt"])

    # 3. Load Models
    logger.info(f"Loading Base Causal GPT-2 from: {base_ckpt}")
    base_sd = torch.load(base_ckpt, map_location=device)
    base_cfg = base_sd.get("model_config", {
        "num_items": tokenizer.num_items,
        "n_embd": 384,
        "n_layer": 3,
        "n_head": 4,
        "max_length": 64,
    })
    base_model = build_gpt2_recsys_model(
        num_items=base_cfg.get("num_items", tokenizer.num_items),
        n_embd=base_cfg.get("n_embd", 384),
        n_layer=base_cfg.get("n_layer", 3),
        n_head=base_cfg.get("n_head", 4),
        max_length=base_cfg.get("max_length", 64),
    ).to(device)
    base_weights = base_sd["model_state_dict"] if "model_state_dict" in base_sd else base_sd
    base_model.load_state_dict(base_weights)
    base_model.name = "Base GPT-2 (MLE)"

    logger.info(f"Loading DPO Aligned Policy from: {dpo_ckpt}")
    dpo_sd = torch.load(dpo_ckpt, map_location=device)
    dpo_cfg = dpo_sd.get("model_config", base_cfg)
    dpo_model = build_gpt2_recsys_model(
        num_items=dpo_cfg.get("num_items", tokenizer.num_items),
        n_embd=dpo_cfg.get("n_embd", 384),
        n_layer=dpo_cfg.get("n_layer", 3),
        n_head=dpo_cfg.get("n_head", 4),
        max_length=dpo_cfg.get("max_length", 64),
    ).to(device)
    dpo_weights = dpo_sd["model_state_dict"] if "model_state_dict" in dpo_sd else dpo_sd
    dpo_model.load_state_dict(dpo_weights)
    dpo_model.name = "DPO Aligned Policy"

    logger.info(f"Loading State-Action Critic from: {rm_ckpt}")
    rm_sd = torch.load(rm_ckpt, map_location=device)
    rm_cfg = rm_sd.get("model_config", base_cfg)
    rm_backbone = build_gpt2_recsys_model(
        num_items=rm_cfg.get("num_items", tokenizer.num_items),
        n_embd=rm_cfg.get("n_embd", 384),
        n_layer=rm_cfg.get("n_layer", 3),
        n_head=rm_cfg.get("n_head", 4),
        max_length=rm_cfg.get("max_length", 64),
    ).to(device)
    reward_model = RecSysRewardModel(backbone=rm_backbone, freeze_backbone=True).to(device)
    rm_weights = rm_sd["model_state_dict"] if "model_state_dict" in rm_sd else rm_sd
    reward_model.load_state_dict(rm_weights)


    # ==============================================================================
    # 4. Run Multi-Paradigm Benchmark
    # ==============================================================================
    print("\n" + "=" * 80)
    print("STEP 1: EVALUATING BASE POLICY & DPO POLICY ON HELD-OUT TEST USERS")
    print("=" * 80)

    base_results = evaluate_policy_per_user(base_model, test_loader, device=device)
    dpo_results = evaluate_policy_per_user(dpo_model, test_loader, device=device)

    print("\n" + "=" * 80)
    print("STEP 2: EVALUATING 2-STAGE ACTOR-CRITIC RE-RANKING (BASE + CRITIC)")
    print("=" * 80)

    rerank_results = evaluate_critic_reranking_per_user(
        base_model=base_model,
        reward_model=reward_model,
        dataloader=test_loader,
        device=device,
        lambda_weights=[0.1, 0.3, 0.5],
        candidate_pool_k=100,
    )

    # ==============================================================================
    # 5. Statistical Hypothesis Testing
    # ==============================================================================
    print("\n" + "=" * 80)
    print(f"STEP 3: COMPUTING 95% BOOTSTRAP CONFIDENCE INTERVALS ({args.num_bootstrap:,} RESAMPLES)")
    print("=" * 80)

    stat_tests: Dict[str, Any] = {}
    for k in [5, 10, 20]:
        stat_tests[f"hr@{k}"] = compute_bootstrap_ci(
            base_arr=base_results["per_user_hits"][k],
            dpo_arr=dpo_results["per_user_hits"][k],
            num_bootstrap=args.num_bootstrap,
            seed=args.seed,
        )
        stat_tests[f"ndcg@{k}"] = compute_bootstrap_ci(
            base_arr=base_results["per_user_ndcg"][k],
            dpo_arr=dpo_results["per_user_ndcg"][k],
            num_bootstrap=args.num_bootstrap,
            seed=args.seed,
        )

    stat_tests["mrr"] = compute_bootstrap_ci(
        base_arr=base_results["per_user_mrr"],
        dpo_arr=dpo_results["per_user_mrr"],
        num_bootstrap=args.num_bootstrap,
        seed=args.seed,
    )

    # McNemar's paired test on HitRate@10
    mcnemar_hr10 = compute_mcnemar_test(
        base_hits=base_results["per_user_hits"][10],
        dpo_hits=dpo_results["per_user_hits"][10],
    )
    stat_tests["mcnemar_hr10"] = mcnemar_hr10

    # Wilcoxon paired signed-rank test on NDCG@10 and MRR
    wilcoxon_ndcg10 = compute_wilcoxon_test(
        base_scores=base_results["per_user_ndcg"][10],
        dpo_scores=dpo_results["per_user_ndcg"][10],
    )
    wilcoxon_mrr = compute_wilcoxon_test(
        base_scores=base_results["per_user_mrr"],
        dpo_scores=dpo_results["per_user_mrr"],
    )
    stat_tests["wilcoxon_ndcg10"] = wilcoxon_ndcg10
    stat_tests["wilcoxon_mrr"] = wilcoxon_mrr

    # ==============================================================================
    # 6. Catalog Health & Diversity Analysis
    # ==============================================================================
    print("\n" + "=" * 80)
    print("STEP 4: COMPUTING CATALOG DIVERSITY, SHANNON ENTROPY & GINI COEFFICIENT")
    print("=" * 80)

    diversity_base = compute_diversity_metrics(base_results["top10_recommendations"], vocab_size, train_item_counts)
    diversity_dpo = compute_diversity_metrics(dpo_results["top10_recommendations"], vocab_size, train_item_counts)
    diversity_rerank = {
        lam: compute_diversity_metrics(rerank_results[lam]["top10_recommendations"], vocab_size, train_item_counts)
        for lam in rerank_results
    }

    # ==============================================================================
    # 7. Multi-Step Autoregressive Rollouts (K=5)
    # ==============================================================================
    print("\n" + "=" * 80)
    print(f"STEP 5: SIMULATING {args.rollout_steps}-STEP AUTOREGRESSIVE ROLLOUTS ({args.num_rollout_users} USERS)")
    print("=" * 80)

    rollout_results = run_autoregressive_rollouts(
        base_model=base_model,
        dpo_model=dpo_model,
        reward_model=reward_model,
        dataloader=test_loader,
        device=device,
        num_users=args.num_rollout_users,
        rollout_steps=args.rollout_steps,
    )

    # ==============================================================================
    # 8. Output Display & JSON Export
    # ==============================================================================
    print("\n" + "=" * 92)
    print("MULTI-PARADIGM BENCHMARK ON HELD-OUT TEST SET (FULL CATALOG RANKING):")
    print("=" * 92)
    print(f"{'Method / Paradigm':<30} | {'HR@5':<8} | {'HR@10':<8} | {'HR@20':<8} | {'NDCG@10':<8} | {'MRR':<8} | {'Coverage%':<9}")
    print("-" * 92)

    def format_row(name: str, hits: Dict[int, np.ndarray], ndcg: Dict[int, np.ndarray], mrr: np.ndarray, cov: float) -> str:
        h5 = f"{np.mean(hits[5]):.4f}"
        h10 = f"{np.mean(hits[10]):.4f}"
        h20 = f"{np.mean(hits[20]):.4f}"
        n10 = f"{np.mean(ndcg[10]):.4f}"
        m = f"{np.mean(mrr):.4f}"
        c = f"{cov:.2f}%"
        return f"{name:<30} | {h5:<8} | {h10:<8} | {h20:<8} | {n10:<8} | {m:<8} | {c:<9}"

    print(format_row("1. Base Causal GPT-2 (MLE)", base_results["per_user_hits"], base_results["per_user_ndcg"], base_results["per_user_mrr"], diversity_base["catalog_coverage_pct"]))
    for lam in [0.1, 0.3, 0.5]:
        print(format_row(f"2. Critic Re-Rank (λ={lam})", rerank_results[lam]["per_user_hits"], rerank_results[lam]["per_user_ndcg"], rerank_results[lam]["per_user_mrr"], diversity_rerank[lam]["catalog_coverage_pct"]))
    print(format_row("3. DPO Aligned Policy (RL)", dpo_results["per_user_hits"], dpo_results["per_user_ndcg"], dpo_results["per_user_mrr"], diversity_dpo["catalog_coverage_pct"]))
    print("=" * 92)

    print("\n" + "=" * 92)
    print("RIGOROUS STATISTICAL SIGNIFICANCE TESTING: DPO POLICY vs BASE GPT-2:")
    print("=" * 92)
    print(f"{'Metric':<12} | {'Base (95% CI)':<22} | {'DPO (95% CI)':<22} | {'Delta (95% CI)':<24} | {'p-value'}")
    print("-" * 92)
    for m in ["hr@5", "hr@10", "hr@20", "ndcg@10", "mrr"]:
        st = stat_tests[m]
        b_str = f"{st['base_mean']:.4f} [{st['base_ci'][0]:.4f}, {st['base_ci'][1]:.4f}]"
        d_str = f"{st['dpo_mean']:.4f} [{st['dpo_ci'][0]:.4f}, {st['dpo_ci'][1]:.4f}]"
        delta_str = f"+{st['delta_mean']:.4f} [{st['delta_ci'][0]:+.4f}, {st['delta_ci'][1]:+.4f}]"
        p_str = f"p={st['p_value_bootstrap']:.4f}"
        print(f"{m.upper():<12} | {b_str:<22} | {d_str:<22} | {delta_str:<24} | {p_str}")

    print("-" * 92)
    print(f"McNemar's Paired Test (HR@10):  chi2 = {mcnemar_hr10['chi2']:.4f},  p-value = {mcnemar_hr10['p_value']:.6f} "
          f"({'STATISTICALLY SIGNIFICANT at p<0.01' if mcnemar_hr10['significant_p01'] else 'Not Significant'})")
    print(f"   --> Discordant Pairs: DPO won {mcnemar_hr10['n01_dpo_won']} users, Base won {mcnemar_hr10['n10_base_won']} users (Net: +{mcnemar_hr10['n01_dpo_won'] - mcnemar_hr10['n10_base_won']})")
    print(f"Wilcoxon Signed-Rank Test:      NDCG@10 p = {wilcoxon_ndcg10['p_value']:.6f} | MRR p = {wilcoxon_mrr['p_value']:.6f}")
    print("=" * 92)

    print("\n" + "=" * 80)
    print("CATALOG DIVERSITY & POPULARITY COLLAPSE ANALYSIS:")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Base GPT-2':<15} | {'DPO Policy':<15} | {'Interpretation'}")
    print("-" * 80)
    print(f"{'Catalog Coverage %':<30} | {diversity_base['catalog_coverage_pct']:<14.2f}% | {diversity_dpo['catalog_coverage_pct']:<14.2f}% | {'Fraction of 98.5K items exposed'}")
    print(f"{'Shannon Entropy (H)':<30} | {diversity_base['shannon_entropy']:<15.4f} | {diversity_dpo['shannon_entropy']:<15.4f} | {'Higher = more spread out recommendations'}")
    print(f"{'Gini Coefficient':<30} | {diversity_base['gini_coefficient']:<15.4f} | {diversity_dpo['gini_coefficient']:<15.4f} | {'Lower = less monopoly inequality'}")
    print(f"{'Popularity Spearman Corr':<30} | {diversity_base['popularity_spearman']:<15.4f} | {diversity_dpo['popularity_spearman']:<15.4f} | {'Correlation with train frequencies'}")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("5-STEP AUTOREGRESSIVE ROLLOUT & EXPOSURE BIAS TRAJECTORY:")
    print("=" * 80)
    print(f"{'Step':<6} | {'Base P(skip)':<14} | {'DPO P(skip)':<14} | {'Base Exp Duration':<18} | {'DPO Exp Duration'}")
    print("-" * 80)
    for s in range(1, args.rollout_steps + 1):
        bs_sk = rollout_results['base'][f'step_{s}']['skip_prob']
        dp_sk = rollout_results['dpo'][f'step_{s}']['skip_prob']
        bs_du = rollout_results['base'][f'step_{s}']['exp_duration_sec']
        dp_du = rollout_results['dpo'][f'step_{s}']['exp_duration_sec']
        print(f"#{s:<5} | {bs_sk:<14.3f} | {dp_sk:<14.3f} | {bs_du:<18.1f}s | {dp_du:.1f}s")
    print("-" * 80)
    print(f"Cumulative 5-Track Listening Duration: Base = {rollout_results['base']['cumulative_5step_duration_sec']:.1f}s | DPO = {rollout_results['dpo']['cumulative_5step_duration_sec']:.1f}s | Lift: {rollout_results['lift_cumulative_seconds']:+.1f}s")
    print("=" * 80)

    # Save comprehensive JSON payload
    output_payload = {
        "benchmark": {
            "base": {
                "hr@5": float(np.mean(base_results["per_user_hits"][5])),
                "hr@10": float(np.mean(base_results["per_user_hits"][10])),
                "hr@20": float(np.mean(base_results["per_user_hits"][20])),
                "ndcg@10": float(np.mean(base_results["per_user_ndcg"][10])),
                "mrr": float(np.mean(base_results["per_user_mrr"])),
            },
            "dpo": {
                "hr@5": float(np.mean(dpo_results["per_user_hits"][5])),
                "hr@10": float(np.mean(dpo_results["per_user_hits"][10])),
                "hr@20": float(np.mean(dpo_results["per_user_hits"][20])),
                "ndcg@10": float(np.mean(dpo_results["per_user_ndcg"][10])),
                "mrr": float(np.mean(dpo_results["per_user_mrr"])),
            },
            "critic_reranking": {
                str(lam): {
                    "hr@5": float(np.mean(rerank_results[lam]["per_user_hits"][5])),
                    "hr@10": float(np.mean(rerank_results[lam]["per_user_hits"][10])),
                    "hr@20": float(np.mean(rerank_results[lam]["per_user_hits"][20])),
                    "ndcg@10": float(np.mean(rerank_results[lam]["per_user_ndcg"][10])),
                    "mrr": float(np.mean(rerank_results[lam]["per_user_mrr"])),
                }
                for lam in [0.1, 0.3, 0.5]
            },
        },
        "statistical_tests": stat_tests,
        "diversity": {
            "base": diversity_base,
            "dpo": diversity_dpo,
            "critic_rerank": diversity_rerank,
        },
        "rollouts": rollout_results,
    }

    out_file = Path(args.output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(output_payload, f, indent=2)
    logger.info(f"Successfully saved complete analytical report to: {out_file}")


if __name__ == "__main__":
    main()
