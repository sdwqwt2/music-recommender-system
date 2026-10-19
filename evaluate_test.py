"""
evaluate_test.py
----------------
Full-Catalog Test-Set Evaluation and Qualitative Manual Inspection
for Causal GPT-2 Next-Item Recommender.

Features:
1. Strict Held-Out Evaluation: Evaluates on test_interactions.parquet with train+val historical context.
2. Multi-Cutoff Metrics: Computes HR@{5,10,20}, NDCG@{5,10,20}, MRR, Test Loss, and Perplexity over the full catalog.
3. Qualitative Manual Inspection: Prints user trajectories, ground-truth targets, and model Top-10 predictions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import ItemTokenizer
from dataset import SequentialRecDataset, create_dataloader
from model import RecSysGPT2, build_gpt2_recsys_model
from train_base import get_device

# Prevent CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


@torch.no_grad()
def evaluate_full_test_metrics(
    model: RecSysGPT2,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    cutoffs: List[int] = [5, 10, 20],
    min_engaged_sec: float = 30.0,
) -> Dict[str, Any]:
    """
    Computes Dual-Track ranking metrics over the full item catalog:
    1. Raw Benchmark: All held-out targets (standard baseline comparability).
    2. Engaged Benchmark: Only targets where user listened >= min_engaged_sec (satisfied listens).
    3. Duration-Weighted NDCG: Relevance scaled by listen duration completion.
    """
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    max_k = max(cutoffs)
    total_loss = 0.0
    num_batches = 0

    # Raw counters
    total_raw = 0
    raw_hits = {k: 0.0 for k in cutoffs}
    raw_ndcg = {k: 0.0 for k in cutoffs}
    raw_mrr = 0.0

    # Engaged counters (duration >= min_engaged_sec)
    total_engaged = 0
    engaged_hits = {k: 0.0 for k in cutoffs}
    engaged_ndcg = {k: 0.0 for k in cutoffs}
    engaged_mrr = 0.0

    # Duration-weighted NDCG
    d_ndcg = {k: 0.0 for k in cutoffs}

    pad_id = model.pad_token_id

    for batch in tqdm(dataloader, desc="Evaluating Dual-Track Test Set", leave=False, mininterval=5.0):
        batch = batch.to(device)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            output = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
                last_token_only=True,
            )

        total_loss += output.loss.item()
        num_batches += 1

        # Extract target items and durations at last valid position
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
        n_valid = targets.size(0)

        _, topk_indices = torch.topk(logits, k=max_k, dim=-1)

        for i in range(n_valid):
            target = targets[i].item()
            dur = max(0.0, durations[i].item())
            preds = topk_indices[i].tolist()
            is_engaged = (dur >= min_engaged_sec)

            # Relevance score normalized by 3-minute standard completion
            rel_score = min(1.0, dur / 180.0)

            total_raw += 1
            if is_engaged:
                total_engaged += 1

            if target in preds:
                rank = preds.index(target) + 1
                recip_rank = 1.0 / rank

                # Raw metrics
                raw_mrr += recip_rank
                for k in cutoffs:
                    if rank <= k:
                        raw_hits[k] += 1.0
                        raw_ndcg[k] += 1.0 / math.log2(rank + 1)
                        # Duration-weighted NDCG
                        d_ndcg[k] += rel_score / math.log2(rank + 1)

                # Engaged metrics
                if is_engaged:
                    engaged_mrr += recip_rank
                    for k in cutoffs:
                        if rank <= k:
                            engaged_hits[k] += 1.0
                            engaged_ndcg[k] += 1.0 / math.log2(rank + 1)

    avg_loss = total_loss / max(1, num_batches)
    ppl = math.exp(min(avg_loss, 20.0))

    results: Dict[str, Any] = {
        "test_loss": round(avg_loss, 4),
        "test_perplexity": round(ppl, 2),
        "total_raw_samples": total_raw,
        "total_engaged_samples": total_engaged,
        "engaged_percentage": round((total_engaged / max(1, total_raw)) * 100.0, 1),
        "raw_mrr": round(raw_mrr / max(1, total_raw), 4),
        "engaged_mrr": round(engaged_mrr / max(1, total_engaged), 4),
    }

    for k in cutoffs:
        results[f"raw_hr@{k}"] = round(raw_hits[k] / max(1, total_raw), 4)
        results[f"raw_ndcg@{k}"] = round(raw_ndcg[k] / max(1, total_raw), 4)
        results[f"engaged_hr@{k}"] = round(engaged_hits[k] / max(1, total_engaged), 4)
        results[f"engaged_ndcg@{k}"] = round(engaged_ndcg[k] / max(1, total_engaged), 4)
        results[f"duration_ndcg@{k}"] = round(d_ndcg[k] / max(1, total_raw), 4)

    return results


def run_qualitative_inspection(
    model: RecSysGPT2,
    test_df: pd.DataFrame,
    history_df: pd.DataFrame,
    tokenizer: ItemTokenizer,
    device: torch.device,
    user_col: str = "uid",
    item_col: str = "item_id",
    duration_col: str = "listen_duration_sec",
    num_samples: int = 10,
    top_k: int = 10,
) -> None:
    """Prints qualitative human-readable inspection cases from real user trajectories."""
    model.eval()
    print("\n" + "=" * 80)
    print("QUALITATIVE MANUAL INSPECTION: REAL USER TEST PREDICTIONS")
    print("=" * 80)

    # Pick users with active test targets
    test_users = test_df[user_col].unique()
    rng = np.random.RandomState(42)
    selected_users = rng.choice(test_users, size=min(num_samples, len(test_users)), replace=False)

    hit_count = 0

    for idx, uid in enumerate(selected_users, start=1):
        # 1. Gather historical sequence up to the test target
        user_history = history_df[history_df[user_col] == uid].sort_values("timestamp")
        user_test = test_df[test_df[user_col] == uid].sort_values("timestamp")

        if len(user_history) == 0 or len(user_test) == 0:
            continue

        hist_items = user_history[item_col].tolist()
        hist_tokens = user_history["item_token"].tolist()
        hist_durations = user_history[duration_col].tolist() if duration_col in user_history else [30.0] * len(hist_items)

        target_item = user_test[item_col].iloc[0]
        target_token = user_test["item_token"].iloc[0]
        target_duration = user_test[duration_col].iloc[0] if duration_col in user_test else 30.0

        # Model input: last up to 64 tokens
        input_tokens = hist_tokens[-64:]
        input_tensor = torch.tensor([input_tokens], dtype=torch.long, device=device)

        scores, topk_tokens = model.predict_next_items(input_tensor, top_k=top_k)
        pred_tokens = topk_tokens[0].cpu().tolist()
        pred_items = tokenizer.inverse_transform(pred_tokens)
        probs = torch.softmax(scores[0], dim=-1).cpu().tolist()

        is_hit = target_token in pred_tokens
        rank = (pred_tokens.index(target_token) + 1) if is_hit else None
        if is_hit:
            hit_count += 1

        print(f"\n[{idx:02d}/{num_samples}] User: {uid}")
        recent_preview = [f"{item} ({dur:.0f}s)" for item, dur in zip(hist_items[-4:], hist_durations[-4:])]
        print(f"  Recent History (last 4 tracks): {' -> '.join(recent_preview)}")
        print(f"  Ground-Truth Next Track:        {target_item} (listened: {target_duration:.1f}s)")

        hit_tag = f"HIT at Rank #{rank} (NDCG: {1.0 / math.log2(rank + 1):.4f})" if is_hit else "MISS (Not in Top-10)"
        print(f"  Model Result:                   {hit_tag}")
        print("  Top-5 Recommended Tracks:")
        for r, (p_item, p_prob) in enumerate(zip(pred_items[:5], probs[:5]), start=1):
            mark = " <--- [MATCH!]" if p_item == str(target_item) else ""
            print(f"     #{r:02d}: {p_item:<25} (Confidence: {p_prob * 100:.2f}%){mark}")

    print("\n" + "-" * 80)
    print(f"Manual Inspection Sample Hit Rate: {hit_count}/{num_samples} ({hit_count / max(1, num_samples) * 100:.1f}%)")
    print("-" * 80)


def main() -> None:
    is_kaggle = os.path.exists("/kaggle") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    default_processed = "/kaggle/working/data/processed" if is_kaggle else "data/processed"
    default_ckpt = "/kaggle/working/checkpoints/best_gpt2_recsys.pt" if is_kaggle else "checkpoints/best_gpt2_recsys.pt"

    parser = argparse.ArgumentParser(description="Evaluate RecSys GPT-2 on Held-Out Test Set")
    parser.add_argument("--processed_dir", type=str, default=default_processed, help="Directory containing processed splits")
    parser.add_argument("--checkpoint_path", type=str, default=default_ckpt, help="Path to best_gpt2_recsys.pt")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of qualitative users to inspect")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 evaluation")
    args = parser.parse_args()

    device = get_device()

    # Resilient Processed Directory Resolution
    processed_candidates = [
        Path(args.processed_dir),
        Path("/kaggle/working/downloaded_weights/data/processed"),
        Path("downloaded_weights/data/processed"),
        Path("/kaggle/working/data/processed"),
        Path("data/processed"),
    ]
    processed_dir = None
    for p in processed_candidates:
        if (p / "test_interactions.parquet").exists():
            processed_dir = p
            break
    if processed_dir is None:
        raise FileNotFoundError(f"Could not locate test_interactions.parquet in: {processed_candidates}")

    # Resilient Checkpoint Resolution
    ckpt_candidates = [
        Path(args.checkpoint_path),
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
        raise FileNotFoundError(f"Checkpoint not found in any expected location: {ckpt_candidates}")

    # 1. Load Preprocessed Data
    train_df = pd.read_parquet(processed_dir / "train_interactions.parquet")
    val_df = pd.read_parquet(processed_dir / "val_interactions.parquet")
    test_df = pd.read_parquet(processed_dir / "test_interactions.parquet")
    tokenizer = ItemTokenizer.load(processed_dir / "item_tokenizer.json")

    print(f"[Data] Loaded: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,} | Vocab={tokenizer.num_items:,}")

    # 2. Load Checkpoint and Restore Model
    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg = checkpoint["model_config"]
    model = build_gpt2_recsys_model(
        num_items=cfg["num_items"],
        max_length=cfg["max_length"],
        n_embd=cfg["n_embd"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"[Model] Restored checkpoint from Epoch {checkpoint.get('epoch', '?')} (Val HR@10: {checkpoint.get('val_hr10', '?')})")

    # 3. Create Full Historical Test DataLoader
    history_df = pd.concat([train_df, val_df], ignore_index=True)
    test_loader = create_dataloader(
        df=test_df,
        history_df=history_df,
        max_length=cfg["max_length"],
        batch_size=args.batch_size,
        shuffle=False,
        sliding_window=False,
        user_col="uid",
        num_workers=2 if is_kaggle else 0,
        pin_memory=(device.type == "cuda"),
    )

    # 4. Compute Test Metrics
    print("\n" + "=" * 60)
    print("RUNNING FINAL TEST SET EVALUATION OVER FULL CATALOG")
    print("=" * 60)
    use_amp = args.fp16 and (device.type == "cuda")
    metrics = evaluate_full_test_metrics(
        model=model,
        dataloader=test_loader,
        device=device,
        use_amp=use_amp,
        cutoffs=[5, 10, 20],
    )

    print("\n" + "=" * 75)
    print("DUAL-TRACK HELD-OUT TEST METRICS (FULL CATALOG RANKING):")
    print("=" * 75)
    print(f"Total Test Users: {metrics['total_raw_samples']:,} | Engaged Plays (>=30s): {metrics['total_engaged_samples']:,} ({metrics['engaged_percentage']}%)")
    print(f"Test Loss: {metrics['test_loss']:.4f} | Perplexity: {metrics['test_perplexity']:.2f}")
    print("-" * 75)
    print(f"{'Metric':<18} | {'Raw Benchmark (All)':<22} | {'Engaged Plays (>=30s)':<22} | {'Lift'}")
    print("-" * 75)
    for k in [5, 10, 20]:
        raw_h = metrics[f"raw_hr@{k}"]
        eng_h = metrics[f"engaged_hr@{k}"]
        lift_val = ((eng_h - raw_h) / max(0.0001, raw_h)) * 100.0
        lift_str = f"+{lift_val:.1f}%" if lift_val >= 0 else f"{lift_val:.1f}%"
        print(f"HitRate@{k:<10} | {raw_h:<22.4f} | {eng_h:<22.4f} | {lift_str}")
    print("-" * 75)
    for k in [5, 10, 20]:
        raw_n = metrics[f"raw_ndcg@{k}"]
        eng_n = metrics[f"engaged_ndcg@{k}"]
        lift_val = ((eng_n - raw_n) / max(0.0001, raw_n)) * 100.0
        lift_str = f"+{lift_val:.1f}%" if lift_val >= 0 else f"{lift_val:.1f}%"
        print(f"NDCG@{k:<13} | {raw_n:<22.4f} | {eng_n:<22.4f} | {lift_str}")
    print("-" * 75)
    print(f"MRR                | {metrics['raw_mrr']:<22.4f} | {metrics['engaged_mrr']:<22.4f} |")
    print(f"Duration-NDCG@10   | {metrics['duration_ndcg@10']:<22.4f} | (Relevance scaled by listen duration)")
    print("=" * 75)

    # Save test metrics
    metrics_path = ckpt_path.parent / "test_evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Export] Saved dual-track test metrics to: {metrics_path}")

    # 5. Qualitative Inspection
    run_qualitative_inspection(
        model=model,
        test_df=test_df,
        history_df=history_df,
        tokenizer=tokenizer,
        device=device,
        num_samples=args.num_samples,
        top_k=10,
    )


if __name__ == "__main__":
    main()
