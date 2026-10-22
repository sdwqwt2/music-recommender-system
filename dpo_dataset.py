"""
dpo_dataset.py
--------------
Dataset and Pairwise Mining for Direct Preference Optimization (DPO)
in Sequential Music Recommendation.

Pairs user history trajectories (prompts/contexts) with:
- Winning Candidate (y_w): Engaged interaction (listen_duration >= 30s or high played_ratio).
- Losing Candidate (y_l): Dispreferred / skipped interaction (listen_duration < 30s or mined session skip).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] (%(name)s) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DPODataset")


@dataclass
class DPORecBatch:
    """
    Batched Tensors for Direct Preference Optimization (DPO).

    Attributes:
        input_ids: LongTensor [B, L] user history prefix tokens.
        attention_mask: LongTensor [B, L] 1 for active prefix items, 0 for <PAD>.
        winning_ids: LongTensor [B] target item tokens for satisfied / engaged listens.
        losing_ids: LongTensor [B] target item tokens for skips / dispreferred listens.
        winning_durations: FloatTensor [B] listening duration of winning action.
        losing_durations: FloatTensor [B] listening duration of losing action.
        user_ids: List of raw user identifiers.
    """
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    winning_ids: torch.Tensor
    losing_ids: torch.Tensor
    winning_durations: torch.Tensor
    losing_durations: torch.Tensor
    user_ids: List[Union[str, int]]

    def to(self, device: Union[torch.device, str]) -> DPORecBatch:
        return DPORecBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            winning_ids=self.winning_ids.to(device),
            losing_ids=self.losing_ids.to(device),
            winning_durations=self.winning_durations.to(device),
            losing_durations=self.losing_durations.to(device),
            user_ids=self.user_ids,
        )


class DPORecDataset(Dataset):
    """
    PyTorch Dataset generating (context, winning_item, losing_item) triplets
    for Direct Preference Optimization.

    Mining Strategy:
    1. Chronological User Trajectory:
       For each user interaction history sorted by timestamp:
       - Context s_t: prefix of items up to step t.
       - Target item at step t+1:
         - Case A (Observed Engaged Listen): If item_{t+1} was engaged (>= skip_threshold_sec):
           winning_item = item_{t+1}.
           losing_item is mined from:
             (1) A track that the same user skipped elsewhere in their session/history.
             (2) If no skip exists in user history, a track with a high global skip rate,
                 or a random unengaged item from the training catalog.
         - Case B (Observed Immediate Skip): If item_{t+1} was an instant skip (< skip_threshold_sec):
           losing_item = item_{t+1}.
           winning_item is the subsequent engaged track that the user accepted next in the session!
    """

    def __init__(
        self,
        df: pd.DataFrame,
        history_df: Optional[pd.DataFrame] = None,
        max_length: int = 64,
        skip_threshold_sec: float = 30.0,
        preference_strategy: str = "binary_skip",
        max_pairs_per_user: int = 2,
        user_col: str = "uid",
        item_token_col: str = "item_token",
        timestamp_col: str = "timestamp",
        duration_col: str = "listen_duration_sec",
        pad_token_id: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.skip_threshold_sec = skip_threshold_sec
        self.preference_strategy = preference_strategy
        self.max_pairs_per_user = max_pairs_per_user
        self.pad_token_id = pad_token_id
        self.rng = np.random.RandomState(seed)

        # Harmonize user column alias
        if user_col not in df.columns:
            if "uid" in df.columns:
                user_col = "uid"
            elif "user_id" in df.columns:
                user_col = "user_id"

        # Combine evaluation split with prior history context if supplied
        if history_df is not None:
            active_users = set(df[user_col].unique())
            filtered_history = history_df[history_df[user_col].isin(active_users)]
            combined_df = pd.concat([filtered_history, df], ignore_index=True)
        else:
            combined_df = df

        # Compute catalog-level skip items for fallback negative sampling
        self.global_skipped_tokens: np.ndarray = np.array([], dtype=np.int64)
        if duration_col in combined_df.columns:
            skip_mask = combined_df[duration_col] < skip_threshold_sec
            if skip_mask.any():
                self.global_skipped_tokens = combined_df.loc[skip_mask, item_token_col].unique()

        self.all_item_tokens = combined_df[item_token_col].unique()

        # Build preference pairs
        self.samples: List[Dict[str, Any]] = []
        self._mine_preference_pairs(
            df=combined_df,
            user_col=user_col,
            item_token_col=item_token_col,
            timestamp_col=timestamp_col,
            duration_col=duration_col,
        )

    def _mine_preference_pairs(
        self,
        df: pd.DataFrame,
        user_col: str,
        item_token_col: str,
        timestamp_col: str,
        duration_col: str,
    ) -> None:
        # Preprocessed dataset is already chronologically sorted by user.
        # Using sort=False in groupby avoids a redundant and expensive 33-million row re-sort!
        grouped = df.groupby(user_col, sort=False)
        has_duration = duration_col in df.columns
        total_pairs = 0
        skipped_users = 0

        logger.info(
            "Mining high-signal DPO pairs from %d user sessions (max %d pairs/user)...",
            len(grouped),
            self.max_pairs_per_user,
        )

        for user_id, group in grouped:
            item_seq = group[item_token_col].values.astype(np.int64)
            dur_seq = (
                group[duration_col].values.astype(np.float32)
                if has_duration
                else np.ones(len(item_seq), dtype=np.float32) * 60.0
            )

            if len(item_seq) < 2:
                skipped_users += 1
                continue

            # Define engagement & skip thresholds according to chosen strategy
            if self.preference_strategy == "played_ratio":
                # Relative completion proxy (normalized to typical 180s track)
                is_skip_mask = (dur_seq / 180.0) < 0.25
                engaged_threshold = 180.0 * 0.65  # ~117s
            elif self.preference_strategy == "margin_threshold":
                # Strict high-margin: >= 60s win vs < 20s loss (guaranteed Delta >= 40s)
                is_skip_mask = dur_seq < 20.0
                engaged_threshold = 60.0
            else:
                # Default binary skip cutoff (< 30s)
                is_skip_mask = dur_seq < self.skip_threshold_sec
                engaged_threshold = self.skip_threshold_sec

            # Identify all tracks this specific user skipped in their history
            user_skip_indices = np.where(is_skip_mask)[0]
            user_skipped_items = item_seq[user_skip_indices]

            # Mine backwards from the most recent interactions (highest context fidelity up to max_length)
            user_pairs = 0
            for t in range(len(item_seq) - 1, 0, -1):
                if user_pairs >= self.max_pairs_per_user:
                    break

                current_item = item_seq[t]
                current_dur = dur_seq[t]
                prefix_items = item_seq[:t]
                is_engaged = current_dur >= engaged_threshold

                if is_engaged:
                    win_item = current_item
                    win_dur = current_dur

                    # Mine dispreferred track:
                    # 1. Sample from user's own skipped tracks
                    if len(user_skipped_items) > 0:
                        lose_item = self.rng.choice(user_skipped_items)
                        lose_dur = 10.0
                    elif len(self.global_skipped_tokens) > 0:
                        # 2. Fallback to globally high-skip track
                        lose_item = self.rng.choice(self.global_skipped_tokens)
                        lose_dur = 10.0
                    else:
                        # 3. Uniform catalog sample
                        lose_item = self.rng.choice(self.all_item_tokens)
                        lose_dur = 10.0

                    if lose_item != win_item:
                        self.samples.append({
                            "user_id": user_id,
                            "prefix": prefix_items,
                            "win_item": win_item,
                            "lose_item": lose_item,
                            "win_dur": win_dur,
                            "lose_dur": lose_dur,
                        })
                        user_pairs += 1
                        total_pairs += 1

                elif is_skip_mask[t]:
                    # Current item was skipped: item_t is a natural negative for prefix_items!
                    lose_item = current_item
                    lose_dur = current_dur

                    # Look ahead for the next engaged track the user accepted
                    future_engaged = [k for k in range(t + 1, len(item_seq)) if dur_seq[k] >= engaged_threshold]
                    if future_engaged:
                        next_engaged_idx = future_engaged[0]
                        win_item = item_seq[next_engaged_idx]
                        win_dur = dur_seq[next_engaged_idx]

                        if lose_item != win_item:
                            self.samples.append({
                                "user_id": user_id,
                                "prefix": prefix_items,
                                "win_item": win_item,
                                "lose_item": lose_item,
                                "win_dur": win_dur,
                                "lose_dur": lose_dur,
                            })
                            user_pairs += 1
                            total_pairs += 1

        logger.info(
            "DPO Pairwise Mining Complete: %d high-signal pairs extracted from %d users.",
            len(self.samples),
            len(grouped),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        raw_prefix = sample["prefix"]

        # Truncate prefix if longer than max_length
        if len(raw_prefix) > self.max_length:
            raw_prefix = raw_prefix[-self.max_length:]

        curr_len = len(raw_prefix)
        pad_len = self.max_length - curr_len

        # Right padding with pad_token_id (0)
        input_ids = np.pad(raw_prefix, (0, pad_len), constant_values=self.pad_token_id)
        attention_mask = np.zeros(self.max_length, dtype=np.int64)
        attention_mask[:curr_len] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "winning_id": sample["win_item"],
            "losing_id": sample["lose_item"],
            "win_dur": sample["win_dur"],
            "lose_dur": sample["lose_dur"],
            "user_id": sample["user_id"],
        }


def dpo_collate_fn(batch_list: List[Dict[str, Any]]) -> DPORecBatch:
    """Collates individual DPO dictionary samples into a unified DPORecBatch."""
    input_ids = torch.from_numpy(np.stack([b["input_ids"] for b in batch_list])).long()
    attention_mask = torch.from_numpy(np.stack([b["attention_mask"] for b in batch_list])).long()
    winning_ids = torch.tensor([b["winning_id"] for b in batch_list], dtype=torch.long)
    losing_ids = torch.tensor([b["losing_id"] for b in batch_list], dtype=torch.long)
    win_durs = torch.tensor([b["win_dur"] for b in batch_list], dtype=torch.float32)
    lose_durs = torch.tensor([b["lose_dur"] for b in batch_list], dtype=torch.float32)
    user_ids = [b["user_id"] for b in batch_list]

    return DPORecBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        winning_ids=winning_ids,
        losing_ids=losing_ids,
        winning_durations=win_durs,
        losing_durations=lose_durs,
        user_ids=user_ids,
    )


def create_dpo_dataloader(
    df: pd.DataFrame,
    history_df: Optional[pd.DataFrame] = None,
    max_length: int = 64,
    batch_size: int = 64,
    shuffle: bool = True,
    skip_threshold_sec: float = 30.0,
    preference_strategy: str = "binary_skip",
    max_pairs_per_user: int = 2,
    user_col: str = "uid",
    num_workers: int = 0,
    pin_memory: bool = False,
    pad_token_id: int = 0,
    seed: int = 42,
) -> DataLoader:
    """Factory creating an optimized PyTorch DataLoader for DPO training and validation."""
    dataset = DPORecDataset(
        df=df,
        history_df=history_df,
        max_length=max_length,
        skip_threshold_sec=skip_threshold_sec,
        preference_strategy=preference_strategy,
        max_pairs_per_user=max_pairs_per_user,
        user_col=user_col,
        pad_token_id=pad_token_id,
        seed=seed,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dpo_collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
    )
