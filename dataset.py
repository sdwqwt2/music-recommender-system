"""
dataset.py
----------
PyTorch Dataset and DataLoader for Autoregressive Sequential Recommender Systems (Next Item Prediction).

Implements:
1. Chronological aggregation per user (sorted by timestamp).
2. Fixed-length sequence windowing (max_length).
3. Causal language modeling target alignment:
   - Input Sequence:  [s_1, s_2, ..., s_k]
   - Shifted Label:   [s_2, s_3, ..., s_{k+1}]
4. Attention masking and proper zero-padding (<PAD> = 0).
5. Listen duration alignment [r_2, r_3, ..., r_{k+1}] to support Phase 2 RL reward fine-tuning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] (%(name)s) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Dataset")


@dataclass
class RecBatch:
    """
    Structured PyTorch Batch for Causal Sequential Recommendation.
    
    Attributes:
        input_ids: LongTensor [B, L] containing item token indices.
        labels: LongTensor [B, L] containing next-item target token indices.
        attention_mask: LongTensor [B, L] containing 1 for valid items, 0 for <PAD>.
        durations: FloatTensor [B, L] containing listening duration in seconds (RL rewards).
        user_ids: List[Union[str, int]] containing raw user identifiers for evaluation grouping.
    """
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    durations: torch.Tensor
    user_ids: List[Union[str, int]]

    def to(self, device: Union[torch.device, str]) -> RecBatch:
        """Transfers batch tensors to target compute device (CPU/GPU/MPS)."""
        return RecBatch(
            input_ids=self.input_ids.to(device),
            labels=self.labels.to(device),
            attention_mask=self.attention_mask.to(device),
            durations=self.durations.to(device),
            user_ids=self.user_ids,
        )


class SequentialRecDataset(Dataset):
    """
    PyTorch Dataset generating causal (input, label) sequence pairs for Next Item Prediction.

    Args:
        df: DataFrame containing interaction logs. Must contain user_col, item_token_col,
            timestamp_col, and optionally duration_col.
        max_length: Maximum sequence length (context window) for the GPT-2 model.
        user_col: Column name for user identifiers.
        item_token_col: Column name for mapped continuous item tokens (1 ... N).
        timestamp_col: Column name for UNIX timestamps.
        duration_col: Column name for listen duration in seconds.
        sliding_window: If True, uses sliding window chunking to extract multiple subsequences
            from long user histories (recommended for training data augmentation).
            If False, retains only the most recent sequence (recommended for validation/test evaluation).
        stride: Stride step size when sliding_window is True. Defaults to max_length // 2.
        pad_token_id: Special token index for padding (default 0).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        history_df: Optional[pd.DataFrame] = None,
        max_length: int = 64,
        user_col: str = "uid",
        item_token_col: str = "item_token",
        timestamp_col: str = "timestamp",
        duration_col: str = "listen_duration_sec",
        sliding_window: bool = False,
        stride: Optional[int] = None,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.max_length: int = max_length
        self.pad_token_id: int = pad_token_id
        self.sliding_window: bool = sliding_window
        self.stride: int = stride if stride is not None else max(1, max_length // 2)

        # Resolve user_col alias if needed
        if user_col not in df.columns:
            if "uid" in df.columns:
                user_col = "uid"
            elif "user_id" in df.columns:
                user_col = "user_id"

        # In evaluation (val/test), concatenate historical interactions with target split
        # to ensure users have full chronological context up to the evaluation target.
        if history_df is not None:
            # Filter history to only users present in current target split
            active_users = set(df[user_col].unique())
            filtered_history = history_df[history_df[user_col].isin(active_users)]
            combined_df = pd.concat([filtered_history, df], ignore_index=True)
        else:
            combined_df = df

        # Store sequence samples: List of tuples (user_id, item_seq, duration_seq)
        self.samples: List[Tuple[Union[str, int], np.ndarray, np.ndarray]] = []
        self._build_sequences(
            df=combined_df,
            user_col=user_col,
            item_token_col=item_token_col,
            timestamp_col=timestamp_col,
            duration_col=duration_col,
        )

    def _build_sequences(
        self,
        df: pd.DataFrame,
        user_col: str,
        item_token_col: str,
        timestamp_col: str,
        duration_col: str,
    ) -> None:
        """Groups user interactions, sorts chronologically, and constructs sequences."""
        # Ensure chronological ordering
        sorted_df = df.sort_values(by=[user_col, timestamp_col]).reset_index(drop=True)
        has_duration = duration_col in sorted_df.columns

        grouped = sorted_df.groupby(user_col)
        skipped_short = 0

        for user_id, group in grouped:
            item_seq = group[item_token_col].values.astype(np.int64)
            duration_seq = (
                group[duration_col].values.astype(np.float32)
                if has_duration
                else np.ones(len(item_seq), dtype=np.float32) * 30.0
            )

            # Causal LM next-item prediction requires at least 2 tokens (1 input, 1 label)
            if len(item_seq) < 2:
                skipped_short += 1
                continue

            if self.sliding_window:
                # Sliding window chunking for training data augmentation
                # A window of size (max_length + 1) provides max_length inputs and max_length shifted labels
                window_size = self.max_length + 1
                if len(item_seq) <= window_size:
                    self.samples.append((user_id, item_seq, duration_seq))
                else:
                    for start_idx in range(0, len(item_seq) - 1, self.stride):
                        end_idx = min(start_idx + window_size, len(item_seq))
                        chunk_items = item_seq[start_idx:end_idx]
                        chunk_durations = duration_seq[start_idx:end_idx]
                        if len(chunk_items) >= 2:
                            self.samples.append((user_id, chunk_items, chunk_durations))
                        if end_idx == len(item_seq):
                            break
            else:
                # Evaluation mode: retain the most recent history up to (max_length + 1)
                chunk_items = item_seq[-(self.max_length + 1):]
                chunk_durations = duration_seq[-(self.max_length + 1):]
                self.samples.append((user_id, chunk_items, chunk_durations))

        logger.info(
            "Constructed %d sequence samples from %d users (skipped %d users with <2 interactions). "
            "sliding_window=%s, max_length=%d",
            len(self.samples),
            len(grouped),
            skipped_short,
            self.sliding_window,
            self.max_length,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, Union[str, int]]]:
        """
        Retrieves a single sequence sample, aligns inputs and shifted causal labels,
        and applies zero-padding to reach max_length.

        Let raw items be [s_1, s_2, ..., s_k, s_{k+1}].
        - Input sequence:   [s_1, s_2, ..., s_k]
        - Shifted Label:    [s_2, s_3, ..., s_{k+1}]
        - Shifted Duration: [r_2, r_3, ..., r_{k+1}]
        """
        user_id, items, durations = self.samples[idx]

        # Shift sequence for next-token prediction
        raw_inputs = items[:-1]
        raw_labels = items[1:]
        raw_durations = durations[1:]

        # Truncate if longer than max_length (e.g. from rightmost recent history)
        if len(raw_inputs) > self.max_length:
            raw_inputs = raw_inputs[-self.max_length:]
            raw_labels = raw_labels[-self.max_length:]
            raw_durations = raw_durations[-self.max_length:]

        curr_len = len(raw_inputs)
        pad_len = self.max_length - curr_len

        # Right-padding with pad_token_id (0)
        # Input IDs: [token_1, token_2, ..., token_k, 0, ..., 0]
        input_ids = np.pad(raw_inputs, (0, pad_len), constant_values=self.pad_token_id)
        # Labels: [token_2, token_3, ..., token_{k+1}, 0, ..., 0]
        labels = np.pad(raw_labels, (0, pad_len), constant_values=self.pad_token_id)
        # Attention Mask: 1 for active item tokens, 0 for <PAD>
        attention_mask = np.zeros(self.max_length, dtype=np.int64)
        attention_mask[:curr_len] = 1
        # Durations: aligned with labels for RL rewards
        durations_padded = np.pad(raw_durations, (0, pad_len), constant_values=0.0)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "durations": durations_padded,
            "user_id": user_id,
        }


def recsys_collate_fn(batch_list: List[Dict[str, Union[np.ndarray, Union[str, int]]]]) -> RecBatch:
    """
    Collate function that batches individual dictionary samples into a unified RecBatch.
    """
    input_ids = torch.from_numpy(np.stack([b["input_ids"] for b in batch_list])).long()
    labels = torch.from_numpy(np.stack([b["labels"] for b in batch_list])).long()
    attention_mask = torch.from_numpy(np.stack([b["attention_mask"] for b in batch_list])).long()
    durations = torch.from_numpy(np.stack([b["durations"] for b in batch_list])).float()
    user_ids = [b["user_id"] for b in batch_list]

    return RecBatch(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        durations=durations,
        user_ids=user_ids,
    )


def create_dataloader(
    df: pd.DataFrame,
    history_df: Optional[pd.DataFrame] = None,
    max_length: int = 64,
    batch_size: int = 128,
    shuffle: bool = True,
    sliding_window: bool = False,
    stride: Optional[int] = None,
    user_col: str = "uid",
    num_workers: int = 0,
    pin_memory: bool = False,
    pad_token_id: int = 0,
) -> DataLoader:
    """
    Factory function creating an optimized PyTorch DataLoader for sequential recommendation.
    """
    dataset = SequentialRecDataset(
        df=df,
        history_df=history_df,
        max_length=max_length,
        user_col=user_col,
        sliding_window=sliding_window,
        stride=stride,
        pad_token_id=pad_token_id,
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=recsys_collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
    )
