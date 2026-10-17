"""
data_utils.py
-------------
Data Preprocessing, Temporal Splitting, and Item Tokenization for Sequence-Based Recommenders.

Designed for Yambda (Yandex Music) interaction logs with columns:
    ['user_id', 'item_id', 'timestamp', 'listen_duration_sec']

Key Guarantees:
1. Strict Temporal Splitting: Zero future-data leakage into training/validation.
2. Low-Frequency Pruning: Removes rare items (< min_item_frequency) to curb vocabulary explosion.
3. Continuous Item Remapping: 1-indexed continuous integer IDs [1, ..., N], reserving 0 for <PAD>.
4. Artifact Serialization: Serializes tokenizers and dataset statistics for seamless downstream evaluation
   and future Reinforcement Learning (RL) reward modeling phases.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] (%(name)s) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DataUtils")


@dataclass
class DatasetStats:
    """Statistical summary of processed interaction data."""
    num_users: int
    num_items: int
    num_interactions: int
    min_timestamp: int
    max_timestamp: int
    train_interactions: int
    val_interactions: int
    test_interactions: int
    density: float
    avg_seq_length: float


class ItemTokenizer:
    """
    Tokenizes raw item identifiers (strings, ints, uuids) into continuous integers.
    
    Vocabulary convention:
        - 0: <PAD> token (reserved for sequence padding and masked in loss calculation)
        - 1 ... num_items: valid item indices mapped from training catalog
        - (Optional) num_items + 1: <UNK> token for cold-start / unseen items
    """

    def __init__(self, pad_token_id: int = 0, unk_token_id: Optional[int] = None) -> None:
        self.pad_token_id: int = pad_token_id
        self.item2idx: Dict[Union[int, str], int] = {}
        self.idx2item: Dict[int, Union[int, str]] = {}
        self.unk_token_id: Optional[int] = unk_token_id
        self.is_fitted: bool = False

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens (<PAD> and items)."""
        # Base count: num items + 1 (<PAD>) + (1 if UNK else 0)
        base = len(self.item2idx) + 1
        if self.unk_token_id is not None:
            base += 1
        return base

    @property
    def num_items(self) -> int:
        """Number of active unique items registered in the vocabulary."""
        return len(self.item2idx)

    def fit(self, items: Union[pd.Series, List[Union[int, str]], np.ndarray], allow_unk: bool = False) -> ItemTokenizer:
        """
        Builds the item-to-index mapping from an iterable of unique items.
        Items are mapped starting at index 1 (leaving 0 for PAD).

        Args:
            items: Collection of unique item IDs from the training split.
            allow_unk: If True, reserves the last token as <UNK>.
        """
        unique_items = pd.Series(items).drop_duplicates().tolist()
        self.item2idx = {}
        self.idx2item = {}

        # 1-based indexing for real items
        for idx, item in enumerate(unique_items, start=1):
            self.item2idx[item] = idx
            self.idx2item[idx] = item

        if allow_unk:
            self.unk_token_id = len(self.item2idx) + 1
        else:
            self.unk_token_id = None

        self.is_fitted = True
        logger.info(
            "ItemTokenizer fitted with %d unique items. Vocab size (inc PAD): %d",
            len(self.item2idx),
            self.vocab_size,
        )
        return self

    def transform(self, items: Union[pd.Series, List[Union[int, str]], np.ndarray]) -> np.ndarray:
        """
        Converts raw item IDs to continuous token indices.
        Unseen items are mapped to <UNK> if enabled, or raise KeyError / return -1.
        """
        if not self.is_fitted:
            raise RuntimeError("ItemTokenizer must be fitted before calling transform().")

        if isinstance(items, pd.Series):
            iterable = items.values
        else:
            iterable = items

        encoded = np.empty(len(iterable), dtype=np.int64)
        for i, val in enumerate(iterable):
            if val in self.item2idx:
                encoded[i] = self.item2idx[val]
            elif self.unk_token_id is not None:
                encoded[i] = self.unk_token_id
            else:
                # Mark as out-of-vocabulary
                encoded[i] = -1

        return encoded

    def inverse_transform(self, indices: Union[List[int], np.ndarray]) -> List[Union[int, str]]:
        """Converts token indices back to raw item IDs."""
        return [self.idx2item.get(idx, "<PAD>" if idx == self.pad_token_id else "<UNK>") for idx in indices]

    def save(self, filepath: Union[str, Path]) -> None:
        """Persists the tokenizer mapping to a JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        # Note: JSON keys must be strings; store item2idx and idx2item
        payload = {
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "item2idx": {str(k): v for k, v in self.item2idx.items()},
            "idx2item": {int(k): v for k, v in self.idx2item.items()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved ItemTokenizer to %s", filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> ItemTokenizer:
        """Loads a persisted ItemTokenizer from a JSON file."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)

        tokenizer = cls(
            pad_token_id=payload["pad_token_id"],
            unk_token_id=payload.get("unk_token_id"),
        )
        # Reconstruct mappings with appropriate types
        tokenizer.item2idx = payload["item2idx"]
        tokenizer.idx2item = {int(k): v for k, v in payload["idx2item"].items()}
        tokenizer.is_fitted = True
        return tokenizer


def filter_rare_items(
    df: pd.DataFrame,
    min_item_frequency: int = 5,
    min_user_interactions: int = 3,
    max_items: Optional[int] = None,
    item_col: str = "item_id",
    user_col: str = "uid",
) -> pd.DataFrame:
    """
    Prunes items appearing fewer than `min_item_frequency` times.
    Optionally caps vocabulary to top `max_items` to strictly bound VRAM.
    After pruning items, users with fewer than `min_user_interactions` are removed
    to ensure every remaining user sequence can yield at least one valid (input, label) pair.

    Args:
        df: Input DataFrame containing interaction logs.
        min_item_frequency: Threshold below which items are filtered out.
        min_user_interactions: Threshold below which user sequences are discarded.
        max_items: Optional hard cap on unique items (retains top frequent items).
        item_col: Column name representing item identifiers.
        user_col: Column name representing user identifiers (default: 'uid').

    Returns:
        Filtered DataFrame with reduced sparsity and controlled vocabulary size.
    """
    # Auto-resolve user column alias if needed
    if user_col not in df.columns:
        if "uid" in df.columns:
            user_col = "uid"
        elif "user_id" in df.columns:
            user_col = "user_id"

    initial_rows = len(df)
    initial_items = df[item_col].nunique()
    initial_users = df[user_col].nunique()

    # Step 1: Filter rare items and optionally cap vocabulary
    item_counts = df[item_col].value_counts()
    frequent_items = item_counts[item_counts >= min_item_frequency].index
    if max_items is not None and len(frequent_items) > max_items:
        logger.info(
            "Capping item catalog from %d frequent items to top %d items to strictly control VRAM",
            len(frequent_items),
            max_items,
        )
        frequent_items = frequent_items[:max_items]

    filtered_df = df[df[item_col].isin(frequent_items)].copy()

    # Step 2: Filter users with too few interactions after item pruning
    user_counts = filtered_df[user_col].value_counts()
    active_users = user_counts[user_counts >= min_user_interactions].index
    filtered_df = filtered_df[filtered_df[user_col].isin(active_users)].copy()

    logger.info(
        "Rare-item filtering (freq >= %d, user_interactions >= %d, max_items=%s): "
        "Rows: %d -> %d (%.2f%% retained), Items: %d -> %d, Users: %d -> %d",
        min_item_frequency,
        min_user_interactions,
        max_items,
        initial_rows,
        len(filtered_df),
        (len(filtered_df) / max(1, initial_rows)) * 100.0,
        initial_items,
        filtered_df[item_col].nunique(),
        initial_users,
        filtered_df[user_col].nunique(),
    )
    return filtered_df


def split_data_temporally(
    df: pd.DataFrame,
    split_mode: Literal["global_temporal", "user_temporal"] = "global_temporal",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    timestamp_col: str = "timestamp",
    user_col: str = "uid",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs a strict temporal train/val/test split guaranteeing zero data leakage.

    Splitting Modes:
    1. 'global_temporal':
       Computes global timestamp cutoffs (T_train, T_val).
       Interactions < T_train -> Train.
       T_train <= Interactions < T_val -> Val.
       Interactions >= T_val -> Test.
       Simulates a true production offline-to-online time horizon.

    2. 'user_temporal' (Leave-K-out):
       For each user's sorted interaction history:
       - The final interaction is held out for Test.
       - The penultimate interaction is held out for Val.
       - All preceding interactions constitute Train.
       Guarantees historical sequences for every user in val and test.

    Args:
        df: Input DataFrame.
        split_mode: 'global_temporal' or 'user_temporal'.
        train_ratio: Proportion of timeline / interactions for train (used in global_temporal).
        val_ratio: Proportion of timeline / interactions for val (used in global_temporal).
        timestamp_col: Column containing UNIX timestamps.
        user_col: Column containing user identifiers.

    Returns:
        Tuple of (train_df, val_df, test_df) sorted chronologically.
    """
    # Ensure chronological order
    df = df.sort_values(by=[user_col, timestamp_col]).reset_index(drop=True)

    if split_mode == "global_temporal":
        # Global timestamp quantile split
        timestamps = df[timestamp_col].values
        t_train = np.quantile(timestamps, train_ratio)
        t_val = np.quantile(timestamps, train_ratio + val_ratio)

        logger.info(
            "Global Temporal Split Cutoffs: Train < %.1f, Val [%.1f, %.1f), Test >= %.1f",
            t_train,
            t_train,
            t_val,
            t_val,
        )

        train_mask = df[timestamp_col] < t_train
        val_mask = (df[timestamp_col] >= t_train) & (df[timestamp_col] < t_val)
        test_mask = df[timestamp_col] >= t_val

        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        test_df = df[test_mask].copy()

    elif split_mode == "user_temporal":
        # Leave-two-out per user (last for test, 2nd-to-last for val, rest for train)
        logger.info("Performing per-user temporal leave-k-out split (last=test, 2nd-last=val)...")

        # Rank interactions chronologically per user: 1 is oldest, max is newest
        df["rank_asc"] = df.groupby(user_col).cumcount() + 1
        df["total_user_len"] = df.groupby(user_col)[user_col].transform("count")
        # Reverse rank: 0 is newest, 1 is 2nd newest
        df["rev_rank"] = df["total_user_len"] - df["rank_asc"]

        test_mask = df["rev_rank"] == 0
        val_mask = df["rev_rank"] == 1
        train_mask = df["rev_rank"] >= 2

        train_df = df[train_mask].drop(columns=["rank_asc", "total_user_len", "rev_rank"]).copy()
        val_df = df[val_mask].drop(columns=["rank_asc", "total_user_len", "rev_rank"]).copy()
        test_df = df[test_mask].drop(columns=["rank_asc", "total_user_len", "rev_rank"]).copy()

    else:
        raise ValueError(f"Unknown split_mode '{split_mode}'. Choose 'global_temporal' or 'user_temporal'.")

    logger.info(
        "Temporal Split Completed: Train=%d, Val=%d, Test=%d interactions",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


def preprocess_pipeline(
    data_path: Union[str, Path],
    output_dir: Union[str, Path],
    min_item_frequency: int = 5,
    min_user_interactions: int = 3,
    max_items: Optional[int] = None,
    split_mode: Literal["global_temporal", "user_temporal"] = "global_temporal",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    user_col: str = "uid",
    item_col: str = "item_id",
    timestamp_col: str = "timestamp",
    duration_col: str = "listen_duration_sec",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ItemTokenizer, DatasetStats]:
    """
    End-to-end reproducible preprocessing pipeline for Yambda-style interaction datasets.

    Pipeline Steps:
    1. Load data from CSV or Parquet.
    2. Harmonize schema (uid vs user_id, derive listen_duration_sec).
    3. Filter out rare items and low-interaction users to cap vocab & memory.
    4. Perform strict time-series split (Train / Val / Test).
    5. Fit ItemTokenizer strictly on the training catalog (zero leakage of test catalog).
    6. Encode item_id into continuous integer indices [1, ..., N] (0 = <PAD>).
    7. Filter out cold-start/unseen items from Val and Test to maintain evaluateable sequences.
    8. Save tokenized datasets, tokenizer mapping, and metadata stats to disk.

    Returns:
        (train_df, val_df, test_df, tokenizer, stats)
    """
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading interaction data from: %s", data_path)
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    elif data_path.suffix in [".csv", ".tsv"]:
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format '{data_path.suffix}'. Expected .parquet or .csv")

    # 1. Schema Harmonization: Map user identifier flexibly
    if user_col not in df.columns:
        if "uid" in df.columns:
            logger.info("Harmonizing user identifier column 'uid' -> '%s'", user_col)
            df[user_col] = df["uid"]
        elif "user_id" in df.columns:
            logger.info("Harmonizing user identifier column 'user_id' -> '%s'", user_col)
            df[user_col] = df["user_id"]
        else:
            raise KeyError(
                f"Missing user identifier column '{user_col}' (and aliases 'uid', 'user_id'). "
                f"Available columns: {list(df.columns)}"
            )

    # 2. Derive listen_duration_sec from native Yambda columns if missing
    if duration_col not in df.columns:
        if "played_ratio_pct" in df.columns and "track_length_seconds" in df.columns:
            logger.info("Computing '%s' from Yambda 'played_ratio_pct' and 'track_length_seconds'", duration_col)
            df[duration_col] = (df["played_ratio_pct"].clip(lower=0.0) / 100.0) * df["track_length_seconds"].clip(lower=0.0)
        elif "track_length_seconds" in df.columns:
            df[duration_col] = df["track_length_seconds"].clip(lower=0.0)
        else:
            logger.warning(
                "Column '%s' not found. Creating default dummy listening durations (30.0s).",
                duration_col,
            )
            df[duration_col] = 30.0

    # 3. Validate essential schema
    required_cols = [user_col, item_col, timestamp_col, duration_col]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in dataset. Available: {list(df.columns)}")

    # 1. Filter rare items & short sequences
    df = filter_rare_items(
        df=df,
        min_item_frequency=min_item_frequency,
        min_user_interactions=min_user_interactions,
        max_items=max_items,
        item_col=item_col,
        user_col=user_col,
    )

    # 2. Strict temporal train / val / test split
    train_df, val_df, test_df = split_data_temporally(
        df=df,
        split_mode=split_mode,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        timestamp_col=timestamp_col,
        user_col=user_col,
    )

    # 3. Fit Tokenizer strictly on train_df
    # In recsys, vocabulary must be formed exclusively from known training items.
    tokenizer = ItemTokenizer(pad_token_id=0)
    tokenizer.fit(train_df[item_col].unique())

    # 4. Map item_id to continuous integer tokens
    train_df["item_token"] = tokenizer.transform(train_df[item_col])
    val_df["item_token"] = tokenizer.transform(val_df[item_col])
    test_df["item_token"] = tokenizer.transform(test_df[item_col])

    # 5. Drop cold-start items (unseen in train) from validation and test sets
    val_unseen = (val_df["item_token"] == -1).sum()
    test_unseen = (test_df["item_token"] == -1).sum()
    if val_unseen > 0:
        logger.info("Pruning %d unseen/cold-start item interactions from Val set", val_unseen)
        val_df = val_df[val_df["item_token"] != -1].copy()
    if test_unseen > 0:
        logger.info("Pruning %d unseen/cold-start item interactions from Test set", test_unseen)
        test_df = test_df[test_df["item_token"] != -1].copy()

    # 6. Compute Dataset Statistics
    total_users = df[user_col].nunique()
    total_items = tokenizer.num_items
    total_interactions = len(train_df) + len(val_df) + len(test_df)
    stats = DatasetStats(
        num_users=total_users,
        num_items=total_items,
        num_interactions=total_interactions,
        min_timestamp=int(df[timestamp_col].min()),
        max_timestamp=int(df[timestamp_col].max()),
        train_interactions=len(train_df),
        val_interactions=len(val_df),
        test_interactions=len(test_df),
        density=float(total_interactions / (max(1, total_users) * max(1, total_items))),
        avg_seq_length=float(df.groupby(user_col)[item_col].count().mean()),
    )

    # 7. Persist artifacts
    tokenizer.save(output_dir / "item_tokenizer.json")
    with open(output_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, indent=2)

    train_df.to_parquet(output_dir / "train_interactions.parquet", index=False)
    val_df.to_parquet(output_dir / "val_interactions.parquet", index=False)
    test_df.to_parquet(output_dir / "test_interactions.parquet", index=False)

    logger.info("Preprocessing complete. Artifacts successfully saved to: %s", output_dir)
    return train_df, val_df, test_df, tokenizer, stats
