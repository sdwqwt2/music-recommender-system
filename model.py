"""
model.py
--------
Causal Language Model (GPT-2) for Autoregressive Next-Item Recommendation.

Initializes a compact GPT-2 architecture from scratch (no pre-trained NLP weights)
tailored for item catalogs and sequential interaction dynamics.

Architecture Specifications:
    - Base Class: transformers.GPT2LMHeadModel (initialized from scratch)
    - vocab_size = num_items + 1 (Index 0 is <PAD>)
    - n_embd = 128 (Compact latent embeddings)
    - n_layer = 3 (Transformer decoder blocks)
    - n_head = 4 (Multi-head self-attention heads)
    - n_positions = max_length (Context window length)
    - pad_token_id = 0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] (%(name)s) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Model")


@dataclass
class RecSysOutput:
    """
    Standardized model output container for sequential recommendation.
    
    Attributes:
        logits: FloatTensor [B, L, vocab_size] unnormalized prediction scores.
        loss: Optional scalar cross-entropy loss (if labels were provided).
        last_hidden_state: Optional FloatTensor [B, L, n_embd] contextual representations.
    """
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None


class RecSysGPT2(nn.Module):
    """
    Sequence-based Recommendation Model using a Causal GPT-2 Transformer initialized from scratch.

    In Next-Item Prediction, each interaction token sequence [s_1, ..., s_k] is projected
    into an embedding space, passed through causal self-attention layers with autoregressive
    masking, and projected to catalog logits [B, L, vocab_size].

    Args:
        num_items: Total number of active items in catalog (excluding <PAD>).
        max_length: Maximum sequence context length (n_positions).
        n_embd: Latent embedding dimension (default: 128).
        n_layer: Number of transformer decoder layers (default: 3).
        n_head: Number of attention heads (default: 4).
        dropout: Dropout probability applied to embeddings and residual connections.
        pad_token_id: Reserved padding token index (default: 0).
    """

    def __init__(
        self,
        num_items: int,
        max_length: int = 64,
        n_embd: int = 128,
        n_layer: int = 3,
        n_head: int = 4,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.num_items: int = num_items
        self.max_length: int = max_length
        self.pad_token_id: int = pad_token_id

        # Vocab size = num_items + 1 (item IDs 1..N, plus 0 for PAD)
        vocab_size = num_items + 1

        # Configure custom GPT-2 architecture
        self.config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=max_length,
            n_ctx=max_length,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            pad_token_id=pad_token_id,
            bos_token_id=None,  # Clear default text NLP tokens
            eos_token_id=None,
        )

        # STRICT REQUIREMENT: Initialized from scratch (NOT from pre-trained weights)
        self.transformer = GPT2LMHeadModel(self.config)

        # Cross-Entropy Loss with ignore_index=0 (<PAD>)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)

        # Count parameters
        num_params = sum(p.numel() for p in self.parameters())
        num_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "Initialized RecSysGPT2 from scratch: vocab_size=%d (num_items=%d + PAD), "
            "n_embd=%d, n_layer=%d, n_head=%d, max_length=%d. "
            "Total parameters: %d (trainable: %d)",
            vocab_size,
            num_items,
            n_embd,
            n_layer,
            n_head,
            max_length,
            num_params,
            num_trainable,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> RecSysOutput:
        """
        Forward pass through Causal GPT-2.

        Args:
            input_ids: LongTensor [B, L] containing item tokens.
            attention_mask: Optional LongTensor [B, L] with 1 for real items, 0 for <PAD>.
            labels: Optional LongTensor [B, L] containing shifted ground-truth item tokens.

        Returns:
            RecSysOutput containing logits [B, L, vocab_size], optional scalar loss,
            and optional last hidden state representations [B, L, n_embd].
        """
        # Auto-generate attention mask if omitted
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()

        # Forward through transformer (without internal labels to compute custom loss)
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        logits = transformer_outputs.logits  # [B, L, vocab_size]
        last_hidden_state = transformer_outputs.hidden_states[-1]  # [B, L, n_embd]

        loss = None
        if labels is not None:
            # Flatten batch and sequence dimensions for cross-entropy
            # Padded positions in labels (value 0) are ignored in loss computation
            loss = self.loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

        return RecSysOutput(
            logits=logits,
            loss=loss,
            last_hidden_state=last_hidden_state,
        )

    def get_last_action_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extracts the contextual representation vector h_t at the last valid interaction token.
        
        Essential for Phase 2 (RL Fine-Tuning): This vector serves as the MDP State s_t
        for the RL policy actor/critic networks.

        Args:
            input_ids: LongTensor [B, L]
            attention_mask: Optional LongTensor [B, L]

        Returns:
            FloatTensor [B, n_embd] state representations.
        """
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()

        out = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # [B, L, n_embd]

        # Find the index of the last active token in each sequence
        # Sequence lengths: sum across attention_mask
        seq_lens = attention_mask.sum(dim=-1) - 1  # 0-indexed [B]
        seq_lens = torch.clamp(seq_lens, min=0)

        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        last_states = hidden[batch_indices, seq_lens]  # [B, n_embd]
        return last_states

    @torch.no_grad()
    def predict_next_items(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        top_k: int = 10,
        filter_history: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates Top-K next item recommendations for the active user sequence.

        Args:
            input_ids: LongTensor [B, L]
            attention_mask: Optional LongTensor [B, L]
            top_k: Number of recommendations to return per sequence.
            filter_history: If True, masks previously interacted items so they are not re-recommended.

        Returns:
            Tuple of (topk_scores [B, top_k], topk_indices [B, top_k]).
        """
        self.eval()
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()

        out = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits  # [B, L, vocab_size]

        # Extract logits corresponding to the last observed item
        seq_lens = attention_mask.sum(dim=-1) - 1
        seq_lens = torch.clamp(seq_lens, min=0)
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        next_item_logits = logits[batch_indices, seq_lens]  # [B, vocab_size]

        # Never recommend the <PAD> token (index 0)
        next_item_logits[:, self.pad_token_id] = -float("inf")

        if filter_history:
            # Mask out items that appear in the user's historical context
            history_mask = torch.zeros_like(next_item_logits, dtype=torch.bool)
            history_mask.scatter_(1, input_ids, True)
            history_mask[:, self.pad_token_id] = False  # preserve pad flag
            next_item_logits = next_item_logits.masked_fill(history_mask, -float("inf"))

        # Compute top_k scores and indices
        topk_scores, topk_indices = torch.topk(next_item_logits, k=top_k, dim=-1)
        return topk_scores, topk_indices


def build_gpt2_recsys_model(
    num_items: int,
    max_length: int = 64,
    n_embd: int = 128,
    n_layer: int = 3,
    n_head: int = 4,
    dropout: float = 0.1,
    pad_token_id: int = 0,
) -> RecSysGPT2:
    """Factory helper to build and return a RecSysGPT2 model instance."""
    return RecSysGPT2(
        num_items=num_items,
        max_length=max_length,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        dropout=dropout,
        pad_token_id=pad_token_id,
    )
