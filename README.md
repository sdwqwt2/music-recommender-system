# T-Lab RecSys 2026: Causal GPT-2 Sequential Recommender System

A modular, production-grade, and leak-proof sequential recommendation pipeline built from scratch using causal transformer language models (GPT-2 architecture) on music streaming interaction logs (**Yambda / Yandex Music**).

---

## 🎯 Project Overview & Objective

In modern sequential recommender systems, user interaction trajectories are modeled as discrete token sequences. While traditional text language models predict the next word token, here we train a **Causal Transformer from scratch** to predict the **Next Item (Track)** interacted with by a user:

$$\hat{y}_{t+1} = \arg\max_{i \in \mathcal{V}} P(s_{t+1} = i \mid s_1, s_2, \dots, s_t)$$

- **Phase 1 (Current):** Build a leak-proof data pipeline, causal sequence generator, and base GPT-2 model trained with Cross-Entropy Loss to master sequential item transition dynamics.
- **Phase 2 (Future RL Iteration):** Fine-tune the base policy using Reinforcement Learning (PPO / Policy Gradient / Offline RL) to optimize long-term user engagement and cumulative listening duration (`listen_duration_sec`).

---

## 📂 Repository Structure

```
Music-Recommender-System/
├── data_utils.py       # Frequency filtering, strict temporal splits, ItemTokenizer
├── dataset.py          # PyTorch Dataset, causal sequence shifting, padding, batching
├── model.py            # GPT2LMHeadModel initialized from scratch (custom GPT2Config)
├── train_base.py       # Training loop, ranking metrics (HR@10, NDCG@10), WandB, checkpointing
├── requirements.txt    # Project dependencies
└── README.md           # Architecture, mathematical formulation, and documentation
```

---

## 🔬 Architectural & Engineering Components

### 1. Data Preprocessing & Leak-Proof Splitting (`data_utils.py`)
- **Rare-Item Pruning:** Filters tracks with interaction count $< \text{min\_item\_frequency}$ (default: 5) to control vocabulary growth and avoid bloating the token embedding matrix and final output projection head.
- **Continuous Re-Indexing:** Item IDs are mapped to contiguous integers in range $[1, \dots, |\mathcal{V}|]$, strictly reserving index `0` for `<PAD>`.
- **Temporal Splitting Strategies:**
  - **`global_temporal`:** Determines global timestamp cutoffs $T_{\text{train}}$ and $T_{\text{val}}$. All events prior to $T_{\text{train}}$ form the train set, events in $[T_{\text{train}}, T_{\text{val}})$ form validation, and events $\ge T_{\text{val}}$ form testing. This mirrors true offline-to-online deployment horizons.
  - **`user_temporal` (Leave-K-Out):** For each user's chronologically sorted interactions, holds out the penultimate interaction for validation and the final interaction for test.
- **Leak-Proof Guarantee:** The vocabulary and `ItemTokenizer` are fitted **exclusively on the training split**, ensuring future items do not leak into the model's vocabulary table. Cold-start items in validation/test splits are pruned cleanly.

### 2. PyTorch Dataset & Causal Alignment (`dataset.py`)
- **Autoregressive Target Shifting:** For a user interaction trajectory $[s_1, s_2, \dots, s_{k+1}]$:
  - Input IDs: $[s_1, s_2, \dots, s_k]$
  - Target Labels: $[s_2, s_3, \dots, s_{k+1}]$
- **Padding & Masking:** Sequences shorter than `max_length` (64) are right-padded with `0`. The `attention_mask` marks active items as `1` and `<PAD>` as `0`.
- **Training Sequence Augmentation:** Supports configurable `sliding_window` chunking for training data augmentation across long user listening histories.
- **Historical Context in Evaluation:** For validation and testing, user historical interactions prior to the evaluation timestamp are concatenated to provide the transformer with complete past trajectory context.
- **Phase 2 RL Readiness:** Aligns listening durations $[r_2, r_3, \dots, r_{k+1}]$ with target labels, preserving dense step rewards for reinforcement learning fine-tuning.

### 3. Model Architecture (`model.py`)
- **Zero Pre-Trained NLP Weights:** Initialized strictly from scratch using HuggingFace `GPT2Config` and `GPT2LMHeadModel` without pre-trained text checkpoints.
- **Hyperparameter Specifications:**
  - `vocab_size = num_items + 1` (index 0 reserved for `<PAD>`)
  - `n_embd = 128` (compact latent item representations)
  - `n_layer = 3` (Transformer decoder layers)
  - `n_head = 4` (Multi-head self-attention heads)
  - `n_positions = 64` (Context length)
  - `pad_token_id = 0`
- **RL State Representation:** Provides `get_last_action_states()` to extract contextual representations $h_t \in \mathbb{R}^{d}$ from the final active interaction token, serving as the MDP state vector $s_t$ for actor-critic networks.
- **Inference & Recommendation:** Provides `predict_next_items()` returning Top-K item recommendations with optional masking of previously consumed tracks.

### 4. Training & RecSys Ranking Metrics (`train_base.py`)
- **Compute Optimization:** Auto-detects Apple Silicon (`mps`), NVIDIA (`cuda`), or fallback (`cpu`).
- **Loss Computation:** Autoregressive Cross-Entropy Loss with `ignore_index=0`, masking `<PAD>` positions from gradient updates.
- **RecSys Evaluation Metrics:** Computes **HitRate@10** and **NDCG@10** at the final sequence position:
  $$\text{HR@10} = \frac{1}{|U|} \sum_{u \in U} \mathbf{1}_{\{y_u \in \text{Top10}_u\}}$$
  $$\text{NDCG@10} = \frac{1}{|U|} \sum_{u \in U} \frac{1}{\log_2(\text{rank}_u + 1)}$$
- **Optimization:** AdamW with weight decay, Cosine Annealing learning rate schedule, and gradient norm clipping.
- **Tracking & Checkpointing:** Built-in Weights & Biases (`wandb`) integration and automated checkpointing of the best model based on validation ranking metrics.

---

## 🚀 Quickstart

### 1. Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. End-to-End Smoke Test (Synthetic Yambda Data)
Run training on a synthetically generated Yambda-style interaction log with 5,000 events:
```bash
python train_base.py --synthetic --epochs 5 --batch_size 32
```

### 3. Training on Custom / Real Yambda Parquet Dataset
```bash
python train_base.py \
    --data_path /path/to/yambda_interactions.parquet \
    --processed_dir data/processed \
    --output_dir checkpoints \
    --min_item_freq 5 \
    --max_length 64 \
    --n_embd 128 \
    --n_layer 3 \
    --n_head 4 \
    --batch_size 64 \
    --lr 5e-4 \
    --epochs 10 \
    --split_mode global_temporal
```

To enable Weights & Biases logging, add the flag:
```bash
--use_wandb --wandb_project yambda-gpt2-recsys
```

### 4. Running on Kaggle (GPU & Mixed Precision FP16)
Kaggle provides **30 hours/week of free NVIDIA GPU (T4 x 2 or P100)** with 16GB VRAM, which accelerates training by ~10x over standard CPU setups.

1. **Pre-flight settings in Kaggle Notebook:**
   - In the right-hand panel under **Settings -> Accelerator**, select **GPU T4 x 2** or **GPU P100**.
   - Under **Settings -> Internet**, toggle **Internet ON** (required for packages and cloning).
   - Attach your Yambda dataset under `/kaggle/input/` via **+ Add Input**.

2. **Option A: Upload the Turnkey Runner Notebook:**
   - Simply upload [kaggle_run.ipynb](file:///Users/lex/Antigravity/Music-Recommender-System/kaggle_run.ipynb) to Kaggle via **File -> Upload Notebook**.
   - Run all cells sequentially to verify GPU, install dependencies, train with `--fp16`, test recommendations, and export checkpoints.

3. **Option B: Run via Terminal / Git Clone in Kaggle:**
   ```bash
   !git clone https://github.com/<your-username>/Music-Recommender-System.git
   %cd Music-Recommender-System
   !pip install -q -r requirements.txt
   
   !python train_base.py \
       --data_path /kaggle/input/<dataset-folder>/yambda_interactions.parquet \
       --processed_dir /kaggle/working/data/processed \
       --output_dir /kaggle/working/checkpoints \
       --batch_size 128 \
       --epochs 10 \
       --fp16 \
       --num_workers 2
   ```
4. **Artifacts Output:**
   - Checkpoints are saved to `/kaggle/working/checkpoints/best_gpt2_recsys.pt`.
   - Download the model weights directly from Kaggle's **Output** panel.

---

## 🔮 Phase 2 Roadmap: Reinforcement Learning (RL) Fine-Tuning

```
Phase 1: Supervised Next-Item Prediction (Causal GPT-2 Pretraining)
  │
  ▼
Phase 2: Markov Decision Process (MDP) RL Fine-Tuning
  ├── State (s_t):      Contextual hidden state h_t from RecSysGPT2.get_last_action_states()
  ├── Action (a_t):     Recommended item token index a_t \in {1, ..., |\mathcal{V}|}
  ├── Reward (r_t):     Function of listen_duration_sec:
  │                     r_t = \log(1 + listen_duration_sec) - penalty_{skip}
  └── Policy Optimization:
        - Value Network (Critic) initialized on top of h_t
        - PPO / REINFORCE with KL-divergence penalty against Phase 1 Base Policy:
          L_{RL}(\theta) = \mathbb{E} [ r_t - \beta D_{KL}(\pi_\theta || \pi_{\text{base}}) ]
```
