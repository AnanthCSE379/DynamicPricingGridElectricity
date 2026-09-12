#  Technical Specification: Agentic AI Fine-Tuning & Apple MLX LoRA Architecture

This document details the machine learning architecture, dataset stratification engineering, and parameter-efficient fine-tuning (PEFT) methodology used to train the **Autonomous Grid Operator Agent** on Apple Silicon via Apple MLX.

---

## 1. Base Foundation Model: Qwen2.5-0.5B-Instruct

The agent utilizes the `Qwen2.5-0.5B-Instruct` decoder-only transformer (Alibaba Cloud / Qwen Team, 2024), selected for its exceptional reasoning-to-parameter ratio, low inference memory footprint, and native structured JSON instruction obedience.

### Structural Architecture
- **Total Parameters**: 494,032,896 (~494M).
- **Transformer Depth**: **24 hidden layers**.
- **Hidden Embedding Dimension ($d_{\text{model}}$)**: 896.
- **Intermediate Feed-Forward Dimension ($d_{\text{ffn}}$)**: 4,864 (SwiGLU activation).
- **Attention Heads**: 14 query heads, 2 key/value heads (Grouped-Query Attention, GQA).
- **Vocabulary Size**: 151,936 tokens.
- **Rotary Position Embedding (RoPE)**: Base $\theta = 1,000,000$ (32k context window).
- **Precision**: 16-bit Brain Floating Point (`bfloat16`).

---

## 2. Low-Rank Adaptation (LoRA) Formulation

Rather than performing full parameter fine-tuning, Low-Rank Adaptation (Hu et al., 2021) freezes the base weight matrix $W_0 \in \mathbb{R}^{d \times k}$ and injects low-rank trainable decomposition matrices $A$ and $B$:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$

*Where:*
- $A \in \mathbb{R}^{r \times k}$ is initialized with a Gaussian distribution $\mathcal{N}(0, \sigma^2)$.
- $B \in \mathbb{R}^{d \times r}$ is initialized to zero, ensuring $\Delta W = 0$ at step 0.
- $r$ is the intrinsic rank ($r \ll \min(d, k)$).
- $\frac{\alpha}{r}$ is a constant scaling factor.

### Configured Hyperparameters
| LoRA Parameter | Setting | Engineering Justification |
| :--- | :---: | :--- |
| **Adapted Depth** | **16 layers (out of 24)** | Adapts the upper two-thirds of the transformer depth where high-level policy reasoning resides. |
| **LoRA Rank ($r$)** | **16** | Sufficient rank capacity to learn non-linear dispatch curves and wholesale spread logic. |
| **LoRA Alpha ($\alpha$)** | **32** | $2\times$ scale factor maintains strong gradient flow into the adapter adapters. |
| **LoRA Dropout** | **0.05** | Prevents adapter co-adaptation and over-memorization of exact timestamps. |
| **Trainable Parameters** | **2,932,736** | **0.594% of total weights** (11.0 MB adapter checkpoint file `adapters.safetensors`). |

---

## 3. The Dataset Curation & Anti-Collapse Protocol

### The Mode Collapse Problem
In standard residential power grids, ~80% of half-hours operate under benign conditions (where optimal tariff $\approx 11.76\text{ p}$). In preliminary trials, chronological sampling caused the cross-entropy loss from the 80% majority class to overwhelm the 20% critical peak events, teaching the agent to predict $11.76\text{ p}$ flat for every step and incurring heavy financial losses during wholesale spikes.

### Stratified Peak-Balanced Sampling (15,000 Samples)
To solve mode collapse, `src/create_finetune_dataset_large.py` implemented a **tri-modal stratified partition** drawn across all 4 seasons from the training split:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   STRATIFIED 15,000-SAMPLE TRAINING DISTRIBUTION                 │
├─────────────────────────┬────────────────────────────┬───────────────────────────┤
│ Regime A: Peak Stress   │ Regime B: Off-Peak Valley  │ Regime C: Baseline Norm   │
│ 5,250 samples (35.0%)   │ 3,750 samples (25.0%)      │ 6,000 samples (40.0%)     │
│ Load > 0.23 or Wh > 15p │ Load < 0.16 & Wh < 10p     │ 0.16 <= Load <= 0.23      │
│ Target: 18p - 48p       │ Target: 3.99p - 8.00p      │ Target: 11.76p - 14.23p   │
└─────────────────────────┴────────────────────────────┴───────────────────────────┘
```

A parallel **1,000-sample balanced validation set** (`data/valid.jsonl`) was held out strictly from the validation split.

---

## 4. Hardware Optimization on Apple Silicon (M-Series GPU)

Fine-tuning was executed natively via Apple MLX (`mlx-lm`) utilizing Apple Silicon Unified Memory Architecture (UMA):

1. **Selective Prompt Loss Masking (`--mask-prompt`)**:
   In standard causal language modeling, loss is computed over all tokens in the sequence. With prompt masking, tokens corresponding to system instructions and input histories are masked with zero weight. The loss is computed **strictly on the assistant's JSON response tokens**:
   $$\mathcal{L} = -\frac{1}{M} \sum_{t \in \text{Target}} \log P(w_t \mid w_{<t})$$
   This focused 100% of gradient updates on numerical forecast precision and retail price setting.

2. **Gradient Checkpointing (`--grad-checkpoint`)**:
   Trades recomputation for memory by discarding intermediate activation tensors during the forward pass and recomputing them during backpropagation.
   - **Peak Unified Memory**: **`2.242 GB`** (guaranteeing zero swap pressure on 8GB RAM systems).

3. **Optimization Schedule**:
   - **Total Iterations**: **10,000 steps**.
   - **Batch Size**: 2 samples per step ($\approx 20,000$ sample exposures, $\approx 1.33$ epochs).
   - **Learning Rate**: $1 \times 10^{-4}$ with Adam optimizer.
   - **Checkpoints**: Serialized every 1,000 iterations to `adapters/`.
   - **Total Overnight Runtime**: ~6.8 hours on Apple M3 GPU.
