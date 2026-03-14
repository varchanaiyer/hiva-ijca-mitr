# MITR Logical Reasoning Results — All MI Strategies

## What is this experiment?

We trained a DistilBERT model on **BoolQ** (yes/no questions about Wikipedia passages) and tested whether adding a **mutual information (MI) penalty** between transformer layers improves logical reasoning.

The idea is simple: each transformer layer should learn something *different*. If two layers learn the same thing, that's wasted capacity. By penalizing this redundancy, we force the model to use each layer for a distinct "reasoning step," which should help with logical consistency.

We tested **4 different ways** to measure layer similarity (and thus penalize redundancy):

| Strategy | What it does | Has learnable parameters? |
|----------|-------------|--------------------------|
| **CLUB** | Trains a small network to estimate mutual information between layers | Yes |
| **InfoNCE** | Uses contrastive learning — treats matching layer pairs as positives, random pairs as negatives | Yes |
| **Cosine** | Measures the angle between layer outputs (cosine similarity) | No |
| **CKA** | Compares the *shape* of representations across the whole batch, not just individual samples | No |

## The two experiments

**Experiment 1 — Accuracy:** Can MITR maintain or improve accuracy on yes/no questions?

**Experiment 2 — Contradiction Rate:** If you ask "Is water wet?" and "Is water not wet?", a logical model should give opposite answers. The contradiction rate measures how often the model gives the *same* answer to both (which is logically impossible).

## Results

![MITR All Strategies Comparison](cka_results.png)

### What the charts show

**Training Loss (top left):**
- Cosine and CKA track closely with the Baseline — they don't destabilize training
- CLUB diverges early (loss goes negative) before recovering
- InfoNCE is very volatile and struggles to converge

**Validation Accuracy (top middle):**
- **Cosine** and **CKA** both maintain accuracy close to Baseline (~0.64-0.65)
- **CLUB** drops to ~0.60 — the MI penalty is too aggressive with learned estimators
- **InfoNCE** collapses to ~0.53 — the contrastive loss fights with the task loss

**MI Regularisation Loss (top right):**
- **CKA is the most stable** — nearly flat line, no spikes
- **Cosine** is also stable but slightly noisier
- **CLUB** starts high and decreases as its internal network learns
- **InfoNCE** is unstable throughout

**Final Accuracy (bottom left):**
- Cosine: ~0.6493 (best, slightly above Baseline)
- Baseline: ~0.6480
- CKA: ~0.6413 (competitive, slight drop)
- CLUB: ~0.5960 (significant drop)
- InfoNCE: ~0.5347 (collapsed)

**Contradiction Rate (bottom middle) — lower is better:**
- **CKA: lowest contradiction rate** — the most logically consistent model
- Baseline and CLUB are similar
- Cosine and InfoNCE have higher contradiction rates

## Key takeaways

### CKA is the best strategy for logical consistency

Even though CKA has slightly lower accuracy than Cosine, it produces the **most logically consistent** model (lowest contradiction rate). This is exactly what we predicted — CKA measures *structural* similarity between layers, not just point-wise similarity. Two layers that learn the same information in a rotated coordinate system will look different to Cosine but identical to CKA.

For logical reasoning, this matters: each deduction step needs to be *truly* independent, not just superficially different.

### Stability matters

The parameter-free methods (CKA and Cosine) are far more stable than the learned methods (CLUB and InfoNCE). This makes sense — CLUB and InfoNCE add extra networks that need their own training, which can conflict with the main task.

### Summary table

| Model | Accuracy | Contradiction Rate | Stability |
|-------|----------|-------------------|-----------|
| Baseline | ~0.648 | Medium | N/A |
| MITR-CLUB | ~0.596 | Medium | Unstable early |
| MITR-InfoNCE | ~0.535 | High | Very unstable |
| MITR-Cosine | ~0.649 | Higher | Stable |
| **MITR-CKA** | **~0.641** | **Lowest** | **Most stable** |

### Bottom line

If you care about **accuracy only** -> Cosine (or just Baseline)

If you care about **logical consistency** -> CKA is the clear winner

If you care about **both** -> CKA gives you the best trade-off (tiny accuracy drop for the biggest consistency gain)

## Setup

- Model: `distilbert-base-uncased`
- Dataset: BoolQ (8,000 train / 1,500 val)
- GPU: A100
- Epochs: 5
- Batch size: 64
- MI lambda: 0.01 (with 200-step linear warmup)
- Precision: BF16
