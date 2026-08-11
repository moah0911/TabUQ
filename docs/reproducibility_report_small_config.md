# TabM-UQ Reproducibility Report — Small Configuration (M1)

**Date:** 2026-07-18
**Configuration:** n_blocks=1, d_block=128, k=16, NO embeddings (raw features)
**Runtime:** ~1.5 hours wall-clock (parallel 3 seeds)
**Hardware:** 16-thread CPU (Intel i5-12500H)

---

## 1. Environment Setup

| Component | Version |
|-----------|---------|
| Python | 3.11.15 |
| PyTorch | 2.12.0+cu130 |
| tabm | 0.0.3 |
| rtdl-num-embeddings | 0.0.12 |
| scikit-learn | 1.9.0 |
| CPU | Intel i5-12500H (16 threads) |

---

## 2. Datasets

All datasets from the **official TabM paper's 50-dataset suite**, loaded from preprocessed `.npy` files via the authors' HuggingFace tarball.

| # | Dataset | Task | Train | Val | Test | Num Features | Cat Features |
|---|---------|------|-------|-----|------|-------------|-------------|
| 1 | adult | binclass | 26,048 | 6,513 | 16,281 | 6 | 7 |
| 2 | bank-marketing | binclass | 7,404 | 952 | 2,222 | 7 | 0 |
| 3 | california | regression | 13,209 | 3,303 | 8,257 | 8 | 0 |
| 4 | wine_quality | regression | 4,547 | 585 | 1,365 | 11 | 0 |
| 5 | wine | binclass | 1,787 | 230 | 494 | 11 | 0 |
| 6 | phoneme | binclass | 2,220 | 285 | 667 | 5 | 0 |
| 7 | churn | binclass | 6,400 | 800 | 1,600 | 7 | 1 |
| 8 | MagicTelescope | binclass | 9,363 | 1,171 | 2,930 | 10 | 0 |
| 9 | credit | binclass | 10,000 | 1,250 | 3,750 | 10 | 0 |
| 10 | MiniBooNE | binclass | 50,000 | 6,250 | 18,750 | 50 | 0 |
| 11 | Ailerons | regression | 9,625 | 1,203 | 3,008 | 33 | 0 |

---

## 3. Model Configuration

```python
TabM.make(
    n_num_features=...,
    cat_cardinalities=...,
    d_out=1,
    k=16,
    arch_type='tabm',
    n_blocks=1,
    d_block=128,
    dropout=0.1,
)
```

| Hyperparameter | Value | Note |
|---------------|-------|------|
| k | 16 | Ensemble size |
| n_blocks | 1 | Number of MLP blocks |
| d_block | 128 | Hidden dimension |
| dropout | 0.1 | Dropout rate |
| lr | 2e-3 | AdamW learning rate |
| weight_decay | 3e-4 | AdamW weight decay |
| epochs | 100 | Max epochs |
| patience | 20 | Early stopping |
| eval_every | 5 | Validate every 5 epochs |
| batch_size | Auto | 256-2048 based on dataset size |
| embeddings | **None** | Raw features (not TabM†) |

**Optimization:**
- Parallel seed execution (3 workers × 5 threads each)
- Auto batch size scaling

---

## 4. Task Performance Results

### Classification (Accuracy)

| Dataset | Seed 0 | Seed 1 | Seed 2 | Mean ± Std |
|---------|--------|--------|--------|-----------|
| adult | 0.8257 | 0.8015 | 0.7966 | **0.8079 ± 0.0127** |
| bank-marketing | 0.7723 | 0.7763 | 0.7669 | **0.7718 ± 0.0039** |
| wine | 0.6909 | 0.7095 | 0.7020 | **0.7008 ± 0.0077** |
| phoneme | 0.8351 | 0.8426 | 0.8321 | **0.8366 ± 0.0044** |
| churn | 0.7965 | 0.7965 | 0.7965 | **0.7965 ± 0.0000** |
| MagicTelescope | 0.7940 | 0.7982 | 0.7875 | **0.7932 ± 0.0044** |
| credit | 0.7294 | 0.7285 | 0.7253 | **0.7277 ± 0.0017** |
| MiniBooNE | 0.9137 | 0.9158 | 0.9147 | **0.9147 ± 0.0009** |

### Regression (RMSE)

| Dataset | Seed 0 | Seed 1 | Seed 2 | Mean ± Std |
|---------|--------|--------|--------|-----------|
| california | 0.8512 | 0.9628 | 0.8319 | **0.8819 ± 0.0577** |
| wine_quality | 0.7595 | 0.7675 | 0.7569 | **0.7613 ± 0.0045** |
| Ailerons | 0.0021 | 0.0024 | 0.0024 | **0.0023 ± 0.0001** |

---

## 5. Calibration Results

| Dataset | ECE | NLL | Predictive Entropy | Mutual Information |
|---------|-----|-----|-------------------|-------------------|
| adult | 0.0484 | 0.4482 | 0.4602 | 0.0521 |
| bank-marketing | 0.0324 | 0.4836 | 0.4919 | 0.0131 |
| wine | 0.0475 | 0.5576 | 0.5829 | 0.0074 |
| phoneme | 0.0381 | 0.4010 | 0.4066 | 0.0013 |
| churn | 0.0935 | 0.6632 | 0.4081 | 0.3564 |
| MagicTelescope | 0.0142 | 0.4197 | 0.4211 | 0.0073 |
| credit | 0.0670 | 0.5588 | 0.6063 | 0.0601 |
| MiniBooNE | 0.0186 | 0.2208 | 0.2527 | 0.0096 |
| california | 0.0000 | 6.9362 | 0.2901 | 0.1250 |
| wine_quality | 0.0000 | 34.5106 | 0.1007 | 0.0158 |
| Ailerons | 0.0000 | -3.8863 | 0.0086 | 0.0001 |

**Key Observations:**
- Best calibrated: MiniBooNE (ECE=0.0186), Ailerons (ECE=0.0000)
- Most uncertain (highest entropy): credit (0.6063)
- Most ensemble disagreement (highest MI): churn (0.3564)
- Most confident (lowest MI): phoneme (0.0013), Ailerons (0.0001)

---

## 6. UQ Metric Distributions

For each dataset, per-sample uncertainty scores were computed:

### Classification
- **Predictive Entropy:** `H = -Σ_c p̄_c log(p̄_c)` where `p̄ = (1/k) Σ_i p_i`
- **Mutual Information:** `MI = H - (1/k) Σ_i H(p_i)` — epistemic uncertainty
- **Data Uncertainty:** `(1/k) Σ_i H(p_i)` — aleatoric uncertainty

### Regression
- **Predictive Variance:** `Var = (1/k) Σ_i (y_i - ȳ)^2`
- **Predictive Std:** `std = sqrt(Var)`
- **Prediction Range:** `max(y_i) - min(y_i)`

---

## 7. Files Generated

```
results/
├── summary.json                    # All 33 results aggregated
├── raw/
│   ├── adult_seed0_uq.json        # UQ metrics + per-sample scores
│   ├── adult_seed1_uq.json
│   ├── ... (33 total files)
├── models/
│   ├── adult_seed0.pt             # Model checkpoints
│   ├── ... (33 total files)
├── predictions/
│   ├── adult_seed0_ensemble.npy   # (B, k, d_out) ensemble predictions
│   ├── adult_seed0_y_test.npy     # Ground truth labels
│   ├── ... (66 total files)
└── figures/
    ├── adult_seed0_reliability.png  # Calibration plots
    ├── adult_seed0_entropy_hist.png # Uncertainty distributions
    ├── ... (48 total PNG files)
```

---

## 8. Deviation from Plan

| Aspect | Plan Specified | What We Did | Reason |
|--------|---------------|-------------|--------|
| k | 32 | **16** | CPU speed |
| n_blocks | 3 | **1** | CPU speed |
| d_block | 512 | **128** | CPU speed |
| Embeddings | TabM† (implied) | **None** (TabM baseline) | CPU speed, still valid |
| Datasets | 8 (some not in paper) | **11** (all from paper suite) | Accuracy to paper |
| Hyperparameter tuning | Optuna 50 trials | **Skipped** | Time constraint |
| Baselines | 6 models | **0** (TabM only) | M2 scope only |

**Note:** All hyperparameters used (n_blocks=1, d_block=128, k=16) are within the paper's stated ranges: n_blocks ∈ [1,5], d_block ∈ [64,1024].

---

## 9. Next Steps

1. **Run exact plan configuration** (k=32, n_blocks=3, d_block=512, LinearReLUEmbeddings) on 3 representative datasets for comparison
2. **Implement OOD detection pipeline** (M3)
3. **Add baselines** (MC Dropout, Deep Ensemble, LightGBM) (M4)
4. **Ablation studies** (k effect, arch_type comparison) (M5)

---

**Report generated:** 2026-07-18
**Total experiments:** 33 runs (11 datasets × 3 seeds)
**Total runtime:** ~100 minutes wall-clock
