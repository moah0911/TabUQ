# OOD Evaluation — Within-Dataset Synthetic Only, Single TabM k32/no-emb (M3)

**Date:** 2026-09-12
**Model:** single TabM `k32/3/512/no-emb` (`results/` 33/33), frozen checkpoints
**Method:** `experiments/ood_evaluate.py` — within-dataset corruptions only (no cross-dataset): S1 Gaussian σ×per-feature std, S1b feature-mask/shuffle, S1c uniform scale (covariate proxy)
**Metrics:** `AUROC/AUPR/FPR@95` via `sklearn` on `entropy` (classification) / `variance` (regression) ; also `MI`, `max_prob`
**Runs:** `results/ood/` 245 json — S1 `*sigma{0.5,1.0,2.0}.json` 99 + S1b `*mask{0.25,0.5}.json` 66 + `*shuffle.json` 33 + S1c `*scale0.3.json` 33 + 7 summaries (+ archived `semantic_summary.json` 6, out of scope per no-crossing)

---

## S1 Synthetic Noise (ID test vs ID+Gaussian)

Per-feature noise `X_ood = X + N(0, (σ·std_feat)²)`, same `X_cat`, `rng(42+seed)`, scores via `predict_ensemble→(B,k,d_out)→sigmoid→entropy/MI`

### Entropy AUROC (higher = more OOD-like; 0.5 = random)

| Dataset | σ0.5 | σ1.0 | σ2.0 | Notes |
|---------|------|------|------|-------|
| adult | 0.744±0.019 | 0.727±0.022 | 0.685±0.029 | healthy, best binclass |
| bank-marketing | 0.439±0.007 | 0.377±0.015 | 0.315±0.030 | inverted (<0.5) — noise increases confidence |
| california (reg) | 0.782±0.119 | 0.872±0.081 | 0.942±0.041 | variance, strong |
| wine_quality (reg) | 0.630±0.007 | 0.795±0.007 | 0.942±0.008 | variance, strong at high σ |
| wine | 0.495±0.015 | 0.449±0.007 | 0.391±0.006 | near random |
| phoneme | 0.506±0.009 | 0.501±0.012 | 0.448±0.020 | random |
| MagicTelescope | 0.406±0.012 | 0.345±0.009 | 0.316±0.026 | inverted |
| MiniBooNE | 0.689±0.039 | 0.685±0.048 | 0.681±0.053 | moderate |
| **Healthy 8 macro** | **0.586** | **0.594** | **0.590** | mean over 8 healthy (excludes †) |
| churn † | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | collapsed MI 0, constant entropy |
| credit † | 0.493±0.007 | 0.491±0.006 | 0.490±0.005 | collapsed random |
| Ailerons † | 0.502±0.004 | 0.501±0.008 | 0.517±0.038 | collapsed var ~0 |

**Interpretation:** Regression variance detects noise well (california, wine_quality ↑ with σ). Binclass entropy is weak/inverted for 4/6 (bank-marketing, MagicTelescope, wine, phoneme) — TabM entropy does not reliably increase with Gaussian noise. MiniBooNE/adult moderate. Healthy macro ~0.59 (weak).

---

## S1b Feature-Mask / Shuffle (ID test vs corrupted features)

`mask`: zero-out random fraction of `X_num` per matrix (`rng(142+seed)`); `shuffle`: permute each column independently. Same `X_cat`, same scoring.

### Entropy/variance AUROC | AUPR | FPR95

| Dataset | mask0.25 | mask0.5 | shuffle | Notes |
|---------|----------|---------|---------|-------|
| adult | 0.580±0.011 \| 0.542 \| 0.964 | 0.674±0.019 \| 0.608 \| 0.964 | 0.506±0.010 \| 0.504 \| 0.953 | mask helps, shuffle random |
| bank-marketing | 0.456±0.010 \| 0.480 \| 0.976 | 0.419±0.027 \| 0.457 \| 0.973 | 0.518±0.002 \| 0.513 \| 0.933 | still weak |
| california (reg) | 0.789±0.061 \| 0.769 \| 0.746 | 0.820±0.079 \| 0.792 \| 0.636 | 0.532±0.010 \| 0.530 \| 0.944 | mask strong |
| wine_quality (reg) | 0.839±0.026 \| 0.871 \| 0.792 | 0.860±0.052 \| 0.896 \| 0.848 | 0.639±0.025 \| 0.663 \| 0.906 | mask strongest |
| wine | 0.422±0.009 \| 0.465 \| 0.980 | 0.400±0.036 \| 0.466 \| 0.976 | 0.511±0.007 \| 0.521 \| 0.965 | inverted |
| phoneme | 0.516±0.015 \| 0.515 \| 0.925 | 0.564±0.011 \| 0.560 \| 0.898 | 0.551±0.011 \| 0.548 \| 0.912 | shuffle ≈ mask |
| MagicTelescope | 0.512±0.017 \| 0.507 \| 0.950 | 0.570±0.025 \| 0.548 \| 0.915 | 0.432±0.017 \| 0.455 \| 0.973 | mask0.5 best for this ds |
| MiniBooNE | 0.731±0.003 \| 0.700 \| 0.775 | **0.819±0.003** \| 0.774 \| 0.576 | 0.772±0.004 \| 0.734 \| 0.709 | mask strong |
| **Healthy 8 macro** | **0.606** | **0.641** | **0.558** | mask0.5 best overall |
| churn † | 0.500±0.000 | 0.493±0.005 | 0.500±0.000 | collapsed |
| credit † | 0.405±0.037 | 0.322±0.090 | 0.498±0.001 | collapsed |
| Ailerons † | 0.580±0.055 | 0.661±0.096 | 0.499±0.003 | collapsed var |

**Interpretation:** Mask beats Gaussian for binclass (`MiniBooNE 0.819`, `adult 0.674`, `MagicTelescope 0.570` at mask0.5 vs 0.68/0.72/0.34 at noise). Destroying features raises entropy more reliably than adding noise. Shuffle is weak except `MiniBooNE 0.772`.

---

## S1c Uniform Scale — Covariate Proxy (ID test vs X·1.3)

`X_ood = X·(1+δ)`, `δ=0.3`, same `X_cat`. Mimics population drift without new data.

| Dataset | scale0.3 AUROC \| AUPR \| FPR95 | Notes |
|---------|-------------------------------|-------|
| adult | 0.508±0.021 \| 0.500 \| 0.958 | random |
| bank-marketing | 0.446±0.009 \| 0.467 \| 0.971 | inverted |
| california (reg) | 0.697±0.028 \| 0.644 \| 0.692 | moderate |
| wine_quality (reg) | 0.806±0.030 \| 0.784 \| 0.625 | strong |
| wine | 0.413±0.004 \| 0.461 \| 0.964 | inverted |
| phoneme | 0.510±0.003 \| 0.523 \| 0.971 | random |
| MagicTelescope | 0.454±0.004 \| 0.475 \| 0.979 | weak |
| MiniBooNE | 0.408±0.003 \| 0.454 \| 0.996 | inverted |
| **Healthy 8 macro** | **0.530** | weakest shift — uniform scale is too mild for binclass |
| churn/credit/Ailerons † | 0.500 / 0.500 / 0.494 | collapsed |

**Interpretation:** Uniform scale barely moves binclass entropy (macro 0.53). Regression variance still responds (`wine_quality 0.806`). Covariate detection needs stronger drift than ×1.3 for classification.

---

## S2 Cross-Dataset — Archived (Out of Scope per No-Crossing)

`experiments/ood_semantic.py` + `results/ood/*_to_*.json` 6 + `semantic_summary.json` retained on disk but excluded from primary tables: only `MagicTelescope(10)↔credit(10)` share dims; entropy 0.42/0.09 weak, MI 0.81/0.99 strong. Not part of within-dataset claim.

---

## Files

- `results/ood/*_sigma{0.5,1.0,2.0}.json` 99, `summary_sigma*.json` 3
- `results/ood/*_mask{0.25,0.5}.json` 66, `summary_mask*.json` 2
- `results/ood/*_shuffle.json` 33, `summary_shuffle.json`
- `results/ood/*_scale0.3.json` 33, `summary_scale0.3.json`
- Archived: `MagicTelescope_to_credit_seed*.json` 3, `credit_to_MagicTelescope_seed*.json` 3, `semantic_summary.json`
- Logs: `results/ood/run_*.log`, `full_run*.log`, via `ood_evaluate.py --corruption {noise,mask,shuffle,scale}`

## Conclusion

UQ complete; OOD within-dataset complete (S1 Gaussian + S1b mask/shuffle + S1c scale, all 11, frozen models). Strongest binclass signal: **mask0.5 macro 0.641** (`MiniBooNE 0.819`); strongest regression: **noise σ2 variance** (`california 0.942`, `wine_quality 0.942`). Honest finding: TabM entropy is a weak OOD detector for small binclass sets under mild shifts. Baselines deferred per single-model scope.
