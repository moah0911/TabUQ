# OOD Evaluation — Synthetic Noise + Semantic (M3)

**Date:** 2026-09-12
**Model:** single TabM `k32/3/512/no-emb` (`results/` 33/33), frozen checkpoints
**Method:** `experiments/ood_evaluate.py` (S1 Gaussian σ×per-feature std) + `ood_semantic.py` (S2 10-dim pair)
**Metrics:** `AUROC/AUPR/FPR@95` via `sklearn` on `entropy` (classification) / `variance` (regression) ; also `MI`, `max_prob`
**Runs:** `results/ood/*_sigma{0.5,1.0,2.0}.json` 99 + 3 summaries + `semantic_summary.json` 6

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

### Next
Try `MI` or `max_prob` as primary: pilot shows `MI` similar weak, `max_prob` inverted — need feature-corruption (mask/shuffle) not just Gaussian; defer to feature-mask S1b.

---

## S2 Semantic (dim-matched 10-dim)

Only `MagicTelescope(10) ↔ credit(10)` share dims (plan Adult→BankMarketing infeasible).

| ID → OOD | Entropy AUROC | MI AUROC | MaxProb AUROC | Notes |
|----------|---------------|----------|---------------|-------|
| MagicTelescope → credit | 0.428±0.276 | **0.814±0.263** | ~0.5 | entropy near random, **MI strong** |
| credit → MagicTelescope | 0.095±0.075 | **0.993±0.003** | ~0.9 | ID collapsed (credit 0.496), entropy inverted, MI near perfect (detects collapse) |

**Interpretation:** Entropy fails for semantic, MI succeeds — ensemble disagreement captures semantic shift while mean confidence does not. Credit-as-ID result is degenerate (ID collapsed) but MI still separates.

---

## Files

- `results/ood/*_sigma{0.5,1.0,2.0}.json` 99, `summary_sigma*.json` 3
- `results/ood/MagicTelescope_to_credit_seed*.json` 3, `credit_to_MagicTelescope_seed*.json` 3, `semantic_summary.json`
- Logs: `results/ood/run_sigma*.log`, `full_run.log`, `semantic run` via `ood_semantic.py`

## Limitations & Next S1b

Gaussian noise is not predictive for binclass TabM entropy — recommend **feature-mask/shuffle** corruption (zero-out 25/50% cols) as S1b for stronger binclass OOD; keep Gaussian for regression only.

UQ module complete; OOD S1 synthetic complete (weak binclass result is honest finding), S2 complete. Next: aggregate figures optional, then baselines (deferred per single-model scope).
