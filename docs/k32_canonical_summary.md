# TabM-UQ Canonical Results — Default TabM k32/3/512/no-emb (M2)

**Date:** 2026-09-12
**Configuration:** `arch_type='tabm'`, `k=32` (fixed), `n_blocks=3`, `d_block=512`, `dropout=0.1`, `num_embeddings=None` (true default TabM, `tabm.py:1700` fallback), `AdamW lr2e-3 wd3e-4`, epochs 200 patience20 eval_every5
**Hardware:** 16-thread i5-12500H, 3 workers ×5 threads, ~30h wall (`results/k32_no_embeddings.log` 9m–12.6h/seed)
**Pipeline:** `uq/metrics.py:predictive_entropy/MI` (binclass) / `var unbiased=False` (regression), `uq/calibration.py:ECE(15)/NLL`
**Scope:** Single-model only — `TabM†`/`mini`/`packed` out of scope per single-model correction

---

## Per-Dataset (mean ± std, 3 seeds)

### Classification

| Dataset | Acc | ECE | NLL | Ent mean | MI mean | Notes |
|---------|-----|-----|-----|----------|---------|-------|
| adult | 0.8037±0.005 | 0.0498±0.050 | 0.469 | 0.501 | 0.0071 | seed1 ECE 0.123 outlier |
| bank-marketing | 0.7726±0.002 | 0.0291±0.001 | 0.481 | 0.483 | 0.0050 | healthy |
| wine | 0.7163±0.003 | 0.0432±0.002 | 0.522 | 0.532 | 0.0173 | healthy |
| phoneme | 0.8581±0.002 | 0.0249±0.005 | 0.329 | 0.324 | 0.0047 | healthy |
| churn | 0.7965±0.000 | 0.0004±0.000 | 0.505 | 0.506±0.000 | 0.0000 | **collapsed †** — constant pred, MI 0 |
| MagicTelescope | 0.8205±0.004 | 0.0205±0.002 | 0.387 | 0.387 | 0.0133 | healthy |
| credit | 0.4963±0.000 | 0.0057±0.002 | 0.693 | 0.693±0.000 | 0.0001 | **collapsed †** — random (0.496 vs pilot 0.727) |
| MiniBooNE | 0.9376±0.000 | 0.0105±0.002 | 0.165 | 0.185 | 0.0093 | healthy, best |

### Regression

| Dataset | RMSE | NLL | Var mean | Std mean | Notes |
|---------|------|-----|----------|----------|-------|
| california | 0.8434±0.020 | 16.8±4.1 | 0.048 | 0.177 | healthy, var unbiased |
| wine_quality | 0.8071±0.004 | 29.2±3.2 | 0.0126 | 0.108 | healthy |
| Ailerons | 0.00040±0.000 | 587.3±295 | 1.99e-10 | 6.0e-06 | **collapsed †** — var ~0, NLL 170–791 |

**Files:** `results/models/*.pt` 33, `predictions/*_ensemble.npy` 33+33, `raw/*_uq.json` 33, `figures/*reliability/entropy` 48 (regression no reliability by design), `summary.json` 33/33

## Macro Means

- **Primary (8 healthy, excludes 3 †):** compute mean over `adult,bank-marketing,california,wine_quality,wine,phoneme,MagicTelescope,MiniBooNE`
- **Full 11 (appendix):** includes collapsed — report with †

## Notes vs Small Pilot k16/1/128

- Pilot `results_small_config/` 33/33 ~1.5h: `adult 0.8079, MiniBooNE 0.9147, credit 0.727, churn MI 0.356 healthy, Ailerons var 7e-05 NLL -3.9` — k16 healthy on all 11
- k32 collapse correlates with depth/width on small/imbalanced (`churn 6.4k 20% pos, credit 10k, Ailerons targets ~0.0004`) with 815k params + missing `X_bin` (adult 1, churn 3) — not data corruption

## Next

UQ module ✅ complete (recomputed from `.npy`, unbiased var). Next P2 S1 synthetic OOD reuses these scores; do not mix collapsed † into primary OOD mean.
