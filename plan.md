# TabM-UQ: Uncertainty-Aware Tabular Deep Learning via Parameter-Efficient Ensembling

**Project Plan (FYP — 8 months)**

---

## 1. Project Title

**TabM-UQ: Uncertainty-Aware Tabular Deep Learning via Parameter-Efficient Ensembling**

## 2. Abstract

Tabular data dominates high-stakes domains (medical diagnosis, credit scoring, fault detection), yet uncertainty quantification (UQ) and out-of-distribution (OOD) detection for tabular models remain under-studied. The recent **TabM** model (Gorishniy et al., ICLR 2025) introduces a parameter-efficient MLP ensemble that produces $k$ predictions per object, but its authors explicitly leave the evaluation of TabM for UQ and OOD detection as future work. This project addresses that gap using a **single model — TabM (vanilla, `k=32`, no embeddings)**. We (i) reproduce this TabM on 11 tabular benchmarks, (ii) develop UQ scoring methods (predictive entropy, mutual information, variance) over TabM's $k$ predictions, (iii) evaluate OOD detection across covariate and semantic shifts, and (iv) report calibration (ECE/NLL). The result is the first systematic UQ/OOD evaluation of TabM, providing actionable uncertainty estimates without modifying the architecture.

## 3. Background & Motivation

### 3.1 The Tabular Deep Learning Landscape

For decades, gradient-boosted decision trees (GBDTs) like XGBoost, LightGBM, and CatBoost have dominated tabular ML. Deep learning approaches (TabNet, FT-Transformer, TabR) have struggled to consistently outperform GBDTs. The recent **TabM** paper (ICLR 2025) breaks this trend using **parameter-efficient ensembling**: a single MLP with weight sharing implicitly represents $k$ sub-models, achieving state-of-the-art performance with manageable compute.

### 3.2 The Missing Piece: Uncertainty

Neither GBDTs nor prior tabular DL methods provide reliable uncertainty estimates. In high-stakes applications, a model's confidence is as important as its accuracy. The TabM paper concludes with this future-work statement:

> *"Another idea is to evaluate TabM for uncertainty estimation and out-of-distribution (OOD) detection on tabular data, which is inspired by works like Lakshminarayanan et al. (2017)."*

To date, **no follow-up work has addressed this**. This project fills that gap.

### 3.3 Key References

- **Gorishniy et al. (2024)**: "TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling" — ICLR 2025
- **Lakshminarayanan et al. (2017)**: "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles" — NeurIPS 2017
- **Wen et al. (2020)**: "BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning" — ICLR 2020 (the underlying technique TabM uses)
- **Gal & Ghahramani (2016)**: "Dropout as a Bayesian Approximation" — ICML 2016 (MC Dropout baseline)
- **Hendrycks & Gimpel (2017)**: "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks" — ICLR 2017 (OOD baseline metrics)
- **Guo et al. (2017)**: "On Calibration of Modern Neural Networks" — ICML 2017 (calibration metrics)

## 4. Research Questions

1. **RQ1**: Can the $k=32$ predictions from a single TabM (vanilla, no embeddings) be used directly as an uncertainty estimate?
2. **RQ2**: How well does TabM detect OOD samples (covariate shift and semantic shift) via its UQ scores?
3. **RQ3**: How well-calibrated is TabM out-of-the-box (ECE/NLL) and can post-hoc temperature scaling help?
4. **RQ4**: How does ensemble size $k$ affect the trade-off between accuracy, uncertainty quality, and computational cost? (deferred — single fixed $k=32$)

## 5. Proposed Method

### 5.1 Baseline Reproduction

Reproduce TabM on **11 datasets from the official paper's 50-dataset benchmark suite**, loaded from the authors' preprocessed `.npy` files (HuggingFace tarball). All datasets are CPU-feasible and directly comparable to the paper's reported metrics:

| # | Dataset | Task | Num Features | Cat Features | Train | Val | Test |
|---|---------|------|-------------|-------------|-------|-----|------|
| 1 | **adult** | Binary Cls. | 6 | 7 | 26,048 | 6,513 | 16,281 |
| 2 | **bank-marketing** | Binary Cls. | 7 | 0 | 7,404 | 952 | 2,222 |
| 3 | **california** | Regression | 8 | 0 | 13,209 | 3,303 | 8,257 |
| 4 | **wine_quality** | Regression | 11 | 0 | 4,547 | 585 | 1,365 |
| 5 | **wine** | Binary Cls. | 11 | 0 | 1,787 | 230 | 494 |
| 6 | **phoneme** | Binary Cls. | 5 | 0 | 2,220 | 285 | 667 |
| 7 | **churn** | Binary Cls. | 7 | 1 | 6,400 | 800 | 1,600 |
| 8 | **MagicTelescope** | Binary Cls. | 10 | 0 | 9,363 | 1,171 | 2,930 |
| 9 | **credit** | Binary Cls. | 10 | 0 | 10,000 | 1,250 | 3,750 |
| 10 | **MiniBooNE** | Binary Cls. | 50 | 0 | 50,000 | 6,250 | 18,750 |
| 11 | **Ailerons** | Regression | 33 | 0 | 9,625 | 1,203 | 3,008 |

**Task mix:** 8 binary classification + 2 regression + 1 small multiclass (`wine`).
**Total train samples:** ~128K across all datasets.

*Note: All datasets are confirmed present in the TabM paper's official 50-dataset benchmark suite (HuggingFace: `rototoHF/tabm-data`). Train/val/test splits are pre-defined by the paper authors.*

### 5.2 Uncertainty Quantification from TabM

TabM produces outputs of shape $(B, k, d_{out})$. We derive three uncertainty measures:

**Predictive Entropy** (classification):
$$H = -\sum_{c} \bar{p}_c \log \bar{p}_c, \quad \text{where } \bar{p}_c = \frac{1}{k}\sum_{i=1}^{k} p_{i,c}$$

**Mutual Information** (epistemic uncertainty, from Depeweg et al.):
$$MI = H - \frac{1}{k}\sum_{i=1}^{k} H(p_i)$$

**Predictive Variance** (regression):
$$\text{Var} = \frac{1}{k}\sum_{i=1}^{k} (y_i - \bar{y})^2$$

### 5.3 OOD Detection Pipeline

For each (ID, OOD) dataset pair:
1. Train TabM on ID dataset
2. Score all ID and OOD samples with the UQ measure
3. Compute AUROC, AUPR, FPR@95TPR
4. Compare against baselines

**OOD scenarios**:
- **Semantic shift**: train on Heart, test on Diabetes (different class semantics)
- **Covariate shift**: train on Adult, test on Bank Marketing (similar task, different feature distribution)
- **Synthetic shift**: add noise / feature corruption to ID test set

### 5.4 Calibration Evaluation

For each dataset and model:
- **Expected Calibration Error (ECE)**: bin predictions into $M=15$ bins, compute weighted average of |acc - conf|
- **Negative Log-Likelihood (NLL)**: probabilistic quality of the predictive distribution
- **Reliability diagrams**: visual comparison

### 5.5 Baselines — Deferred (Post-Acceptance Gate)

Single-model focus for Acceptance: **no baselines required**. Baselines below are planned for Phase 2 (after R1):

| Baseline | Description | Status |
|----------|-------------|--------|
| **MLP (vanilla)** | Single MLP, no ensembling; softmax confidence as UQ | Deferred |
| **MC Dropout** | $k=32$ forward passes with dropout enabled | Deferred |
| **Deep Ensemble** | $k=32$ independently-trained MLPs | Deferred |
| **Temperature Scaling** | Post-hoc calibration on a held-out set | Deferred |
| **GBDT (LightGBM)** | Tree-based baseline | Deferred |
| **TabR** | Retrieval-based tabular DL baseline | Out of scope |

### 5.6 Ablations — Deferred

Single-model focus: **no arch/embedding ablations for Acceptance**. Planned post-R1 (optional):

- **Effect of $k$**: $k \in \{16, 32\}$ (paper fixes $k=32$, not tuned)
- **UQ score choice**: entropy vs MI vs variance
- **Calibration method**: raw vs temperature scaling
- `arch_type` (`tabm-mini`/`packed`) and `TabM†` (with `PiecewiseLinearEmbeddings`) are **out of scope** — they are enhanced variants, not the default TabM.

## 6. Experimental Protocol

### 6.1 Training Protocol — Single Model (TabM default)

**Canonical model (paper-default, `TabM.make` fallback):**
`arch_type='tabm'`, `k=32` (fixed, not tuned — paper §3.3), `n_blocks=3`, `d_block=512`, `dropout=0.1`, `num_embeddings=None` (raw features + one-hot cats), `AdamW(lr=2e-3, wd=3e-4)`. Paper tunes `n_blocks∈[1,5]`, `d_block∈[64,1024]` per-dataset (`Table 6`); `3/512` is the untuned fallback — used here for CPU-feasible single-model evaluation without Optuna. `TabM†` (with `PiecewiseLinearEmbeddings`/`LinearReLUEmbeddings`, `Table 7`) is the enhanced variant and is **out of scope**.

#### Phase 1: CPU-Feasible Reproduction (M1 — Completed)
Lightweight pilot for pipeline validation:
- **Architecture**: `n_blocks=1`, `d_block=128`, `k=16`, `num_embeddings=None`
- **Optimizer**: AdamW (`lr=2e-3`, `weight_decay=3e-4`)
- **Epochs**: 100 (early stopping patience=20), `eval_every=5`
- **Batch size**: Auto-scaled (256-2048), Parallel: 3 seeds ×5 threads
- **Runtime (measured)**: ~1.5h wall for 11×3 (see `results_small_config/full_run.log`)
- **Result**: `docs/reproducibility_report_small_config.md`, `results_small_config/` 33/33 — within paper ranges, used only for validation.

#### Phase 2: Default TabM (M2 — Canonical, 11 Datasets)
- **Architecture**: `n_blocks=3`, `d_block=512`, `k=32`, `num_embeddings=None` (true default TabM)
- **Optimizer**: AdamW (`lr=2e-3`, `weight_decay=3e-4`)
- **Epochs**: 200 (early stopping patience=20), `eval_every=5`
- **Seeds**: 3 per dataset, **all 11 datasets**
- **Runtime (measured)**: ~30h wall for 11×3 (`results/k32_no_embeddings.log` — small 9–27m/seed, medium 58m–3.7h/seed, large 3.5–12.6h/seed e.g. MiniBooNE). `results/` 33/33 on disk.

#### Hyperparameter Tuning
- Skipped for Acceptance (fixed defaults). Optuna 30-50 trials deferred to Phase 2 stretch if needed.

### 6.2 Compute Budget (Measured)

All experiments run on CPU (16-thread Intel i5-12500H, 3 workers ×5 threads).

**Phase 1 (Pilot k16/1/128, M1 — measured):**
- Small (<3K): 9.5–27m / seed; Medium (3–15K): 58m–3.7h / seed; Large (>15K): 3.5–12.6h / seed
- **Total**: ~1.5h wall for 11×3 (`results_small_config/full_run.log`)

**Phase 2 (Default TabM k32/3/512/no-emb, M2 — measured):**
- Small: 9–27m / seed; Medium: 58m–3.7h / seed; Large: 3.5–12.6h / seed (MiniBooNE 12.6h/seed)
- **Total**: ~30h wall for 11×3 (`results/k32_no_embeddings.log`)

**Overall (single-model scope):**
- M1 repro + UQ pipeline: ~2 days (completed)
- M2 canonical TabM + calibration: ~2 days (completed, 30h)
- M3 OOD detection: ~2–3 weeks (next)

### 6.3 Reproducibility

- All code published as a public GitHub repository
- Hyperparameters logged via TOML files (matching TabM's convention)
- Final results reported in JSON and CSV with means and standard deviations
- Pretrained model checkpoints stored for verification

## 7. Expected Contributions

1. **First systematic UQ/OOD evaluation of TabM** (directly addresses paper's future work)
2. **A practical recipe** for turning TabM into a calibrated, uncertainty-aware model
3. **Open-source benchmark** of uncertainty methods on tabular data
4. **Publication-ready paper** for an IEEE or Springer conference (e.g., IEEE TNNLS, ESWA, NeurReps, PAKDD)

## 8. Timeline (8 months) — Single-Model

| Month | Activities | Deliverable | Status |
|-------|-----------|-------------|--------|
| **M1** | Setup, clone TabM repo, install deps, lightweight pilot `k16/1/128/no-emb` on 11 datasets, write reproducibility report | Reproducibility report | ✅ **COMPLETED** (`results_small_config/` 33/33, ~1.5h) |
| **M2** | UQ scoring (entropy/MI/variance) + calibration (ECE/NLL, 15 bins); **default TabM** `k32/3/512/no-emb` on **11 datasets** | UQ pipeline + `results/` 33/33 (~30h) | ✅ **COMPLETED** (on disk; `summary.json` stale) |
| **M3** | OOD detection pipeline (noise / feature-corruption / population shift); AUROC/AUPR/FPR@95TPR | OOD results table | ⏳ Next |
| **M4** | Baselines (MC Dropout/DeepEns/TempScale/LightGBM) — **deferred** | Comparison table | ⏳ Deferred |
| **M5** | Ablations (`k` 16 vs 32, UQ score) — **deferred** | Ablation plots | ⏳ Deferred |
| **M6** | Draft paper (intro, method, experiments) | First draft | ⏳ Pending |
| **M7** | Code cleanup, README, reproducibility check | Manuscript | ⏳ Pending |
| **M8** | Submit (workshop target: NeurIPS Tabular / ICLR Practical DL) | Submitted paper | ⏳ Pending |

**Note:** Single model only — `TabM` default (`k=32`, no embeddings). `TabM†` (+embeddings) and `arch_type` variants are out of scope. AW-TabM (adaptive weighting) is optional Phase-2 stretch, not required for Acceptance (R1 13 Jul 2026).

## 9. Risk Assessment & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| TabM reproducibility issues | Medium | Medium | Use official code; match hyperparameter configs exactly |
| OOD results not significantly better than baselines | Medium | High | Focus on calibration and per-sample UQ quality as alternative contributions |
| CPU compute too slow for hyperparameter search | Low | Medium | Reduce Optuna trials to 30; use smaller subset of datasets |
| Calibration improvements marginal | Medium | Medium | Combine with input perturbation-based UQ (e.g., test-time augmentation) |
| IEEE/Springer rejection | Medium | Low | Have backup venues (workshops, arXiv preprint) |

## 10. Tools & Dependencies

- **Python**: 3.11
- **Core**: PyTorch 2.x, scikit-learn, numpy, pandas
- **TabM**: `pip install tabm` (or local install from source)
- **Hyperparameter tuning**: Optuna
- **Visualization**: matplotlib, seaborn
- **GBDT baselines**: LightGBM, XGBoost
- **Datasets**: UCI ML Repository, OpenML, `rtdl_revisiting_models` (used by TabM)

## 11. Evaluation Metrics Summary

| Metric | Purpose |
|--------|---------|
| Accuracy / RMSE | Task performance |
| AUROC | OOD detection quality |
| AUPR | OOD detection (precision-recall) |
| FPR@95TPR | OOD detection at high recall |
| ECE | Calibration error |
| NLL | Probabilistic prediction quality |
| Inference time (CPU) | Practical efficiency |

## 12. Publication Venues (Target)

- **Tier 1**: IEEE TNNLS, Neurocomputing, ESWA (Expert Systems with Applications)
- **Tier 2**: PAKDD, ECML-PKDD, ACML, IDA (Intelligent Data Analysis)
- **Workshops**: NeurIPS Workshop on Tabular Learning, ICLR Workshop on Practical DL
- **Backup**: arXiv preprint with code repository

## 13. References

1. Gorishniy, Y., Kotelnikov, A., & Babenko, A. (2024). TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling. *ICLR 2025*.
2. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *NeurIPS 2017*.
3. Wen, Y., Tran, D., & Ba, J. (2020). BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning. *ICLR 2020*.
4. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation. *ICML 2016*.
5. Hendrycks, D., & Gimpel, K. (2017). A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks. *ICLR 2017*.
6. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML 2017*.
7. Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2022). Revisiting Deep Learning Models for Tabular Data. *NeurIPS 2021*.

---

## Appendix A: Why TabM (vs. TabR)?

We considered **TabR** (ICLR 2024), but **TabM** is a better fit — single vanilla `TabM k=32/no-emb` without architectural changes:

| Aspect | TabR path | **TabM (single-model) path** |
|--------|-----------|------------------------------|
| Architectural changes needed | High (new similarity/value modules) | None (post-hoc UQ scoring on `(B,k,d_out)`) |
| Implementation complexity | High | Low (~50 lines UQ) |
| Novelty source | Architectural improvement | First UQ/OOD evaluation of default TabM (paper future work §7) |
| Paper clarity | "Better retrieval" | "First UQ/OOD evaluation of TabM" |
| CPU efficiency | ~minutes per dataset | ~seconds–minutes per dataset |
| Reproducibility | Complex (k-NN retrieval) | Simple (pure MLP, `pip install tabm`) |
| Model scope | Multiple variants | **Single model only**: `TabM k=32/3/512/no-emb` (TabM† out of scope) |

## Appendix B: Proposal Submission

The formal **project proposal document** for the first review (due in 2 days) should follow this plan, expanded to include:
- Detailed literature review (Section 3 expanded)
- Problem statement
- Methodology details
- Expected outcomes
- References
- Format: 3-5 pages, IEEE conference template (or as required by the institution)
