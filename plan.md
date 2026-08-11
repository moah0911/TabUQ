# TabM-UQ: Uncertainty-Aware Tabular Deep Learning via Parameter-Efficient Ensembling

**Project Plan (FYP — 8 months)**

---

## 1. Project Title

**TabM-UQ: Uncertainty-Aware Tabular Deep Learning via Parameter-Efficient Ensembling**

## 2. Abstract

Tabular data dominates high-stakes domains (medical diagnosis, credit scoring, fault detection), yet uncertainty quantification (UQ) and out-of-distribution (OOD) detection for tabular models remain under-studied. The recent **TabM** model (Gorishniy et al., ICLR 2025) introduces a parameter-efficient MLP ensemble that produces $k$ predictions per object, but its authors explicitly leave the evaluation of TabM for UQ and OOD detection as future work. This project addresses that gap: we (i) reproduce TabM on 5-10 tabular benchmarks, (ii) develop UQ scoring methods (predictive entropy, mutual information, variance) over TabM's $k$ predictions, (iii) evaluate OOD detection across covariate and semantic shifts, and (iv) benchmark against MC Dropout, deep ensembles, and temperature scaling. The result is a calibrated, interpretable tabular model that is competitive on accuracy while providing actionable uncertainty estimates.

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

1. **RQ1**: Can the $k$ predictions from a single TabM model be used directly as an uncertainty estimate, and how does it compare to a traditional deep ensemble of $k$ independent MLPs?
2. **RQ2**: How well does TabM detect OOD samples (covariate shift and semantic shift) compared to established baselines?
3. **RQ3**: Is TabM better-calibrated than prior tabular DL models and GBDTs out-of-the-box? Can calibration be further improved via post-hoc methods?
4. **RQ4**: How does ensemble size $k$ affect the trade-off between accuracy, uncertainty quality, and computational cost?

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

### 5.5 Baselines

| Baseline | Description |
|----------|-------------|
| **MLP (vanilla)** | Single MLP, no ensembling; softmax confidence as UQ |
| **MC Dropout** | $k=32$ forward passes with dropout enabled |
| **Deep Ensemble** | $k=32$ independently-trained MLPs |
| **Temperature Scaling** | Post-hoc calibration on a held-out set |
| **GBDT (LightGBM)** | Tree-based baseline |
| **TabR** | Retrieval-based tabular DL baseline (for accuracy reference) |

### 5.6 Ablations

- **Effect of $k$**: $k \in \{4, 8, 16, 32, 64\}$
- **Effect of arch_type**: `tabm` vs `tabm-mini` vs `tabm-packed`
- **UQ score choice**: entropy vs MI vs variance
- **Calibration method**: raw vs temperature scaling

## 6. Experimental Protocol

### 6.1 Training Protocol (Two-Phase)

#### Phase 1: CPU-Feasible Reproduction (M1)
For initial pipeline validation on CPU, we use a lightweight configuration:
- **Architecture**: `n_blocks=1`, `d_block=128`, `k=16`
- **Embeddings**: **None** (raw features) — this is the paper's default TabM baseline
- **Optimizer**: AdamW (`lr=2e-3`, `weight_decay=3e-4`)
- **Epochs**: 100 (early stopping with patience=20)
- **Batch size**: Auto-scaled (256-2048 based on dataset size)
- **Parallel**: 3 seeds in parallel, 5 CPU threads per worker
- **Runtime**: ~2-3 hours for all 11 datasets × 3 seeds

**Rationale:** This configuration is within the paper's hyperparameter ranges (`n_blocks` ∈ [1,5], `d_block` ∈ [64,1024]) and produces valid UQ metrics while remaining CPU-feasible. See `docs/reproducibility_report_small_config.md` for full results.

#### Phase 2: Exact Paper Configuration (M2 Validation)
For final comparison with the paper's reported metrics:
- **Architecture**: `n_blocks=3`, `d_block=512`, `k=32` (paper defaults)
- **Embeddings**: `LinearReLUEmbeddings` (TabM† variant)
- **Optimizer**: AdamW (`lr=2e-3`, `weight_decay=3e-4`)
- **Epochs**: 200 (early stopping with patience=20)
- **Seeds**: 3 seeds per dataset
- **Runtime**: ~40-60 min per seed for small datasets, longer for large

*Note: Due to CPU constraints, exact-config experiments are run on a subset of 3 representative datasets (e.g., phoneme, wine, churn) for validation, rather than all 11.*

#### Hyperparameter Tuning
- **M1**: Skipped (fixed lightweight defaults for speed)
- **M4+**: Optuna with 30-50 trials if time permits (GPU cluster or extended runtime)

### 6.2 Compute Budget

All experiments run on CPU (16-thread Intel i5-12500H).

**Phase 1 (Small Config, M1):**
- Small datasets (<3K samples): ~2-3 min per seed
- Medium datasets (3-15K): ~5-8 min per seed
- Large datasets (>15K): ~15-25 min per seed
- **Total**: ~2-3 hours wall-clock for 11 datasets × 3 seeds (with parallelization)

**Phase 2 (Exact Config, M2):**
- Small datasets: ~15-25 min per seed
- Medium datasets: ~30-45 min per seed
- Large datasets: ~60-90 min per seed
- **Total**: ~4-6 hours for 3 validation datasets × 3 seeds

**Overall project timeline:**
- M1 (Reproduction): ~1-2 days (including documentation)
- M2 (UQ pipeline): ~1-2 weeks
- M3 (OOD detection): ~2-3 weeks
- M4 (Baselines): ~2-3 weeks
- M5 (Ablations): ~2 weeks
- M6-M8 (Paper writing): ~3 months

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

## 8. Timeline (8 months) — Revised

| Month | Activities | Deliverable | Status |
|-------|-----------|-------------|--------|
| **M1** | Setup environment, clone TabM repo, install dependencies, reproduce 11 datasets with lightweight config (n_blocks=1, d_block=128, k=16), write reproducibility report | Reproducibility report | ✅ **COMPLETED** (Day 1) |
| **M2** | Implement UQ scoring functions (entropy, MI, variance); add calibration metrics (ECE, NLL); run exact config validation (n_blocks=3, d_block=512, k=32, LinearReLUEmbeddings) on 3 datasets | Working UQ pipeline + validation results | 🔄 **IN PROGRESS** |
| **M3** | Implement OOD detection pipeline (semantic/covariate/synthetic shifts); compute AUROC, AUPR, FPR@95TPR | OOD detection results table | ⏳ Pending |
| **M4** | Run baselines (MC Dropout, Deep Ensemble, Temperature Scaling, LightGBM); compare UQ quality | Comparison table + plots | ⏳ Pending |
| **M5** | Ablations (k effect, arch_type comparison, UQ score choice); reliability diagrams; final hyperparameter sweep if time permits | Ablation analysis | ⏳ Pending |
| **M6** | Draft paper (intro, method, experiments); internal review; camera-ready prep | First paper draft | ⏳ Pending |
| **M7** | Code cleanup; README; reproducibility check; supplementary materials | Submission-ready manuscript | ⏳ Pending |
| **M8** | Submit to chosen venue; address reviewer comments if applicable | Submitted paper | ⏳ Pending |

**Note:** M1 and initial M2 work were completed in a single day by using a CPU-optimized configuration. The exact-config validation (full paper defaults) is the current active task.
| **M3** | Implement OOD detection pipeline; run main experiments on all datasets | Initial OOD results table |
| **M4** | Run baselines (MC Dropout, Deep Ensemble, Temperature Scaling); compare | Comparison table + plots |
| **M5** | Ablations (k, arch_type, UQ score choice); reliability diagrams | Ablation analysis |
| **M6** | Draft paper (intro, method, experiments); internal review | First paper draft |
| **M7** | Camera-ready preparation; code cleanup; README; reproducibility check | Submission-ready manuscript |
| **M8** | Submit to chosen venue; address reviewer comments if applicable | Submitted paper |

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

We considered improving **TabR** (ICLR 2024) instead, but **TabM** is a better fit for the FYP scope:

| Aspect | TabR path | TabM path |
|--------|-----------|-----------|
| Architectural changes needed | High (new similarity/value modules) | None (post-hoc scoring) |
| Implementation complexity | High | Low |
| Novelty source | Architectural improvement | Application of future-work item |
| Paper clarity | "Better retrieval" | "First UQ/OOD evaluation of TabM" |
| CPU efficiency | ~minutes per dataset | ~seconds per dataset |
| Reproducibility | Complex (k-NN retrieval) | Simple (pure MLP) |

## Appendix B: Proposal Submission

The formal **project proposal document** for the first review (due in 2 days) should follow this plan, expanded to include:
- Detailed literature review (Section 3 expanded)
- Problem statement
- Methodology details
- Expected outcomes
- References
- Format: 3-5 pages, IEEE conference template (or as required by the institution)
