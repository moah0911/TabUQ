# Thesis Fill — TabM-UQ (paste into official format, Times New Roman)

Scope locked: one TabM model only (vanilla default: 32 sub-models, 3 blocks, width 512, no feature embeddings), uncertainty quantification plus within-dataset out-of-distribution evaluation, 11 benchmarks. Written in third person; past tense for completed work. Replace every `[bracketed]` placeholder. Figures: Fig 1.1 (solution), Fig 4.1 (architecture), DFD figure (Ch 4), Fig A.1 (timeline) — 4 diagram files; remaining tables are typed directly as Word tables.

---

## Chapter 1 — Introduction

### 1.1 Background
- Tabular data dominates high-stakes domains such as medical diagnosis, credit scoring and industrial fault detection, where a wrong prediction carries real cost.
- For decades, gradient-boosted decision trees (GBDTs) have dominated tabular machine learning, while deep learning approaches have struggled to consistently outperform them.
- The TabM model of Gorishniy et al. [1] breaks this trend with parameter-efficient ensembling: a single multilayer perceptron with shared weights implicitly represents 32 sub-models and produces 32 predictions per object, achieving state-of-the-art accuracy among tabular deep learning models at manageable compute cost.
- TabM is publicly available as an official Python package (version 0.0.3) and trains on CPU, which makes it accessible for academic laboratories without accelerators.

### 1.2 Problem Statement
- Neither GBDTs nor prior tabular deep learning methods provide reliable uncertainty estimates, yet a model's confidence is as important as its accuracy in high-stakes applications.
- The TabM paper closes with an explicit future-work statement calling for TabM to be evaluated for uncertainty estimation and out-of-distribution (OOD) detection on tabular data [1].
- To date, no published follow-up work has addressed this gap: TabM's 32 predictions per object are mean-averaged for accuracy and their uncertainty content is discarded.
- Without uncertainty quantification, practitioners cannot distinguish confident predictions from guesswork, and models fail silently on shifted or corrupted inputs.

### 1.3 Motivation
- TabM is current (International Conference on Learning Representations, 2025) and practically relevant: it has been used in winning Kaggle competition solutions (including game-playing-strength prediction and survival prediction) as of mid-2025.
- Its 32 predictions per object are a natural ensemble for uncertainty, available at no extra training cost — unlike deep ensembles, which require 32 independent training runs [2].
- A CPU-feasible, single-model evaluation of these predictions for uncertainty and OOD detection is therefore both timely and practically useful, and it answers the original authors' open call directly.

### 1.4 Objectives
- O1. Reproduce the default TabM configuration (32 sub-models, no embeddings) on 11 benchmarks with 3 random seeds each — 33 training runs in total.
- O2. Derive per-sample uncertainty scores (predictive entropy, mutual information, predictive variance) from the 32 predictions of each trained model.
- O3. Evaluate within-dataset OOD detection across 7 synthetic input shifts using AUROC, AUPR and FPR at 95% TPR.
- O4. Measure calibration via Expected Calibration Error (15 bins) and Negative Log-Likelihood, supported by reliability diagrams and entropy histograms.

### 1.5 Proposed Solution Overview (Figure 1.1)
- The system wraps a frozen, already-trained TabM model with a post-hoc uncertainty-scoring layer of about 50 lines of code; the TabM model source is never modified.
- Model outputs of shape (batch, 32, outputs) are converted to probabilities, aggregated into entropy, mutual information and variance scores, checked for calibration with ECE/NLL, and reused as OOD scores on corrupted inputs.
- Training optimises the mean loss over the 32 members — not the loss of the mean prediction — which is the critical detail that keeps the members diverse enough for uncertainty scoring.

### 1.6 Scope of the Project (Table 1.1)
- IN scope: vanilla default TabM (32 sub-models, 3 blocks, width 512, no embeddings); 11 datasets with 3 seeds each; entropy, mutual information and variance scores; ECE and NLL calibration; within-dataset OOD with Gaussian noise (3 levels), feature masking (2 levels), column shuffling and input rescaling.
- OUT of scope: the embedding-enhanced TabM variant, lightweight/packed TabM builds, hyperparameter tuning, GBDT or deep-ensemble baselines, cross-dataset semantic shift, and retraining with additional binary features.

### 1.7 Contributions
1. First systematic uncertainty and OOD evaluation of the default TabM, directly addressing the future work of [1].
2. A practical recipe for turning TabM into an uncertainty-aware model without any retraining.
3. An open benchmark of 33 trained models and 245 OOD evaluation runs with all numerical artifacts preserved.

### 1.8 Organisation of the Report
- Chapter 2 surveys the literature; Chapter 3 specifies requirements; Chapter 4 presents the design; Chapter 5 describes implementation; Chapter 6 reports testing and results; Chapter 7 concludes and lists future scope.

### Summary
- This chapter introduced the problem (missing uncertainty for tabular models), the objectives (reproduce, quantify, detect, calibrate), the solution (post-hoc scoring on frozen TabM outputs) and the scope of TabM-UQ.

---

## Chapter 2 — Literature Survey

### 2.1 Survey Method
- Papers from 2016–2025 on tabular deep learning, ensemble uncertainty, OOD detection and calibration were reviewed, prioritising ICLR/NeurIPS/ICML venues.
- Each paper was recorded with its method, key result and limitation; the limitations column directly motivates the research gap in Section 2.6.

### 2.2 Background Concepts
- Parameter-efficient ensembling [3] shares most weights across ensemble members while keeping member-specific scaling factors, giving 32 predictions at near-single-model cost.
- Predictive entropy measures total uncertainty of the averaged prediction; mutual information (entropy of the mean minus mean of the entropies) isolates epistemic (model) uncertainty; predictive variance serves the same role for regression.
- OOD detection is scored threshold-free with AUROC/AUPR and operationally with FPR at 95% TPR [5]; calibration is measured with ECE and NLL [6].

### 2.3–2.4 Review (Table 2.1)
| Paper | Method | Key result | Limitation |
|---|---|---|---|
| Gorishniy et al. [1] | TabM with 32 sub-models | Best tabular deep learning accuracy on the official benchmark suite | No uncertainty or OOD evaluation |
| Lakshminarayanan et al. [2] | Deep ensembles | Gold-standard uncertainty estimation | 32 independent training runs required |
| Wen et al. [3] | BatchEnsemble | Efficient ensembling via shared weights plus rank-1 factors | Needs adaptation to tabular data |
| Gal and Ghahramani [4] | MC Dropout | Cheap single-model uncertainty via test-time dropout | Weaker calibration than ensembles |
| Hendrycks and Gimpel [5] | Softmax-confidence OOD baseline | Standard AUROC/AUPR evaluation protocol | Uses output confidence only, no ensemble signal |
| Guo et al. [6] | Temperature scaling and ECE | Standard calibration metrics and post-hoc fix | Post-hoc only; does not improve the model itself |

### 2.5 Existing Systems
- The official TabM package provides a model constructor with 32 sub-models as the default (not a tuned choice) and the paper's data suite provides 50 preprocessed benchmarks; neither ships any uncertainty or OOD scoring.
- Standard libraries used: a PyTorch-based training stack, scikit-learn for preprocessing and OOD metrics, and matplotlib for reliability diagrams.

### 2.6 Research Gap
- No published work evaluates TabM's 32 predictions as uncertainty or OOD scores; this project fills exactly that gap for the default (embedding-free) TabM configuration.

### Summary
- The survey establishes the method base (efficient ensembling, uncertainty scores, OOD protocol, calibration) and the unaddressed TabM-UQ gap.

---

## Chapter 3 — System Analysis and Requirements Specification

### 3.1 Existing vs Proposed System
- Existing: TabM trains for accuracy only; its 32 outputs are averaged and the ensemble structure is discarded.
- Proposed: the same frozen outputs are additionally scored for uncertainty, calibration and OOD detection without any retraining.

### 3.2 Feasibility
- Technical: CPU-only PyTorch stack on a 16-thread workstation; measured total training time about 30 hours for all 33 runs.
- Operational: frozen trained models are reused for all OOD scoring (minutes per shift), so no additional training is ever needed for analysis.
- Schedule: the 8-month plan was met for uncertainty plus OOD work; external baselines were formally deferred as out of scope.

### 3.3–3.5 Users, Functional and Non-Functional Requirements
- Users: tabular machine-learning practitioners who need confidence values attached to predictions.
- FR1: train the default TabM on all 11 datasets with 3 seeds (measurable: 33 saved models).
- FR2: compute entropy, mutual information and variance per test sample (measurable: 33 uncertainty result files).
- FR3: score 7 within-dataset shifts with AUROC, AUPR and FPR at 95% TPR (measurable: 245 OOD result files).
- FR4: report ECE with 15 bins and NLL with reliability plots (measurable: 48 figures).
- NFR1: CPU-only execution. NFR2: reproducibility (fixed seeds 0–2, logged configurations). NFR3: honest reporting (failed runs flagged in an appendix, not hidden).

### 3.6–3.8 Hardware, Software, Dataset
- Hardware: Intel i5-12500H processor, 16 threads, 16 GB RAM; no GPU used at any stage.
- Software (Table 3.8): Python 3.11 with a virtual environment; PyTorch 2.12 (CPU build); official TabM package 0.0.3; numerical-embeddings helper library (installed but not activated, since the embedding-free variant is used); scikit-learn 1.9 for encoding and metrics; matplotlib 3.10 for plots; 3 parallel workers with 5 threads each.
- Dataset: 11 benchmarks from the official TabM data suite (about 1 GB archive, 50 sets available, 11 used): adult 26,048 / bank-marketing 7,404 / california 13,209 / wine_quality 4,547 / wine 1,787 / phoneme 2,220 / churn 6,400 / MagicTelescope 9,363 / credit 10,000 / MiniBooNE 50,000 / Ailerons 9,625 training samples.
- Note (footnoted limitation): binary indicator features present in two datasets (1 in adult, 3 in churn) were excluded by the data loader; in-distribution and OOD inputs use identical features, so AUROC comparisons remain valid.

### Summary
- Requirements, hardware/software stack and data are fixed, counted and measurable.

---

## Chapter 4 — System Design

### 4.1 Design Method
- Post-hoc scoring design: no architecture search was performed; the model is used exactly in its published default configuration.

### 4.2 System Architecture (Figure 4.1)
- Six stages: (1) benchmark archive → (2) data-loading module → (3) training module → (4) stored models with ensemble predictions → (5) uncertainty and calibration scorer → (6) OOD scorer.
- Each stage lists its inputs and outputs; stored artifacts total 33 models, 33 ensemble prediction files, 33 uncertainty result files, 48 figures and 245 OOD result files.

### 4.3 Module Design
| Module | Function | Input | Output |
|---|---|---|---|
| Data loader | Reads NumPy data splits, encodes string categories | Preprocessed dataset splits | Numeric/categorical arrays with metadata |
| Model wrapper | Builds and trains default TabM (32 sub-models, 3 blocks, width 512) | Feature arrays | Raw ensemble outputs of shape (batch, 32, outputs); saved model |
| Uncertainty scorer | Converts outputs to probabilities, aggregates scores | Ensemble probabilities | Entropy, mutual information (classification) / variance (regression) |
| Calibration scorer | Compares mean predictions with labels | Labels, mean probabilities | ECE (15 bins), NLL, reliability plot |
| OOD evaluator | Corrupts inputs deterministically, re-scores frozen model | Test arrays, saved model | AUROC/AUPR/FPR95 per shift |

### 4.4–4.6 Data Flow, DFD, Data Storage (DFD figure)
- Level 0: dataset archive → training process → scoring processes → result stores.
- Level 1 expands scoring into uncertainty, calibration and OOD branches with four data stores (models, uncertainty results, figures, OOD results).
- No relational database is used; the data store is a versioned file archive. There is no end-user interface; the interface is a command-line tool with three commands (train; evaluate uncertainty; evaluate OOD with a corruption-type option).

### 4.7 Algorithm and Model
- Training minimises the mean loss over the 32 members (Equation 4.1), not the loss of the mean prediction — the critical TabM detail that preserves member diversity.
- Uncertainty: entropy H of the mean prediction; mutual information = H minus mean of member entropies; variance = mean squared deviation from the member mean using population (not sample) variance.
- OOD corruptions: Gaussian noise at 3 intensities; feature masking at 2 ratios; column shuffling; uniform rescaling. Scores feed standard AUROC/AUPR and ROC-curve (FPR at 95% TPR) computations.

### Summary
- The design reuses one frozen model for all uncertainty and OOD analysis through five specialised modules.

---

## Chapter 5 — Implementation

### 5.1 Environment
- Python 3.11 in an isolated virtual environment; all dependencies installed from a pinned requirements file.
- CPU-only PyTorch; official TabM package; scikit-learn; matplotlib; dataset archive extracted once to a local data folder.

### 5.2 Project Structure
- Data-loading module (dataset registry with 11 entries; ordinal encoding of string categories with unknown-category handling).
- Model-wrapper module (construction, optimisation, checkpoint saving, ensemble prediction).
- Uncertainty-metrics module (classification and regression scoring functions).
- Calibration module (ECE, NLL, reliability and histogram plots).
- Three pipeline scripts: single-dataset trainer; post-training uncertainty evaluator; OOD evaluator with corruption-type option; plus a parallel experiment runner (3 workers).

### 5.3 Module-wise Implementation
- Data loader: maps 11 dataset names to files; encodes string categories to integers, mapping unseen test categories to a reserved code.
- Model wrapper: constructs TabM with 32 sub-models, 3 blocks, width 512; AdamW optimiser (learning rate 2e-3, weight decay 3e-4); batch size auto-scaled 256–2048 by training-set size; early stopping with patience 20 evaluated every 5 epochs; ensemble prediction returns raw (batch, 32, outputs) logits.
- Uncertainty scorer: sigmoid/softmax to probabilities; entropy, mutual information, variance and standard deviation per sample.
- OOD evaluator: deterministic seeded corruptions (seed formula fixed); per-seed result files plus aggregated summaries.
- Runner: 3 parallel workers with 5 threads each; automatic skipping of completed runs; aggregated summary over all runs.

### 5.4 Novel Contribution
- The contribution is methodological-evaluative rather than architectural: the first protocol that turns TabM's native 32 outputs into calibrated uncertainty and within-dataset OOD scores without touching the model.

### 5.5–5.6 Integration and Deployment
- Integration means frozen trained models are reused across uncertainty and OOD analysis with zero retraining.
- Deployment is a CPU workstation; the demo chain (load → predict → uncertainty → reliability plot) is documented in Appendix D.

### 5.7 Challenges and Solutions
- Collapse of 3 of 11 full-size runs (churn: zero mutual information; credit: near-random accuracy; Ailerons: near-zero variance) → excluded from primary means, retained with a dagger (†) mark in the appendix with honest discussion.
- Newer PyTorch versions changed checkpoint-loading defaults → explicit trusted-load flag for local checkpoints.
- Stale small-config defaults found in older scripts and notes → reconciled to the canonical 32-sub-model configuration everywhere.

### Summary
- All modules are implemented and integrated on frozen artifacts; challenges were resolved without changing the locked scope.

---

## Chapter 6 — Testing, Results and Discussion

### 6.1–6.2 Strategy and Environment
- Verification by recomputation (all 33 uncertainty evaluations rerun from saved ensembles), artifact counts, and single-dataset smoke runs; 16-thread CPU workstation; no automated pytest suite (stated honestly).

### 6.3–6.5 Test Tables (Tables 6.3–6.5)
- Unit checks: imports of wrapper, metrics and calibration modules — pass.
- Integration: uncertainty recomputation 33/33 with matching summaries — pass.
- System: OOD scoring for 7 shifts × 33 runs = 245 result files — pass.

### 6.6 Metrics (Equations)
- Accuracy and RMSE for task performance; ECE with 15 bins and NLL for calibration; entropy, mutual information and variance for uncertainty; AUROC, AUPR and FPR at 95% TPR for OOD detection. Every symbol is defined after its equation.

### 6.7 Experimental Setup
- Default TabM (32 sub-models, 3 blocks, width 512, no embeddings), seeds 0–2, all 11 datasets; OOD scored on frozen models with deterministic corruptions.

### 6.8 Results
- Table 6.1, uncertainty (8 healthy datasets primary; 3 marked † in appendix): MiniBooNE accuracy 0.9376 with ECE 0.010 (best); phoneme 0.8581/0.0249; adult 0.8037/0.0498; california RMSE 0.8434; wine_quality RMSE 0.8071.
- Table 6.2, OOD macro AUROC over 8 healthy datasets: noise σ0.5 → 0.586; σ1.0 → 0.594; σ2.0 → 0.590; mask 0.25 → 0.606; mask 0.5 → 0.641 (best); shuffle → 0.558; scale → 0.530.
- Peaks: regression variance reaches 0.942 AUROC under strong noise (california, wine_quality); classification peaks are MiniBooNE mask 0.5 at 0.819 and adult noise at 0.744.
- Figures: reliability diagrams and entropy histograms per run; demo figure uses the phoneme dataset.

### 6.9 Comparison with Existing Work
- Pilot small configuration (16 sub-models, all 11 healthy, e.g. credit 0.727, churn mutual information 0.356) versus canonical 32-sub-model configuration (3 collapsed) shows capacity-driven collapse on small or imbalanced sets, not data corruption.
- No external baselines (deferred per scope); comparison is within-model (corruption types; small versus canonical configuration).

### 6.10 Objectives vs Outcomes (Table 6.10)
| Objective | Evidence | Status |
|---|---|---|
| O1 reproduce 33/33 | 33 trained models | ACHIEVED |
| O2 uncertainty scores | 8-dataset primary table | ACHIEVED |
| O3 OOD over 7 shifts | 245 result files | ACHIEVED |
| O4 calibration + plots | ECE/NLL values + 48 figures | ACHIEVED |

### 6.11 Discussion
- Regression variance detects shifts reliably and strengthens with corruption intensity; classification entropy is weak or inverted under mild noise (two small datasets below 0.5) but improves with destructive masking; uniform 1.3× rescaling is too mild for classification (macro 0.530).
- The honest weak-entropy finding bounds TabM's out-of-box OOD use and is the key practitioner takeaway.

### Summary
- All four objectives are met with measured evidence; limitations are flagged, not hidden.

---

## Chapter 7 — Conclusion and Future Scope

### 7.1 Summary
- Default TabM was reproduced on 11 benchmarks (33 runs, about 30 CPU hours); its 32 predictions were scored for uncertainty and calibration; 7 within-dataset OOD shifts were evaluated (245 runs) reusing frozen models.

### 7.2 Key Contributions
- As listed in Section 1.7: first evaluation, practical recipe, open benchmark.

### 7.3 Limitations
- Three full-size runs collapsed (constant or random predictor, near-zero variance); binary indicator features were excluded from two datasets; no cross-dataset semantic shift; no external baselines; uniform rescaling too mild.

### 7.4–7.5 Impact and Future Scope
- Societal: calibrated tabular uncertainty enables safer triage in medical screening and credit decisions.
- Future: retraining with binary features included; stronger shift design; MC Dropout and gradient-boosting baselines; learned (e.g. gating-based) combination of the 32 members.

### 7.6 Concluding Remarks
- TabM's free ensemble yields useful uncertainty for regression and large classification sets, but its entropy alone is an unreliable OOD detector for small tabular sets — the bound practitioners need.

---

## Appendices

### Appendix A — Plan, Timeline, Review Record
- Timeline (Fig A.1): M1 pilot (~1.5 h, done); M2 full-size training plus uncertainty (~30 h, 33/33, done); M3 OOD over 7 shifts (245 files, done); paper draft (workshop target, pending); thesis plus viva (pending).
- Table A.2, review feedback record:
| Remark | Action taken | Evidence |
|---|---|---|
| Single-model scope? | Fixed to vanilla 32-sub-model, embedding-free; enhanced variant out | Scope section of thesis Chapters 1 and 3 |
| Configuration exact? | 32 sub-models fixed; 3 blocks / width 512 confirmed | Official TabM paper default configuration plus training logs |
| Runtime claim of 2 hours? | Corrected to measured 30 hours | Measured training log, wall-clock time |
| External baselines required? | Formally deferred (workshop needs uncertainty plus OOD only) | Scope section plus workshop requirements |
- Table A.3, chapter approvals: guide sign-off recorded per chapter before the next review.
- Table A.4, roles: solo project — data loading, model wrapper, uncertainty metrics, calibration, OOD pipeline, results, plots and documentation by a single owner ([Your Name], [Roll No]); guide [Guide Name]; batch [Batch].

### Appendix B — Research Paper
- Title, authors, target workshop (tabular machine learning workshop at NeurIPS, or practical deep learning workshop at ICLR), status, abstract draft (first uncertainty/OOD study of default TabM; mask-0.5 macro AUROC 0.641; regression strong-noise AUROC 0.942), proof of submission when available.

### Appendix C — Originality Report
- Similarity-check summary page (below 15%), tool name, date, percentage (attach at submission).

### Appendix D — User Manual and Installation Guide
1. Create a Python 3.11 virtual environment and install the pinned dependencies.
2. Extract the benchmark data archive into the local data folder (about 1 GB, 11 datasets used).
3. Train: run the parallel experiment runner with the canonical configuration (32 sub-models, 3 blocks, width 512, 200 epochs, 3 workers).
4. Uncertainty: run the uncertainty evaluator for each dataset-seed pair (writes accuracy/ECE/NLL plus reliability plot).
5. OOD: run the OOD evaluator for each corruption type (noise, mask, shuffle, scale).
6. Screenshots: one reliability diagram and one terminal listing of a result file.

### Appendix E — Source Code
- Module table (data loader; model wrapper; uncertainty metrics; calibration; trainer; uncertainty evaluator; OOD evaluator; parallel runner) with ≤25-line key listings for the model construction call, the scoring formulas and one corruption function; full listing attached as a printout.

### Appendix F — PO Mapping
- Keep exactly as given (3/3/3/3/3/2/3/3/3/2/3); remove only the guidance box.

---

## References (IEEE format — all 6 cited works)

[1] Y. Gorishniy, A. Kotelnikov, and A. Babenko, "TabM: Advancing tabular deep learning with parameter-efficient ensembling," in Proc. Int. Conf. Learn. Representations (ICLR), Singapore, Singapore, 2025. [Online]. Available: https://arxiv.org/abs/2410.24210

[2] B. Lakshminarayanan, A. Pritzel, and C. Blundell, "Simple and scalable predictive uncertainty estimation using deep ensembles," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 30, Long Beach, CA, USA, 2017, pp. 6402–6413.

[3] Y. Wen, D. Tran, and J. Ba, "BatchEnsemble: An alternative approach to efficient ensemble and lifelong learning," in Proc. Int. Conf. Learn. Representations (ICLR), Addis Ababa, Ethiopia, 2020.

[4] Y. Gal and Z. Ghahramani, "Dropout as a Bayesian approximation: Representing model uncertainty in deep learning," in Proc. Int. Conf. Mach. Learn. (ICML), vol. 48, New York, NY, USA, 2016, pp. 1050–1059.

[5] D. Hendrycks and K. Gimpel, "A baseline for detecting misclassified and out-of-distribution examples in neural networks," in Proc. Int. Conf. Learn. Representations (ICLR), Toulon, France, 2017.

[6] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in Proc. Int. Conf. Mach. Learn. (ICML), vol. 70, Sydney, Australia, 2017, pp. 1321–1330.
