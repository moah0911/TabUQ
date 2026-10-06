# PPT Fill — TabM-UQ (paste into BATCH-9A.pptx v3.0)

Every slide below is complete on its own. Copy the bullets onto the slide, draw the table shown
(if any) as a native PowerPoint table, and insert the named image (if any) with the given caption.
Blank underlines ____________ are only for personal details (names, rolls, email) — fill by hand.

Project title (all covers and title slides): TabM-UQ: Uncertainty-Aware Tabular Deep Learning
via Parameter-Efficient Ensembling. Short form: TabM-UQ.

---

## Deck R1 — Project Acceptance Review (PPT slides 4–14)

### Cover (slide 4)

- Title: TabM-UQ: Uncertainty-Aware Tabular Deep Learning via Parameter-Efficient Ensembling
- Short form: TabM-UQ
- Team table on the cover: ____________ (name, roll), ____________ (name, roll), ____________ (name, roll)
- Guide: ____________ , ____________ (designation), Department of Computer Science and Engineering
- Batch: ____________ , B.Tech, MVGR College of Engineering (Autonomous)
- Status line: Phase 1, Review 1, Acceptance Gate, 13 July 2026

### Slide 1/10 — Title Slide (slide 5)

- TabM-UQ: Uncertainty-Aware Tabular Deep Learning via Parameter-Efficient Ensembling
- One default TabM model with 32 sub-models, studied on CPU with no accelerator
- 11 benchmark datasets times 3 random seeds gives 33 complete training runs
- Presented by ____________ (roll ____________), guided by ____________
- Phase 1, Review 1, Acceptance Gate, 13 July 2026
- Visual: text only, the cover already carries the title and team table

### Slide 2/10 — Problem Statement (slide 6)

- Tabular deep-learning models output class probabilities but give no reliable uncertainty estimate
- TabM produces 32 predictions per input, yet no published work evaluates what their disagreement means
- Silent high-confidence errors threaten medical diagnosis, credit decisions and fault detection
- The TabM paper (ICLR 2025) explicitly leaves uncertainty quantification as open future work
- This project closes that gap with a retraining-free scoring layer on the frozen model
- Visual: text only, the problem is stated aloud with these five points

### Slide 3/10 — Motivation and Need (slide 7)

- TabM (ICLR 2025) is the current accuracy leader among deep tabular models
- TabM-based solutions already power winning entries in Kaggle competitions
- Gradient-boosted trees are accurate but output no uncertainty signal at all
- TabM's 32 sub-model predictions are a free ensemble signal waiting to be scored
- The full study runs on CPU in about 30 hours, so the work is feasible without accelerators
- Visual: text only

### Slide 4/10 — Project Objectives (slide 8)

- O1: reproduce default TabM (32 sub-models, 3 blocks, width 512, no embeddings) on 11 datasets times 3 seeds
- O2: quantify per-sample uncertainty with predictive entropy, mutual information and predictive variance
- O3: detect distribution shift with 7 within-dataset corruptions: noise at 3 levels, masking at 2 levels, shuffling, scaling
- O4: calibrate the model with expected calibration error, negative log-likelihood and reliability diagrams
- All four objectives close with file evidence: 33 models, 33 result files, 48 figures, 245 OOD files
- Visual: text only, the four objectives are read aloud

### Slide 5/10 — Proposed Solution Overview (slide 9)

- Keep the trained TabM completely frozen and add a scoring layer of about 50 lines
- Mean-member training loss keeps the 32 sub-models diverse, which is what makes disagreement informative
- Each test sample yields three outputs: predicted class, uncertainty score, OOD score
- No retraining is ever needed: scoring runs on stored predictions in minutes on CPU
- The same protocol serves both classification (entropy, mutual information) and regression (variance)

Insert: r1_solution.png
Caption on slide: Proposed solution — frozen TabM plus scoring layer.
What the diagram shows: five boxes left to right — tabular input features; frozen TabM with
32 sub-models; per-member probability predictions; uncertainty scorer computing entropy,
mutual information and variance; final outputs prediction plus uncertainty plus OOD score.

### Slide 6/10 — Scope and Boundaries (slide 10)

- In scope: the default TabM configuration exactly as published, 32 sub-models, no embeddings
- In scope: 11 datasets times 3 seeds, uncertainty scores, ECE and NLL calibration, 7 synthetic shifts
- Out of scope: embedding variants, hyperparameter tuning, external baselines such as ensembles or dropout
- Out of scope: cross-dataset semantic shifts, which need matching feature dimensions that rarely exist
- A fixed written scope protects every later claim at the viva: anything outside it is explicitly excluded

Draw this table on the slide:

| In scope | Out of scope |
|---|---|
| Default TabM: 32 sub-models, 3 blocks, width 512, no embeddings | Embedding variant and mini or packed builds |
| 11 datasets times 3 seeds, 33 runs, CPU only | Hyperparameter tuning of any kind |
| Entropy, mutual information, variance plus ECE and NLL | External baselines such as deep ensembles or MC dropout |
| 7 within-dataset shifts: noise, mask, shuffle, scale | Cross-dataset semantic shifts |

### Slide 7/10 — Technology Stack (Proposed) (slide 11)

- CPU-only stack: 16-thread i5 processor, 3 parallel workers with 5 threads each
- Official TabM package version 0.0.3 in its default configuration, nothing customised
- All versions pinned so any reviewer can reproduce every number bit for bit
- Preprocessed benchmark suite: 50 datasets available, 11 used, fixed train and test splits

Draw this table on the slide:

| Tool | Version | Why it is used |
|---|---|---|
| Python | 3.11 | Project language, isolated environment |
| PyTorch, CPU build | 2.12 | Model training, no accelerator needed |
| TabM package | 0.0.3 | Official model in default configuration |
| scikit-learn | 1.9 | Category encoding and OOD metrics |
| matplotlib | 3.10 | Reliability diagrams and histograms |
| Benchmark suite | 50 datasets, about 1 GB, 11 used | Official TabM data with fixed splits |

### Slide 8/10 — Initial Literature Survey (slide 12)

- Six papers anchor the method and the metrics: TabM, ensembles, efficient ensembling, dropout, OOD baseline, calibration
- The identified gap: TabM's 32 outputs were never scored for uncertainty or shift detection
- Citations in the thesis follow the IEEE numbered format with exactly these six references
- The survey spans 2016 to 2025 across ICLR, ICML and NeurIPS venues

Draw this table on the slide:

| Paper | Method | Key result | Limitation for this work |
|---|---|---|---|
| Gorishniy et al., ICLR 2025 | TabM, 32 sub-models | Best tabular accuracy | No uncertainty study |
| Lakshminarayanan et al., NeurIPS 2017 | Deep ensembles | Uncertainty gold standard | Needs 32 separate trainings |
| Wen et al., ICLR 2020 | BatchEnsemble | Efficient ensembling idea | Needs tabular adaptation |
| Gal and Ghahramani, ICML 2016 | MC Dropout | Cheap single-model scores | Weaker calibration |
| Hendrycks and Gimpel, ICLR 2017 | Softmax OOD baseline | Standard AUROC protocol | Confidence only |
| Guo et al., ICML 2017 | Calibration and ECE | Standard metrics plus fix | Post-hoc only |

### Slide 9/10 — Project Plan and Timeline (slide 13)

- Pilot study finished: small configuration on all 11 datasets in about 1.5 hours
- Full training finished: 33 runs of the default configuration in about 30 hours wall clock
- Uncertainty and OOD scoring finished: 33 result files, 48 figures, 245 OOD files
- Still pending: workshop paper draft, full thesis write-up, Reviews 3 and 4 in January and March 2027
- Four review gates pace the work: acceptance, Phase 1 completion, paper review, final mock

Insert: r1_gantt.png
Caption on slide: Project timeline with four review gates.
What the diagram shows: a month-axis Gantt chart from mid-2026 to March 2027 with bars for
pilot, full training, uncertainty scoring, OOD evaluation, paper writing and thesis writing,
plus gate markers for Review 1 (13 July 2026), Review 2 (05 Oct 2026), Review 3 (05 Jan 2027)
and Review 4 (08 Mar 2027).

### Slide 10/10 — Team Roles and Report Organisation (slide 14)

- Solo project: one owner designed, implemented and evaluated all five modules
- Data loading, model training, uncertainty scoring, calibration and OOD evaluation by the same owner
- Thesis Chapters 1 through 7 plus appendices written by the same owner
- Review material (slides, diagrams, report) prepared by the same owner
- Visual: text only

---

## Deck R2 — System Design, Implementation and Phase 1 Completion (PPT slides 15–25)

### Cover (slide 15)

- Title: TabM-UQ: Uncertainty-Aware Tabular Deep Learning via Parameter-Efficient Ensembling
- Short form: TabM-UQ
- Team table on the cover: ____________ (name, roll), ____________ (name, roll), ____________ (name, roll)
- Guide: ____________ , ____________ (designation), Department of Computer Science and Engineering
- Batch: ____________ , B.Tech, MVGR College of Engineering (Autonomous)
- Status line: Phase 1, Review 2, Phase 1 Completion Gate, 05 October 2026
- Progress since Review 1: 33 of 33 models trained, 33 uncertainty files scored, 245 OOD files produced

### Slide 1/10 — Title and Review 1 Feedback, Actions Taken (slide 16)

- The Review 1 panel gave four remarks; each one is resolved with evidence below
- Scope is fixed to the vanilla 32-sub-model TabM with no embeddings anywhere
- Configuration is set to the paper defaults on every dataset and every seed
- Runtime claim corrected from 2 hours to the measured 30 hours of wall clock
- External baselines deferred: the workshop contribution is the UQ plus OOD protocol itself

Draw this table on the slide:

| Panel remark | Action taken | Evidence |
|---|---|---|
| Single-model scope required | Fixed to vanilla 32-sub-model TabM, no embeddings | Scope table and configuration logs |
| Configuration must be exact | Paper defaults on all 11 datasets and 3 seeds | Training configuration records |
| Runtime claim of 2 hours wrong | Corrected to measured 30 hours wall clock | Training log with start and end times |
| Baselines required | Deferred; workshop needs the UQ plus OOD protocol | Scope statement and venue requirements |

### Slide 2/10 — Literature Review and Research Gap (slide 17)

- Survey spans 2016 to 2025 across ICLR, ICML and NeurIPS venues
- Six core papers anchor the method (TabM, ensembles) and the metrics (ECE, AUROC)
- The gap: no published study scores TabM outputs for uncertainty or shift detection
- Refined task: score the 32 frozen sub-model outputs post-hoc with entropy, mutual information and variance
- Calibration follows the standard 15-bin expected calibration error with reliability diagrams

Draw this table on the slide:

| Paper | Method | Key result | Limitation for this work |
|---|---|---|---|
| Gorishniy et al., ICLR 2025 | TabM, 32 sub-models | Best tabular accuracy | No uncertainty study |
| Lakshminarayanan et al., NeurIPS 2017 | Deep ensembles | Uncertainty gold standard | Needs 32 separate trainings |
| Wen et al., ICLR 2020 | BatchEnsemble | Efficient ensembling idea | Needs tabular adaptation |
| Gal and Ghahramani, ICML 2016 | MC Dropout | Cheap single-model scores | Weaker calibration |
| Hendrycks and Gimpel, ICLR 2017 | Softmax OOD baseline | Standard AUROC protocol | Confidence only |
| Guo et al., ICML 2017 | Calibration and ECE | Standard metrics plus fix | Post-hoc only |

### Slide 3/10 — System Architecture and Module Design (slide 18)

- Six pipeline stages: data archive, data loader, trainer, model store, uncertainty scorer, OOD evaluator
- Five modules with fixed contracts: arrays in, predictions or scores out
- All 33 trained models are reused frozen; the analysis stages never retrain anything
- Artefact counts: 33 checkpoints, 33 uncertainty files, 48 figures, 245 OOD files

Insert: r2_arch.png
Caption on slide: System architecture — six stages with frozen-model reuse.
What the diagram shows: six stage boxes connected left to right — dataset archive, data loader,
trainer, checkpoint store with 33 models, uncertainty and calibration scorer, OOD evaluator —
with arrows showing that scoring and detection read the frozen checkpoints and never loop back
to training, plus artefact counts on each store.

### Slide 4/10 — Database, Data Flow and Interface Design (slide 19)

- Training flow writes 33 checkpoints into the model store, one per dataset-seed pair
- Scoring flow reads checkpoints and test splits, then writes uncertainty and calibration results
- Detection flow corrupts the test features with 7 seeded shifts and writes 245 OOD files
- There is no database: a versioned file archive holds models, results and figures
- Every flow is reproducible from its seed: training seeds 0 to 2, corruption seeds fixed per shift

Insert: r2_dfd.png
Caption on slide: Data-flow diagram, levels 0 and 1.
What the diagram shows: an external entity (analyst) connected to process circles for training,
scoring and shift detection, and four data stores holding datasets, checkpoints, uncertainty
results and OOD results, with labelled arrows for features, predictions, entropy and AUROC scores.

### Slide 5/10 — Module Implementation, Code Walkthrough (slide 20)

- Default TabM: 32 sub-models, 3 blocks, width 512, dropout 0.1, no embeddings
- AdamW optimiser at learning rate 0.002 with weight decay 0.0003, 200 epochs maximum
- Early stopping with patience 20 and evaluation every 5 epochs; batch size scales 256 to 2048 with train size
- Categorical strings are encoded to integers; member probabilities feed entropy, mutual information and variance
- Seeded shifts (noise, mask, shuffle, scale) are scored with standard AUROC, AUPR and FPR at 95 percent recall
- Visual: text only, the walkthrough is spoken over these five points

### Slide 6/10 — Live Demo, Full System End to End (slide 21)

- Live chain on the phoneme dataset: load split, load frozen checkpoint, predict, score, plot
- Phoneme result: accuracy 0.8581, expected calibration error 0.0249, negative log-likelihood 0.329
- The reliability curve hugs the diagonal, so predicted confidence matches observed accuracy
- Entropy histogram separates confident correct samples from unsure ones at a glance
- Masking half the features gives an OOD score of 0.564 on this dataset

Insert: r2_demo_reliability.png
Caption on slide: Phoneme reliability diagram — predicted confidence versus observed accuracy.
What the diagram shows: a calibration curve with 15 bins lying close to the diagonal reference
line, plus a small inset of the entropy histogram; backup image r2_demo_entropy.png shows the
full entropy histogram alone.

### Slide 7/10 — Testing Results, Unit Integration System (slide 22)

- Unit level: every module imports cleanly and every function contract is exercised
- Integration level: uncertainty metrics recomputed from stored predictions for all 33 runs
- Integration level: all 33 summary files regenerate identically from the same inputs
- System level: all 7 shifts across all runs verified present — 245 OOD files with valid scores
- Honest note: there is no automated test suite; verification is recomputation with exact file counts

Draw this table on the slide:

| Level | Check performed | Result |
|---|---|---|
| Unit | Module imports and function contracts | Pass |
| Integration | Uncertainty recomputed from stored predictions, 33 of 33 | Pass |
| Integration | Summary files regenerated identically, 33 of 33 | Pass |
| System | 7 shifts across all runs, 245 OOD files with valid scores | Pass |
| Honesty note | No automated suite; counts audited against stored files | Stated |

### Slide 8/10 — Performance Metrics and Known Issues (slide 23)

- Best accuracy: MiniBooNE at 0.9376 with calibration error 0.0105, the best-calibrated result
- Best shift detection: masking half the numeric features, macro AUROC 0.641 over 8 healthy datasets
- Regression peak: variance detects strong Gaussian noise at AUROC 0.942 on both california and wine quality
- Known issue: 3 runs collapsed (churn constant predictions, credit near-random 0.4963, Ailerons near-zero variance)
- Collapsed runs are excluded from all primary means and reported separately with a dagger mark

Draw this table on the slide:

| Highlight | Value |
|---|---|
| Best accuracy: MiniBooNE | 0.9376, calibration error 0.0105 |
| Phoneme accuracy and calibration error | 0.8581 and 0.0249 |
| Best OOD: mask half the features | Macro AUROC 0.641 |
| Regression peak under strong noise | AUROC 0.942 on california and wine quality |
| Known issue | 3 collapsed runs, kept in the appendix with a dagger mark |

### Slide 9/10 — Objectives versus Outcomes, Phase 1 Gate Check (slide 24)

- O1 reproduce default TabM on 11 datasets times 3 seeds: 33 checkpoints, achieved
- O2 quantify uncertainty per sample: full 8-dataset primary table, achieved
- O3 detect 7 shifts: 245 OOD result files, achieved
- O4 calibrate with ECE, NLL and reliability plots: values plus 48 figures, achieved
- Panel scorecard: four of four objectives met, so Phase 1 is complete and the project proceeds to the paper

Draw this table on the slide:

| Objective | Evidence | Status |
|---|---|---|
| O1: reproduce, 33 of 33 runs | 33 trained checkpoints | Achieved |
| O2: uncertainty table | 8-dataset primary table with entropy, MI, variance | Achieved |
| O3: OOD over 7 shifts | 245 result files | Achieved |
| O4: calibration and plots | ECE and NLL values plus 48 figures | Achieved |

### Slide 10/10 — Phase 2 Plan and Thesis Status (slide 25)

- Next analysis: AUPR tables and overlay figures comparing shifts side by side
- Workshop paper draft on the first UQ and OOD study of TabM, co-authored with the guide
- Thesis Chapters 1 through 6 filled and reviewed before the January 2027 review
- Explicitly deferred: external baselines and any feature or embedding retraining
- Fixed milestones: Review 3 on 05 January 2027, final mock review on 08 March 2027
- Visual: text only

---

## Deck R3 — Integration, Evaluation and Research Paper (PPT slides 26–36)

### Cover (slide 26)

- Title: TabM-UQ: Uncertainty-Aware Tabular Deep Learning via Parameter-Efficient Ensembling
- Short form: TabM-UQ
- Team table on the cover: ____________ (name, roll), ____________ (name, roll), ____________ (name, roll)
- Guide: ____________ , ____________ (designation), Department of Computer Science and Engineering
- Batch: ____________ , B.Tech, MVGR College of Engineering (Autonomous)
- Status line: Phase 2, Review 3, Research Paper Mandatory, 05 January 2027
- Phase 1 outcomes: 33 trained models, 245 OOD files, four of four objectives met

### Slide 1/10 — Phase 2 Overview and Phase 1 Outcomes (slide 27)

- Phase 2 adds three things: the workshop paper, the final metric tables, and the completed thesis
- Phase 1 delivered 33 frozen models, 33 uncertainty files, 48 figures and 245 OOD files
- New targets: AUPR tables for every shift, overlay figures, and the submitted paper draft
- Scope stays locked: no embeddings, no tuning, no external baselines, no cross-dataset shifts
- Visual: text only

### Slide 2/10 — Review 2 Feedback and Phase 1 Resolutions (slide 28)

- Review 2 was held on 05 October 2026 as the Phase 1 completion gate
- The panel examined 33 checkpoints, 33 uncertainty files, 48 figures and 245 OOD files
- Each panel remark is recorded below with its resolution and the file evidence behind it
- Scope changes after this point need written guide approval

Draw this table on the slide and fill each row from the panel remarks:

| Panel remark | Resolution | Evidence |
|---|---|---|
| ____________ | ____________ | ____________ |
| ____________ | ____________ | ____________ |
| ____________ | ____________ | ____________ |

### Slide 3/10 — Full Integrated System Demo (slide 29)

- End to end on phoneme: data load, frozen checkpoint, 32-member prediction, scoring, calibration plot
- Phoneme accuracy 0.8581 with calibration error 0.0249; reliability curve hugs the diagonal
- Edge cases shown honestly: the 3 collapsed runs are flagged, never hidden
- The full chain runs on CPU in minutes because no retraining is involved

Insert: r2_demo_reliability.png
Caption on slide: Phoneme reliability diagram — predicted confidence versus observed accuracy.
What the diagram shows: a calibration curve with 15 bins lying close to the diagonal reference
line; backup image r2_demo_entropy.png shows the entropy histogram alone.

### Slide 4/10 — Integration Architecture, Final (slide 30)

- All stages connect through frozen-model reuse: training output becomes scoring input
- Data contracts are fixed: feature arrays in, predictions and scores out, no hidden state
- A single forward pipeline with no retraining loops and no feedback edges
- Artefact totals: 33 checkpoints, 33 uncertainty files, 48 figures, 245 OOD files

Insert: r2_arch.png
Caption on slide: Final integration architecture — six stages with frozen-model reuse.
What the diagram shows: six stage boxes connected left to right — dataset archive, data loader,
trainer, checkpoint store with 33 models, uncertainty and calibration scorer, OOD evaluator —
with arrows showing that scoring and detection read the frozen checkpoints and never loop back
to training, plus artefact counts on each store.

### Slide 5/10 — Novel Module and Research Contribution Demo (slide 31)

- First published protocol that scores TabM outputs for uncertainty post-hoc
- No model edit and no retraining: about 50 lines of scoring code on stored predictions
- Live on sample inputs: entropy and mutual information for classification, variance for regression
- The recipe transfers to any future TabM checkpoint without modification

Insert: r1_solution.png
Caption on slide: Proposed solution — frozen TabM plus scoring layer.
What the diagram shows: five boxes left to right — tabular input features; frozen TabM with
32 sub-models; per-member probability predictions; uncertainty scorer computing entropy,
mutual information and variance; final outputs prediction plus uncertainty plus OOD score.

### Slide 6/10 — Full Test Suite Results (slide 32)

- Uncertainty metrics recomputed from stored predictions for all 33 runs
- All 33 summary files regenerate identically from the same inputs
- All 7 shifts verified across all runs: 245 OOD files carrying valid AUROC scores
- Pass rate is 100 percent on every defined suite at unit, integration and system level
- Honest note: there is no automated suite; verification is recomputation with exact file counts

Draw this table on the slide:

| Level | Check performed | Result |
|---|---|---|
| Unit | Module imports and function contracts | Pass |
| Integration | Uncertainty recomputed from stored predictions, 33 of 33 | Pass |
| Integration | Summary files regenerated identically, 33 of 33 | Pass |
| System | 7 shifts across all runs, 245 OOD files with valid scores | Pass |
| Honesty note | No automated suite; counts audited against stored files | Stated |

### Slide 7/10 — Quantitative Evaluation Results (slide 33)

- Classification accuracy on healthy sets spans 0.7163 (wine) to 0.9376 (MiniBooNE)
- Calibration error stays between 0.0105 (MiniBooNE) and 0.0498 (adult, with one seed outlier at 0.123)
- OOD macro AUROC spans 0.530 (uniform scale, weakest) to 0.641 (mask half, strongest)
- Regression variance peaks at 0.942 under strong Gaussian noise on california and wine quality
- Inverted signals are reported as found: noise lowers entropy on bank-marketing, MagicTelescope, wine and phoneme

Draw this table on the slide:

| Dataset | Accuracy or RMSE | Calibration error | Strongest OOD signal |
|---|---|---|---|
| MiniBooNE | Accuracy 0.9376 | 0.0105 | Mask half: 0.819 |
| phoneme | Accuracy 0.8581 | 0.0249 | Mask half: 0.564 |
| MagicTelescope | Accuracy 0.8205 | 0.0205 | Mask half: 0.570 |
| adult | Accuracy 0.8037 | 0.0498 | Noise level 0.5: 0.744 |
| bank-marketing | Accuracy 0.7726 | 0.0291 | No signal: noise inverted |
| wine | Accuracy 0.7163 | 0.0432 | No signal: near random |
| california | RMSE 0.8434 | NLL 16.8 | Noise level 2: 0.942 |
| wine quality | RMSE 0.8071 | NLL 29.2 | Noise level 2: 0.942 |

### Slide 8/10 — Research Paper Draft (slide 34)

- Working title: the first uncertainty and OOD study of TabM
- Contribution one: a retraining-free post-hoc scoring protocol of about 50 lines
- Contribution two: a frozen-model reuse recipe — 33 checkpoints scored with zero retraining
- Contribution three: an open benchmark of 33 models and 245 shift runs with full tables
- Target venue: a tabular machine-learning workshop; draft with the guide, submitted before 08 March 2027
- Visual: text only

### Slide 9/10 — Originality Report, Thesis and System (slide 35)

- Thesis similarity target is below 15 percent on the institutional check
- All training code, scoring code, plots and text are original work by the project owner
- Datasets and the TabM package are third-party and are cited as such in the thesis
- The originality check runs after the full thesis draft completes, before March 2027
- Visual: text only

### Slide 10/10 — Final Review Plan and Mock Viva Timeline (slide 36)

- Thesis completed and reviewed chapter by chapter through January 2027
- Workshop paper polished and submitted before the March 2027 mock review
- Mock viva rehearses the full 20-slide final deck with live phoneme demo
- Final internal mock review on 08 March 2027, viva window opens late March 2027
- Visual: text only

---

## Deck R4 Final — Final Internal Mock Review, 20 slides (PPT slides 37–57)

Condensed viva deck. Diagram files are reused; no new visuals are needed.

### Cover (slide 37)

- Title: TabM-UQ: Uncertainty-Aware Tabular Deep Learning via Parameter-Efficient Ensembling
- Short form: TabM-UQ
- Team table on the cover: ____________ (name, roll), ____________ (name, roll), ____________ (name, roll)
- Guide: ____________ , ____________ (designation), Department of Computer Science and Engineering
- Batch: ____________ , B.Tech, MVGR College of Engineering (Autonomous)
- Status line: Phase 2, Review 4 Final, viva in 3 weeks, 08 March 2027
- Complete work: four of four objectives met, workshop paper in preparation with the guide

### Slide 1/20 — Title Slide (slide 38)

- TabM-UQ: Uncertainty-Aware Tabular Deep Learning via Parameter-Efficient Ensembling
- One default TabM, 11 datasets, 3 seeds, CPU only, about 30 hours of training
- All four objectives met with file evidence; workshop paper in preparation
- Presented by ____________ (roll ____________), guided by ____________
- Final internal mock review, 08 March 2027
- Visual: text only, the cover already carries the title and team table

### Slide 2/20 — Problem Statement and Motivation (slide 39)

- Tabular models output probabilities but give no reliable uncertainty estimate
- TabM produces 32 predictions per input; their meaning was never evaluated until this work
- Silent high-confidence errors threaten medical, credit and fault-detection uses
- TabM leads tabular deep learning and already powers winning Kaggle solutions, so its uncertainty matters
- Visual: text only

### Slide 3/20 — Objectives and Scope (slide 40)

- O1 reproduce default TabM on 11 datasets times 3 seeds, 33 runs total
- O2 quantify uncertainty with entropy, mutual information and variance
- O3 detect 7 within-dataset shifts: noise, mask, shuffle and scale
- O4 calibrate with expected calibration error, negative log-likelihood and reliability diagrams
- Out of scope: embeddings, tuning, external baselines, cross-dataset shifts

Draw this table on the slide:

| In scope | Out of scope |
|---|---|
| Default TabM: 32 sub-models, 3 blocks, width 512, no embeddings | Embedding variant and mini or packed builds |
| 11 datasets times 3 seeds, 33 runs, CPU only | Hyperparameter tuning of any kind |
| Entropy, mutual information, variance plus ECE and NLL | External baselines such as deep ensembles or MC dropout |
| 7 within-dataset shifts: noise, mask, shuffle, scale | Cross-dataset semantic shifts |

### Slide 4/20 — Literature Review Summary (slide 41)

- Six papers anchor the method and the metrics across ICLR, ICML and NeurIPS, 2016 to 2025
- Ensemble literature proves disagreement measures uncertainty; calibration literature standardises ECE
- The gap this work fills: TabM outputs were never scored for uncertainty or shift detection
- Thesis citations follow the IEEE numbered format with exactly these six references

Draw this table on the slide:

| Paper | Method | Key result | Limitation for this work |
|---|---|---|---|
| Gorishniy et al., ICLR 2025 | TabM, 32 sub-models | Best tabular accuracy | No uncertainty study |
| Lakshminarayanan et al., NeurIPS 2017 | Deep ensembles | Uncertainty gold standard | Needs 32 separate trainings |
| Wen et al., ICLR 2020 | BatchEnsemble | Efficient ensembling idea | Needs tabular adaptation |
| Gal and Ghahramani, ICML 2016 | MC Dropout | Cheap single-model scores | Weaker calibration |
| Hendrycks and Gimpel, ICLR 2017 | Softmax OOD baseline | Standard AUROC protocol | Confidence only |
| Guo et al., ICML 2017 | Calibration and ECE | Standard metrics plus fix | Post-hoc only |

### Slide 5/20 — System Architecture, Final (slide 42)

- Six stages: dataset archive, data loader, trainer, checkpoint store, uncertainty scorer, OOD evaluator
- All 33 trained models reused frozen with zero retraining in the analysis stages
- Artefact totals: 33 checkpoints, 33 uncertainty files, 48 figures, 245 OOD files
- Single forward pipeline with fixed contracts: feature arrays in, predictions and scores out

Insert: r2_arch.png
Caption on slide: Final system architecture — six stages with frozen-model reuse.
What the diagram shows: six stage boxes connected left to right — dataset archive, data loader,
trainer, checkpoint store with 33 models, uncertainty and calibration scorer, OOD evaluator —
with arrows showing that scoring and detection read the frozen checkpoints and never loop back
to training, plus artefact counts on each store.

### Slide 6/20 — Technology Stack and Database Design (slide 43)

- CPU-only stack with pinned versions: Python 3.11, PyTorch 2.12, TabM 0.0.3, scikit-learn 1.9, matplotlib 3.10
- 11 benchmarks with 1,700 to 50,000 training rows each and fixed preprocessed splits
- No database: a versioned file archive holds the 33 checkpoints, results and figures
- Every artefact regenerates from its seed, so the whole study is bit-for-bit reproducible

Draw this table on the slide:

| Tool | Version | Why it is used |
|---|---|---|
| Python | 3.11 | Project language, isolated environment |
| PyTorch, CPU build | 2.12 | Model training, no accelerator needed |
| TabM package | 0.0.3 | Official model in default configuration |
| scikit-learn | 1.9 | Category encoding and OOD metrics |
| matplotlib | 3.10 | Reliability diagrams and histograms |
| Benchmark suite | 50 datasets, about 1 GB, 11 used | Official TabM data with fixed splits |

### Slide 7/20 — Module Breakdown and Data Flow (slide 44)

- Five modules: data loader, model wrapper, uncertainty scorer, calibration scorer, OOD evaluator
- Training writes checkpoints; scoring reads them and writes uncertainty results; detection writes OOD files
- Four file stores hold datasets, checkpoints, uncertainty results and OOD results respectively
- Fixed seeds everywhere: training seeds 0 to 2, one fixed corruption seed per shift

Insert: r2_dfd.png
Caption on slide: Data-flow diagram, levels 0 and 1.
What the diagram shows: an external entity (analyst) connected to process circles for training,
scoring and shift detection, and four data stores holding datasets, checkpoints, uncertainty
results and OOD results, with labelled arrows for features, predictions, entropy and AUROC scores.

### Slide 8/20 — Implementation Overview, All Modules (slide 45)

- All five modules complete under a single owner with no pending implementation
- Training finished at 33 of 33 runs; scoring and detection finished over the frozen outputs
- Remaining work is writing only: paper draft plus thesis chapters, no further code
- Module handover is documented in the thesis appendices with contracts and file lists

Draw this table on the slide:

| Module | Function | Status |
|---|---|---|
| Data loader | Reads fixed splits, encodes categories to integers | Done |
| Model wrapper | Trains default TabM, 33 runs on CPU | Done |
| Uncertainty scorer | Entropy, mutual information, variance per sample | Done |
| Calibration scorer | ECE over 15 bins, NLL, reliability and entropy plots | Done |
| OOD evaluator | 7 seeded shifts, AUROC, AUPR and FPR at 95 percent recall | Done |

### Slide 9/20 — Key Module Walkthrough (slide 46)

- Default TabM: 32 sub-models, 3 blocks, width 512, dropout 0.1, no embeddings
- AdamW at learning rate 0.002, weight decay 0.0003, 200 epochs, patience 20, evaluation every 5 epochs
- Member probabilities feed entropy and mutual information; member regressions feed variance
- Seeded shifts scored with standard AUROC; training ran on a 16-thread CPU in about 30 hours
- Visual: text only, the walkthrough is spoken over these four points

### Slide 10/20 — Novel Contribution, Deep Dive (slide 47)

- First protocol that scores TabM outputs for uncertainty post-hoc, with no model edit
- About 50 lines of scoring code that runs on stored predictions in minutes on CPU
- Same recipe covers classification and regression, and transfers to any TabM checkpoint
- Live demonstration uses real phoneme outputs: accuracy 0.8581, calibration error 0.0249

Insert: r1_solution.png
Caption on slide: Proposed solution — frozen TabM plus scoring layer.
What the diagram shows: five boxes left to right — tabular input features; frozen TabM with
32 sub-models; per-member probability predictions; uncertainty scorer computing entropy,
mutual information and variance; final outputs prediction plus uncertainty plus OOD score.

### Slide 11/20 — Full System Demo, End to End (slide 48)

- Phoneme accuracy 0.8581 with calibration error 0.0249 and negative log-likelihood 0.329
- Live chain: load split, load frozen checkpoint, 32-member prediction, scoring, calibration plot
- Reliability curve hugs the diagonal; entropy histogram separates confident from unsure samples
- Masking half the features gives an OOD score of 0.564 on this dataset

Insert: r2_demo_reliability.png
Caption on slide: Phoneme reliability diagram — predicted confidence versus observed accuracy.
What the diagram shows: a calibration curve with 15 bins lying close to the diagonal reference
line; backup image r2_demo_entropy.png shows the entropy histogram alone.

### Slide 12/20 — Testing Results Summary (slide 49)

- Unit level: module imports and function contracts, pass
- Integration level: uncertainty recomputed for 33 of 33 runs, pass
- Integration level: summaries regenerated identically for 33 of 33 files, pass
- System level: 245 OOD files with valid scores across all 7 shifts, pass
- Honest note: no automated suite exists; verification is recomputation with exact file counts

Draw this table on the slide:

| Level | Check performed | Result |
|---|---|---|
| Unit | Module imports and function contracts | Pass |
| Integration | Uncertainty recomputed from stored predictions, 33 of 33 | Pass |
| Integration | Summary files regenerated identically, 33 of 33 | Pass |
| System | 7 shifts across all runs, 245 OOD files with valid scores | Pass |
| Honesty note | No automated suite; counts audited against stored files | Stated |

### Slide 13/20 — Quantitative Evaluation, Key Metrics (slide 50)

- Best accuracy 0.9376 on MiniBooNE with the best calibration error 0.0105
- Best shift detection: masking half the features, macro AUROC 0.641 over 8 healthy datasets
- Regression peaks at AUROC 0.942 under strong noise on california and wine quality
- Three collapsed runs excluded from all primary means and reported in the appendix

Draw this table on the slide:

| Highlight | Value |
|---|---|
| Best accuracy: MiniBooNE | 0.9376, calibration error 0.0105 |
| Phoneme accuracy and calibration error | 0.8581 and 0.0249 |
| Best OOD: mask half the features | Macro AUROC 0.641 |
| Regression peak under strong noise | AUROC 0.942 on california and wine quality |
| Known issue | 3 collapsed runs, kept in the appendix with a dagger mark |

### Slide 14/20 — Objectives Achievement Table (slide 51)

- O1 reproduce default TabM with 33 checkpoints: achieved
- O2 quantify uncertainty with the full 8-dataset primary table: achieved
- O3 detect 7 shifts with 245 OOD result files: achieved
- O4 calibrate with ECE, NLL and 48 figures: achieved
- Panel scorecard reads four of four; Phase 1 closed and the project moved to the paper

Draw this table on the slide:

| Objective | Evidence | Status |
|---|---|---|
| O1: reproduce, 33 of 33 runs | 33 trained checkpoints | Achieved |
| O2: uncertainty table | 8-dataset primary table with entropy, MI, variance | Achieved |
| O3: OOD over 7 shifts | 245 result files | Achieved |
| O4: calibration and plots | ECE and NLL values plus 48 figures | Achieved |

### Slide 15/20 — Comparison with Existing Work (slide 52)

- Small pilot configuration stayed healthy on all 11 datasets, including credit at 0.727 and churn MI at 0.356
- Full default configuration collapsed 3 runs: churn to constant predictions, credit to 0.4963, Ailerons variance to 2e-10
- Collapse is driven by capacity (about 815,000 parameters on small or imbalanced sets), not by data corruption
- No external baselines are claimed; the comparison is within-model, pilot versus canonical
- Visual: text only

### Slide 16/20 — Research Contributions (slide 53)

- Contribution 1: the first uncertainty and OOD evaluation of TabM outputs
- Contribution 2: a retraining-free recipe of about 50 lines that scores any frozen TabM checkpoint
- Contribution 3: an open benchmark of 33 trained models and 245 shift runs with full result tables
- Practical finding: regression variance detects shifts reliably while classification entropy often fails
- Visual: text only

### Slide 17/20 — Limitations and Future Work (slide 54)

- Three runs collapsed under the full configuration and are reported in the appendix, not the main tables
- No cross-dataset shifts and no external baselines such as deep ensembles or MC dropout
- Uniform scaling at factor 1.3 is too mild to move classification entropy; stronger drift is needed
- Future work: embedding variants, stronger shifts, external baselines, and uncertainty-gated abstention
- Visual: text only

### Slide 18/20 — Research Paper and Originality (slide 55)

- Workshop paper on the first UQ and OOD study of TabM, co-authored with the guide
- Three contributions: the scoring protocol, the frozen-reuse recipe, and the open benchmark
- Draft in preparation; submission target is before the final mock review on 08 March 2027
- Thesis similarity target below 15 percent; the originality check runs after the full draft completes
- Visual: text only

### Slide 19/20 — Conclusion, What this Work Means (slide 56)

- Regression variance detects feature corruption reliably, peaking at 0.942 under strong noise
- Classification entropy alone fails on small datasets under mild shifts, with macro scores near 0.59
- Practitioners gain a clear usage bound: trust variance signals, distrust mild-shift entropy signals
- The full protocol costs minutes on CPU and needs no retraining, so it is deployable as-is
- Visual: text only

### Slide 20/20 — Thank You & Q&A (slide 57)

- TabM-UQ by ____________ , guided by ____________
- Four of four objectives met: 33 models, 48 figures, 245 OOD files, one protocol
- Thank you; questions are welcome
- Contact: ____________
- Visual: text only
