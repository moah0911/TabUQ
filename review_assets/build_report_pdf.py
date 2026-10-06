"""Assemble R1+R2 combined report PDF (A4, selectable text via PdfPages).

Only 5 figures are embedded (4 redesigned diagrams + 1 demo plot);
all other pages are detailed text-only bullets in faculty-facing wording.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

OUT = Path(__file__).parent
PDF = OUT / "R1_R2_UQ_OOD_Report.pdf"

NAVY = "#1E2761"
TEAL = "#0B8A7C"
DARK = "#1A1A2E"
GREY = "#6B7280"

# (heading, bullets, image-or-None, caption/thesis-ref, fontsize)
PAGES = [
    ("TabM-UQ: Uncertainty-Aware Tabular Deep Learning\nvia Parameter-Efficient Ensembling",
     ["Single model: default TabM, 32 sub-models, no embeddings (paper default)",
      "First uncertainty + within-dataset OOD evaluation of TabM (ICLR 2025 future work)",
      "11 benchmarks x 3 seeds = 33 runs, CPU-only, about 30 hours total",
      "Team: [Your Name], [Roll No]  |  Guide: [Guide Name]  |  Batch: [9A]",
      "Covers Review 1 (Acceptance) + Review 2 (Design / Phase-1 gate)"],
     None, "Cover - replace bracketed placeholders before submission", 10.5),
    ("R1. Problem Statement (Thesis 1.2)",
     ["Tabular models lack reliable uncertainty estimates for high-stakes use",
      "TabM gives 32 predictions per object, but their meaning was never evaluated",
      "Silent failures possible in medical, credit and fault-detection domains",
      "Paper leaves uncertainty/OOD as explicit future work; still unaddressed"],
     None, "Thesis Sec 1.2 - text-only slide (no picture needed)", 10.5),
    ("R1. Motivation and Need (Thesis 1.3)",
     ["TabM (ICLR 2025): state-of-the-art tabular deep learning, used in Kaggle wins",
      "Gradient-boosted trees are accurate but provide no uncertainty",
      "Gap is explicit and open; a CPU-only study is feasible",
      "'I don't know' is more valuable than a wrong confident prediction"],
     None, "Thesis Sec 1.3 - text-only slide", 10.5),
    ("R1. Project Objectives (Thesis 1.4)",
     ["O1: reproduce default TabM on 11 benchmarks (33 runs)",
      "O2: quantify uncertainty - entropy, MI (classification) + variance (regression)",
      "O3: detect OOD within-dataset - noise / mask / shuffle / scale (7 shifts)",
      "O4: calibrate - ECE (15 bins), NLL + reliability diagrams"],
     None, "Thesis Sec 1.4, Table 1.4 - type as Word table in thesis", 10.5),
    ("R1. Proposed Solution Overview (Thesis 1.5, Fig 1.1)",
     ["Frozen TabM (B,32,1) + post-hoc scoring layer (about 50 lines)",
      "Training uses mean loss over members, not loss of the mean (keeps diversity)",
      "Outputs: prediction + uncertainty + OOD score; no retraining, CPU only"],
     "r1_solution.png", "Fig 1.1: detailed UQ-layer pipeline with shapes and formulas", 10.5),
    ("R1. Scope and Boundaries (Thesis 1.6, Table 1.1)",
     ["IN: vanilla 32-sub-model TabM, 11 datasets, entropy/MI/variance, ECE/NLL, 7 shifts",
      "OUT: embedding variant, tuning, external baselines, cross-dataset shifts",
      "Baselines deferred - workshop target needs uncertainty + OOD only"],
     None, "Thesis Sec 1.6, Table 1.1 - type as Word table in thesis", 10.5),
    ("R1. Technology Stack (Thesis 3.8)",
     ["Python 3.11, PyTorch 2.12 (CPU), official TabM package 0.0.3",
      "scikit-learn 1.9 (encoding + metrics), matplotlib 3.10 (plots)",
      "Data: official 50-set TabM suite, about 1 GB, 11 sets used",
      "Compute: 3 parallel workers x 5 threads, no GPU required"],
     None, "Thesis Sec 3.8, Table 3.8 - type as Word table in thesis", 10.5),
    ("R1. Literature Survey (Thesis Ch 2, Table 2.1)",
     ["[1] TabM: best tabular accuracy, no uncertainty/OOD - our starting point",
      "[2] Deep ensembles: uncertainty gold standard, needs 32 training runs",
      "[3] BatchEnsemble: efficient ensembling idea behind TabM",
      "[4] MC Dropout: cheap single-model uncertainty, weaker calibration",
      "[5] Softmax OOD baseline: standard AUROC protocol",
      "[6] Calibration (ECE/NLL): our calibration metrics",
      "Gap: no published work evaluates TabM outputs as uncertainty/OOD scores"],
     None, "Thesis Table 2.1 + References (see last page) - IEEE format", 10.5),
    ("R1. Project Plan and Timeline (Appendix A, Fig A.1)",
     ["M1 pilot (small config, 1.5 h) done  |  M2 full-size + UQ (30 h, 33/33) done",
      "M3 OOD over 7 shifts (245 files) done  |  Paper: workshop target",
      "Reviews: R1 acceptance - R2 gate - R3 paper - R4 final"],
     "r1_gantt.png", "Fig A.1: Gantt chart with measured runtimes and review gates", 10.5),
    ("R1. Team Roles and Report Organisation (Thesis 1.8)",
     ["Solo project: data loading, model wrapper, metrics, OOD, results by one owner",
      "Thesis Chapters 1-7 plus Appendices A-F; fill name/roll/guide/batch",
      "Fill placeholders on the PPT title slide before the review"],
     None, "Thesis Sec 1.8, Table A.4 - type as Word table in thesis", 10.5),
    ("R2. R1 Feedback - Actions Taken (Table A.2)",
     ["Scope fixed to single vanilla TabM (32 sub-models, no embeddings)",
      "Configuration corrected to paper defaults; runtime corrected to 30 h measured",
      "All documents reconciled to one configuration; baselines formally deferred"],
     None, "Thesis Table A.2 - evidence points to thesis sections and logs", 10.5),
    ("R2. System Architecture - Final (Fig 4.1)",
     ["6 stages: archive - loader - trainer - stored models - UQ/calibration - OOD",
      "33 trained models reused frozen for all analysis (zero retraining)",
      "Outputs: 33 uncertainty files, 48 plots, 245 OOD files"],
     "r2_arch.png", "Fig 4.1: six stages with inputs, outputs and artifact counts", 10.5),
    ("R2. Data Flow and Interface (DFD Levels 0-1)",
     ["P1 trains 33 models from dataset splits into the checkpoint store",
      "P2 scores uncertainty + calibration (test splits + frozen model)",
      "P4 scores OOD over 7 shifts; results to dedicated stores",
      "No database: versioned file archive; 3 command-line operations"],
     "r2_dfd.png", "DFD: external entity, processes P1/P2/P4, stores D1-D4", 10.5),
    ("R2. Module Implementation (Thesis 5.3)",
     ["Data loader: reads splits, encodes string categories, reserves unknown code",
      "Model wrapper: TabM with 32 sub-models, AdamW, auto batch, early stopping",
      "Scorers: sigmoid to probabilities, entropy/MI/variance, ECE/NLL, AUROC",
      "OOD: seeded corruptions (noise/mask/shuffle/scale); per-seed result files"],
     None, "Thesis Sec 5.3 - detailed bullets, key listings in Appendix E", 10.5),
    ("R2. Live Demo - Full System (Fig 5.2)",
     ["Phoneme dataset: load - predict - entropy 0.324, ECE 0.024",
      "Reliability diagram and entropy histogram shown below",
      "OOD on phoneme: mask-0.5 AUROC 0.564 (entropy histogram in folder)"],
     "r2_demo_reliability.png", "Fig 5.2: phoneme reliability diagram (live demo backup)", 10.5),
    ("R2. Testing Results (Tables 6.3-6.5)",
     ["Unit: wrapper/metrics/calibration imports - pass",
      "Integration: uncertainty recompute 33/33 with matching summaries - pass",
      "System: OOD 7 shifts x 33 runs = 245 files - pass",
      "Honest status: no pytest suite; recomputation counts as evidence"],
     None, "Thesis Tables 6.3-6.5 - type as Word tables in thesis", 10.5),
    ("R2. Performance Metrics and Known Issues (6.8, 7.3)",
     ["Best accuracy MiniBooNE 0.9376 (ECE 0.010); phoneme 0.8581",
      "Best OOD mask-0.5 macro AUROC 0.641; regression strong-noise 0.942",
      "Known issue: 3 collapsed runs (churn/credit/Ailerons) detailed in appendix"],
     None, "Thesis Tables 6.1, 6.2, 6.10 - type as Word tables in thesis", 10.5),
    ("R2. Objectives vs Outcomes - Phase-1 Gate (Table 6.10)",
     ["O1 reproduce 33/33: ACHIEVED  |  O2 uncertainty table: ACHIEVED",
      "O3 OOD 7 shifts / 245 files: ACHIEVED  |  O4 ECE/NLL + 48 plots: ACHIEVED",
      "Gate decision: Phase 1 complete, proceed to paper drafting"],
     None, "Thesis Table 6.10 - type as Word table in thesis", 10.5),
    ("R2. Phase-2 Plan and Thesis Status (App A)",
     ["Remaining: AUPR/FPR tables, overlay figures, workshop paper draft",
      "Thesis Chapters 1-5 drafted; Chapter 6 tables ready; appendices stubbed",
      "Deferred: external baselines, extra-feature retrain, embedding variant"],
     None, "Thesis App A, Table A.3 - chapter approvals by guide", 10.5),
    ("References (IEEE format - all 6 cited works)",
     ["[1] Y. Gorishniy, A. Kotelnikov, and A. Babenko, 'TabM: Advancing tabular deep "
      "learning with parameter-efficient ensembling,' in Proc. Int. Conf. Learn. "
      "Representations (ICLR), Singapore, 2025. Available: https://arxiv.org/abs/2410.24210",
      "[2] B. Lakshminarayanan, A. Pritzel, and C. Blundell, 'Simple and scalable "
      "predictive uncertainty estimation using deep ensembles,' in Proc. Adv. Neural Inf. "
      "Process. Syst. (NeurIPS), vol. 30, Long Beach, CA, USA, 2017, pp. 6402-6413.",
      "[3] Y. Wen, D. Tran, and J. Ba, 'BatchEnsemble: An alternative approach to efficient "
      "ensemble and lifelong learning,' in Proc. Int. Conf. Learn. Representations "
      "(ICLR), Addis Ababa, Ethiopia, 2020.",
      "[4] Y. Gal and Z. Ghahramani, 'Dropout as a Bayesian approximation: Representing "
      "model uncertainty in deep learning,' in Proc. Int. Conf. Mach. Learn. (ICML), "
      "vol. 48, New York, NY, USA, 2016, pp. 1050-1059.",
      "[5] D. Hendrycks and K. Gimpel, 'A baseline for detecting misclassified and "
      "out-of-distribution examples in neural networks,' in Proc. Int. Conf. Learn. "
      "Representations (ICLR), Toulon, France, 2017.",
      "[6] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, 'On calibration of modern "
      "neural networks,' in Proc. Int. Conf. Mach. Learn. (ICML), vol. 70, Sydney, "
      "Australia, 2017, pp. 1321-1330."],
     None, "Thesis References - numbered in order of first citation", 8.5),
]

with PdfPages(PDF) as pdf:
    for heading, bullets, img, caption, fs in PAGES:
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        fig.patch.set_facecolor("white")
        fig.text(0.5, 0.955, heading, fontsize=15, fontweight="bold", color=NAVY,
                 ha="center", va="top", family="serif",
                 bbox=dict(facecolor="#E8EDF5", edgecolor=NAVY, boxstyle="round,pad=0.4"))
        y = 0.86
        step = 0.035 if fs >= 10 else 0.030
        for b in bullets:
            fig.text(0.08, y, "-  " + b, fontsize=fs, color=DARK, ha="left", va="top",
                     wrap=True)
            nlines = max(1, len(b) // 95 + 1)
            y -= step * nlines
        if img:
            p = OUT / img
            if p.exists():
                from matplotlib import image as mpimg
                im = mpimg.imread(str(p))
                ax = fig.add_axes([0.08, 0.08, 0.84, max(0.30, y - 0.12)])
                ax.imshow(im)
                ax.axis("off")
                ax.set_title(caption, fontsize=9, color="#555555", pad=6)
            else:
                fig.text(0.5, 0.3, f"[missing image: {img}]", ha="center", color="red")
        else:
            fig.text(0.5, 0.08, caption, ha="center", fontsize=9, color=GREY,
                     style="italic")
        fig.text(0.5, 0.03, "TabM-UQ - R1+R2 review report (single default TabM)",
                 ha="center", fontsize=8, color=GREY)
        pdf.savefig(fig)
        plt.close(fig)

print("wrote", PDF, f"({len(PAGES)} pages)")
