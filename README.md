# TabM-UQ: Uncertainty Quantification for TabM

**TabM-UQ** is a project that implements uncertainty quantification (UQ) and calibration metrics for the TabM (Tabular Deep Learning with Parameter-Efficient Ensembling) model, as described in Gorishniy et al. (ICLR 2025).

This project directly addresses the paper's stated future work: *"evaluate TabM for uncertainty estimation and out-of-distribution (OOD) detection on tabular data."*

---

## Overview

TabM produces **k predictions per object** (shape `(B, k, d_out)`) via parameter-efficient ensembling. This project:

1. **Reproduces TabM** on 10 tabular benchmarks from the paper's official dataset suite
2. **Derives uncertainty estimates** from the k ensemble predictions:
   - Predictive Entropy
   - Mutual Information (epistemic uncertainty)
   - Predictive Variance (regression)
3. **Evaluates calibration** via:
   - Expected Calibration Error (ECE)
   - Negative Log-Likelihood (NLL)
   - Reliability diagrams

---

## Architecture

| Component | Setting | Paper Reference |
|-----------|---------|-----------------|
| Model | TabM (without embeddings) | Default in paper |
| Ensemble size k | 16 | Section 6.1 |
| Blocks | 2 | Hyperparameter range |
| Hidden dim | 256 | Hyperparameter range |
| Optimizer | AdamW (lr=2e-3, wd=3e-4) | Default |
| Epochs | 100 (early stopping) | Standard protocol |

**Note:** We use TabM *without* feature embeddings (`LinearReLUEmbeddings`) for computational efficiency. This is explicitly documented as the paper's default baseline configuration. For the highest accuracy, one can enable `PiecewiseLinearEmbeddings` (TabM†).

---

## Uncertainty Quantification Metrics

For **classification** (binary/multiclass), given ensemble predictions `p_i` for i=1..k:

### 1. Predictive Entropy (Total Uncertainty)
```
H = -Σ_c p̄_c log(p̄_c), where p̄ = (1/k) Σ_i p_i
```

### 2. Mutual Information (Epistemic Uncertainty)
```
MI = H - (1/k) Σ_i H(p_i)
```
This measures disagreement *between* ensemble members.

### 3. Data Uncertainty
```
(1/k) Σ_i H(p_i)
```
This measures average uncertainty *within* each member.

For **regression**, given ensemble predictions `y_i`:

### 4. Predictive Variance
```
Var = (1/k) Σ_i (y_i - ȳ)^2
```

---

## Calibration Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| **ECE** | Σ_m (|B_m|/N) \|acc(B_m) - conf(B_m)\| | Measures alignment between confidence and accuracy |
| **NLL** | -(1/N) Σ_n log(p_{y_n}) | Measures probabilistic prediction quality |
| **Reliability Diagram** | Visual (15 bins) | Visualizes confidence vs. accuracy |

---

## Datasets

All 10 datasets are from the official TabM paper benchmark (50-dataset suite) and are accessed via the preprocessed `.npy` files from the authors' HuggingFace tarball.

| # | Dataset | Task | Train Size | Features |
|---|---------|------|------------|----------|
| 1 | adult | Binary Cls. | 26,048 | 6 num + 7 cat |
| 2 | bank-marketing | Binary Cls. | 7,404 | 7 num |
| 3 | california | Regression | 13,209 | 8 num |
| 4 | wine_quality | Regression | 4,547 | 11 num |
| 5 | wine | Binary Cls. | 1,787 | 11 num |
| 6 | phoneme | Binary Cls. | 2,220 | 5 num |
| 7 | churn | Binary Cls. | 6,400 | 7 num + 1 cat |
| 8 | MagicTelescope | Binary Cls. | 9,363 | 10 num |
| 9 | credit | Binary Cls. | 10,000 | 10 num |
| 10 | MiniBooNE | Binary Cls. | 50,000 | 50 num |
| 11 | Ailerons | Regression | 9,625 | 33 num |

---

## Project Structure

```
TabM/
├── data/
│   └── loaders.py          # Dataset loading from .npy files
├── models/
│   └── tabm_wrapper.py     # TabM training + inference wrapper
├── uq/
│   ├── metrics.py          # Entropy, MI, Variance
│   └── calibration.py      # ECE, NLL, reliability diagrams
├── experiments/
│   ├── train.py            # Single dataset trainer
│   └── evaluate_uq.py      # Post-training UQ analysis
├── results/
│   ├── raw/                # JSON results per run
│   ├── models/             # Saved model checkpoints
│   ├── predictions/        # Ensemble predictions (.npy)
│   └── figures/            # Reliability diagrams, entropy hists
├── run_all.py              # Parallel experiment runner
└── README.md               # This file
```

---

## Setup

### Prerequisites
- Python 3.11
- 16 GB RAM (for MiniBooNE)
- CPU-only (no GPU required)

### Install Dependencies

```bash
# Using the project's virtual environment
.venv/bin/pip install -r requirements.txt
```

Key dependencies:
- `torch>=2.0`
- `tabm>=0.0.3` (official PyPI package)
- `rtdl-num-embeddings>=0.0.12`
- `scikit-learn>=1.3`
- `matplotlib>=3.7`

### Download Datasets

```bash
# Clone the official TabM repo (for data access)
git clone https://github.com/yandex-research/tabm.git tabm_repo

# Download the 50-dataset preprocessed tarball
wget https://huggingface.co/datasets/rototoHF/tabm-data/resolve/main/data.tar \
  -O tabm_repo/local/tabm-data.tar
mkdir -p tabm_repo/data
tar -xf tabm_repo/local/tabm-data.tar -C tabm_repo/data
```

---

## Usage

### Run Full Experiment Suite

Train and evaluate UQ on all 10 datasets with 3 random seeds:

```bash
# Start the full run (runs in background)
nohup .venv/bin/python run_all.py \
  --k 16 \
  --n-blocks 2 \
  --d-block 256 \
  --epochs 100 \
  --patience 20 \
  --eval-every 5 \
  --workers 3 \
  --threads 5 \
  > results/full_run.log 2>&1 &
```

**Default settings** (can be omitted):
- k=16, n_blocks=2, d_block=256, epochs=100
- workers=3 (parallel seeds), threads=5 (per worker)

### Run Single Dataset

```bash
# Train
.venv/bin/python -m experiments.train phoneme \
  --seed 0 --k 16 --n-blocks 2 --d-block 256 --epochs 100

# Evaluate UQ
.venv/bin/python -m experiments.evaluate_uq phoneme --seed 0
```

### Check Status

```bash
# See live log
tail -f results/full_run.log

# Count completed results
ls results/raw/*_uq.json | wc -l

# See running processes
ps aux | grep "experiments.train" | grep -v grep

# CPU usage
top -p $(pgrep -d',' -f "experiments.train")
```

### Resume After Interrupt

The runner automatically skips completed seeds. Just re-run:

```bash
nohup .venv/bin/python run_all.py \
  --k 16 --n-blocks 2 --d-block 256 --epochs 100 \
  --workers 3 --threads 5 \
  > results/full_run.log 2>&1 &
```

---

## Results Format

Each run produces:

### 1. Raw Results JSON
`results/raw/{dataset}_seed{seed}_uq.json`

```json
{
  "dataset": "phoneme",
  "seed": 0,
  "task_type": "binclass",
  "accuracy": 0.8651,
  "ECE": 0.0298,
  "NLL": 0.3065,
  "uq": {
    "predictive_entropy_mean": 0.3295,
    "predictive_entropy_std": 0.2275,
    "mutual_information_mean": 0.0133,
    "mutual_information_std": 0.0152,
    "total_uncertainty_mean": 0.3295,
    "data_uncertainty_mean": 0.3162,
    "epistemic_uncertainty_mean": 0.0133
  }
}
```

### 2. Figures
- `results/figures/{dataset}_seed{seed}_reliability.png` — Reliability diagram
- `results/figures/{dataset}_seed{seed}_entropy_hist.png` — Entropy distribution

### 3. Model Checkpoints
- `results/models/{dataset}_seed{seed}.pt` — Full model state

### 4. Ensemble Predictions
- `results/predictions/{dataset}_seed{seed}_ensemble.npy` — (B, k, d_out) predictions
- `results/predictions/{dataset}_seed{seed}_y_test.npy` — Ground truth labels

### 5. Aggregated Summary
- `results/summary.json` — All results combined

---

## Execution & Runtime Performance

### Estimated vs. Measured Runtimes

*Note: Initial planning estimates assumed GPU-like speed. Actual benchmark execution on a 16-thread CPU (Intel i5-12500H, 3 parallel worker processes) yielded the following measured runtimes:*

| Dataset Category | Initial Estimate | Actual Measured Time (per seed) | Actual Wall-Clock (3 Parallel Seeds) | Representative Datasets |
|:---|:---|:---|:---|:---|
| **Small (<3K)** | ~2–3 min | 9.5 – 27 min | **~14 – 27 min** | `phoneme` (~14m), `wine` (~27m), `wine_quality` (~14m) |
| **Medium (3–15K)** | ~5–8 min | 58 min – 3.7 hours | **~1.4 – 3.7 hours** | `churn` (~2h), `bank-marketing` (~1.7h), `california` (~3.7h) |
| **Large (>15K)** | ~15–25 min | 3.5 – 12.6 hours | **~6.7 – 12.6 hours** | `adult` (~6.7h), `MiniBooNE` (~12.6h) |

**Total Suite Runtime:** **~30+ hours** wall-clock across 11 datasets × 3 seeds on CPU.

### Runtime Bottlenecks & CPU Execution Factors:
1. **CPU Tensor Math:** PyTorch forward and backward passes run 10×–30× slower on CPU compared to GPU acceleration.
2. **CPU Thread Contention:** Spawning 3 parallel PyTorch processes (`--workers 3`) each utilizing 5 threads (`--threads 5`) on a 16-thread CPU introduces cache thrashing and thread scheduling overhead.
3. **Dataset Scale & Ensemble Size:** Processing large datasets like `MiniBooNE` (50,000 samples × 50 features) over 100 epochs across $k=16$ ensemble heads involves updating millions of values per epoch.

---


## References

1. Gorishniy, Y., Kotelnikov, A., & Babenko, A. (2024). **TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling.** *ICLR 2025*.
   - arXiv: https://arxiv.org/abs/2410.24210
   - Code: https://github.com/yandex-research/tabm

2. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). **Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.** *NeurIPS 2017*.

3. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). **On Calibration of Modern Neural Networks.** *ICML 2017*.

---

## License

This project uses the official `tabm` package (Apache-2.0) and datasets from the TabM paper. Original paper code: https://github.com/yandex-research/tabm
