# AGENTS.md — TabM-UQ Project

## Project Overview

**TabM-UQ** implements uncertainty quantification (UQ) for the TabM model (Gorishniy et al., ICLR 2025). This is an FYP (Final Year Project) addressing the paper's stated future work on UQ and OOD detection.

**Key files:**
- `plan.md` — Original project plan with timeline
- `docs/reproducibility_report_small_config.md` — Results from Phase 1 (small config)
- `README.md` — User-facing documentation
- `run_all.py` — Parallel experiment runner

---

## Architecture

### TabM Model
- **Package:** `tabm` (PyPI, official)
- **Variants:**
  - **TabM** (default): No embeddings, raw features
  - **TabM†** (enhanced): With `LinearReLUEmbeddings` or `PiecewiseLinearEmbeddings`
- **Current config (Phase 1):** n_blocks=1, d_block=128, k=16, NO embeddings
- **Target config (Phase 2):** n_blocks=3, d_block=512, k=32, LinearReLUEmbeddings

### UQ Pipeline
- `uq/metrics.py` — Entropy, MI, Variance
- `uq/calibration.py` — ECE, NLL, reliability diagrams

---

## File Structure

```
TabM/
├── plan.md                           # Project plan (8-month timeline)
├── README.md                         # User documentation
├── requirements.txt                  # Dependencies
├── AGENTS.md                         # This file
├── docs/
│   └── reproducibility_report_small_config.md  # Phase 1 results
├── data/
│   └── loaders.py                    # Dataset loading from .npy files
├── models/
│   └── tabm_wrapper.py               # TabM training + inference
├── uq/
│   ├── metrics.py                    # UQ metrics (entropy, MI, variance)
│   └── calibration.py                # Calibration metrics + plots
├── experiments/
│   ├── train.py                      # Single dataset trainer
│   └── evaluate_uq.py              # Post-training UQ analysis
├── results/
│   ├── raw/                          # JSON results per run
│   ├── models/                       # Saved checkpoints
│   ├── predictions/                  # Ensemble predictions (.npy)
│   └── figures/                      # Reliability diagrams + entropy hists
├── tabm_repo/                        # Official TabM repo (data access)
│   ├── data/                         # 50 preprocessed datasets
│   └── local/
│       └── tabm-data.tar             # HuggingFace tarball
└── run_all.py                        # Parallel experiment runner
```

---

## Key Commands

### Run Full Experiment
```bash
# Phase 1 (small config — CPU fast)
nohup .venv/bin/python run_all.py \
  --k 16 --n-blocks 1 --d-block 128 --epochs 100 \
  --workers 3 --threads 5 \
  > results/full_run.log 2>&1 &

# Phase 2 (exact config — paper defaults)
nohup .venv/bin/python run_all.py \
  --k 32 --n-blocks 3 --d-block 512 --epochs 200 \
  --workers 3 --threads 5 \
  > results/full_run.log 2>&1 &
```

### Check Status
```bash
bash check_status.sh
# Or manually:
ls results/raw/*_uq.json | wc -l    # Count completed
tail -f results/full_run.log         # Watch live log
```

### Resume After Interrupt
```bash
# Just rerun — completed seeds are automatically skipped
nohup .venv/bin/python run_all.py \
  --k 16 --n-blocks 1 --d-block 128 --epochs 100 \
  --workers 3 --threads 5 \
  > results/full_run.log 2>&1 &
```

### Run Single Dataset
```bash
# Train
.venv/bin/python -m experiments.train phoneme \
  --seed 0 --k 16 --n-blocks 1 --d-block 128 --epochs 100

# Evaluate UQ
.venv/bin/python -m experiments.evaluate_uq phoneme --seed 0
```

---

## Datasets

All from the **TabM paper's official 50-dataset suite** (HuggingFace: `rototoHF/tabm-data`):

1. adult — binclass, 26K train
2. bank-marketing — binclass, 7K train
3. california — regression, 13K train
4. wine_quality — regression, 4.5K train
5. wine — binclass, 1.7K train
6. phoneme — binclass, 2.2K train
7. churn — binclass, 6.4K train
8. MagicTelescope — binclass, 9.3K train
9. credit — binclass, 10K train
10. MiniBooNE — binclass, 50K train
11. Ailerons — regression, 9.6K train

**Loading:** `from data.loaders import load_dataset; X_num, X_cat, Y, info = load_dataset('adult')`

---

## Environment

```bash
# Virtual environment
.venv/bin/python --version  # 3.11.15

# Key packages
.venv/bin/pip list | grep -E "torch|tabm|sklearn|numpy"

# CPU threads per worker (set automatically)
export OMP_NUM_THREADS=5
export MKL_NUM_THREADS=5
```

---

## Results Format

### JSON (`results/raw/{dataset}_seed{seed}_uq.json`)
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
    "mutual_information_mean": 0.0133
  }
}
```

### Figures
- `results/figures/{dataset}_seed{seed}_reliability.png`
- `results/figures/{dataset}_seed{seed}_entropy_hist.png`

### Checkpoints
- `results/models/{dataset}_seed{seed}.pt` (PyTorch state_dict)

---

## Important Notes

1. **Embeddings:** We currently use TabM **without** embeddings for speed. To enable `LinearReLUEmbeddings`, edit `models/tabm_wrapper.py`:
   ```python
   from rtdl_num_embeddings import LinearReLUEmbeddings
   num_embeddings = LinearReLUEmbeddings(n_num_features)
   ```

2. **Parallel execution:** `run_all.py` spawns 3 parallel processes (seeds). Each process gets 5 CPU threads via environment variables.

3. **Resumability:** Experiments check for existing `results/raw/{dataset}_seed{seed}_uq.json` before running. Interrupted runs resume automatically.

4. **Categorical features:** String categories are encoded to integers via `OrdinalEncoder` in `data/loaders.py`.

5. **Batch size:** Auto-scaled based on train set size (256-2048).

---

## Current Status (as of last update)

- ✅ Phase 1 complete: 11 datasets × 3 seeds with small config
- 🔄 Phase 2 in progress: Exact paper config (k=32, n_blocks=3, d_block=512, with embeddings)
- ⏳ M3-M8 pending

---

## Dependencies

```
torch>=2.0
tabm>=0.0.3
rtdl-num-embeddings>=0.0.12
scikit-learn>=1.3
numpy>=1.24
matplotlib>=3.7
```

---

## References

- TabM Paper: https://arxiv.org/abs/2410.24210
- TabM Code: https://github.com/yandex-research/tabm
- Datasets: https://huggingface.co/datasets/rototoHF/tabm-data
