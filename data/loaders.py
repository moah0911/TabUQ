"""Data loaders for TabM preprocessed datasets."""
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from sklearn.preprocessing import OrdinalEncoder

# Map from canonical names to directory names in the tarball
DATASET_NAME_MAP = {
    "adult": "adult",
    "bank-marketing": "classif-num-medium-0-bank-marketing",
    "california": "california",
    "wine_quality": "regression-num-medium-0-wine_quality",
    "wine": "classif-num-medium-0-wine",
    "phoneme": "classif-num-medium-0-phoneme",
    "churn": "churn",
    "MagicTelescope": "classif-num-medium-0-MagicTelescope",
    "credit": "classif-num-medium-0-credit",
    "MiniBooNE": "classif-num-large-0-MiniBooNE",
    "Ailerons": "regression-num-medium-0-Ailerons",
}

DATASETS = list(DATASET_NAME_MAP.keys())


def load_dataset(
    dataset_name: str,
    data_root: str = "tabm_repo/data",
    device: str = "cpu",
) -> Tuple[Dict[str, np.ndarray], Dict[str, Optional[np.ndarray]], Dict[str, np.ndarray], Dict[str, Any]]:
    """Load a preprocessed TabM dataset.
    
    Args:
        dataset_name: Canonical dataset name (e.g., 'adult', 'credit')
        data_root: Root directory containing dataset folders
        device: torch device (for future tensor conversion)
        
    Returns:
        X_num: dict with 'train', 'val', 'test' numerical features
        X_cat: dict with 'train', 'val', 'test' categorical features (or None)
        Y: dict with 'train', 'val', 'test' targets
        info: dict with dataset metadata
    """
    dir_name = DATASET_NAME_MAP.get(dataset_name, dataset_name)
    path = Path(data_root) / dir_name
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {path}")
    
    with open(path / "info.json") as f:
        info = json.load(f)
    
    # Load numerical features
    X_num = {}
    for split in ["train", "val", "test"]:
        fpath = path / f"X_num_{split}.npy"
        X_num[split] = np.load(fpath).astype(np.float32)
    
    # Load categorical features (optional)
    X_cat_raw = {}
    for split in ["train", "val", "test"]:
        fpath = path / f"X_cat_{split}.npy"
        if fpath.exists():
            X_cat_raw[split] = np.load(fpath, allow_pickle=True)
        else:
            X_cat_raw[split] = None
    
    # Encode string categorical features to integer indices
    X_cat = {}
    if X_cat_raw["train"] is not None:
        if X_cat_raw["train"].dtype.kind in ('U', 'S', 'O'):  # String or object dtype
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            encoder.fit(X_cat_raw["train"])
            for split in ["train", "val", "test"]:
                encoded = encoder.transform(X_cat_raw[split])
                X_cat[split] = encoded.astype(np.int64)
        else:
            for split in ["train", "val", "test"]:
                X_cat[split] = X_cat_raw[split].astype(np.int64)
    else:
        for split in ["train", "val", "test"]:
            X_cat[split] = None
    
    # Load targets
    Y = {}
    for split in ["train", "val", "test"]:
        fpath = path / f"Y_{split}.npy"
        Y[split] = np.load(fpath)
        # Squeeze if shape is (N, 1)
        if Y[split].ndim == 2 and Y[split].shape[1] == 1:
            Y[split] = Y[split].squeeze(1)
    
    return X_num, X_cat, Y, info


def get_dataset_info(dataset_name: str, data_root: str = "tabm_repo/data") -> Dict[str, Any]:
    """Get dataset metadata without loading full arrays."""
    dir_name = DATASET_NAME_MAP.get(dataset_name, dataset_name)
    path = Path(data_root) / dir_name
    with open(path / "info.json") as f:
        return json.load(f)


def print_dataset_summary():
    """Print a summary of all available datasets."""
    print("=" * 60)
    print("TabM-UQ Dataset Summary")
    print("=" * 60)
    for name in DATASETS:
        try:
            info = get_dataset_info(name)
            task = info.get("task_type", "unknown")
            n_num = info.get("n_num_features", 0)
            n_cat = info.get("n_cat_features", 0)
            n_bin = info.get("n_bin_features", 0)
            train_n = info.get("train_size", 0)
            print(f"{name:20s} | {task:12s} | num={n_num:3d} cat={n_cat:3d} bin={n_bin:3d} | train={train_n:6d}")
        except Exception as e:
            print(f"{name:20s} | ERROR: {e}")
    print("=" * 60)


if __name__ == "__main__":
    print_dataset_summary()
