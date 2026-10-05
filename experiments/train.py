"""Training script for a single dataset."""
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loaders import load_dataset, DATASETS
from models.tabm_wrapper import TabMWrapper


def train_on_dataset(
    dataset_name: str,
    seed: int = 0,
    data_root: str = "tabm_repo/data",
    epochs: int = 200,
    batch_size: int = None,
    patience: int = 20,
    eval_every: int = 5,
    k: int = 32,
    n_blocks: int = 3,
    d_block: int = 512,
    use_embeddings: bool = False,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train TabM on a single dataset and return results.
    
    Args:
        dataset_name: Canonical dataset name.
        seed: Random seed.
        data_root: Path to preprocessed data.
        epochs: Max training epochs.
        batch_size: Batch size. If None, auto-selected based on dataset size.
        patience: Early stopping patience.
        eval_every: Evaluate validation every N epochs.
        k: Ensemble size.
        n_blocks: Number of MLP blocks.
        d_block: Hidden dimension.
        use_embeddings: Whether to use LinearReLUEmbeddings (TabM†).
        device: Training device.
        verbose: Print progress.
        
    Returns:
        Dict with training history and final metrics.
    """
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    
    # Load data
    X_num, X_cat, Y, info = load_dataset(dataset_name, data_root)
    
    # Auto batch size based on dataset size
    if batch_size is None:
        train_size = len(Y["train"])
        if train_size < 2000:
            batch_size = 256
        elif train_size < 10000:
            batch_size = 512
        elif train_size < 50000:
            batch_size = 1024
        else:
            batch_size = 2048
        if verbose:
            print(f"Auto batch_size: {batch_size} (train_size={train_size})")
    task_type = info["task_type"]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name} | Task: {task_type} | Seed: {seed}")
        print(f"Train: {len(Y['train'])} | Val: {len(Y['val'])} | Test: {len(Y['test'])}")
        print(f"Num features: {X_num['train'].shape[1]} | Cat features: {X_cat['train'].shape[1] if X_cat['train'] is not None else 0}")
        print(f"{'='*60}")
    
    # Determine output dimension and task
    if task_type == "binclass":
        d_out = 1
        # Ensure binary targets are 0/1
        for split in ["train", "val", "test"]:
            if Y[split].ndim == 1:
                Y[split] = Y[split].astype(np.float32)
    elif task_type == "multiclass":
        d_out = int(np.max(Y['train'])) + 1
        for split in ["train", "val", "test"]:
            Y[split] = Y[split].astype(np.int64)
    elif task_type == "regression":
        d_out = 1
        for split in ["train", "val", "test"]:
            Y[split] = Y[split].astype(np.float32)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    
    # Get categorical cardinalities
    cat_cardinalities = None
    if X_cat["train"] is not None:
        cat_cardinalities = []
        for i in range(X_cat["train"].shape[1]):
            # Count unique values across all splits, ignoring -1 (unknown)
            all_vals = np.concatenate([X_cat[split][:, i] for split in ["train", "val", "test"]])
            unique_vals = np.unique(all_vals)
            unique_vals = unique_vals[unique_vals != -1]
            cat_cardinalities.append(int(len(unique_vals)))
    
    # Initialize model
    model = TabMWrapper(
        n_num_features=X_num["train"].shape[1],
        cat_cardinalities=cat_cardinalities,
        d_out=d_out,
        task_type=task_type,
        k=k,
        n_blocks=n_blocks,
        d_block=d_block,
        device=device,
        use_embeddings=use_embeddings,
    )
    
    # Train
    history = model.fit(
        X_num["train"], X_cat["train"], Y["train"],
        X_num["val"], X_cat["val"], Y["val"],
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        eval_every=eval_every,
        verbose=verbose,
    )
    
    # Evaluate on test set
    test_loss, test_metric = model.evaluate(
        X_num["test"], X_cat["test"], Y["test"],
        batch_size=batch_size,
    )
    
    # Get ensemble predictions for UQ
    ensemble_preds = model.predict_ensemble(
        X_num["test"], X_cat["test"], batch_size=batch_size
    )
    
    results = {
        "dataset": dataset_name,
        "seed": seed,
        "task_type": task_type,
        "k": k,
        "n_blocks": n_blocks,
        "d_block": d_block,
        "test_loss": float(test_loss),
        "test_metric": float(test_metric),
        "best_epoch": int(np.argmin(history["val_loss"])),
        "history": {
            "train_loss": [float(x) for x in history["train_loss"]],
            "val_loss": [float(x) for x in history["val_loss"]],
            "val_metric": [float(x) for x in history["val_metric"]],
        },
        "ensemble_predictions_shape": ensemble_preds.shape,
    }
    
    # Save model
    model_dir = Path("results/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir / f"{dataset_name}_seed{seed}.pt"))
    
    # Save ensemble predictions for UQ analysis
    preds_dir = Path("results/predictions")
    preds_dir.mkdir(parents=True, exist_ok=True)
    np.save(preds_dir / f"{dataset_name}_seed{seed}_ensemble.npy", ensemble_preds)
    np.save(preds_dir / f"{dataset_name}_seed{seed}_y_test.npy", Y["test"])
    
    if verbose:
        print(f"\nFinal Test Metric: {test_metric:.4f}")
        print(f"Best Epoch: {results['best_epoch']}")
        print(f"Model saved to: {model_dir / f'{dataset_name}_seed{seed}.pt'}")
    
    return results, ensemble_preds, Y["test"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Dataset name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--n-blocks", type=int, default=3)
    parser.add_argument("--d-block", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use-embeddings", action="store_true", help="Use LinearReLUEmbeddings (TabM†)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    
    results, preds, y_test = train_on_dataset(
        args.dataset,
        seed=args.seed,
        k=args.k,
        n_blocks=args.n_blocks,
        d_block=args.d_block,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        eval_every=args.eval_every,
        use_embeddings=args.use_embeddings,
        device=args.device,
        verbose=not args.quiet,
    )
    
    # Save results JSON
    results_dir = Path("results/raw")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{args.dataset}_seed{args.seed}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_dir / f'{args.dataset}_seed{args.seed}.json'}")
