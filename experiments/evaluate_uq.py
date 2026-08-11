"""Evaluation script: compute UQ metrics and calibration from saved predictions."""
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from uq.metrics import (
    compute_all_uq_metrics_classification,
    compute_all_uq_metrics_regression,
)
from uq.calibration import (
    compute_calibration_metrics,
    plot_reliability_diagram,
    plot_entropy_histogram,
)


def evaluate_uq_on_dataset(
    dataset_name: str,
    seed: int = 0,
    n_bins: int = 15,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compute UQ and calibration metrics from saved predictions.
    
    Args:
        dataset_name: Canonical dataset name.
        seed: Random seed.
        n_bins: Number of bins for ECE.
        verbose: Print progress.
        
    Returns:
        Dict with all UQ and calibration metrics.
    """
    preds_dir = Path("results/predictions")
    
    # Load predictions
    ensemble_preds = np.load(preds_dir / f"{dataset_name}_seed{seed}_ensemble.npy")
    y_test = np.load(preds_dir / f"{dataset_name}_seed{seed}_y_test.npy")
    
    # Load results to get task type
    results_path = Path("results/raw") / f"{dataset_name}_seed{seed}.json"
    with open(results_path) as f:
        train_results = json.load(f)
    task_type = train_results["task_type"]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"UQ Evaluation: {dataset_name} | Seed: {seed} | Task: {task_type}")
        print(f"Ensemble predictions shape: {ensemble_preds.shape}")
        print(f"Test samples: {len(y_test)}")
        print(f"{'='*60}")
    
    import torch
    
    if task_type in ["binclass", "multiclass"]:
        # Convert logits to probabilities
        # ensemble_preds shape: (B, k, d_out)
        preds_tensor = torch.from_numpy(ensemble_preds).float()
        
        if task_type == "binclass":
            # Sigmoid then add channel dimension if needed
            probs = torch.sigmoid(preds_tensor)  # (B, k, 1)
            # For binary, we need 2-class probabilities for entropy computation
            probs_2class = torch.stack([1 - probs, probs], dim=2)  # (B, k, 2, 1) -> wait
            # Actually for binary classification with single output:
            # We can just compute entropy as: -p*log(p) - (1-p)*log(1-p)
            # But our functions expect multi-class probabilities
            # So let's stack: probs_class1 = probs, probs_class0 = 1 - probs
            probs_bk2 = torch.cat([1 - probs, probs], dim=2)  # (B, k, 2)
        else:
            # Multiclass: softmax over last dim
            probs_bk2 = torch.softmax(preds_tensor, dim=2)  # (B, k, n_classes)
        
        uq_metrics = compute_all_uq_metrics_classification(probs_bk2)
        
        # Get mean probabilities for calibration
        mean_probs = probs_bk2.mean(dim=1).numpy()  # (B, n_classes)
        
        # Calibration metrics
        if task_type == "binclass":
            # For binary, mean_probs is (B, 2), y_test should be int 0/1
            y_test_int = y_test.astype(int)
            cal_metrics = compute_calibration_metrics(y_test_int, mean_probs, n_bins)
            accuracy = np.mean((mean_probs[:, 1] > 0.5).astype(int) == y_test_int)
        else:
            y_test_int = y_test.astype(int)
            cal_metrics = compute_calibration_metrics(y_test_int, mean_probs, n_bins)
            accuracy = np.mean(mean_probs.argmax(axis=1) == y_test_int)
        
        # Plot reliability diagram
        fig_dir = Path("results/figures")
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_reliability_diagram(
            y_test_int, mean_probs, n_bins,
            title=f"{dataset_name} (seed={seed}) - Reliability Diagram",
            save_path=str(fig_dir / f"{dataset_name}_seed{seed}_reliability.png"),
        )
        
        # Plot entropy histogram
        plot_entropy_histogram(
            uq_metrics["predictive_entropy"],
            title=f"{dataset_name} (seed={seed}) - Predictive Entropy",
            save_path=str(fig_dir / f"{dataset_name}_seed{seed}_entropy_hist.png"),
        )
        
        results = {
            "dataset": dataset_name,
            "seed": seed,
            "task_type": task_type,
            "accuracy": float(accuracy),
            **cal_metrics,
            "uq": {
                "predictive_entropy_mean": float(np.mean(uq_metrics["predictive_entropy"])),
                "predictive_entropy_std": float(np.std(uq_metrics["predictive_entropy"])),
                "mutual_information_mean": float(np.mean(uq_metrics["mutual_information"])),
                "mutual_information_std": float(np.std(uq_metrics["mutual_information"])),
                "total_uncertainty_mean": float(np.mean(uq_metrics["total_uncertainty"])),
                "data_uncertainty_mean": float(np.mean(uq_metrics["data_uncertainty"])),
                "epistemic_uncertainty_mean": float(np.mean(uq_metrics["epistemic_uncertainty"])),
            },
        }
        
    elif task_type == "regression":
        preds_tensor = torch.from_numpy(ensemble_preds).float()
        uq_metrics = compute_all_uq_metrics_regression(preds_tensor)
        
        # Regression NLL (Gaussian assumption)
        from uq.calibration import negative_log_likelihood_regression
        nll = negative_log_likelihood_regression(
            y_test, uq_metrics["mean_prediction"], uq_metrics["predictive_variance"]
        )
        
        rmse = np.sqrt(np.mean((uq_metrics["mean_prediction"] - y_test) ** 2))
        
        results = {
            "dataset": dataset_name,
            "seed": seed,
            "task_type": task_type,
            "rmse": float(rmse),
            "NLL": float(nll),
            "uq": {
                "predictive_variance_mean": float(np.mean(uq_metrics["predictive_variance"])),
                "predictive_variance_std": float(np.std(uq_metrics["predictive_variance"])),
                "predictive_std_mean": float(np.mean(uq_metrics["predictive_std"])),
                "prediction_range_mean": float(np.mean(uq_metrics["prediction_range"])),
            },
        }
    
    if verbose:
        print("\nResults:")
        for key, val in results.items():
            if key != "uq":
                print(f"  {key}: {val}")
        print("  UQ Metrics:")
        for key, val in results["uq"].items():
            print(f"    {key}: {val:.4f}")
    
    # Save UQ results
    results_dir = Path("results/raw")
    uq_path = results_dir / f"{dataset_name}_seed{seed}_uq.json"
    with open(uq_path, "w") as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print(f"\nUQ results saved to: {uq_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Dataset name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    
    evaluate_uq_on_dataset(
        args.dataset,
        seed=args.seed,
        n_bins=args.n_bins,
        verbose=not args.quiet,
    )
