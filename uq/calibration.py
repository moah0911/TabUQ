"""Calibration metrics and visualization for TabM."""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, Optional


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE).
    
    Args:
        y_true: Array of shape (N,) with true class labels (integers).
        y_prob: Array of shape (N, n_classes) with predicted probabilities.
        n_bins: Number of bins for calibration.
        
    Returns:
        ECE value (float).
    """
    n_samples = len(y_true)
    if n_samples == 0:
        return 0.0
    
    # Get predicted class and confidence
    y_pred = y_prob.argmax(axis=1)
    confidences = y_prob.max(axis=1)
    accuracies = (y_pred == y_true).astype(float)
    
    # Equal-width bins
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]
        
        # Include right edge for last bin
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)
    
    return float(ece)


def negative_log_likelihood(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """Compute Negative Log-Likelihood.
    
    Args:
        y_true: Array of shape (N,) with true class labels.
        y_prob: Array of shape (N, n_classes) with predicted probabilities.
        
    Returns:
        NLL value.
    """
    n_samples = len(y_true)
    if n_samples == 0:
        return 0.0
    
    # Gather probabilities for true classes
    true_class_probs = y_prob[np.arange(n_samples), y_true]
    nll = -np.mean(np.log(true_class_probs + eps))
    return float(nll)


def negative_log_likelihood_regression(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_var: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """Compute Negative Log-Likelihood for regression (Gaussian assumption).
    
    Args:
        y_true: Array of shape (N,).
        y_pred_mean: Array of shape (N,).
        y_pred_var: Array of shape (N,) with predictive variance.
        
    Returns:
        NLL value.
    """
    var = np.maximum(y_pred_var, eps)
    nll = 0.5 * np.mean(
        np.log(2 * np.pi * var) + (y_true - y_pred_mean) ** 2 / var
    )
    return float(nll)


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, float]:
    """Compute all calibration metrics for classification.
    
    Args:
        y_true: True labels, shape (N,).
        y_prob: Predicted probabilities, shape (N, n_classes).
        n_bins: Number of bins for ECE.
        
    Returns:
        Dict with 'ECE' and 'NLL'.
    """
    return {
        "ECE": expected_calibration_error(y_true, y_prob, n_bins),
        "NLL": negative_log_likelihood(y_true, y_prob),
    }


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
    title: str = "Reliability Diagram",
    save_path: Optional[str] = None,
) -> None:
    """Plot and optionally save a reliability diagram.
    
    Args:
        y_true: True labels, shape (N,).
        y_prob: Predicted probabilities, shape (N, n_classes).
        n_bins: Number of bins.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    y_pred = y_prob.argmax(axis=1)
    confidences = y_prob.max(axis=1)
    accuracies = (y_pred == y_true).astype(float)
    
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        if i == n_bins - 1:
            in_bin = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
        else:
            in_bin = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        
        bin_counts[i] = np.sum(in_bin)
        if bin_counts[i] > 0:
            bin_accuracies[i] = np.mean(accuracies[in_bin])
            bin_confidences[i] = np.mean(confidences[in_bin])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot bars
    ax.bar(bin_centers, bin_accuracies, width=1.0 / n_bins, alpha=0.6, edgecolor='black', label="Accuracy")
    ax.bar(bin_centers, bin_confidences, width=1.0 / n_bins, alpha=0.3, edgecolor='red', label="Confidence", color='red')
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration")
    
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_entropy_histogram(
    entropy: np.ndarray,
    title: str = "Predictive Entropy Distribution",
    save_path: Optional[str] = None,
) -> None:
    """Plot histogram of predictive entropy.
    
    Args:
        entropy: Array of entropy values, shape (N,).
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    n_bins = min(50, max(2, len(np.unique(entropy))))
    ax.hist(entropy, bins=n_bins, alpha=0.7, edgecolor='black', color='steelblue')
    ax.axvline(np.mean(entropy), color='red', linestyle='--', linewidth=2, label=f"Mean: {np.mean(entropy):.3f}")
    ax.axvline(np.median(entropy), color='orange', linestyle='--', linewidth=2, label=f"Median: {np.median(entropy):.3f}")
    ax.set_xlabel("Predictive Entropy")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
