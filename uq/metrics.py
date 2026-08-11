"""Uncertainty Quantification metrics for TabM."""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Union


def predictive_entropy(predictions: torch.Tensor) -> torch.Tensor:
    """Compute predictive entropy from ensemble predictions.
    
    Args:
        predictions: Tensor of shape (B, k, n_classes) containing softmax probabilities.
        
    Returns:
        Tensor of shape (B,) with entropy values.
    """
    mean_probs = predictions.mean(dim=1)  # (B, n_classes)
    entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
    return entropy


def mutual_information(predictions: torch.Tensor) -> torch.Tensor:
    """Compute mutual information (epistemic uncertainty).
    
    MI = H(mean) - mean(H(individual))
    
    Args:
        predictions: Tensor of shape (B, k, n_classes) containing softmax probabilities.
        
    Returns:
        Tensor of shape (B,) with MI values.
    """
    mean_probs = predictions.mean(dim=1)  # (B, n_classes)
    H_mean = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
    
    individual_entropies = -torch.sum(
        predictions * torch.log(predictions + 1e-10), dim=2
    )  # (B, k)
    mean_individual_entropy = individual_entropies.mean(dim=1)
    
    return H_mean - mean_individual_entropy


def predictive_variance(predictions: torch.Tensor) -> torch.Tensor:
    """Compute predictive variance for regression.
    
    Args:
        predictions: Tensor of shape (B, k, 1) or (B, k).
        
    Returns:
        Tensor of shape (B,) with variance values.
    """
    if predictions.dim() == 3 and predictions.shape[2] == 1:
        predictions = predictions.squeeze(2)
    return predictions.var(dim=1)


def entropy_of_mean_vs_mean_of_entropy(predictions: torch.Tensor) -> dict:
    """Compute both total uncertainty (entropy of mean) and data uncertainty (mean of entropy).
    
    Args:
        predictions: Tensor of shape (B, k, n_classes).
        
    Returns:
        Dict with 'total_uncertainty' and 'data_uncertainty' tensors.
    """
    mean_probs = predictions.mean(dim=1)
    total_uncertainty = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
    
    individual_entropies = -torch.sum(
        predictions * torch.log(predictions + 1e-10), dim=2
    )
    data_uncertainty = individual_entropies.mean(dim=1)
    
    return {
        "total_uncertainty": total_uncertainty,
        "data_uncertainty": data_uncertainty,
        "epistemic_uncertainty": total_uncertainty - data_uncertainty,  # Same as MI
    }


def compute_all_uq_metrics_classification(
    predictions: torch.Tensor,
) -> dict:
    """Compute all UQ metrics for classification in one call.
    
    Args:
        predictions: Tensor of shape (B, k, n_classes) with softmax probabilities.
        
    Returns:
        Dict with numpy arrays for each metric.
    """
    entropy = predictive_entropy(predictions)
    mi = mutual_information(predictions)
    decomp = entropy_of_mean_vs_mean_of_entropy(predictions)
    
    return {
        "predictive_entropy": entropy.cpu().numpy(),
        "mutual_information": mi.cpu().numpy(),
        "total_uncertainty": decomp["total_uncertainty"].cpu().numpy(),
        "data_uncertainty": decomp["data_uncertainty"].cpu().numpy(),
        "epistemic_uncertainty": decomp["epistemic_uncertainty"].cpu().numpy(),
    }


def compute_all_uq_metrics_regression(
    predictions: torch.Tensor,
) -> dict:
    """Compute all UQ metrics for regression in one call.
    
    Args:
        predictions: Tensor of shape (B, k, 1) or (B, k).
        
    Returns:
        Dict with numpy arrays for each metric.
    """
    if predictions.dim() == 3 and predictions.shape[2] == 1:
        pred_2d = predictions.squeeze(2)
    else:
        pred_2d = predictions
    
    mean_pred = pred_2d.mean(dim=1)
    variance = pred_2d.var(dim=1)
    std = pred_2d.std(dim=1)
    
    # Also compute min/max range as a simple uncertainty proxy
    min_pred = pred_2d.min(dim=1)[0]
    max_pred = pred_2d.max(dim=1)[0]
    range_pred = max_pred - min_pred
    
    return {
        "mean_prediction": mean_pred.cpu().numpy(),
        "predictive_variance": variance.cpu().numpy(),
        "predictive_std": std.cpu().numpy(),
        "prediction_range": range_pred.cpu().numpy(),
    }
