"""
History Summary - Generate textual summaries of learning curves for LLM consumption.

This module analyzes training history and produces concise, human-readable
summaries that describe learning dynamics, overfitting patterns, and convergence
behavior.
"""

from typing import Dict, Any, List


def build_history_summary(history: Dict[str, Any]) -> str:
    """
    Generate a textual summary of training history for LLM analysis.
    
    Analyzes per-epoch training and validation metrics to identify patterns such as:
    - Monotonic improvement vs. overfitting
    - Early stopping behavior
    - Train/val gap evolution
    - Convergence characteristics
    
    Args:
        history: Dictionary with keys:
                 - "best_epoch": int
                 - "epochs": list of dicts with keys "epoch", "train_mse", "val_mse", "lr"
    
    Returns:
        A multi-sentence textual summary describing learning dynamics
    
    Example:
        >>> history = {
        ...     "best_epoch": 17,
        ...     "epochs": [
        ...         {"epoch": 0, "train_mse": 1.02, "val_mse": 1.05},
        ...         # ... more epochs
        ...         {"epoch": 29, "train_mse": 0.24, "val_mse": 0.40}
        ...     ]
        ... }
        >>> summary = build_history_summary(history)
        >>> print(summary)
        30 epochs, best_epoch=17. train_mse decreased from 1.02 to 0.24. ...
    """
    if not history or "epochs" not in history or len(history["epochs"]) == 0:
        return "No training history available."
    
    epochs = history["epochs"]
    best_epoch = history.get("best_epoch", 0)
    num_epochs = len(epochs)
    
    # Extract key statistics
    first_epoch = epochs[0]
    last_epoch = epochs[-1]
    
    train_mse_first = first_epoch.get("train_mse", 0.0)
    train_mse_last = last_epoch.get("train_mse", 0.0)
    val_mse_first = first_epoch.get("val_mse", 0.0)
    val_mse_last = last_epoch.get("val_mse", 0.0)
    
    # Find best validation MSE
    val_mses = [epoch.get("val_mse", float('inf')) for epoch in epochs]
    best_val_mse = min(val_mses) if val_mses else 0.0
    best_val_epoch = val_mses.index(best_val_mse) if val_mses else 0
    
    # Build summary components
    summary_parts = []
    
    # Basic info
    summary_parts.append(f"{num_epochs} epochs, best_epoch={best_epoch}.")
    
    # Train MSE trajectory
    if train_mse_first > 0:
        train_direction = "decreased" if train_mse_last < train_mse_first else "increased"
        summary_parts.append(
            f"train_mse {train_direction} from {train_mse_first:.4f} to {train_mse_last:.4f}."
        )
    
    # Val MSE trajectory and best point
    if val_mse_first > 0:
        summary_parts.append(
            f"val_mse started at {val_mse_first:.4f}, "
            f"reached minimum of {best_val_mse:.4f} at epoch {best_val_epoch}, "
            f"ended at {val_mse_last:.4f}."
        )
    
    # Analyze overfitting pattern
    overfitting_analysis = _analyze_overfitting(epochs, best_val_epoch)
    if overfitting_analysis:
        summary_parts.append(overfitting_analysis)
    
    # Analyze train/val gap
    gap_analysis = _analyze_train_val_gap(epochs)
    if gap_analysis:
        summary_parts.append(gap_analysis)
    
    return " ".join(summary_parts)


def _analyze_overfitting(epochs: List[Dict[str, Any]], best_val_epoch: int) -> str:
    """
    Analyze whether the model shows signs of overfitting.
    
    Args:
        epochs: List of epoch dictionaries
        best_val_epoch: Index of the epoch with best validation MSE
    
    Returns:
        A string describing overfitting patterns, or empty string if none detected
    """
    num_epochs = len(epochs)
    
    if best_val_epoch < num_epochs - 1:
        # Validation MSE increased after best epoch
        val_mses = [epoch.get("val_mse", 0.0) for epoch in epochs]
        val_at_best = val_mses[best_val_epoch] if best_val_epoch < len(val_mses) else 0.0
        val_at_end = val_mses[-1]
        
        if val_at_end > val_at_best * 1.05:  # 5% threshold
            return (
                f"This suggests overfitting: validation loss starts rising after epoch {best_val_epoch}, "
                f"increasing from {val_at_best:.4f} to ~{val_at_end:.4f} by epoch {num_epochs - 1}."
            )
        else:
            return (
                f"Validation loss plateaued after epoch {best_val_epoch} "
                f"with minimal change to {val_at_end:.4f}."
            )
    elif best_val_epoch == num_epochs - 1:
        return "Model was still improving at the last epoch; more training may help."
    
    return ""


def _analyze_train_val_gap(epochs: List[Dict[str, Any]]) -> str:
    """
    Analyze the evolution of the train/val gap throughout training.
    
    Args:
        epochs: List of epoch dictionaries
    
    Returns:
        A string describing train/val gap patterns, or empty string if insufficient data
    """
    if len(epochs) < 2:
        return ""
    
    # Calculate gap at first and last epoch
    first_train = epochs[0].get("train_mse", 0.0)
    first_val = epochs[0].get("val_mse", 0.0)
    last_train = epochs[-1].get("train_mse", 0.0)
    last_val = epochs[-1].get("val_mse", 0.0)
    
    if first_train > 0 and first_val > 0 and last_train > 0 and last_val > 0:
        first_gap = abs(first_val - first_train)
        last_gap = abs(last_val - last_train)
        
        if last_gap > first_gap * 1.5:  # Gap increased significantly
            return (
                f"The train/val gap widened from {first_gap:.4f} to {last_gap:.4f}, "
                f"indicating potential overfitting."
            )
        elif last_gap < first_gap * 0.7:  # Gap decreased
            return (
                f"The train/val gap narrowed from {first_gap:.4f} to {last_gap:.4f}, "
                f"suggesting good generalization."
            )
        else:
            return f"The train/val gap remained relatively stable at ~{last_gap:.4f}."
    
    return ""
