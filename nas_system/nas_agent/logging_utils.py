"""
Logging Utilities - Functions for saving and loading metrics and training history.

This module provides JSON-based logging for experiment metrics and per-epoch
training history, making it easy for LLM agents to read and analyze results.
"""

from pathlib import Path
import json
from typing import Dict, Any


def save_metrics(run_dir: Path, metrics: Dict[str, Any]) -> None:
    """
    Save metrics dictionary to metrics.json in the run directory.
    
    Args:
        run_dir: Path to the run directory
        metrics: Dictionary containing evaluation metrics
                 (e.g., best_epoch, val_mse, test_mse, train_time_sec)
    
    Example:
        >>> metrics = {
        ...     "best_epoch": 17,
        ...     "val_mse": 0.324,
        ...     "test_mse": 0.341,
        ...     "train_mse": 0.245,
        ...     "train_time_sec": 127.5
        ... }
        >>> save_metrics(Path("runs/run_0000"), metrics)
    """
    run_dir = Path(run_dir).resolve()
    metrics_path = run_dir / "metrics.json"
    
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def save_history(run_dir: Path, history: Dict[str, Any]) -> None:
    """
    Save training history dictionary to history.json in the run directory.
    
    Args:
        run_dir: Path to the run directory
        history: Dictionary containing per-epoch training history
                 with keys: best_epoch, epochs (list of epoch dicts)
    
    Example:
        >>> history = {
        ...     "best_epoch": 17,
        ...     "epochs": [
        ...         {"epoch": 0, "train_mse": 1.02, "val_mse": 1.05, "lr": 0.001},
        ...         {"epoch": 1, "train_mse": 0.85, "val_mse": 0.89, "lr": 0.001},
        ...         # ...
        ...     ]
        ... }
        >>> save_history(Path("runs/run_0000"), history)
    """
    run_dir = Path(run_dir).resolve()
    history_path = run_dir / "history.json"
    
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def load_metrics(run_dir: Path) -> Dict[str, Any]:
    """
    Load metrics dictionary from metrics.json in the run directory.
    
    Args:
        run_dir: Path to the run directory
        
    Returns:
        Dictionary containing evaluation metrics
        
    Raises:
        FileNotFoundError: If metrics.json does not exist
    
    Example:
        >>> metrics = load_metrics(Path("runs/run_0000"))
        >>> print(metrics["test_mse"])
        0.341
    """
    run_dir = Path(run_dir).resolve()
    metrics_path = run_dir / "metrics.json"
    
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    
    with open(metrics_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_history(run_dir: Path) -> Dict[str, Any]:
    """
    Load training history dictionary from history.json in the run directory.
    
    Args:
        run_dir: Path to the run directory
        
    Returns:
        Dictionary containing per-epoch training history
        
    Raises:
        FileNotFoundError: If history.json does not exist
    
    Example:
        >>> history = load_history(Path("runs/run_0000"))
        >>> print(len(history["epochs"]))
        30
    """
    run_dir = Path(run_dir).resolve()
    history_path = run_dir / "history.json"
    
    if not history_path.exists():
        raise FileNotFoundError(f"History file not found: {history_path}")
    
    with open(history_path, 'r', encoding='utf-8') as f:
        return json.load(f)
