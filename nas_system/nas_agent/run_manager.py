"""
Run Manager - Utilities for managing training run directories.

This module provides helpers for creating unique run directories,
snapshotting configurations, and organizing experiment outputs.
"""

from pathlib import Path
import shutil


def ensure_runs_root(root: Path = Path("runs")) -> Path:
    """
    Ensure the root runs directory exists.
    
    Args:
        root: Path to the runs root directory (default: "runs")
        
    Returns:
        The absolute path to the runs directory
    """
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_run_dir(tag: str, root: Path = Path("runs")) -> Path:
    """
    Get or create a run directory for the given tag.
    
    Args:
        tag: Unique identifier for this run (e.g., "run_0000")
        root: Path to the runs root directory (default: "runs")
        
    Returns:
        The absolute path to the run directory
        
    Example:
        >>> run_dir = get_run_dir("run_0000")
        >>> print(run_dir)
        /path/to/project/runs/run_0000
    """
    ensure_runs_root(root)
    run_dir = (Path(root) / tag).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def snapshot_config(config_path: Path, run_dir: Path) -> Path:
    """
    Copy the configuration file to the run directory.
    
    This creates a snapshot of the exact configuration used for this run,
    enabling reproducibility and tracking of hyperparameter changes.
    
    Args:
        config_path: Path to the original config file
        run_dir: Path to the run directory where the config should be copied
        
    Returns:
        The path to the copied config file
        
    Example:
        >>> snapshot_config(Path("config.yaml"), Path("runs/run_0000"))
        PosixPath('runs/run_0000/config_used.yaml')
    """
    config_path = Path(config_path).resolve()
    run_dir = Path(run_dir).resolve()
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    dest_path = run_dir / "config_used.yaml"
    shutil.copy2(config_path, dest_path)
    
    return dest_path
