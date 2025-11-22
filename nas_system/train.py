"""
Training Pipeline - Main script for training time series forecasting models with real Informer.

This script implements the actual training logic from the notebook, using
the real Informer model from Informer2020 and ETTh1 dataset.

Usage:
    python train.py --config config.yaml --tag run_0000
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import importlib.util
import sys

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from model import build_model
from nas_agent import (
    get_run_dir,
    snapshot_config,
    save_metrics,
    save_history,
)


class TimeSeriesDataset(Dataset):
    """
    Real time series dataset for ETTh1/ETTh2 data.
    
    This implementation matches the notebook and uses StandardScaler
    for normalization and supports label_len for decoder input.
    """
    
    def __init__(self, data, seq_len: int, label_len: int, pred_len: int, 
                 features: list, target: str = 'OT', flag: str = 'train', 
                 scale: bool = True, scaler=None):
        """
        Initialize time series dataset.
        
        Args:
            data: Pandas DataFrame with the time series data
            seq_len: Input sequence length
            label_len: Decoder start token length
            pred_len: Prediction length
            features: List of feature column names
            target: Target column name
            flag: One of "train", "val", "test"
            scale: Whether to apply StandardScaler
            scaler: Pre-fitted scaler (for val/test sets)
        """
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.features = features
        self.target = target
        self.flag = flag
        
        # Extract features and target
        self.data_x = data[features].values
        self.data_y = data[target].values
        
        # Apply scaling
        if scale:
            if scaler is None:
                self.scaler = StandardScaler()
                self.data_x = self.scaler.fit_transform(self.data_x)
            else:
                self.scaler = scaler
                self.data_x = self.scaler.transform(self.data_x)
        else:
            self.scaler = None
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_x[r_begin:r_end]
        
        # Time marks (placeholder - zeros for now)
        seq_x_mark = np.zeros((self.seq_len, 4))
        seq_y_mark = np.zeros((self.label_len + self.pred_len, 4))
        
        return (torch.FloatTensor(seq_x), 
                torch.FloatTensor(seq_y),
                torch.FloatTensor(seq_x_mark),
                torch.FloatTensor(seq_y_mark))


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from Python file."""
    # Import config module dynamically
    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config from {config_path}")
    
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config_module
    spec.loader.exec_module(config_module)
    
    # Get config as dict
    if hasattr(config_module, 'get_config'):
        return config_module.get_config()
    else:
        # Build config from module attributes
        return {
            "data": config_module.data,
            "model": config_module.model,
            "training": config_module.training,
            "logging": config_module.logging,
            "seed": config_module.seed,
        }


def load_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and split ETTh1 data into train/val/test."""
    data_config = config["data"]
    data_path = data_config["data_path"]
    
    # Load data (from URL or local file)
    if data_path.startswith("http"):
        print(f"Downloading data from {data_path}...")
        df = pd.read_csv(data_path)
    else:
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
    
    # Split into train/val/test
    train_size = data_config["train_size"]
    val_size = data_config["val_size"]
    test_size = data_config.get("test_size", val_size)
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:train_size + val_size + test_size]
    
    print(f"Data loaded: Train={train_df.shape}, Val={val_df.shape}, Test={test_df.shape}")
    
    return train_df, val_df, test_df


def create_dataloaders(config: Dict[str, Any], 
                       train_df: pd.DataFrame,
                       val_df: pd.DataFrame,
                       test_df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders."""
    data_config = config["data"]
    model_config = config["model"]
    
    seq_len = model_config["seq_len"]
    label_len = model_config["label_len"]
    pred_len = model_config["pred_len"]
    features = data_config["features"]
    target = data_config["target"]
    batch_size = data_config["batch_size"]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(
        train_df, seq_len, label_len, pred_len, 
        features, target, 'train', scale=data_config.get("scale", True)
    )
    
    val_dataset = TimeSeriesDataset(
        val_df, seq_len, label_len, pred_len,
        features, target, 'val', scale=data_config.get("scale", True), 
        scaler=train_dataset.scaler
    )
    
    test_dataset = TimeSeriesDataset(
        test_df, seq_len, label_len, pred_len,
        features, target, 'test', scale=data_config.get("scale", True), 
        scaler=train_dataset.scaler
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_config.get("num_workers", 0)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_config.get("num_workers", 0)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_config.get("num_workers", 0)
    )
    
    return train_loader, val_loader, test_loader


def train_epoch(model, dataloader, criterion, optimizer, label_len, pred_len, device, grad_clip=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch_x, batch_y, batch_x_mark, batch_y_mark in dataloader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)
        
        # Prepare decoder input
        dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        
        # Extract predictions for the prediction horizon
        f_dim = -1
        outputs = outputs[:, -pred_len:, f_dim:]
        batch_y = batch_y[:, -pred_len:, f_dim:].to(device)
        
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # Gradient clipping
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, label_len, pred_len, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in dataloader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            # Prepare decoder input
            dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
            
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            # Extract predictions for the prediction horizon
            f_dim = -1
            outputs = outputs[:, -pred_len:, f_dim:]
            batch_y = batch_y[:, -pred_len:, f_dim:].to(device)
            
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def plot_learning_curve(history: Dict[str, Any], save_path: Path):
    """Plot and save learning curve."""
    epochs_data = history["epochs"]
    epochs = [e["epoch"] for e in epochs_data]
    train_losses = [e["train_mse"] for e in epochs_data]
    val_losses = [e["val_mse"] for e in epochs_data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train MSE", marker='o')
    plt.plot(epochs, val_losses, label="Val MSE", marker='s')
    plt.axvline(x=history["best_epoch"], color='r', linestyle='--', label=f'Best Epoch ({history["best_epoch"]})')
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train(config: Dict[str, Any], run_dir: Path):
    """
    Main training function.
    
    Args:
        config: Configuration dictionary
        run_dir: Directory to save outputs
    """
    # Set random seeds
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Setup device
    device_name = config["training"].get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    train_df, val_df, test_df = load_data(config)
    
    # Create model
    print("Building model...")
    model = build_model(config)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_params:,} trainable parameters")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config, train_df, val_df, test_df)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # Setup loss function
    criterion_class = config["training"].get("criterion_class", nn.MSELoss)
    criterion_params = config["training"].get("criterion_params", {})
    criterion = criterion_class(**criterion_params)
    
    # Setup optimizer
    optimizer_class = config["training"].get("optimizer_class", optim.Adam)
    optimizer_params = config["training"].get("optimizer_params", {"lr": 1e-4})
    optimizer = optimizer_class(model.parameters(), **optimizer_params)
    
    # Setup scheduler
    scheduler = None
    if config["training"].get("use_scheduler", False):
        scheduler_class = config["training"].get("scheduler_class")
        scheduler_params = config["training"].get("scheduler_params", {})
        
        if scheduler_class is not None:
            # Special handling for ReduceLROnPlateau (needs optimizer as first arg)
            if scheduler_class == optim.lr_scheduler.ReduceLROnPlateau:
                scheduler_params.setdefault("mode", "min")
                scheduler = scheduler_class(optimizer, **scheduler_params)
            else:
                scheduler = scheduler_class(optimizer, **scheduler_params)
    
    # Get model config for training
    label_len = config["model"]["label_len"]
    pred_len = config["model"]["pred_len"]
    
    # Get gradient clipping value
    grad_clip = config["training"].get("grad_clip", None)
    
    # Training loop
    max_epochs = config["training"]["max_epochs"]
    patience = config["training"]["patience"]
    
    best_val_mse = float('inf')
    best_epoch = 0
    best_model_state = None
    patience_counter = 0
    
    history_epochs = []
    start_time = time.time()
    
    print(f"\nStarting training for up to {max_epochs} epochs with patience={patience}...")
    print("-" * 80)
    
    for epoch in range(max_epochs):
        # Train
        train_mse = train_epoch(model, train_loader, criterion, optimizer, label_len, pred_len, device, grad_clip)
        
        # Evaluate
        val_mse = validate(model, val_loader, criterion, label_len, pred_len, device)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history_epochs.append({
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "lr": current_lr
        })
        
        # Print progress
        print(f"Epoch {epoch:3d}/{max_epochs} | Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f} | LR: {current_lr:.6f}")
        
        # Check for improvement
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  → New best validation MSE: {best_val_mse:.6f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch} (no improvement for {patience} epochs)")
            break
        
        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_mse)
            else:
                scheduler.step()
    
    train_time = time.time() - start_time
    print("-" * 80)
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Load best model and evaluate on test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    test_mse = validate(model, test_loader, criterion, label_len, pred_len, device)
    train_mse_at_best = history_epochs[best_epoch]["train_mse"]
    
    print(f"\nBest model (epoch {best_epoch}):")
    print(f"  Train MSE: {train_mse_at_best:.6f}")
    print(f"  Val MSE:   {best_val_mse:.6f}")
    print(f"  Test MSE:  {test_mse:.6f}")
    
    # Save metrics
    metrics = {
        "best_epoch": best_epoch,
        "val_mse": best_val_mse,
        "test_mse": test_mse,
        "train_mse": train_mse_at_best,
        "train_time_sec": train_time,
        "n_params": n_params
    }
    save_metrics(run_dir, metrics)
    print(f"\nMetrics saved to {run_dir / 'metrics.json'}")
    
    # Save history
    history = {
        "best_epoch": best_epoch,
        "epochs": history_epochs
    }
    save_history(run_dir, history)
    print(f"History saved to {run_dir / 'history.json'}")
    
    # Plot learning curve
    if config.get("logging", {}).get("plot_learning_curve", True):
        plot_path = run_dir / "learning_curve.png"
        plot_learning_curve(history, plot_path)
        print(f"Learning curve saved to {plot_path}")
    
    # Save best model checkpoint
    if config["training"].get("save_checkpoints", True):
        checkpoint_path = run_dir / "best_model.pth"
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'val_mse': best_val_mse,
            'test_mse': test_mse,
        }, checkpoint_path)
        print(f"Best model checkpoint saved to {checkpoint_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train time series forecasting model")
    parser.add_argument("--config", type=str, required=True, help="Path to config Python file")
    parser.add_argument("--tag", type=str, required=True, help="Run tag (e.g., run_0000)")
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = load_config(config_path)
    
    # Setup run directory
    run_dir = get_run_dir(args.tag)
    print(f"Run directory: {run_dir}")
    
    # Snapshot config
    snapshot_config(config_path, run_dir)
    config_snapshot_name = "config_used.py"
    print(f"Config snapshot saved to {run_dir / config_snapshot_name}")
    
    # Train
    train(config, run_dir)
    
    print(f"\n{'=' * 80}")
    print(f"Training complete! Results saved to: {run_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
