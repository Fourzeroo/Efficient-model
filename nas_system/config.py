"""
Configuration for Time Series Forecasting with Informer

This is a Python config file that can be modified by LLM agents during NAS.
It allows specifying optimizer classes, scheduler classes, loss functions, etc.
directly as code for maximum flexibility.
"""

import torch
import torch.nn as nn
import torch.optim as optim


# Dataset configuration
data = {
    # Path to the dataset (ETTh1, ETTh2, ETTm1, etc.)
    "data_path": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    
    # Data preprocessing
    "target": "OT",  # Target column to forecast
    "features": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],  # Feature columns
    
    # Sequence lengths (from best Optuna params)
    "seq_len": 336,      # Input sequence length
    "label_len": 24,     # Start token length for decoder
    "pred_len": 96,      # Prediction length
    
    # Train/val/test split (hours)
    "train_size": 8640,  # 12 months * 30 days * 24 hours
    "val_size": 2880,    # 4 months * 30 days * 24 hours
    "test_size": 2880,   # 4 months * 30 days * 24 hours
    
    # Data loader settings (from best Optuna params)
    "batch_size": 64,
    "num_workers": 0,
    "scale": True,
}


# Model architecture (from best Optuna params: val_loss=0.1855)
model = {
    "enc_in": 7,         # Number of encoder input features
    "dec_in": 7,         # Number of decoder input features
    "c_out": 7,          # Number of output features
    "seq_len": 336,      # Input sequence length
    "label_len": 24,     # Start token length
    "pred_len": 96,      # Prediction length
    "d_model": 512,      # Model dimension
    "n_heads": 16,       # Number of attention heads
    "e_layers": 3,       # Number of encoder layers
    "d_layers": 1,       # Number of decoder layers
    "d_ff": 1024,        # Feedforward network dimension
    "dropout": 0.337,    # Dropout rate
    "factor": 4,         # ProbSparse attention factor
}


# Training configuration
training = {
    # Optimization (from best Optuna params)
    "optimizer_class": optim.Adam,  # Can be changed to optim.AdamW, optim.SGD, etc.
    "optimizer_params": {
        "lr": 1.603e-05,            # Learning rate
        "weight_decay": 0.0,
        # "betas": (0.9, 0.999),    # For Adam/AdamW
        # "momentum": 0.9,          # For SGD
    },
    
    # Learning rate scheduler
    "use_scheduler": True,
    "scheduler_class": optim.lr_scheduler.CosineAnnealingLR,  # Can be changed
    "scheduler_params": {
        "T_max": 10,
        # "step_size": 10,          # For StepLR
        # "gamma": 0.5,             # For StepLR
        # "patience": 3,            # For ReduceLROnPlateau
    },
    
    # Loss function
    "criterion_class": nn.MSELoss,  # Can be nn.L1Loss, nn.SmoothL1Loss, etc.
    "criterion_params": {},          # Parameters for loss function
    
    # Training parameters
    "max_epochs": 20,
    "patience": 5,        # Early stopping patience
    
    # Gradient clipping
    "grad_clip": 1.0,     # Set to None to disable
    
    # Device
    "device": "cuda",     # cuda or cpu (auto-detected if cuda not available)
    
    # Checkpointing
    "save_checkpoints": True,
    "checkpoint_metric": "val_mse",  # Metric to use for best checkpoint
}


# Logging and reproducibility
logging = {
    "print_every": 10,           # Print training stats every N batches
    "plot_learning_curve": True,
}


# Random seed for reproducibility
seed = 42


# Export config as dict for compatibility
def get_config():
    """Return config as a dictionary."""
    return {
        "data": data,
        "model": model,
        "training": training,
        "logging": logging,
        "seed": seed,
    }
