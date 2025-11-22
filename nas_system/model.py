"""
Model Definition - Real Informer model for time series forecasting.

This module provides the actual Informer model from Informer2020 repository,
which can be modified by LLM agents to explore different architectures during NAS.
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from typing import Dict, Any

# Add Informer2020 to path
informer_path = Path(__file__).parent.parent / "Informer2020"
if informer_path.exists():
    sys.path.insert(0, str(informer_path))
    from models.model import Informer
else:
    raise ImportError(
        f"Informer2020 not found at {informer_path}. "
        f"Please clone it: git clone https://github.com/zhouhaoyi/Informer2020.git"
    )


def build_model(config: Dict[str, Any]) -> nn.Module:
    """
    Build the real Informer model from configuration dictionary.
    
    This function is called by train.py to instantiate the model
    based on the current config.yaml. LLM agents can modify the
    config to explore different architectures.
    
    Args:
        config: Configuration dictionary containing model hyperparameters
        
    Returns:
        Instantiated Informer model from Informer2020
        
    Example config structure:
        {
            "model": {
                "enc_in": 7,
                "dec_in": 7,
                "c_out": 7,
                "seq_len": 336,
                "label_len": 24,
                "pred_len": 96,
                "d_model": 512,
                "n_heads": 16,
                "e_layers": 3,
                "d_layers": 1,
                "d_ff": 1024,
                "dropout": 0.33,
                "factor": 4
            }
        }
    """
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    
    # Extract model parameters
    enc_in = model_config.get("enc_in", 7)
    dec_in = model_config.get("dec_in", 7)
    c_out = model_config.get("c_out", 7)
    seq_len = model_config.get("seq_len", 336)
    label_len = model_config.get("label_len", 24)
    pred_len = model_config.get("pred_len", 96)
    d_model = model_config.get("d_model", 512)
    n_heads = model_config.get("n_heads", 16)
    e_layers = model_config.get("e_layers", 3)
    d_layers = model_config.get("d_layers", 1)
    d_ff = model_config.get("d_ff", 1024)
    dropout = model_config.get("dropout", 0.33)
    factor = model_config.get("factor", 4)
    
    # Get device
    device_name = config.get("training", {}).get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    
    # Create Informer model
    model = Informer(
        enc_in=enc_in,
        dec_in=dec_in,
        c_out=c_out,
        seq_len=seq_len,
        label_len=label_len,
        out_len=pred_len,
        factor=factor,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        d_layers=d_layers,
        d_ff=d_ff,
        dropout=dropout,
        attn='prob',
        embed='timeF',
        freq='h',
        activation='gelu',
        output_attention=False,
        distil=True,
        mix=True
    ).to(device)
    
    return model
