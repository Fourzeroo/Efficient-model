# Real Implementation Setup

This guide explains how to set up the real Informer-based NAS system that uses actual ETTh1 data and the Informer2020 model.

## Prerequisites

The system now uses the **real Informer model** from the original Informer2020 repository and trains on **real ETTh1 time series data**.

## Setup Instructions

### 1. Clone Informer2020 Repository

The model requires the Informer2020 repository to be cloned in the parent directory:

```bash
cd ..  # Go to parent directory (Efficient-model/)
git clone https://github.com/zhouhaoyi/Informer2020.git
cd nas_system
```

Your directory structure should look like:
```
Efficient-model/
├── Informer2020/          # Cloned repository
│   ├── models/
│   │   └── model.py      # Real Informer implementation
│   └── ...
└── nas_system/           # This NAS system
    ├── train.py
    ├── model.py
    ├── config.yaml
    └── ...
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- torch >= 2.0.0
- pyyaml >= 6.0
- matplotlib >= 3.5.0
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0

### 3. Configuration

The `config.py` is pre-configured with the **best hyperparameters** found via Optuna optimization (validation MSE: 0.1855):

```python
import torch.optim as optim
import torch.nn as nn

model = {
    "d_model": 512,
    "n_heads": 16,
    "e_layers": 3,
    "d_layers": 1,
    "d_ff": 1024,
    "dropout": 0.337,
    "factor": 4,
    "seq_len": 336,
    "label_len": 24,
    "pred_len": 96,
}

training = {
    "optimizer_class": optim.Adam,
    "optimizer_params": {"lr": 1.603e-05},
    "scheduler_class": optim.lr_scheduler.CosineAnnealingLR,
    "scheduler_params": {"T_max": 10},
    "criterion_class": nn.MSELoss,
    "max_epochs": 20,
    "patience": 5,
}
```

**Формат Python позволяет:**
- Использовать любые классы из torch.optim и torch.nn
- Не ограничиваться предопределенным списком
- Добавлять кастомные оптимизаторы/schedulers/losses

The system automatically downloads ETTh1 data from:
`https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv`

## Running Training

### Quick Test Run

```bash
python train.py --config config.py --tag test_run
```

This will:
1. Download ETTh1 dataset automatically
2. Build the real Informer model (4.9M parameters)
3. Train for up to 20 epochs with early stopping
4. Save results to `runs/test_run/`

### Expected Output

```
Run directory: D:\path\to\nas_system\runs\test_run
Config snapshot saved to runs\test_run\config_used.yaml
Using device: cuda
Downloading data from https://raw.githubusercontent.com/...
Data loaded: Train=(8640, 8), Val=(2880, 8), Test=(2880, 8)
Building model...
Model has 4,932,103 trainable parameters
Creating data loaders...
Train batches: 130, Val batches: 42, Test batches: 42

Starting training for up to 20 epochs with patience=5...
--------------------------------------------------------------------------------
Epoch   0/20 | Train MSE: 1.234567 | Val MSE: 1.345678 | LR: 0.000016
  → New best validation MSE: 1.345678
Epoch   1/20 | Train MSE: 0.987654 | Val MSE: 1.123456 | LR: 0.000015
  → New best validation MSE: 1.123456
...
```

### Output Files

After training, `runs/test_run/` will contain:

- `config_used.yaml` - Configuration snapshot
- `metrics.json` - Final metrics
  ```json
  {
    "best_epoch": 12,
    "val_mse": 0.185,
    "test_mse": 0.192,
    "train_mse": 0.156,
    "train_time_sec": 324.5,
    "n_params": 4932103
  }
  ```
- `history.json` - Per-epoch training history
- `learning_curve.png` - Visualization
- `best_model.pth` - Model checkpoint

## Using with NAS Agent

### 1. Read Current State

```python
from nas_agent import load_agent_state, load_metrics, load_history, build_history_summary

# Load global state
state = load_agent_state()

# Load results from a run
metrics = load_metrics("runs/test_run")
history = load_history("runs/test_run")
summary = build_history_summary(history)

print(f"Val MSE: {metrics['val_mse']}")
print(f"Summary: {summary}")
```

### 2. Modify Architecture

Edit `config.py` to try different architectures and optimizers:

```python
# config.py

import torch.optim as optim
import torch.nn as nn

# Try smaller model
model = {
    "d_model": 256,     # Reduced
    "n_heads": 8,       # Reduced
    "e_layers": 2,      # Reduced
    # ... other params
}

# Try different optimizer and loss
training = {
    "optimizer_class": optim.AdamW,  # Changed from Adam
    "optimizer_params": {
        "lr": 1e-4,
        "weight_decay": 0.01,  # Added regularization
    },
    "criterion_class": nn.SmoothL1Loss,  # Changed loss
    # ... other params
}
```

### 3. Run Training

```bash
python train.py --config config.py --tag run_0001
```

### 4. Update Global State

```python
from nas_agent import add_run, save_agent_state, RunInfo

run_info = RunInfo(
    run_id="run_0001",
    val_mse=metrics["val_mse"],
    test_mse=metrics["test_mse"],
    history_summary=summary,
    accepted=True
)

state = add_run(state, run_info)
save_agent_state(state)
```

## Key Differences from Placeholder

The real implementation:

1. **Real Model**: Uses actual Informer from Informer2020 with ProbSparse attention
2. **Real Data**: Downloads and processes ETTh1 dataset (electricity transformer temperature)
3. **Best Params**: Pre-configured with Optuna-optimized hyperparameters (val_mse=0.1855)
4. **Real Training**: Matches the notebook training loop exactly
5. **Proper Dataset**: Uses label_len for decoder input, StandardScaler for normalization

## Troubleshooting

### Import Error: models.model

```
ImportError: No module named 'models.model'
```

**Solution**: Clone Informer2020 in the parent directory:
```bash
cd ..
git clone https://github.com/zhouhaoyi/Informer2020.git
cd nas_system
```

### CUDA Out of Memory

**Solution**: Reduce batch size in `config.py`:
```python
data = {
    "batch_size": 32,  # Reduced from 64
    # ... other params
}
```

### Download Fails

**Solution**: Download ETTh1.csv manually:
```bash
curl -O https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv
```

Then update `config.py`:
```python
data = {
    "data_path": "./ETTh1.csv",
    # ... other params
}
```

## Baseline Performance

With the best Optuna parameters:
- **Val MSE**: 0.1855
- **Test MSE**: ~0.19
- **Parameters**: 4.9M
- **Training time**: ~5-10 minutes on GPU

You can try to beat this with different architectures!

## Next Steps

1. Run `example_agent.py` to see automated NAS in action
2. Try different model configurations
3. Integrate with external LLM agents (Claude, Cursor, Windsurf)
4. Experiment with ETTh2 or other datasets

## References

- [Informer Paper](https://arxiv.org/abs/2012.07436)
- [Informer2020 GitHub](https://github.com/zhouhaoyi/Informer2020)
- [ETT Dataset](https://github.com/zhouhaoyi/ETDataset)
