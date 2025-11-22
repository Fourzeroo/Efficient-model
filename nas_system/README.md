# NAS System - Training Infrastructure

Инфраструктура для обучения модели Informer на временных рядах с поддержкой Neural Architecture Search.

## Обзор

Базовая система обучения, которая предоставляет:
- Полный тренировочный пайплайн для модели Informer
- Управление экспериментами и состоянием
- Утилиты для логирования метрик и анализа
- API для внешних агентов (включая LLM)

Эта система может использоваться:
- **Автоматически** через `nas_agent_graph` (LLM-оркестрация)
- **Вручную** для запуска отдельных экспериментов

## Структура

```
nas_system/
├── train.py                 # Скрипт обучения (НЕ изменяется агентами)
├── model.py                 # Обертка для Informer (можно изменять)
├── config.py                # Конфигурация (изменяется агентами)
├── requirements.txt         # Зависимости
│
├── nas_agent/               # Библиотека утилит
│   ├── agent_state.py       # Управление глобальным состоянием NAS
│   ├── logging_utils.py     # Сохранение/загрузка метрик
│   ├── history_summary.py   # Анализ кривых обучения
│   └── run_manager.py       # Управление директориями
│
├── runs/                    # Результаты (создается автоматически)
│   ├── run_0000/
│   │   ├── metrics.json
│   │   ├── history.json
│   │   ├── learning_curve.png
│   │   ├── best_model.pth
│   │   └── config_used.py
│   └── run_0001/
│
└── agent_state.json         # Глобальное состояние NAS
```

## Установка

```bash
pip install -r requirements.txt
```

### Требуется Informer2020

Система использует реальную модель Informer. Склонируйте репозиторий:

```bash
cd ..  # Перейти в Efficient-model/
git clone https://github.com/zhouhaoyi/Informer2020.git
```

Структура должна быть:
```
Efficient-model/
├── Informer2020/           # Репозиторий с моделью
└── nas_system/             # Эта система
```

## Использование

### Запуск обучения

```bash
python train.py --config config.py --tag run_0000
```

Это создаст директорию `runs/run_0000/` с результатами:
- `metrics.json` - Финальные метрики (val_mse, test_mse, etc.)
- `history.json` - История по эпохам
- `learning_curve.png` - Визуализация кривых
- `best_model.pth` - Чекпоинт лучшей модели
- `config_used.py` - Снапшот использованной конфигурации

### Изменение конфигурации

Отредактируйте `config.py` (формат Python для максимальной гибкости):

```python
import torch.optim as optim
import torch.nn as nn

# Архитектура модели
model = {
    "d_model": 256,         # Размерность модели
    "n_heads": 4,           # Attention heads
    "e_layers": 3,          # Слои энкодера
    "d_layers": 1,          # Слои декодера
    "d_ff": 1024,           # Feedforward размерность
    "dropout": 0.1,         # Dropout
}

# Обучение
training = {
    "optimizer_class": optim.Adam,  # Или optim.AdamW, optim.SGD
    "optimizer_params": {
        "lr": 0.0001,
        "weight_decay": 0.0001,
    },
    "criterion_class": nn.MSELoss,  # Или nn.L1Loss, nn.SmoothL1Loss
    "scheduler_class": optim.lr_scheduler.CosineAnnealingLR,
    "scheduler_params": {"T_max": 10},
    "max_epochs": 20,
    "patience": 5,          # Early stopping
}
```

## Библиотека nas_agent

Утилиты для управления NAS процессом.

### Управление состоянием (`agent_state.py`)

```python
from nas_agent import (
    load_agent_state, save_agent_state, add_run,
    get_best_run, get_recent_runs, RunInfo
)

# Загрузить состояние
state = load_agent_state()

# Добавить результат запуска
run_info = RunInfo(
    run_id="run_0000",
    val_mse=0.185,
    test_mse=0.192,
    history_summary="...",
    accepted=True
)
state = add_run(state, run_info)
save_agent_state(state)

# Получить лучший запуск
best = get_best_run(state)
print(f"Best: {best.run_id}, MSE: {best.val_mse}")

# Последние запуски
recent = get_recent_runs(state, k=5)
```

### Логирование (`logging_utils.py`)

```python
from pathlib import Path
from nas_agent import save_metrics, load_metrics, save_history, load_history

# Сохранить метрики
metrics = {
    "val_mse": 0.185,
    "test_mse": 0.192,
    "best_epoch": 17,
}
save_metrics(Path("runs/run_0000"), metrics)

# Загрузить метрики
metrics = load_metrics(Path("runs/run_0000"))

# История обучения
history = {
    "best_epoch": 17,
    "epochs": [
        {"epoch": 0, "train_mse": 1.02, "val_mse": 1.05},
        {"epoch": 1, "train_mse": 0.85, "val_mse": 0.89},
        # ...
    ]
}
save_history(Path("runs/run_0000"), history)
```

### Анализ истории (`history_summary.py`)

```python
from nas_agent import load_history, build_history_summary

history = load_history(Path("runs/run_0000"))
summary = build_history_summary(history)
print(summary)
# "20 epochs, best_epoch=17. train_mse decreased from 1.02 to 0.24.
#  val_mse started at 1.05, reached minimum of 0.32 at epoch 17..."
```

### Управление директориями (`run_manager.py`)

```python
from nas_agent import get_run_dir, snapshot_config

# Создать/получить директорию запуска
run_dir = get_run_dir("run_0000")

# Сохранить снапшот конфига
snapshot_config(Path("config.py"), run_dir)
```

## Форматы файлов

### metrics.json

```json
{
  "best_epoch": 17,
  "val_mse": 0.1855,
  "test_mse": 0.1923,
  "train_mse": 0.1621,
  "train_time_sec": 285.4,
  "n_params": 4932103
}
```

### history.json

```json
{
  "best_epoch": 17,
  "epochs": [
    {"epoch": 0, "train_mse": 1.02, "val_mse": 1.05, "lr": 0.0001},
    {"epoch": 1, "train_mse": 0.85, "val_mse": 0.89, "lr": 0.0001}
  ]
}
```

### agent_state.json

```json
{
  "max_runs": 100,
  "best_run_id": "run_0003",
  "runs": [
    {
      "run_id": "run_0000",
      "val_mse": 0.1855,
      "test_mse": 0.1923,
      "history_summary": "20 epochs, best_epoch=17...",
      "accepted": true
    }
  ]
}
```

## Модель

### Informer (`model.py`)

Обертка для модели из Informer2020:

```python
from model import build_model

# Построить модель из конфига
model = build_model(config)

# Модель: Informer с ProbSparse Attention
# - Параметры: 4.9M (с текущим конфигом)
# - Используется для прогнозирования временных рядов
```

## Датасет

**ETTh1** (Electricity Transformer Temperature):
- Временной ряд с часовой гранулярностью
- 7 признаков: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
- Таргет: OT (Oil Temperature)
- Split: 8640 train / 2880 val / 2880 test часов

Автоматически загружается с GitHub при запуске обучения.

## Параметры конфигурации

### Модель

- `d_model` - размерность модели (256, 512, 768)
- `n_heads` - количество attention heads (4, 8, 16)
- `e_layers` - слои энкодера (2, 3, 4)
- `d_layers` - слои декодера (1, 2)
- `d_ff` - размерность feedforward (512, 1024, 2048)
- `dropout` - dropout rate (0.1, 0.2, 0.3)
- `factor` - ProbSparse attention factor (3, 4, 5)

### Обучение

- `optimizer_class` - класс оптимизатора (Adam, AdamW, SGD)
- `lr` - learning rate (1e-5 до 1e-3)
- `weight_decay` - L2 regularization (0.0 до 0.01)
- `batch_size` - размер батча (16, 32, 64)
- `max_epochs` - максимум эпох (20, 30, 50)
- `patience` - early stopping patience (5, 10)
- `criterion_class` - функция потерь (MSELoss, L1Loss, SmoothL1Loss)
- `scheduler_class` - scheduler (CosineAnnealingLR, StepLR, ReduceLROnPlateau)

## Базовая производительность

С лучшими параметрами из Optuna:
- **Val MSE**: 0.1855
- **Test MSE**: ~0.19
- **Параметры**: 4.9M
- **Время обучения**: ~5-10 минут (GPU)

## Использование с LLM-агентами

Система спроектирована для работы с внешними LLM-агентами:

1. **Агент читает** `agent_state.json` для контекста
2. **Агент изменяет** `config.py` на основе анализа
3. **Агент запускает** `train.py --config config.py --tag run_XXXX`
4. **Агент читает** результаты из `runs/run_XXXX/metrics.json`
5. **Агент обновляет** `agent_state.json` через библиотеку `nas_agent`

См. `nas_agent_graph/` для автоматической LLM-оркестрации.

## Устранение неполадок

### Ошибка импорта Informer

```
ImportError: No module named 'models.model'
```

Решение: Склонировать Informer2020 в родительскую директорию

### CUDA Out of Memory

Решение: Уменьшить `batch_size` в `config.py`:
```python
data = {
    "batch_size": 32,  # Было 64
}
```

### Нестабильные результаты

Нормально отклонение ±0.01-0.02 MSE.
Для воспроизводимости: установить `seed: 42` в config.

## Лицензия

Та же, что и у родительского проекта.

## Благодарности

- **Informer**: [Informer2020](https://github.com/zhouhaoyi/Informer2020)
- **ETT Dataset**: [ETDataset](https://github.com/zhouhaoyi/ETDataset)
