# LLM-Orchestrated Neural Architecture Search

Автоматическая система улучшения архитектуры нейронных сетей для прогнозирования временных рядов на основе LLM-агентов.

## Описание

Система использует LLM (Claude 3.5 Sonnet) для автоматической оптимизации модели Informer на датасете временных рядов ETTh1. Система анализирует результаты обучения, принимает решения об изменении гиперпараметров и архитектуры, и итеративно улучшает метрики модели.

### Ключевые возможности

- **LLM-планирование**: Интеллектуальное принятие решений о следующих шагах оптимизации
- **Автоматическое выполнение**: Применение изменений и запуск обучения
- **Гибридная оценка**: Детерминированные правила + LLM для граничных случаев
- **Полная наблюдаемость**: Отслеживание всех экспериментов и метрик
- **Возобновление работы**: Автоматическое продолжение с последней итерации

## Архитектура системы

```
┌─────────────────────────────────────────────────────────────┐
│                  LangGraph NAS Workflow                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. PLANNER (LLM) → 2. EXECUTOR (Python) → 3. EVALUATOR    │
│         ↑                                            ↓       │
│         └──────────────────── Loop ─────────────────┘       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

Система состоит из двух основных частей:

### `nas_agent_graph/` - Оркестрация LLM-агентов

LangGraph-воркфлоу, который автоматизирует процесс NAS:
- **Planner**: LLM-агент, анализирующий историю и решающий, что изменить
- **Executor**: Применяет изменения в конфигурацию и запускает обучение
- **Evaluator**: Оценивает результаты (правила + LLM для сложных случаев)

### `nas_system/` - Инфраструктура обучения

Базовая система для обучения моделей:
- **train.py**: Полный пайплайн обучения модели Informer
- **model.py**: Обертка для модели из Informer2020
- **config.py**: Конфигурация (изменяется агентами)
- **nas_agent/**: Утилиты для управления состоянием и логированием

## Быстрый старт

### 1. Установка зависимостей

```bash
# Установить зависимости для обучения
pip install -r nas_system/requirements.txt

# Установить зависимости для LLM-агентов
pip install -r nas_agent_graph/requirements.txt
```

### 2. Настройка API ключа

Создайте файл `.env` в корне проекта:

```bash
OPENROUTER_API_KEY=your_openrouter_api_key
```

Получить ключ можно на https://openrouter.ai/

### 3. Запуск оптимизации

```bash
# Быстрый тест (5 итераций)
python quickstart_nas_graph.py

# Полная оптимизация (20 итераций)
python -m nas_agent_graph.main --max-iterations 20
```

## Использование

### Python API

```python
from nas_agent_graph import run_nas_optimization

# Запустить 20 итераций NAS
run_nas_optimization(max_iterations=20)
```

### Командная строка

```bash
# Стандартный запуск
python -m nas_agent_graph.main --max-iterations 20

# С подробным выводом
python -m nas_agent_graph.main --max-iterations 20 --verbose
```

### Анализ результатов

```python
from pathlib import Path
from nas_system.nas_agent import load_agent_state, get_best_run

# Загрузить состояние
state = load_agent_state(Path("nas_system/agent_state.json"))

# Получить лучший запуск
best = get_best_run(state)
print(f"Лучший запуск: {best.run_id}")
print(f"Val MSE: {best.val_mse:.6f}")
print(f"Test MSE: {best.test_mse:.6f}")
```

## Структура проекта

```
.
├── README.md                        # Этот файл
├── .env                             # API ключи (создать вручную)
├── quickstart_nas_graph.py          # Скрипт быстрого теста
│
├── nas_agent_graph/                 # LLM-оркестрация
│   ├── __init__.py                  # Экспорты пакета
│   ├── main.py                      # Точка входа
│   ├── config.py                    # Конфигурация системы
│   ├── planner.py                   # LLM-агент планирования
│   ├── executor.py                  # Применение изменений и обучение
│   ├── evaluator.py                 # Оценка результатов
│   ├── graph.py                     # LangGraph воркфлоу
│   ├── prompts.py                   # Шаблоны промптов
│   ├── phoenix_tracing.py           # Трейсинг для отладки
│   ├── README.md                    # Документация агентной системы
│   └── requirements.txt             # Зависимости
│
├── nas_system/                      # Инфраструктура обучения
│   ├── train.py                     # Скрипт обучения
│   ├── model.py                     # Определение модели
│   ├── config.py                    # Конфигурация (изменяется LLM)
│   ├── README.md                    # Документация системы обучения
│   ├── requirements.txt             # Зависимости
│   │
│   ├── nas_agent/                   # Библиотека утилит
│   │   ├── agent_state.py           # Управление глобальным состоянием
│   │   ├── logging_utils.py         # Сохранение/загрузка метрик
│   │   ├── history_summary.py       # Анализ кривых обучения
│   │   └── run_manager.py           # Управление директориями запусков
│   │
│   ├── runs/                        # Результаты запусков (создается автоматически)
│   │   ├── run_0000/
│   │   │   ├── metrics.json
│   │   │   ├── history.json
│   │   │   ├── learning_curve.png
│   │   │   ├── best_model.pth
│   │   │   └── config_used.py
│   │   └── run_0001/
│   │       └── ...
│   │
│   └── agent_state.json             # Глобальное состояние NAS
│
└── Informer2020/                    # Модель Informer (внешний репозиторий)
    └── models/
        └── model.py
```

## Возможности изменения конфигурации

LLM-планировщик может предлагать изменения в любые параметры:

### Гиперпараметры
```python
"training.optimizer_params.lr": 0.0001          # Learning rate
"training.optimizer_params.weight_decay": 0.01  # Регуляризация L2
"model.dropout": 0.3                            # Dropout
"data.batch_size": 32                           # Размер батча
```

### Архитектура модели
```python
"model.d_model": 256        # Размерность модели
"model.n_heads": 8          # Количество attention heads
"model.e_layers": 3         # Слои энкодера
"model.d_ff": 1024          # Размерность feedforward сети
```

### Оптимизатор и функция потерь
```python
"training.optimizer_class": "AdamW"           # Adam, AdamW, SGD
"training.scheduler_class": "StepLR"          # Scheduler
"training.criterion_class": "SmoothL1Loss"    # Функция потерь
```

## Процесс работы

### Итерация NAS

1. **Planner анализирует контекст**:
   - Текущая итерация / максимум итераций
   - Лучший запуск (val_mse, run_id)
   - Последние 5 запусков
   - Текущая конфигурация

2. **Planner принимает решение**:
   - Тип шага: small/medium/radical/stop
   - План: что попробовать
   - Причина: почему это имеет смысл
   - Конкретные изменения конфигурации

3. **Executor применяет изменения**:
   - Модифицирует `nas_system/config.py`
   - Запускает обучение
   - Загружает результаты

4. **Evaluator оценивает результаты**:
   - Детерминированные правила (быстро)
   - LLM для граничных случаев
   - Решение: принять или отклонить

5. **Обновление состояния**:
   - Добавить RunInfo в agent_state.json
   - Обновить best_run_id при улучшении
   - Инкрементировать счетчик итераций

6. **Цикл или остановка**:
   - Продолжить, если iteration < max_iterations и step_type != "stop"
   - Иначе завершить и показать результаты

## Ожидаемые результаты

- **Базовая производительность**: val_mse ≈ 0.1855 (из Optuna оптимизации)
- **После 10 итераций**: val_mse ≈ 0.17-0.18 (улучшение 5-10%)
- **После 20 итераций**: val_mse ≈ 0.16-0.17 (улучшение 10-15%)
- **Время на итерацию**: 5-10 минут (с GPU)
- **Стоимость**: ~$0.30-0.60 за 20 итераций (Claude 3.5 Sonnet)

## Документация

- **README.md** (этот файл) - Общий обзор системы
- **nas_agent_graph/README.md** - Подробная документация LLM-агентов
- **nas_system/README.md** - Документация системы обучения

## Требования

- Python 3.10+
- PyTorch с поддержкой CUDA (опционально, но рекомендуется)
- OpenRouter API ключ
- ~4GB RAM (CPU) или ~2GB VRAM (GPU)

## Устранение неполадок

### Ошибка импорта

```bash
pip install -r nas_agent_graph/requirements.txt
pip install -r nas_system/requirements.txt
```

### API ключ не найден

Создайте файл `.env` в корне проекта:
```bash
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### Обучение падает

Проверьте работу системы обучения отдельно:
```bash
cd nas_system
python train.py --config config.py --tag test_run
```
