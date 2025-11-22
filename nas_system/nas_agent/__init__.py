"""
NAS Agent - A Python library for LLM-driven Neural Architecture Search.

This package provides utilities for managing training runs, logging metrics,
and maintaining global NAS state for time series forecasting models.

Modules:
    run_manager: Run directory management and config snapshotting
    logging_utils: Metrics and history logging utilities
    history_summary: Learning curve summarization for LLM consumption
    agent_state: Global NAS state management (agent_state.json)
"""

from .run_manager import get_run_dir, snapshot_config, ensure_runs_root
from .logging_utils import save_metrics, save_history, load_metrics, load_history
from .history_summary import build_history_summary
from .agent_state import (
    RunInfo,
    AgentState,
    load_agent_state,
    save_agent_state,
    add_run,
    get_best_run,
    get_recent_runs,
)

__all__ = [
    "get_run_dir",
    "snapshot_config",
    "ensure_runs_root",
    "save_metrics",
    "save_history",
    "load_metrics",
    "load_history",
    "build_history_summary",
    "RunInfo",
    "AgentState",
    "load_agent_state",
    "save_agent_state",
    "add_run",
    "get_best_run",
    "get_recent_runs",
]
