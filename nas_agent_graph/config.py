"""
Configuration for LangGraph NAS System

This module defines settings for the LLM orchestration system,
including API keys, model selection, and graph parameters.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# OpenRouter API Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# LLM Model Configuration
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "anthropic/claude-3.5-sonnet")
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "anthropic/claude-3.5-sonnet")

# Temperature settings
PLANNER_TEMPERATURE = 0.7  # Higher for creative exploration
EVALUATOR_TEMPERATURE = 0.3  # Lower for consistent evaluation

# NAS System Paths
NAS_SYSTEM_ROOT = Path(__file__).parent.parent / "nas_system"
CONFIG_PATH = NAS_SYSTEM_ROOT / "config.py"
RUNS_ROOT = NAS_SYSTEM_ROOT / "runs"
AGENT_STATE_PATH = NAS_SYSTEM_ROOT / "agent_state.json"

# NAS Parameters
DEFAULT_MAX_ITERATIONS = 20
RECENT_RUNS_WINDOW = 5  # Number of recent runs to show in context

# Validation
if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY not found in environment. "
        "Please set it in a .env file or as an environment variable."
    )

# Debug settings
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"
