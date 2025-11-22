"""
Quick Start Script for LangGraph NAS System

This script provides a simple way to test the LangGraph NAS system
with a small number of iterations.
"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from nas_agent_graph import run_nas_optimization


def main():
    """Run a quick test of the NAS system."""
    print("=" * 80)
    print("LANGGRAPH NAS SYSTEM - QUICK START")
    print("=" * 80)
    print()
    print("This will run 5 iterations of the NAS system as a test.")
    print("Press Ctrl+C at any time to stop.")
    print()
    
    try:
        run_nas_optimization(max_iterations=5)
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        raise


if __name__ == "__main__":
    main()
