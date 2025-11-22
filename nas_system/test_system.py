"""
Quick test script to verify the NAS system is working correctly.

Run this after installation to ensure all modules can be imported
and basic functionality works.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from nas_agent import (
            get_run_dir,
            snapshot_config,
            ensure_runs_root,
            save_metrics,
            save_history,
            load_metrics,
            load_history,
            build_history_summary,
            RunInfo,
            AgentState,
            load_agent_state,
            save_agent_state,
            add_run,
            get_best_run,
            get_recent_runs,
        )
        print("  ✓ nas_agent modules imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_run_manager():
    """Test run directory management."""
    print("\nTesting run manager...")
    try:
        from nas_agent import get_run_dir, ensure_runs_root
        
        # Test creating run directory
        ensure_runs_root()
        run_dir = get_run_dir("test_run")
        
        assert run_dir.exists(), "Run directory not created"
        print(f"  ✓ Run directory created: {run_dir}")
        
        # Cleanup
        import shutil
        shutil.rmtree("runs", ignore_errors=True)
        
        return True
    except Exception as e:
        print(f"  ✗ Run manager test failed: {e}")
        return False


def test_logging():
    """Test metrics and history logging."""
    print("\nTesting logging utilities...")
    try:
        from nas_agent import save_metrics, save_history, load_metrics, load_history, get_run_dir
        
        # Create test run directory
        run_dir = get_run_dir("test_run")
        
        # Test saving metrics
        metrics = {"val_mse": 0.123, "test_mse": 0.456}
        save_metrics(run_dir, metrics)
        loaded_metrics = load_metrics(run_dir)
        assert loaded_metrics["val_mse"] == 0.123
        print("  ✓ Metrics save/load works")
        
        # Test saving history
        history = {
            "best_epoch": 10,
            "epochs": [
                {"epoch": 0, "train_mse": 1.0, "val_mse": 1.1},
                {"epoch": 1, "train_mse": 0.9, "val_mse": 1.0},
            ]
        }
        save_history(run_dir, history)
        loaded_history = load_history(run_dir)
        assert loaded_history["best_epoch"] == 10
        print("  ✓ History save/load works")
        
        # Cleanup
        import shutil
        shutil.rmtree("runs", ignore_errors=True)
        
        return True
    except Exception as e:
        print(f"  ✗ Logging test failed: {e}")
        return False


def test_history_summary():
    """Test history summarization."""
    print("\nTesting history summarization...")
    try:
        from nas_agent import build_history_summary
        
        history = {
            "best_epoch": 10,
            "epochs": [
                {"epoch": i, "train_mse": 1.0 - i*0.05, "val_mse": 1.1 - i*0.04}
                for i in range(20)
            ]
        }
        
        summary = build_history_summary(history)
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "20 epochs" in summary
        print(f"  ✓ Generated summary: {summary[:100]}...")
        
        return True
    except Exception as e:
        print(f"  ✗ History summary test failed: {e}")
        return False


def test_agent_state():
    """Test agent state management."""
    print("\nTesting agent state management...")
    try:
        from nas_agent import (
            load_agent_state, save_agent_state, add_run,
            get_best_run, get_recent_runs, RunInfo
        )
        
        # Load/create state
        state = load_agent_state(Path("test_agent_state.json"))
        assert state.max_runs == 100
        print("  ✓ Agent state loaded")
        
        # Add a run
        run1 = RunInfo(
            run_id="run_0000",
            val_mse=0.5,
            test_mse=0.6,
            history_summary="Test run 1",
            accepted=True
        )
        state = add_run(state, run1)
        
        run2 = RunInfo(
            run_id="run_0001",
            val_mse=0.3,
            test_mse=0.4,
            history_summary="Test run 2",
            accepted=True
        )
        state = add_run(state, run2)
        
        # Test best run
        best = get_best_run(state)
        assert best.run_id == "run_0001"
        assert best.val_mse == 0.3
        print("  ✓ Best run detection works")
        
        # Test recent runs
        recent = get_recent_runs(state, k=2)
        assert len(recent) == 2
        print("  ✓ Recent runs retrieval works")
        
        # Save state
        save_agent_state(state, Path("test_agent_state.json"))
        print("  ✓ Agent state saved")
        
        # Cleanup
        Path("test_agent_state.json").unlink(missing_ok=True)
        
        return True
    except Exception as e:
        print(f"  ✗ Agent state test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_import():
    """Test model building."""
    print("\nTesting model import...")
    try:
        from model import build_model
        
        config = {
            "model": {
                "input_dim": 7,
                "d_model": 64,
                "n_heads": 4,
                "n_encoder_layers": 1,
                "n_decoder_layers": 1,
                "d_ff": 256,
                "dropout": 0.1,
                "seq_len": 96,
                "pred_len": 24,
                "output_dim": 1
            }
        }
        
        model = build_model(config)
        assert model is not None
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Model built successfully ({n_params:,} parameters)")
        
        return True
    except Exception as e:
        print(f"  ✗ Model import test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("NAS System Quick Test")
    print("="*60)
    
    tests = [
        test_imports,
        test_run_manager,
        test_logging,
        test_history_summary,
        test_agent_state,
        test_model_import,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Run a test training: python train.py --config config.yaml --tag test")
        print("  2. Try the example agent: python example_agent.py")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
