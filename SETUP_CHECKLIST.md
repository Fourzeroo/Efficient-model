# ✅ Setup and Testing Checklist

Complete this checklist to verify the LangGraph NAS system is ready to use.

## 📋 Pre-Installation

- [ ] Python 3.10+ installed
- [ ] Git installed
- [ ] CUDA/GPU available (optional, but recommended)
- [ ] OpenRouter account created at https://openrouter.ai/
- [ ] OpenRouter API key obtained

## 📦 Installation

### Step 1: Base System
```bash
cd nas_system
pip install -r requirements.txt
```
- [ ] All dependencies installed without errors
- [ ] No import errors when running `python -c "import torch; import pandas"`

### Step 2: Informer2020 (if not already present)
```bash
cd ..  # Go to Efficient-model/
git clone https://github.com/zhouhaoyi/Informer2020.git
```
- [ ] `Informer2020` directory exists in project root
- [ ] `Informer2020/models/model.py` exists

### Step 3: LangGraph System
```bash
cd nas_agent_graph
pip install -r requirements.txt
```
- [ ] langgraph installed
- [ ] langchain-openai installed
- [ ] pydantic installed
- [ ] python-dotenv installed

## 🔧 Configuration

### Create .env file
```bash
cd ..  # Go to Efficient-model/
cp .env.example .env
```

Edit `.env` and add:
```bash
OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here
```

- [ ] `.env` file created
- [ ] `OPENROUTER_API_KEY` set with real key
- [ ] No spaces around `=` sign
- [ ] Key starts with `sk-or-v1-`

## 🧪 Testing

### Test 1: Base Training System
```bash
cd nas_system
python train.py --config config.py --tag test_run_manual
```

Expected output:
```
Building model...
Model has X,XXX,XXX trainable parameters
Creating data loaders...
Starting training for up to 20 epochs...
...
Training complete! Results saved to: ...
```

- [ ] Training completes without errors
- [ ] `runs/test_run_manual/` directory created
- [ ] `metrics.json` exists in run directory
- [ ] `history.json` exists in run directory
- [ ] `best_model.pth` exists in run directory
- [ ] `learning_curve.png` exists in run directory

Check metrics:
```bash
cat runs/test_run_manual/metrics.json
```

- [ ] `val_mse` value present (should be ~0.18-0.20)
- [ ] `test_mse` value present
- [ ] `best_epoch` value present

### Test 2: Import LangGraph System
```bash
cd ..
python -c "from nas_agent_graph import run_nas_optimization; print('✓ Import successful')"
```

- [ ] No import errors
- [ ] "✓ Import successful" printed

### Test 3: Quick NAS Run (5 iterations)
```bash
python quickstart_nas_graph.py
```

Expected behavior:
1. Prints system information
2. Shows "PLANNER - Iteration 0/5"
3. Shows "EXECUTOR - Running iteration 0"
4. Shows "EVALUATOR - Evaluating run_0000"
5. Shows "UPDATE STATE NODE"
6. Repeats for iterations 1-4
7. Shows final summary with best run

- [ ] Script runs without crashes
- [ ] All 5 iterations complete
- [ ] Planner outputs visible
- [ ] Training runs for each iteration
- [ ] Evaluator decisions shown
- [ ] Final summary displayed

Check outputs:
```bash
ls nas_system/runs/
```

- [ ] `run_0000` through `run_0004` directories exist (or fewer if some were rejected)

```bash
cat nas_system/agent_state.json
```

- [ ] JSON is valid
- [ ] `runs` array contains entries
- [ ] `best_run_id` is set
- [ ] Each run has `val_mse`, `test_mse`, `accepted` fields

### Test 4: Single Iteration Test
```python
# test_single_iteration.py
from nas_agent_graph import run_nas_optimization

print("Testing single iteration...")
run_nas_optimization(max_iterations=1)
print("Test complete!")
```

- [ ] Single iteration completes
- [ ] Config potentially modified
- [ ] New run directory created
- [ ] Agent state updated

## 🔍 Verification Checklist

### File Structure
```bash
ls -la
```

Should see:
- [ ] `.env` (your secrets)
- [ ] `.env.example` (template)
- [ ] `COMPLETE_OVERVIEW.md`
- [ ] `LANGGRAPH_NAS_GUIDE.md`
- [ ] `IMPLEMENTATION_SUMMARY.md`
- [ ] `quickstart_nas_graph.py`
- [ ] `nas_system/` directory
- [ ] `nas_agent_graph/` directory
- [ ] `Informer2020/` directory

### nas_agent_graph structure
```bash
ls nas_agent_graph/
```

Should see:
- [ ] `__init__.py`
- [ ] `config.py`
- [ ] `evaluator.py`
- [ ] `executor.py`
- [ ] `graph.py`
- [ ] `main.py`
- [ ] `planner.py`
- [ ] `prompts.py`
- [ ] `README.md`
- [ ] `requirements.txt`

### API Connection Test
```python
# test_api.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
assert api_key is not None, "API key not found!"
assert api_key.startswith("sk-or-v1-"), "Invalid API key format!"

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    model="anthropic/claude-3.5-sonnet",
)

response = llm.invoke("Say 'API test successful' and nothing else.")
print(f"✓ API Response: {response.content}")
```

- [ ] API key loads correctly
- [ ] LLM responds successfully
- [ ] No authentication errors

## 🎯 Ready to Use

If all checkboxes above are checked ✅, your system is ready!

### Next Steps

1. **Run full optimization**:
```bash
python -m nas_agent_graph.main --max-iterations 20 --verbose
```

2. **Monitor progress**:
```bash
# In another terminal
watch -n 10 "tail -20 nas_system/agent_state.json"
```

3. **Analyze results**:
```python
from nas_system.nas_agent import load_agent_state, get_best_run
from pathlib import Path

state = load_agent_state(Path("nas_system/agent_state.json"))
best = get_best_run(state)

print(f"Best run: {best.run_id}")
print(f"Val MSE: {best.val_mse:.6f}")
print(f"Improvement: {(0.1855 - best.val_mse) / 0.1855 * 100:.1f}%")
```

## 🐛 Troubleshooting

### Common Issues During Testing

#### Issue: "No module named 'langgraph'"
```bash
pip install langgraph langchain-openai pydantic python-dotenv
```

#### Issue: "OPENROUTER_API_KEY not found"
- Check `.env` file exists
- Check key format: `OPENROUTER_API_KEY=sk-or-v1-...`
- No spaces around `=`
- File is in project root, not in subdirectory

#### Issue: "Import error: models.model"
- Check `Informer2020` directory exists
- Should be at same level as `nas_system`
- Run: `ls ../Informer2020/models/model.py` from nas_system

#### Issue: Training fails with CUDA error
- Check GPU: `nvidia-smi`
- Or use CPU: Set `training.device = "cpu"` in config.py

#### Issue: Config not being modified
- Check config.py syntax is standard Python dict
- Enable debug: `DEBUG=true` in .env
- Try manual modification first

#### Issue: LLM timeout
- Check internet connection
- Try different model: `PLANNER_MODEL=openai/gpt-3.5-turbo`
- Check OpenRouter status page

## 📊 Expected Performance

After completing tests, you should see:

### Training Performance
- Baseline: `val_mse ≈ 0.185`
- Training time: 5-10 minutes per run (GPU) or 20-30 minutes (CPU)
- Convergence: 15-20 epochs typically

### NAS Performance  
- First run: Establishes baseline
- Iterations 1-5: Initial exploration (~5-10% improvement)
- Iterations 6-15: Refinement (~10-15% improvement)
- Iterations 16-20: Fine-tuning (~15-20% improvement)

### Resource Usage
- Disk: ~50MB per run (model + logs)
- Memory: ~4GB (with GPU) or ~8GB (CPU only)
- API cost: ~$0.02-0.03 per iteration

## 🎓 You're Ready!

✅ **All tests passed**: System is fully operational
⚠️ **Some tests failed**: Check troubleshooting section
❌ **Many tests failed**: Review installation steps

---

**Happy optimizing! 🚀**
