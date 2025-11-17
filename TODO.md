# MAGRPO Execution TODO
## All Code Implemented - Ready to Run!

✅ **Status**: All code modules are implemented and ready
**Your Task**: Execute the training and analyze results

**Budget**: $10-15 for Claude Haiku evaluations
**Time**: 1-2 hours setup + 6-8 hours training (overnight)
**Hardware**: Colab Pro with A100 GPU

---

## Quick Start (5 Minutes)

**If you just want to start training immediately:**

1. Open `notebooks/magrpo_training.ipynb` in Colab
2. Select Runtime → Change runtime type → A100 GPU
3. Run all cells
4. Enter your Anthropic API key when prompted
5. Let it train overnight (6-8 hours)

✅ Done! Results will be in your Google Drive.

---

## Detailed Execution Guide

### Phase 1: Pre-Setup (30 minutes)

#### 1.1 Get Anthropic API Key
- [ ] Go to https://console.anthropic.com/
- [ ] Create account or login
- [ ] Go to API Keys section
- [ ] Create new API key
- [ ] Copy the key (starts with `sk-ant-...`)
- [ ] **IMPORTANT**: Set budget alert to $20 in Anthropic console

#### 1.2 Verify Colab Pro
- [ ] Login to https://colab.google.com/
- [ ] Verify you see "Colab Pro" badge in top-right
- [ ] Test GPU access: Runtime → Change runtime type → Check A100 is available
- [ ] If no A100: Upgrade to Colab Pro ($10/month) or use V100 (will be slower)

#### 1.3 Prepare Repository
- [ ] Ensure latest code pushed to GitHub: `git push origin feat/full-learning`
- [ ] Note your GitHub URL: `https://github.com/YOUR_USERNAME/Orchestry.git`
- [ ] Update notebook cell 3 with your GitHub URL

#### 1.4 Mount Google Drive
- [ ] Open Colab, run: `from google.colab import drive; drive.mount('/content/drive')`
- [ ] Authorize Google Drive access
- [ ] Verify mount: `!ls /content/drive/MyDrive`

✅ **Checkpoint**: You should have API key, Colab Pro, and GitHub repo ready

---

### Phase 2: Training Execution (6-8 hours, mostly unattended)

#### 2.1 Open Notebook
- [ ] Go to GitHub: `https://github.com/YOUR_USERNAME/Orchestry`
- [ ] Navigate to `notebooks/magrpo_training.ipynb`
- [ ] Click "Open in Colab" button (or upload to Colab)

#### 2.2 Setup Environment (30 mins)
- [ ] **Cell 1**: Check GPU - verify you see A100 GPU
- [ ] **Cell 2**: Install dependencies (~10 mins)
- [ ] **Cell 3**: Clone repository - UPDATE WITH YOUR GITHUB URL FIRST!
- [ ] **Cell 4**: Install package
- [ ] **Cell 5**: Mount Google Drive
- [ ] **Cell 6**: Enter Anthropic API key (paste when prompted)
- [ ] **Cell 7**: Load configuration - verify settings look correct

#### 2.3 Load Models (20 mins)
- [ ] **Cell 8**: Setup logging
- [ ] **Cell 9**: Load models - **THIS TAKES ~15-20 MINS**
  - Watch for: "Model loaded successfully"
  - Check model size: should be ~3GB per agent
  - Check trainable parameters: should be ~50M

#### 2.4 Initialize Task & Trainer (5 mins)
- [ ] **Cell 10**: Create task - verify "15 problems" loaded
- [ ] **Cell 11**: Create trainer

#### 2.5 Smoke Test (RECOMMENDED, 10 mins)
- [ ] **Cell 12**: Run 5-episode smoke test
- [ ] Verify no errors
- [ ] Check that cooperation rate is being tracked
- [ ] Check that rewards are non-zero
- [ ] **If any errors, STOP and debug before full training**

#### 2.6 Start Full Training (6-8 hours)
- [ ] **Cell 13**: Start full training
- [ ] Monitor first 10 episodes (~30 mins) to ensure:
  - No errors appearing
  - GPU utilization >80% (check `!nvidia-smi` in new cell)
  - Rewards being computed and logged
  - Budget tracker showing reasonable costs (~$0.02 per episode)
  - Checkpoints saving to Google Drive

✅ **Critical**: After first 10 episodes look good, you can leave it running overnight

#### 2.7 Monitor Progress (Optional)
While training runs, you can periodically check:
- [ ] Cooperation rate increasing?
- [ ] Mean reward increasing?
- [ ] Budget spent (should be <$15 total)
- [ ] Latest checkpoint exists in Google Drive
- [ ] No error messages in logs

**Expected Progress**:
- Episode 100: Cooperation ~30-40%
- Episode 250: Cooperation ~50-60%
- Episode 500: Cooperation ~70-80%

#### 2.8 Training Complete
- [ ] **Cell 14**: Save final models to Google Drive
- [ ] **Cell 15**: Run evaluation on test set
- [ ] **Cell 16**: Generate quick visualization
- [ ] **Cell 17**: Print budget summary - verify <$15 spent

✅ **Checkpoint**: Training complete, models saved, budget within limit

---

### Phase 3: Download & Analyze Results (1-2 hours)

#### 3.1 Download from Google Drive
From your Google Drive, download:
- [ ] `/MyDrive/magrpo_experiments/checkpoints/` - All checkpoints
- [ ] `/MyDrive/magrpo_experiments/final_models/` - Final trained models
- [ ] `/MyDrive/magrpo_experiments/training_curves.png` - Quick plot

Save to your local machine: `~/Downloads/magrpo_results/`

#### 3.2 Generate Detailed Analysis
On your local machine:

```bash
cd ~/gitrepos/Orchestry

# Generate learning curves
python analysis/plot_results.py ~/Downloads/magrpo_results/final_models/training_metrics.json

# Analyze cooperation schemes
python analysis/detect_schemes.py ~/Downloads/magrpo_results/checkpoints/episode_450/
```

- [ ] Run plot_results.py - generates `results/learning_curves.png`
- [ ] Run detect_schemes.py - prints cooperation patterns
- [ ] Review generated plots in `results/` directory

#### 3.3 Run Baseline Comparisons (Optional, 2-3 hours)
To compare against baselines:

```bash
# In Colab or locally with GPU
python -c "
from orchestry.baselines import FixedModelBaseline, NaiveConcatenationBaseline, SequentialPipelineBaseline
from orchestry.tasks.code_collaboration import CodeCollaborationTask
import json

task = CodeCollaborationTask()
_, test_problems = task.get_train_test_split()

# Run each baseline
fixed = FixedModelBaseline()
results_fixed = fixed.evaluate(test_problems)

naive = NaiveConcatenationBaseline()
results_naive = naive.evaluate(test_problems)

sequential = SequentialPipelineBaseline()
results_sequential = sequential.evaluate(test_problems)

# Save results
with open('baseline_results.json', 'w') as f:
    json.dump({
        'fixed': results_fixed,
        'naive': results_naive,
        'sequential': results_sequential,
    }, f)

print('Baselines complete!')
"
```

- [ ] Run baseline comparisons
- [ ] Compare with your MAGRPO results
- [ ] Generate comparison plot

---

### Phase 4: Document Findings (1 hour)

#### 4.1 Create Results Summary
- [ ] Open `results/summary.txt` (auto-generated by plot_results.py)
- [ ] Review key metrics:
  - Initial vs Final cooperation rate
  - Initial vs Final reward
  - Test pass rate
  - Budget spent
- [ ] Take screenshots of best learning curves

#### 4.2 Identify Cooperation Schemes
- [ ] Run scheme detection on final checkpoint
- [ ] Document which schemes emerged:
  - [ ] Fallback pattern?
  - [ ] Decorator pattern?
  - [ ] Coordinator pattern?
  - [ ] Strategy filter pattern?
- [ ] Copy 2-3 example trajectories showing good cooperation

#### 4.3 Create RESULTS.md
- [ ] Copy template below and fill in your results
- [ ] Save as `RESULTS.md` in repo root
- [ ] Commit and push to GitHub

**RESULTS.md Template:**
```markdown
# MAGRPO Experiment Results

## Summary
- **Date**: [DATE]
- **Total Episodes**: 500
- **Training Time**: [X] hours
- **Budget Spent**: $[X.XX]
- **Hardware**: Colab Pro A100 GPU

## Key Findings
- Cooperation rate improved from [X%] → [Y%]
- Test pass rate: [Z%]
- [4/4] cooperation schemes emerged

## Learning Curves
![Learning Curves](results/learning_curves.png)

## Cooperation Schemes Observed
1. **Fallback** - [X] instances
2. **Decorator** - [Y] instances
3. **Coordinator** - [Z] instances
4. **Strategy Filter** - [W] instances

## Example Trajectories
[Paste 2-3 examples showing evolution]

## Baseline Comparison
| Method | Mean Reward | Cooperation Rate |
|--------|-------------|------------------|
| Fixed Model | X.XX | X% |
| Naive Concat | X.XX | X% |
| Sequential | X.XX | X% |
| **MAGRPO (Ours)** | **X.XX** | **X%** |

## Lessons Learned
- What worked well?
- What could be improved?
- Novel findings vs paper?
```

---

## Success Criteria Checklist

### Minimum Success (Must Achieve)
- [ ] Training completed all 500 episodes without crashing
- [ ] Cooperation rate increased (any amount counts)
- [ ] At least 1 cooperation scheme emerged clearly
- [ ] Budget within $15 limit
- [ ] Final models saved successfully

### Target Success (Should Achieve)
- [ ] Cooperation rate: 20% → 70%+ improvement
- [ ] Test pass rate: 30% → 50%+ improvement
- [ ] All 4 cooperation schemes observed
- [ ] Beats all baseline methods
- [ ] Clear learning curves showing improvement

### Stretch Success (Publication Quality)
- [ ] Novel cooperation schemes not in paper
- [ ] Cooperation rate >80%
- [ ] Pass@1 rate >55%
- [ ] Detailed analysis of why cooperation works
- [ ] Ready to writeup findings

---

## Troubleshooting

### Problem: GPU Out of Memory
**Solution**:
- Reduce group_size from 4 to 2 in config
- Use 8-bit quantization instead of 4-bit
- Reduce max_new_tokens from 512 to 256

### Problem: API Rate Limits
**Solution**:
- Add sleep between episodes: `time.sleep(1)`
- Reduce Claude evaluation frequency
- Use heuristic rewards only (set `use_claude=False`)

### Problem: Budget Exceeding
**Solution**:
- STOP training immediately
- Review budget_tracker logs
- Reduce episodes or disable Claude eval
- Resume from last checkpoint

### Problem: Cooperation Not Improving
**Solution**:
- Check reward signal is correct (inspect first 10 episodes manually)
- Verify helper actually being called
- Increase cooperation reward weight
- Try different problems that require cooperation
- Increase training episodes to 1000

### Problem: Colab Disconnect
**Solution**:
- Checkpoints are saved every 50 episodes to Google Drive
- Re-run notebook and use `trainer.load_checkpoint(path)` to resume
- Consider Colab Pro+ for longer sessions

---

## Quick Commands Reference

### Check GPU
```bash
!nvidia-smi
```

### Monitor Training (in separate cell)
```python
import json
with open('/content/drive/MyDrive/magrpo_experiments/checkpoints/episode_450/metrics.json') as f:
    metrics = json.load(f)
print(f"Latest episode: {metrics[-1]['episode']}")
print(f"Cooperation: {metrics[-1]['cooperation_rate']:.2%}")
print(f"Reward: {metrics[-1]['mean_reward']:.3f}")
```

### Resume from Checkpoint
```python
trainer.load_checkpoint('/content/drive/MyDrive/magrpo_experiments/checkpoints/episode_450')
# Continue training
trainer.config.episodes = 1000  # Train to 1000
trainer.train()
```

### Quick Evaluation
```python
eval_results = trainer.evaluate(num_samples=10)
print(eval_results)
```

---

## What's Next?

After completing this TODO:

### If Results Are Good
1. Write up findings in RESULTS.md
2. Compare against paper's results
3. Identify your contributions (what's new/different?)
4. Consider submitting to workshop/conference
5. Implement research extensions (heterogeneous agents, curriculum learning, etc.)

### If Results Are Weak
1. Analyze failure modes (inspect generated code)
2. Debug reward signal
3. Try different hyperparameters
4. Increase training time (1000 episodes)
5. Simplify task (easier problems first)

### Research Extensions Ideas
- **Heterogeneous agents**: Use different models for helper vs main
- **Adaptive group size**: Vary k based on task difficulty
- **Curriculum learning**: Start easy, increase difficulty
- **3+ agents**: Add reviewer/refactor agents
- **Alternative rewards**: Learn critic instead of using Claude

---

## Files Generated

After completion, you'll have:

**Code (already exists)**:
- `orchestry/marl/algorithms/magrpo.py` - MAGRPO algorithm
- `orchestry/marl/local_inference.py` - Local LLM wrapper
- `orchestry/marl/rewards/code_reward.py` - Reward model
- `orchestry/tasks/code_collaboration.py` - Task environment
- `orchestry/marl/training/magrpo_trainer.py` - Training loop
- `orchestry/baselines/*.py` - 4 baseline methods
- `analysis/*.py` - Analysis tools
- `configs/magrpo_local.yaml` - Configuration
- `notebooks/magrpo_training.ipynb` - Training notebook

**Results (you'll generate)**:
- `checkpoints/episode_*/` - Training checkpoints
- `final_models/` - Final trained LoRA weights
- `results/learning_curves.png` - Training visualization
- `results/baseline_comparison.png` - Baseline plots
- `results/summary.txt` - Text summary
- `RESULTS.md` - Your findings document

---

## Time Estimates

| Phase | Task | Time | Can Skip? |
|-------|------|------|-----------|
| 1 | Pre-setup | 30 mins | No |
| 2 | Training execution | 6-8 hours | No (overnight) |
| 3 | Download results | 10 mins | No |
| 3 | Analysis plots | 30 mins | No |
| 3 | Baseline comparison | 2-3 hours | Yes |
| 4 | Document findings | 1 hour | Recommended |
| **Total** | **Your active time** | **~3 hours** | - |
| **Total** | **With training** | **~10 hours** | - |

---

## Support

**Stuck?** Check:
1. Error messages in Colab output
2. Budget tracker logs
3. Latest checkpoint metrics
4. This TODO troubleshooting section

**Still stuck?** Open an issue on GitHub with:
- Error message
- Last successful checkpoint
- Budget spent so far
- What you were trying to do

---

**YOU ARE HERE** → Phase 1: Pre-Setup, Step 1.1 (Get API key)

✅ All code is implemented and ready to run!
✅ Just follow this TODO step-by-step
✅ Expected total cost: $10-15
✅ Expected time: 3 hours active + 8 hours training

**Good luck! 🚀**
