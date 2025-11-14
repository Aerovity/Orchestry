# Quick Start Guide

Get Orchestry running in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Or use the setup script:

```bash
python setup.py
```

## Step 2: Configure API Key

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=sk-ant-...your-key-here
   ```

Get your API key from: https://console.anthropic.com/

## Step 3: Run a Test

Run a quick 3-episode test to verify everything works:

```bash
python main.py --test --verbose
```

You should see:
- 3 agents collaborating on stories
- Real-time conversation output
- Reward calculations
- Results saved to `runs/` directory

## Step 4: Run Full Training

Start a full 20-episode training run:

```bash
python main.py --episodes 20 --verbose
```

Or use the default from config:

```bash
python main.py
```

## Step 5: View Results

After training completes, check the `runs/` directory:

```
runs/2025-11-14_10-30/
├── episodes.json          # All conversations
├── rewards.csv           # Reward data
├── metrics.json          # Summary stats
└── plots/
    ├── training_curves.png
    └── collaboration_analysis.png
```

Open the plot images to see learning curves!

## Common Commands

```bash
# Quick test (3 episodes)
python main.py --test --verbose

# Training with custom episode count
python main.py --episodes 10

# Show best episode at the end
python main.py --episodes 20 --show-best

# Use custom config
python main.py --config my_config.yaml

# Just run without verbose output
python main.py --episodes 5
```

## Customization

Edit `config.yaml` to:
- Change number of episodes
- Adjust reward weights
- Add new story themes
- Modify agent roles
- Tune learning parameters

## Troubleshooting

### API Key Error
```
Error: ANTHROPIC_API_KEY not found
```
**Fix**: Make sure `.env` file exists and contains your API key

### Rate Limiting
```
Error: Rate limit exceeded
```
**Fix**: Increase `rate_limit_delay` in `config.yaml`:
```yaml
api:
  rate_limit_delay: 2.0
```

### Import Errors
```
ModuleNotFoundError: No module named 'anthropic'
```
**Fix**: Install dependencies:
```bash
pip install -r requirements.txt
```

## What to Expect

### First Few Episodes (1-5)
- Agents are learning collaboration
- Stories may be disconnected
- Rewards typically 5-7

### Middle Episodes (6-15)
- Noticeable improvement
- Better story coherence
- Rewards 7-8+

### Later Episodes (16+)
- Strong collaboration
- High-quality complete stories
- Consistent high rewards

## Next Steps

1. ✓ Run the test
2. ✓ Try a full training run
3. ✓ View the generated plots
4. ✓ Read the full README.md
5. ✓ Customize the config
6. ✓ Experiment with different themes

## Example Session

```bash
$ python main.py --test --verbose

╔═══════════════════════════════════════════════════════════╗
║                    ORCHESTRY                              ║
║         Multi-Agent LLM Reinforcement Learning            ║
╚═══════════════════════════════════════════════════════════╝

Setting up Orchestry...
Creating agents...
  ✓ Creative Writer - Generate creative and engaging story content
  ✓ Editor - Refine content, improve clarity and coherence
  ✓ Narrator - Maintain story flow and tie elements together

Starting training: 3 episodes

Episode 1/3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Turn 1 | Creative Writer:
[Story content appears here...]

Turn 2 | Editor:
[Refined content appears here...]

...

Episode Reward: 7.2
  Story Quality: 7.5
  Collaboration: 7.8
  Efficiency: 6.5

Training Summary
════════════════════════════════════════════════════════════
Total Episodes:     3
Average Reward:     7.1
Best Reward:        7.8

Results saved to: runs/2025-11-14_10-30
```

Enjoy Orchestry! 🎭
