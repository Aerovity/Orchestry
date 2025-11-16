# Orchestry 🎭

**Production-Ready Multi-Agent Reinforcement Learning Platform for LLMs**

Orchestry trains multiple LLM agents to collaborate using **real Multi-Agent Reinforcement Learning (MARL)**. Watch AI agents learn to work together on production tasks like code review, documentation, and more.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![MARL](https://img.shields.io/badge/MARL-Production-brightgreen.svg)

---

## 🚀 What's New in v1.0 - Production MARL

Orchestry has been **completely rewritten** from a prompt-engineering demo into a **production-ready MARL platform**:

### ✨ Real Multi-Agent RL (Not Prompt Hacking)
- **API-based GRPO**: Group Relative Policy Optimization adapted for LLM APIs
- **Beam Search**: Explores 10+ trajectory candidates per episode
- **Centralized Value Estimation**: Claude acts as judge agent for multi-agent interactions
- **Behavior Pattern Extraction**: Meta-learning from successful episodes

### 🎯 Production Tasks (Beyond Story Writing)
- **Code Review** (NEW): 3 agents collaborate to write, review, and refactor code
- **Documentation** (Coming Soon): Generate comprehensive technical docs
- **Research Synthesis** (Coming Soon): Multi-agent research and analysis
- Story Writing (Legacy task, still supported)

### 📊 Real Learning with Measurable Improvement
- Agents **actually improve** through RL, not just prompt updates
- Quantitative collaboration metrics
- Reproducible results with experiment tracking

---

## 🎯 How It Works - The MARL System

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   MARL Training Loop                    │
│                                                         │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────┐ │
│  │ Multi-Sample│───▶│ Beam Search  │───▶│ Best     │ │
│  │ Generation  │    │ (Top N)      │    │ Selection│ │
│  │ (k=5)       │    │              │    │ (GRPO)   │ │
│  └─────────────┘    └──────────────┘    └──────────┘ │
│         │                   │                  │       │
│         ▼                   ▼                  ▼       │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────┐ │
│  │ Centralized │    │  Advantage   │    │ Behavior │ │
│  │ Value Est.  │───▶│  Calculation │───▶│ Learning │ │
│  │ (Judge)     │    │              │    │          │ │
│  └─────────────┘    └──────────────┘    └──────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Key Innovation: API-Based MARL

Since we can't fine-tune Claude's weights directly, Orchestry implements **MARL through intelligent trajectory search**:

1. **Multi-Sampling**: Generate k=5 responses per agent per turn
2. **Beam Search**: Keep top-N=10 trajectories, prune rest
3. **Group-Relative Advantages**: A(τ) = R(τ) - mean(all trajectories)
4. **Best Selection**: Pick trajectory with highest advantage
5. **Meta-Learning**: Extract patterns from top-20% episodes → update agent prompts

Think of it as: **"MARL-guided Monte Carlo Tree Search over conversation space"**

---

## 📦 Installation

### Prerequisites
- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip
- Anthropic API key ([Get one here](https://console.anthropic.com/))

### Quick Setup (Recommended - using uv)

```bash
# Clone repository
git clone https://github.com/Aerovity/Orchestry.git
cd Orchestry

# Create virtual environment and install (uv handles everything)
uv venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Set up API key
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=your-key-here

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

### Alternative Setup (using pip)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package
pip install -e ".[dev]"

# Set up API key
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=your-key-here
```

---

## 🎮 Quick Start

### Run Your First MARL Training

```bash
# Using the CLI commands (recommended)
orchestry-marl --dry-run --verbose  # Quick test (2 episodes)
orchestry-marl --episodes 10 --verbose  # Small run
orchestry-marl --task code_review --episodes 20  # Full training

# Or run directly from examples/
python examples/run_marl.py --dry-run --verbose

# Using Make (if you have Makefile)
make run-marl  # Runs in dry-run mode
```

### Example Output

```
╔═══════════════════════════════════════════════════════════╗
║                     ORCHESTRY MARL                        ║
║        Multi-Agent Reinforcement Learning Platform        ║
╚═══════════════════════════════════════════════════════════╝

Configuration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Beam Width                    10
Samples per Turn (k)          5
Task Type                     code_review
Episodes                      20

Episode 5/20
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Task: Binary Search

Turn 1 | Code Writer:
```python
def binary_search(arr: list[int], target: int) -> int:
    """Binary search implementation..."""
    ...
```

Turn 2 | Code Reviewer:
The code looks good, but I notice a few edge cases...

Turn 3 | Code Refactorer:
FINAL CODE: Improved version with edge case handling

Selected trajectory 3
Reward: 8.2 (Q=8.5, C=8.8, E=7.5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Training: 100%|████████████| 20/20 [reward=7.8, avg_10=8.1]

Training Complete!
Average Reward: 7.65
Best Reward: 9.20
Final 10 Avg: 8.15 ⬆ IMPROVING!
```

---

## ⚙️ Configuration

Edit `config_marl.yaml` to customize training:

### MARL Algorithm

```yaml
marl:
  beam_width: 10          # Trajectories to explore
  k_samples: 5            # Responses per agent per turn
  temperature: 0.8        # Sampling diversity
  exploration_rate: 0.1   # Exploration vs exploitation
```

### Reward Weights

```yaml
rewards:
  quality_weight: 0.4        # Code correctness, readability
  collaboration_weight: 0.4  # How well agents built on each other
  efficiency_weight: 0.2     # Turn count efficiency
```

---

## 📊 Understanding Results

After training, find results in `runs/marl_<timestamp>/`:

```
runs/marl_2025-01-15_14-30-00/
├── episodes.json           # All episode transcripts
├── rewards.csv            # Reward history
├── learned_behaviors.json # Extracted behavioral patterns
├── summary.json           # Training statistics
└── checkpoint_ep*.json    # Periodic checkpoints
```

### Key Metrics

- **Quality** (0-10): Code correctness, readability, handles edge cases
- **Collaboration** (0-10): Agents reference each other, build incrementally
- **Efficiency** (0-10): Completed in optimal turn count (6-12 turns ideal)
- **Total**: Weighted sum (configurable)

---

## 🎨 Advanced Usage

### Command Line Options

```bash
# Override config settings
python main_marl.py --episodes 50 --beam-width 15 --k-samples 7

# Show best episode at end
python main_marl.py --episodes 10 --show-best

# Custom config file
python main_marl.py --config my_custom_config.yaml

# Dry run (fast test)
python main_marl.py --dry-run --verbose
```

---

## 🏗️ Project Structure

```
Orchestry/
├── main_marl.py              # MARL entry point (NEW)
├── config_marl.yaml          # MARL configuration (NEW)
│
├── src/
│   ├── marl/                 # MARL implementation (NEW)
│   │   ├── api_grpo.py       # Group Relative Policy Optimization
│   │   ├── trainer.py        # Training loop with beam search
│   │   ├── trajectory.py     # Multi-turn trajectory tracking
│   │   ├── value_estimator.py # Centralized value estimation
│   │   └── behavior_library.py # Pattern extraction
│   │
│   ├── tasks/                # Task implementations (NEW)
│   │   ├── base.py           # Abstract task interface
│   │   └── code_review.py    # Code review task
│   │
│   └── legacy/               # Original implementation
│
└── runs/                     # Training outputs
```

---

## 🔬 Technical Deep Dive

### Why API-Based MARL?

Traditional MARL requires fine-tuning model weights. With API-only access to Claude, we implement **"policy optimization through trajectory search"**:

```python
# Orchestry's approach
for episode in episodes:
    # Generate multiple candidate trajectories
    candidates = beam_search(k_samples=5, beam_width=10)

    # Score all candidates
    rewards = [evaluate(t) for t in candidates]

    # Pick best using group-relative advantages
    advantages = rewards - mean(rewards)
    best = candidates[argmax(advantages)]

    # Learn patterns from top episodes (meta-learning)
    if episode % 5 == 0:
        patterns = extract_behaviors(top_episodes)
        update_agent_prompts(patterns)
```

### GRPO: Group Relative Policy Optimization

The key insight is **group-relative advantages**:

```
A(trajectory_i) = R(trajectory_i) - mean(R(all trajectories))
```

This encourages agents to:
- Collaborate (maximize joint reward)
- Coordinate (find strategies that work together)
- Avoid local optima (compare against alternatives)

---

## 💰 Cost Considerations

### API Usage per Episode

```
Calls = k_samples × beam_width × num_turns × num_agents
      = 5 × 10 × 10 × 3 = 1,500 API calls per episode

With caching (~40% hit rate): ≈900 actual API calls
```

### Cost Estimates (Claude 3.5 Sonnet)

- **Dry run** (2 episodes): ~$0.50
- **Small training** (10 episodes): ~$5-10
- **Full training** (20 episodes): ~$10-20
- **Production run** (100 episodes): ~$50-100

**Cost Optimization:**
1. Use `--dry-run` for testing
2. Enable caching (default)
3. Start with lower beam_width
4. Use cheaper models for experiments

---

## 🧪 Extending Orchestry

### Adding a New Task

```python
# src/tasks/documentation.py
from .base import BaseTask

class DocumentationTask(BaseTask):
    def reset(self):
        return {'task_description': ...}

    def step(self, agent_id, agent_role, action):
        return observation, done

    def evaluate(self):
        return {'quality': ..., 'collaboration': ..., 'efficiency': ...}
```

Register in main_marl.py and run:

```bash
python main_marl.py --task documentation --episodes 20
```

---

## 🐛 Troubleshooting

### API Key Issues
```
Error: ANTHROPIC_API_KEY not found
```
**Solution:** Create `.env` file with `ANTHROPIC_API_KEY=your-key-here`

### Rate Limiting
```
Error: Rate limit exceeded
```
**Solution:** Increase `rate_limit_delay` in config_marl.yaml

### Slow Training
**Solution:** Use dry-run mode:
```bash
python main_marl.py --dry-run  # k=1, beam=1, episodes=2
```

---

## 🎯 Roadmap

### ✅ Completed (v1.0)
- API-based GRPO implementation
- Beam search trajectory optimization
- Code review production task
- Behavior pattern learning

### 🚧 In Progress (v1.1)
- [ ] Streamlit web dashboard
- [ ] Weights & Biases integration
- [ ] Documentation task
- [ ] Research synthesis task

### 🔮 Planned (v1.2+)
- [ ] Docker containerization
- [ ] Multi-task transfer learning
- [ ] Human-in-the-loop feedback
- [ ] Distributed training

---

## 📚 Learn More

### Key Concepts

- **MARL**: Multi-Agent Reinforcement Learning
- **GRPO**: Group Relative Policy Optimization
- **Beam Search**: Keeping top-N candidates
- **CTDE**: Centralized Training, Decentralized Execution
- **Meta-Learning**: Learning patterns from learning

### Research Papers

- [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- [Multi-Agent RL: A Selective Overview](https://arxiv.org/abs/1911.10635)

---

## 📁 Project Structure

```
orchestry/
├── pyproject.toml          # Modern Python project config & dependencies
├── uv.lock                 # Locked dependencies for reproducibility
├── Makefile                # Development commands (format, test, lint)
├── .pre-commit-config.yaml # Git hooks for code quality
│
├── orchestry/              # Main package (installable)
│   ├── __init__.py
│   ├── py.typed           # PEP 561 type marker
│   │
│   ├── marl/              # MARL implementation
│   │   ├── api_grpo.py    # Group Relative Policy Optimization
│   │   ├── trainer.py     # Main training loop with beam search
│   │   ├── trajectory.py  # Trajectory and turn management
│   │   ├── value_estimator.py  # Centralized value estimation
│   │   └── behavior_library.py  # Meta-learning from success
│   │
│   ├── tasks/             # Task definitions
│   │   ├── base.py        # Abstract task interface
│   │   └── code_review.py # Code review task (3-agent collaboration)
│   │
│   ├── legacy/            # Legacy story writing system
│   │   ├── agent.py
│   │   ├── environment.py
│   │   ├── rewards.py
│   │   ├── trainer.py
│   │   └── utils.py
│   │
│   └── cli/               # Command-line interfaces
│       ├── marl.py        # MARL CLI (orchestry-marl command)
│       └── legacy.py      # Legacy CLI (orchestry-legacy command)
│
├── tests/                 # Test suite (mirrors package structure)
│   ├── marl/
│   ├── tasks/
│   └── legacy/
│
├── examples/              # Standalone example scripts
│   ├── run_marl.py        # MARL training example
│   └── run_legacy.py      # Legacy training example
│
├── configs/               # Configuration files
│   ├── marl.yaml          # MARL system configuration
│   └── legacy.yaml        # Legacy system configuration
│
└── docs/                  # Documentation
    ├── QUICKSTART.md
    ├── PROJECT_SUMMARY.md
    ├── IMPLEMENTATION_SUMMARY.md
    └── MIGRATION_GUIDE.md
```

### Development Tools

- **Black**: Code formatting (line length: 100)
- **Ruff**: Fast linting and import sorting
- **Mypy**: Strict type checking
- **Pytest**: Testing framework with coverage
- **Pre-commit**: Automated code quality checks

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

Areas for improvement:
- New tasks (business planning, creative writing, debates)
- Better reward functions (automated testing)
- Performance optimization (parallel beam search)
- Documentation and tutorials

---

## 📝 License

MIT License - free to use for research or commercial projects.

---

## 🙏 Acknowledgments

Built with:
- [Anthropic Claude](https://www.anthropic.com/claude) - LLM agents
- [Rich](https://github.com/Textualize/rich) - Beautiful CLI
- [NumPy](https://numpy.org/) - Numerical computations

---

**Happy Orchestrating! 🎭**

Transform your LLM agents from solo performers into a coordinated ensemble.
