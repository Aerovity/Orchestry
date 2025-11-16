# Orchestry 🎭

**Multi-Agent LLM Reinforcement Learning Environment**

Orchestry is a CLI-based system that trains multiple LLM agents to collaborate through reinforcement learning. Watch as AI agents learn to work together, building on each other's ideas to create collaborative stories.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- **Multi-Agent Collaboration**: 3 specialized agents (Creative Writer, Editor, Narrator) work together
- **Reinforcement Learning**: Agents improve through experience using prompt-based learning
- **Real-Time Visualization**: Watch agents collaborate in beautifully formatted CLI output
- **Comprehensive Metrics**: Track story quality, collaboration, and efficiency
- **Learning Analytics**: Visualize improvement with automatically generated plots
- **Extensible Architecture**: Easily add new tasks, agents, or reward functions

## 🎯 How It Works

### The Collaborative Story Writing Task

Three LLM agents collaborate to write creative short stories:

1. **Creative Writer**: Generates creative and engaging story content
2. **Editor**: Refines content, improves clarity and coherence
3. **Narrator**: Maintains story flow and ties elements together

### Reinforcement Learning Loop

```
Episode Start → Agents Collaborate → Story Complete → Calculate Rewards → Update Agents → Repeat
```

**Reward Components:**
- **Story Quality (40%)**: Judge agent rates creativity, coherence, completeness, engagement
- **Collaboration (40%)**: Measures "yes, and" thinking, building on ideas, maintaining continuity
- **Efficiency (20%)**: Rewards completing stories in optimal turn count

### Learning Mechanism

Instead of complex policy gradients, Orchestry uses **prompt engineering as policy**:

1. High-reward episodes → Extract successful behavior patterns
2. Update agent system prompts with learned behaviors
3. Agents incorporate successful patterns in future episodes
4. Track collaboration patterns and inject them into prompts

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Anthropic API key ([Get one here](https://console.anthropic.com/))

### Installation

```bash
# Clone or download Orchestry
cd Orchestry

# Install dependencies
pip install -r requirements.txt

# Set up your API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Run Your First Training

```bash
# Quick test (3 episodes)
python main.py --test --verbose

# Full training run (20 episodes)
python main.py --episodes 20 --verbose

# Custom number of episodes
python main.py --episodes 10
```

### Example Output

```
╔═══════════════════════════════════════════════════════════╗
║                    ORCHESTRY                              ║
║         Multi-Agent LLM Reinforcement Learning            ║
╚═══════════════════════════════════════════════════════════╝

Episode 5/20
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Turn 1 | Creative Writer:
In a dusty antique shop, Elena discovered a peculiar mirror that showed
not her reflection, but glimpses of parallel moments in time...

Turn 2 | Editor:
Building on this intriguing premise, let me refine: Elena noticed the
mirror's ornate frame was inscribed with symbols that seemed to shift
when she wasn't looking directly at them...

Turn 3 | Narrator:
The story deepens as Elena reaches toward the mirror's surface, her
fingertips passing through as if touching water...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Episode Reward: 7.8
  Story Quality: 8.0
  Collaboration: 8.5
  Efficiency: 6.5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Average Reward (last 5): 6.9 → 7.3 ✓ Improving!
```

## 📊 Understanding Results

After training, results are saved to `runs/{timestamp}/`:

```
runs/2025-11-14_10-30/
├── episodes.json          # Complete conversation logs
├── rewards.csv           # Episode-by-episode rewards
├── metrics.json          # Summary statistics
├── agent_stats.json      # Agent performance data
└── plots/
    ├── training_curves.png
    └── collaboration_analysis.png
```

### Key Metrics

- **Total Reward**: Weighted combination of all components (0-10)
- **Story Quality**: Judge's rating of the final story (0-10)
- **Collaboration**: How well agents built on each other's ideas (0-10)
- **Efficiency**: Based on turn count (bonus <10 turns, penalty >20 turns)

## ⚙️ Configuration

Edit `config.yaml` to customize:

### Agent Configuration

```yaml
agents:
  - role: "Creative Writer"
    goal: "Generate creative and engaging story content"
    color: "green"
  # Add more agents...
```

### Training Parameters

```yaml
training:
  num_episodes: 20
  learning_rate: 0.1        # How much to update prompts
  exploration_rate: 0.2     # Probability of trying variations
```

### Reward Weights

```yaml
rewards:
  story_quality_weight: 0.4
  collaboration_weight: 0.4
  efficiency_weight: 0.2
```

### Story Themes

```yaml
story_task:
  themes:
    - "A mysterious discovery"
    - "An unexpected friendship"
    - "A journey to an unknown place"
    # Add your own...
```

## 🎨 Advanced Usage

### Custom Training Run

```bash
# Run 50 episodes with detailed output and show best story
python main.py --episodes 50 --verbose --show-best

# Use custom config
python main.py --config my_config.yaml --episodes 15
```

### Command Line Options

```
--episodes N      Number of training episodes (overrides config)
--test           Run quick 3-episode test
--verbose        Show detailed episode conversations
--show-best      Display the best episode at the end
--config PATH    Path to custom config file
```

## 🏗️ Architecture

### Project Structure

```
Orchestry/
├── main.py              # CLI entry point
├── config.yaml          # Configuration
├── requirements.txt     # Dependencies
├── .env.example        # Environment template
├── src/
│   ├── agent.py        # LLM Agent with memory and learning
│   ├── environment.py  # RL environment and episode management
│   ├── rewards.py      # Reward calculation system
│   ├── trainer.py      # Training loop and learning updates
│   └── utils.py        # Plotting and helper functions
└── runs/               # Training outputs (auto-generated)
```

### Core Components

**Agent** (`agent.py`)
- LLM-powered agent with role and goal
- Episodic and long-term memory
- Learned behaviors that evolve through training
- Prompt construction with learned patterns

**Environment** (`environment.py`)
- Manages episode lifecycle (reset, step, done)
- Tracks conversation state
- Rotates agent turns
- Detects story completion

**Reward Calculator** (`rewards.py`)
- Judge agent evaluates story quality
- Analyzes collaboration patterns
- Calculates efficiency scores
- Extracts successful behaviors for learning

**Trainer** (`trainer.py`)
- Runs training episodes
- Updates agents based on rewards
- Tracks metrics and progress
- Saves checkpoints and results

## 🔧 Customization

### Adding a New Task

Create a new environment class in `environment.py`:

```python
class DebateEnvironment(CollaborativeStoryEnvironment):
    def __init__(self, agents, topic):
        super().__init__(agents)
        self.topic = topic

    def reset(self):
        task = f"Debate the topic: {self.topic}"
        # Custom initialization...
```

### Adding More Agents

In `config.yaml`:

```yaml
agents:
  - role: "Fact Checker"
    goal: "Verify accuracy and add credible details"
    color: "magenta"
```

### Custom Reward Functions

Add to `rewards.py`:

```python
def _evaluate_creativity(self, episode: Episode) -> float:
    # Your custom evaluation logic
    return creativity_score
```

Update weights in `config.yaml`:

```yaml
rewards:
  creativity_weight: 0.3
  # Adjust other weights to sum to 1.0
```

## 📈 Learning Insights

### What to Expect

**Episodes 1-5**: Agents learn basic collaboration patterns
- Initial stories may be disconnected
- Collaboration scores typically 5-7

**Episodes 6-15**: Noticeable improvement
- Agents start building on each other's ideas
- More coherent stories
- Collaboration scores 7-8+

**Episodes 16+**: Refined collaboration
- Smooth story flow
- Strong "yes, and" thinking
- High-quality complete stories

### Signs of Successful Learning

✓ **Increasing average rewards** over episodes
✓ **Collaboration scores improving** faster than other metrics
✓ **Agents referencing previous contributions** explicitly
✓ **More complete stories** with clear structure
✓ **Efficient completion** (optimal turn count)

## 🐛 Troubleshooting

### API Key Issues

```
Error: ANTHROPIC_API_KEY not found
```
**Solution**: Create `.env` file with `ANTHROPIC_API_KEY=your-key-here`

### Rate Limiting

```
Error: Rate limit exceeded
```
**Solution**: Increase `rate_limit_delay` in `config.yaml`:
```yaml
api:
  rate_limit_delay: 2.0  # Increase delay between API calls
```

### Memory/Token Issues

**Solution**: Reduce episode length or token limits:
```yaml
environment:
  max_turns: 10  # Reduce from 15

api:
  max_tokens: 512  # Reduce from 1024
```

## 💡 Tips for Best Results

1. **Start with test mode** (`--test`) to verify everything works
2. **Use verbose mode** initially to understand agent behavior
3. **Adjust reward weights** based on what you want to optimize
4. **Experiment with themes** - some lead to better collaboration
5. **Monitor plots** to see if agents are truly learning
6. **Save successful configs** for future experiments

## 📚 Learn More

### Key Concepts

- **Episodic Learning**: Agents update after each complete story
- **Prompt-Based Policy**: System prompts are the "policy" that improves
- **Collaborative Rewards**: Rewards explicitly measure teamwork
- **Exploration vs Exploitation**: Balance trying new things vs using what works

### Extending Orchestry

The modular architecture makes it easy to:
- Add new collaboration tasks (code review, planning, brainstorming)
- Experiment with different agent roles
- Implement more sophisticated learning algorithms
- Add human-in-the-loop feedback
- Scale to more agents

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional tasks beyond story writing
- More sophisticated learning algorithms
- Interactive mode for human feedback
- Multi-task transfer learning
- Performance optimizations

## 📝 License

MIT License - feel free to use for research or commercial projects.

## 🙏 Acknowledgments

Built with:
- [Anthropic Claude](https://www.anthropic.com/claude) - LLM agents
- [Rich](https://github.com/Textualize/rich) - Beautiful CLI
- [Matplotlib](https://matplotlib.org/) - Visualization

## 📧 Support

For issues, questions, or ideas:
- Open an issue on GitHub
- Check the troubleshooting section
- Review example outputs in `runs/`

---

## 🚀 MVP Product Roadmap

This section outlines the path to turn Orchestry from a research prototype into a production-ready MVP product.

### Phase 1: Core Stability & UX (Weeks 1-2)
**Goal**: Make the product reliable and user-friendly

- [ ] **Error Handling & Recovery**
  - [ ] Graceful handling of API failures with retry logic
  - [ ] Better error messages with actionable suggestions
  - [ ] Auto-save progress during long training runs
  - [ ] Resume interrupted training sessions

- [ ] **Improved CLI Experience**
  - [ ] Interactive setup wizard for first-time users
  - [ ] Progress bars with ETA for long training runs
  - [ ] Keyboard shortcuts (pause/resume training, skip episode)
  - [ ] Export training runs to shareable format

- [ ] **Configuration Validation**
  - [ ] Validate config.yaml on startup
  - [ ] Provide helpful error messages for invalid configs
  - [ ] Config templates for common use cases
  - [ ] In-app config editor with validation

### Phase 2: Web Dashboard (Weeks 3-4)
**Goal**: Provide a modern web interface for better visualization

- [ ] **Real-Time Web Dashboard**
  - [ ] Live training monitoring via web browser
  - [ ] Interactive plots with zoom/pan
  - [ ] Real-time agent conversation streaming
  - [ ] Training control panel (start/stop/pause)

- [ ] **Results Gallery**
  - [ ] Browse all training runs
  - [ ] Compare multiple runs side-by-side
  - [ ] Search and filter episodes
  - [ ] Export best episodes as PDF/HTML

- [ ] **Analytics & Insights**
  - [ ] Advanced metrics dashboard
  - [ ] Collaboration pattern visualization
  - [ ] Agent performance comparison
  - [ ] Learning curve predictions

### Phase 3: Multi-Task Support (Weeks 5-6)
**Goal**: Expand beyond story writing to multiple domains

- [ ] **New Collaboration Tasks**
  - [ ] Code Review: One writes, another reviews, third refactors
  - [ ] Business Planning: Brainstorming, analysis, refinement
  - [ ] Debate/Discussion: Agents take positions and debate
  - [ ] Research Synthesis: Agents research and synthesize findings

- [ ] **Task Management System**
  - [ ] Task marketplace/library
  - [ ] Easy task switching via CLI/web
  - [ ] Custom task builder wizard
  - [ ] Task-specific reward functions

- [ ] **Transfer Learning**
  - [ ] Save learned behaviors per task
  - [ ] Transfer knowledge between similar tasks
  - [ ] Meta-learning across task families

### Phase 4: Collaboration Features (Weeks 7-8)
**Goal**: Enable teams to collaborate on training

- [ ] **Team Features**
  - [ ] Multi-user support with authentication
  - [ ] Shared training runs and results
  - [ ] Comments and annotations on episodes
  - [ ] Team leaderboards and challenges

- [ ] **Human-in-the-Loop**
  - [ ] Manual reward override for specific episodes
  - [ ] Human feedback injection during training
  - [ ] Guided exploration based on user preferences
  - [ ] A/B testing different configurations

- [ ] **Community Platform**
  - [ ] Share trained agents publicly
  - [ ] Download and use community agents
  - [ ] Rate and review agent behaviors
  - [ ] Competitions and challenges

### Phase 5: Production Scale (Weeks 9-10)
**Goal**: Make it production-ready for enterprises

- [ ] **Performance & Scale**
  - [ ] Parallel episode execution
  - [ ] GPU acceleration for training
  - [ ] Distributed training across machines
  - [ ] Caching and optimization for API calls

- [ ] **Enterprise Features**
  - [ ] Docker containerization
  - [ ] Kubernetes deployment configs
  - [ ] API for programmatic access
  - [ ] Webhooks for training events

- [ ] **Security & Compliance**
  - [ ] Secure API key management (vault integration)
  - [ ] Audit logs for all actions
  - [ ] Role-based access control
  - [ ] Data privacy controls (PII filtering)

### Phase 6: Advanced RL & AI (Weeks 11-12)
**Goal**: State-of-the-art learning capabilities

- [ ] **Advanced Learning Algorithms**
  - [ ] PPO (Proximal Policy Optimization) implementation
  - [ ] Multi-agent RL algorithms (MADDPG)
  - [ ] Curriculum learning (start simple, increase difficulty)
  - [ ] Meta-learning and few-shot adaptation

- [ ] **Agent Capabilities**
  - [ ] Long-term memory with vector databases
  - [ ] Tool use and API calling
  - [ ] Multi-modal inputs (images, documents)
  - [ ] Agent introspection and self-improvement

- [ ] **Research Features**
  - [ ] Experiment tracking (MLflow/Weights & Biases)
  - [ ] Hyperparameter optimization
  - [ ] Automated ablation studies
  - [ ] Research paper export templates

### Phase 7: Productization (Weeks 13-14)
**Goal**: Launch-ready product with business model

- [ ] **Pricing & Monetization**
  - [ ] Free tier (limited episodes/month)
  - [ ] Pro tier (unlimited, advanced features)
  - [ ] Enterprise tier (on-premise, custom support)
  - [ ] API usage-based pricing

- [ ] **Documentation & Support**
  - [ ] Video tutorials and demos
  - [ ] Interactive documentation site
  - [ ] In-app help and tooltips
  - [ ] 24/7 support channel (community + paid)

- [ ] **Marketing & Launch**
  - [ ] Landing page with live demos
  - [ ] Blog posts and case studies
  - [ ] Social media presence
  - [ ] Launch on Product Hunt

### Quick Wins (Can Do Immediately)
**Low-effort, high-impact improvements**

- [x] ✅ GitHub repository with clean README
- [ ] Add GitHub Actions for CI/CD
- [ ] Create demo video (3-5 minutes)
- [ ] Set up documentation site (GitHub Pages)
- [ ] Add more example tasks to showcase flexibility
- [ ] Create Discord/Slack community
- [ ] Write blog post: "How we built multi-agent RL"
- [ ] Submit to AI/ML newsletters and communities

### Success Metrics

**MVP Launch Criteria:**
- [ ] 100+ GitHub stars
- [ ] 50+ active users per week
- [ ] <2% error rate in production
- [ ] Average training completion rate >90%
- [ ] User satisfaction score >4.0/5.0

**Growth Metrics:**
- [ ] 1,000+ total users in first 3 months
- [ ] 10+ community-contributed tasks
- [ ] 5+ case studies from real users
- [ ] 50+ trained agent models shared
- [ ] Partnership with 3+ AI research labs

---

**MVP Timeline**: ~14 weeks (3.5 months) from prototype to product
**Investment Needed**: $50K-100K (1-2 engineers, infrastructure, API costs)
**Revenue Potential**: $10K-50K MRR within 6 months post-launch

---

**Happy Orchestrating! 🎭**

Watch your agents learn to collaborate and create amazing stories together.
