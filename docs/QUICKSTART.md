# Orchestry MARL - Quick Start Guide

Get started with Orchestry in 5 minutes! 🚀

## Step 1: Installation (2 minutes)

```bash
# Clone the repository
git clone https://github.com/Aerovity/Orchestry.git
cd Orchestry

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Configure API Key (1 minute)

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your API key
# For Claude:
# ANTHROPIC_API_KEY=sk-ant-...
#
# For Gemini:
# GEMINI_API_KEY=AIzaSy...
```

Get your API keys at:
- Claude: https://console.anthropic.com/
- Gemini: https://makersuite.google.com/app/apikey

**Note**: You can use either Claude or Gemini for research lab. Configure the provider in `configs/research_lab.yaml`:

## Step 3: Test Installation (1 minute)

```bash
# Run quick test to verify everything works
python test_marl.py
```

Expected output:
```
✓ All imports successful
✓ API key found
✓ Trajectory system working
✓ Task system working
✓ Agent system working
```

## Step 4: Run Dry Run (1 minute)

```bash
# Fast test: 2 episodes, k=1, beam=1 (~$0.50)
python main_marl.py --dry-run --verbose
```

## Step 5: Run Real Training (10-20 minutes)

```bash
# Small training: 10 episodes (~$5-10)
python main_marl.py --episodes 10 --verbose --show-best
```

Check results in `runs/marl_<timestamp>/`!

---

For full documentation, see [README.md](README.md)
