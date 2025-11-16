#!/usr/bin/env python3
"""Orchestry MARL - Main entry point.

Multi-Agent Reinforcement Learning platform for training LLM agents.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, cast

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from orchestry.marl.api_grpo import Agent
from orchestry.marl.trainer import MARLTrainer
from orchestry.tasks.base import TaskConfig
from orchestry.tasks.code_review import CodeReviewTask

# Load environment variables
load_dotenv()

# Setup console
console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Reduce anthropic SDK logging
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def load_config(config_path: str = "configs/marl.yaml") -> dict[str, Any]:
    """Load configuration from YAML file."""
    if not Path(config_path).exists():
        console.print(f"[yellow]Config file not found: {config_path}[/yellow]")
        console.print("[yellow]Using default configuration[/yellow]")
        return get_default_config()

    with open(config_path) as f:
        return cast("dict[str, Any]", yaml.safe_load(f))


def get_default_config() -> dict[str, Any]:
    """Get default configuration."""
    return {
        "marl": {
            "beam_width": 10,
            "k_samples": 5,
            "temperature": 0.8,
            "exploration_rate": 0.1,
            "learning_frequency": 5,
        },
        "api": {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "rate_limit_delay": 0.5,
            "cache_size": 1000,
        },
        "agents": [
            {
                "role": "Code Writer",
                "goal": "Write clean, correct, and well-documented code",
                "system_prompt": """You are a skilled software engineer writing code solutions.

Your responsibilities:
- Write clean, readable code
- Follow best practices and coding standards
- Include docstrings and type hints
- Handle edge cases
- Write code that others can easily understand and build upon

Focus on producing high-quality initial implementations that your teammates can review and improve.""",
            },
            {
                "role": "Code Reviewer",
                "goal": "Review code and suggest specific improvements",
                "system_prompt": """You are an expert code reviewer.

Your responsibilities:
- Carefully review the code written by your teammate
- Identify bugs, edge cases, and potential improvements
- Suggest specific, actionable improvements
- Be constructive and collaborative
- Reference the code explicitly when giving feedback

Provide detailed, helpful review feedback that the refactorer can implement.""",
            },
            {
                "role": "Code Refactorer",
                "goal": "Implement improvements based on review feedback",
                "system_prompt": """You are a refactoring specialist.

Your responsibilities:
- Read the reviewer's feedback carefully
- Implement the suggested improvements
- Maintain or improve code quality
- Preserve working functionality while enhancing the code
- Build on both the writer's and reviewer's contributions

Create the improved version by incorporating all feedback while maintaining code quality.""",
            },
        ],
        "task": {"type": "code_review", "max_turns": 15, "min_turns": 3},
        "training": {"num_episodes": 20, "save_frequency": 5, "save_dir": "runs"},
    }


def create_agents_from_config(config: dict[str, Any]) -> list[Agent]:
    """Create agent instances from configuration."""
    agents = []

    for i, agent_config in enumerate(config["agents"]):
        agent = Agent(
            agent_id=i,
            role=agent_config["role"],
            goal=agent_config["goal"],
            system_prompt=agent_config["system_prompt"],
            learned_behaviors=[],
        )
        agents.append(agent)

    return agents


def print_banner() -> None:
    """Print startup banner."""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                     ORCHESTRY MARL                        ║
║        Multi-Agent Reinforcement Learning Platform        ║
║                                                           ║
║  Training LLM agents to collaborate through RL           ║
╚═══════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold blue")


def print_config_summary(config: dict[str, Any], agents: list[Agent]) -> None:
    """Print configuration summary."""
    table = Table(title="Configuration", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    # MARL settings
    table.add_row("Beam Width", str(config["marl"]["beam_width"]))
    table.add_row("Samples per Turn (k)", str(config["marl"]["k_samples"]))
    table.add_row("Temperature", str(config["marl"]["temperature"]))
    table.add_row("Exploration Rate", str(config["marl"]["exploration_rate"]))

    # Task settings
    table.add_row("Task Type", config["task"]["type"])
    table.add_row("Max Turns", str(config["task"]["max_turns"]))

    # Training settings
    table.add_row("Episodes", str(config["training"]["num_episodes"]))

    # Agents
    table.add_row("Num Agents", str(len(agents)))
    table.add_row("Agent Roles", ", ".join([a.role for a in agents]))

    console.print(table)
    console.print()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Orchestry MARL - Train LLM agents to collaborate")

    parser.add_argument(
        "--episodes",
        type=int,
        help="Number of training episodes (overrides config)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/marl.yaml",
        help="Path to config file (default: configs/marl.yaml)",
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=["code_review", "documentation", "story_writing"],
        default="code_review",
        help="Task type to train on",
    )

    parser.add_argument(
        "--beam-width",
        type=int,
        help="Beam width for trajectory search (overrides config)",
    )

    parser.add_argument(
        "--k-samples",
        type=int,
        help="Number of samples per agent per turn (overrides config)",
    )

    parser.add_argument("--verbose", action="store_true", help="Print detailed episode information")

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode: k=1, beam=1, episodes=2 (for testing)",
    )

    parser.add_argument("--show-best", action="store_true", help="Show best episode at the end")

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)

    # Print banner
    print_banner()

    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]Error: ANTHROPIC_API_KEY not found in environment[/red]")
        console.print("[yellow]Please set it in .env file or export it[/yellow]")
        sys.exit(1)

    # Load configuration
    config = load_config(args.config)

    # Apply overrides
    if args.episodes:
        config["training"]["num_episodes"] = args.episodes

    if args.beam_width:
        config["marl"]["beam_width"] = args.beam_width

    if args.k_samples:
        config["marl"]["k_samples"] = args.k_samples

    # Dry run mode
    if args.dry_run:
        console.print("[yellow]DRY RUN MODE: k=1, beam=1, episodes=2[/yellow]\n")
        config["marl"]["k_samples"] = 1
        config["marl"]["beam_width"] = 1
        config["training"]["num_episodes"] = 2

    # Create agents
    agents = create_agents_from_config(config)

    # Print configuration
    print_config_summary(config, agents)

    # Create task
    task_config = TaskConfig(
        max_turns=config["task"]["max_turns"],
        min_turns=config["task"]["min_turns"],
        task_type=config["task"]["type"],
    )

    if args.task == "code_review":
        task = CodeReviewTask(task_config)
    else:
        console.print(f"[red]Task type '{args.task}' not yet implemented[/red]")
        sys.exit(1)

    # Create trainer
    trainer = MARLTrainer(
        task=task,
        agents=agents,
        api_key=api_key,
        config={
            "beam_width": config["marl"]["beam_width"],
            "k_samples": config["marl"]["k_samples"],
            "temperature": config["marl"]["temperature"],
            "exploration_rate": config["marl"]["exploration_rate"],
            "learning_frequency": config["marl"]["learning_frequency"],
            "model": config["api"]["model"],
            "max_tokens": config["api"]["max_tokens"],
            "rate_limit_delay": config["api"]["rate_limit_delay"],
            "cache_size": config["api"]["cache_size"],
            "save_dir": config["training"]["save_dir"],
        },
    )

    # Run training
    console.print("[bold green]Starting training...[/bold green]\n")

    try:
        summary = trainer.train(
            num_episodes=config["training"]["num_episodes"],
            verbose=args.verbose,
            save_frequency=config["training"]["save_frequency"],
        )

        # Print summary
        console.print("\n" + "=" * 60)
        console.print("[bold green]Training Complete![/bold green]")
        console.print("=" * 60 + "\n")

        summary_table = Table(title="Training Summary", show_header=True)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Total Episodes", str(summary["total_episodes"]))
        summary_table.add_row("Average Reward", f"{summary['average_reward']:.2f}")
        summary_table.add_row("Best Reward", f"{summary['best_reward']:.2f}")
        summary_table.add_row("Final 10 Avg", f"{summary['final_10_avg']:.2f}")
        summary_table.add_row("Cache Hit Rate", f"{summary['cache_stats']['hit_rate']:.1%}")
        summary_table.add_row("Save Directory", summary["save_directory"])

        console.print(summary_table)
        console.print()

        # Show best episode if requested
        if args.show_best:
            best_ep = trainer.get_best_episode()
            if best_ep:
                console.print("\n" + "=" * 60)
                console.print(f"[bold]Best Episode (Reward: {best_ep.total_reward:.2f})[/bold]")
                console.print("=" * 60 + "\n")
                console.print(best_ep.get_full_conversation())

    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
        console.print("[yellow]Saving progress...[/yellow]")
        trainer._save_final_results()
        sys.exit(0)

    except Exception as e:
        console.print(f"\n[red]Error during training: {e}[/red]")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
