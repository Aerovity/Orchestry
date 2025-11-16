#!/usr/bin/env python3
"""Orchestry Legacy - Command-line interface.

Legacy collaborative story writing system.
For new projects, use the MARL system instead.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console

from orchestry.legacy.agent import LLMAgent
from orchestry.legacy.environment import CollaborativeStoryEnvironment
from orchestry.legacy.rewards import RewardCalculator
from orchestry.legacy.trainer import Trainer
from orchestry.legacy.utils import (
    format_episode_summary,
    get_learning_insights,
    load_config,
    plot_collaboration_analysis,
    plot_training_curves,
    setup_logging,
)

console = Console()


def display_banner() -> None:
    """Display Orchestry banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║         ORCHESTRY LEGACY - Story Writing System           ║
    ║                                                           ║
    ║  Note: Consider using the MARL system for new projects   ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold yellow")


def setup_environment() -> tuple[dict[str, Any], str]:
    """Setup and validate environment.

    Returns:
        Tuple of (config, api_key)

    """
    # Load environment variables
    load_dotenv()

    # Get API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]Error: ANTHROPIC_API_KEY not found in environment[/red]")
        console.print("Please set your API key in a .env file or environment variable")
        sys.exit(1)

    # Load config
    config_path = os.getenv("ORCHESTRY_CONFIG", "configs/legacy.yaml")
    try:
        config = load_config(config_path)
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        sys.exit(1)

    return config, api_key


def create_agents(config: dict[str, Any], api_key: str) -> list[LLMAgent]:
    """Create LLM agents from config.

    Args:
        config: Configuration dictionary
        api_key: Anthropic API key

    Returns:
        List of LLMAgent instances

    """
    agents = []

    for agent_config in config["agents"]:
        agent = LLMAgent(
            name=agent_config["name"],
            role=agent_config["role"],
            api_key=api_key,
            model=agent_config.get("model", config["llm"]["model"]),
            max_tokens=agent_config.get("max_tokens", config["llm"]["max_tokens"]),
            temperature=agent_config.get("temperature", config["llm"]["temperature"]),
        )
        agents.append(agent)

    return agents


def main() -> None:
    """Main entry point for legacy system."""
    parser = argparse.ArgumentParser(
        description="Orchestry Legacy - Multi-Agent Story Writing Training",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of training episodes (default: 10)",
    )

    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Max turns per episode (default: 10)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate training plots after completion",
    )

    args = parser.parse_args()

    # Display banner
    display_banner()

    # Setup logging
    setup_logging(verbose=args.verbose)

    # Setup environment
    config, api_key = setup_environment()

    # Create agents
    console.print("[cyan]Creating agents...[/cyan]")
    agents = create_agents(config, api_key)
    console.print(f"[green]Created {len(agents)} agents[/green]\n")

    # Create environment
    console.print("[cyan]Initializing environment...[/cyan]")
    env = CollaborativeStoryEnvironment(
        agents=agents,
        max_turns=args.max_turns,
        theme=config["environment"]["theme"],
    )

    # Create reward calculator
    reward_calc = RewardCalculator(
        quality_weight=config["rewards"]["quality_weight"],
        collaboration_weight=config["rewards"]["collaboration_weight"],
    )

    # Create trainer
    trainer = Trainer(env=env, reward_calculator=reward_calc, num_episodes=args.episodes)

    # Run training
    console.print("[bold green]Starting training...[/bold green]\n")

    try:
        history = trainer.train(verbose=args.verbose)

        # Print summary
        console.print("\n" + "=" * 60)
        console.print("[bold green]Training Complete![/bold green]")
        console.print("=" * 60 + "\n")

        # Show insights
        insights = get_learning_insights(history)
        console.print(format_episode_summary(insights))

        # Generate plots if requested
        if args.plot:
            console.print("\n[cyan]Generating training plots...[/cyan]")
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)

            plot_training_curves(history, save_path=output_dir / "training_curves.png")
            plot_collaboration_analysis(
                history,
                save_path=output_dir / "collaboration_analysis.png",
            )

            console.print(f"[green]Plots saved to {output_dir}/[/green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
        sys.exit(0)

    except Exception as e:
        console.print(f"\n[red]Error during training: {e}[/red]")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
