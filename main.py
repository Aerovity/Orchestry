#!/usr/bin/env python3
"""
Orchestry: Multi-Agent LLM Reinforcement Learning Environment

Main CLI entry point for running collaborative story writing training.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich import box
import time

from src.agent import LLMAgent
from src.environment import CollaborativeStoryEnvironment
from src.rewards import RewardCalculator
from src.trainer import Trainer
from src.utils import (
    load_config,
    setup_logging,
    plot_training_curves,
    plot_collaboration_analysis,
    format_episode_summary,
    get_learning_insights
)

console = Console()


def display_banner():
    """Display Orchestry banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   ██████╗ ██████╗  ██████╗██╗  ██╗███████╗███████╗████████╗██████╗ ██╗   ██╗
    ║  ██╔═══██╗██╔══██╗██╔════╝██║  ██║██╔════╝██╔════╝╚══██╔══╝██╔══██╗╚██╗ ██╔╝
    ║  ██║   ██║██████╔╝██║     ███████║█████╗  ███████╗   ██║   ██████╔╝ ╚████╔╝
    ║  ██║   ██║██╔══██╗██║     ██╔══██║██╔══╝  ╚════██║   ██║   ██╔══██╗  ╚██╔╝
    ║  ╚██████╔╝██║  ██║╚██████╗██║  ██║███████╗███████║   ██║   ██║  ██║   ██║
    ║   ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚══════╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝
    ║                                                           ║
    ║         Multi-Agent LLM Reinforcement Learning            ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


def setup_environment() -> tuple:
    """
    Setup and validate environment.

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
    config_path = os.getenv("ORCHESTRY_CONFIG", "config.yaml")
    try:
        config = load_config(config_path)
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        sys.exit(1)

    return config, api_key


def create_agents(config: dict, api_key: str) -> list:
    """
    Create LLM agents from config.

    Args:
        config: Configuration dictionary
        api_key: Anthropic API key

    Returns:
        List of LLMAgent instances
    """
    agents = []

    for agent_config in config["agents"]:
        agent = LLMAgent(
            role=agent_config["role"],
            goal=agent_config["goal"],
            api_key=api_key,
            model=config["api"]["model"],
            temperature=config["api"]["temperature"],
            max_tokens=config["api"]["max_tokens"],
            color=agent_config.get("color", "white")
        )
        agents.append(agent)

    return agents


def display_episode_progress(episode_num: int, total_episodes: int, conversation: list, rewards: Optional[dict] = None):
    """
    Display episode progress in real-time.

    Args:
        episode_num: Current episode number
        total_episodes: Total episodes
        conversation: Conversation history
        rewards: Rewards (if episode completed)
    """
    # Create header
    header = f"Episode {episode_num}/{total_episodes}"

    # Display conversation
    for turn_data in conversation:
        agent_role = turn_data["agent"]
        content = turn_data["content"]

        # Color based on agent
        color_map = {
            "Creative Writer": "green",
            "Editor": "yellow",
            "Narrator": "blue"
        }
        color = color_map.get(agent_role, "white")

        console.print(f"\n[bold {color}]Turn {turn_data['turn']} | {agent_role}:[/bold {color}]")
        console.print(f"[{color}]{content}[/{color}]")

    # Display rewards if available
    if rewards:
        console.print("\n" + "━" * 80)

        rewards_table = Table(show_header=False, box=box.SIMPLE)
        rewards_table.add_column(style="bold")
        rewards_table.add_column(style="cyan")

        rewards_table.add_row("Episode Reward:", f"{rewards['total']:.2f}")
        rewards_table.add_row("  Story Quality:", f"{rewards['story_quality']:.2f}")
        rewards_table.add_row("  Collaboration:", f"{rewards['collaboration']:.2f}")
        rewards_table.add_row("  Efficiency:", f"{rewards['efficiency']:.2f}")

        console.print(rewards_table)
        console.print("━" * 80)


def display_training_summary(metrics: dict, save_dir: Path):
    """
    Display training summary.

    Args:
        metrics: Training metrics
        save_dir: Save directory
    """
    console.print("\n[bold cyan]Training Summary[/bold cyan]")
    console.print("=" * 80)

    # Create summary table
    summary_table = Table(show_header=False, box=box.ROUNDED)
    summary_table.add_column(style="bold")
    summary_table.add_column(style="cyan")

    summary_table.add_row("Total Episodes:", str(metrics["total_episodes"]))
    summary_table.add_row("Average Reward:", f"{metrics['average_reward']:.2f}")
    summary_table.add_row("Best Reward:", f"{metrics['best_reward']:.2f}")
    summary_table.add_row("Worst Reward:", f"{metrics['worst_reward']:.2f}")

    console.print(summary_table)

    # Learning insights
    console.print("\n[bold]Learning Insights:[/bold]")
    insights = get_learning_insights(metrics)
    for insight in insights:
        console.print(f"  {insight}")

    # Save location
    console.print(f"\n[bold]Results saved to:[/bold] {save_dir}")
    console.print("=" * 80)


def run_training(args):
    """
    Run training loop.

    Args:
        args: Command line arguments
    """
    display_banner()

    # Setup
    console.print("[bold]Setting up Orchestry...[/bold]")
    config, api_key = setup_environment()

    # Setup logging
    log_level = config.get("output", {}).get("log_level", "INFO")
    setup_logging(log_level)

    # Create agents
    console.print("[bold]Creating agents...[/bold]")
    agents = create_agents(config, api_key)

    for agent in agents:
        console.print(f"  ✓ {agent.role} - {agent.goal}")

    # Create environment
    env = CollaborativeStoryEnvironment(
        agents=agents,
        max_turns=config["environment"]["max_turns"],
        story_target_length=config["environment"]["story_target_length"],
        themes=config["story_task"]["themes"]
    )

    # Create reward calculator
    reward_calc = RewardCalculator(
        api_key=api_key,
        model=config["api"]["model"],
        story_quality_weight=config["rewards"]["story_quality_weight"],
        collaboration_weight=config["rewards"]["collaboration_weight"],
        efficiency_weight=config["rewards"]["efficiency_weight"],
        efficiency_threshold_good=config["rewards"]["efficiency_threshold_good"],
        efficiency_threshold_bad=config["rewards"]["efficiency_threshold_bad"]
    )

    # Create trainer
    trainer = Trainer(
        agents=agents,
        environment=env,
        reward_calculator=reward_calc,
        learning_rate=config["training"]["learning_rate"],
        exploration_rate=config["training"]["exploration_rate"],
        save_dir="runs"
    )

    # Determine number of episodes
    if args.test:
        num_episodes = config["training"]["test_episodes"]
        console.print(f"\n[yellow]Running in TEST mode: {num_episodes} episodes[/yellow]\n")
    else:
        num_episodes = args.episodes or config["training"]["num_episodes"]

    console.print(f"\n[bold green]Starting training: {num_episodes} episodes[/bold green]\n")

    # Training loop
    for i in range(1, num_episodes + 1):
        console.print(f"\n[bold cyan]{'=' * 80}[/bold cyan]")
        console.print(f"[bold cyan]Episode {i}/{num_episodes}[/bold cyan]")
        console.print(f"[bold cyan]{'=' * 80}[/bold cyan]\n")

        # Run episode with progress display
        with console.status(f"[bold green]Running episode {i}...") as status:
            episode = env.run_episode()

            # Display conversation if verbose
            if args.verbose or args.test:
                display_episode_progress(i, num_episodes, episode.conversation)

            # Calculate rewards
            status.update("[bold green]Calculating rewards...")
            rewards = reward_calc.calculate_rewards(episode)

            episode.rewards = rewards
            episode.total_reward = rewards["total"]

            trainer.episodes.append(episode)
            trainer.metrics.add_episode(rewards, len(episode.conversation))

            # Learning update
            if rewards["total"] > 6.0:
                trainer._update_agents(episode, rewards)

        # Display rewards
        if not (args.verbose or args.test):
            # Just show summary
            recent_avg = trainer.metrics.get_recent_average()
            improving = "✓ Improving!" if trainer.metrics.is_improving() else ""

            console.print(
                f"\n[bold]Episode {i}:[/bold] "
                f"Reward={rewards['total']:.2f}, "
                f"Avg(5)={recent_avg:.2f} {improving}"
            )
        else:
            display_episode_progress(i, num_episodes, episode.conversation, rewards)

            # Show progress
            recent_avg = trainer.metrics.get_recent_average()
            prev_avg = trainer.metrics.get_recent_average(10) if i > 5 else recent_avg

            if recent_avg > prev_avg:
                console.print(f"\n[green]Average Reward (last 5): {prev_avg:.2f} → {recent_avg:.2f} ✓ Improving![/green]")
            else:
                console.print(f"\n[yellow]Average Reward (last 5): {recent_avg:.2f}[/yellow]")

        # Save checkpoint
        if i % config["training"]["save_frequency"] == 0:
            trainer._save_checkpoint(i)

    # Final save
    console.print("\n[bold]Saving results...[/bold]")
    trainer._save_final_results()

    # Generate plots
    if config["output"]["generate_plots"]:
        console.print("[bold]Generating plots...[/bold]")
        try:
            plot_training_curves(trainer.metrics.to_dict(), trainer.save_dir)
            plot_collaboration_analysis(trainer.metrics.to_dict(), trainer.save_dir)
            console.print("  ✓ Plots generated")
        except Exception as e:
            console.print(f"  [yellow]Warning: Could not generate plots: {e}[/yellow]")

    # Display summary
    display_training_summary(trainer.metrics.to_dict(), trainer.save_dir)

    # Show best episode
    best_episode = trainer.get_best_episode()
    if best_episode and args.show_best:
        console.print("\n[bold cyan]Best Episode:[/bold cyan]")
        console.print(format_episode_summary(best_episode.to_dict()))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Orchestry: Multi-Agent LLM Reinforcement Learning"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        help="Number of episodes to run (overrides config)"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (3 quick episodes)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed episode output"
    )

    parser.add_argument(
        "--show-best",
        action="store_true",
        help="Display best episode at the end"
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file"
    )

    args = parser.parse_args()

    # Set config path
    if args.config:
        os.environ["ORCHESTRY_CONFIG"] = args.config

    try:
        run_training(args)
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Training interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
