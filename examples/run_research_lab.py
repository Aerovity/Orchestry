#!/usr/bin/env python3
"""
Autonomous Research Lab - Multi-Agent Scientific Research System

This example demonstrates a 5-agent system that conducts autonomous scientific research:
1. Literature Synthesizer - Reviews and synthesizes research papers
2. Hypothesis Generator - Creates novel, testable hypotheses
3. Experimental Designer - Designs rigorous experiments
4. Data Analyst - Analyzes experimental results
5. Paper Writer - Writes research paper drafts

Usage:
    python examples/run_research_lab.py --domain materials_science --episodes 20
    python examples/run_research_lab.py --domain climate --dry-run
    python examples/run_research_lab.py --domain protein --show-best
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from orchestry.marl.trainer import MARLTrainer
from orchestry.tasks.research_lab import ResearchLabTask


console = Console()


def print_header() -> None:
    """Print fancy header."""
    header = """
    ╔═══════════════════════════════════════════════════════════╗
    ║         AUTONOMOUS RESEARCH LABORATORY v1.0               ║
    ║      Multi-Agent Scientific Research Collaboration        ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(header, style="bold cyan")


def load_config(config_path: str = "configs/research_lab.yaml") -> dict:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        sys.exit(1)

    with open(path) as f:
        config = yaml.safe_load(f)

    return config


def print_config_summary(config: dict, args: argparse.Namespace) -> None:
    """Print configuration summary."""
    table = Table(title="Configuration", show_header=False, box=None)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="yellow")

    # Override config with command line args
    domain = args.domain or config["task"]["domain"]
    episodes = args.episodes or config["marl"]["episodes"]
    beam_width = args.beam_width or config["marl"]["beam_width"]
    k_samples = args.k_samples or config["marl"]["k_samples"]

    table.add_row("Research Domain", domain.replace("_", " ").title())
    table.add_row("Episodes", str(episodes))
    table.add_row("Beam Width", str(beam_width))
    table.add_row("Samples per Turn (k)", str(k_samples))
    table.add_row("Agent Roles", str(config["agents"]["num_agents"]))
    table.add_row("Max Turns per Episode", str(config["task"]["max_turns"]))

    if args.dry_run:
        table.add_row("Mode", "DRY RUN (fast test)", style="bold red")

    console.print(table)
    console.print()


def create_task(config: dict, args: argparse.Namespace) -> ResearchLabTask:
    """Create research lab task."""
    domain = args.domain or config["task"]["domain"]

    task = ResearchLabTask(
        domain=domain,
        max_turns=config["task"]["max_turns"],
        require_novelty=config["task"]["require_novelty"],
        use_real_data=config["task"]["use_real_data"],
    )

    console.print(f"[green]✓[/green] Created {domain.replace('_', ' ').title()} research task")
    return task


def train_research_lab(
    config: dict, task: ResearchLabTask, args: argparse.Namespace
) -> dict:
    """Train the research lab agents."""
    # Override config with CLI args
    if args.episodes:
        config["marl"]["episodes"] = args.episodes
    if args.beam_width:
        config["marl"]["beam_width"] = args.beam_width
    if args.k_samples:
        config["marl"]["k_samples"] = args.k_samples

    # Dry run mode
    if args.dry_run:
        config["marl"]["episodes"] = 2
        config["marl"]["beam_width"] = 2
        config["marl"]["k_samples"] = 1
        config["task"]["max_turns"] = 10

    # Create trainer
    trainer = MARLTrainer(
        task=task,
        config=config,
        agent_roles=config["agents"]["roles"],
    )

    console.print(f"[green]✓[/green] Initialized MARL trainer")
    console.print()

    # Training progress
    console.print("[bold cyan]Starting Research Lab Training...[/bold cyan]")
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task_progress = progress.add_task(
            "[cyan]Training research agents...",
            total=config["marl"]["episodes"],
        )

        # Custom callback to update progress
        def episode_callback(episode_num: int, episode_results: dict) -> None:
            progress.update(task_progress, advance=1)

            # Print episode summary every N episodes
            if episode_num % 5 == 0 or args.verbose:
                print_episode_summary(episode_num, episode_results)

        # Train with callback
        results = trainer.train(episode_callback=episode_callback)

    return results


def print_episode_summary(episode_num: int, results: dict) -> None:
    """Print summary of an episode."""
    scores = results.get("scores", {})

    summary = f"""
Episode {episode_num}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Topic: {results.get('topic', 'Unknown')}

Research Progress:
  Literature reviewed: {results.get('literature_count', 0)} sources
  Hypotheses generated: {results.get('hypothesis_count', 0)}
  Experiments designed: {results.get('experiment_count', 0)}
  Analyses completed: {results.get('analysis_count', 0)}
  Paper draft: {results.get('paper_length', 0)} characters

Evaluation Scores:
  Scientific Rigor:   {scores.get('scientific_rigor', 0):.1f}/10
  Novelty:            {scores.get('novelty', 0):.1f}/10
  Completeness:       {scores.get('completeness', 0):.1f}/10
  Collaboration:      {scores.get('collaboration', 0):.1f}/10
  Feasibility:        {scores.get('feasibility', 0):.1f}/10
  ─────────────────────────────────────────────────────────
  Total Score:        {scores.get('total', 0):.2f}/10

Selected trajectory: {results.get('selected_trajectory', 'N/A')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    console.print(summary)


def print_final_results(results: dict, args: argparse.Namespace) -> None:
    """Print final training results."""
    console.print()
    console.print("[bold green]Training Complete![/bold green]")
    console.print()

    # Summary statistics
    table = Table(title="Final Results", box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")

    metrics = results.get("summary", {})
    table.add_row("Total Episodes", str(results.get("total_episodes", 0)))
    table.add_row("Average Scientific Rigor", f"{metrics.get('avg_rigor', 0):.2f}/10")
    table.add_row("Average Novelty", f"{metrics.get('avg_novelty', 0):.2f}/10")
    table.add_row("Average Completeness", f"{metrics.get('avg_completeness', 0):.2f}/10")
    table.add_row("Average Collaboration", f"{metrics.get('avg_collaboration', 0):.2f}/10")
    table.add_row("Average Total Score", f"{metrics.get('avg_total', 0):.2f}/10")
    table.add_row("Best Episode Score", f"{metrics.get('best_score', 0):.2f}/10")

    # Check for improvement
    final_10 = metrics.get("final_10_avg", 0)
    first_10 = metrics.get("first_10_avg", 0)
    improvement = final_10 - first_10

    if improvement > 0:
        table.add_row(
            "Improvement (First 10 → Last 10)",
            f"+{improvement:.2f} ⬆ IMPROVING!",
            style="bold green",
        )
    else:
        table.add_row("Improvement", f"{improvement:.2f}", style="yellow")

    console.print(table)
    console.print()

    # Cost information
    if "total_cost" in results:
        cost_panel = Panel(
            f"Total API Cost: ${results['total_cost']:.2f}\n"
            f"Average per Episode: ${results['avg_cost_per_episode']:.2f}",
            title="Budget Summary",
            border_style="cyan",
        )
        console.print(cost_panel)
        console.print()

    # Save location
    output_dir = results.get("output_dir", "runs/research_lab_latest")
    console.print(f"[green]✓[/green] Results saved to: [cyan]{output_dir}[/cyan]")
    console.print()

    # Show best episode if requested
    if args.show_best and "best_episode" in results:
        show_best_episode(results["best_episode"])


def show_best_episode(episode: dict) -> None:
    """Display the best research episode."""
    console.print()
    console.print("[bold cyan]Best Research Episode[/bold cyan]")
    console.print("=" * 70)
    console.print()

    console.print(f"[bold]Topic:[/bold] {episode.get('topic', 'Unknown')}")
    console.print()

    # Show trajectory
    console.print("[bold]Research Process:[/bold]")
    for i, turn in enumerate(episode.get("trajectory", []), 1):
        role = turn.get("role", "unknown").replace("_", " ").title()
        action = turn.get("action", "")

        console.print(f"\n[cyan]Turn {i} | {role}:[/cyan]")
        console.print(action[:500] + ("..." if len(action) > 500 else ""))

    console.print()
    console.print(f"[bold green]Final Score:[/bold green] {episode.get('score', 0):.2f}/10")
    console.print()


def save_research_papers(results: dict, output_dir: Path) -> None:
    """Save generated research papers to files."""
    papers_dir = output_dir / "generated_papers"
    papers_dir.mkdir(parents=True, exist_ok=True)

    for i, episode in enumerate(results.get("episodes", []), 1):
        paper_draft = episode.get("paper_draft", "")
        if paper_draft:
            paper_file = papers_dir / f"paper_episode_{i:03d}.md"
            with open(paper_file, "w") as f:
                f.write(f"# Research Paper - Episode {i}\n\n")
                f.write(f"**Topic:** {episode.get('topic', 'Unknown')}\n\n")
                f.write(f"**Score:** {episode.get('score', 0):.2f}/10\n\n")
                f.write("---\n\n")
                f.write(paper_draft)

    console.print(f"[green]✓[/green] Saved {len(results.get('episodes', []))} research papers")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Autonomous Research Laboratory")

    # Task configuration
    parser.add_argument(
        "--domain",
        type=str,
        choices=["materials_science", "climate", "protein", "physics"],
        help="Research domain",
    )

    # Training parameters
    parser.add_argument("--episodes", type=int, help="Number of training episodes")
    parser.add_argument("--beam-width", type=int, help="Beam search width")
    parser.add_argument("--k-samples", type=int, help="Samples per agent per turn")

    # Modes
    parser.add_argument("--dry-run", action="store_true", help="Quick test run (2 episodes)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--show-best", action="store_true", help="Show best episode at end")

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="configs/research_lab.yaml",
        help="Path to config file",
    )

    args = parser.parse_args()

    # Print header
    print_header()

    # Load configuration
    config = load_config(args.config)

    # Print configuration
    print_config_summary(config, args)

    # Create task
    task = create_task(config, args)

    # Train
    results = train_research_lab(config, task, args)

    # Print results
    print_final_results(results, args)

    # Save research papers
    output_dir = Path(results.get("output_dir", "runs/research_lab_latest"))
    save_research_papers(results, output_dir)

    # Success message
    console.print(
        Panel(
            "[bold green]Research Lab Training Complete![/bold green]\n\n"
            "Your AI research agents are now trained to:\n"
            "• Synthesize scientific literature\n"
            "• Generate novel hypotheses\n"
            "• Design rigorous experiments\n"
            "• Analyze data\n"
            "• Write research papers\n\n"
            f"Results saved to: {output_dir}",
            title="Success",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
