#!/usr/bin/env python3
"""
Orchestry Main Entry Point - Unified Research Lab Interface

This is the main script for running research lab with:
- Interactive question input
- Multi-episode training (e.g., 50 episodes)
- LLM-as-judge reward system
- Full progress visualization
- Structured output (runs/research_lab_*)

Usage:
    # Interactive mode with LLM judge
    python main.py --mode research --episodes 50 --question "cats" --use-llm-judge

    # With custom parameters
    python main.py --mode research --episodes 20 --question "battery materials" --use-llm-judge --verbose

    # Dry run test
    python main.py --mode research --episodes 2 --question "test" --dry-run
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from orchestry.marl.rewards.research_reward import ResearchRewardModel
from orchestry.marl.trainer import MARLTrainer
from orchestry.tasks.research_lab import ResearchLabTask


console = Console()


def print_header() -> None:
    """Print application header."""
    header = """
    ╔═══════════════════════════════════════════════════════════╗
    ║            ORCHESTRY RESEARCH LABORATORY v2.0             ║
    ║       Multi-Agent AI Research with LLM Judge              ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(header, style="bold cyan")


def get_research_details_interactive(question: str) -> dict:
    """
    Interactively get detailed research parameters.

    Args:
        question: Main research question/topic

    Returns:
        Research problem dictionary
    """
    console.print()
    console.print(f"[bold cyan]Research Topic: {question}[/bold cyan]")
    console.print()
    console.print("[yellow]Let's define the research details...[/yellow]")
    console.print()

    # Get objective
    console.print("[bold]What is the specific research objective?[/bold]")
    console.print("[dim]Example: 'Identify optimal protein levels and dietary patterns for feline health'[/dim]")
    objective = input("> ").strip()
    if not objective:
        objective = f"Conduct comprehensive research on {question}"

    console.print()

    # Get background context
    console.print("[bold]Provide background context (press Enter twice when done):[/bold]")
    console.print("[dim]Example: 'Cats are obligate carnivores requiring high protein...'[/dim]")
    context_lines = []
    while True:
        line = input()
        if line == "" and (not context_lines or context_lines[-1] == ""):
            break
        context_lines.append(line)

    # Remove trailing empty line
    if context_lines and context_lines[-1] == "":
        context_lines.pop()

    context = "\n".join(context_lines).strip()
    if not context:
        context = f"Research on {question} to understand key mechanisms and applications."

    console.print()

    # Get success metrics
    console.print("[bold]What metrics will define research success?[/bold]")
    console.print("[dim]Example: 'protein content, digestibility, palatability, cost'[/dim]")
    metrics_input = input("> ").strip()
    if metrics_input:
        success_metrics = [m.strip() for m in metrics_input.split(",")]
    else:
        success_metrics = ["scientific rigor", "novelty", "completeness"]

    console.print()

    # Optional: key papers/references
    use_papers = Prompt.ask(
        "[yellow]Do you have specific papers to reference?[/yellow]",
        choices=["yes", "no"],
        default="no"
    )

    key_papers = []
    if use_papers == "yes":
        console.print("[yellow]Enter papers/findings (one per line, empty line to finish):[/yellow]")
        while True:
            paper = input()
            if paper == "":
                break
            key_papers.append(paper)

    # Determine domain
    domain = "other"
    question_lower = question.lower()
    if any(word in question_lower for word in ["battery", "material", "catalyst", "electrolyte"]):
        domain = "materials_science"
    elif any(word in question_lower for word in ["climate", "carbon", "co2", "emission"]):
        domain = "climate"
    elif any(word in question_lower for word in ["protein", "enzyme", "drug", "antibody"]):
        domain = "protein"
    elif any(word in question_lower for word in ["quantum", "superconductor", "physics"]):
        domain = "physics"

    # Build research problem
    research_problem = {
        "topic": question,
        "domain": domain,
        "context": context,
        "objective": objective,
        "success_metrics": success_metrics,
        "key_papers": key_papers,
    }

    # Show summary
    console.print()
    console.print(Panel.fit(
        f"[bold]Topic:[/bold] {question}\n"
        f"[bold]Domain:[/bold] {domain}\n"
        f"[bold]Objective:[/bold] {objective}\n"
        f"[bold]Metrics:[/bold] {', '.join(success_metrics)}\n"
        f"[bold]Key Papers:[/bold] {len(key_papers)} references",
        title="Research Configuration",
        border_style="green",
    ))

    console.print()
    confirm = Prompt.ask(
        "[yellow]Start training with this configuration?[/yellow]",
        choices=["yes", "no"],
        default="yes"
    )

    if confirm == "no":
        console.print("[red]Exiting...[/red]")
        sys.exit(0)

    return research_problem


def load_config(config_path: str = "configs/research_lab.yaml") -> dict:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        console.print(f"[yellow]Warning: Config file not found: {config_path}[/yellow]")
        console.print("[yellow]Using default configuration[/yellow]")
        return get_default_config()

    with open(path) as f:
        config = yaml.safe_load(f)

    return config


def get_default_config() -> dict:
    """Get default configuration if YAML not found."""
    return {
        "agents": {
            "num_agents": 5,
            "roles": [
                "literature_synthesizer",
                "hypothesis_generator",
                "experimental_designer",
                "data_analyst",
                "paper_writer",
            ],
            "model": "claude-3-5-sonnet-20241022",
            "temperature": 0.7,
            "max_tokens": 1024,
        },
        "marl": {
            "algorithm": "api_grpo",
            "beam_width": 8,
            "k_samples": 4,
            "episodes": 50,
            "temperature": 0.8,
        },
        "rewards": {
            "scientific_rigor_weight": 0.25,
            "novelty_weight": 0.25,
            "completeness_weight": 0.20,
            "collaboration_weight": 0.15,
            "feasibility_weight": 0.15,
        },
        "task": {
            "max_turns": 20,
            "require_novelty": True,
            "use_real_data": False,
        },
    }


def print_config_summary(config: dict, args: argparse.Namespace, research_problem: dict) -> None:
    """Print configuration summary before training."""
    table = Table(title="Training Configuration", show_header=False, box=None)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="yellow")

    table.add_row("Research Topic", research_problem["topic"])
    table.add_row("Domain", research_problem["domain"].replace("_", " ").title())
    table.add_row("Episodes", str(args.episodes or config["marl"]["episodes"]))
    table.add_row("Beam Width", str(args.beam_width or config["marl"]["beam_width"]))
    table.add_row("Samples per Turn (k)", str(args.k_samples or config["marl"]["k_samples"]))
    table.add_row("Agent Roles", str(config["agents"]["num_agents"]))
    table.add_row("Max Turns per Episode", str(config["task"]["max_turns"]))

    if args.use_llm_judge:
        table.add_row("Reward System", "LLM-as-Judge (Claude)", style="bold green")
    else:
        table.add_row("Reward System", "Heuristic (keyword-based)", style="yellow")

    if args.dry_run:
        table.add_row("Mode", "DRY RUN (fast test)", style="bold red")

    console.print(table)
    console.print()


def create_research_task(research_problem: dict, config: dict) -> ResearchLabTask:
    """Create research lab task with custom problem."""
    task = ResearchLabTask(
        domain=research_problem["domain"],
        max_turns=config["task"]["max_turns"],
        require_novelty=config["task"]["require_novelty"],
        use_real_data=config["task"]["use_real_data"],
        custom_problems=[research_problem],
    )

    console.print(f"[green]✓[/green] Created research task: {research_problem['topic']}")
    return task


def create_llm_judge(api_key: str) -> ResearchRewardModel:
    """Create LLM-as-judge reward model."""
    try:
        reward_model = ResearchRewardModel(api_key=api_key)
        console.print("[green]✓[/green] Initialized LLM Judge (Claude Sonnet)")
        return reward_model
    except Exception as e:
        console.print(f"[red]Error initializing LLM judge: {e}[/red]")
        console.print("[yellow]Falling back to heuristic rewards[/yellow]")
        return None


def train_research_lab(
    config: dict,
    task: ResearchLabTask,
    args: argparse.Namespace,
    llm_judge: ResearchRewardModel | None = None,
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

    # Add LLM judge to config if enabled
    if llm_judge:
        config["rewards"]["llm_judge"] = llm_judge
        config["rewards"]["use_llm_judge"] = True

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

        # Episode callback
        def episode_callback(episode_num: int, episode_results: dict) -> None:
            progress.update(task_progress, advance=1)

            # Print episode summary
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

    # Show structure
    console.print("[bold]Output Structure:[/bold]")
    console.print(f"  {output_dir}/")
    console.print(f"  ├── episodes.json              # All research episodes")
    console.print(f"  ├── rewards.csv               # Training metrics")
    console.print(f"  ├── learned_behaviors.json    # Successful research patterns")
    console.print(f"  ├── summary.json              # Final statistics")
    console.print(f"  ├── generated_papers/         # Research paper drafts")
    console.print(f"  │   ├── paper_episode_001.md")
    console.print(f"  │   └── ...")
    console.print(f"  └── checkpoints/              # Training checkpoints")
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
    parser = argparse.ArgumentParser(
        description="Orchestry - Multi-Agent AI Research Laboratory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive research with LLM judge
  python main.py --mode research --episodes 50 --question "cats" --use-llm-judge

  # Quick test run
  python main.py --mode research --episodes 2 --question "test" --dry-run

  # Custom parameters
  python main.py --mode research --episodes 20 --question "battery materials" \\
                 --use-llm-judge --beam-width 16 --k-samples 8 --verbose
        """
    )

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["research"],
        default="research",
        help="Execution mode (currently only 'research' supported)"
    )

    # Research question
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Research question or topic"
    )

    # Training parameters
    parser.add_argument(
        "--episodes",
        type=int,
        help="Number of training episodes (default: 50)"
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        help="Beam search width (default: 8)"
    )
    parser.add_argument(
        "--k-samples",
        type=int,
        help="Samples per agent per turn (default: 4)"
    )

    # LLM Judge
    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        help="Use Claude as judge for rewards (more accurate, ~$0.05/eval)"
    )

    # Modes
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Quick test run (2 episodes)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output (show all episodes)"
    )
    parser.add_argument(
        "--show-best",
        action="store_true",
        help="Show best episode at end"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=True,
        help="Interactive mode (ask detailed questions)"
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="configs/research_lab.yaml",
        help="Path to config file"
    )

    args = parser.parse_args()

    # Print header
    print_header()

    # Get research details interactively
    if args.interactive:
        research_problem = get_research_details_interactive(args.question)
    else:
        # Non-interactive: use minimal config
        research_problem = {
            "topic": args.question,
            "domain": "other",
            "context": f"Research on {args.question}",
            "objective": f"Comprehensive analysis of {args.question}",
            "success_metrics": ["scientific rigor", "novelty", "completeness"],
            "key_papers": [],
        }

    # Load configuration
    config = load_config(args.config)

    # Print configuration summary
    console.print()
    print_config_summary(config, args, research_problem)

    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]Error: ANTHROPIC_API_KEY not set[/red]")
        console.print("Set it in .env file or environment variable")
        sys.exit(1)

    # Create LLM judge if requested
    llm_judge = None
    if args.use_llm_judge:
        llm_judge = create_llm_judge(api_key)

    # Create task
    task = create_research_task(research_problem, config)

    # Train
    results = train_research_lab(config, task, args, llm_judge)

    # Print results
    print_final_results(results, args)

    # Save research papers
    output_dir = Path(results.get("output_dir", "runs/research_lab_latest"))
    save_research_papers(results, output_dir)

    # Success message
    console.print(
        Panel(
            "[bold green]Research Lab Training Complete![/bold green]\n\n"
            "Your AI research agents have been trained to:\n"
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
