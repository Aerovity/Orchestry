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
import signal
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rich.console import Console

# Load environment variables from .env file
load_dotenv()
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from orchestry.marl.rewards.research_reward import ResearchRewardModel
from orchestry.marl.trainer import MARLTrainer
from orchestry.tasks.research_lab import ResearchLabTask


console = Console()

# Global trainer reference for signal handler
_global_trainer = None


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully - save papers before exit."""
    global _global_trainer
    console.print("\n\n[yellow]⚠ Interrupted by user (Ctrl+C)[/yellow]")

    if _global_trainer and hasattr(_global_trainer, 'episodes') and _global_trainer.episodes:
        console.print("[cyan]Saving research papers before exit...[/cyan]")
        try:
            _global_trainer._save_final_results()
            console.print(f"[green]✓ Saved {len(_global_trainer.episodes)} papers to {_global_trainer.save_dir}/papers/[/green]")
        except Exception as e:
            console.print(f"[red]✗ Error saving papers: {e}[/red]")
    else:
        console.print("[yellow]No episodes to save[/yellow]")

    console.print("[dim]Exiting...[/dim]")
    sys.exit(0)


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

    with open(path, encoding='utf-8') as f:
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
        provider = config.get("agents", {}).get("provider", "claude")
        provider_name = "Claude" if provider == "claude" else "Gemini"
        table.add_row("Reward System", f"LLM-as-Judge ({provider_name})", style="bold green")
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


def create_llm_judge(api_key: str, provider: str = "claude", gemini_api_key: str | None = None) -> ResearchRewardModel:
    """Create LLM-as-judge reward model."""
    try:
        # Set model based on provider
        if provider == "gemini":
            model = "gemini-2.0-flash"  # STABLE model with high RPM - fast and reliable
        else:
            model = "claude-3-5-sonnet-20241022"

        reward_model = ResearchRewardModel(
            api_key=api_key,
            model=model,
            provider=provider,
            gemini_api_key=gemini_api_key
        )
        provider_name = "Claude Sonnet" if provider == "claude" else "Gemini"
        console.print(f"[green]✓[/green] Initialized LLM Judge ({provider_name})")
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
    api_key: str = None,
    provider: str = "claude",
    gemini_api_key: str | None = None,
) -> dict:
    """Train the research lab agents."""
    global _global_trainer

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

    # Create agents from roles with proper system prompts from task
    from orchestry.marl.api_grpo import Agent
    agents = []

    # Get a sample observation to build base prompts
    sample_obs = task.reset()

    for i, role in enumerate(config["agents"]["roles"]):
        # Get the full agent prompt from the task (includes role-specific instructions)
        full_prompt = task.get_agent_prompt(agent_id=i, agent_role=role, observation=sample_obs)

        agent = Agent(
            agent_id=i,
            role=role,
            goal=f"Expert {role.replace('_', ' ')} for research collaboration",
            system_prompt=full_prompt,
            learned_behaviors=[]
        )
        agents.append(agent)

    # Create trainer
    trainer = MARLTrainer(
        task=task,
        agents=agents,
        api_key=api_key,
        config=config,
        provider=provider,
        gemini_api_key=gemini_api_key
    )

    # Set global trainer for signal handler
    _global_trainer = trainer

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

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

        # Train without callback (the trainer has its own progress tracking)
        # Just run training with appropriate parameters
        results = trainer.train(
            num_episodes=config["marl"]["episodes"],
            verbose=args.verbose if hasattr(args, 'verbose') else False,
            save_frequency=5
        )

        # Update progress bar to complete
        progress.update(task_progress, completed=config["marl"]["episodes"])

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
    import logging
    logger = logging.getLogger(__name__)

    papers_dir = output_dir / "generated_papers"
    papers_dir.mkdir(parents=True, exist_ok=True)

    episodes_data = results.get("episodes", [])
    if not episodes_data:
        console.print("[yellow]⚠ No episodes found to save papers[/yellow]")
        return

    papers_saved = 0
    for i, episode in enumerate(episodes_data, 1):
        # Extract all research content from trajectory turns
        paper_draft = ""
        literature = ""
        hypotheses = ""
        experiments = ""
        analysis = ""

        turns = episode.get("turns", [])
        logger.info(f"Episode {i}: Processing {len(turns)} turns")

        for turn in turns:
            agent_role = turn.get("agent_role", "")
            action = turn.get("action", "")

            if agent_role == "literature_synthesizer":
                literature += action + "\n\n"
            elif agent_role == "hypothesis_generator":
                hypotheses += action + "\n\n"
            elif agent_role == "experimental_designer":
                experiments += action + "\n\n"
            elif agent_role == "data_analyst":
                analysis += action + "\n\n"
            elif agent_role == "paper_writer":
                paper_draft += action + "\n\n"

        # Check if we have any content to save
        has_content = any([
            paper_draft.strip(),
            literature.strip(),
            hypotheses.strip(),
            experiments.strip(),
            analysis.strip()
        ])

        if has_content:
            paper_file = papers_dir / f"paper_episode_{i:03d}.md"
            logger.info(f"Saving paper to {paper_file}")

            with open(paper_file, "w", encoding='utf-8') as f:
                f.write(f"# Research Paper - Episode {i}\n\n")
                f.write(f"**Score:** {episode.get('total_reward', 0):.2f}/10\n\n")

                # Write reward components
                comps = episode.get("reward_components", {})
                if comps:
                    f.write("**Reward Components:**\n")
                    for key, val in comps.items():
                        if key != 'total':
                            f.write(f"- {key.replace('_', ' ').title()}: {val:.2f}/10\n")
                    f.write("\n")

                f.write("---\n\n")

                # Write full research process
                if literature.strip():
                    f.write("## Literature Review\n\n")
                    f.write(literature)
                    f.write("\n---\n\n")

                if hypotheses.strip():
                    f.write("## Hypotheses\n\n")
                    f.write(hypotheses)
                    f.write("\n---\n\n")

                if experiments.strip():
                    f.write("## Experimental Design\n\n")
                    f.write(experiments)
                    f.write("\n---\n\n")

                if analysis.strip():
                    f.write("## Data Analysis\n\n")
                    f.write(analysis)
                    f.write("\n---\n\n")

                if paper_draft.strip():
                    f.write("## Paper Draft\n\n")
                    f.write(paper_draft)

            papers_saved += 1
            logger.info(f"Successfully saved paper {i}")
        else:
            logger.warning(f"Episode {i}: No content to save (turns: {len(turns)})")

    console.print(f"[green]✓[/green] Saved {papers_saved} research papers to {papers_dir}")
    if papers_saved == 0:
        console.print("[yellow]⚠ Warning: No papers had content to save. Check agent outputs.[/yellow]")


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

    # Check for API keys based on provider
    provider = config.get("agents", {}).get("provider", "claude")

    if provider == "claude":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            console.print("[red]Error: ANTHROPIC_API_KEY not set[/red]")
            console.print("Set it in .env file or environment variable")
            sys.exit(1)
        gemini_api_key = None
    elif provider == "gemini":
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            console.print("[red]Error: GEMINI_API_KEY not set[/red]")
            console.print("Set it in .env file or environment variable")
            sys.exit(1)
        api_key = gemini_api_key
    else:
        console.print(f"[red]Error: Unknown provider: {provider}[/red]")
        console.print("Set provider to 'claude' or 'gemini' in config file")
        sys.exit(1)

    # Create LLM judge if requested
    llm_judge = None
    if args.use_llm_judge:
        llm_judge = create_llm_judge(api_key, provider, gemini_api_key)

    # Create task
    task = create_research_task(research_problem, config)

    # Train
    results = train_research_lab(config, task, args, llm_judge, api_key, provider, gemini_api_key)

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
