#!/usr/bin/env python3
"""
Solve Custom Research Problem - Interactive CLI

This script allows users to input their own research questions and have
the 5-agent AI research lab solve them autonomously.

Usage:
    python examples/solve_custom_research.py
    python examples/solve_custom_research.py --interactive
    python examples/solve_custom_research.py --from-file research_questions.json
"""

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from orchestry.marl.trainer import MARLTrainer
from orchestry.tasks.research_lab import ResearchLabTask


console = Console()


def print_welcome() -> None:
    """Print welcome message."""
    welcome = """
    ╔═══════════════════════════════════════════════════════════╗
    ║       AUTONOMOUS RESEARCH LAB - Custom Research           ║
    ║                                                           ║
    ║  Enter your research question and AI agents will:        ║
    ║  1. Review relevant literature                           ║
    ║  2. Generate testable hypotheses                         ║
    ║  3. Design experiments                                   ║
    ║  4. Analyze results                                      ║
    ║  5. Write a complete research paper                      ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(welcome, style="bold cyan")
    console.print()


def get_research_question_interactive() -> dict:
    """Interactively collect research question from user."""
    console.print("[bold cyan]Let's define your research question[/bold cyan]")
    console.print()

    # Get research topic
    topic = Prompt.ask(
        "[yellow]What is your research topic?[/yellow]\n"
        "Example: 'Novel catalysts for ammonia synthesis at low pressure'"
    )

    # Get domain
    domain = Prompt.ask(
        "[yellow]Which domain does this belong to?[/yellow]",
        choices=["materials_science", "climate", "protein", "physics", "other"],
        default="materials_science",
    )

    # Get context
    console.print()
    console.print("[yellow]Provide background context (what's known, what's the problem):[/yellow]")
    console.print("(Press Enter twice when done)")
    context_lines = []
    while True:
        line = input()
        if line == "" and context_lines and context_lines[-1] == "":
            break
        context_lines.append(line)
    context = "\n".join(context_lines).strip()

    # Get objective
    objective = Prompt.ask(
        "\n[yellow]What is the specific research objective?[/yellow]\n"
        "Example: 'Design catalyst with >80% yield at <10 bar pressure'"
    )

    # Get success metrics
    console.print()
    console.print("[yellow]What metrics will define success? (comma-separated)[/yellow]")
    console.print("Example: yield percentage, operating pressure, catalyst stability, cost")
    metrics_input = input()
    success_metrics = [m.strip() for m in metrics_input.split(",")]

    # Optional: Literature context
    use_literature = Prompt.ask(
        "\n[yellow]Do you have specific papers/findings to reference?[/yellow]",
        choices=["yes", "no"],
        default="no",
    )

    key_papers = []
    if use_literature == "yes":
        console.print("[yellow]Enter key papers/findings (one per line, empty line to finish):[/yellow]")
        while True:
            paper = input()
            if paper == "":
                break
            key_papers.append(paper)

    # Build research problem
    research_problem = {
        "topic": topic,
        "domain": domain,
        "context": context,
        "objective": objective,
        "success_metrics": success_metrics,
        "key_papers": key_papers,
    }

    # Confirm
    console.print()
    console.print(Panel.fit(
        f"[bold]Topic:[/bold] {topic}\n"
        f"[bold]Domain:[/bold] {domain}\n"
        f"[bold]Objective:[/bold] {objective}\n"
        f"[bold]Metrics:[/bold] {', '.join(success_metrics)}",
        title="Research Problem Summary",
        border_style="green",
    ))

    confirm = Prompt.ask("\n[yellow]Does this look correct?[/yellow]", choices=["yes", "no"], default="yes")

    if confirm == "no":
        console.print("[red]Let's start over...[/red]")
        return get_research_question_interactive()

    return research_problem


def load_research_from_file(filepath: str) -> list[dict]:
    """Load research problems from JSON file."""
    path = Path(filepath)
    if not path.exists():
        console.print(f"[red]File not found: {filepath}[/red]")
        return []

    with open(path) as f:
        data = json.load(f)

    # Handle both single problem and list of problems
    if isinstance(data, dict):
        return [data]
    return data


def solve_research_problem(research_problem: dict, beam_width: int = 8, k_samples: int = 4) -> dict:
    """
    Solve a research problem using AI agents.

    Args:
        research_problem: Dict with topic, context, objective, success_metrics
        beam_width: Number of research trajectories to explore
        k_samples: Samples per agent per turn

    Returns:
        Results including generated paper
    """
    console.print()
    console.print(f"[bold cyan]Starting research on: {research_problem['topic']}[/bold cyan]")
    console.print()

    # Create task with custom problem
    task = ResearchLabTask(
        domain=research_problem.get("domain", "materials_science"),
        max_turns=20,
        require_novelty=True,
        custom_problems=[research_problem],
    )

    # Create minimal config for single-problem solving
    config = {
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
            "beam_width": beam_width,
            "k_samples": k_samples,
            "episodes": 1,  # Just solve this one problem
            "temperature": 0.8,
        },
        "rewards": {
            "scientific_rigor_weight": 0.25,
            "novelty_weight": 0.25,
            "completeness_weight": 0.20,
            "collaboration_weight": 0.15,
            "feasibility_weight": 0.15,
        },
    }

    # Create trainer
    trainer = MARLTrainer(
        task=task,
        config=config,
        agent_roles=config["agents"]["roles"],
    )

    console.print("[green]✓[/green] Initialized AI research team (5 agents)")
    console.print(f"[green]✓[/green] Exploring {beam_width} research trajectories")
    console.print()

    # Show progress
    console.print("[bold]Research in progress...[/bold]")
    console.print("This may take 5-10 minutes depending on problem complexity")
    console.print()

    with console.status("[bold cyan]AI agents working...", spinner="dots"):
        results = trainer.train()

    return results


def display_results(results: dict, research_problem: dict) -> None:
    """Display research results."""
    console.print()
    console.print("[bold green]Research Complete![/bold green]")
    console.print("=" * 70)
    console.print()

    # Get episode results (only 1 episode for custom research)
    episode = results.get("episodes", [{}])[0]
    scores = episode.get("scores", {})

    # Show summary
    console.print(Panel.fit(
        f"[bold]Topic:[/bold] {research_problem['topic']}\n\n"
        f"[bold]Scores:[/bold]\n"
        f"  Scientific Rigor:   {scores.get('scientific_rigor', 0):.1f}/10\n"
        f"  Novelty:            {scores.get('novelty', 0):.1f}/10\n"
        f"  Completeness:       {scores.get('completeness', 0):.1f}/10\n"
        f"  Collaboration:      {scores.get('collaboration', 0):.1f}/10\n"
        f"  Feasibility:        {scores.get('feasibility', 0):.1f}/10\n"
        f"  ───────────────────────────────\n"
        f"  [bold]Total Score:        {scores.get('total', 0):.2f}/10[/bold]",
        title="Research Results",
        border_style="green",
    ))
    console.print()

    # Show research progress
    console.print("[bold cyan]Research Progress:[/bold cyan]")
    console.print(f"  • Literature sources reviewed: {episode.get('literature_count', 0)}")
    console.print(f"  • Hypotheses generated: {episode.get('hypothesis_count', 0)}")
    console.print(f"  • Experiments designed: {episode.get('experiment_count', 0)}")
    console.print(f"  • Data analyses completed: {episode.get('analysis_count', 0)}")
    console.print(f"  • Paper draft length: {episode.get('paper_length', 0)} characters")
    console.print()

    # Show generated paper
    paper_draft = episode.get("paper_draft", "")
    if paper_draft:
        console.print("[bold cyan]Generated Research Paper:[/bold cyan]")
        console.print("─" * 70)
        console.print(paper_draft)
        console.print("─" * 70)
        console.print()

        # Offer to save
        save = Prompt.ask("[yellow]Save research paper to file?[/yellow]", choices=["yes", "no"], default="yes")
        if save == "yes":
            filename = f"research_paper_{research_problem['topic'][:30].replace(' ', '_')}.md"
            with open(filename, "w") as f:
                f.write(f"# {research_problem['topic']}\n\n")
                f.write(f"**Objective:** {research_problem['objective']}\n\n")
                f.write(f"**Metrics:** {', '.join(research_problem['success_metrics'])}\n\n")
                f.write("---\n\n")
                f.write(paper_draft)

            console.print(f"[green]✓[/green] Saved to: [cyan]{filename}[/cyan]")

    # Show trajectory highlights
    show_trajectory = Prompt.ask("\n[yellow]Show detailed agent interactions?[/yellow]", choices=["yes", "no"], default="no")
    if show_trajectory == "yes":
        trajectory = episode.get("trajectory", [])
        for i, turn in enumerate(trajectory, 1):
            role = turn.get("role", "unknown").replace("_", " ").title()
            action = turn.get("action", "")
            console.print(f"\n[cyan]Turn {i} | {role}:[/cyan]")
            console.print(action[:500] + ("..." if len(action) > 500 else ""))


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Solve custom research problems with AI")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode (default)")
    parser.add_argument("--from-file", type=str, help="Load research problems from JSON file")
    parser.add_argument("--beam-width", type=int, default=8, help="Beam search width (default: 8)")
    parser.add_argument("--k-samples", type=int, default=4, help="Samples per turn (default: 4)")

    args = parser.parse_args()

    # Print welcome
    print_welcome()

    # Get research problem(s)
    if args.from_file:
        research_problems = load_research_from_file(args.from_file)
        if not research_problems:
            return
    else:
        # Interactive mode (default)
        research_problem = get_research_question_interactive()
        research_problems = [research_problem]

    # Solve each problem
    for i, problem in enumerate(research_problems, 1):
        if len(research_problems) > 1:
            console.print(f"\n[bold]Problem {i}/{len(research_problems)}[/bold]")

        results = solve_research_problem(
            problem,
            beam_width=args.beam_width,
            k_samples=args.k_samples,
        )

        display_results(results, problem)

        # Cost info
        if "total_cost" in results:
            console.print(f"\n[cyan]API Cost: ${results['total_cost']:.2f}[/cyan]")

        # Continue?
        if i < len(research_problems):
            continue_research = Prompt.ask("\n[yellow]Continue to next problem?[/yellow]", choices=["yes", "no"], default="yes")
            if continue_research == "no":
                break

    console.print()
    console.print("[bold green]All research complete![/bold green]")


if __name__ == "__main__":
    main()
