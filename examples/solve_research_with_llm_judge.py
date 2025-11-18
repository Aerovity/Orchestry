#!/usr/bin/env python3
"""
Custom Research with LLM-as-Judge Rewards

This version uses Claude as a judge to evaluate research quality
instead of heuristic keyword matching. More accurate but costs ~$0.05 per eval.

Usage:
    python examples/solve_research_with_llm_judge.py
"""

import argparse
import os

from rich.console import Console
from rich.prompt import Prompt

from orchestry.marl.rewards.research_reward import HybridRewardModel
from orchestry.tasks.research_lab import ResearchLabTask


console = Console()


def solve_with_llm_judge():
    """Solve research problem with LLM judge for higher quality."""

    # Get API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]Error: ANTHROPIC_API_KEY not set[/red]")
        console.print("Set it in .env file or environment variable")
        return

    console.print("[bold cyan]Research Lab with LLM Judge[/bold cyan]")
    console.print("Using Claude to evaluate research quality")
    console.print()

    # Get research question (simplified for demo)
    topic = Prompt.ask(
        "[yellow]What's your research topic?[/yellow]\n",
        default="Low-cost catalysts for hydrogen evolution from seawater"
    )

    objective = Prompt.ask(
        "[yellow]What's the objective?[/yellow]\n",
        default="Catalyst with >80% efficiency, <$10/kg cost, >1000h seawater stability"
    )

    context = Prompt.ask(
        "[yellow]Brief background (or press Enter to use default)[/yellow]\n",
        default="Electrolysis uses expensive Pt catalysts. Seawater chlorides corrode electrodes. Need cheap, stable alternative."
    )

    # Create research problem
    custom_problem = {
        'topic': topic,
        'context': context,
        'objective': objective,
        'success_metrics': ['efficiency', 'cost', 'stability', 'overpotential'],
        'key_papers': []
    }

    console.print()
    console.print("[green]Creating research task with LLM judge...[/green]")

    # Create task
    task = ResearchLabTask(custom_problems=[custom_problem])

    # Create LLM judge reward model
    reward_model = HybridRewardModel(
        api_key=api_key,
        use_llm_for_final=True  # Use Claude to judge final result
    )

    console.print("[green]✓[/green] Initialized LLM judge (Claude Sonnet)")
    console.print()

    # Run research (simplified - in real version this would be in trainer)
    console.print("[bold]Starting research...[/bold]")
    console.print("This will use Claude to judge the research quality")
    console.print()

    # Reset task
    observation = task.reset()

    # Simulate agent collaboration (in real version, agents would actually run)
    console.print("[cyan]Agents working...[/cyan]")
    console.print("• Literature Synthesizer: Reviewing papers...")
    console.print("• Hypothesis Generator: Creating hypotheses...")
    console.print("• Experimental Designer: Designing experiments...")
    console.print("• Data Analyst: Analyzing results...")
    console.print("• Paper Writer: Writing paper...")
    console.print()

    # For demo, show comparison
    console.print("[bold yellow]Reward Comparison:[/bold yellow]")
    console.print()

    # Heuristic score (built-in)
    heuristic_scores = task.evaluate(trajectory=[])
    console.print("[cyan]Heuristic Rewards (keyword matching):[/cyan]")
    console.print(f"  Scientific Rigor:  {heuristic_scores['scientific_rigor']:.1f}/10")
    console.print(f"  Novelty:           {heuristic_scores['novelty']:.1f}/10")
    console.print(f"  Completeness:      {heuristic_scores['completeness']:.1f}/10")
    console.print(f"  Collaboration:     {heuristic_scores['collaboration']:.1f}/10")
    console.print(f"  Feasibility:       {heuristic_scores['feasibility']:.1f}/10")
    console.print(f"  [bold]Total:             {heuristic_scores['total']:.2f}/10[/bold]")
    console.print()

    # LLM score (more accurate)
    console.print("[cyan]LLM Judge (Claude evaluating research):[/cyan]")
    console.print("[dim]Calling Claude API for evaluation...[/dim]")

    try:
        llm_scores = reward_model.llm_judge.evaluate_research(
            topic=topic,
            objective=objective,
            trajectory=[],  # Would have full trajectory in real version
            literature_reviewed=task.literature_reviewed,
            hypotheses=task.hypotheses_generated,
            experiments=task.experiments_designed,
            analyses=task.analyses_completed,
            paper_draft=task.paper_draft,
        )

        console.print(f"  Scientific Rigor:  {llm_scores['scientific_rigor']:.1f}/10")
        console.print(f"  Novelty:           {llm_scores['novelty']:.1f}/10")
        console.print(f"  Completeness:      {llm_scores['completeness']:.1f}/10")
        console.print(f"  Collaboration:     {llm_scores['collaboration']:.1f}/10")
        console.print(f"  Feasibility:       {llm_scores['feasibility']:.1f}/10")
        console.print(f"  [bold]Total:             {llm_scores['total']:.2f}/10[/bold]")
        console.print()

        # Show difference
        diff = abs(llm_scores['total'] - heuristic_scores['total'])
        console.print(f"[yellow]Difference: {diff:.1f} points[/yellow]")
        console.print()

        if llm_scores['total'] > heuristic_scores['total']:
            console.print("[green]✓ LLM judge scored higher (caught quality heuristics missed)[/green]")
        else:
            console.print("[yellow]! LLM judge scored lower (caught issues heuristics missed)[/yellow]")

    except Exception as e:
        console.print(f"[red]Error calling LLM judge: {e}[/red]")
        console.print("[yellow]Falling back to heuristic rewards[/yellow]")

    console.print()
    console.print("[bold]Cost Breakdown:[/bold]")
    console.print("  Heuristic eval: $0.00 (free)")
    console.print("  LLM judge eval: ~$0.05 (one-time)")
    console.print("  [dim]In hybrid mode, LLM is only used for final trajectory[/dim]")


if __name__ == "__main__":
    solve_with_llm_judge()
