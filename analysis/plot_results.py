"""
Plotting and visualization for MAGRPO results.

Generates figures matching the paper:
- Figure 2: Learning curves (cooperation, test pass rate, rewards)
- Scheme distribution evolution
- Baseline comparison
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(
    metrics_file: str,
    output_file: str = "results/learning_curves.png",
    window: int = 20,
) -> None:
    """
    Plot learning curves over training.

    Args:
        metrics_file: Path to metrics JSON file
        output_file: Where to save the plot
        window: Smoothing window size

    """
    # Load metrics
    with Path(metrics_file).open() as f:
        metrics = json.load(f)

    episodes = [m["episode"] for m in metrics]
    rewards = [m["mean_reward"] for m in metrics]
    structure = [m.get("structure_rate", 0) for m in metrics]
    syntax = [m.get("syntax_rate", 0) for m in metrics]
    tests = [m.get("test_pass_rate", 0) for m in metrics]
    cooperation = [m.get("cooperation_rate", 0) for m in metrics]

    # Smooth curves
    def smooth(
        values: list[float],
        window: int,
    ) -> np.ndarray[tuple[int], np.dtype[np.floating[Any]]]:
        if len(values) < window:
            return np.array(values)
        return np.convolve(values, np.ones(window) / window, mode="valid")

    smooth_rewards = smooth(rewards, window)
    smooth_structure = smooth(structure, window)
    smooth_syntax = smooth(syntax, window)
    smooth_tests = smooth(tests, window)
    smooth_cooperation = smooth(cooperation, window)

    # Create figure with 2x2 subplots
    _, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Total reward
    axes[0, 0].plot(episodes, rewards, alpha=0.3, color="blue")
    axes[0, 0].plot(
        episodes[window - 1 :],
        smooth_rewards,
        linewidth=2,
        color="blue",
        label="Total Reward",
    )
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Mean Reward")
    axes[0, 0].set_title("Total Return Over Time")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Plot 2: Individual metrics
    axes[0, 1].plot(
        episodes[window - 1 :],
        smooth_structure,
        label="Structure",
        linestyle="--",
        color="gray",
    )
    axes[0, 1].plot(
        episodes[window - 1 :],
        smooth_syntax,
        label="Syntax",
        linestyle="--",
        color="green",
    )
    axes[0, 1].plot(
        episodes[window - 1 :],
        smooth_tests,
        label="Tests",
        linestyle="--",
        color="red",
    )
    axes[0, 1].plot(
        episodes[window - 1 :],
        smooth_cooperation,
        label="Cooperation",
        linestyle="--",
        color="orange",
    )
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Rate")
    axes[0, 1].set_title("Reward Components Over Time")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Plot 3: Cooperation quality (larger)
    axes[1, 0].plot(episodes, cooperation, alpha=0.3, color="orange")
    axes[1, 0].plot(episodes[window - 1 :], smooth_cooperation, linewidth=2, color="orange")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Cooperation Rate")
    axes[1, 0].set_title("Cooperation Quality Over Time")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0.7, color="r", linestyle=":", label="Target (70%)")
    axes[1, 0].legend()

    # Plot 4: Test pass rate (larger)
    axes[1, 1].plot(episodes, tests, alpha=0.3, color="red")
    axes[1, 1].plot(episodes[window - 1 :], smooth_tests, linewidth=2, color="red")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Test Pass Rate")
    axes[1, 1].set_title("Test Pass Rate Over Time")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Learning curves saved to {output_file}")

    plt.show()


def plot_baseline_comparison(
    magrpo_results: dict[str, float],
    baseline_results: dict[str, dict[str, float]],
    output_file: str = "results/baseline_comparison.png",
) -> None:
    """
    Plot comparison with baselines.

    Args:
        magrpo_results: MAGRPO results dict
        baseline_results: Dict mapping baseline name to results
        output_file: Where to save the plot

    """
    methods = [
        "Fixed\nModel",
        "Naive\nConcat",
        "Sequential\nPipeline",
        "Discussion",
        "MAGRPO\n(Ours)",
    ]
    rewards = [
        baseline_results.get("fixed", {}).get("mean_reward", 0),
        baseline_results.get("naive", {}).get("mean_reward", 0),
        baseline_results.get("sequential", {}).get("mean_reward", 0),
        baseline_results.get("discussion", {}).get("mean_reward", 0),
        magrpo_results.get("mean_reward", 0),
    ]

    cooperation = [
        baseline_results.get("fixed", {}).get("cooperation_rate", 0),
        baseline_results.get("naive", {}).get("cooperation_rate", 0),
        baseline_results.get("sequential", {}).get("cooperation_rate", 0),
        baseline_results.get("discussion", {}).get("cooperation_rate", 0),
        magrpo_results.get("cooperation_rate", 0),
    ]

    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Mean reward comparison
    bars1 = axes[0].bar(
        methods,
        rewards,
        color=["gray", "lightblue", "lightgreen", "lightyellow", "orange"],
    )
    bars1[-1].set_color("red")  # Highlight MAGRPO
    axes[0].set_ylabel("Mean Reward")
    axes[0].set_title("Mean Reward Comparison")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Cooperation rate comparison
    bars2 = axes[1].bar(
        methods,
        cooperation,
        color=["gray", "lightblue", "lightgreen", "lightyellow", "orange"],
    )
    bars2[-1].set_color("red")  # Highlight MAGRPO
    axes[1].set_ylabel("Cooperation Rate")
    axes[1].set_title("Cooperation Rate Comparison")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Baseline comparison saved to {output_file}")

    plt.show()


def plot_scheme_distribution(
    scheme_counts: dict[str, int],
    output_file: str = "results/scheme_distribution.png",
) -> None:
    """
    Plot cooperation scheme distribution.

    Args:
        scheme_counts: Dict mapping scheme name to count
        output_file: Where to save the plot

    """
    schemes = list(scheme_counts.keys())
    counts = list(scheme_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(schemes, counts, color=["steelblue", "coral", "lightgreen", "gold", "gray"])
    plt.xlabel("Cooperation Scheme")
    plt.ylabel("Count")
    plt.title("Cooperation Scheme Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Scheme distribution saved to {output_file}")

    plt.show()


def create_summary_report(
    metrics_file: str,
    output_file: str = "results/summary.txt",
) -> None:
    """
    Create text summary of results.

    Args:
        metrics_file: Path to metrics JSON
        output_file: Where to save summary

    """
    with Path(metrics_file).open() as f:
        metrics = json.load(f)

    # Get final metrics
    final = metrics[-1] if metrics else {}

    # Calculate statistics
    rewards = [m["mean_reward"] for m in metrics]
    cooperation = [m.get("cooperation_rate", 0) for m in metrics]

    initial_reward = rewards[0] if rewards else 0
    final_reward = rewards[-1] if rewards else 0
    improvement = final_reward - initial_reward

    initial_coop = cooperation[0] if cooperation else 0
    final_coop = cooperation[-1] if cooperation else 0
    coop_improvement = final_coop - initial_coop

    summary = f"""
MAGRPO TRAINING SUMMARY
{"=" * 60}

TRAINING CONFIGURATION
- Total Episodes: {len(metrics)}
- Final Episode: {final.get("episode", 0)}

PERFORMANCE METRICS
- Initial Mean Reward: {initial_reward:.3f}
- Final Mean Reward: {final_reward:.3f}
- Improvement: {improvement:.3f} ({improvement / initial_reward * 100:.1f}%)

COOPERATION METRICS
- Initial Cooperation Rate: {initial_coop:.2%}
- Final Cooperation Rate: {final_coop:.2%}
- Improvement: {coop_improvement:.2%}

FINAL BREAKDOWN
- Structure Rate: {final.get("structure_rate", 0):.2%}
- Syntax Rate: {final.get("syntax_rate", 0):.2%}
- Test Pass Rate: {final.get("test_pass_rate", 0):.2%}
- Cooperation Rate: {final.get("cooperation_rate", 0):.2%}

BUDGET
- Total Spent: ${final.get("budget_spent", 0):.2f}

{"=" * 60}
"""

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        f.write(summary)

    print(summary)
    print(f"✓ Summary saved to {output_file}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <metrics_file>")
        sys.exit(1)

    metrics_file = sys.argv[1]

    # Generate all plots
    plot_learning_curves(metrics_file)
    create_summary_report(metrics_file)
