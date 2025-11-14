"""
Utility functions for Orchestry.

Includes plotting, configuration loading, and helper functions.
"""

from typing import Dict, Any, List, Optional
import yaml
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {e}")
        raise


def setup_logging(level: str = "INFO") -> None:
    """
    Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def plot_training_curves(
    metrics: Dict[str, Any],
    save_dir: Path,
    show: bool = False
) -> None:
    """
    Plot training curves and save to file.

    Args:
        metrics: Training metrics dictionary
        save_dir: Directory to save plots
        show: Whether to display plots
    """
    plots_dir = save_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Orchestry Training Metrics", fontsize=16, fontweight="bold")

    episodes = list(range(1, len(metrics["episode_rewards"]) + 1))

    # Plot 1: Total Rewards
    ax1 = axes[0, 0]
    ax1.plot(episodes, metrics["episode_rewards"], marker="o", linewidth=2, markersize=4)
    ax1.axhline(
        y=np.mean(metrics["episode_rewards"]),
        color="r",
        linestyle="--",
        label="Mean"
    )
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Total Reward per Episode")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Reward Components
    ax2 = axes[0, 1]
    ax2.plot(episodes, metrics["story_quality_scores"], marker="s", label="Story Quality", alpha=0.7)
    ax2.plot(episodes, metrics["collaboration_scores"], marker="^", label="Collaboration", alpha=0.7)
    ax2.plot(episodes, metrics["efficiency_scores"], marker="d", label="Efficiency", alpha=0.7)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Score")
    ax2.set_title("Reward Components")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Episode Lengths
    ax3 = axes[1, 0]
    ax3.bar(episodes, metrics["episode_lengths"], alpha=0.6, color="green")
    ax3.axhline(
        y=np.mean(metrics["episode_lengths"]),
        color="r",
        linestyle="--",
        label="Mean Length"
    )
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Number of Turns")
    ax3.set_title("Episode Lengths")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    # Plot 4: Moving Average
    ax4 = axes[1, 1]
    window = min(5, len(metrics["episode_rewards"]))
    if window > 1:
        moving_avg = np.convolve(
            metrics["episode_rewards"],
            np.ones(window) / window,
            mode="valid"
        )
        moving_episodes = list(range(window, len(episodes) + 1))
        ax4.plot(moving_episodes, moving_avg, marker="o", linewidth=2, color="purple")
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Moving Average Reward")
        ax4.set_title(f"Moving Average (window={window})")
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    plot_path = plots_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    logger.info(f"Training curves saved to: {plot_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_collaboration_analysis(
    metrics: Dict[str, Any],
    save_dir: Path
) -> None:
    """
    Plot collaboration-specific analysis.

    Args:
        metrics: Training metrics
        save_dir: Directory to save plots
    """
    plots_dir = save_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    episodes = list(range(1, len(metrics["collaboration_scores"]) + 1))

    # Scatter plot: collaboration vs story quality
    scatter = ax.scatter(
        metrics["collaboration_scores"],
        metrics["story_quality_scores"],
        c=episodes,
        cmap="viridis",
        s=100,
        alpha=0.6
    )

    ax.set_xlabel("Collaboration Score")
    ax.set_ylabel("Story Quality Score")
    ax.set_title("Collaboration vs Story Quality")
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Episode")

    # Save
    plot_path = plots_dir / "collaboration_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    logger.info(f"Collaboration analysis saved to: {plot_path}")
    plt.close()


def format_episode_summary(episode_data: Dict[str, Any]) -> str:
    """
    Format episode data as readable summary.

    Args:
        episode_data: Episode dictionary

    Returns:
        Formatted summary string
    """
    lines = [
        f"Episode {episode_data['episode_id']}",
        "=" * 50,
        f"Theme: {episode_data['metadata'].get('theme', 'Unknown')}",
        f"Turns: {episode_data['num_turns']}",
        "",
        "Rewards:",
        f"  Total: {episode_data['rewards']['total']:.2f}",
        f"  Story Quality: {episode_data['rewards']['story_quality']:.2f}",
        f"  Collaboration: {episode_data['rewards']['collaboration']:.2f}",
        f"  Efficiency: {episode_data['rewards']['efficiency']:.2f}",
        "",
        "Conversation:",
        "-" * 50
    ]

    for turn in episode_data["conversation"]:
        lines.append(f"\n[{turn['agent']}]:")
        lines.append(turn['content'])
        lines.append("-" * 50)

    return "\n".join(lines)


def calculate_improvement_rate(rewards: List[float], window: int = 5) -> float:
    """
    Calculate rate of improvement in rewards.

    Args:
        rewards: List of episode rewards
        window: Window size for comparison

    Returns:
        Improvement rate (positive = improving)
    """
    if len(rewards) < window * 2:
        return 0.0

    older_avg = np.mean(rewards[-window * 2:-window])
    recent_avg = np.mean(rewards[-window:])

    improvement = recent_avg - older_avg
    return improvement


def get_learning_insights(metrics: Dict[str, Any]) -> List[str]:
    """
    Generate insights about learning progress.

    Args:
        metrics: Training metrics

    Returns:
        List of insight strings
    """
    insights = []

    # Overall performance
    avg_reward = metrics["average_reward"]
    if avg_reward >= 8.0:
        insights.append("✓ Excellent overall performance!")
    elif avg_reward >= 6.5:
        insights.append("✓ Good overall performance")
    else:
        insights.append("⚠ Performance could be improved")

    # Improvement trend
    if len(metrics["episode_rewards"]) >= 10:
        improvement = calculate_improvement_rate(metrics["episode_rewards"])
        if improvement > 0.5:
            insights.append("✓ Strong improvement trend")
        elif improvement > 0:
            insights.append("✓ Slight improvement trend")
        else:
            insights.append("⚠ No clear improvement trend")

    # Best components
    avg_quality = np.mean(metrics["story_quality_scores"])
    avg_collab = np.mean(metrics["collaboration_scores"])
    avg_eff = np.mean(metrics["efficiency_scores"])

    best_component = max(
        [("Story Quality", avg_quality), ("Collaboration", avg_collab), ("Efficiency", avg_eff)],
        key=lambda x: x[1]
    )
    insights.append(f"✓ Strongest area: {best_component[0]} ({best_component[1]:.2f})")

    # Areas for improvement
    worst_component = min(
        [("Story Quality", avg_quality), ("Collaboration", avg_collab), ("Efficiency", avg_eff)],
        key=lambda x: x[1]
    )
    if worst_component[1] < 7.0:
        insights.append(f"⚠ Area to improve: {worst_component[0]} ({worst_component[1]:.2f})")

    return insights
