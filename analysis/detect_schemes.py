"""
Cooperation Scheme Detection

Identifies cooperation patterns from trajectories:
- Fallback: try-except around helper call
- Decorator: helper call + additional logic
- Coordinator: loop calling helper
- Strategy Filter: helper in conditional
"""

import re
from collections import Counter

from orchestry.marl.algorithms.magrpo import Trajectory


def detect_fallback(_helper_code: str, main_code: str) -> bool:
    """
    Detect fallback pattern.

    Main agent wraps helper call in try-except for robustness.
    """
    # Look for try-except pattern with helper call
    pattern = r"try:.*helper.*except"
    return bool(re.search(pattern, main_code, re.DOTALL | re.IGNORECASE))


def detect_decorator(_helper_code: str, main_code: str) -> bool:
    """
    Detect decorator pattern.

    Main agent calls helper and adds additional logic/edge cases.
    """
    # Helper is called, and main has significant additional logic
    if "helper" not in main_code.lower():
        return False

    # Main should have more than just a simple helper call
    main_lines = [
        line.strip()
        for line in main_code.split("\n")
        if line.strip() and not line.strip().startswith("#")
    ]

    # Simple heuristic: more than 5 lines beyond helper call
    return len(main_lines) > 5


def detect_coordinator(_helper_code: str, main_code: str) -> bool:
    """
    Detect coordinator pattern.

    Main agent uses loop to coordinate multiple helper calls.
    """
    # Look for loop (for/while) that calls helper
    loop_patterns = [
        r"for .* in .*:.*helper",
        r"while .*:.*helper",
    ]

    for pattern in loop_patterns:
        if re.search(pattern, main_code, re.DOTALL | re.IGNORECASE):
            return True

    return False


def detect_strategy_filter(_helper_code: str, main_code: str) -> bool:
    """
    Detect strategy filter pattern.

    Helper is used in conditional to guide main logic.
    """
    # Look for helper call in if statement
    pattern = r"if\s+.*helper\s*\("
    return bool(re.search(pattern, main_code, re.IGNORECASE))


def classify_scheme(trajectory: Trajectory) -> str:
    """
    Classify cooperation scheme for a trajectory.

    Args:
        trajectory: Trajectory with helper and main turns

    Returns:
        scheme: One of "fallback", "decorator", "coordinator", "strategy_filter", "other"

    """
    if len(trajectory.turns) < 2:
        return "other"

    helper_code = trajectory.turns[0]["action"]
    main_code = trajectory.turns[1]["action"]

    # Check patterns in priority order
    if detect_fallback(helper_code, main_code):
        return "fallback"

    if detect_coordinator(helper_code, main_code):
        return "coordinator"

    if detect_strategy_filter(helper_code, main_code):
        return "strategy_filter"

    if detect_decorator(helper_code, main_code):
        return "decorator"

    return "other"


def analyze_schemes(trajectories: list[Trajectory]) -> dict[str, int]:
    """
    Analyze cooperation schemes across multiple trajectories.

    Args:
        trajectories: List of trajectories to analyze

    Returns:
        scheme_counts: Dict mapping scheme name to count

    """
    schemes = [classify_scheme(t) for t in trajectories]
    return dict(Counter(schemes))


def analyze_scheme_evolution(checkpoints: list[str]) -> dict[int, dict[str, int]]:
    """
    Analyze how cooperation schemes evolve during training.

    Args:
        checkpoints: List of checkpoint paths (e.g., ["episode_0", "episode_100", ...])

    Returns:
        evolution: Dict mapping episode number to scheme distribution

    """
    from pathlib import Path

    evolution: dict[int, dict[str, int]] = {}

    for checkpoint_path in checkpoints:
        # Extract episode number from path
        episode = int(Path(checkpoint_path).name.split("_")[-1])

        # Load metrics
        metrics_file = Path(checkpoint_path) / "metrics.json"
        if not metrics_file.exists():
            continue

        # Get scheme distribution for this episode
        # Note: This requires schemes to be stored in metrics
        # For now, return empty dict
        evolution[episode] = {}

    return evolution


def print_scheme_examples(trajectories: list[Trajectory], max_examples: int = 3) -> None:
    """
    Print example code for each cooperation scheme.

    Args:
        trajectories: List of trajectories
        max_examples: Maximum examples per scheme

    """
    schemes: dict[str, list[Trajectory]] = {}

    for traj in trajectories:
        scheme = classify_scheme(traj)

        if scheme not in schemes:
            schemes[scheme] = []

        if len(schemes[scheme]) < max_examples:
            schemes[scheme].append(traj)

    for scheme, examples in schemes.items():
        print(f"\n{'=' * 60}")
        print(f"SCHEME: {scheme.upper()}")
        print("=" * 60)

        for i, traj in enumerate(examples):
            if len(traj.turns) < 2:
                continue

            print(f"\nExample {i + 1}:")
            print("\n--- Helper Code ---")
            print(traj.turns[0]["action"])
            print("\n--- Main Code ---")
            print(traj.turns[1]["action"])
            print(f"\nReward: {traj.total_reward:.3f}")
            print(f"Cooperation: {traj.reward_components.get('cooperation', 0):.3f}")
            print("-" * 60)
