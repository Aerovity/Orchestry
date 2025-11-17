"""
Multi-Agent Group Relative Policy Optimization (MAGRPO)

Implementation of Algorithm 1 from the paper:
"LLM Collaboration with Multi-Agent Reinforcement Learning"

Key equations:
- Equation 1: Â(g) = R(g) - (1/G) * Σ R(g')  [Group relative advantage]
- Equation 2: J(θ) = E[Â(g) * log π_θ(a|h)]  [Policy gradient]
"""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch  # type: ignore[import-not-found]


class AgentModel(Protocol):
    """Protocol for agent models with compute_log_prob method."""

    def compute_log_prob(self, action: str, observation: str) -> torch.Tensor:
        """Compute log probability of action given observation."""
        ...


@dataclass
class Trajectory:
    """Single trajectory through the environment."""

    turns: list[dict[str, Any]]  # List of {agent_id, observation, action, log_prob}
    total_reward: float
    reward_components: dict[str, float]

    def __post_init__(self) -> None:
        """Initialize empty turns list if not provided."""
        if not self.turns:
            self.turns = []


def compute_advantages(returns: np.ndarray) -> np.ndarray:
    """
    Compute group-relative advantages (Equation 1 from paper).

    Â(g) = R(g) - (1/G) * Σ_{g'=1}^G R(g')

    The advantage of each trajectory is its return minus the mean return
    across all trajectories in the group.

    Args:
        returns: Array of shape (G,) containing returns for each trajectory

    Returns:
        advantages: Array of shape (G,) with group-relative advantages

    Note:
        Advantages will sum to zero by construction.

    """
    mean_return = np.mean(returns)
    advantages: np.ndarray[tuple[int], np.dtype[np.floating[Any]]] = returns - mean_return
    return advantages


def compute_policy_loss(
    trajectories: list[Trajectory],
    advantages: np.ndarray,
    agent_id: int,
    agent_model: AgentModel,
) -> torch.Tensor:
    """
    Compute policy gradient loss (Equation 2 from paper).

    J(θ_i) = E[Â(g) * log π_θi(a_i|h_i)]

    This is an on-policy gradient (no importance sampling, no clipping).

    Args:
        trajectories: List of G trajectories
        advantages: Array of shape (G,) with advantages
        agent_id: Which agent we're updating (0 or 1)
        agent_model: The agent's model with compute_log_prob method

    Returns:
        loss: Scalar tensor (negative for gradient ascent)

    """
    total_loss = 0.0
    count = 0

    for traj, advantage in zip(trajectories, advantages, strict=True):
        for turn in traj.turns:
            if turn["agent_id"] == agent_id:
                # Get log probability from stored value or recompute
                if "log_prob" in turn:
                    log_prob = turn["log_prob"]
                else:
                    log_prob = agent_model.compute_log_prob(turn["action"], turn["observation"])

                # Policy gradient: advantage * log_prob
                # Negative because we do gradient descent (maximizing is minimizing negative)
                total_loss -= advantage * log_prob
                count += 1

    if count == 0:
        return torch.tensor(0.0, requires_grad=True)

    # Average over all turns for this agent
    return total_loss / count


class MAGRPOOptimizer:
    """
    MAGRPO optimizer implementing Algorithm 1 from the paper.

    Usage:
        optimizer = MAGRPOOptimizer(agents, learning_rate=1e-4)

        # Collect batch of episodes
        trajectories = collect_trajectories(k=4)

        # Update policies
        optimizer.update(trajectories)
    """

    def __init__(
        self,
        agents: list[Any],
        learning_rate: float = 1e-4,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 100,
    ) -> None:
        """
        Initialize MAGRPO optimizer.

        Args:
            agents: List of agent models (with LoRA parameters)
            learning_rate: Learning rate for Adam optimizer
            max_grad_norm: Maximum gradient norm for clipping
            warmup_steps: Number of steps for learning rate warmup

        """
        self.agents = agents
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps

        # Create optimizer for each agent's LoRA parameters
        self.optimizers = []
        for agent in agents:
            params = agent.model.parameters() if hasattr(agent, "model") else agent.parameters()
            optimizer = torch.optim.AdamW(params, lr=learning_rate)
            self.optimizers.append(optimizer)

        self.step_count = 0

    def update(self, trajectories: list[Trajectory]) -> dict[str, float]:
        """
        Perform MAGRPO update on all agents.

        Args:
            trajectories: List of G trajectories from episode batch

        Returns:
            metrics: Dict with loss values and other metrics

        """
        # 1. Extract returns
        returns = np.array([traj.total_reward for traj in trajectories])

        # 2. Compute group-relative advantages (Equation 1)
        advantages = compute_advantages(returns)

        # 3. Update each agent's policy (Equation 2)
        metrics = {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_advantage": float(np.mean(np.abs(advantages))),
        }

        for agent_id, (agent, optimizer) in enumerate(
            zip(self.agents, self.optimizers, strict=True),
        ):
            # Compute policy loss
            loss = compute_policy_loss(trajectories, advantages, agent_id, agent)

            # Backward pass
            optimizer.zero_grad()
            if loss.requires_grad:
                loss.backward()

                # Gradient clipping
                params = agent.model.parameters() if hasattr(agent, "model") else agent.parameters()
                torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)

                # Optimizer step with learning rate warmup
                if self.step_count < self.warmup_steps:
                    lr_scale = (self.step_count + 1) / self.warmup_steps
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = self.learning_rate * lr_scale

                optimizer.step()

            metrics[f"agent_{agent_id}_loss"] = float(loss.detach())

        self.step_count += 1
        return metrics

    def get_learning_rates(self) -> list[float]:
        """Get current learning rates for all agents."""
        return [opt.param_groups[0]["lr"] for opt in self.optimizers]
