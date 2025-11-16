"""Tests for MARL trajectory management."""

from orchestry.marl.trajectory import MultiTurnTrajectory, TrajectoryBeam, Turn


def test_turn_creation() -> None:
    """Test Turn dataclass creation."""
    turn = Turn(
        agent_id=0,
        agent_role="Test Role",
        observation="Test observation",
        action="Test action",
        turn_number=1,
        metadata={"test": True},
    )

    assert turn.agent_id == 0
    assert turn.agent_role == "Test Role"
    assert turn.observation == "Test observation"
    assert turn.action == "Test action"
    assert turn.turn_number == 1
    assert turn.metadata == {"test": True}


def test_trajectory_adds_turns() -> None:
    """Test that trajectory correctly adds turns."""
    traj = MultiTurnTrajectory()

    traj.add_turn(
        agent_id=0,
        agent_role="Writer",
        observation="Task: Write code",
        action="First action",
    )
    traj.add_turn(
        agent_id=1,
        agent_role="Reviewer",
        observation="Task: Review code",
        action="Second action",
    )

    assert len(traj.turns) == 2
    assert traj.turns[0].agent_id == 0
    assert traj.turns[1].agent_id == 1


def test_trajectory_calculates_total_reward() -> None:
    """Test trajectory total reward calculation."""
    traj = MultiTurnTrajectory()

    traj.add_turn(
        agent_id=0,
        agent_role="Writer",
        observation="Task",
        action="A",
    )
    traj.add_turn(
        agent_id=1,
        agent_role="Reviewer",
        observation="Task",
        action="B",
    )
    traj.add_turn(
        agent_id=0,
        agent_role="Writer",
        observation="Task",
        action="C",
    )

    # Set rewards
    traj.set_rewards(5.0, {"quality": 3.0, "collaboration": 2.0})

    assert traj.total_reward == 5.0


def test_trajectory_beam_initialization() -> None:
    """Test TrajectoryBeam initialization."""
    beam = TrajectoryBeam(beam_width=5)

    assert beam.beam_width == 5
    assert len(beam.trajectories) == 0


def test_trajectory_beam_adds_trajectories() -> None:
    """Test adding trajectories to beam."""
    beam = TrajectoryBeam(beam_width=3)

    traj1 = MultiTurnTrajectory()
    traj1.add_turn(agent_id=0, agent_role="Writer", observation="Task", action="A")

    traj2 = MultiTurnTrajectory()
    traj2.add_turn(agent_id=0, agent_role="Writer", observation="Task", action="B")

    beam.add(traj1, 5.0)
    beam.add(traj2, 3.0)

    assert len(beam.trajectories) == 2


def test_trajectory_beam_maintains_width() -> None:
    """Test that beam maintains maximum width."""
    beam = TrajectoryBeam(beam_width=2)

    # Add 3 trajectories
    for i, score in enumerate([5.0, 3.0, 7.0]):
        traj = MultiTurnTrajectory()
        traj.add_turn(
            agent_id=0,
            agent_role="Writer",
            observation="Task",
            action=f"Content {i}",
        )
        beam.add(traj, score)

    # Prune to maintain width
    beam.prune()

    # Should only keep top 2
    assert len(beam.trajectories) <= 2


def test_trajectory_beam_get_best() -> None:
    """Test getting best trajectory from beam."""
    beam = TrajectoryBeam(beam_width=3)

    traj1 = MultiTurnTrajectory()
    traj1.add_turn(agent_id=0, agent_role="Writer", observation="Task", action="Low")
    traj1.set_rewards(3.0, {})

    traj2 = MultiTurnTrajectory()
    traj2.add_turn(agent_id=0, agent_role="Writer", observation="Task", action="High")
    traj2.set_rewards(8.0, {})

    traj3 = MultiTurnTrajectory()
    traj3.add_turn(agent_id=0, agent_role="Writer", observation="Task", action="Medium")
    traj3.set_rewards(5.0, {})

    beam.add(traj1, 3.0)
    beam.add(traj2, 8.0)
    beam.add(traj3, 5.0)

    best = beam.get_best()
    assert best is not None
    assert best.total_reward == 8.0
    assert best.turns[0].action == "High"


def test_empty_beam_get_best_returns_none() -> None:
    """Test that getting best from empty beam returns None."""
    beam = TrajectoryBeam(beam_width=3)
    assert beam.get_best() is None
