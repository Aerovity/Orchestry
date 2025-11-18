"""
Autonomous Research Lab Task - Multi-agent scientific research collaboration.

This task implements a 5-agent system for autonomous scientific research:
1. Literature Synthesizer - Reads and synthesizes research papers
2. Hypothesis Generator - Generates testable hypotheses
3. Experimental Designer - Designs experiments to test hypotheses
4. Data Analyst - Analyzes experimental results
5. Paper Writer - Writes research papers from findings
"""

from typing import Any

import numpy as np

from .base import BaseTask, TaskConfig


class ResearchLabTask(BaseTask):
    """
    Multi-agent autonomous research lab.

    Agents collaborate to:
    1. Review literature on a research topic
    2. Generate novel hypotheses
    3. Design experiments
    4. Analyze simulated/real data
    5. Write research paper drafts

    Domains supported:
    - Materials science (battery compositions, catalysts)
    - Climate modeling (carbon capture, renewable materials)
    - Protein folding (drug discovery, enzyme design)
    - Physics simulations (quantum systems, particle physics)
    """

    def __init__(
        self,
        domain: str = "materials_science",
        max_turns: int = 15,
        require_novelty: bool = True,
        use_real_data: bool = False,
        custom_problems: list[dict[str, Any]] | None = None,
        config: TaskConfig | None = None,
    ) -> None:
        """
        Initialize research lab task.

        Args:
            domain: Research domain (materials_science, climate, protein, physics)
            max_turns: Maximum conversation turns
            require_novelty: Whether to penalize non-novel hypotheses
            use_real_data: Whether to use real datasets (requires API access)
            custom_problems: User-provided research problems (overrides built-in problems)
            config: Task configuration (optional)
        """
        # Initialize base class with config
        if config is None:
            config = TaskConfig(max_turns=max_turns, task_type="research_lab")
        super().__init__(config)

        self.domain = domain
        self.max_turns = max_turns
        self.require_novelty = require_novelty
        self.use_real_data = use_real_data

        # Define agent roles
        self.agent_roles = [
            "literature_synthesizer",
            "hypothesis_generator",
            "experimental_designer",
            "data_analyst",
            "paper_writer",
        ]

        # Load research problems (custom or built-in)
        if custom_problems:
            self.research_problems = custom_problems
        else:
            self.research_problems = self._load_research_problems()

        # Track research progress
        self.current_problem: dict[str, Any] = {}
        self.literature_reviewed: list[str] = []
        self.hypotheses_generated: list[str] = []
        self.experiments_designed: list[dict[str, Any]] = []
        self.analyses_completed: list[str] = []
        self.paper_draft: str = ""
        self.turn_count: int = 0

    def _load_research_problems(self) -> list[dict[str, Any]]:
        """Load domain-specific research problems."""
        problems = {
            "materials_science": [
                {
                    "topic": "Next-generation solid-state battery electrolytes",
                    "context": "Current lithium-ion batteries use liquid electrolytes that are flammable. "
                    "Solid-state electrolytes promise higher energy density and safety.",
                    "key_papers": [
                        "Li7La3Zr2O12 garnet electrolytes (conductivity: 10^-4 S/cm)",
                        "NASICON-type electrolytes (conductivity: 10^-3 S/cm)",
                        "Sulfide-based electrolytes (conductivity: 10^-2 S/cm, air-sensitive)",
                    ],
                    "objective": "Discover novel solid electrolyte compositions with >10^-2 S/cm "
                    "conductivity and air stability",
                    "success_metrics": [
                        "Ionic conductivity",
                        "Electrochemical stability window",
                        "Air/moisture stability",
                        "Cost of materials",
                    ],
                },
                {
                    "topic": "CO2 reduction catalysts for carbon-neutral fuels",
                    "context": "Converting CO2 to useful chemicals (methanol, ethanol, formic acid) "
                    "requires efficient electrocatalysts.",
                    "key_papers": [
                        "Copper-based catalysts (high C2+ selectivity, poor stability)",
                        "Gold nanoparticles (selective CO production)",
                        "Bismuth catalysts (formic acid production, low overpotential)",
                    ],
                    "objective": "Design catalyst with >80% Faradaic efficiency for C2+ products "
                    "at <0.5V overpotential",
                    "success_metrics": [
                        "Faradaic efficiency",
                        "Overpotential",
                        "Current density",
                        "Catalyst stability",
                    ],
                },
            ],
            "climate": [
                {
                    "topic": "Direct air capture materials for CO2 sequestration",
                    "context": "Removing CO2 from atmosphere requires sorbent materials with high "
                    "capacity and low regeneration energy.",
                    "key_papers": [
                        "Amine-functionalized sorbents (capacity: 2-3 mmol/g, high energy)",
                        "MOF materials (capacity: 5-10 mmol/g, expensive synthesis)",
                        "Porous carbons (low capacity, cheap, regenerable)",
                    ],
                    "objective": "Discover sorbent with >5 mmol/g capacity, <80°C regeneration",
                    "success_metrics": [
                        "CO2 uptake capacity",
                        "Regeneration temperature",
                        "Cycling stability",
                        "Material cost",
                    ],
                },
            ],
            "protein": [
                {
                    "topic": "De novo enzyme design for plastic degradation",
                    "context": "PET plastic accumulates in environment. Natural enzymes exist "
                    "(PETase) but are slow.",
                    "key_papers": [
                        "IsPETase from Ideonella sakaiensis (kcat: 0.04 s^-1)",
                        "LCC from Leaf Compost Cutinase (kcat: 0.26 s^-1)",
                        "Engineered PETase variants (kcat: up to 0.8 s^-1)",
                    ],
                    "objective": "Design enzyme with kcat >1.0 s^-1 and stability at 50-70°C",
                    "success_metrics": [
                        "Catalytic rate (kcat)",
                        "Thermal stability (Tm)",
                        "Substrate binding (Km)",
                        "Stability over time",
                    ],
                },
            ],
            "physics": [
                {
                    "topic": "Room-temperature superconductor candidates",
                    "context": "High-pressure hydrides show high Tc but require extreme pressures. "
                    "Need ambient-pressure alternatives.",
                    "key_papers": [
                        "H3S (Tc = 203K at 155 GPa)",
                        "LaH10 (Tc = 250K at 170 GPa)",
                        "Carbonaceous sulfur hydride (Tc = 288K at 267 GPa, disputed)",
                    ],
                    "objective": "Predict materials with Tc >200K at ambient pressure",
                    "success_metrics": [
                        "Critical temperature",
                        "Required pressure",
                        "Material stability",
                        "Synthesis feasibility",
                    ],
                },
            ],
        }

        return problems.get(self.domain, problems["materials_science"])

    def reset(self) -> dict[str, Any]:
        """Reset task with a new research problem."""
        # Select random research problem
        self.current_problem = np.random.choice(self.research_problems)

        # Reset tracking
        self.literature_reviewed = []
        self.hypotheses_generated = []
        self.experiments_designed = []
        self.analyses_completed = []
        self.paper_draft = ""
        self.turn_count = 0

        # Initial observation for all agents
        initial_obs = {
            "topic": self.current_problem["topic"],
            "context": self.current_problem["context"],
            "objective": self.current_problem["objective"],
            "success_metrics": self.current_problem["success_metrics"],
            "current_phase": "literature_review",
        }

        return initial_obs

    def step(
        self, agent_id: int, agent_role: str, action: str
    ) -> tuple[dict[str, Any], bool]:
        """
        Execute agent action and return observation.

        Args:
            agent_id: Agent identifier
            agent_role: Agent's role in research process
            action: Agent's contribution (text)

        Returns:
            observation: Updated research state (includes info in observation dict)
            done: Whether research is complete
        """
        self.turn_count += 1

        # Process action based on agent role
        if agent_role == "literature_synthesizer":
            self._process_literature_synthesis(action)
        elif agent_role == "hypothesis_generator":
            self._process_hypothesis(action)
        elif agent_role == "experimental_designer":
            self._process_experiment_design(action)
        elif agent_role == "data_analyst":
            self._process_data_analysis(action)
        elif agent_role == "paper_writer":
            self._process_paper_writing(action)

        # Determine current research phase
        phase = self._determine_phase()

        # Check if done
        done = self._is_research_complete()

        # Create observation (includes info)
        observation = {
            "topic": self.current_problem["topic"],
            "current_phase": phase,
            "literature_reviewed": len(self.literature_reviewed),
            "hypotheses_generated": len(self.hypotheses_generated),
            "experiments_designed": len(self.experiments_designed),
            "analyses_completed": len(self.analyses_completed),
            "paper_draft_length": len(self.paper_draft),
            "turn_count": self.turn_count,
            "last_action_role": agent_role,
            "phase_progress": self._calculate_phase_progress(),
        }

        return observation, done

    def _process_literature_synthesis(self, action: str) -> None:
        """Process literature synthesis contribution."""
        # Extract key findings from action
        if len(action) > 50:  # Meaningful synthesis
            self.literature_reviewed.append(action)

    def _process_hypothesis(self, action: str) -> None:
        """Process hypothesis generation."""
        # Check for hypothesis markers
        hypothesis_markers = ["hypothesis:", "propose:", "predict:", "theory:"]
        if any(marker in action.lower() for marker in hypothesis_markers):
            self.hypotheses_generated.append(action)

    def _process_experiment_design(self, action: str) -> None:
        """Process experimental design."""
        # Check for experimental details
        experiment_markers = ["experiment:", "method:", "procedure:", "measure:"]
        if any(marker in action.lower() for marker in experiment_markers):
            experiment = {
                "design": action,
                "timestamp": self.turn_count,
            }
            self.experiments_designed.append(experiment)

    def _process_data_analysis(self, action: str) -> None:
        """Process data analysis."""
        # Check for analysis content
        analysis_markers = ["result:", "analysis:", "finding:", "data shows:"]
        if any(marker in action.lower() for marker in analysis_markers):
            self.analyses_completed.append(action)

    def _process_paper_writing(self, action: str) -> None:
        """Process paper writing."""
        # Accumulate paper draft
        if len(action) > 100:  # Substantial writing
            self.paper_draft += f"\n\n{action}"

    def _determine_phase(self) -> str:
        """Determine current research phase."""
        if len(self.literature_reviewed) < 2:
            return "literature_review"
        if len(self.hypotheses_generated) < 1:
            return "hypothesis_generation"
        if len(self.experiments_designed) < 1:
            return "experimental_design"
        if len(self.analyses_completed) < 1:
            return "data_analysis"
        return "paper_writing"

    def _calculate_phase_progress(self) -> dict[str, float]:
        """Calculate progress in each phase."""
        return {
            "literature_review": min(1.0, len(self.literature_reviewed) / 3),
            "hypothesis_generation": min(1.0, len(self.hypotheses_generated) / 2),
            "experimental_design": min(1.0, len(self.experiments_designed) / 2),
            "data_analysis": min(1.0, len(self.analyses_completed) / 2),
            "paper_writing": min(1.0, len(self.paper_draft) / 1000),
        }

    def _is_research_complete(self) -> bool:
        """Check if research process is complete."""
        # Complete if all phases done or max turns reached
        all_phases_complete = (
            len(self.literature_reviewed) >= 2
            and len(self.hypotheses_generated) >= 1
            and len(self.experiments_designed) >= 1
            and len(self.analyses_completed) >= 1
            and len(self.paper_draft) >= 500
        )

        return all_phases_complete or self.turn_count >= self.max_turns

    def evaluate(self, trajectory: list[dict[str, Any]]) -> dict[str, float]:
        """
        Evaluate research quality.

        Metrics:
        - Scientific rigor (0-10): Complete methodology, proper citations
        - Novelty (0-10): Novel hypotheses and approaches
        - Completeness (0-10): All research phases completed
        - Collaboration (0-10): Agents building on each other's work
        - Feasibility (0-10): Realistic experimental designs

        Args:
            trajectory: Sequence of agent interactions

        Returns:
            Dictionary of evaluation scores
        """
        # Calculate scientific rigor
        rigor_score = self._calculate_rigor_score()

        # Calculate novelty
        novelty_score = self._calculate_novelty_score()

        # Calculate completeness
        completeness_score = self._calculate_completeness_score()

        # Calculate collaboration quality
        collaboration_score = self._calculate_collaboration_score(trajectory)

        # Calculate feasibility
        feasibility_score = self._calculate_feasibility_score()

        # Weighted total
        total_score = (
            0.25 * rigor_score
            + 0.25 * novelty_score
            + 0.20 * completeness_score
            + 0.15 * collaboration_score
            + 0.15 * feasibility_score
        )

        return {
            "scientific_rigor": rigor_score,
            "novelty": novelty_score,
            "completeness": completeness_score,
            "collaboration": collaboration_score,
            "feasibility": feasibility_score,
            "total": total_score,
        }

    def _calculate_rigor_score(self) -> float:
        """Calculate scientific rigor score."""
        score = 0.0

        # Literature review depth
        if len(self.literature_reviewed) >= 2:
            score += 2.0
        if len(self.literature_reviewed) >= 4:
            score += 1.0

        # Hypothesis quality (mentions metrics, testable)
        for hyp in self.hypotheses_generated:
            if any(metric in hyp.lower() for metric in ["measure", "test", "quantify"]):
                score += 1.5

        # Experimental design completeness
        for exp in self.experiments_designed:
            design = exp["design"].lower()
            if "control" in design and "measure" in design:
                score += 1.5

        # Analysis depth
        for analysis in self.analyses_completed:
            if any(word in analysis.lower() for word in ["significant", "correlation", "trend"]):
                score += 1.0

        return min(10.0, score)

    def _calculate_novelty_score(self) -> float:
        """Calculate novelty score."""
        score = 5.0  # Base score

        # Penalize if hypotheses too similar to literature
        if self.require_novelty:
            for hyp in self.hypotheses_generated:
                # Check if hypothesis extends beyond known work
                if any(word in hyp.lower() for word in ["novel", "new", "improve", "beyond"]):
                    score += 2.0

        # Bonus for multiple diverse hypotheses
        if len(self.hypotheses_generated) >= 2:
            score += 1.0

        return min(10.0, score)

    def _calculate_completeness_score(self) -> float:
        """Calculate research completeness score."""
        score = 0.0

        # Each phase completion
        if len(self.literature_reviewed) >= 2:
            score += 2.0
        if len(self.hypotheses_generated) >= 1:
            score += 2.0
        if len(self.experiments_designed) >= 1:
            score += 2.0
        if len(self.analyses_completed) >= 1:
            score += 2.0
        if len(self.paper_draft) >= 500:
            score += 2.0

        return score

    def _calculate_collaboration_score(self, trajectory: list[dict[str, Any]]) -> float:
        """Calculate how well agents collaborated."""
        if len(trajectory) < 2:
            return 0.0

        collaboration_score = 0.0

        # Check for references to other agents' work
        for i, turn in enumerate(trajectory[1:], 1):
            action = turn.get("action", "").lower()

            # Look for references to previous contributions
            references = [
                "based on",
                "building on",
                "as mentioned",
                "according to",
                "following",
                "using the",
            ]
            if any(ref in action for ref in references):
                collaboration_score += 1.0

            # Check for cross-phase integration
            if i >= 2:
                prev_role = trajectory[i - 1].get("role", "")
                curr_role = turn.get("role", "")
                if prev_role != curr_role:  # Different agents
                    # Check if current agent references previous
                    if len(action) > 50:
                        collaboration_score += 0.5

        return min(10.0, collaboration_score)

    def _calculate_feasibility_score(self) -> float:
        """Calculate experimental feasibility."""
        score = 5.0  # Base score

        # Check experimental designs for feasibility indicators
        for exp in self.experiments_designed:
            design = exp["design"].lower()

            # Positive indicators
            if any(word in design for word in ["standard", "established", "validated"]):
                score += 1.0

            # Negative indicators (unrealistic)
            if any(word in design for word in ["assume perfect", "infinite", "zero cost"]):
                score -= 2.0

        return max(0.0, min(10.0, score))

    def get_agent_prompt(self, agent_id: int, agent_role: str, observation: dict[str, Any]) -> str:
        """
        Generate role-specific prompt for agent.

        Args:
            agent_id: Agent identifier
            agent_role: Agent's research role
            observation: Current research state

        Returns:
            Formatted prompt string
        """
        base_prompt = f"""You are the {agent_role.replace('_', ' ').title()} in an autonomous research laboratory.

Current Research Project: {observation['topic']}

Objective: {observation.get('objective', 'Advance scientific understanding')}

Current Phase: {observation.get('current_phase', 'unknown')}

Progress:
- Literature reviewed: {observation.get('literature_reviewed', 0)} sources
- Hypotheses generated: {observation.get('hypotheses_generated', 0)}
- Experiments designed: {observation.get('experiments_designed', 0)}
- Analyses completed: {observation.get('analyses_completed', 0)}
- Paper draft: {observation.get('paper_draft_length', 0)} characters

"""

        # Role-specific instructions
        role_instructions = {
            "literature_synthesizer": """Your role: Synthesize existing research literature.

Tasks:
1. Identify key findings from the provided papers
2. Highlight gaps in current knowledge
3. Summarize state-of-the-art approaches
4. Provide context for hypothesis generation

Be concise but thorough. Focus on quantitative results and methods.""",
            "hypothesis_generator": """Your role: Generate novel, testable hypotheses.

Tasks:
1. Based on literature review, identify promising directions
2. Propose specific, testable hypotheses
3. Explain expected outcomes
4. Suggest metrics for validation

Make hypotheses concrete and measurable.""",
            "experimental_designer": """Your role: Design experiments to test hypotheses.

Tasks:
1. Design controlled experiments
2. Specify materials, methods, and measurements
3. Include control groups and variables
4. Estimate resource requirements

Be realistic and specific about experimental procedures.""",
            "data_analyst": """Your role: Analyze experimental results.

Tasks:
1. Interpret experimental data
2. Identify trends and patterns
3. Assess statistical significance
4. Compare to predictions from hypotheses

Provide quantitative analysis where possible.""",
            "paper_writer": """Your role: Write research paper draft.

Tasks:
1. Synthesize all findings into coherent narrative
2. Structure: Abstract, Introduction, Methods, Results, Discussion
3. Highlight novel contributions
4. Suggest future work

Write clearly and concisely in scientific style.""",
        }

        return base_prompt + role_instructions.get(agent_role, "")

    def is_done(self) -> bool:
        """Check if research episode is complete."""
        return self._is_research_complete()
