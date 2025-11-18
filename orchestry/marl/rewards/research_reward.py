"""
LLM-as-Judge Reward Model for Research Lab

Uses Claude to evaluate research quality instead of heuristic rules.
More accurate but costs ~$0.05 per evaluation.
"""

from typing import Any

from anthropic import Anthropic


class ResearchRewardModel:
    """
    LLM-based reward model for research evaluation.

    Uses Claude Sonnet to judge research quality on 5 dimensions:
    - Scientific rigor
    - Novelty
    - Completeness
    - Collaboration
    - Feasibility
    """

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022") -> None:
        """
        Initialize reward model.

        Args:
            api_key: Anthropic API key
            model: Claude model to use for evaluation
        """
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def evaluate_research(
        self,
        topic: str,
        objective: str,
        trajectory: list[dict[str, Any]],
        literature_reviewed: list[str],
        hypotheses: list[str],
        experiments: list[dict[str, Any]],
        analyses: list[str],
        paper_draft: str,
    ) -> dict[str, float]:
        """
        Evaluate research quality using Claude as judge.

        Args:
            topic: Research topic
            objective: Research objective
            trajectory: Sequence of agent interactions
            literature_reviewed: Literature synthesis outputs
            hypotheses: Generated hypotheses
            experiments: Designed experiments
            analyses: Data analyses
            paper_draft: Research paper draft

        Returns:
            Dictionary with scores for each dimension (0-10)
        """
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(
            topic=topic,
            objective=objective,
            trajectory=trajectory,
            literature_reviewed=literature_reviewed,
            hypotheses=hypotheses,
            experiments=experiments,
            analyses=analyses,
            paper_draft=paper_draft,
        )

        # Get Claude's evaluation
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0.0,  # Deterministic scoring
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse scores from response
        scores = self._parse_scores(response.content[0].text)

        return scores

    def _build_evaluation_prompt(
        self,
        topic: str,
        objective: str,
        trajectory: list[dict[str, Any]],
        literature_reviewed: list[str],
        hypotheses: list[str],
        experiments: list[dict[str, Any]],
        analyses: list[str],
        paper_draft: str,
    ) -> str:
        """Build evaluation prompt for Claude."""
        # Format trajectory
        trajectory_text = "\n\n".join(
            f"**{turn.get('role', 'agent').replace('_', ' ').title()}:**\n{turn.get('action', '')}"
            for turn in trajectory
        )

        # Format literature
        literature_text = "\n\n".join(f"- {lit}" for lit in literature_reviewed)

        # Format hypotheses
        hypotheses_text = "\n\n".join(f"{i+1}. {h}" for i, h in enumerate(hypotheses))

        # Format experiments
        experiments_text = "\n\n".join(
            f"**Experiment {i+1}:**\n{exp.get('design', '')}"
            for i, exp in enumerate(experiments)
        )

        # Format analyses
        analyses_text = "\n\n".join(f"- {a}" for a in analyses)

        prompt = f"""You are an expert scientific reviewer evaluating AI-generated research.

**Research Topic:** {topic}

**Objective:** {objective}

## Research Process

### Literature Review
{literature_text or "No literature reviewed"}

### Hypotheses Generated
{hypotheses_text or "No hypotheses generated"}

### Experimental Designs
{experiments_text or "No experiments designed"}

### Data Analyses
{analyses_text or "No analyses completed"}

### Paper Draft
{paper_draft or "No paper draft"}

---

## Full Trajectory
{trajectory_text}

---

## Evaluation Task

Rate this research on 5 dimensions (0-10 scale):

**1. Scientific Rigor (0-10)**
- Are hypotheses testable with specific metrics?
- Do experiments include controls and proper methodology?
- Is statistical analysis appropriate?
- Are citations and references appropriate?

**2. Novelty (0-10)**
- Do hypotheses extend beyond existing literature?
- Are approaches creative or just incremental?
- Is there potential for high-impact findings?

**3. Completeness (0-10)**
- Are all research phases completed (lit review, hypothesis, experiment, analysis, paper)?
- Is each phase sufficiently detailed?
- Is the paper draft publication-ready?

**4. Collaboration Quality (0-10)**
- Do agents build on each other's work?
- Is there clear information flow between phases?
- Do later agents reference earlier findings?

**5. Feasibility (0-10)**
- Are experiments realistic and achievable?
- Are resources/costs reasonable?
- Could this actually be tested in a lab?
- Are timelines realistic?

## Output Format

Respond ONLY with scores in this exact format (no other text):

```
SCIENTIFIC_RIGOR: [0-10 score]
NOVELTY: [0-10 score]
COMPLETENESS: [0-10 score]
COLLABORATION: [0-10 score]
FEASIBILITY: [0-10 score]
```

Provide decimal scores (e.g., 7.5) for precision.
"""

        return prompt

    def _parse_scores(self, response_text: str) -> dict[str, float]:
        """Parse scores from Claude's response."""
        scores = {
            "scientific_rigor": 5.0,
            "novelty": 5.0,
            "completeness": 5.0,
            "collaboration": 5.0,
            "feasibility": 5.0,
        }

        # Parse each line
        for line in response_text.split("\n"):
            line = line.strip()
            if "SCIENTIFIC_RIGOR:" in line:
                scores["scientific_rigor"] = float(line.split(":")[-1].strip())
            elif "NOVELTY:" in line:
                scores["novelty"] = float(line.split(":")[-1].strip())
            elif "COMPLETENESS:" in line:
                scores["completeness"] = float(line.split(":")[-1].strip())
            elif "COLLABORATION:" in line:
                scores["collaboration"] = float(line.split(":")[-1].strip())
            elif "FEASIBILITY:" in line:
                scores["feasibility"] = float(line.split(":")[-1].strip())

        # Calculate weighted total
        scores["total"] = (
            0.25 * scores["scientific_rigor"]
            + 0.25 * scores["novelty"]
            + 0.20 * scores["completeness"]
            + 0.15 * scores["collaboration"]
            + 0.15 * scores["feasibility"]
        )

        return scores


class HybridRewardModel:
    """
    Hybrid reward model: heuristic + LLM judge.

    - Uses fast heuristics during beam search
    - Uses LLM judge for final trajectory selection
    - Balances cost vs accuracy
    """

    def __init__(self, api_key: str, use_llm_for_final: bool = True) -> None:
        """
        Initialize hybrid model.

        Args:
            api_key: Anthropic API key
            use_llm_for_final: Whether to use LLM for final evaluation
        """
        self.llm_judge = ResearchRewardModel(api_key) if use_llm_for_final else None
        self.use_llm_for_final = use_llm_for_final

    def evaluate_intermediate(self, task: Any) -> dict[str, float]:
        """
        Fast heuristic evaluation during beam search.

        Args:
            task: ResearchLabTask instance

        Returns:
            Quick heuristic scores
        """
        # Use existing heuristic evaluation
        # (the code that's already in ResearchLabTask.evaluate)
        return task.evaluate(trajectory=[])

    def evaluate_final(
        self,
        topic: str,
        objective: str,
        trajectory: list[dict[str, Any]],
        task: Any,
    ) -> dict[str, float]:
        """
        High-quality LLM evaluation for final trajectory.

        Args:
            topic: Research topic
            objective: Research objective
            trajectory: Full agent interaction sequence
            task: ResearchLabTask instance

        Returns:
            LLM-judged scores
        """
        if self.llm_judge and self.use_llm_for_final:
            return self.llm_judge.evaluate_research(
                topic=topic,
                objective=objective,
                trajectory=trajectory,
                literature_reviewed=task.literature_reviewed,
                hypotheses=task.hypotheses_generated,
                experiments=task.experiments_designed,
                analyses=task.analyses_completed,
                paper_draft=task.paper_draft,
            )

        # Fallback to heuristic
        return task.evaluate(trajectory=trajectory)
