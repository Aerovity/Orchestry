"""
LLM-as-Judge Reward Model for Research Lab

Uses Claude or Gemini to evaluate research quality instead of heuristic rules.
More accurate but costs ~$0.05 per evaluation.
"""

from typing import Any

from anthropic import Anthropic
import google.generativeai as genai


class ResearchRewardModel:
    """
    LLM-based reward model for research evaluation.

    Uses Claude Sonnet or Gemini to judge research quality on 5 dimensions:
    - Scientific rigor
    - Novelty
    - Completeness
    - Collaboration
    - Feasibility
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        provider: str = "claude",
        gemini_api_key: str | None = None,
    ) -> None:
        """
        Initialize reward model.

        Args:
            api_key: Anthropic API key (for Claude)
            model: Model to use for evaluation
            provider: "claude" or "gemini"
            gemini_api_key: Google Gemini API key (if using Gemini)
        """
        self.provider = provider.lower()
        self.model = model

        if self.provider == "claude":
            self.client = Anthropic(api_key=api_key)
        elif self.provider == "gemini":
            if gemini_api_key:
                genai.configure(api_key=gemini_api_key)
            elif api_key:
                genai.configure(api_key=api_key)
            # Default to gemini-2.0-flash - stable and fast with high RPM
            self.client = genai.GenerativeModel(model or "gemini-2.0-flash")
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'claude' or 'gemini'")

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
        Evaluate research quality using Claude or Gemini as judge.

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
        import logging
        import time
        logger = logging.getLogger(__name__)

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

        # Retry logic for LLM judge
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Get LLM evaluation based on provider
                if self.provider == "claude":
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=1024,
                        temperature=0.0,  # Deterministic scoring
                        messages=[{"role": "user", "content": prompt}],
                    )
                    response_text = response.content[0].text
                elif self.provider == "gemini":
                    # Disable safety filters for research evaluation
                    from google.generativeai.types import HarmCategory, HarmBlockThreshold
                    safety_settings = {
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    }

                    response = self.client.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.0, max_output_tokens=1024
                        ),
                        safety_settings=safety_settings,
                    )

                    # Handle potential safety blocks
                    try:
                        response_text = response.text
                    except (ValueError, AttributeError) as e:
                        logger.warning(f"Gemini blocked LLM judge response (attempt {attempt + 1}/{max_retries}): {e}")
                        if attempt < max_retries - 1:
                            time.sleep(1)
                            continue
                        # Use fallback scores on final attempt
                        return self._get_fallback_scores()
                else:
                    raise ValueError(f"Unknown provider: {self.provider}")

                # Parse scores from response
                scores = self._parse_scores(response_text)

                # Verify we got valid scores (not all defaults)
                if scores.get("total", 0) > 0:
                    return scores
                else:
                    logger.warning(f"LLM judge returned invalid scores (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    return self._get_fallback_scores()

            except Exception as e:
                logger.error(f"LLM judge error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                # Return fallback scores on final failure
                return self._get_fallback_scores()

        # Should never reach here, but just in case
        return self._get_fallback_scores()

    def _get_fallback_scores(self) -> dict[str, float]:
        """Return fallback scores when LLM judge fails."""
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("Using fallback scores for LLM judge failure")

        return {
            "scientific_rigor": 5.0,
            "novelty": 5.0,
            "completeness": 5.0,
            "collaboration": 5.0,
            "feasibility": 5.0,
            "total": 5.0,
        }

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
        """Build evaluation prompt for Claude or Gemini."""
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
        """Parse scores from Claude or Gemini response."""
        import logging
        logger = logging.getLogger(__name__)

        scores = {
            "scientific_rigor": 5.0,
            "novelty": 5.0,
            "completeness": 5.0,
            "collaboration": 5.0,
            "feasibility": 5.0,
        }

        # Parse each line
        parsed_count = 0
        for line in response_text.split("\n"):
            line = line.strip()
            try:
                if "SCIENTIFIC_RIGOR:" in line:
                    scores["scientific_rigor"] = float(line.split(":")[-1].strip())
                    parsed_count += 1
                elif "NOVELTY:" in line:
                    scores["novelty"] = float(line.split(":")[-1].strip())
                    parsed_count += 1
                elif "COMPLETENESS:" in line:
                    scores["completeness"] = float(line.split(":")[-1].strip())
                    parsed_count += 1
                elif "COLLABORATION:" in line:
                    scores["collaboration"] = float(line.split(":")[-1].strip())
                    parsed_count += 1
                elif "FEASIBILITY:" in line:
                    scores["feasibility"] = float(line.split(":")[-1].strip())
                    parsed_count += 1
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse score from line: {line} - {e}")

        if parsed_count == 0:
            logger.warning(f"LLM Judge returned no parseable scores! Response:\n{response_text[:500]}")

        # Calculate weighted total
        scores["total"] = (
            0.25 * scores["scientific_rigor"]
            + 0.25 * scores["novelty"]
            + 0.20 * scores["completeness"]
            + 0.15 * scores["collaboration"]
            + 0.15 * scores["feasibility"]
        )

        logger.info(f"LLM Judge Scores: Rigor={scores['scientific_rigor']:.1f}, Novel={scores['novelty']:.1f}, Complete={scores['completeness']:.1f}, Collab={scores['collaboration']:.1f}, Feasible={scores['feasibility']:.1f} → Total={scores['total']:.2f}")

        return scores


class HybridRewardModel:
    """
    Hybrid reward model: heuristic + LLM judge.

    - Uses fast heuristics during beam search
    - Uses LLM judge for final trajectory selection
    - Balances cost vs accuracy
    """

    def __init__(
        self,
        api_key: str,
        use_llm_for_final: bool = True,
        provider: str = "claude",
        model: str | None = None,
        gemini_api_key: str | None = None,
    ) -> None:
        """
        Initialize hybrid model.

        Args:
            api_key: API key (Anthropic for Claude, or Google for Gemini)
            use_llm_for_final: Whether to use LLM for final evaluation
            provider: "claude" or "gemini"
            model: Model name (optional, defaults per provider)
            gemini_api_key: Separate Gemini API key if needed
        """
        self.use_llm_for_final = use_llm_for_final
        self.provider = provider

        if use_llm_for_final:
            self.llm_judge = ResearchRewardModel(
                api_key=api_key,
                model=model or ("claude-3-5-sonnet-20241022" if provider == "claude" else "gemini-2.0-flash-exp"),
                provider=provider,
                gemini_api_key=gemini_api_key,
            )
        else:
            self.llm_judge = None

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
