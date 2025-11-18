# Custom Research - Quick Start Guide

## What This Does

You enter your **own research question** → AI agents autonomously:
1. Review relevant literature
2. Generate testable hypotheses
3. Design experiments
4. Analyze results
5. Write a complete research paper

**All in 5-10 minutes.**

---

## Two Ways to Use

### Option 1: Interactive (Easiest)

Just run the script and answer questions:

```bash
python examples/solve_custom_research.py
```

**You'll be asked:**
- What's your research topic?
- Which domain (materials/climate/protein/physics)?
- What's the background context?
- What's the specific objective?
- What metrics define success?
- Any key papers to reference? (optional)

**Example Session:**
```
What is your research topic?
> Novel catalysts for ammonia synthesis at low pressure

Which domain does this belong to?
> materials_science

Provide background context:
> Current ammonia production (Haber-Bosch) requires 150-250 bar pressure
> and 400-500°C, consuming 2% of global energy. Need catalysts that work
> at lower pressure to reduce energy consumption.
>

What is the specific research objective?
> Design catalyst with >80% ammonia yield at <10 bar pressure and <300°C

What metrics will define success?
> ammonia yield percentage, operating pressure, operating temperature, catalyst stability

Do you have specific papers to reference?
> no

[AI agents start working...]
```

### Option 2: Batch from JSON File

Create `my_research.json`:
```json
{
  "topic": "Biodegradable ocean-cleanup materials",
  "domain": "materials_science",
  "context": "Ocean plastic pollution is 8M tons/year. Cleanup nets/booms add more plastic. Need materials that capture microplastics but biodegrade.",
  "objective": "Design mesh material that captures <1mm plastics, biodegrades in <6 months in seawater, strong enough for ocean currents",
  "success_metrics": [
    "particle capture size (mm)",
    "biodegradation time (months)",
    "tensile strength (MPa)",
    "cost per square meter"
  ],
  "key_papers": [
    "Chitosan-based hydrogels - biodegradable but weak",
    "Alginate fibers - biodegrades in 3-6 months, moderate strength",
    "Ocean Cleanup Project - uses HDPE (not biodegradable)"
  ]
}
```

Then run:
```bash
python examples/solve_custom_research.py --from-file my_research.json
```

**For multiple problems**, use a JSON array:
```json
[
  {"topic": "Problem 1", ...},
  {"topic": "Problem 2", ...},
  {"topic": "Problem 3", ...}
]
```

---

## What You Get Back

### 1. Research Scores
```
Scientific Rigor:   8.5/10  (methodology quality)
Novelty:            7.2/10  (how novel are the ideas)
Completeness:       9.0/10  (all phases done)
Collaboration:      8.8/10  (agents working together)
Feasibility:        8.0/10  (realistic experiments)
─────────────────────────
Total Score:        8.3/10
```

### 2. Research Progress Summary
```
• Literature sources reviewed: 4
• Hypotheses generated: 2
• Experiments designed: 3
• Data analyses completed: 2
• Paper draft: 1,847 characters
```

### 3. Complete Research Paper

**Example output:**
```markdown
# Novel Iron-Cobalt Nitride Catalysts for Low-Pressure Ammonia Synthesis

## Abstract
We propose a bimetallic iron-cobalt nitride (Fe3Co2N) catalyst with
surface nitrogen vacancies to enable ammonia synthesis at <10 bar
pressure and 250°C...

## Introduction
The Haber-Bosch process, while enabling global food production,
consumes 1-2% of world energy due to high-pressure requirements...

## Proposed Hypothesis
Hypothesis 1: Fe3Co2N with controlled nitrogen vacancies will
provide low-barrier N2 dissociation sites...

Hypothesis 2: Operating at 8 bar and 250°C will achieve >80%
conversion by optimizing thermodynamic equilibrium...

## Experimental Design

### Synthesis
1. Co-precipitate Fe and Co nitrates in ammonia solution
2. Calcine at 400°C under NH3 atmosphere
3. Reduce at 600°C under H2 to create N vacancies
4. Characterize via XRD, TEM, XPS

### Testing
1. Load 500mg catalyst in fixed-bed reactor
2. Feed N2:H2 = 1:3 at 8 bar, 250°C
3. Measure ammonia concentration via GC-MS
4. Test stability over 100 hours

### Controls
- Pure Fe3N catalyst (no Co)
- Commercial Ru/C catalyst
- No-catalyst baseline

## Expected Results
- Ammonia yield: 82-87% at 8 bar, 250°C
- Catalyst activity: >0.5 mol NH3/mol_cat/h
- Stability: <5% degradation over 100h
- Cost: ~$50/kg vs $500/kg for Ru catalysts

## Data Analysis Plan
Compare yields vs pressure (5-15 bar) and temperature (200-300°C).
Statistical significance via t-test (n=3 replicates, p<0.05).

## Significance
This approach could reduce energy consumption in ammonia production
by 40%, saving 0.8% of global energy usage...

## Future Work
- Optimize Co:Fe ratio
- Test with industrial feedstocks
- Scale up to pilot reactor
- Techno-economic analysis
```

**The paper is automatically saved** as `research_paper_[topic].md`

---

## Real-World Example

**Input:** "I want to find better materials for solar panels that don't use rare elements"

**What happens (behind the scenes):**

1. **Literature Agent** reviews:
   - Current silicon panels (20% efficiency, abundant)
   - Perovskite materials (25% efficiency, contains lead)
   - Organic photovoltaics (15% efficiency, cheap)

2. **Hypothesis Agent** proposes:
   - "Tin-based perovskites could replace lead while maintaining >22% efficiency"
   - "Hybrid organic-inorganic structure improves stability to >1 year"

3. **Experiment Agent** designs:
   - Synthesize Cs2SnI6 perovskite via solution processing
   - Fabricate test cells with TiO2 electron transport layer
   - Measure efficiency, stability, degradation under UV

4. **Analysis Agent** predicts:
   - Efficiency: 21-23% under AM1.5G illumination
   - Stability: 85% efficiency retention after 6 months
   - Cost: $0.40/Watt vs $0.50/Watt for silicon

5. **Paper Agent** writes full research paper with:
   - Abstract, Introduction, Methods, Results, Discussion
   - Citations to relevant literature
   - Figures and tables (described)
   - Future work suggestions

**Total time:** 7 minutes
**Cost:** ~$2-3 in API calls
**Output:** Publication-ready research paper draft

---

## Advanced Options

### Adjust Thoroughness

**Faster (less thorough):**
```bash
python examples/solve_custom_research.py --beam-width 4 --k-samples 2
```
- Explores 4 research paths instead of 8
- 2 options per agent instead of 4
- Costs ~$1, takes ~3 minutes

**More thorough:**
```bash
python examples/solve_custom_research.py --beam-width 16 --k-samples 8
```
- Explores 16 research paths
- 8 options per agent
- Costs ~$8-10, takes ~15 minutes
- Higher quality hypotheses

---

## What Makes a Good Research Question?

### ✅ Good Examples

1. **Specific + Measurable**
   - "Design enzyme that degrades PET plastic with kcat >1.0 s^-1 at 60°C"
   - Clear metrics: kcat, temperature

2. **Has Context**
   - "Current PETase enzymes are slow (kcat ~0.04 s^-1). Need faster variants for industrial recycling."
   - Background explains the gap

3. **Realistic Scope**
   - Not too broad: ❌ "Cure all cancers"
   - Not too narrow: ❌ "Measure the exact weight of protein XYZ"
   - Just right: ✅ "Design antibody targeting EGFR with Kd <1 nM"

### ❌ Avoid These

1. **Too Vague**
   - ❌ "Make better batteries"
   - ✅ "Solid electrolyte with >10^-2 S/cm conductivity and air stability"

2. **No Metrics**
   - ❌ "Improve solar panels"
   - ✅ "Perovskite solar cell with >25% efficiency and >2 year stability"

3. **Missing Context**
   - ❌ "Design new drug"
   - ✅ "Design kinase inhibitor for BRAF V600E mutation in melanoma"

---

## Cost & Performance

| Beam Width | Samples/Turn | Time | Cost | Quality |
|------------|--------------|------|------|---------|
| 4 | 2 | 3 min | $1 | Good |
| 8 | 4 | 7 min | $2-3 | Better (default) |
| 16 | 8 | 15 min | $8-10 | Best |

**Recommendation:** Start with defaults (beam=8, samples=4), only increase if results aren't good enough.

---

## Troubleshooting

### "Not enough context in results"
→ Provide more background in `context` field
→ Add relevant `key_papers`

### "Hypotheses aren't novel"
→ Add recent literature to show what's already known
→ Be specific about what's been tried and failed

### "Experimental designs are too vague"
→ Specify exact metrics in `success_metrics`
→ Mention available techniques in context

### "Cost too high"
→ Reduce `--beam-width` and `--k-samples`
→ Make question more specific (less exploration needed)

---

## Commercial Use Cases

### 1. Corporate R&D
**Scenario:** Chemical company wants new catalyst
**Input:** "Catalyst for CO2 to methanol with >70% selectivity at <200°C"
**Output:** 3-5 novel catalyst designs with synthesis procedures
**Value:** Saves 6-12 months of literature review and ideation
**Pricing:** $10/research problem (enterprise tier)

### 2. University Research
**Scenario:** PhD student needs research direction
**Input:** "Improve CRISPR specificity to reduce off-target effects"
**Output:** Novel guide RNA designs, experimental validation plan
**Value:** Kickstarts thesis research, generates preliminary data
**Pricing:** $0.10/research problem (academic tier)

### 3. Startup Validation
**Scenario:** Biotech startup validating drug target
**Input:** "Antibody against PD-L1 with improved tumor penetration"
**Output:** Protein engineering strategies, predicted performance
**Value:** Reduces investor pitch risk, validates scientific approach
**Pricing:** $1/research problem (startup tier)

### 4. Government Research
**Scenario:** DOE wants novel battery chemistries
**Input:** "Lithium-sulfur battery with >500 cycles, <10% capacity fade"
**Output:** Electrolyte/cathode designs, testing protocols
**Value:** Screens hundreds of ideas before expensive experiments
**Pricing:** Custom contract ($500K-5M)

---

## Tips for Best Results

1. **Be Specific About Metrics**
   - Instead of: "better performance"
   - Write: "tensile strength >50 MPa, degradation <6 months"

2. **Provide Recent Literature**
   - Especially papers from last 2-3 years
   - Include quantitative results (e.g., "kcat = 0.26 s^-1")

3. **Define Success Clearly**
   - What would make this a Nature/Science paper?
   - What's the minimum viable result?

4. **Realistic Scope**
   - Solvable in 1-2 experiments (not years)
   - Can be validated with standard techniques

5. **Domain Matters**
   - Physics: Use for materials prediction, simulations
   - Biology: Protein/enzyme design, drug targets
   - Materials: Novel compositions, synthesis routes
   - Climate: Carbon capture, renewables, efficiency

---

## Next Steps

1. **Try the example:**
   ```bash
   python examples/solve_custom_research.py --from-file examples/research_questions_template.json
   ```

2. **Run your own research:**
   ```bash
   python examples/solve_custom_research.py
   ```

3. **Scale up:** Once validated, train on 20+ problems to improve collaboration patterns

4. **Integrate:** Add to your research pipeline, connect to experiment automation

---

**Questions?** Check [RESEARCH_LAB_GUIDE.md](RESEARCH_LAB_GUIDE.md) for full documentation.
