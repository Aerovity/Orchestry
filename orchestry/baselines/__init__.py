"""Baseline methods for comparison with MAGRPO."""

from orchestry.baselines.discussion import OneRoundDiscussionBaseline
from orchestry.baselines.fixed_model import FixedModelBaseline
from orchestry.baselines.naive_concat import NaiveConcatenationBaseline
from orchestry.baselines.sequential import SequentialPipelineBaseline

__all__ = [
    "FixedModelBaseline",
    "NaiveConcatenationBaseline",
    "OneRoundDiscussionBaseline",
    "SequentialPipelineBaseline",
]
