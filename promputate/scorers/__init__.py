"""
Scorers package for LLM integration

This package provides adapters for various LLM APIs to evaluate prompt fitness.
"""

from .base_scorer import BaseScorer, BrandAnalysis, ScoringResult
from .openai_scorer import OpenAIScorer, create_openai_scorer
from .fitness_evaluator import ScorerFitnessEvaluator, create_openai_evaluator

__all__ = [
    'BaseScorer',
    'BrandAnalysis', 
    'ScoringResult',
    'OpenAIScorer',
    'create_openai_scorer',
    'ScorerFitnessEvaluator',
    'create_openai_evaluator',
]
