"""
Fitness Evaluator adapter for GA integration

This module provides adapters that connect LLM scorers with the GA engine's
FitnessEvaluator interface.
"""

import asyncio
from typing import List, Optional
import logging

from ..mutate import FitnessEvaluator
from .base_scorer import BaseScorer

logger = logging.getLogger(__name__)


class ScorerFitnessEvaluator(FitnessEvaluator):
    """
    Adapter that connects any BaseScorer with the GA engine's FitnessEvaluator interface
    
    This allows the professional GA engine to use LLM scorers for fitness evaluation.
    """
    
    def __init__(self, scorer: BaseScorer):
        """
        Initialize the adapter
        
        Args:
            scorer: Any BaseScorer implementation (OpenAI, Claude, etc.)
        """
        self.scorer = scorer
        logger.info(f"Initialized fitness evaluator with {type(scorer).__name__}")
    
    async def evaluate(self, prompt: str) -> float:
        """
        Evaluate a single prompt and return fitness score
        
        Args:
            prompt: The prompt to evaluate
            
        Returns:
            Fitness score (0.0 to ~2.0, higher is better)
        """
        try:
            result = await self.scorer.evaluate_prompt(prompt)
            return result.fitness_score
        except Exception as e:
            logger.error(f"Error evaluating prompt '{prompt[:50]}...': {e}")
            return 0.0
    
    async def evaluate_batch(self, prompts: List[str]) -> List[float]:
        """
        Evaluate a batch of prompts and return fitness scores
        
        Args:
            prompts: List of prompts to evaluate
            
        Returns:
            List of fitness scores
        """
        try:
            results = await self.scorer.evaluate_prompts_batch(prompts)
            return [result.fitness_score for result in results]
        except Exception as e:
            logger.error(f"Error evaluating batch of {len(prompts)} prompts: {e}")
            return [0.0] * len(prompts)


# Convenience function to create evaluator with OpenAI
def create_openai_evaluator(target_brands: List[str], 
                           competitor_brands: Optional[List[str]] = None,
                           api_key: Optional[str] = None,
                           **scorer_kwargs) -> ScorerFitnessEvaluator:
    """
    Create a fitness evaluator using OpenAI scorer
    
    Args:
        target_brands: List of brands to optimize for (e.g., ["MacBook"])
        competitor_brands: List of competing brands
        api_key: OpenAI API key
        **scorer_kwargs: Additional arguments for OpenAIScorer
        
    Returns:
        ScorerFitnessEvaluator configured with OpenAI
    """
    from .openai_scorer import create_openai_scorer
    
    scorer = create_openai_scorer(
        target_brands=target_brands,
        competitor_brands=competitor_brands,
        api_key=api_key,
        **scorer_kwargs
    )
    
    return ScorerFitnessEvaluator(scorer) 