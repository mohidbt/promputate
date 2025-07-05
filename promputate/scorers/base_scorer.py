"""
Base scorer abstract class for LLM integration

This module provides the abstract base class for all LLM scorers,
defining the interface for brand analysis and fitness scoring.
"""

import re
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BrandSentiment(Enum):
    """Sentiment analysis for brand mentions"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


@dataclass
class BrandAnalysis:
    """Analysis of brand mentions in a response"""
    brand_name: str
    mention_rank: int  # 1-based position, 0 = not mentioned
    mention_count: int
    sentiment: BrandSentiment
    context_snippet: str
    confidence: float = 0.0


@dataclass
class ScoringResult:
    """Complete scoring result for a prompt-response pair"""
    prompt: str
    response: str
    response_length: int
    
    # Brand analysis
    target_brand_analysis: Optional[BrandAnalysis]
    competitor_analyses: List[BrandAnalysis]
    
    # Fitness metrics
    fitness_score: float
    brand_mention_rank: int  # 0 = not mentioned
    competition_penalty: float
    length_bonus: float
    sentiment_bonus: float
    
    # Raw data
    raw_response: str
    processing_time: float
    
    def __post_init__(self):
        """Calculate derived metrics"""
        if self.target_brand_analysis:
            self.brand_mention_rank = self.target_brand_analysis.mention_rank
        else:
            self.brand_mention_rank = 0


class BaseScorer(ABC):
    """
    Abstract base class for LLM scorers
    
    Provides the interface for evaluating prompts and computing fitness scores
    based on brand mention ranking and other metrics.
    """
    
    def __init__(self, 
                 target_brands: List[str],
                 competitor_brands: List[str],
                 max_response_length: int = 1000,
                 timeout: float = 30.0):
        """
        Initialize the scorer
        
        Args:
            target_brands: List of brands to optimize for (e.g., ["MacBook"])
            competitor_brands: List of competing brands to track
            max_response_length: Maximum expected response length
            timeout: Request timeout in seconds
        """
        self.target_brands = [brand.lower() for brand in target_brands]
        self.competitor_brands = [brand.lower() for brand in competitor_brands]
        self.max_response_length = max_response_length
        self.timeout = timeout
        
        # Compile regex patterns for efficient brand detection
        self._compile_brand_patterns()
        
        logger.info(f"Initialized scorer with {len(self.target_brands)} target brands, "
                   f"{len(self.competitor_brands)} competitors")
    
    def _compile_brand_patterns(self):
        """Compile regex patterns for brand detection"""
        self.target_patterns = []
        self.competitor_patterns = []
        
        # Create patterns for target brands
        for brand in self.target_brands:
            # Word boundary patterns to avoid partial matches
            pattern = re.compile(r'\b' + re.escape(brand) + r'\b', re.IGNORECASE)
            self.target_patterns.append((brand, pattern))
        
        # Create patterns for competitor brands
        for brand in self.competitor_brands:
            pattern = re.compile(r'\b' + re.escape(brand) + r'\b', re.IGNORECASE)
            self.competitor_patterns.append((brand, pattern))
    
    @abstractmethod
    async def generate_response(self, prompt: str) -> str:
        """
        Generate a response for the given prompt
        
        Args:
            prompt: The input prompt
            
        Returns:
            The LLM response
        """
        pass
    
    @abstractmethod
    async def generate_responses_batch(self, prompts: List[str]) -> List[str]:
        """
        Generate responses for multiple prompts in batch
        
        Args:
            prompts: List of input prompts
            
        Returns:
            List of LLM responses
        """
        pass
    
    def analyze_brand_mentions(self, response: str) -> Tuple[Optional[BrandAnalysis], List[BrandAnalysis]]:
        """
        Analyze brand mentions in the response
        
        Args:
            response: The LLM response to analyze
            
        Returns:
            Tuple of (target_brand_analysis, competitor_analyses)
        """
        response_lower = response.lower()
        words = response_lower.split()
        
        # Find all brand mentions with positions
        target_mentions = []
        competitor_mentions = []
        
        # Check target brands
        for brand, pattern in self.target_patterns:
            matches = list(pattern.finditer(response))
            if matches:
                # Find first mention position in words
                first_match = matches[0]
                word_position = self._get_word_position(response, first_match.start())
                
                # Extract context around the mention
                context = self._extract_context(response, first_match.start(), first_match.end())
                
                # Simple sentiment analysis
                sentiment = self._analyze_sentiment(context)
                
                analysis = BrandAnalysis(
                    brand_name=brand,
                    mention_rank=word_position,
                    mention_count=len(matches),
                    sentiment=sentiment,
                    context_snippet=context,
                    confidence=0.8  # Default confidence
                )
                target_mentions.append(analysis)
        
        # Check competitor brands
        for brand, pattern in self.competitor_patterns:
            matches = list(pattern.finditer(response))
            if matches:
                first_match = matches[0]
                word_position = self._get_word_position(response, first_match.start())
                context = self._extract_context(response, first_match.start(), first_match.end())
                sentiment = self._analyze_sentiment(context)
                
                analysis = BrandAnalysis(
                    brand_name=brand,
                    mention_rank=word_position,
                    mention_count=len(matches),
                    sentiment=sentiment,
                    context_snippet=context,
                    confidence=0.8
                )
                competitor_mentions.append(analysis)
        
        # Return the first (highest ranked) target brand analysis
        target_analysis = min(target_mentions, key=lambda x: x.mention_rank) if target_mentions else None
        
        return target_analysis, competitor_mentions
    
    def _get_word_position(self, text: str, char_position: int) -> int:
        """Get the word position (1-based) for a character position"""
        words_before = text[:char_position].split()
        return len(words_before) + 1
    
    def _extract_context(self, text: str, start_pos: int, end_pos: int, window: int = 50) -> str:
        """Extract context around a brand mention"""
        context_start = max(0, start_pos - window)
        context_end = min(len(text), end_pos + window)
        return text[context_start:context_end].strip()
    
    def _analyze_sentiment(self, context: str) -> BrandSentiment:
        """Simple sentiment analysis based on keywords"""
        context_lower = context.lower()
        
        positive_words = ['best', 'excellent', 'great', 'amazing', 'perfect', 'outstanding', 
                         'superior', 'premium', 'top', 'recommend', 'love', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'poor', 'worst', 'hate', 'horrible',
                         'disappointing', 'overpriced', 'expensive', 'limited', 'lacking']
        
        positive_count = sum(1 for word in positive_words if word in context_lower)
        negative_count = sum(1 for word in negative_words if word in context_lower)
        
        if positive_count > negative_count:
            return BrandSentiment.POSITIVE
        elif negative_count > positive_count:
            return BrandSentiment.NEGATIVE
        else:
            return BrandSentiment.NEUTRAL
    
    def calculate_fitness_score(self, 
                              target_analysis: Optional[BrandAnalysis],
                              competitor_analyses: List[BrandAnalysis],
                              response_length: int) -> Tuple[float, float, float, float]:
        """
        Calculate fitness score based on brand analysis
        
        Args:
            target_analysis: Analysis of target brand mentions
            competitor_analyses: List of competitor brand analyses
            response_length: Length of the response
            
        Returns:
            Tuple of (fitness_score, competition_penalty, length_bonus, sentiment_bonus)
        """
        # Base fitness from brand mention rank
        if target_analysis and target_analysis.mention_rank > 0:
            base_fitness = 1.0 / target_analysis.mention_rank
        else:
            base_fitness = 0.0
        
        # Competition penalty
        competition_penalty = len(competitor_analyses) * 0.1
        
        # Length bonus (prefer shorter responses)
        length_bonus = max(0, (self.max_response_length - response_length) / self.max_response_length * 0.2)
        
        # Sentiment bonus
        sentiment_bonus = 0.0
        if target_analysis:
            if target_analysis.sentiment == BrandSentiment.POSITIVE:
                sentiment_bonus = 0.2
            elif target_analysis.sentiment == BrandSentiment.NEGATIVE:
                sentiment_bonus = -0.1
        
        # Calculate final fitness
        fitness_score = base_fitness + length_bonus - competition_penalty + sentiment_bonus
        
        return fitness_score, competition_penalty, length_bonus, sentiment_bonus
    
    async def evaluate_prompt(self, prompt: str) -> ScoringResult:
        """
        Evaluate a single prompt and return complete scoring result
        
        Args:
            prompt: The prompt to evaluate
            
        Returns:
            ScoringResult with complete analysis
        """
        import time
        start_time = time.time()
        
        try:
            # Generate response
            response = await self.generate_response(prompt)
            
            # Analyze brand mentions
            target_analysis, competitor_analyses = self.analyze_brand_mentions(response)
            
            # Calculate fitness score
            fitness_score, competition_penalty, length_bonus, sentiment_bonus = \
                self.calculate_fitness_score(target_analysis, competitor_analyses, len(response))
            
            # Create result
            result = ScoringResult(
                prompt=prompt,
                response=response,
                response_length=len(response),
                target_brand_analysis=target_analysis,
                competitor_analyses=competitor_analyses,
                fitness_score=fitness_score,
                brand_mention_rank=target_analysis.mention_rank if target_analysis else 0,
                competition_penalty=competition_penalty,
                length_bonus=length_bonus,
                sentiment_bonus=sentiment_bonus,
                raw_response=response,
                processing_time=time.time() - start_time
            )
            
            logger.debug(f"Evaluated prompt: {prompt[:50]}... -> fitness: {fitness_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating prompt '{prompt[:50]}...': {e}")
            
            # Return a zero-fitness result
            return ScoringResult(
                prompt=prompt,
                response="",
                response_length=0,
                target_brand_analysis=None,
                competitor_analyses=[],
                fitness_score=0.0,
                brand_mention_rank=0,
                competition_penalty=0.0,
                length_bonus=0.0,
                sentiment_bonus=0.0,
                raw_response="",
                processing_time=time.time() - start_time
            )
    
    async def evaluate_prompts_batch(self, prompts: List[str]) -> List[ScoringResult]:
        """
        Evaluate multiple prompts in batch
        
        Args:
            prompts: List of prompts to evaluate
            
        Returns:
            List of ScoringResult objects
        """
        logger.info(f"Evaluating batch of {len(prompts)} prompts")
        
        # Use asyncio to evaluate all prompts concurrently
        tasks = [self.evaluate_prompt(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that occurred
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error evaluating prompt {i}: {result}")
                # Create a zero-fitness result for failed evaluations
                final_results.append(ScoringResult(
                    prompt=prompts[i],
                    response="",
                    response_length=0,
                    target_brand_analysis=None,
                    competitor_analyses=[],
                    fitness_score=0.0,
                    brand_mention_rank=0,
                    competition_penalty=0.0,
                    length_bonus=0.0,
                    sentiment_bonus=0.0,
                    raw_response="",
                    processing_time=0.0
                ))
            else:
                final_results.append(result)
        
        return final_results 