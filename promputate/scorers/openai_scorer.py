"""
OpenAI API scorer implementation

This module provides the OpenAI-specific scorer for evaluating prompts
using GPT-4o-mini with rate limiting and error handling.
"""

import asyncio
import os
from typing import List, Optional
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base_scorer import BaseScorer

# Try to import OpenAI - graceful degradation if not available
try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    # Create dummy classes to avoid type errors
    class DummyOpenAI:
        class RateLimitError(Exception):
            pass
        class APITimeoutError(Exception):
            pass
        class APIConnectionError(Exception):
            pass
    
    class DummyAsyncOpenAI:
        def __init__(self, *args, **kwargs):
            pass
        
        @property
        def chat(self):
            class DummyChat:
                @property 
                def completions(self):
                    class DummyCompletions:
                        async def create(self, *args, **kwargs):
                            raise ImportError("OpenAI not available")
                    return DummyCompletions()
            return DummyChat()
    
    openai = DummyOpenAI()  # type: ignore
    AsyncOpenAI = DummyAsyncOpenAI  # type: ignore

logger = logging.getLogger(__name__)


class OpenAIScorer(BaseScorer):
    """
    OpenAI API scorer using GPT-4o-mini
    
    Provides async batch evaluation with rate limiting and error handling.
    """
    
    def __init__(self,
                 target_brands: List[str],
                 competitor_brands: List[str],
                 api_key: Optional[str] = None,
                 model: str = "gpt-4o-mini",
                 max_tokens: int = 500,
                 temperature: float = 0.7,
                 max_response_length: int = 1000,
                 timeout: float = 30.0,
                 max_concurrent: int = 5,
                 rate_limit_delay: float = 1.0):
        """
        Initialize OpenAI scorer
        
        Args:
            target_brands: List of brands to optimize for
            competitor_brands: List of competing brands to track
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
            model: OpenAI model to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            max_response_length: Maximum expected response length
            timeout: Request timeout in seconds
            max_concurrent: Maximum concurrent requests
            rate_limit_delay: Delay between requests to respect rate limits
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        
        super().__init__(target_brands, competitor_brands, max_response_length, timeout)
        
        # API configuration
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_concurrent = max_concurrent
        self.rate_limit_delay = rate_limit_delay
        
        # Initialize async client
        self.client = AsyncOpenAI(api_key=self.api_key)
        
        # Semaphore for rate limiting
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
        logger.info(f"Initialized OpenAI scorer with model {model}, "
                   f"max_concurrent={max_concurrent}, rate_limit_delay={rate_limit_delay}s")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError))
    )
    async def _make_api_request(self, prompt: str) -> str:
        """
        Make a single API request with retry logic
        
        Args:
            prompt: The input prompt
            
        Returns:
            The response text
        """
        async with self._semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    timeout=self.timeout
                )
                
                # Add rate limiting delay
                await asyncio.sleep(self.rate_limit_delay)
                
                return response.choices[0].message.content or ""
                
            except openai.RateLimitError as e:
                logger.warning(f"Rate limit hit: {e}")
                raise
            except openai.APITimeoutError as e:
                logger.warning(f"API timeout: {e}")
                raise
            except openai.APIConnectionError as e:
                logger.warning(f"API connection error: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected OpenAI API error: {e}")
                raise
    
    async def generate_response(self, prompt: str) -> str:
        """
        Generate a response for a single prompt
        
        Args:
            prompt: The input prompt
            
        Returns:
            The LLM response
        """
        try:
            response = await self._make_api_request(prompt)
            logger.debug(f"Generated response for prompt: {prompt[:50]}...")
            return response
        except Exception as e:
            logger.error(f"Failed to generate response for prompt '{prompt[:50]}...': {e}")
            return ""
    
    async def generate_responses_batch(self, prompts: List[str]) -> List[str]:
        """
        Generate responses for multiple prompts in batch
        
        Args:
            prompts: List of input prompts
            
        Returns:
            List of LLM responses
        """
        logger.info(f"Generating responses for {len(prompts)} prompts")
        
        # Create tasks for all prompts
        tasks = [self.generate_response(prompt) for prompt in prompts]
        
        # Execute all tasks concurrently with rate limiting
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        final_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Error generating response for prompt {i}: {response}")
                final_responses.append("")
            else:
                final_responses.append(response)
        
        logger.info(f"Generated {len(final_responses)} responses")
        return final_responses
    
    def get_usage_estimate(self, prompts: List[str]) -> dict:
        """
        Estimate token usage and cost for a list of prompts
        
        Args:
            prompts: List of prompts to estimate for
            
        Returns:
            Dictionary with usage estimates
        """
        # Rough token estimation (4 chars ≈ 1 token)
        total_input_chars = sum(len(prompt) for prompt in prompts)
        estimated_input_tokens = total_input_chars // 4
        estimated_output_tokens = len(prompts) * self.max_tokens
        total_tokens = estimated_input_tokens + estimated_output_tokens
        
        # GPT-4o-mini pricing (as of 2024)
        cost_per_1k_input = 0.00015  # $0.15 per 1K input tokens
        cost_per_1k_output = 0.0006  # $0.60 per 1K output tokens
        
        estimated_cost = (
            (estimated_input_tokens / 1000) * cost_per_1k_input +
            (estimated_output_tokens / 1000) * cost_per_1k_output
        )
        
        return {
            "prompts": len(prompts),
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost,
            "model": self.model
        }


# Convenience function for creating OpenAI scorer
def create_openai_scorer(target_brands: List[str], 
                        competitor_brands: Optional[List[str]] = None,
                        api_key: Optional[str] = None,
                        **kwargs) -> OpenAIScorer:
    """
    Create an OpenAI scorer with default competitor brands for laptop example
    
    Args:
        target_brands: List of brands to optimize for (e.g., ["MacBook"])
        competitor_brands: List of competing brands (defaults to laptop competitors)
        api_key: OpenAI API key
        **kwargs: Additional arguments for OpenAIScorer
        
    Returns:
        Configured OpenAIScorer instance
    """
    if competitor_brands is None:
        competitor_brands = [
            "ThinkPad", "Dell XPS", "HP Spectre", "Surface Laptop", 
            "Alienware", "ASUS", "Acer", "Lenovo", "Samsung"
        ]
    
    return OpenAIScorer(
        target_brands=target_brands,
        competitor_brands=competitor_brands,
        api_key=api_key,
        **kwargs
    ) 