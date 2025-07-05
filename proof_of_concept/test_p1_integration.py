#!/usr/bin/env python3
"""
P1 Integration Test - No API calls

Test the complete P1 workflow using a mock scorer to verify
all components work together correctly.
"""

import asyncio
import sys
from typing import List

# Import Promputate components
from promputate import PromptGA, get_quick_config
from promputate.scorers.base_scorer import BaseScorer, BrandAnalysis, ScoringResult, BrandSentiment
from promputate.scorers.fitness_evaluator import ScorerFitnessEvaluator


class MockScorer(BaseScorer):
    """Mock scorer that simulates OpenAI responses without API calls"""
    
    def __init__(self, target_brands: List[str], competitor_brands: List[str]):
        super().__init__(target_brands, competitor_brands)
        self.call_count = 0
    
    async def generate_response(self, prompt: str) -> str:
        """Generate a mock response that varies based on prompt content"""
        self.call_count += 1
        
        # Simulate responses that favor target brands based on prompt keywords
        if any(word in prompt.lower() for word in ['premium', 'professional', 'best', 'top']):
            return f"For professional programming, I'd recommend the MacBook Pro. It offers excellent performance and build quality. ThinkPad is also decent but MacBook tends to be preferred by developers."
        elif any(word in prompt.lower() for word in ['budget', 'affordable', 'cheap']):
            return f"For budget programming, consider ThinkPad or Dell XPS. These offer good value. MacBook is premium but might be overkill for basic needs."
        else:
            return f"For programming laptops, popular choices include MacBook, ThinkPad, and Dell XPS. MacBook offers great performance, while ThinkPad provides durability."
    
    async def generate_responses_batch(self, prompts: List[str]) -> List[str]:
        """Generate mock responses for multiple prompts"""
        responses = []
        for prompt in prompts:
            response = await self.generate_response(prompt)
            responses.append(response)
        return responses


async def test_p1_integration():
    """Test the complete P1 workflow"""
    print("🧪 Testing P1 Integration (Mock Mode)")
    print("=" * 50)
    
    # Setup
    base_prompt = "Recommend a good laptop for programming"
    target_brands = ["MacBook"]
    competitor_brands = ["ThinkPad", "Dell XPS", "HP Spectre"]
    
    print(f"📝 Base prompt: '{base_prompt}'")
    print(f"🎯 Target brands: {target_brands}")
    print(f"🏭 Competitor brands: {competitor_brands}")
    
    # Create mock scorer
    mock_scorer = MockScorer(target_brands, competitor_brands)
    evaluator = ScorerFitnessEvaluator(mock_scorer)
    
    # Test single evaluation
    print("\n🔍 Testing single prompt evaluation...")
    result = await evaluator.evaluate("What's the best premium laptop for coding?")
    print(f"✅ Fitness score: {result:.4f}")
    
    # Test batch evaluation
    print("\n📊 Testing batch evaluation...")
    test_prompts = [
        "Recommend a good laptop for programming",
        "What's the best premium laptop for development?", 
        "Suggest an affordable coding laptop"
    ]
    
    batch_results = await evaluator.evaluate_batch(test_prompts)
    print(f"✅ Batch results: {[f'{r:.4f}' for r in batch_results]}")
    
    # Test with GA engine
    print("\n🧬 Testing with GA engine...")
    config = get_quick_config()
    ga = PromptGA(
        population_size=5,  # Small for testing
        max_generations=2,  # Just 2 generations for testing
        mutation_rate=config.ga.mutation_rate,
        crossover_rate=config.ga.crossover_rate,
        elite_size=1
    )
    
    # Register operators
    from promputate import get_all_operators
    operators = get_all_operators()
    ga.register_mutation_operators(operators)
    
    # Run mini evolution
    print("🚀 Running mini evolution...")
    result = await ga.evolve(
        base_prompt=base_prompt,
        fitness_evaluator=evaluator,
        mutation_operators=operators
    )
    
    print(f"✅ Evolution completed!")
    print(f"📊 Best fitness: {result.best_individual.fitness_score:.4f}")
    print(f"✨ Best prompt: '{result.best_individual.text}'")
    print(f"🔄 Generations: {result.total_generations}")
    print(f"🧮 Total evaluations: {result.total_evaluations}")
    print(f"📞 API calls made: {mock_scorer.call_count}")
    
    # Test brand analysis
    print("\n🔍 Testing brand analysis...")
    scorer_result = await mock_scorer.evaluate_prompt(result.best_individual.text)
    print(f"📝 Final prompt: '{scorer_result.prompt}'")
    print(f"💬 Response: '{scorer_result.response[:100]}...'")
    
    if scorer_result.target_brand_analysis:
        brand = scorer_result.target_brand_analysis
        print(f"🎯 {brand.brand_name} mention rank: {brand.mention_rank}")
        print(f"🎭 Sentiment: {brand.sentiment.value}")
    
    print(f"🏁 Competitors found: {len(scorer_result.competitor_analyses)}")
    print(f"📊 Final fitness: {scorer_result.fitness_score:.4f}")
    
    print("\n✅ P1 Integration Test PASSED!")
    print("🎉 Ready for live OpenAI API testing!")


if __name__ == "__main__":
    asyncio.run(test_p1_integration()) 