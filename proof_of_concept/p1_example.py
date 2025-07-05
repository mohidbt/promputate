#!/usr/bin/env python3
"""
P1 Example: Professional GA-driven Prompt Optimization

This example demonstrates how to use the Promputate library with OpenAI
to optimize prompts for brand mention ranking. It replicates the original
proof-of-concept functionality using the professional library structure.

Usage:
    python3 p1_example.py

Requirements:
    - OPENAI_API_KEY environment variable set
    - pip install openai tenacity
"""

import asyncio
import os
import sys
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Promputate components
try:
    from promputate import (
        PromptGA, get_all_operators, get_quick_config,
        PromptIndividual, EvolutionResult
    )
    from promputate.scorers import create_openai_evaluator
except ImportError as e:
    logger.error(f"Failed to import Promputate: {e}")
    logger.error("Make sure you've installed the package: pip install -e .")
    sys.exit(1)


class PromputateExperiment:
    """
    Main experiment class for running GA-based prompt optimization
    """
    
    def __init__(self, 
                 base_prompt: str,
                 target_brands: List[str],
                 competitor_brands: Optional[List[str]] = None,
                 api_key: Optional[str] = None):
        """
        Initialize the experiment
        
        Args:
            base_prompt: Starting prompt to optimize
            target_brands: Brands to optimize for (e.g., ["MacBook"])
            competitor_brands: Competing brands to track
            api_key: OpenAI API key
        """
        self.base_prompt = base_prompt
        self.target_brands = target_brands
        self.competitor_brands = competitor_brands or [
            "ThinkPad", "Dell XPS", "HP Spectre", "Surface Laptop", "Alienware"
        ]
        
        # Setup configuration
        self.config = get_quick_config()  # Small population for demo
        logger.info(f"Using configuration: {self.config.ga.population_size} population, "
                   f"{self.config.ga.max_generations} max generations")
        
        # Create GA engine
        self.ga = PromptGA(
            population_size=self.config.ga.population_size,
            max_generations=self.config.ga.max_generations,
            mutation_rate=self.config.ga.mutation_rate,
            crossover_rate=self.config.ga.crossover_rate,
            elite_size=self.config.ga.elite_size
        )
        
        # Register mutation operators
        operators = get_all_operators()
        self.ga.register_mutation_operators(operators)
        logger.info(f"Registered {len(operators)} mutation operators")
        
        # Create fitness evaluator with OpenAI
        try:
            self.evaluator = create_openai_evaluator(
                target_brands=self.target_brands,
                competitor_brands=self.competitor_brands,
                api_key=api_key,
                model="gpt-4o-mini",
                max_tokens=300,
                temperature=0.7,
                max_concurrent=3,  # Conservative rate limiting
                rate_limit_delay=1.5
            )
            logger.info("OpenAI evaluator created successfully")
        except Exception as e:
            logger.error(f"Failed to create OpenAI evaluator: {e}")
            raise
    
    async def run_evolution(self) -> Optional[EvolutionResult]:
        """
        Run the complete genetic algorithm evolution
        
        Returns:
            EvolutionResult with best prompt and statistics
        """
        logger.info("="*60)
        logger.info("🧬 Starting Genetic Algorithm Evolution")
        logger.info("="*60)
        logger.info(f"📝 Base prompt: '{self.base_prompt}'")
        logger.info(f"🎯 Target brands: {self.target_brands}")
        logger.info(f"🏭 Competitor brands: {self.competitor_brands}")
        
        # Estimate cost
        initial_population = self.ga.create_initial_population(self.base_prompt)
        # Type cast to access OpenAI-specific methods
        from promputate.scorers.openai_scorer import OpenAIScorer
        if isinstance(self.evaluator.scorer, OpenAIScorer):
            cost_estimate = self.evaluator.scorer.get_usage_estimate(
                [ind[0] for ind in initial_population]
            )
        else:
            cost_estimate = {"estimated_cost_usd": 0.0}
        logger.info(f"💰 Estimated cost: ${cost_estimate['estimated_cost_usd']:.3f} USD")
        
        # Ask for confirmation
        if cost_estimate['estimated_cost_usd'] > 0.50:
            response = input(f"💰 Estimated cost is ${cost_estimate['estimated_cost_usd']:.3f}. Continue? (y/N): ")
            if response.lower() != 'y':
                logger.info("🛑 Evolution cancelled by user")
                return None
        
        # Run evolution
        logger.info("🚀 Starting evolution...")
        result = await self.ga.evolve(
            base_prompt=self.base_prompt,
            fitness_evaluator=self.evaluator,
            mutation_operators=get_all_operators()
        )
        
        # Log results
        logger.info("="*60)
        logger.info("🏆 Evolution Complete!")
        logger.info("="*60)
        logger.info(f"✨ Best prompt: '{result.best_individual.text}'")
        logger.info(f"📊 Best fitness: {result.best_individual.fitness_score:.4f}")
        logger.info(f"🔄 Total generations: {result.total_generations}")
        logger.info(f"🧮 Total evaluations: {result.total_evaluations}")
        
        return result
    
    def print_evolution_summary(self, result: EvolutionResult):
        """Print a detailed summary of the evolution results"""
        if not result:
            return
        
        print("\n" + "="*80)
        print("📈 EVOLUTION SUMMARY")
        print("="*80)
        
        print(f"🎯 Target Optimization: {self.target_brands[0]} first-mention ranking")
        print(f"📝 Original Prompt: '{self.base_prompt}'")
        print(f"✨ Best Evolved Prompt: '{result.best_individual.text}'")
        print(f"📊 Final Fitness Score: {result.best_individual.fitness_score:.4f}")
        
        print(f"\n📈 Evolution Progress:")
        for i, fitness in enumerate(result.best_fitness_history):
            print(f"   Generation {i+1}: {fitness:.4f}")
        
        print(f"\n🔢 Statistics:")
        print(f"   • Total Generations: {result.total_generations}")
        print(f"   • Total Evaluations: {result.total_evaluations}")
        print(f"   • Population Diversity: {result.population_diversity[-1]:.2f}" if result.population_diversity else "N/A")
        
        if result.best_fitness_history:
            improvement = result.best_fitness_history[-1] - result.best_fitness_history[0]
            print(f"   • Fitness Improvement: {improvement:+.4f}")
        
        print("="*80)


async def main():
    """Main function to run the P1 example"""
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("❌ OPENAI_API_KEY environment variable not set")
        logger.error("   Please set your OpenAI API key:")
        logger.error("   export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    # Setup experiment
    experiment = PromputateExperiment(
        base_prompt="Recommend a good laptop for programming",
        target_brands=["MacBook"],
        competitor_brands=["ThinkPad", "Dell XPS", "HP Spectre", "Surface Laptop", "Alienware"]
    )
    
    try:
        # Run evolution
        result = await experiment.run_evolution()
        
        if result:
            # Print detailed summary
            experiment.print_evolution_summary(result)
            
            # Test the best prompt
            logger.info("\n🧪 Testing best evolved prompt...")
            best_result = await experiment.evaluator.scorer.evaluate_prompt(result.best_individual.text)
            
            print(f"\n🧪 TEST RESULTS:")
            print(f"   📝 Prompt: '{best_result.prompt}'")
            print(f"   💬 Response: '{best_result.response[:200]}...'")
            
            if best_result.target_brand_analysis:
                brand_analysis = best_result.target_brand_analysis
                print(f"   🎯 {brand_analysis.brand_name} rank: {brand_analysis.mention_rank}")
                print(f"   🎭 Sentiment: {brand_analysis.sentiment.value}")
                print(f"   📖 Context: '{brand_analysis.context_snippet}'")
            else:
                print(f"   ❌ Target brand not mentioned")
            
            print(f"   🏁 Competitors mentioned: {len(best_result.competitor_analyses)}")
            print(f"   📊 Final fitness: {best_result.fitness_score:.4f}")
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Evolution interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error during evolution: {e}")
        raise


if __name__ == "__main__":
    print("🧬 Promputate P1 Example - GA-driven Prompt Optimization")
    print("=" * 60)
    
    # Run the async main function
    asyncio.run(main()) 