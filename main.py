#!/usr/bin/env python3
"""
P2 Main App: Terminal-based GA Prompt Optimization

A simple terminal interface for the Promputate library that allows users
to configure and run genetic algorithm optimization for prompt engineering.

Usage:
    python3 main.py
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
        PromptGA, get_all_operators, get_quick_config, get_balanced_config, get_thorough_config,
        PromptIndividual, EvolutionResult
    )
    from promputate.scorers import create_openai_evaluator
except ImportError as e:
    logger.error(f"Failed to import Promputate: {e}")
    logger.error("Make sure you've installed the package: pip install -e .")
    sys.exit(1)


def get_user_input():
    """Get user input for configuration"""
    print("🧬 Welcome to Promputate - GA Prompt Optimization!")
    print("=" * 60)
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        api_key = input("Enter your OpenAI API key: ").strip()
        if not api_key:
            print("❌ API key is required")
            sys.exit(1)
    
    # Get base prompt
    print("\n📝 PROMPT CONFIGURATION")
    print("-" * 30)
    base_prompt = input("Enter your base prompt to optimize: ").strip()
    if not base_prompt:
        base_prompt = "Recommend a good laptop for programming"
        print(f"Using default: {base_prompt}")
    
    # Get target brand
    print("\n🎯 TARGET BRAND")
    print("-" * 30)
    target_brand = input("Enter target brand to optimize for: ").strip()
    if not target_brand:
        target_brand = "MacBook"
        print(f"Using default: {target_brand}")
    
    # Get competitor brands
    print("\n🏭 COMPETITOR BRANDS")
    print("-" * 30)
    print("Enter competitor brands separated by commas (or press Enter for defaults):")
    competitor_input = input("Competitors: ").strip()
    
    if competitor_input:
        competitor_brands = [brand.strip() for brand in competitor_input.split(",")]
    else:
        competitor_brands = ["ThinkPad", "Dell XPS", "HP Spectre", "Surface Laptop", "Alienware"]
        print(f"Using defaults: {', '.join(competitor_brands)}")
    
    # Get configuration level
    print("\n⚙️ CONFIGURATION LEVEL")
    print("-" * 30)
    print("1. Quick (10 pop, 5 gen) - Fast testing")
    print("2. Balanced (50 pop, 20 gen) - Good balance")
    print("3. Thorough (100 pop, 50 gen) - Best results")
    
    config_choice = input("Choose configuration (1-3) [default: 2]: ").strip()
    
    if config_choice == "1":
        config = get_quick_config()
        config_name = "Quick"
    elif config_choice == "3":
        config = get_thorough_config()
        config_name = "Thorough"
    else:
        config = get_balanced_config()
        config_name = "Balanced"
    
    print(f"Selected: {config_name} ({config.ga.population_size} pop, {config.ga.max_generations} gen)")
    
    return {
        'api_key': api_key,
        'base_prompt': base_prompt,
        'target_brand': target_brand,
        'competitor_brands': competitor_brands,
        'config': config,
        'config_name': config_name
    }


def print_summary(user_config: dict):
    """Print configuration summary"""
    print("\n" + "=" * 60)
    print("📋 CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"📝 Base Prompt: '{user_config['base_prompt']}'")
    print(f"🎯 Target Brand: {user_config['target_brand']}")
    print(f"🏭 Competitors: {', '.join(user_config['competitor_brands'])}")
    print(f"⚙️ Configuration: {user_config['config_name']}")
    print(f"   • Population Size: {user_config['config'].ga.population_size}")
    print(f"   • Max Generations: {user_config['config'].ga.max_generations}")
    print(f"   • Mutation Rate: {user_config['config'].ga.mutation_rate}")
    print("=" * 60)


async def run_optimization(user_config: dict):
    """Run the genetic algorithm optimization"""
    print("\n🚀 STARTING OPTIMIZATION")
    print("=" * 60)
    
    # Create GA engine
    ga = PromptGA(
        population_size=user_config['config'].ga.population_size,
        max_generations=user_config['config'].ga.max_generations,
        mutation_rate=user_config['config'].ga.mutation_rate,
        crossover_rate=user_config['config'].ga.crossover_rate,
        elite_size=user_config['config'].ga.elite_size
    )
    
    # Register mutation operators
    operators = get_all_operators()
    ga.register_mutation_operators(operators)
    print(f"✅ Registered {len(operators)} mutation operators")
    
    # Create fitness evaluator
    evaluator = create_openai_evaluator(
        target_brands=[user_config['target_brand']],
        competitor_brands=user_config['competitor_brands'],
        api_key=user_config['api_key'],
        model="gpt-4o-mini",
        max_tokens=300,
        temperature=0.7,
        max_concurrent=3,
        rate_limit_delay=1.5
    )
    print("✅ Created OpenAI evaluator")
    
    # Estimate cost
    initial_population = ga.create_initial_population(user_config['base_prompt'])
    from promputate.scorers.openai_scorer import OpenAIScorer
    if isinstance(evaluator.scorer, OpenAIScorer):
        cost_estimate = evaluator.scorer.get_usage_estimate(
            [ind[0] for ind in initial_population]
        )
        print(f"💰 Estimated cost: ${cost_estimate['estimated_cost_usd']:.3f} USD")
        
        # Ask for confirmation
        if cost_estimate['estimated_cost_usd'] > 0.10:
            confirm = input(f"💰 Proceed with estimated cost of ${cost_estimate['estimated_cost_usd']:.3f}? (y/N): ")
            if confirm.lower() != 'y':
                print("🛑 Optimization cancelled")
                return None
    
    # Run evolution
    print("\n🧬 Running genetic algorithm evolution...")
    result = await ga.evolve(
        base_prompt=user_config['base_prompt'],
        fitness_evaluator=evaluator,
        mutation_operators=operators
    )
    
    return result


def print_results(result: EvolutionResult, user_config: dict):
    """Print optimization results"""
    if not result:
        return
    
    print("\n" + "=" * 80)
    print("🏆 OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"🎯 Target: {user_config['target_brand']} optimization")
    print(f"📝 Original: '{user_config['base_prompt']}'")
    print(f"✨ Optimized: '{result.best_individual.text}'")
    print(f"📊 Final Fitness: {result.best_individual.fitness_score:.4f}")
    
    print(f"\n📈 Evolution Progress:")
    for i, fitness in enumerate(result.best_fitness_history):
        print(f"   Generation {i+1}: {fitness:.4f}")
    
    if result.best_fitness_history:
        improvement = result.best_fitness_history[-1] - result.best_fitness_history[0]
        print(f"\n📊 Statistics:")
        print(f"   • Total Generations: {result.total_generations}")
        print(f"   • Total Evaluations: {result.total_evaluations}")
        print(f"   • Fitness Improvement: {improvement:+.4f}")
        if result.population_diversity:
            print(f"   • Final Diversity: {result.population_diversity[-1]:.2f}")
    
    print("=" * 80)


async def main():
    """Main application function"""
    try:
        # Get user input
        user_config = get_user_input()
        
        # Print summary
        print_summary(user_config)
        
        # Final confirmation
        proceed = input("\n🚀 Start optimization? (Y/n): ")
        if proceed.lower() == 'n':
            print("👋 Goodbye!")
            return
        
        # Run optimization
        result = await run_optimization(user_config)
        
        # Print results
        if result:
            print_results(result, user_config)
            
            # Test the best prompt
            test_prompt = input("\n🧪 Test the best prompt with OpenAI? (Y/n): ")
            if test_prompt.lower() != 'n':
                print("🧪 Testing optimized prompt...")
                evaluator = create_openai_evaluator(
                    target_brands=[user_config['target_brand']],
                    competitor_brands=user_config['competitor_brands'],
                    api_key=user_config['api_key']
                )
                
                test_result = await evaluator.scorer.evaluate_prompt(result.best_individual.text)
                
                print(f"\n🧪 TEST RESULTS:")
                print(f"   📝 Prompt: '{test_result.prompt}'")
                print(f"   💬 Response: '{test_result.response[:200]}...'")
                
                if test_result.target_brand_analysis:
                    brand_analysis = test_result.target_brand_analysis
                    print(f"   🎯 {brand_analysis.brand_name} rank: {brand_analysis.mention_rank}")
                    print(f"   🎭 Sentiment: {brand_analysis.sentiment.value}")
                    print(f"   📖 Context: '{brand_analysis.context_snippet}'")
                else:
                    print(f"   ❌ Target brand not mentioned")
                
                print(f"   🏁 Competitors: {len(test_result.competitor_analyses)}")
                print(f"   📊 Fitness: {test_result.fitness_score:.4f}")
        
        print("\n✅ Optimization complete!")
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Application error: {e}")
        raise


if __name__ == "__main__":
    print("🧬 Promputate P2 - Terminal GA Prompt Optimizer")
    print("=" * 50)
    
    asyncio.run(main()) 