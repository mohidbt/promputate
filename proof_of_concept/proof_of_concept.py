#!/usr/bin/env python3
"""
Minimal Proof of Concept: Genetic Algorithm for Prompt Optimization
Goal: Optimize prompts to get "MacBook" mentioned first in LLM responses
"""

import random
import re
import asyncio
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from openai import AsyncOpenAI
import time

# Try to import NLTK
try:
    import nltk
    from nltk.corpus import wordnet
    NLTK_AVAILABLE = True
    
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not available. Synonym replacement will use predefined dictionary only.")

@dataclass
class PromptResult:
    """Result of testing a prompt variant"""
    prompt: str
    response: str
    macbook_rank: int  # Position where MacBook is first mentioned (1-based, 0 = not mentioned)
    fitness: float
    response_length: int
    brands_mentioned: List[str]
    sentiment_score: float = 0.0
    
class SimpleGA:
    """Simple Genetic Algorithm for prompt optimization"""
    
    def __init__(self, base_prompt: str, population_size: int = 20, mutation_rate: float = 0.3):
        self.base_prompt = base_prompt
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generation = 0
        self.best_fitness_history = []
        
        # Common laptop brands to track
        self.laptop_brands = ['MacBook', 'MacBook Pro', 'MacBook Air', 'ThinkPad', 'Dell XPS', 
                             'HP Spectre', 'Surface Laptop', 'Alienware', 'ASUS', 'Lenovo']
        
        # Synonyms for common words
        self.synonyms = {
            'good': ['excellent', 'great', 'outstanding', 'superior', 'quality', 'top', 'best'],
            'recommend': ['suggest', 'advise', 'propose', 'endorse', 'favor'],
            'laptop': ['notebook', 'computer', 'machine', 'device'],
            'programming': ['coding', 'development', 'software development', 'coding work'],
            'for': ['suitable for', 'ideal for', 'perfect for', 'designed for']
        }
        
        # Modifiers that can be added/removed
        self.modifiers = ['high-performance', 'reliable', 'professional', 'portable', 'powerful', 
                         'affordable', 'premium', 'lightweight', 'durable', 'versatile']
    
    def get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word using our dictionary and WordNet"""
        synonyms = set()
        
        # Use our predefined synonyms
        if word.lower() in self.synonyms:
            synonyms.update(self.synonyms[word.lower()])
        
        # Use WordNet if available
        if NLTK_AVAILABLE:
            try:
                for syn in wordnet.synsets(word):
                    if syn and hasattr(syn, 'lemmas'):
                        for lemma in syn.lemmas():
                            if lemma and hasattr(lemma, 'name'):
                                synonym = lemma.name().replace('_', ' ')
                                if synonym.lower() != word.lower():
                                    synonyms.add(synonym)
            except:
                pass
        
        return list(synonyms)[:5]  # Limit to 5 synonyms
    
    def mutate_synonym_replace(self, prompt: str) -> str:
        """Replace a random word with a synonym"""
        words = prompt.split()
        if not words:
            return prompt
        
        # Pick a random word to replace
        word_idx = random.randint(0, len(words) - 1)
        word = words[word_idx].strip('.,!?')
        
        synonyms = self.get_synonyms(word)
        if synonyms:
            words[word_idx] = random.choice(synonyms)
        
        return ' '.join(words)
    
    def mutate_add_modifier(self, prompt: str) -> str:
        """Add a random modifier to the prompt"""
        modifier = random.choice(self.modifiers)
        
        # Insert modifier before "laptop" if it exists
        if 'laptop' in prompt.lower():
            return prompt.replace('laptop', f'{modifier} laptop', 1)
        else:
            return f"{prompt} with {modifier} features"
    
    def mutate_reorder(self, prompt: str) -> str:
        """Slightly reorder the prompt structure"""
        variations = [
            f"What's the best {prompt.lower().replace('recommend a good', '').strip()}?",
            f"Can you suggest a {prompt.lower().replace('recommend a good', '').strip()}?",
            f"I need a {prompt.lower().replace('recommend a good', '').strip()}. What do you recommend?",
            f"Looking for a {prompt.lower().replace('recommend a good', '').strip()}. Any suggestions?"
        ]
        return random.choice(variations)
    
    def mutate_prompt(self, prompt: str) -> str:
        """Apply random mutations to a prompt"""
        mutations = [
            self.mutate_synonym_replace,
            self.mutate_add_modifier,
            self.mutate_reorder
        ]
        
        # Apply 1-2 random mutations
        num_mutations = random.randint(1, 2)
        result = prompt
        
        for _ in range(num_mutations):
            mutation = random.choice(mutations)
            result = mutation(result)
        
        return result
    
    def create_initial_population(self) -> List[str]:
        """Create initial population of prompt variants"""
        population = [self.base_prompt]  # Always include the original
        
        for _ in range(self.population_size - 1):
            variant = self.mutate_prompt(self.base_prompt)
            population.append(variant)
        
        return population
    
    def calculate_fitness(self, result: PromptResult) -> float:
        """Calculate fitness score for a prompt result"""
        if result.macbook_rank == 0:
            # MacBook not mentioned
            return 0.0
        
        # Primary score: inverse of MacBook rank (first mention = highest score)
        rank_score = 1.0 / result.macbook_rank
        
        # Bonus for shorter responses (more concise)
        length_bonus = max(0, 1.0 - (result.response_length / 1000.0))
        
        # Bonus for mentioning fewer competing brands
        competing_brands = [b for b in result.brands_mentioned if 'macbook' not in b.lower()]
        competition_penalty = len(competing_brands) * 0.1
        
        total_fitness = rank_score + length_bonus * 0.2 - competition_penalty
        return max(0.0, total_fitness)
    
    def find_brand_mentions(self, text: str) -> Tuple[List[str], int]:
        """Find brand mentions and MacBook rank"""
        mentioned_brands = []
        macbook_rank = 0
        
        # Look for brand mentions in order
        text_lower = text.lower()
        
        for brand in self.laptop_brands:
            if brand.lower() in text_lower:
                mentioned_brands.append(brand)
                
                # Check if this is the first MacBook mention
                if macbook_rank == 0 and 'macbook' in brand.lower():
                    macbook_rank = len(mentioned_brands)
        
        return mentioned_brands, macbook_rank
    
    async def evaluate_prompt(self, client: AsyncOpenAI, prompt: str) -> PromptResult:
        """Evaluate a single prompt using OpenAI API"""
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7
            )
            
            response_text = response.choices[0].message.content or ""
            brands_mentioned, macbook_rank = self.find_brand_mentions(response_text)
            
            result = PromptResult(
                prompt=prompt,
                response=response_text,
                macbook_rank=macbook_rank,
                fitness=0.0,  # Will be calculated
                response_length=len(response_text),
                brands_mentioned=brands_mentioned
            )
            
            result.fitness = self.calculate_fitness(result)
            return result
            
        except Exception as e:
            print(f"Error evaluating prompt: {e}")
            return PromptResult(
                prompt=prompt,
                response="Error",
                macbook_rank=0,
                fitness=0.0,
                response_length=0,
                brands_mentioned=[]
            )
    
    async def evaluate_population(self, client: AsyncOpenAI, population: List[str]) -> List[PromptResult]:
        """Evaluate entire population in parallel"""
        print(f"Evaluating {len(population)} prompts...")
        
        # Rate limiting: batch requests
        results = []
        batch_size = 5
        
        for i in range(0, len(population), batch_size):
            batch = population[i:i+batch_size]
            batch_results = await asyncio.gather(*[
                self.evaluate_prompt(client, prompt) for prompt in batch
            ])
            results.extend(batch_results)
            
            # Small delay between batches
            if i + batch_size < len(population):
                await asyncio.sleep(1)
        
        return results
    
    def select_parents(self, results: List[PromptResult], num_parents: int) -> List[PromptResult]:
        """Select best performers as parents"""
        # Sort by fitness (descending)
        sorted_results = sorted(results, key=lambda x: x.fitness, reverse=True)
        return sorted_results[:num_parents]
    
    def crossover_and_mutate(self, parents: List[PromptResult]) -> List[str]:
        """Create next generation through crossover and mutation"""
        new_population = []
        
        # Keep best performers
        for parent in parents[:self.population_size // 4]:
            new_population.append(parent.prompt)
        
        # Create offspring
        while len(new_population) < self.population_size:
            # Select random parent
            parent = random.choice(parents)
            
            # Mutate parent's prompt
            if random.random() < self.mutation_rate:
                offspring = self.mutate_prompt(parent.prompt)
            else:
                offspring = parent.prompt
            
            new_population.append(offspring)
        
        return new_population
    
    async def evolve(self, client: AsyncOpenAI, generations: int = 5) -> List[PromptResult]:
        """Run the genetic algorithm evolution"""
        print(f"Starting evolution with {self.population_size} prompts for {generations} generations")
        
        # Create initial population
        population = self.create_initial_population()
        print(f"Initial population created: {len(population)} variants")
        
        best_results = []
        
        for gen in range(generations):
            print(f"\n=== Generation {gen + 1} ===")
            
            # Evaluate current population
            results = await self.evaluate_population(client, population)
            
            # Track best result
            best_result = max(results, key=lambda x: x.fitness)
            best_results.append(best_result)
            self.best_fitness_history.append(best_result.fitness)
            
            print(f"Best fitness: {best_result.fitness:.3f}")
            print(f"Best prompt: {best_result.prompt}")
            print(f"MacBook rank: {best_result.macbook_rank}")
            print(f"Brands mentioned: {best_result.brands_mentioned}")
            
            # Select parents and create next generation
            if gen < generations - 1:  # Don't create new generation on last iteration
                parents = self.select_parents(results, self.population_size // 2)
                population = self.crossover_and_mutate(parents)
        
        return best_results

# Example usage
async def main():
    # Initialize OpenAI client (uses OPENAI_API_KEY environment variable)
    import os
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Please set your OPENAI_API_KEY environment variable:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    client = AsyncOpenAI(api_key=api_key)
    
    # Base prompt
    base_prompt = "Recommend a good laptop for programming"
    
    # Create genetic algorithm
    ga = SimpleGA(base_prompt, population_size=15, mutation_rate=0.4)
    
    # Run evolution
    best_results = await ga.evolve(client, generations=3)
    
    # Print final results
    print("\n" + "="*50)
    print("EVOLUTION COMPLETE")
    print("="*50)
    
    for i, result in enumerate(best_results):
        print(f"\nGeneration {i+1} Best:")
        print(f"Prompt: {result.prompt}")
        print(f"MacBook Rank: {result.macbook_rank}")
        print(f"Fitness: {result.fitness:.3f}")
        print(f"Response: {result.response[:200]}...")

if __name__ == "__main__":
    asyncio.run(main()) 