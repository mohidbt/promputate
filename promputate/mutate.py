"""
Core Genetic Algorithm Engine for Prompt Optimization

This module provides the main GA engine using DEAP for evolving prompts
to optimize for specific objectives (like brand mention ranking).
"""

import random
import logging
from typing import List, Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from deap import base, creator, tools, algorithms
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PromptIndividual:
    """Represents an individual prompt in the GA population"""
    text: str
    fitness_score: float = 0.0
    generation: int = 0
    parent_id: Optional[str] = None
    mutation_history: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # Generate unique ID for tracking
        self.id = f"gen{self.generation}_{hash(self.text) % 10000:04d}"


@dataclass
class EvolutionResult:
    """Results from a complete GA evolution run"""
    best_individual: PromptIndividual
    best_fitness_history: List[float]
    population_diversity: List[float]
    total_generations: int
    total_evaluations: int
    convergence_generation: Optional[int] = None


class FitnessEvaluator(ABC):
    """Abstract base class for fitness evaluation"""
    
    @abstractmethod
    async def evaluate(self, prompt: str) -> float:
        """Evaluate a single prompt and return fitness score"""
        pass
    
    @abstractmethod
    async def evaluate_batch(self, prompts: List[str]) -> List[float]:
        """Evaluate a batch of prompts and return fitness scores"""
        pass


class PromptGA:
    """
    Main Genetic Algorithm class for prompt optimization
    
    Uses DEAP framework for professional GA implementation with:
    - Configurable operators and parameters
    - Population diversity tracking
    - Convergence detection
    - Detailed evolution statistics
    """
    
    def __init__(self, 
                 population_size: int = 50,
                 max_generations: int = 100,
                 mutation_rate: float = 0.3,
                 crossover_rate: float = 0.7,
                 selection_pressure: float = 1.5,
                 elite_size: int = 2,
                 convergence_threshold: float = 0.001,
                 convergence_generations: int = 10):
        """
        Initialize the Genetic Algorithm
        
        Args:
            population_size: Number of individuals in each generation
            max_generations: Maximum number of generations to evolve
            mutation_rate: Probability of mutation for each individual
            crossover_rate: Probability of crossover between parents
            selection_pressure: Selection pressure for tournament selection
            elite_size: Number of best individuals to preserve each generation
            convergence_threshold: Fitness improvement threshold for convergence
            convergence_generations: Generations without improvement to declare convergence
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_pressure = selection_pressure
        self.elite_size = elite_size
        self.convergence_threshold = convergence_threshold
        self.convergence_generations = convergence_generations
        
        # Evolution tracking
        self.current_generation = 0
        self.best_fitness_history = []
        self.diversity_history = []
        self.total_evaluations = 0
        
        # Mutation operators (will be set by operators module)
        self.mutation_operators = []
        
        # Setup DEAP framework
        self._setup_deap()
        
        logger.info(f"PromptGA initialized with population_size={population_size}, "
                   f"max_generations={max_generations}")
    
    def _setup_deap(self):
        """Setup DEAP genetic algorithm framework"""
        # Clear any existing creators
        if hasattr(creator, 'FitnessMax'):
            del creator.FitnessMax  # type: ignore
        if hasattr(creator, 'Individual'):
            del creator.Individual  # type: ignore
        
        # Create fitness function (maximization)
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)  # type: ignore
        
        # Create toolbox
        self.toolbox = base.Toolbox()
        
        # Selection: Tournament selection
        self.toolbox.register("select", tools.selTournament, 
                            tournsize=int(self.selection_pressure * 2))
        
        # Crossover: Will be set by operators
        self.toolbox.register("mate", self._crossover)
        
        # Mutation: Will be set by operators
        self.toolbox.register("mutate", self._mutate)
        
        # Evaluation: Will be set by user
        self.toolbox.register("evaluate", self._evaluate_individual)
        
        logger.info("DEAP framework configured successfully")
    
    def register_mutation_operators(self, operators: List[Callable[[str], str]]):
        """Register mutation operators for prompt modification"""
        self.mutation_operators = operators
        logger.info(f"Registered {len(operators)} mutation operators")
    
    def register_fitness_evaluator(self, evaluator: FitnessEvaluator):
        """Register fitness evaluator for prompt scoring"""
        self.fitness_evaluator = evaluator
        logger.info("Fitness evaluator registered")
    
    def _crossover(self, ind1: List[str], ind2: List[str]) -> Tuple[List[str], List[str]]:
        """
        Crossover operator for prompts with validation
        
        Combines two prompts by mixing their components
        """
        if len(ind1) == 0 or len(ind2) == 0:
            return ind1, ind2
        
        # Store originals for fallback
        original1 = ind1[0]
        original2 = ind2[0]
        
        # Simple single-point crossover on prompt text
        prompt1, prompt2 = ind1[0], ind2[0]
        
        # Split prompts into words
        words1 = prompt1.split()
        words2 = prompt2.split()
        
        if len(words1) > 1 and len(words2) > 1:
            # Random crossover points
            cp1 = random.randint(1, len(words1) - 1)
            cp2 = random.randint(1, len(words2) - 1)
            
            # Create offspring
            offspring1 = words1[:cp1] + words2[cp2:]
            offspring2 = words2[:cp2] + words1[cp1:]
            
            # Validate offspring
            candidate1 = " ".join(offspring1)
            candidate2 = " ".join(offspring2)
            
            # Only apply if both offspring are valid
            if (self._is_valid_mutation(original1, candidate1) and 
                self._is_valid_mutation(original2, candidate2)):
                ind1[0] = candidate1
                ind2[0] = candidate2
                logger.debug(f"Applied crossover: '{original1}' + '{original2}' -> '{candidate1}' + '{candidate2}'")
            else:
                logger.debug("Crossover failed validation, keeping originals")
        
        return ind1, ind2
    
    def _mutate(self, individual: List[str]) -> Tuple[List[str]]:
        """
        Mutation operator for prompts with quality validation
        
        Tries multiple operators and validates results to prevent broken text
        """
        if not self.mutation_operators or len(individual) == 0:
            return individual,
        
        original_text = individual[0]
        
        # Try up to 3 different operators to find a good mutation
        operators_to_try = self.mutation_operators.copy()
        random.shuffle(operators_to_try)
        
        for operator in operators_to_try[:3]:  # Try max 3 operators
            try:
                # Apply mutation
                mutated_prompt = operator(original_text)
                
                # Validate the result
                if self._is_valid_mutation(original_text, mutated_prompt):
                    individual[0] = mutated_prompt
                    logger.debug(f"Applied mutation: {operator.__name__}: '{original_text}' -> '{mutated_prompt}'")
                    return individual,
                else:
                    logger.warning(f"Mutation {operator.__name__} failed validation: '{original_text}' -> '{mutated_prompt}'")
                    
            except Exception as e:
                logger.warning(f"Mutation {operator.__name__} failed: {e}")
                continue
        
        # If all mutations failed validation, keep original
        logger.debug("All mutations failed validation, keeping original")
        return individual,
    
    def _is_valid_mutation(self, original: str, mutated: str) -> bool:
        """
        Validate that a mutation produces reasonable text
        
        Args:
            original: Original text
            mutated: Mutated text
            
        Returns:
            True if mutation is valid, False otherwise
        """
        if not mutated or not mutated.strip():
            return False
        
        words = mutated.split()
        
        # Check basic word count constraints
        if len(words) < 3 or len(words) > 30:
            return False
        
        # Check for duplicate consecutive words
        for i in range(len(words) - 1):
            if words[i].lower() == words[i + 1].lower():
                return False
        
        # Check for reasonable text structure
        if mutated.count('?') > 2 or mutated.count('.') > 3:
            return False
        
        # Check for incomplete sentences (ending with articles/conjunctions)
        if mutated.lower().endswith((' a ', ' an ', ' the ', ' and ', ' or ', ' but ')):
            return False
        
        # Check for inappropriate technical/medical terms
        technical_terms = ['antiophthalmic', 'factor', 'syndrome', 'pathology', 'diagnosis', 
                          'therapeutic', 'pharmaceutical', 'clinical', 'medical', 'anatomical']
        mutated_lower = mutated.lower()
        if any(term in mutated_lower for term in technical_terms):
            return False
        
        # Check for overly long or complex words (likely technical jargon)
        for word in words:
            clean_word = word.strip('.,!?;:"\'').lower()
            if len(clean_word) > 15 or clean_word.endswith(('ology', 'osis', 'itis', 'emia', 'ism', 'otic')):
                return False
        
        # Check for reasonable similarity to original (not completely different)
        original_words = set(original.lower().split())
        mutated_words = set(mutated.lower().split())
        overlap = len(original_words & mutated_words)
        
        # At least 25% word overlap to maintain meaning
        if overlap < min(3, len(original_words) * 0.25):
            return False
        
        # Check that core nouns are preserved (especially for tech prompts)
        core_nouns = {'laptop', 'computer', 'device', 'machine', 'notebook', 'system',
                      'phone', 'smartphone', 'tablet', 'programming', 'coding', 'development',
                      'underwear', 'clothing', 'garment', 'apparel', 'wear', 'attire', 'gear',
                      'socks', 'shoes', 'shirt', 'pants', 'jacket', 'coat'}
        original_core_nouns = original_words & core_nouns
        mutated_core_nouns = mutated_words & core_nouns
        
        # If original had core nouns, mutated should have at least one
        if original_core_nouns and not mutated_core_nouns:
            return False
        
        # Special check for critical words being completely removed
        critical_words = {'laptop', 'computer', 'programming', 'coding', 'development', 'underwear', 
                         'clothing', 'socks', 'shoes', 'shirt', 'pants', 'jacket'}
        for word in critical_words:
            if word in original_words and word not in mutated_words:
                # Allow if replaced by a synonym
                synonyms = {
                    'laptop': ['computer', 'notebook', 'machine', 'device', 'system'],
                    'computer': ['laptop', 'notebook', 'machine', 'device', 'system'],
                    'programming': ['coding', 'development', 'software'],
                    'coding': ['programming', 'development', 'software'],
                    'development': ['programming', 'coding', 'software'],
                    'underwear': ['underclothing', 'underclothes', 'undergarment', 'clothing', 'garment', 'apparel'],
                    'clothing': ['garment', 'apparel', 'wear', 'attire', 'gear'],
                    'socks': ['hosiery', 'footwear', 'clothing'],
                    'shoes': ['footwear', 'sneakers', 'boots'],
                    'shirt': ['top', 'blouse', 'clothing'],
                    'pants': ['trousers', 'bottoms', 'clothing'],
                    'jacket': ['coat', 'outerwear', 'clothing']
                }
                
                if word in synonyms:
                    has_synonym = any(syn in mutated_words for syn in synonyms[word])
                    if not has_synonym:
                        return False
        
        # Check for basic grammar coherence (no repeated question words, etc.)
        question_words = ['what', 'which', 'how', 'why', 'when', 'where', 'who']
        question_count = sum(1 for word in words if word.lower() in question_words)
        if question_count > 2:
            return False
        
        # Check for proper article placement (grammar check)
        for i, word in enumerate(words):
            word_lower = word.lower()
            
            # Articles should not appear at the end
            if word_lower in ['a', 'an', 'the'] and i == len(words) - 1:
                return False
            
            # Articles should not appear right before prepositions
            if (word_lower in ['a', 'an', 'the'] and i < len(words) - 1 and 
                words[i + 1].lower() in ['for', 'of', 'in', 'on', 'at', 'by', 'with']):
                return False
            
            # Check for nonsensical patterns like "laptop a for"
            if (i < len(words) - 2 and 
                words[i + 1].lower() in ['a', 'an'] and 
                words[i + 2].lower() in ['for', 'of', 'in', 'on', 'at', 'by', 'with']):
                return False
            
            # Check for broken patterns like "programming for a good laptop"
            if (word_lower == 'programming' and i < len(words) - 1 and 
                words[i + 1].lower() == 'for'):
                return False
            
            # Check for patterns like "laptop good for" (adjective without article)
            if (i < len(words) - 2 and 
                words[i].lower() in ['laptop', 'computer', 'device', 'machine'] and
                words[i + 1].lower() in ['good', 'great', 'excellent', 'quality'] and
                words[i + 2].lower() == 'for'):
                return False
        
        # Check for basic sentence structure (should start with verb/question word/pronoun)
        first_word = words[0].lower()
        valid_starters = ['recommend', 'suggest', 'advise', 'what', 'which', 'how', 'can', 'could', 
                         'would', 'should', 'i', 'help', 'find', 'looking', 'need', 'want', 'please']
        if first_word not in valid_starters:
            return False
        
        return True
    
    def _evaluate_individual(self, individual: List[str]) -> Tuple[float]:
        """Evaluate a single individual's fitness"""
        if not hasattr(self, 'fitness_evaluator'):
            raise ValueError("No fitness evaluator registered")
        
        # Note: This is a placeholder - actual evaluation should be async
        # For now, return random fitness for structure
        return (random.random(),)
    
    def create_initial_population(self, base_prompt: str, 
                                variation_strategies: Optional[List[str]] = None) -> List:
        """
        Create initial population of prompt variants
        
        Args:
            base_prompt: Starting prompt to create variations from
            variation_strategies: List of strategies for creating initial variants
            
        Returns:
            List of DEAP individuals representing the initial population
        """
        population = []
        
        # Always include the original prompt
        original = creator.Individual([base_prompt])  # type: ignore
        population.append(original)
        
        # Create variations using mutation operators
        valid_variants = 0
        max_attempts = (self.population_size - 1) * 5  # 5 attempts per needed variant
        
        for attempt in range(max_attempts):
            if valid_variants >= self.population_size - 1:
                break
                
            # Start with base prompt
            variant = creator.Individual([base_prompt])  # type: ignore
            
            # Apply only 1 mutation to prevent cascading failures
            original_text = variant[0]
            self._mutate(variant)
            
            # Verify the mutation was valid
            if variant[0] != original_text:  # Something was changed
                logger.debug(f"Created variant {valid_variants + 1}: '{variant[0]}'")
                # Double-check it's valid
                if self._is_valid_mutation(base_prompt, variant[0]):
                    population.append(variant)
                    valid_variants += 1
                else:
                    logger.warning(f"Initial population variant failed validation: '{variant[0]}'")
            else:
                # Mutation was rejected, add original
                population.append(variant)
                valid_variants += 1
        
        # If we didn't get enough valid variants, fill with originals
        while len(population) < self.population_size:
            population.append(creator.Individual([base_prompt]))  # type: ignore
        
        logger.info(f"Created initial population of {len(population)} individuals ({valid_variants} variants)")
        return population
    
    def calculate_population_diversity(self, population: List) -> float:
        """Calculate diversity of current population"""
        if len(population) < 2:
            return 0.0
        
        # Calculate average edit distance between all pairs
        total_distance = 0
        comparisons = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                # Simple character-based distance
                # Access the prompt text from the DEAP individual structure
                text1 = population[i][0] if isinstance(population[i][0], str) else population[i][0][0]
                text2 = population[j][0] if isinstance(population[j][0], str) else population[j][0][0]
                distance = len(set(text1.split()) ^ set(text2.split()))
                total_distance += distance
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0
    
    def check_convergence(self) -> bool:
        """Check if population has converged"""
        if len(self.best_fitness_history) < self.convergence_generations:
            return False
        
        # Check if fitness hasn't improved significantly
        recent_fitness = self.best_fitness_history[-self.convergence_generations:]
        fitness_range = max(recent_fitness) - min(recent_fitness)
        
        return fitness_range < self.convergence_threshold
    
    async def evolve(self, base_prompt: str, 
                    fitness_evaluator: FitnessEvaluator,
                    mutation_operators: List[Callable[[str], str]]) -> EvolutionResult:
        """
        Run the complete evolution process
        
        Args:
            base_prompt: Starting prompt for evolution
            fitness_evaluator: Evaluator for fitness scoring
            mutation_operators: List of mutation functions
            
        Returns:
            EvolutionResult with best individual and evolution statistics
        """
        # Setup
        self.register_fitness_evaluator(fitness_evaluator)
        self.register_mutation_operators(mutation_operators)
        
        # Create initial population
        population = self.create_initial_population(base_prompt)
        
        # Evaluate initial population
        fitnesses = await self._evaluate_population(population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = (fit,)
        
        self.total_evaluations += len(population)
        
        # Evolution loop
        for generation in range(self.max_generations):
            self.current_generation = generation
            
            # Track best fitness
            best_fitness = max(ind.fitness.values[0] for ind in population)
            self.best_fitness_history.append(best_fitness)
            
            # Track diversity
            diversity = self.calculate_population_diversity(population)
            self.diversity_history.append(diversity)
            
            logger.info(f"Generation {generation}: Best fitness = {best_fitness:.4f}, "
                       f"Diversity = {diversity:.2f}")
            
            # Check convergence
            if self.check_convergence():
                logger.info(f"Converged at generation {generation}")
                break
            
            # Selection
            offspring = self.toolbox.select(population, self.population_size - self.elite_size)  # type: ignore
            offspring = [self.toolbox.clone(ind) for ind in offspring]  # type: ignore
            
            # Crossover - DISABLED due to grammar issues
            # for i in range(0, len(offspring) - 1, 2):
            #     if random.random() < self.crossover_rate:
            #         self.toolbox.mate(offspring[i], offspring[i + 1])  # type: ignore
            #         del offspring[i].fitness.values
            #         del offspring[i + 1].fitness.values
            
            # Mutation
            for mutant in offspring:
                if random.random() < self.mutation_rate:
                    self.toolbox.mutate(mutant)  # type: ignore
                    del mutant.fitness.values
            
            # Evaluate offspring
            invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = await self._evaluate_population(invalid_individuals)
            for ind, fit in zip(invalid_individuals, fitnesses):
                ind.fitness.values = (fit,)
            
            self.total_evaluations += len(invalid_individuals)
            
            # Elite preservation
            population.sort(key=lambda x: x.fitness.values[0], reverse=True)
            elite = population[:self.elite_size]
            
            # Replace population
            population = elite + offspring
            
            # Ensure population size
            population = population[:self.population_size]
        
        # Find best individual
        best_individual = max(population, key=lambda x: x.fitness.values[0])
        
        # Create result
        result = EvolutionResult(
            best_individual=PromptIndividual(
                text=best_individual[0],
                fitness_score=best_individual.fitness.values[0],
                generation=self.current_generation
            ),
            best_fitness_history=self.best_fitness_history,
            population_diversity=self.diversity_history,
            total_generations=self.current_generation + 1,
            total_evaluations=self.total_evaluations
        )
        
        logger.info(f"Evolution complete. Best fitness: {result.best_individual.fitness_score:.4f}")
        return result
    
    async def _evaluate_population(self, population: List) -> List[float]:
        """Evaluate a population of individuals"""
        prompts = [ind[0] for ind in population]
        return await self.fitness_evaluator.evaluate_batch(prompts)


# Utility functions for GA operations
def setup_ga_statistics():  # type: ignore
    """Setup statistics tracking for GA"""
    stats = base.Statistics(lambda ind: ind.fitness.values)  # type: ignore
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("std", np.std)
    return stats


def create_logbook():  # type: ignore
    """Create logbook for evolution tracking"""
    logbook = base.Logbook()  # type: ignore
    logbook.header = "gen", "avg", "min", "max", "std"
    return logbook 