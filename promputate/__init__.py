"""
Promputate: Genetic Algorithm for Prompt Optimization

A library for optimizing prompts using genetic algorithms to influence LLM responses.
"""

__version__ = "0.1.0"
__author__ = "Promputate Team"
__description__ = "Genetic Algorithm for Prompt Optimization"

# Core GA engine
from .mutate import (
    PromptGA,
    PromptIndividual,
    EvolutionResult,
    FitnessEvaluator,
    setup_ga_statistics,
    create_logbook
)

# Mutation operators
from .operators import (
    synonym_replace,
    chunk_shuffle,
    modifier_toggle,
    prompt_reorder,
    intensity_modifier,
    get_all_operators,
    get_operator_by_name,
    operator_registry
)

# Configuration management
from .config import (
    GAConfig,
    OperatorConfig,
    FitnessConfig,
    PromputateConfig,
    get_quick_config,
    get_thorough_config,
    get_balanced_config,
    validate_config_file,
    create_default_config,
    DEFAULT_CONFIG
)

# Convenience imports for common use cases
__all__ = [
    # Core engine
    'PromptGA',
    'PromptIndividual',
    'EvolutionResult',
    'FitnessEvaluator',
    
    # Operators
    'synonym_replace',
    'chunk_shuffle',
    'modifier_toggle',
    'prompt_reorder',
    'intensity_modifier',
    'get_all_operators',
    'get_operator_by_name',
    'operator_registry',
    
    # Configuration
    'GAConfig',
    'OperatorConfig',
    'FitnessConfig',
    'PromputateConfig',
    'get_quick_config',
    'get_thorough_config',
    'get_balanced_config',
    'validate_config_file',
    'create_default_config',
    'DEFAULT_CONFIG',
    
    # Utilities
    'setup_ga_statistics',
    'create_logbook',
]

# Version information
__version_info__ = tuple(map(int, __version__.split('.')))

# Package metadata
__title__ = 'promputate'
__author_email__ = 'team@promputate.ai'
__license__ = 'MIT'
__url__ = 'https://github.com/promputate/promputate'
__download_url__ = 'https://pypi.org/project/promputate/'

# Development status
__status__ = 'Development'  # Development, Production, or Mature

# Quick start example in docstring
__doc__ = """
Promputate: Genetic Algorithm for Prompt Optimization

Quick Start Example:
    >>> from promputate import PromptGA, get_all_operators, get_quick_config
    >>> 
    >>> # Create a GA with quick configuration
    >>> config = get_quick_config()
    >>> ga = PromptGA(
    ...     population_size=config.ga.population_size,
    ...     max_generations=config.ga.max_generations
    ... )
    >>> 
    >>> # Register mutation operators
    >>> operators = get_all_operators()
    >>> ga.register_mutation_operators(operators)
    >>> 
    >>> # Create initial population
    >>> population = ga.create_initial_population("recommend a good laptop")
    >>> print(f"Created {len(population)} individuals")
    Created 10 individuals

For more detailed examples, see the documentation or examples directory.
"""
