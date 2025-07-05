"""
Configuration Classes and Schemas for Promputate

This module provides configuration management for genetic algorithm parameters,
mutation operators, and other settings.
"""

import os
import json
import yaml
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class GAConfig:
    """Configuration for Genetic Algorithm parameters"""
    
    # Population settings
    population_size: int = 50
    max_generations: int = 100
    
    # Evolution parameters
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    selection_pressure: float = 1.5
    elite_size: int = 2
    
    # Convergence settings
    convergence_threshold: float = 0.001
    convergence_generations: int = 10
    early_stopping: bool = True
    
    # Logging and monitoring
    log_level: str = "INFO"
    track_diversity: bool = True
    save_population_history: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self.validate()
    
    def validate(self):
        """Validate configuration parameters"""
        if self.population_size < 4:
            raise ValueError("Population size must be at least 4")
        
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("Mutation rate must be between 0 and 1")
        
        if not 0 <= self.crossover_rate <= 1:
            raise ValueError("Crossover rate must be between 0 and 1")
        
        if self.elite_size >= self.population_size:
            raise ValueError("Elite size must be less than population size")
        
        if self.max_generations < 1:
            raise ValueError("Max generations must be at least 1")
        
        if self.convergence_threshold < 0:
            raise ValueError("Convergence threshold must be non-negative")
        
        logger.debug("GA configuration validated successfully")


@dataclass
class OperatorConfig:
    """Configuration for mutation operators"""
    
    # Operator weights (higher = more likely to be selected)
    synonym_replace_weight: float = 2.0
    chunk_shuffle_weight: float = 1.5
    modifier_toggle_weight: float = 2.0
    prompt_reorder_weight: float = 1.8
    intensity_modifier_weight: float = 1.0
    
    # Operator-specific settings
    max_synonym_replacements: int = 2
    max_modifiers: int = 1
    preserve_question_format: bool = True
    
    # Dependencies
    require_nltk: bool = True
    require_spacy: bool = True
    fallback_on_error: bool = True
    
    def __post_init__(self):
        """Validate operator configuration"""
        self.validate()
    
    def validate(self):
        """Validate operator configuration"""
        weights = [
            self.synonym_replace_weight,
            self.chunk_shuffle_weight,
            self.modifier_toggle_weight,
            self.prompt_reorder_weight,
            self.intensity_modifier_weight
        ]
        
        if any(w < 0 for w in weights):
            raise ValueError("All operator weights must be non-negative")
        
        if sum(weights) == 0:
            raise ValueError("At least one operator must have positive weight")
        
        if self.max_synonym_replacements < 0:
            raise ValueError("Max synonym replacements must be non-negative")
        
        if self.max_modifiers < 0:
            raise ValueError("Max modifiers must be non-negative")
        
        logger.debug("Operator configuration validated successfully")
    
    def get_operator_weights(self) -> Dict[str, float]:
        """Get operator weights as dictionary"""
        return {
            'synonym_replace': self.synonym_replace_weight,
            'chunk_shuffle': self.chunk_shuffle_weight,
            'modifier_toggle': self.modifier_toggle_weight,
            'prompt_reorder': self.prompt_reorder_weight,
            'intensity_modifier': self.intensity_modifier_weight,
        }


@dataclass
class FitnessConfig:
    """Configuration for fitness evaluation"""
    
    # Primary metrics
    brand_mention_weight: float = 1.0
    position_penalty_factor: float = 0.1
    
    # Secondary metrics
    length_bonus_weight: float = 0.2
    competition_penalty_weight: float = 0.1
    sentiment_bonus_weight: float = 0.1
    naturalness_weight: float = 0.05
    
    # Evaluation settings
    max_response_length: int = 1000
    min_response_length: int = 10
    timeout_seconds: float = 30.0
    
    # Target settings
    target_brands: List[str] = field(default_factory=lambda: ['MacBook'])
    competitor_brands: List[str] = field(default_factory=lambda: [
        'ThinkPad', 'Dell XPS', 'HP Spectre', 'Surface Laptop', 'Alienware'
    ])
    
    def __post_init__(self):
        """Validate fitness configuration"""
        self.validate()
    
    def validate(self):
        """Validate fitness configuration"""
        if self.brand_mention_weight < 0:
            raise ValueError("Brand mention weight must be non-negative")
        
        if self.max_response_length <= self.min_response_length:
            raise ValueError("Max response length must be greater than min response length")
        
        if self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")
        
        if not self.target_brands:
            raise ValueError("At least one target brand must be specified")
        
        logger.debug("Fitness configuration validated successfully")


@dataclass
class PromputateConfig:
    """Main configuration class for Promputate"""
    
    # Sub-configurations
    ga: GAConfig = field(default_factory=GAConfig)
    operators: OperatorConfig = field(default_factory=OperatorConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    
    # General settings
    random_seed: Optional[int] = None
    output_dir: str = "./promputate_output"
    save_results: bool = True
    
    # API settings (for future LLM integration)
    api_timeout: float = 30.0
    max_retries: int = 3
    rate_limit_delay: float = 1.0
    
    def __post_init__(self):
        """Validate main configuration"""
        self.validate()
        self._create_output_dir()
    
    def validate(self):
        """Validate main configuration"""
        if self.api_timeout <= 0:
            raise ValueError("API timeout must be positive")
        
        if self.max_retries < 0:
            raise ValueError("Max retries must be non-negative")
        
        if self.rate_limit_delay < 0:
            raise ValueError("Rate limit delay must be non-negative")
        
        # Validate sub-configurations
        self.ga.validate()
        self.operators.validate()
        self.fitness.validate()
        
        logger.debug("Main configuration validated successfully")
    
    def _create_output_dir(self):
        """Create output directory if it doesn't exist"""
        if self.save_results:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    def to_json(self, filepath: Optional[str] = None) -> str:
        """Export configuration to JSON"""
        json_str = json.dumps(self.to_dict(), indent=2)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
            logger.info(f"Configuration saved to {filepath}")
        
        return json_str
    
    def to_yaml(self, filepath: Optional[str] = None) -> str:
        """Export configuration to YAML"""
        yaml_str = yaml.dump(self.to_dict(), default_flow_style=False, indent=2)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(yaml_str)
            logger.info(f"Configuration saved to {filepath}")
        
        return yaml_str
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PromputateConfig':
        """Create configuration from dictionary"""
        # Handle nested configurations
        if 'ga' in config_dict:
            config_dict['ga'] = GAConfig(**config_dict['ga'])
        
        if 'operators' in config_dict:
            config_dict['operators'] = OperatorConfig(**config_dict['operators'])
        
        if 'fitness' in config_dict:
            config_dict['fitness'] = FitnessConfig(**config_dict['fitness'])
        
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'PromputateConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        logger.info(f"Configuration loaded from {filepath}")
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'PromputateConfig':
        """Load configuration from YAML file"""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from {filepath}")
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_env(cls, prefix: str = "PROMPUTATE_") -> 'PromputateConfig':
        """Load configuration from environment variables"""
        config_dict = {}
        
        # Map environment variables to config fields
        env_mappings = {
            f"{prefix}POPULATION_SIZE": ("ga", "population_size", int),
            f"{prefix}MAX_GENERATIONS": ("ga", "max_generations", int),
            f"{prefix}MUTATION_RATE": ("ga", "mutation_rate", float),
            f"{prefix}CROSSOVER_RATE": ("ga", "crossover_rate", float),
            f"{prefix}RANDOM_SEED": ("random_seed", int),
            f"{prefix}OUTPUT_DIR": ("output_dir", str),
            f"{prefix}API_TIMEOUT": ("api_timeout", float),
            f"{prefix}MAX_RETRIES": ("max_retries", int),
        }
        
        for env_var, (section, *field_info) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    # Handle nested configuration
                    if len(field_info) == 2:
                        field, type_func = field_info
                        if section not in config_dict:
                            config_dict[section] = {}
                        config_dict[section][field] = type_func(value)
                    else:
                        # Top-level configuration
                        type_func = field_info[0]
                        config_dict[section] = type_func(value)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid value for {env_var}: {value} ({e})")
        
        if config_dict:
            logger.info(f"Configuration loaded from environment variables")
            return cls.from_dict(config_dict)
        else:
            logger.info("No environment variables found, using default configuration")
            return cls()


# Predefined configurations for common use cases
def get_quick_config() -> PromputateConfig:
    """Get configuration for quick testing (small population, few generations)"""
    return PromputateConfig(
        ga=GAConfig(
            population_size=10,
            max_generations=5,
            mutation_rate=0.4,
            convergence_generations=3
        )
    )


def get_thorough_config() -> PromputateConfig:
    """Get configuration for thorough optimization (large population, many generations)"""
    return PromputateConfig(
        ga=GAConfig(
            population_size=100,
            max_generations=50,
            mutation_rate=0.25,
            crossover_rate=0.8,
            convergence_generations=15
        )
    )


def get_balanced_config() -> PromputateConfig:
    """Get balanced configuration (medium population, moderate generations)"""
    return PromputateConfig(
        ga=GAConfig(
            population_size=50,
            max_generations=20,
            mutation_rate=0.3,
            crossover_rate=0.7,
            convergence_generations=8
        )
    )


# Configuration validation utilities
def validate_config_file(filepath: str) -> bool:
    """Validate a configuration file without loading it"""
    try:
        if filepath.endswith('.json'):
            config = PromputateConfig.from_json(filepath)
        elif filepath.endswith(('.yaml', '.yml')):
            config = PromputateConfig.from_yaml(filepath)
        else:
            raise ValueError("Unsupported file format. Use .json or .yaml")
        
        logger.info(f"Configuration file {filepath} is valid")
        return True
    
    except Exception as e:
        logger.error(f"Configuration file {filepath} is invalid: {e}")
        return False


def create_default_config(filepath: str, format: str = 'yaml') -> None:
    """Create a default configuration file"""
    config = PromputateConfig()
    
    if format.lower() == 'json':
        config.to_json(filepath)
    elif format.lower() in ['yaml', 'yml']:
        config.to_yaml(filepath)
    else:
        raise ValueError("Format must be 'json' or 'yaml'")
    
    logger.info(f"Default configuration created at {filepath}")


# Default configuration instance
DEFAULT_CONFIG = PromputateConfig() 