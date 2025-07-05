"""
Test suite for Promputate library
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import List

# Import modules to test
from promputate.mutate import (
    PromptGA, PromptIndividual, EvolutionResult, FitnessEvaluator
)
from promputate.operators import (
    synonym_replace, chunk_shuffle, modifier_toggle, prompt_reorder,
    intensity_modifier, get_all_operators, operator_registry
)
from promputate.config import (
    GAConfig, OperatorConfig, FitnessConfig, PromputateConfig,
    get_quick_config, get_thorough_config, get_balanced_config
)


class TestPromputateStructure:
    """Test the basic package structure and imports"""
    
    def test_package_import(self):
        """Test that the main package can be imported"""
        import promputate
        assert hasattr(promputate, '__version__')
        assert hasattr(promputate, '__author__')
        assert hasattr(promputate, '__description__')
    
    def test_version_format(self):
        """Test that version follows semantic versioning"""
        import promputate
        version_parts = promputate.__version__.split('.')
        assert len(version_parts) == 3
        assert all(part.isdigit() for part in version_parts)


class TestMutateModule:
    """Test the mutate module - GA engine"""
    
    def test_prompt_individual_creation(self):
        """Test PromptIndividual creation and ID generation"""
        individual = PromptIndividual("test prompt", fitness_score=0.5, generation=1)
        assert individual.text == "test prompt"
        assert individual.fitness_score == 0.5
        assert individual.generation == 1
        assert individual.id.startswith("gen1_")
    
    def test_prompt_ga_initialization(self):
        """Test PromptGA initialization with default parameters"""
        ga = PromptGA()
        assert ga.population_size == 50
        assert ga.max_generations == 100
        assert ga.mutation_rate == 0.3
        assert ga.crossover_rate == 0.7
        assert ga.elite_size == 2
        assert ga.current_generation == 0
        assert len(ga.best_fitness_history) == 0
        assert len(ga.mutation_operators) == 0
    
    def test_prompt_ga_custom_params(self):
        """Test PromptGA with custom parameters"""
        ga = PromptGA(
            population_size=20,
            max_generations=10,
            mutation_rate=0.5,
            crossover_rate=0.8,
            elite_size=3
        )
        assert ga.population_size == 20
        assert ga.max_generations == 10
        assert ga.mutation_rate == 0.5
        assert ga.crossover_rate == 0.8
        assert ga.elite_size == 3
    
    def test_mutation_operators_registration(self):
        """Test mutation operator registration"""
        ga = PromptGA()
        operators = [lambda x: x + "_mutated", lambda x: x.upper()]
        ga.register_mutation_operators(operators)
        assert len(ga.mutation_operators) == 2
        assert ga.mutation_operators[0]("test") == "test_mutated"
        assert ga.mutation_operators[1]("test") == "TEST"
    
    def test_fitness_evaluator_registration(self):
        """Test fitness evaluator registration"""
        ga = PromptGA()
        
        class MockEvaluator(FitnessEvaluator):
            async def evaluate(self, prompt: str) -> float:
                return 0.5
            
            async def evaluate_batch(self, prompts: List[str]) -> List[float]:
                return [0.5] * len(prompts)
        
        evaluator = MockEvaluator()
        ga.register_fitness_evaluator(evaluator)
        assert hasattr(ga, 'fitness_evaluator')
        assert ga.fitness_evaluator == evaluator
    
    def test_population_diversity_calculation(self):
        """Test population diversity calculation"""
        ga = PromptGA()
        
        # Mock population with different prompts
        population = [
            [["hello world"]],
            [["goodbye world"]],
            [["hello universe"]]
        ]
        
        diversity = ga.calculate_population_diversity(population)
        assert diversity > 0  # Should have some diversity
        
        # Test with identical population
        identical_population = [
            [["same prompt"]],
            [["same prompt"]],
            [["same prompt"]]
        ]
        
        diversity_identical = ga.calculate_population_diversity(identical_population)
        assert diversity_identical == 0  # Should have no diversity
    
    def test_convergence_detection(self):
        """Test convergence detection logic"""
        ga = PromptGA(convergence_generations=3, convergence_threshold=0.01)
        
        # No convergence with insufficient history
        assert not ga.check_convergence()
        
        # No convergence with improving fitness
        ga.best_fitness_history = [0.1, 0.2, 0.3, 0.4]
        assert not ga.check_convergence()
        
        # Convergence with stable fitness
        ga.best_fitness_history = [0.5, 0.501, 0.502, 0.503]
        assert ga.check_convergence()
    
    def test_crossover_operator(self):
        """Test crossover operator"""
        ga = PromptGA()
        
        ind1 = ["hello world programming"]
        ind2 = ["goodbye coding universe"]
        
        # Test crossover
        result1, result2 = ga._crossover(ind1, ind2)
        
        # Results should be different from originals (most of the time)
        # We can't guarantee exact behavior due to randomness
        assert len(result1) == 1
        assert len(result2) == 1
        assert isinstance(result1[0], str)
        assert isinstance(result2[0], str)
    
    def test_mutation_operator(self):
        """Test mutation operator"""
        ga = PromptGA()
        
        # Register a simple mutation operator
        ga.register_mutation_operators([lambda x: x + "_mutated"])
        
        individual = ["test prompt"]
        result = ga._mutate(individual)
        
        assert result[0][0] == "test prompt_mutated"


class TestOperatorsModule:
    """Test the operators module - mutation functions"""
    
    def test_synonym_replace_basic(self):
        """Test basic synonym replacement"""
        prompt = "recommend a good laptop"
        result = synonym_replace(prompt, max_replacements=1)
        
        # Should be different from original (most of the time)
        # We can't guarantee exact behavior due to randomness
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_synonym_replace_empty_prompt(self):
        """Test synonym replacement with empty prompt"""
        result = synonym_replace("")
        assert result == ""
    
    def test_chunk_shuffle_basic(self):
        """Test basic chunk shuffling"""
        prompt = "I need a good laptop for programming"
        result = chunk_shuffle(prompt)
        
        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain same words (though potentially reordered)
        assert all(word in result for word in ["need", "good", "laptop", "programming"])
    
    def test_modifier_toggle_basic(self):
        """Test basic modifier toggle"""
        prompt = "recommend a laptop"
        result = modifier_toggle(prompt)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_prompt_reorder_question_to_statement(self):
        """Test converting question to statement"""
        prompt = "What's the best laptop for programming?"
        result = prompt_reorder(prompt)
        
        assert isinstance(result, str)
        assert len(result) > 0
        # Should no longer end with question mark
        assert not result.endswith("?")
    
    def test_prompt_reorder_statement_to_question(self):
        """Test converting statement to question"""
        prompt = "I need a good laptop for programming"
        result = prompt_reorder(prompt)
        
        assert isinstance(result, str)
        assert len(result) > 0
        # Should end with question mark or question pattern
        assert "?" in result or any(starter in result.lower() for starter in ["what", "can you", "which"])
    
    def test_intensity_modifier_basic(self):
        """Test intensity modifier"""
        prompt = "recommend a laptop"
        result = intensity_modifier(prompt)
        
        assert isinstance(result, str)
        assert len(result) > len(prompt)  # Should add words
    
    def test_operator_registry_functionality(self):
        """Test operator registry functionality"""
        # Test that operators are registered
        assert len(operator_registry.operators) > 0
        
        # Test getting operator by name
        synonym_op = operator_registry.get_operator("synonym_replace")
        assert synonym_op is not None
        assert callable(synonym_op)
        
        # Test getting random operator
        random_op = operator_registry.get_random_operator()
        assert random_op is not None
        assert callable(random_op)
        
        # Test listing operators
        operator_names = operator_registry.list_operators()
        assert "synonym_replace" in operator_names
        assert "chunk_shuffle" in operator_names
        assert "modifier_toggle" in operator_names
    
    def test_get_all_operators(self):
        """Test getting all operators"""
        operators = get_all_operators()
        assert len(operators) > 0
        assert all(callable(op) for op in operators)
    
    def test_operators_with_edge_cases(self):
        """Test operators with edge cases"""
        # Test with single word
        single_word = synonym_replace("laptop")
        assert isinstance(single_word, str)
        
        # Test with short prompt
        short_prompt = modifier_toggle("help")
        assert isinstance(short_prompt, str)
        
        # Test with punctuation
        punctuated = synonym_replace("What's the best laptop?")
        assert isinstance(punctuated, str)


class TestConfigModule:
    """Test the config module - configuration classes"""
    
    def test_ga_config_defaults(self):
        """Test GAConfig with default values"""
        config = GAConfig()
        assert config.population_size == 50
        assert config.max_generations == 100
        assert config.mutation_rate == 0.3
        assert config.crossover_rate == 0.7
        assert config.elite_size == 2
        assert config.early_stopping is True
    
    def test_ga_config_validation(self):
        """Test GAConfig validation"""
        # Valid config should not raise
        GAConfig(population_size=10, max_generations=5)
        
        # Invalid configs should raise
        with pytest.raises(ValueError):
            GAConfig(population_size=2)  # Too small
        
        with pytest.raises(ValueError):
            GAConfig(mutation_rate=1.5)  # Too high
        
        with pytest.raises(ValueError):
            GAConfig(elite_size=60, population_size=50)  # Elite too large
    
    def test_operator_config_defaults(self):
        """Test OperatorConfig with default values"""
        config = OperatorConfig()
        assert config.synonym_replace_weight == 2.0
        assert config.chunk_shuffle_weight == 1.5
        assert config.modifier_toggle_weight == 2.0
        assert config.max_synonym_replacements == 2
        assert config.max_modifiers == 1
    
    def test_operator_config_weights(self):
        """Test operator config weight retrieval"""
        config = OperatorConfig()
        weights = config.get_operator_weights()
        assert "synonym_replace" in weights
        assert "chunk_shuffle" in weights
        assert weights["synonym_replace"] == 2.0
        assert weights["chunk_shuffle"] == 1.5
    
    def test_operator_config_validation(self):
        """Test OperatorConfig validation"""
        # Valid config should not raise
        OperatorConfig(synonym_replace_weight=1.0)
        
        # Invalid configs should raise
        with pytest.raises(ValueError):
            OperatorConfig(synonym_replace_weight=-1.0)  # Negative weight
        
        with pytest.raises(ValueError):
            OperatorConfig(max_synonym_replacements=-1)  # Negative max
    
    def test_fitness_config_defaults(self):
        """Test FitnessConfig with default values"""
        config = FitnessConfig()
        assert config.brand_mention_weight == 1.0
        assert config.max_response_length == 1000
        assert config.min_response_length == 10
        assert "MacBook" in config.target_brands
        assert "ThinkPad" in config.competitor_brands
    
    def test_fitness_config_validation(self):
        """Test FitnessConfig validation"""
        # Valid config should not raise
        FitnessConfig(target_brands=["TestBrand"])
        
        # Invalid configs should raise
        with pytest.raises(ValueError):
            FitnessConfig(target_brands=[])  # Empty target brands
        
        with pytest.raises(ValueError):
            FitnessConfig(max_response_length=10, min_response_length=20)  # Invalid lengths
    
    def test_promputate_config_defaults(self):
        """Test PromputateConfig with default values"""
        config = PromputateConfig()
        assert isinstance(config.ga, GAConfig)
        assert isinstance(config.operators, OperatorConfig)
        assert isinstance(config.fitness, FitnessConfig)
        assert config.output_dir == "./promputate_output"
        assert config.save_results is True
    
    def test_promputate_config_validation(self):
        """Test PromputateConfig validation"""
        # Valid config should not raise
        PromputateConfig()
        
        # Invalid configs should raise
        with pytest.raises(ValueError):
            PromputateConfig(api_timeout=-1.0)  # Negative timeout
        
        with pytest.raises(ValueError):
            PromputateConfig(max_retries=-1)  # Negative retries
    
    def test_config_to_dict(self):
        """Test configuration to dictionary conversion"""
        config = PromputateConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "ga" in config_dict
        assert "operators" in config_dict
        assert "fitness" in config_dict
        assert isinstance(config_dict["ga"], dict)
    
    def test_config_to_json(self):
        """Test configuration to JSON conversion"""
        config = PromputateConfig()
        json_str = config.to_json()
        
        assert isinstance(json_str, str)
        assert "ga" in json_str
        assert "population_size" in json_str
        
        # Test JSON is valid
        import json
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
    
    def test_config_from_dict(self):
        """Test configuration from dictionary"""
        config_dict = {
            "ga": {"population_size": 30, "max_generations": 15},
            "operators": {"synonym_replace_weight": 3.0},
            "fitness": {"target_brands": ["TestBrand"]}
        }
        
        config = PromputateConfig.from_dict(config_dict)
        assert config.ga.population_size == 30
        assert config.ga.max_generations == 15
        assert config.operators.synonym_replace_weight == 3.0
        assert "TestBrand" in config.fitness.target_brands
    
    def test_predefined_configs(self):
        """Test predefined configuration functions"""
        # Test quick config
        quick = get_quick_config()
        assert quick.ga.population_size == 10
        assert quick.ga.max_generations == 5
        
        # Test thorough config
        thorough = get_thorough_config()
        assert thorough.ga.population_size == 100
        assert thorough.ga.max_generations == 50
        
        # Test balanced config
        balanced = get_balanced_config()
        assert balanced.ga.population_size == 50
        assert balanced.ga.max_generations == 20
    
    @patch.dict('os.environ', {'PROMPUTATE_POPULATION_SIZE': '25', 'PROMPUTATE_MAX_GENERATIONS': '15'})
    def test_config_from_env(self):
        """Test configuration from environment variables"""
        config = PromputateConfig.from_env()
        assert config.ga.population_size == 25
        assert config.ga.max_generations == 15


class TestIntegration:
    """Integration tests combining multiple components"""
    
    def test_ga_with_operators_and_config(self):
        """Test GA integration with operators and configuration"""
        # Create configuration
        config = get_quick_config()
        
        # Create GA with config
        ga = PromptGA(
            population_size=config.ga.population_size,
            max_generations=config.ga.max_generations,
            mutation_rate=config.ga.mutation_rate
        )
        
        # Register operators
        operators = get_all_operators()
        ga.register_mutation_operators(operators)
        
        # Create initial population
        population = ga.create_initial_population("test prompt")
        
        assert len(population) == config.ga.population_size
        assert all(len(ind) == 1 for ind in population)
        assert all(isinstance(ind[0], str) for ind in population)
    
    def test_operators_with_config(self):
        """Test operators with configuration parameters"""
        config = OperatorConfig(max_synonym_replacements=1, max_modifiers=2)
        
        # Test synonym replacement with config
        result = synonym_replace("test prompt", max_replacements=config.max_synonym_replacements)
        assert isinstance(result, str)
        
        # Test modifier toggle with config
        result = modifier_toggle("test prompt", max_modifiers=config.max_modifiers)
        assert isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__]) 