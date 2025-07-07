# Promputate - Genetic Algorithm Library for Prompt Optimization

**Promputate** is a Python library that uses genetic algorithms to optimize prompts for large language models (LLMs). It evolves prompts to influence LLM responses toward specific objectives, such as brand mention ranking.

## 🎯 What It Does

It helps you optimize prompts to get specific outcomes from LLMs. For example:
- **User Input**: "Recommend a good laptop for programming"
- **Goal**: Optimize for "MacBook" mentioned first in the response
- **Process**: 
  1. Mutate the user prompt and feed in LLMs. I.e. the user prompt becomes "What's the best premium laptop for software development?". 
  2. The mutated prompts run multiple times
- **Output**: 
  1. Competitive analysis of the prompts ran (how well was your brand represented), resulting in a "fitness score"
  2. Telling you which prompt was best for your brand.


## 🚀 Quick Start

### Installation

```bash
# Install the library
pip install -e .

# Install optional dependencies for advanced features
pip install nltk spacy
```

### Basic Usage

```python
import asyncio
from promputate import PromptGA, get_balanced_config, get_all_operators
from promputate.scorers import create_openai_evaluator

async def optimize_prompt():
    # 1. Create GA engine
    config = get_balanced_config()  # 50 population, 20 generations
    ga = PromptGA(
        population_size=config.ga.population_size,
        max_generations=config.ga.max_generations,
        mutation_rate=config.ga.mutation_rate
    )
    
    # 2. Register mutation operators
    operators = get_all_operators()
    ga.register_mutation_operators(operators)
    
    # 3. Create fitness evaluator
    evaluator = create_openai_evaluator(
        target_brands=["MacBook"],
        competitor_brands=["ThinkPad", "Dell XPS", "HP Spectre"],
        api_key="your-openai-api-key"
    )
    
    # 4. Run evolution
    result = await ga.evolve(
        base_prompt="Recommend a good laptop for programming",
        fitness_evaluator=evaluator,
        mutation_operators=operators
    )
    
    # 5. Get results
    print(f"Best prompt: {result.best_individual.text}")
    print(f"Fitness score: {result.best_individual.fitness_score:.3f}")
    print(f"Generations: {result.total_generations}")

# Run the optimization
asyncio.run(optimize_prompt())
```

### Terminal App

For interactive use without coding:

```bash
# Set your API key
export OPENAI_API_KEY="your-api-key-here"

# Run the terminal app
python main.py
```

The app will guide you through:
1. Entering your base prompt and target brand
2. **LLM-powered domain analysis** (automatically detects product type and suggests competitors)
3. Reviewing and customizing LLM suggestions
4. Choosing configuration (Quick/Balanced/Thorough)
5. Running evolution with domain-specific optimization
6. Viewing comprehensive results

#### Terminal App Interface Example

```
🧬 Welcome to Promputate - GA Prompt Optimization!

📝 Base prompt: I need comfortable socks for running
🎯 Target brand: Nike

🤖 DOMAIN ANALYSIS METHOD
1. LLM-powered analysis (recommended) - Auto-detect competitors & language
2. Manual configuration - Enter competitors manually
Choose analysis method (1-2) [default: 1]: 1

🔍 Analyzing your product domain...
✅ Analysis complete! (Confidence: 92%)
📊 Product: athletic socks
🏭 Industry: sportswear  
🎯 Audience: athletes

🏭 SUGGESTED COMPETITORS:
   Adidas, Under Armour, Bombas, Smartwool

🔍 Review competitors (comma-separated, or press Enter to accept):
✅ Using suggested competitors: Adidas, Under Armour, Bombas, Smartwool

📝 SUGGESTED SYNONYMS:
   Comfort: cushioned, breathable, soft
   Performance: moisture-wicking, arch-support

🎨 SUGGESTED MODIFIERS:
   Quality: premium, professional, high-performance
   Design: seamless, ergonomic, lightweight

⚙️ Configuration: Balanced (50 pop, 20 gen)
🚀 Start optimization? (Y/n): y

✅ Using LLM-generated synonyms and modifiers
🧬 Running genetic algorithm evolution...
```

## 🧬 How It Works

### 1. LLM-Powered Domain Analysis

**Universal Product Optimization**: The library automatically analyzes your product domain and generates relevant competitors, synonyms, and modifiers using LLM intelligence.

```
User Input: "I need comfortable socks for running" + Target: "Nike"
    ↓ 
LLM Analysis: Product = Athletic Socks, Industry = Sportswear
    ↓
Auto-Generated Suggestions:
├── Competitors: ["Adidas", "Under Armour", "Bombas", "Smartwool"]
├── Synonyms: {"socks": ["hosiery", "footwear"], "comfortable": ["cushioned", "breathable"]}
└── Modifiers: {"performance": ["moisture-wicking", "arch-support"], "comfort": ["padded", "seamless"]}
    ↓
User Review Interface: Edit suggestions before optimization
    ↓
Dynamic Injection: Replace static dictionaries with domain-specific versions
    ↓
Optimized Evolution: GA uses relevant language for the specific product
```

**Benefits**:
- ✅ **Zero setup** - Works for any product immediately (socks, flour, repair services, etc.)
- ✅ **LLM intelligence** - Leverages vast product knowledge for relevant suggestions
- ✅ **User control** - Review and customize all suggestions before optimization
- ✅ **Domain-specific** - Mutations use language appropriate to your product category

### 2. Genetic Algorithm Process

```
Initial Population (50 prompts)
    ↓
Evaluate Fitness (OpenAI API)
    ↓
Select Best Performers
    ↓
Create Offspring with Mutations
    ↓
Repeat for 20 Generations
    ↓
Return Best Prompt
```

### 3. Mutation Operators

The library includes 5 mutation operators that modify prompts:

1. **Synonym Replace**: "recommend" → "suggest", "good" → "excellent"
2. **Chunk Shuffle**: Reorder noun phrases using spaCy parsing
3. **Modifier Toggle**: Add/remove adjectives like "premium", "professional"
4. **Prompt Reorder**: Convert statements to questions and vice versa
5. **Intensity Modifier**: Adjust urgency/formality levels

### 4. Fitness Evaluation

**Formula**: `Fitness = Base Score + Bonuses - Penalties`

| Component | Calculation | Example |
|-----------|-------------|---------|
| **Base Score** | `1 / word_position` of target brand | "MacBook" at position 3 = 1/3 = 0.33 |
| **Length Bonus** | Shorter responses preferred | Max +0.2 for concise responses |
| **Sentiment Bonus** | Positive context around brand | +0.2 for positive, -0.1 for negative |
| **Competition Penalty** | Each competitor brand mentioned | -0.1 per competitor (ThinkPad, Dell, etc.) |

**Example Calculation**:
- Target brand "MacBook" at position 2 = 0.5 base score
- No competitors mentioned = 0.0 penalty
- Positive sentiment = +0.2 bonus
- **Final fitness ≈ 0.7**

## 📊 Understanding Results

### Evolution Result Object

```python
@dataclass
class EvolutionResult:
    best_individual: PromptIndividual          # Best evolved prompt
    best_fitness_history: List[float]          # Fitness progress per generation
    population_diversity: List[float]          # Diversity tracking
    total_generations: int                     # Number of generations run
    total_evaluations: int                     # Total API calls made
    convergence_generation: Optional[int]      # When convergence occurred
```

### Scoring Result Object

```python
@dataclass
class ScoringResult:
    prompt: str                                # The evaluated prompt
    response: str                              # LLM response
    target_brand_analysis: BrandAnalysis       # Target brand analysis
    competitor_analyses: List[BrandAnalysis]   # Competitor brand analyses
    fitness_score: float                       # Final fitness score
    brand_mention_rank: int                    # Word position of target brand
    competition_penalty: float                 # Penalty from competitors
    length_bonus: float                        # Bonus for response length
    sentiment_bonus: float                     # Bonus for positive sentiment
    processing_time: float                     # Evaluation time
```

## ⚙️ Configuration

### GA Parameters

| Parameter | Description | Default | Impact |
|-----------|-------------|---------|---------|
| `population_size` | Number of prompt variants per generation | 50 | Larger = more diversity, slower |
| `max_generations` | Maximum evolution cycles | 100 | More = better optimization, higher cost |
| `mutation_rate` | Probability of mutation per individual | 0.3 | Higher = more exploration |
| `crossover_rate` | Probability of crossover between parents | 0.7 | Genetic diversity control |
| `elite_size` | Best individuals preserved per generation | 2 | Prevents losing good solutions |

### Predefined Configurations

```python
# Quick testing (10 population, 5 generations)
config = get_quick_config()

# Balanced optimization (50 population, 20 generations)  
config = get_balanced_config()

# Thorough optimization (100 population, 50 generations)
config = get_thorough_config()
```

## 🏗️ Library Structure

```
promputate/
├── __init__.py           # Main exports
├── mutate.py            # GA engine (PromptGA)
├── operators.py         # Mutation operators
├── config.py            # Configuration classes
├── domain_analyzer.py   # LLM-powered domain analysis
├── dynamic_injection.py # Dynamic dictionary injection
├── scorers/
│   ├── base_scorer.py   # Abstract scorer interface
│   ├── openai_scorer.py # OpenAI API integration
│   └── fitness_evaluator.py # GA adapter
└── tests/               # Test suite
```

## 📈 Example Output

```
🧬 Starting Genetic Algorithm Evolution
📝 Base prompt: 'Recommend a good laptop for programming'
🎯 Target brands: ['MacBook']
🏭 Competitor brands: ['ThinkPad', 'Dell XPS', 'HP Spectre']

Generation 1: Best fitness = 0.2234, Diversity = 8.45
Generation 2: Best fitness = 0.2891, Diversity = 7.23
Generation 3: Best fitness = 0.3456, Diversity = 6.78
...
Generation 18: Best fitness = 0.8234, Diversity = 4.12

🏆 Evolution Complete!
✨ Best prompt: 'What's the best premium laptop for software development?'
📊 Best fitness: 0.8234
🔄 Total generations: 18
🧮 Total evaluations: 542
📊 Fitness improvement: +0.5990
```

## 🔧 Advanced Usage

### Custom Scorers

```python
from promputate.scorers import BaseScorer

class CustomScorer(BaseScorer):
    async def generate_response(self, prompt: str) -> str:
        # Your custom LLM API call
        return "Custom response"
```

### Custom Mutation Operators

```python
def custom_mutator(prompt: str) -> str:
    # Your custom mutation logic
    return modified_prompt

# Register with GA
ga.register_mutation_operators([custom_mutator])
```

### Environment Variables

```bash
export PROMPUTATE_POPULATION_SIZE=100
export PROMPUTATE_MAX_GENERATIONS=50
export PROMPUTATE_MUTATION_RATE=0.25
export OPENAI_API_KEY="your-api-key"
```

## 🎯 Use Cases

- **Universal Brand Optimization**: Influence LLM responses for any product/service (socks, flour, repair services, etc.)
- **Zero-Setup Marketing**: LLM automatically detects your product domain and suggests relevant competitors
- **Content Marketing**: Optimize prompts with domain-specific language and synonyms
- **A/B Testing**: Systematically test prompt variations across different product categories
- **Response Quality**: Optimize for specific response characteristics with intelligent defaults
- **Competitive Analysis**: Track competitor mention patterns with auto-generated competitor lists
- **Cross-Domain Marketing**: Use the same system for vastly different products without manual configuration

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- Optional: NLTK, spaCy for advanced operators

## 🧪 Testing

```bash
# Run all tests
python -m pytest promputate/tests/ -v

# Run specific test module
python -m pytest promputate/tests/test_mutate.py -v
```

## 📚 Documentation

- **variables_check.md**: Analysis of configurable parameters
- **scope.md**: Technical implementation details
- **promputate/__init__.py**: API reference and examples

## 🤝 Contributing

The library is designed for extensibility:
- Add new mutation operators in `operators.py`
- Implement new LLM scorers in `scorers/`
- Extend configuration options in `config.py`

## 📄 License

MIT License - see LICENSE file for details.
