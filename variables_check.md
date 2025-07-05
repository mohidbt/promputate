# Static Variables Analysis for Promputate P2

## Overview

This document analyzes all static variables, constants, and hardcoded values found in the Promputate codebase and example files. These are potential candidates for making dynamic in P2.

## 1. Configuration Default Values

### GA Configuration (`promputate/config.py`)
- **population_size**: `50` (default population size)
- **max_generations**: `100` (default max generations)
- **mutation_rate**: `0.3` (default mutation rate)
- **crossover_rate**: `0.7` (default crossover rate)
- **selection_pressure**: `1.5` (default selection pressure)
- **elite_size**: `2` (default elite size)
- **convergence_threshold**: `0.001` (default convergence threshold)
- **convergence_generations**: `10` (default convergence generations)
- **early_stopping**: `True` (default early stopping)
- **log_level**: `"INFO"` (default log level)
- **track_diversity**: `True` (default diversity tracking)
- **save_population_history**: `False` (default population history saving)

### Operator Configuration (`promputate/config.py`)
- **synonym_replace_weight**: `2.0` (operator weight)
- **chunk_shuffle_weight**: `1.5` (operator weight)
- **modifier_toggle_weight**: `2.0` (operator weight)
- **prompt_reorder_weight**: `1.8` (operator weight)
- **intensity_modifier_weight**: `1.0` (operator weight)
- **max_synonym_replacements**: `2` (max synonym replacements)
- **max_modifiers**: `1` (max modifiers)
- **preserve_question_format**: `True` (preserve question format)
- **require_nltk**: `True` (require NLTK)
- **require_spacy**: `True` (require spaCy)
- **fallback_on_error**: `True` (fallback on error)

### Fitness Configuration (`promputate/config.py`)
- **brand_mention_weight**: `1.0` (brand mention weight)
- **position_penalty_factor**: `0.1` (position penalty factor)
- **length_bonus_weight**: `0.2` (length bonus weight)
- **competition_penalty_weight**: `0.1` (competition penalty weight)
- **sentiment_bonus_weight**: `0.1` (sentiment bonus weight)
- **naturalness_weight**: `0.05` (naturalness weight)
- **max_response_length**: `1000` (max response length)
- **min_response_length**: `10` (min response length)
- **timeout_seconds**: `30.0` (timeout seconds)

### General Configuration (`promputate/config.py`)
- **output_dir**: `"./promputate_output"` (output directory)
- **save_results**: `True` (save results)
- **api_timeout**: `30.0` (API timeout)
- **max_retries**: `3` (max retries)
- **rate_limit_delay**: `1.0` (rate limit delay)

## 2. Brand and Domain Data

### Default Target Brands (`promputate/config.py`)
```python
target_brands: ['MacBook']
```

### Default Competitor Brands (`promputate/config.py`)
```python
competitor_brands: [
    'ThinkPad', 'Dell XPS', 'HP Spectre', 'Surface Laptop', 'Alienware'
]
```

### Extended Competitor Brands (`promputate/scorers/openai_scorer.py`)
```python
competitor_brands: [
    "ThinkPad", "Dell XPS", "HP Spectre", "Surface Laptop", 
    "Alienware", "ASUS", "Acer", "Lenovo", "Samsung"
]
```

### Example App Defaults (`main.py`, `p1_example.py`)
- **base_prompt**: `"Recommend a good laptop for programming"`
- **target_brand**: `"MacBook"`
- **competitor_brands**: `["ThinkPad", "Dell XPS", "HP Spectre", "Surface Laptop", "Alienware"]`

## 3. Mutation Operator Data

### Synonym Dictionary (`promputate/operators.py`)
```python
SYNONYM_DICTIONARY = {
    'good': ['excellent', 'great', 'outstanding', 'superior', 'quality', 'top', 'best', 'fine'],
    'recommend': ['suggest', 'advise', 'propose', 'endorse', 'favor', 'advocate'],
    'laptop': ['notebook', 'computer', 'machine', 'device', 'system'],
    'programming': ['coding', 'development', 'software development', 'coding work', 'software engineering'],
    'for': ['suitable for', 'ideal for', 'perfect for', 'designed for', 'intended for'],
    'need': ['require', 'want', 'seek', 'desire', 'look for'],
    'best': ['top', 'finest', 'premier', 'optimal', 'leading', 'superior'],
    'help': ['assist', 'aid', 'support', 'guide', 'advise'],
    'find': ['locate', 'discover', 'identify', 'seek', 'search for'],
    'looking': ['searching', 'seeking', 'hunting', 'browsing', 'shopping'],
}
```

### Modifiers Dictionary (`promputate/operators.py`)
```python
MODIFIERS = {
    'performance': ['high-performance', 'powerful', 'fast', 'efficient', 'optimized'],
    'quality': ['premium', 'professional', 'quality', 'reliable', 'robust'],
    'portability': ['portable', 'lightweight', 'compact', 'mobile', 'travel-friendly'],
    'price': ['affordable', 'budget-friendly', 'cost-effective', 'economical'],
    'design': ['sleek', 'modern', 'stylish', 'elegant', 'well-designed'],
    'features': ['feature-rich', 'comprehensive', 'versatile', 'advanced', 'cutting-edge'],
}
```

### Question Starters (`promputate/operators.py`)
```python
QUESTION_STARTERS = [
    "What's the best",
    "Can you suggest",
    "I need",
    "Looking for",
    "Help me find",
    "Which",
    "What would you recommend for",
    "I'm searching for",
    "Could you recommend",
    "What are the top",
]
```

### Question Endings (`promputate/operators.py`)
```python
QUESTION_ENDINGS = [
    "?",
    "? Any suggestions?",
    ". What do you recommend?",
    ". Any advice?",
    "? I'd appreciate your help.",
    "? Please help.",
]
```

### Nouns for Modifier Addition (`promputate/operators.py`)
```python
nouns = [
    'laptop', 'computer', 'device', 'machine', 'notebook', 'system',
    'phone', 'smartphone', 'tablet', 'car', 'vehicle', 'product'
]
```

### Intensity Words (`promputate/operators.py`)
```python
intensity_words = {
    'low': ['please', 'kindly', 'if possible', 'perhaps'],
    'medium': ['I would like', 'could you', 'I am looking for'],
    'high': ['urgent', 'immediately', 'ASAP', 'critical', 'must have'],
    'very_high': ['URGENT', 'CRITICAL', 'EMERGENCY', 'MUST', 'REQUIRED']
}
```

### Question-Statement Transformations (`promputate/operators.py`)
```python
transformations = [
    ("what's the best", "recommend the best"),
    ("what is the best", "recommend the best"),
    ("can you suggest", "please suggest"),
    ("which", "I need"),
    ("looking for", "I need"),
    ("help me find", "please recommend"),
]
```

## 4. Sentiment Analysis Data

### Positive Words (`promputate/scorers/base_scorer.py`)
```python
positive_words = [
    'best', 'excellent', 'great', 'amazing', 'perfect', 'outstanding', 
    'superior', 'premium', 'top', 'recommend', 'love', 'fantastic'
]
```

### Negative Words (`promputate/scorers/base_scorer.py`)
```python
negative_words = [
    'bad', 'terrible', 'awful', 'poor', 'worst', 'hate', 'horrible',
    'disappointing', 'overpriced', 'expensive', 'limited', 'lacking'
]
```

## 5. API Configuration

### OpenAI Model Settings (`promputate/scorers/openai_scorer.py`)
- **model**: `"gpt-4o-mini"` (default model)
- **max_tokens**: `500` (default max tokens)
- **temperature**: `0.7` (default temperature)
- **max_response_length**: `1000` (default max response length)
- **timeout**: `30.0` (default timeout)
- **max_concurrent**: `5` (default max concurrent requests)
- **rate_limit_delay**: `1.0` (default rate limit delay)

### Cost Estimation (`promputate/scorers/openai_scorer.py`)
- **cost_per_1k_input**: `0.00015` (GPT-4o-mini input cost)
- **cost_per_1k_output**: `0.0006` (GPT-4o-mini output cost)
- **token_char_ratio**: `4` (characters per token estimation)

### Retry Configuration (`promputate/scorers/openai_scorer.py`)
- **max_attempts**: `3` (retry attempts)
- **wait_min**: `4` (min wait time)
- **wait_max**: `10` (max wait time)
- **wait_multiplier**: `1` (wait multiplier)

## 6. Fitness Calculation Constants

### Scoring Constants (`promputate/scorers/base_scorer.py`)
- **competition_penalty_per_brand**: `0.1` (penalty per competitor brand)
- **length_bonus_factor**: `0.2` (length bonus factor)
- **positive_sentiment_bonus**: `0.2` (positive sentiment bonus)
- **negative_sentiment_penalty**: `-0.1` (negative sentiment penalty)
- **context_window**: `50` (context extraction window)
- **default_confidence**: `0.8` (default confidence score)

## 7. Predefined Configurations

### Quick Configuration (`promputate/config.py`)
- **population_size**: `10`
- **max_generations**: `5`
- **mutation_rate**: `0.4`
- **convergence_generations**: `3`

### Thorough Configuration (`promputate/config.py`)
- **population_size**: `100`
- **max_generations**: `50`
- **mutation_rate**: `0.25`
- **crossover_rate**: `0.8`
- **convergence_generations**: `15`

### Balanced Configuration (`promputate/config.py`)
- **population_size**: `50`
- **max_generations**: `20`
- **mutation_rate**: `0.3`
- **crossover_rate**: `0.7`
- **convergence_generations**: `8`

## 8. User Interface Strings

### Main App (`main.py`)
- Configuration prompts and messages
- Default values for user input
- Success/error messages
- Progress indicators

### P1 Example (`p1_example.py`)
- Logging messages
- Status indicators
- Result formatting

## 9. Environment Variables

### Supported Environment Variables (`promputate/config.py`)
- **PROMPUTATE_POPULATION_SIZE**
- **PROMPUTATE_MAX_GENERATIONS**
- **PROMPUTATE_MUTATION_RATE**
- **PROMPUTATE_CROSSOVER_RATE**
- **PROMPUTATE_RANDOM_SEED**
- **PROMPUTATE_OUTPUT_DIR**
- **PROMPUTATE_API_TIMEOUT**
- **PROMPUTATE_MAX_RETRIES**
- **OPENAI_API_KEY**

## 10. Package Metadata

### Version Information (`promputate/__init__.py`)
- **__version__**: `"0.1.0"`
- **__author__**: `"Promputate Team"`
- **__description__**: `"Genetic Algorithm for Prompt Optimization"`
- **__title__**: `'promputate'`
- **__author_email__**: `'team@promputate.ai'`
- **__license__**: `'MIT'`
- **__url__**: `'https://github.com/promputate/promputate'`
- **__download_url__**: `'https://pypi.org/project/promputate/'`
- **__status__**: `'Development'`

## Analysis Summary

### Total Static Variables Found: 80+

**Categories:**
- Configuration defaults: 25 variables
- Brand/domain data: 8 lists/strings
- Mutation operator data: 6 large dictionaries
- Sentiment analysis data: 2 lists
- API configuration: 12 variables
- Fitness calculation constants: 8 constants
- Predefined configurations: 12 variables
- Package metadata: 9 constants

**High Priority for Dynamic Configuration:**
1. Brand and competitor lists (domain-specific)
2. Mutation operator dictionaries (domain-specific)
3. API model and cost settings (provider-specific)
4. Fitness calculation weights (use case-specific)
5. Configuration presets (user preference-specific)

**Medium Priority:**
1. Sentiment analysis word lists
2. Question transformation patterns
3. Timeout and retry settings

**Low Priority:**
1. Package metadata
2. Environment variable names
3. Default file paths 