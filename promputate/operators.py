"""
Mutation Operators for Prompt Optimization

This module provides individual mutation functions that can be applied
to prompts to create variations for genetic algorithm evolution.
"""

import random
import re
import logging
from typing import List, Dict, Set, Optional, Callable
from functools import lru_cache

# Try to import optional dependencies
try:
    import nltk
    from nltk.corpus import wordnet
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# Global spaCy model (loaded lazily)
_nlp_model = None


def get_spacy_model():
    """Get or load spaCy model lazily"""
    global _nlp_model, SPACY_AVAILABLE
    if _nlp_model is None and SPACY_AVAILABLE:
        try:
            import spacy
            _nlp_model = spacy.load('en_core_web_sm')
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.warning("spaCy English model not found. Run: python -m spacy download en_core_web_sm")
            SPACY_AVAILABLE = False
    return _nlp_model


# Predefined word lists for mutations
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
    'underwear': ['underclothing', 'undergarment', 'underclothes'],
    'clothing': ['garment', 'apparel', 'wear', 'attire'],
    'socks': ['hosiery'],
    'shoes': ['footwear', 'sneakers', 'boots'],
    'sports': ['athletics', 'exercise', 'fitness', 'workout'],
}

MODIFIERS = {
    'performance': ['high-performance', 'powerful', 'fast', 'efficient', 'optimized'],
    'quality': ['premium', 'professional', 'quality', 'reliable', 'robust'],
    'portability': ['portable', 'lightweight', 'compact', 'mobile', 'travel-friendly'],
    'price': ['affordable', 'budget-friendly', 'cost-effective', 'economical'],
    'design': ['sleek', 'modern', 'stylish', 'elegant', 'well-designed'],
    'features': ['feature-rich', 'comprehensive', 'versatile', 'advanced', 'cutting-edge'],
}

# Question starters for prompt reordering
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

# Question endings
QUESTION_ENDINGS = [
    "?",
    "? Any suggestions?",
    ". What do you recommend?",
    ". Any advice?",
    "? I'd appreciate your help.",
    "? Please help.",
]


class MutationOperatorRegistry:
    """Registry for managing mutation operators"""
    
    def __init__(self):
        self.operators: Dict[str, Callable[[str], str]] = {}
        self.weights: Dict[str, float] = {}
    
    def register(self, name: str, operator: Callable[[str], str], weight: float = 1.0):
        """Register a mutation operator with optional weight"""
        self.operators[name] = operator
        self.weights[name] = weight
        logger.info(f"Registered operator '{name}' with weight {weight}")
    
    def get_operator(self, name: str) -> Optional[Callable[[str], str]]:
        """Get operator by name"""
        return self.operators.get(name)
    
    def get_random_operator(self) -> Callable[[str], str]:
        """Get random operator based on weights"""
        if not self.operators:
            raise ValueError("No operators registered")
        
        names = list(self.operators.keys())
        weights = [self.weights[name] for name in names]
        
        selected_name = random.choices(names, weights=weights)[0]
        return self.operators[selected_name]
    
    def list_operators(self) -> List[str]:
        """List all registered operator names"""
        return list(self.operators.keys())


# Global registry instance
operator_registry = MutationOperatorRegistry()


def synonym_replace(prompt: str, max_replacements: int = 2) -> str:
    """
    Replace words with synonyms using NLTK WordNet and predefined dictionary
    
    Args:
        prompt: Input prompt to modify
        max_replacements: Maximum number of words to replace
        
    Returns:
        Modified prompt with synonyms
    """
    words = prompt.split()
    if not words:
        return prompt
    
    # Track replacements
    replacements_made = 0
    result_words = words.copy()
    
    # Get indices to potentially replace
    indices = list(range(len(words)))
    random.shuffle(indices)
    
    for idx in indices:
        if replacements_made >= max_replacements:
            break
        
        word = words[idx].lower().strip('.,!?;:"\'')
        
        # Try predefined dictionary first
        synonyms = set()
        if word in SYNONYM_DICTIONARY:
            synonyms.update(SYNONYM_DICTIONARY[word])
        
        # Skip WordNet for very common words that get weird synonyms
        skip_wordnet = {
            'i', 'me', 'my', 'a', 'an', 'the', 'is', 'are', 'was', 'were', 
            'have', 'has', 'had', 'do', 'does', 'did', 'can', 'could', 'would', 'should'
        }
        if word in skip_wordnet:
            continue
        
        # Try WordNet if available (but be more conservative)
        if NLTK_AVAILABLE and len(synonyms) < 3:
            try:
                for syn in wordnet.synsets(word):
                    if syn and hasattr(syn, 'lemmas'):
                        for lemma in syn.lemmas():
                            if lemma and hasattr(lemma, 'name'):
                                synonym = lemma.name().replace('_', ' ')
                                # More conservative filtering
                                if (synonym.lower() != word.lower() and 
                                    len(synonym.split()) <= 2 and
                                    len(synonym) <= 15 and  # Avoid overly long technical terms
                                    not any(char.isdigit() for char in synonym) and  # No numbers
                                    not synonym.lower().endswith(('ic', 'al', 'ous', 'ine', 'ism', 'tic')) and  # Avoid technical suffixes
                                    synonym.isalpha() or ' ' in synonym):  # Only alphabetic or simple phrases
                                    synonyms.add(synonym)
                                if len(synonyms) >= 3:  # Reduced limit
                                    break
                    if len(synonyms) >= 3:
                        break
            except Exception as e:
                logger.debug(f"WordNet lookup failed for '{word}': {e}")
        
        # Apply synonym if found
        if synonyms:
            synonym = random.choice(list(synonyms))
            
            # Validate synonym is not empty or just whitespace
            if not synonym or not synonym.strip():
                logger.debug(f"Skipping empty synonym for '{words[idx]}'")
                continue
            
            # Preserve capitalization
            if words[idx][0].isupper():
                synonym = synonym.capitalize()
            
            # Preserve punctuation
            punctuation = re.findall(r'[.,!?;:"\']', words[idx])
            if punctuation:
                synonym += ''.join(punctuation)
            
            # Final validation before replacement
            if synonym.strip():
                result_words[idx] = synonym
                replacements_made += 1
                logger.debug(f"Replaced '{words[idx]}' with '{synonym}'")
            else:
                logger.debug(f"Skipping invalid synonym '{synonym}' for '{words[idx]}'")
    
    return ' '.join(result_words)


def chunk_shuffle(prompt: str) -> str:
    """
    Shuffle noun phrases and clauses using spaCy parsing
    
    Args:
        prompt: Input prompt to modify
        
    Returns:
        Modified prompt with shuffled chunks
    """
    if not SPACY_AVAILABLE:
        # Fallback: simple word-level shuffle
        return _simple_word_shuffle(prompt)
    
    nlp = get_spacy_model()
    if nlp is None:
        return _simple_word_shuffle(prompt)
    
    try:
        doc = nlp(prompt)
        
        # Extract noun phrases
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        # If we have multiple noun phrases, try to shuffle them
        if len(noun_phrases) >= 2:
            # Create a mapping of original to shuffled phrases
            shuffled_phrases = noun_phrases.copy()
            random.shuffle(shuffled_phrases)
            
            # Only shuffle if the result is different
            if shuffled_phrases != noun_phrases:
                result = prompt
                for original, shuffled in zip(noun_phrases, shuffled_phrases):
                    if original != shuffled:
                        # Replace first occurrence
                        result = result.replace(original, f"__TEMP_{hash(shuffled)}_", 1)
                
                # Replace temporary placeholders
                for original, shuffled in zip(noun_phrases, shuffled_phrases):
                    if original != shuffled:
                        result = result.replace(f"__TEMP_{hash(shuffled)}_", shuffled, 1)
                
                logger.debug(f"Shuffled noun phrases: {noun_phrases} -> {shuffled_phrases}")
                return result
        
        # Fallback: sentence-level reordering
        sentences = [sent.text.strip() for sent in doc.sents]
        if len(sentences) >= 2:
            random.shuffle(sentences)
            return ' '.join(sentences)
            
    except Exception as e:
        logger.debug(f"spaCy chunk shuffling failed: {e}")
    
    return _simple_word_shuffle(prompt)


def _simple_word_shuffle(prompt: str) -> str:
    """Fallback word shuffling when spaCy is not available"""
    words = prompt.split()
    if len(words) <= 2:
        return prompt
    
    # Keep first and last word, shuffle middle
    if len(words) > 3:
        middle = words[1:-1]
        random.shuffle(middle)
        return ' '.join([words[0]] + middle + [words[-1]])
    
    return prompt


def modifier_toggle(prompt: str, max_modifiers: int = 1) -> str:
    """
    Add or remove modifiers (adjectives) to/from the prompt
    
    Args:
        prompt: Input prompt to modify
        max_modifiers: Maximum number of modifiers to add/remove
        
    Returns:
        Modified prompt with added or removed modifiers
    """
    # 50% chance to add, 50% chance to remove
    if random.random() < 0.5:
        return _add_modifiers(prompt, max_modifiers)
    else:
        return _remove_modifiers(prompt, max_modifiers)


def _add_modifiers(prompt: str, max_modifiers: int) -> str:
    """Add modifiers to the prompt"""
    words = prompt.split()
    if not words:
        return prompt
    
    # Find nouns to modify (simple heuristic)
    nouns = ['laptop', 'computer', 'device', 'machine', 'notebook', 'system',
             'phone', 'smartphone', 'tablet', 'car', 'vehicle', 'product']
    
    result_words = words.copy()
    modifiers_added = 0
    
    for i, word in enumerate(words):
        if modifiers_added >= max_modifiers:
            break
        
        word_clean = word.lower().strip('.,!?;:"\'')
        
        if word_clean in nouns:
            # Select random modifier category
            category = random.choice(list(MODIFIERS.keys()))
            modifier = random.choice(MODIFIERS[category])
            
            # Insert modifier before the noun
            result_words[i] = f"{modifier} {word}"
            modifiers_added += 1
            logger.debug(f"Added modifier '{modifier}' to '{word}'")
    
    # If no nouns found, add modifier to a random word
    if modifiers_added == 0 and len(words) > 1:
        category = random.choice(list(MODIFIERS.keys()))
        modifier = random.choice(MODIFIERS[category])
        
        # Insert at random position (not first or last)
        pos = random.randint(1, len(result_words) - 1)
        result_words.insert(pos, modifier)
        logger.debug(f"Added modifier '{modifier}' at position {pos}")
    
    return ' '.join(result_words)


def _remove_modifiers(prompt: str, max_removals: int) -> str:
    """Remove modifiers from the prompt"""
    words = prompt.split()
    if len(words) <= 3:  # Increased minimum to preserve sentence structure
        return prompt
    
    # Identify potential modifiers (adjectives) - only safe ones
    all_modifiers = set()
    for category_modifiers in MODIFIERS.values():
        all_modifiers.update(category_modifiers)
    
    # Add common adjectives that are safe to remove
    safe_to_remove = {'good', 'great', 'nice', 'fine', 'decent', 'solid', 'reliable', 
                      'excellent', 'outstanding', 'quality', 'premium', 'professional'}
    all_modifiers.update(safe_to_remove)
    
    # Core nouns that should NEVER be removed
    core_nouns = {'laptop', 'computer', 'device', 'machine', 'notebook', 'system',
                  'phone', 'smartphone', 'tablet', 'programming', 'coding', 'development',
                  'work', 'software', 'app', 'application', 'service', 'product'}
    
    result_words = words.copy()
    removals_made = 0
    
    i = 0
    while i < len(result_words) and removals_made < max_removals:
        word_clean = result_words[i].lower().strip('.,!?;:"\'')
        
        # Only remove if it's a known modifier AND not a core noun
        if word_clean in all_modifiers and word_clean not in core_nouns:
            removed_word = result_words.pop(i)
            removals_made += 1
            logger.debug(f"Removed modifier '{removed_word}'")
            # Don't increment i since we removed an element
        else:
            i += 1
    
    # Ensure we still have a reasonable sentence
    if len(result_words) < 3:
        return prompt  # Return original if too short
    
    return ' '.join(result_words)


def prompt_reorder(prompt: str) -> str:
    """
    Reorder the prompt structure (statement to question, etc.)
    
    Args:
        prompt: Input prompt to modify
        
    Returns:
        Modified prompt with different structure
    """
    prompt = prompt.strip()
    
    # Convert statement to question
    if not prompt.endswith('?'):
        return _statement_to_question(prompt)
    else:
        return _question_to_statement(prompt)


def _statement_to_question(prompt: str) -> str:
    """Convert statement to question format"""
    # Remove ending punctuation
    prompt = prompt.rstrip('.,!;:')
    
    # Extract the main concept
    words = prompt.lower().split()
    
    # Common patterns
    if 'recommend' in words:
        # "Recommend a good laptop" -> "What's the best laptop?"
        concept = ' '.join(words[words.index('recommend')+1:])
        concept = concept.replace('a ', '').replace('an ', '')
        starter = random.choice(["What's the best", "Can you suggest", "Which"])
        ending = random.choice(QUESTION_ENDINGS)
        return f"{starter} {concept}{ending}"
    
    elif 'need' in words or 'want' in words:
        # "I need a laptop" -> "Looking for a laptop?"
        marker = 'need' if 'need' in words else 'want'
        concept = ' '.join(words[words.index(marker)+1:])
        starter = random.choice(["Looking for", "Help me find", "I'm searching for"])
        ending = random.choice(QUESTION_ENDINGS)
        return f"{starter} {concept}{ending}"
    
    else:
        # Generic transformation
        starter = random.choice(QUESTION_STARTERS)
        ending = random.choice(QUESTION_ENDINGS)
        return f"{starter} {prompt.lower()}{ending}"


def _question_to_statement(prompt: str) -> str:
    """Convert question to statement format"""
    prompt = prompt.rstrip('?')
    
    # Simple transformations
    transformations = [
        ("what's the best", "recommend the best"),
        ("what is the best", "recommend the best"),
        ("can you suggest", "please suggest"),
        ("which", "I need"),
        ("looking for", "I need"),
        ("help me find", "please recommend"),
    ]
    
    prompt_lower = prompt.lower()
    for question_phrase, statement_phrase in transformations:
        if prompt_lower.startswith(question_phrase):
            return prompt.replace(question_phrase, statement_phrase, 1) + "."
    
    # Generic transformation
    return f"I need {prompt.lower()}."


def intensity_modifier(prompt: str) -> str:
    """
    Modify the intensity/urgency of the prompt
    
    Args:
        prompt: Input prompt to modify
        
    Returns:
        Modified prompt with different intensity
    """
    intensity_words = {
        'low': ['please', 'kindly', 'if possible', 'perhaps'],
        'medium': ['I would like', 'could you', 'I am looking for'],
        'high': ['urgent', 'immediately', 'ASAP', 'critical', 'must have'],
        'very_high': ['URGENT', 'CRITICAL', 'EMERGENCY', 'MUST', 'REQUIRED']
    }
    
    # Random intensity level
    level = random.choice(list(intensity_words.keys()))
    modifier = random.choice(intensity_words[level])
    
    # Add at beginning or end
    if random.random() < 0.5:
        return f"{modifier} {prompt}"
    else:
        return f"{prompt} {modifier}"


# Register all operators
def register_default_operators():
    """Register all default mutation operators"""
    operator_registry.register("synonym_replace", synonym_replace, weight=2.0)
    # operator_registry.register("chunk_shuffle", chunk_shuffle, weight=1.5)  # Disabled - causes grammar issues
    operator_registry.register("modifier_toggle", modifier_toggle, weight=2.0)
    operator_registry.register("prompt_reorder", prompt_reorder, weight=1.8)
    operator_registry.register("intensity_modifier", intensity_modifier, weight=1.0)
    
    logger.info(f"Registered {len(operator_registry.operators)} default operators")


# Convenience function to get all operators
def get_all_operators() -> List[Callable[[str], str]]:
    """Get all registered operators as a list"""
    if not operator_registry.operators:
        register_default_operators()
    
    return list(operator_registry.operators.values())


def get_operator_by_name(name: str) -> Optional[Callable[[str], str]]:
    """Get specific operator by name"""
    if not operator_registry.operators:
        register_default_operators()
    
    return operator_registry.get_operator(name)


# Initialize default operators on module import
register_default_operators() 