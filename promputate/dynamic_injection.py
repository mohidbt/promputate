"""
Dynamic Dictionary Injection for LLM-Generated Content

This module provides functionality to dynamically inject LLM-generated
synonyms and modifiers into the mutation operators, replacing static
dictionaries with domain-specific alternatives.
"""

import logging
from typing import Dict, List, Optional, Any, ContextManager
from contextlib import contextmanager
from copy import deepcopy

from .domain_analyzer import DomainAnalysis

logger = logging.getLogger(__name__)

# References to the original dictionaries from operators.py
_original_synonym_dict = None
_original_modifiers_dict = None


def _backup_original_dictionaries():
    """Backup the original dictionaries from operators.py"""
    global _original_synonym_dict, _original_modifiers_dict
    
    try:
        from . import operators
        
        if _original_synonym_dict is None:
            _original_synonym_dict = deepcopy(operators.SYNONYM_DICTIONARY)
            logger.debug("Backed up original SYNONYM_DICTIONARY")
        
        if _original_modifiers_dict is None:
            _original_modifiers_dict = deepcopy(operators.MODIFIERS)
            logger.debug("Backed up original MODIFIERS")
            
    except ImportError as e:
        logger.error(f"Failed to backup original dictionaries: {e}")


def _restore_original_dictionaries():
    """Restore the original dictionaries to operators.py"""
    global _original_synonym_dict, _original_modifiers_dict
    
    try:
        from . import operators
        
        if _original_synonym_dict is not None:
            operators.SYNONYM_DICTIONARY.clear()
            operators.SYNONYM_DICTIONARY.update(_original_synonym_dict)
            logger.debug("Restored original SYNONYM_DICTIONARY")
        
        if _original_modifiers_dict is not None:
            operators.MODIFIERS.clear()
            operators.MODIFIERS.update(_original_modifiers_dict)
            logger.debug("Restored original MODIFIERS")
            
    except ImportError as e:
        logger.error(f"Failed to restore original dictionaries: {e}")


def _inject_dynamic_dictionaries(synonyms: Dict[str, List[str]], modifiers: Dict[str, List[str]]):
    """Inject dynamic dictionaries into operators.py"""
    try:
        from . import operators
        
        # Clear and update synonyms
        operators.SYNONYM_DICTIONARY.clear()
        operators.SYNONYM_DICTIONARY.update(synonyms)
        logger.info(f"Injected {len(synonyms)} synonym categories")
        
        # Clear and update modifiers
        operators.MODIFIERS.clear()
        operators.MODIFIERS.update(modifiers)
        logger.info(f"Injected {len(modifiers)} modifier categories")
        
    except ImportError as e:
        logger.error(f"Failed to inject dynamic dictionaries: {e}")


def create_enhanced_synonyms(domain_analysis: DomainAnalysis) -> Dict[str, List[str]]:
    """
    Create enhanced synonym dictionary from domain analysis
    
    Args:
        domain_analysis: LLM analysis result with suggested synonyms
        
    Returns:
        Enhanced synonym dictionary combining LLM suggestions with fallbacks
    """
    # Start with LLM-generated synonyms
    enhanced_synonyms = deepcopy(domain_analysis.synonyms)
    
    # Add key product-specific synonyms based on analysis
    product_terms = domain_analysis.product_category.lower().split()
    
    # Add domain-specific synonyms
    if 'laptop' in product_terms or 'computer' in product_terms:
        enhanced_synonyms['laptop'] = enhanced_synonyms.get('laptop', []) + [
            'notebook', 'computer', 'machine', 'device', 'system'
        ]
    
    if 'sock' in product_terms or 'socks' in product_terms:
        enhanced_synonyms['sock'] = enhanced_synonyms.get('sock', []) + [
            'hosiery', 'footwear', 'socks'
        ]
        enhanced_synonyms['socks'] = enhanced_synonyms.get('socks', []) + [
            'hosiery', 'footwear', 'sock'
        ]
    
    if 'flour' in product_terms or 'baking' in product_terms:
        enhanced_synonyms['flour'] = enhanced_synonyms.get('flour', []) + [
            'baking flour', 'wheat flour', 'ingredient'
        ]
    
    # Add common action words if not present
    if 'recommend' not in enhanced_synonyms:
        enhanced_synonyms['recommend'] = ['suggest', 'advise', 'propose', 'endorse']
    
    if 'good' not in enhanced_synonyms:
        enhanced_synonyms['good'] = ['excellent', 'great', 'quality', 'top']
    
    if 'best' not in enhanced_synonyms:
        enhanced_synonyms['best'] = ['top', 'finest', 'premier', 'optimal']
    
    # Remove duplicates and empty lists
    cleaned_synonyms = {}
    for key, synonyms in enhanced_synonyms.items():
        if synonyms:
            cleaned_synonyms[key] = list(set(synonyms))
    
    logger.info(f"Created enhanced synonyms with {len(cleaned_synonyms)} categories")
    return cleaned_synonyms


def create_enhanced_modifiers(domain_analysis: DomainAnalysis) -> Dict[str, List[str]]:
    """
    Create enhanced modifier dictionary from domain analysis
    
    Args:
        domain_analysis: LLM analysis result with suggested modifiers
        
    Returns:
        Enhanced modifier dictionary combining LLM suggestions with fallbacks
    """
    # Start with LLM-generated modifiers
    enhanced_modifiers = deepcopy(domain_analysis.modifiers)
    
    # Add fallback modifiers if categories are missing
    if 'quality' not in enhanced_modifiers:
        enhanced_modifiers['quality'] = ['premium', 'professional', 'quality', 'reliable']
    
    if 'performance' not in enhanced_modifiers:
        enhanced_modifiers['performance'] = ['high-performance', 'efficient', 'powerful']
    
    if 'design' not in enhanced_modifiers:
        enhanced_modifiers['design'] = ['modern', 'stylish', 'elegant', 'sleek']
    
    if 'value' not in enhanced_modifiers:
        enhanced_modifiers['value'] = ['affordable', 'cost-effective', 'budget-friendly']
    
    # Remove duplicates and empty lists
    cleaned_modifiers = {}
    for key, modifiers in enhanced_modifiers.items():
        if modifiers:
            cleaned_modifiers[key] = list(set(modifiers))
    
    logger.info(f"Created enhanced modifiers with {len(cleaned_modifiers)} categories")
    return cleaned_modifiers


@contextmanager
def dynamic_dictionaries(domain_analysis: Optional[DomainAnalysis] = None):
    """
    Context manager for temporarily injecting dynamic dictionaries
    
    Args:
        domain_analysis: Optional domain analysis with LLM-generated content
        
    Usage:
        with dynamic_dictionaries(domain_analysis):
            # Mutation operators will use LLM-generated synonyms/modifiers
            result = some_mutation_function(prompt)
    """
    if domain_analysis is None:
        logger.debug("No domain analysis provided, using original dictionaries")
        yield
        return
    
    # Backup original dictionaries
    _backup_original_dictionaries()
    
    try:
        # Create enhanced dictionaries
        enhanced_synonyms = create_enhanced_synonyms(domain_analysis)
        enhanced_modifiers = create_enhanced_modifiers(domain_analysis)
        
        # Inject dynamic dictionaries
        _inject_dynamic_dictionaries(enhanced_synonyms, enhanced_modifiers)
        
        logger.info("Dynamic dictionaries injected successfully")
        yield
        
    finally:
        # Always restore original dictionaries
        _restore_original_dictionaries()
        logger.info("Original dictionaries restored")


def apply_dynamic_injection(
    domain_analysis: DomainAnalysis,
    synonyms: Optional[Dict[str, List[str]]] = None,
    modifiers: Optional[Dict[str, List[str]]] = None
) -> None:
    """
    Apply dynamic injection permanently (until restore is called)
    
    Args:
        domain_analysis: Domain analysis with LLM-generated content
        synonyms: Optional custom synonyms (overrides domain analysis)
        modifiers: Optional custom modifiers (overrides domain analysis)
    """
    _backup_original_dictionaries()
    
    # Use provided dictionaries or create from domain analysis
    if synonyms is None:
        synonyms = create_enhanced_synonyms(domain_analysis)
    
    if modifiers is None:
        modifiers = create_enhanced_modifiers(domain_analysis)
    
    # Inject the dictionaries
    _inject_dynamic_dictionaries(synonyms, modifiers)
    
    logger.info("Dynamic injection applied permanently")


def restore_original_dictionaries() -> None:
    """
    Restore the original static dictionaries
    """
    _restore_original_dictionaries()
    logger.info("Original dictionaries restored")


def get_current_dictionaries() -> Dict[str, Any]:
    """
    Get the current state of the dictionaries
    
    Returns:
        Dictionary with current synonyms and modifiers
    """
    try:
        from . import operators
        
        return {
            'synonyms': deepcopy(operators.SYNONYM_DICTIONARY),
            'modifiers': deepcopy(operators.MODIFIERS),
            'synonyms_count': len(operators.SYNONYM_DICTIONARY),
            'modifiers_count': len(operators.MODIFIERS),
        }
    except ImportError:
        return {
            'synonyms': {},
            'modifiers': {},
            'synonyms_count': 0,
            'modifiers_count': 0,
        }


def preview_dynamic_injection(domain_analysis: DomainAnalysis) -> Dict[str, Any]:
    """
    Preview what the dynamic injection would look like without applying it
    
    Args:
        domain_analysis: Domain analysis with LLM-generated content
        
    Returns:
        Preview of enhanced dictionaries
    """
    enhanced_synonyms = create_enhanced_synonyms(domain_analysis)
    enhanced_modifiers = create_enhanced_modifiers(domain_analysis)
    
    return {
        'enhanced_synonyms': enhanced_synonyms,
        'enhanced_modifiers': enhanced_modifiers,
        'synonyms_count': len(enhanced_synonyms),
        'modifiers_count': len(enhanced_modifiers),
        'original_synonyms_count': len(_original_synonym_dict) if _original_synonym_dict else 0,
        'original_modifiers_count': len(_original_modifiers_dict) if _original_modifiers_dict else 0,
    } 