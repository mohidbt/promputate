"""
Domain Analyzer for LLM-Powered Dynamic Configuration

This module provides intelligent analysis of product domains using LLM APIs
to automatically generate relevant competitors, synonyms, and modifiers.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Check for OpenAI availability
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI package not available. Domain analysis will be limited.")
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None


@dataclass
class DomainAnalysis:
    """Result of LLM-powered domain analysis"""
    product_category: str
    product_type: str
    industry: str
    target_audience: str
    competitors: List[str]
    synonyms: Dict[str, List[str]]
    modifiers: Dict[str, List[str]]
    confidence: float
    analysis_notes: str


class DomainAnalyzer:
    """
    LLM-powered analyzer for product domains
    
    Analyzes user's base prompt and target brand to automatically generate:
    - Relevant competitor brands
    - Domain-specific synonym dictionaries
    - Product-appropriate modifier lists
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize the domain analyzer
        
        Args:
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
            model: OpenAI model to use for analysis
        """
        if not OPENAI_AVAILABLE or AsyncOpenAI is None:
            raise ImportError("OpenAI package required for domain analysis. Install with: pip install openai")
        
        import os
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required for domain analysis")
        
        self.model = model
        self.client = AsyncOpenAI(api_key=self.api_key)
        
        logger.info(f"Domain analyzer initialized with model {model}")
    
    async def analyze_domain(self, base_prompt: str, target_brand: str) -> DomainAnalysis:
        """
        Analyze the product domain from user input
        
        Args:
            base_prompt: The user's base prompt (e.g., "Recommend comfortable socks for running")
            target_brand: The target brand to optimize for (e.g., "Nike")
            
        Returns:
            DomainAnalysis with generated competitors, synonyms, and modifiers
        """
        analysis_prompt = self._create_analysis_prompt(base_prompt, target_brand)
        
        try:
            logger.info(f"Analyzing domain for prompt: '{base_prompt[:50]}...' and brand: '{target_brand}'")
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a marketing and product analysis expert. Analyze product domains and generate comprehensive brand and language data."
                    },
                    {
                        "role": "user", 
                        "content": analysis_prompt
                    }
                ],
                max_tokens=1500,
                temperature=0.3,  # Lower temperature for more consistent analysis
                response_format={"type": "json_object"}
            )
            
            response_text = response.choices[0].message.content
            if not response_text:
                raise ValueError("Empty response from LLM")
            
            # Parse JSON response
            analysis_data = json.loads(response_text)
            
            # Create DomainAnalysis object
            domain_analysis = DomainAnalysis(
                product_category=analysis_data.get("product_category", "Unknown"),
                product_type=analysis_data.get("product_type", "Unknown"),
                industry=analysis_data.get("industry", "Unknown"),
                target_audience=analysis_data.get("target_audience", "General"),
                competitors=analysis_data.get("competitors", []),
                synonyms=analysis_data.get("synonyms", {}),
                modifiers=analysis_data.get("modifiers", {}),
                confidence=analysis_data.get("confidence", 0.8),
                analysis_notes=analysis_data.get("analysis_notes", "")
            )
            
            logger.info(f"Domain analysis complete: {domain_analysis.product_category} "
                       f"with {len(domain_analysis.competitors)} competitors")
            
            return domain_analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            raise ValueError(f"Invalid JSON response from LLM: {e}")
        except Exception as e:
            logger.error(f"Domain analysis failed: {e}")
            raise RuntimeError(f"Domain analysis error: {e}")
    
    def _create_analysis_prompt(self, base_prompt: str, target_brand: str) -> str:
        """Create the analysis prompt for the LLM"""
        return f"""
Analyze this product optimization request and return a JSON response:

Base Prompt: "{base_prompt}"
Target Brand: "{target_brand}"

Please analyze the product domain and return JSON with the following structure:

{{
    "product_category": "specific product category (e.g., 'athletic socks', 'baking flour', 'auto repair service')",
    "product_type": "general product type (e.g., 'apparel', 'food ingredient', 'service')",
    "industry": "industry sector (e.g., 'sportswear', 'food & beverage', 'automotive')",
    "target_audience": "primary target audience (e.g., 'athletes', 'home bakers', 'car owners')",
    "competitors": ["list of 4-6 main competitor brands that compete directly with the target brand"],
    "synonyms": {{
        "key_product_terms": ["alternative words for main product terms"],
        "action_words": ["alternatives for verbs like recommend, suggest, need"],
        "quality_descriptors": ["alternatives for quality words like good, best, excellent"]
    }},
    "modifiers": {{
        "quality": ["quality-focused adjectives for this product type"],
        "performance": ["performance-focused adjectives"],
        "design": ["design/appearance-focused adjectives"],
        "value": ["price/value-focused adjectives"]
    }},
    "confidence": 0.9,
    "analysis_notes": "Brief explanation of the analysis and any important considerations"
}}

Focus on:
1. Competitors should be direct, well-known competitors in the same market segment
2. Synonyms should be natural alternatives that users might search for
3. Modifiers should be relevant to this specific product category
4. Consider the target audience when suggesting language

Return only valid JSON.
"""
    
    async def get_fallback_analysis(self, base_prompt: str, target_brand: str) -> DomainAnalysis:
        """
        Provide fallback analysis when LLM is unavailable
        
        Args:
            base_prompt: The user's base prompt
            target_brand: The target brand
            
        Returns:
            Basic DomainAnalysis with generic suggestions
        """
        logger.warning("Using fallback analysis - LLM unavailable")
        
        # Extract basic product terms from prompt
        prompt_lower = base_prompt.lower()
        
        # Simple keyword-based detection
        if any(term in prompt_lower for term in ['laptop', 'computer', 'programming', 'coding']):
            product_category = "laptop computer"
            competitors = ["ThinkPad", "Dell XPS", "HP Spectre", "Surface Laptop"]
        elif any(term in prompt_lower for term in ['sock', 'footwear', 'athletic', 'running']):
            product_category = "athletic socks"
            competitors = ["Adidas", "Under Armour", "Bombas", "Smartwool"]
        elif any(term in prompt_lower for term in ['flour', 'baking', 'cooking', 'ingredient']):
            product_category = "baking flour"
            competitors = ["King Arthur", "Gold Medal", "Pillsbury", "Bob's Red Mill"]
        else:
            product_category = "general product"
            competitors = ["Brand A", "Brand B", "Brand C", "Brand D"]
        
        return DomainAnalysis(
            product_category=product_category,
            product_type="product",
            industry="general",
            target_audience="consumers",
            competitors=competitors,
            synonyms={
                "good": ["excellent", "great", "quality", "top"],
                "recommend": ["suggest", "advise", "propose"],
                "best": ["top", "finest", "leading", "premier"]
            },
            modifiers={
                "quality": ["premium", "professional", "reliable"],
                "performance": ["high-performance", "efficient", "powerful"],
                "design": ["modern", "stylish", "elegant"],
                "value": ["affordable", "cost-effective", "budget-friendly"]
            },
            confidence=0.3,
            analysis_notes="Fallback analysis - basic keyword detection used"
        )


# Convenience function for easy integration
async def analyze_product_domain(base_prompt: str, target_brand: str, 
                                api_key: Optional[str] = None) -> DomainAnalysis:
    """
    Convenience function to analyze product domain
    
    Args:
        base_prompt: User's base prompt
        target_brand: Target brand to optimize for
        api_key: OpenAI API key (optional)
        
    Returns:
        DomainAnalysis with LLM-generated suggestions
    """
    try:
        analyzer = DomainAnalyzer(api_key=api_key)
        return await analyzer.analyze_domain(base_prompt, target_brand)
    except Exception as e:
        logger.error(f"LLM analysis failed, using fallback: {e}")
        # Create analyzer for fallback
        analyzer = DomainAnalyzer.__new__(DomainAnalyzer)
        return await analyzer.get_fallback_analysis(base_prompt, target_brand) 