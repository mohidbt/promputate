#!/usr/bin/env python3
"""
Multi-Domain Test Script

Test the LLM-powered domain analysis with different product domains
to verify that the system works universally across various product types.
"""

import asyncio
import os
import sys
from typing import List, Dict, Any

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from promputate.domain_analyzer import analyze_product_domain, DomainAnalysis
    from promputate.dynamic_injection import preview_dynamic_injection
except ImportError as e:
    print(f"❌ Failed to import Promputate: {e}")
    print("Make sure you've installed the package: pip install -e .")
    sys.exit(1)


# Test cases covering different product domains
TEST_CASES = [
    {
        'name': 'Athletic Socks',
        'base_prompt': 'Recommend comfortable socks for running and athletics',
        'target_brand': 'Nike',
        'expected_category': 'athletic socks',
        'expected_industry': 'sportswear'
    },
    {
        'name': 'Baking Flour',
        'base_prompt': 'I need high-quality flour for baking bread and pastries',
        'target_brand': 'King Arthur',
        'expected_category': 'baking flour',
        'expected_industry': 'food'
    },
    {
        'name': 'Auto Repair Service',
        'base_prompt': 'Looking for reliable car repair service for my vehicle',
        'target_brand': 'Jiffy Lube',
        'expected_category': 'auto repair',
        'expected_industry': 'automotive'
    },
    {
        'name': 'Smartphone',
        'base_prompt': 'What is the best smartphone for photography and gaming?',
        'target_brand': 'iPhone',
        'expected_category': 'smartphone',
        'expected_industry': 'technology'
    },
    {
        'name': 'Coffee Beans',
        'base_prompt': 'Help me find premium coffee beans for espresso',
        'target_brand': 'Starbucks',
        'expected_category': 'coffee beans',
        'expected_industry': 'food'
    }
]


async def test_domain_analysis(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Test domain analysis for a specific test case"""
    print(f"\n🧪 Testing: {test_case['name']}")
    print(f"📝 Prompt: \"{test_case['base_prompt']}\"")
    print(f"🎯 Brand: {test_case['target_brand']}")
    
    try:
        # Run domain analysis
        domain_analysis = await analyze_product_domain(
            test_case['base_prompt'],
            test_case['target_brand']
        )
        
        # Get injection preview
        injection_preview = preview_dynamic_injection(domain_analysis)
        
        # Display results
        print(f"✅ Analysis complete!")
        print(f"   📊 Product: {domain_analysis.product_category}")
        print(f"   🏭 Industry: {domain_analysis.industry}")
        print(f"   🎯 Audience: {domain_analysis.target_audience}")
        print(f"   📈 Confidence: {domain_analysis.confidence:.1%}")
        print(f"   🏪 Competitors: {', '.join(domain_analysis.competitors[:4])}")
        print(f"   📝 Synonyms: {injection_preview['synonyms_count']} categories")
        print(f"   🎨 Modifiers: {injection_preview['modifiers_count']} categories")
        
        # Show some examples
        if domain_analysis.synonyms:
            example_syns = list(domain_analysis.synonyms.items())[:2]
            for category, synonyms in example_syns:
                print(f"      • {category}: {', '.join(synonyms[:3])}")
        
        if domain_analysis.modifiers:
            example_mods = list(domain_analysis.modifiers.items())[:2]
            for category, modifiers in example_mods:
                print(f"      • {category}: {', '.join(modifiers[:3])}")
        
        return {
            'success': True,
            'domain_analysis': domain_analysis,
            'injection_preview': injection_preview
        }
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }


async def validate_results(results: List[Dict[str, Any]]) -> None:
    """Validate the test results"""
    print("\n" + "=" * 60)
    print("📊 VALIDATION RESULTS")
    print("=" * 60)
    
    successful_tests = sum(1 for r in results if r['success'])
    total_tests = len(results)
    
    print(f"✅ Successful tests: {successful_tests}/{total_tests}")
    print(f"❌ Failed tests: {total_tests - successful_tests}/{total_tests}")
    
    if successful_tests == total_tests:
        print("🎉 All tests passed! Multi-domain functionality is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    # Analysis of successful tests
    successful_analyses = [r for r in results if r['success']]
    
    if successful_analyses:
        print(f"\n📈 Analysis Summary:")
        
        # Average confidence
        avg_confidence = sum(r['domain_analysis'].confidence for r in successful_analyses) / len(successful_analyses)
        print(f"   • Average confidence: {avg_confidence:.1%}")
        
        # Unique industries detected
        industries = set(r['domain_analysis'].industry for r in successful_analyses)
        print(f"   • Industries detected: {', '.join(industries)}")
        
        # Total synonyms/modifiers generated
        total_synonyms = sum(r['injection_preview']['synonyms_count'] for r in successful_analyses)
        total_modifiers = sum(r['injection_preview']['modifiers_count'] for r in successful_analyses)
        print(f"   • Total synonyms generated: {total_synonyms}")
        print(f"   • Total modifiers generated: {total_modifiers}")
        
        # Domain diversity
        product_categories = set(r['domain_analysis'].product_category for r in successful_analyses)
        print(f"   • Product categories: {len(product_categories)}")
    
    print("=" * 60)


async def main():
    """Main test function"""
    print("🧬 Multi-Domain Test Suite")
    print("🧪 Testing LLM-powered domain analysis across different product domains")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("🔑 Please set your OpenAI API key to run this test")
        return
    
    print(f"🔑 API key found: {api_key[:8]}...")
    
    # Run tests
    results = []
    for test_case in TEST_CASES:
        try:
            result = await test_domain_analysis(test_case)
            results.append(result)
        except Exception as e:
            print(f"❌ Test '{test_case['name']}' failed with error: {e}")
            results.append({'success': False, 'error': str(e)})
    
    # Validate results
    await validate_results(results)
    
    print("\n🎯 Multi-domain testing complete!")
    print("💡 If all tests passed, the system is ready for universal product optimization.")


if __name__ == "__main__":
    asyncio.run(main()) 