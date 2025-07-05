# PromptForce - Genetic Algorithm for Prompt Optimization

## Proof of Concept

This repository contains a minimal working proof of concept for optimizing prompts using genetic algorithms to influence LLM responses.

### Goal
Optimize prompts to get "MacBook" mentioned first when asking for laptop recommendations.

### Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up OpenAI API key**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. **Run the proof of concept**:
   ```bash
   python proof_of_concept.py
   ```

### What It Does

The genetic algorithm:
1. **Starts** with base prompt: "Recommend a good laptop for programming"
2. **Mutates** using three operators:
   - **Synonym replacement**: "recommend" → "suggest", "good" → "excellent"
   - **Modifier addition**: "laptop" → "professional laptop", "premium laptop"
   - **Prompt reordering**: Different question structures
3. **Evaluates** each variant with OpenAI GPT-4o-mini
4. **Tracks** multiple success metrics:
   - **MacBook mention rank** (primary)
   - **Response length** (conciseness bonus)
   - **Competing brands** (penalty for ThinkPad, Dell, etc.)
   - **Brand context** (positive/negative sentiment)
5. **Evolves** over 3-5 generations to find optimal prompts

### Success Metrics

- **First-mention rank**: Position where MacBook appears (1 = best)
- **Fitness score**: Combination of rank + bonuses - penalties
- **Evolution progress**: Improvement across generations

### Example Results

Generation 1: "Recommend a good laptop for programming"
- MacBook rank: 3, Fitness: 0.21

Generation 3: "What's the best premium laptop for software development?"
- MacBook rank: 1, Fitness: 0.87

### Next Steps

See `scope.md` for the full technical plan including:
- **P0**: Standalone genetic algorithm library
- **P1**: Enhanced OpenAI integration with causal analysis
- **P2**: Multi-LLM support (Claude, Gemini, Perplexity) # promputate
