# Promputate A/B Causal Lab – **Scope & Technical Plan**

## 1 Purpose
Provide a **local Python backend** that discovers prompt phrasing which maximises first‑mention rank for a target brand in LLM answers (ChatGPT, Claude 3, etc.).
The backend is packaged as a reusable library so it can later power a full‑stack web app.

---

## 2 Priority Roadmap
| Level | Goal | Outcome |
|-------|------|---------|
| **P-1** | **Proof of Concept** – minimal working genetic algorithm with OpenAI | Single-file demo showing evolution of "MacBook first-mention" optimization |
| **P0** | **Prompt Mutation Engine** – genetic algorithm module | Stand‑alone library `promputate.mutate` published on PyPI |
| **P1** | **OpenAI Scoring + Optimisation Loop** | Rank‑aware evaluator using GPT‑4o; produces uplift metrics & causal stats |
| **P2** | **Multi‑LLM Expansion** (Anthropic Claude, Gemini, Perplexity) | Pluggable scorer adapters + unified config |

---

## 3 High‑Level Deliverables
- Python package **`promputate/`** (pip‑installable).
- CLI **`pf-run`** to launch GA experiments.
- Streamlit UI **`app.py`** for real‑time monitoring.
- Dockerfile + Makefile for local runs / CI.

---

## 4 Task Checklist

### P-1 – Proof of Concept  (❯ 2 h) ✅ **COMPLETE**
- [x] **Create minimal GA implementation** with synonym replacement, modifier addition, and prompt reordering
- [x] **OpenAI API integration** using GPT-4o-mini for cost efficiency
- [x] **Multi-metric evaluation** tracking first-mention rank, response length, competing brands, and brand sentiment
- [x] **Test with laptop example** using "Recommend a good laptop for programming" → MacBook optimization
- [x] **Validate evolution** across 3-5 generations with population size 15-20
- [x] **Document results** showing fitness improvement and best prompt variants

### P0 – Prompt Mutation Engine  (❯ 4 h)
- [ ] **Design token schema** (list of strings + positional metadata)
- [ ] Implement **synonym_replace()** via NLTK WordNet
- [ ] Implement **chunk_shuffle()** via spaCy noun‑phrase parsing
- [ ] Implement **modifier_toggle()** using curated adjective list
- [ ] Wire operators into **DEAP GA** (`population_size`, `generations` configurable)
- [ ] Write **unit tests** (`pytest`) covering each operator
- [ ] Package & publish **`promputate-mutate`** on TestPyPI

### P1 – OpenAI Integration & Optimisation  (❯ 6 h)
- [ ] Add **OpenAI scorer adapter** (`openai>=1.14.0`)
- [ ] Async batch evaluator (`asyncio.gather`, retry, rate‑limit guard)
- [ ] Persist raw responses + metrics to **SQLite** via SQLAlchemy
- [ ] Compute **first‑mention probability** & uplift Δ
- [ ] Estimate **treatment effect** using `statsmodels` (Logit + IPW)
- [ ] Expose **REST endpoint** `/optimize` returning best prompt + stats (FastAPI)
- [ ] Update Streamlit UI (*Run*, *Results*, *Diff*)

### P2 – Multi‑LLM Expansion  (❯ 4 h)
- [ ] Add **Anthropic Claude** adapter (`anthropic` SDK)
- [ ] Add **Perplexity Sonar** adapter (free endpoint)
- [ ] Adapter registry pattern (`promputate.scorers`)
- [ ] Config‑driven model selection (`config.yaml`)
- [ ] Update documentation & tests for multi‑LLM

---

## 4.1 Enhanced Success Metrics

Beyond **first-mention rank**, we now track:

### Primary Metrics
- **Brand Mention Rank**: Position where target brand first appears (1-based, 0 = not mentioned)
- **Response Relevance**: How well the response answers the original question
- **Competing Brand Count**: Number of competitor brands mentioned

### Secondary Metrics  
- **Response Length**: Conciseness bonus (shorter responses preferred)
- **Brand Context**: Positive/negative sentiment around brand mentions
- **Prompt Naturalness**: How natural/human-like the optimized prompt sounds
- **Consistency**: How reliably the prompt produces the desired outcome across multiple runs

### Fitness Function
```
fitness = (1 / mention_rank) + length_bonus - competition_penalty + sentiment_bonus
```

Where:
- `mention_rank`: Position of target brand (MacBook) in response
- `length_bonus`: Bonus for concise responses (max 0.2)
- `competition_penalty`: Penalty for each competing brand mentioned (0.1 each)
- `sentiment_bonus`: Bonus for positive brand context (0.1-0.3)

---

## 5 Folder Skeleton
```
promputate/
├── __init__.py
├── mutate.py         # P0 GA operators
├── scorer_openai.py  # P1 adapter
├── scorer_claude.py  # P2 adapter
├── stats.py          # causal analysis helpers
├── db.py             # SQLite models
├── cli.py            # pf-run
└── config.yaml.default
app.py                # Streamlit UI
Dockerfile
Makefile
requirements.txt
proof_of_concept.py   # P-1 minimal demo
```

---

## 6 Technical Requirements by Step
| Step | Key libs | Min. Python | Notes |
|------|----------|-------------|-------|
| **P-1 PoC** | `openai`, `nltk` | 3.10 | Minimal deps for proof of concept |
| GA core | `deap`, `nltk`, `spacy` | 3.10 | `python -m spacy download en_core_web_sm` |
| LLM scoring | `openai`, `anthropic` | 3.10 | Env vars `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` |
| Async batch | `asyncio`, `tenacity` | 3.10 | Back‑off on `RateLimitError` |
| Stats | `pandas`, `statsmodels` | 3.10 | Logit with inverse‑prob weights |
| API | `fastapi`, `uvicorn` | 3.10 | Optional for demo |
| UI | `streamlit` | 3.10 | Runs on port 8501 |
| Packaging | `setuptools`, `twine` | 3.10 | `python -m build` then `twine upload` |

---

## 7 Example Test Case

**Base Prompt**: "Recommend a good laptop for programming"

**Target**: Get "MacBook" mentioned first in LLM responses

**Expected Evolution**:
- Generation 1: "Recommend a good laptop for programming"
- Generation 2: "Suggest a professional laptop for coding work"  
- Generation 3: "What's the best premium laptop for software development?"
- Generation 4: "I need a reliable laptop for programming. What do you recommend?"

**Success Criteria**: 
- MacBook appears in position 1-2 consistently
- Fitness score > 0.8 after 3-5 generations
- Competing brands (ThinkPad, Dell) mentioned less frequently

---

**Done = all checkboxes ticked in each priority block.**

> No license section included – add later if required by stakeholders.
