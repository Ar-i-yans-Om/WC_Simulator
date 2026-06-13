# FIFA World Cup 2026 — Multi-Agent Match Predictor

A multi-agent match-prediction system built on **LangGraph** and **Google Gemini**
(via the `google-genai` SDK with Google Search grounding). It models a match as the
*collision* of two independent teams — each with its own squad chemistry,
tournament game theory, scouting and tactical plan — filtered through impartial
global factors (live data, environment, betting market), and resolved in a **single
run** into a full scoreline probability distribution.

The reference fixture is **Matchday 3: Portugal vs France**.

---

## What a run produces

One agentic pass per match yields the **complete scoreline probability grid**
(0-0 … 7-7) plus aggregated Win/Draw/Loss, the most likely scorelines, a market
comparison and an executive narrative. There is no Monte-Carlo over many runs:
every scenario — including black-swan chaos — is weighted into the one run.

### Black-swan chaos is folded in, not sampled

Chaos (red card / VAR penalty / freak injury) is **not** a per-run coin flip.
The Chaos Agent emits the full probability-weighted scenario set, and the Judge
builds a **mixture of Poisson grids** — one grid per scenario (including the
no-chaos baseline), combined by probability:

```
P(a,b) = (1 − p)·Grid(λ_A, λ_B) + Σ_i p_i · Grid(λ_A^i, λ_B^i)
```

This preserves the fat tails a chaos model exists to capture (blending the λ first
and running a single Poisson would erase them), at the same mean. Set
`--no-chaos` (or `FIFA_CHAOS_PROB=0`) for a clean baseline.

---

## Core principle: no mock data

The numeric core — the ELO anchor, the Physiologist fitness factor, and the
Poisson/mixture grid — is **always computed deterministically from real inputs**.
The qualitative layer is supplied by Gemini with Google Search grounding, so it
reads **current, public web data** rather than invented numbers.

| Layer | Source |
|---|---|
| Recent form / injuries / squad clubs / weaknesses | Gemini + Google Search (live) |
| Travel fatigue | Real Haversine distance between actual 2026 host-city coordinates |
| Rest / altitude / host effects | Transparent documented formula |
| Baseline xG (λ) | Gemini reasoning, anchored by the deterministic ELO λ |
| Poisson grid + chaos mixture | `math_engine.py` (pure maths) |
| Betting market anchor | the-odds-api.com (if key set) or Gemini + Google Search |

There is no offline stub mode in the product itself: a live `GEMINI_API_KEY` is
required to run a prediction. (The test suite injects a lightweight stub LLM so the
graph topology and maths can be exercised with no key — see Tests.)

---

## Install

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="..."        # or put it in a .env file (free key from aistudio.google.com)
```

## Run

```bash
# Live prediction (Portugal vs France, Matchday 3)
python -m fifa_predictor.main

# Clean baseline — disable the weighted chaos contribution
python -m fifa_predictor.main --no-chaos

# Print the compiled LangGraph diagram (Mermaid) and exit  (no key needed)
python -m fifa_predictor.main --print-graph
```

Real fixtures via the data-driven runner:

```bash
python match_runner.py --fixture K5     # single fixture by ID
python match_runner.py --group K        # whole group
python match_runner.py --md 1           # a whole matchday
python match_runner.py --no-chaos       # baseline only
```

Programmatic use:

```python
from fifa_predictor import run_prediction, Settings, default_portugal_france_scenario

final = run_prediction(
    settings=Settings.from_env(),
    scenario=default_portugal_france_scenario(),
)
print(final["report"]["win_draw_loss"])
print(final["poisson"]["grid"])          # full 0-0..7-7 probability matrix
```

## Tests

```bash
python -m pytest tests/ -q
```

The suite runs fully offline with no API key: `test_math.py` validates the numeric
core (Poisson mass sums to 1, the chaos scenario weights sum to 1, the mixture
preserves tail mass versus blended lambdas) and `test_graph_structure.py` injects a
stub LLM to prove the graph compiles, runs end-to-end to the Judge with a full
mixture grid, fires the Pitch Simulator exactly once, weights chaos into a single
run, and keeps the team branches isolated.

---

## Architecture

**Common** (impartial, global) nodes are separated from **Independent** (per-team,
isolated) nodes to prevent perfect-information leakage.

### Common / symmetric nodes
1. **Manager** — entry point; validates the frozen config and seeds the trace.
2. **Researcher (Global Pulse)** — impartial data lifeline. Gathers raw form,
   injuries, suspensions, squad clubs and public weaknesses for **both** teams via
   Google Search grounding, and bakes in the **Physiologist** `fitness_degradation_factor`.
3. **Pitch Simulator** — the only node that sees both blind plans. Estimates each
   team's **baseline** xG (λ) for normal play (no black-swan events — those are
   modelled downstream to avoid double-counting), anchored by the ELO λ.
4. **Chaos Agent** — emits the probability-weighted black-swan scenario set
   (baseline + red card / VAR penalty / injury × either team). No random roll.
5. **Bookmaker** — market anchor; implied probabilities from real betting markets.
6. **Judge** — builds the mixture of Poisson grids over all chaos scenarios,
   reconciles model vs market, and writes the executive report.

### Independent nodes (one instance per team)
7. **Alchemist** — club-level chemistry clusters → `cohesion_multiplier`.
8. **Strategist** — tournament game theory (rotation vs full intensity).
9. **Scout** — opponent's **public** weaknesses only (isolation enforced).
10. **Tactician** — fuses availability + intensity + scouting into a *blind* Plan A/B.

Team A's branch has **zero edges** to Team B's branch — isolation is structural.

### Topology

```
manager → researcher ─┬─→ alchemist_A → strategist_A → scout_A → tactician_A ─┐
                      ├─→ alchemist_B → strategist_B → scout_B → tactician_B ─┤
                      └─→ bookmaker → END                                     │
                                                                              ▼
                                                                     pitch_simulator
                                                                              │
                                                                         chaos_agent
                                                                              │
                                                                          judge → END
```

The two 4-node team chains are **balanced**, so `tactician_A` and `tactician_B`
finish in the same LangGraph superstep — the fan-in at `pitch_simulator` fires
exactly once (verified by a test). Chaos is no longer a branch: the Chaos Agent
always runs and the Judge folds every scenario into the one run.

---

## The mathematics

**Baseline λ** — the Pitch Simulator (Gemini) sets each team's expected goals for
normal play, using the deterministic ELO λ as a statistical anchor plus the full
tactical/chemistry/fitness picture.

**Chaos mixture** — `build_chaos_scenarios(p)` enumerates the weighted scenarios
(baseline weight `1 − p`; the rest split across event types and the affected team).
Per-event effects are applied over the *expected* remaining match fraction (a card
averaged over when it strikes); a VAR penalty is additive (~0.76 xG to the awarded
side). `mixture_grid(...)` builds a separate Poisson grid per scenario and combines
them by probability.

**Poisson grid** — `P(X=k) = (λ^k · e^−λ) / k!`, independent for each team, over
scorelines 0-0 … 7-7 (configurable via `FIFA_MAX_GOALS`), renormalised for
truncation, then aggregated into Win/Draw/Loss and the most likely scorelines. The
reported expected goals are the mixture mean, so the displayed xG and the displayed
distribution are always consistent.

---

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | Required for live runs |
| `FIFA_RESEARCH_MODEL` | `gemini-2.0-flash` | Research tier |
| `FIFA_REASONING_MODEL` | `gemini-2.0-flash` | Independent strategy agents + pitch |
| `FIFA_JUDGE_MODEL` | `gemini-1.5-pro` | Final critic / report |
| `FIFA_MAX_SEARCHES` | `6` | Google Search cap per Researcher pass |
| `ODDS_API_KEY` | — | Live H2H odds for the Bookmaker (the-odds-api.com) |
| `FIFA_CHAOS_PROB` | `0.22` | Total black-swan probability woven into the run (0 = off) |
| `FIFA_MAX_GOALS` | `7` | Scoreline grid spans 0-0 … N-N |

---

## Project layout

```
fifa_predictor/
  config.py            Settings + MatchScenario (the Portugal vs France spec)
  schemas.py           Pydantic payloads + the LangGraph AgentState TypedDict
  llm.py               Gemini wrapper: structured() (JSON mode) + research() (Google Search)
  data_sources.py      Real host-city geocoords, Haversine, fitness formula, optional scrapers
  math_engine.py       Pure maths: ELO anchor, Poisson grid, chaos scenarios, mixture grid
  graph.py             LangGraph assembly, run_prediction(), Mermaid export
  main.py              CLI
  agents/
    common/            manager, researcher, pitch_simulator, chaos_agent, bookmaker, judge
    independent/       alchemist, strategist, scout, tactician  (one instance per team)
tests/                 offline math + topology/isolation tests (stub LLM injected)
```
