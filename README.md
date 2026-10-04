# WC_Simulator — FIFA World Cup 2026 Multi-Agent Match Predictor

A multi-agent match-prediction engine built on **LangGraph** and **Google Gemini**
(the `google-genai` SDK, with Google Search grounding for research). It models a
football match as the *collision* of two independently analysed teams — each with
its own squad chemistry, tournament game theory, scouting and tactical plan —
filtered through impartial global factors (live team news, travel and climate,
the betting market, black-swan events), and resolves each fixture in a **single
run** into a full scoreline probability distribution.

The engine covers all 104 matches of the 2026 World Cup: the 72 group games and
the full knockout bracket (Round of 32 → Final). Its output, `data/predictions.json`,
powers the companion dashboard:

- **Live dashboard:** https://wc-agentic-simulator.streamlit.app/
- **Dashboard repo:** https://github.com/Ar-i-yans-Om/Fifa

---

## What one run produces

For a single fixture, one pass through the graph yields:

- the **complete scoreline grid** (0-0 … 7-7) as a probability matrix,
- Win / Draw / Loss probabilities aggregated from that grid,
- the most likely scorelines and a confidence rating,
- a model-vs-market comparison against betting odds,
- an executive headline and narrative written by the Judge.

There is no Monte Carlo. Every black-swan scenario is weighted into the one run
analytically (see [The mathematics](#the-mathematics)).

---

## Architecture

The graph enforces a hard split between **common** nodes (impartial, see both
teams) and **independent** nodes (instantiated once per team, isolated from the
other side).

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

| Agent | Type | Role | Output |
|---|---|---|---|
| **Manager** | common | Entry node; reads the frozen scenario and seeds the trace | — |
| **Researcher** | common | Head analyst + physiologist. One Google-Search-grounded brief covering both teams (results, injuries, suspensions, likely XIs, standings, press conferences, head-to-head), cross-referenced against the registered 26-man squads. Fitness is computed deterministically. | `TeamForm`, fitness profile |
| **Alchemist** | independent | Squad chemistry: clusters players who share a club; more and larger blocks → higher cohesion | `CohesionReport` |
| **Strategist** | independent | Tournament game theory: qualification maths, bracket paths, rotation vs push (knockout ties are always Full Intensity) | `StrategyReport` |
| **Scout** | independent | Sees only the opponent's *public* profile; finds exploitable weaknesses | `ScoutReport` |
| **Tactician** | independent | Builds a *blind* game plan (never sees the opponent's plan): one of five styles, a formation, Plan A/B | `TacticalPlan` |
| **Pitch Simulator** | common | The only node that sees both plans. Sets each side's baseline xG, then fuses in the quantified levers | `PitchResult` |
| **Chaos Agent** | common | Emits the probability-weighted black-swan scenario set (no random roll) | `ChaosModel` |
| **Bookmaker** | common | Market anchor: live odds from the-odds-api.com, else a grounded market scan | `BookmakerReport` |
| **Judge** | common | Builds the mixture of Poisson grids, derives W/D/L, scorelines and confidence, compares with the market, writes the report | `PoissonReport`, `FinalReport` |

**Isolation is structural.** Team A's chain has zero edges to Team B's; the Scout
reads the opponent only through `public_opponent_view` (name, rank, results,
injuries, suspensions, publicly reported weaknesses). The two four-node chains
are balanced, so `tactician_A` and `tactician_B` finish in the same LangGraph
superstep and the fan-in at `pitch_simulator` fires exactly once (covered by a
test). The Bookmaker terminates at `END`; the Judge reads its result from shared
state, which avoids a second trigger downstream.

**Numbers are deterministic; the LLM supplies judgement and prose.** Gemini
decides the qualitative inputs (baseline xG, strategy mode, tactical fit, club
clusters) inside code-enforced guardrails, and writes the Judge's prose. Every
probability is computed in `math_engine.py`.

---

## The mathematics

All of this lives in `fifa_predictor/math_engine.py` (pure, no I/O) and
`fifa_predictor/data_sources.py`.

**ELO anchor.** Each side's statistical expected goals:

```
p_win    = 1 / (1 + 10^(-(ELO_team - ELO_opp) / 400))
λ_anchor = tournament_avg × (2 · p_win)^0.75            clamped to [0.3, 3.5]
```

`tournament_avg` is 1.35. The 0.75 exponent keeps the anchor conservative: the
favourite gains less than the underdog loses, so lopsided fixtures aren't
over-stated before the agents weigh in.

**Home advantage.** Applied only to a host nation playing *in its own country*
(e.g. Mexico in Mexico City): ×1.18 for the home side, ×0.94 for the visitor.

**Baseline xG fusion.** Gemini (Pitch Simulator) sets a clean tactical/ELO
baseline per side and is told explicitly *not* to include chemistry, intensity,
fitness or black-swan events. The quantified levers are then multiplied in:

```
λ = base × cohesion × intensity × tactical_fit × fitness × momentum × dark_horse
                                                        clamped to [0.3, 3.5]
```

| Lever | Source | Range |
|---|---|---|
| cohesion | Alchemist clusters, diminishing returns | 1.00 – 1.08 |
| intensity | Strategist mode: Passive/Rotation 0.78–0.95, Full Intensity 1.02–1.14, Targeted Bracket 0.90–1.02 (knockouts: Full Intensity only) | 0.78 – 1.14 |
| tactical_fit | Tactician self-assessment against the scouted weaknesses | 0.88 – 1.15 |
| fitness | Physiologist formula (below) | 0.80 – 1.06 |
| momentum | Actual results so far: 0.03 × goal difference per match (capped ±0.09), averaged | 0.90 – 1.10 |
| dark_horse | Manual per-team factor in `team_ratings.json` (1.0 if unlisted) | — |

**Physiologist (fitness) factor.**

```
fitness = clamp(1 − rest − travel − altitude − climate + host, 0.80, 1.06)
```

- rest: 3% per day below a 4-day baseline (max 12%)
- travel: up to 6%, linear to 8,000 km (Haversine between the real host-city coordinates)
- altitude: up to 5% above 1,500 m for teams not acclimatised to altitude
- climate: up to 4% when the venue is hotter or more humid than the team's home climate
- host: +4% for the three host nations

**Chaos as a mixture, not a dice roll.** With total black-swan probability `p`
(default 0.3), the Chaos Agent enumerates the no-chaos baseline (weight `1 − p`)
plus red card / VAR penalty / freak injury × each team. The split of `p` across
those six scenarios follows each team's `chaos_profile` in `team_ratings.json`.
Severity is fixed: a red card or an injury is applied over the expected half of
the match that remains when it strikes, and a VAR penalty adds 0.76 xG to the
side awarded it. The Judge then combines one Poisson grid per scenario:

```
P(a, b) = (1 − p) · Grid(λ_A, λ_B)  +  Σ_i p_i · Grid(λ_A^i, λ_B^i)
```

Blending the λs first and running a single Poisson would collapse the fat tails
that chaos exists to model; the mixture keeps them at the same mean.

**Poisson grid.** `P(X = k) = λ^k e^(−λ) / k!`, independent per side, over
0 … `max_goals` (default 7), renormalised for truncation, then aggregated into
W/D/L. The predicted scoreline is the grid's most likely cell. Confidence is
*high* if the leading outcome is ≥ 55% or leads the next by ≥ 20 points,
*medium* at ≥ 42% / ≥ 10 points, otherwise *low*.

---

## Data

Everything the runner needs is in `data/`:

| File | Contents |
|---|---|
| `fixtures.json` | The 104-match schedule (id, group or round, date, venue, city). Knockout ties (`M73`–`M104`) carry a `round`, a `match_no` and slot references instead of team names. |
| `team_ratings.json` | Per team: ELO (worldfootballrankings.com, 17 Jun 2026), FIFA rank (11 Jun 2026), confederation and `chaos_profile`, plus the optional `dark_horses` map. Chaos profiles are heuristic (a confederation baseline plus documented team tweaks), not per-match statistics. |
| `results.json` | Played scores: `{id, home_score, away_score, played}`. A knockout tie level after extra time adds `"winner": "<team>"` (the shootout winner). |
| `players.json` | Registered 26-man squads for all 48 teams (number, position, date of birth, club, caps, goals). The Researcher uses it as the roster reference and as the Alchemist's club map. |
| `predictions.json` | Output, keyed by fixture id (see below). |

**Knockout bracket.** `resolve_bracket()` in `match_runner.py` turns slot
references into teams as results arrive. `1A` / `2B` / `3E` are group placings,
filled once a group's six matches are all played and ranked per FIFA World Cup 26
Regulations, Article 13 (`rank_group()`): points; then, for teams level on
points, head-to-head points, goal difference and goals scored (re-applied to any
teams still level); then overall goal difference and goals scored; then the FIFA
world ranking. Team conduct (cards) isn't tracked, so that criterion is skipped.
`W77` / `L101` are the winner / loser of a match. Ties whose feeders aren't
decided yet are skipped.

Knockout scenarios carry their round (`knockout_round`, e.g. `R16`), and the
agents switch to knockout context: the Researcher's brief covers each side's
route and what awaits the winner instead of the group table, the Strategist
plays Full Intensity, and no group points are shown. Travel, rest and momentum
still come from each team's actual previous match.

**`predictions.json` contract** (one entry per fixture):

```json
{
  "fixture_id": "M90", "round": "R16", "match_no": 90,
  "home": "Canada", "away": "Morocco",
  "prob_home_win": 20, "prob_draw": 19, "prob_away_win": 62,
  "expected_goals": {"home": 1.26, "away": 2.37},
  "predicted_scoreline": "Canada 1-2 Morocco",
  "top_scorelines": [{"score": "1-2", "prob": 9}, {"score": "1-1", "prob": 8}, {"score": "1-3", "prob": 7}],
  "top_outcome": {"team": "Morocco", "score": "3-1", "prob": 7},
  "scoreline_grid": [[0.02491, 0.0594, "…"], "…"],
  "confidence": "high",
  "headline": "…", "narrative": "…", "model_vs_market": "…",
  "market_probs": {"implied_home_win": 0.2039, "implied_draw": 0.1203, "implied_away_win": 0.6758},
  "quality": {"ok": true, "grounded": true, "n_issues": 0, "issues": []},
  "updated_at": "…"
}
```

- Probabilities are integer percentages; `scoreline_grid[i][j]` is P(home i, away j).
- `expected_goals` is each side's fused baseline λ. The grid's own mean differs
  slightly because it also folds in the chaos scenarios.
- `top_outcome` is the most emphatic scoreline that still carries more than 5%.
- `quality.ok` records whether every LLM call took its optimal route, and
  `quality.grounded` whether research was backed by live Google Search results.
- The format evolved during the tournament: the matchday-1 fixtures of groups
  E–L used a 0-0 … 4-4 grid (`max_goals` 4), and `market_probs` / `quality` are
  absent from entries written before those fields were introduced.
  `market_probs` is null when no market anchor was recorded.

---

## Install

```bash
pip install -r requirements.txt
```

Create a `.env` file at the repository root (a free key is available from
aistudio.google.com):

```
GEMINI_API_KEY=...
# optional
GEMINI_API_KEYS=key2,key3      # failover pool for quota/auth errors
ODDS_API_KEY=...               # live odds for the Bookmaker
```

## Run

```bash
# Data-driven runner (reads data/, writes data/predictions.json)
python match_runner.py                       # every fixture not yet played
python match_runner.py --fixture K5          # one fixture (or several: --fixture K5 K6)
python match_runner.py --group K             # a whole group
python match_runner.py --md 3                # a matchday
python match_runner.py --round R16           # a knockout round: R32 / R16 / QF / SF / 3P / F
python match_runner.py --all                 # include fixtures already played
python match_runner.py --rerun-degraded      # re-run entries whose quality.ok is false
python match_runner.py --no-chaos            # baseline only (p = 0)
python match_runner.py --no-write            # print, don't touch predictions.json
python match_runner.py --verbose             # DEBUG-level agent logs

# Single-fixture demo (K5, Portugal vs Colombia) with the full report and trace
python -m fifa_predictor.main
python -m fifa_predictor.main --print-graph  # Mermaid diagram of the graph, no key needed
```

During a tournament, enter each result in `data/results.json` as it is played;
group points, momentum, travel and the bracket are all recomputed from that file.

Programmatic use:

```python
from fifa_predictor import Settings, demo_scenario, run_prediction

final = run_prediction(settings=Settings.from_env(), scenario=demo_scenario())
print(final["report"]["win_draw_loss"])
print(final["poisson"]["grid"])        # full 0-0..7-7 probability matrix
```

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | Primary Gemini key (required for live runs) |
| `GEMINI_API_KEYS` | — | Extra keys (comma / space / newline separated), rotated on auth, quota or overload errors |
| `FIFA_RESEARCH_MODEL` | `gemini-2.5-flash-lite` | Grounded research tier (Researcher, Bookmaker) |
| `FIFA_REASONING_MODEL` | `gemini-3.1-flash-lite` | Ungrounded agents and structuring (the bulk of calls) |
| `FIFA_JUDGE_MODEL` | `gemini-3.1-flash-lite` | Judge prose |
| `FIFA_FALLBACK_MODELS` | _(empty)_ | Optional model chain tried on 503 / overload before rotating keys |
| `FIFA_DISABLE_GROUNDING` | `0` | `1` = research from model knowledge only |
| `FIFA_GROUNDING_RETRIES` | `2` | Re-rolls when a grounded call returns text without actually searching |
| `FIFA_MAX_TOKENS` | `10000` | Output-token budget per call (floors apply for thinking models) |
| `FIFA_THINKING_BUDGET` | `-1` | Thinking-token cap (`-1` = model default, `0` = off) |
| `FIFA_RESPONSE_SCHEMA` | `1` | Use Gemini's native `response_schema` for structured output |
| `FIFA_LLM_STRICT` | `1` | A call that still fails after all retries aborts the fixture instead of returning placeholder defaults |
| `ODDS_API_KEY` | — | the-odds-api.com key for live H2H odds |
| `FIFA_CHAOS_PROB` | `0.3` | Total black-swan probability (`0` = off) |
| `FIFA_MAX_GOALS` | `7` | Grid spans 0-0 … N-N |

The research and reasoning tiers use different models on purpose: Gemini's free
tier meters quota per model, so the split gives two independent quota buckets
per key.

---

## Tests

```bash
python -m pytest tests/ -q
```

The suite runs offline with no API key: a stub LLM is injected and all the maths
runs for real. `test_math.py` covers the Poisson grid, the chaos scenario set
(with and without team profiles), the mixture's tail behaviour and the ELO
anchor. `test_graph_structure.py` checks the topology, an end-to-end run to the
Judge, the single fan-in, chaos weighting, Scout isolation, the physiologist,
the Strategist guardrails, scenario building from the data files, that the demo
equals the data-driven K5 scenario, the FIFA group-ranking rules (head-to-head
before goal difference, the world-ranking fallback, unfinished groups staying
unresolved), knockout-aware briefs, and that the bracket resolves real ties
(France–Sweden in the Round of 32, Spain–Argentina in the Final).

---

## Project layout

```
match_runner.py          data-driven runner: loads data/, resolves the bracket, writes predictions.json
requirements.txt
data/
  fixtures.json  team_ratings.json  results.json  players.json  predictions.json
  decode_unicode.py      utility: rewrite a JSON file with readable UTF-8 names
fifa_predictor/
  config.py              Settings, TeamScenario / MatchScenario, demo_scenario()
  schemas.py             Pydantic payloads + the LangGraph AgentState
  graph.py               graph assembly, run_prediction(), Mermaid export
  llm.py                 Gemini wrapper: structured() and research(), retries, key failover, quality ledger
  math_engine.py         ELO anchor, home advantage, Poisson grid, chaos scenarios, mixture
  data_sources.py        host cities, Haversine, climate, fitness formula, odds API
  logger.py              logging setup
  main.py                single-fixture CLI demo
  agents/
    _helpers.py          isolation helpers (public_opponent_view, …)
    common/              manager, researcher, pitch_simulator, chaos_agent, bookmaker, judge
    independent/         alchemist, strategist, scout, tactician (one instance per team)
tests/
  test_math.py  test_graph_structure.py
```
