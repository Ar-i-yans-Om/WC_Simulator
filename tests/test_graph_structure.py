"""
tests/test_graph_structure.py
=============================

Offline structural tests. No API key, no network: the LLM layer is replaced by
an injected stub (FakeLLM) that returns schema defaults, exactly as the graph
expects. All maths (fitness, ELO anchor, Poisson, chaos mixture) runs for real.

Verifies:
- Graph compiles with the expected nodes and NO chaos-loop nodes
- End-to-end pipeline reaches the Judge with a full mixture Poisson grid
- The fan-in at pitch_simulator fires exactly once (balanced chains)
- Chaos is weighted into a single run (not sampled) and shifts the distribution
- Branch isolation: the Scout only sees public opponent fields
- Physiologist distinguishes teams with different travel/rest
"""

from fifa_predictor.agents._helpers import _PUBLIC_FORM_FIELDS, public_opponent_view
from fifa_predictor.config import MatchScenario, Settings, default_portugal_france_scenario
from fifa_predictor.graph import build_graph, run_prediction


# ---------------------------------------------------------------------------
# Stub LLM: returns schema defaults for structured() and "" for research().
# This is the offline substitute for live Gemini calls (injected via `llms`).
# ---------------------------------------------------------------------------
class FakeLLM:
    def structured(self, schema, system, user, context=""):
        return schema()

    def research(self, system, user, max_searches=6):
        return ""


def _fake_llms() -> dict:
    one = FakeLLM()
    return {"research": one, "reasoning": one, "judge": one}


def _settings(chaos_prob: float = 0.22) -> Settings:
    return Settings(gemini_api_key=None, chaos_base_probability=chaos_prob)


def _scenario() -> MatchScenario:
    return default_portugal_france_scenario()


def _run(chaos_prob: float = 0.22) -> dict:
    return run_prediction(settings=_settings(chaos_prob),
                          scenario=_scenario(), llms=_fake_llms())


# ---------------------------------------------------------------------------
# Graph topology
# ---------------------------------------------------------------------------
def test_graph_compiles_with_all_nodes():
    app = build_graph(_settings(), llms=_fake_llms())
    nodes = set(app.get_graph().nodes.keys())
    expected = {
        "manager", "researcher", "bookmaker",
        "pitch_simulator", "chaos_agent", "judge",
        "alchemist_a", "strategist_a", "scout_a", "tactician_a",
        "alchemist_b", "strategist_b", "scout_b", "tactician_b",
    }
    assert expected.issubset(nodes)


def test_graph_has_no_chaos_loop_nodes():
    app = build_graph(_settings(), llms=_fake_llms())
    nodes = set(app.get_graph().nodes.keys())
    for dead in ("chaos_readapt_tactician", "chaos_readapt_alchemist", "pitch_recalc"):
        assert dead not in nodes


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------
def test_end_to_end_reaches_judge_with_poisson_grid():
    final = _run()
    assert final.get("report")
    p = final.get("poisson")
    assert p
    assert abs(p["home_win"] + p["draw"] + p["away_win"] - 1.0) < 1e-3
    assert final["report"]["predicted_scoreline"] != ""
    assert "bookmaker" in final


def test_poisson_output_is_full_grid():
    p = _run()["poisson"]
    assert p["max_goals"] == 7
    assert len(p["grid"]) == 8 and len(p["grid"][0]) == 8


def test_team_names_never_unknown():
    pitch = _run().get("pitch", {})
    assert pitch.get("team_a", {}).get("name") == "Portugal"
    assert pitch.get("team_b", {}).get("name") == "France"


def test_pitch_simulator_fires_exactly_once():
    final = _run(chaos_prob=0.0)
    runs = [l for l in final["step_log"] if l.startswith("[PitchSim]")]
    assert len(runs) == 1


# ---------------------------------------------------------------------------
# Chaos: weighted into a single run, not sampled
# ---------------------------------------------------------------------------
def test_chaos_is_weighted_scenario_set():
    chaos = _run().get("chaos", {})
    scenarios = chaos.get("scenarios", [])
    assert len(scenarios) > 1                       # baseline + events
    assert abs(sum(s["probability"] for s in scenarios) - 1.0) < 1e-6
    assert chaos.get("summary")


def test_no_chaos_is_baseline_only():
    chaos = _run(chaos_prob=0.0).get("chaos", {})
    scenarios = chaos.get("scenarios", [])
    assert len(scenarios) == 1
    assert scenarios[0]["event_type"] == "none"


def test_chaos_shifts_distribution_within_one_run():
    """Same (deterministic) baseline lambdas; only the chaos weight differs."""
    base = _run(chaos_prob=0.0)["poisson"]
    chaotic = _run(chaos_prob=0.22)["poisson"]
    moved = (
        abs(base["home_win"] - chaotic["home_win"])
        + abs(base["draw"] - chaotic["draw"])
        + abs(base["away_win"] - chaotic["away_win"])
    )
    assert moved > 1e-4    # chaos meaningfully changes the distribution


# ---------------------------------------------------------------------------
# Branch isolation
# ---------------------------------------------------------------------------
def test_scout_sees_only_public_fields():
    state = {
        "teams": {
            "B": {
                "form": {
                    "name": "France", "fifa_rank": 1,
                    "recent_results": ["W", "W"],
                    "injuries": ["Key Striker"],
                    "suspensions": [],
                    "public_weaknesses": ["high line"],
                    "squad_clubs": {"Mbappe": "Real Madrid"},   # PRIVATE
                    "avg_goals_scored": 1.9,                     # PRIVATE
                },
                "cohesion":  {"cohesion_multiplier": 1.1},       # PRIVATE
                "tactics":   {"style": "possession"},             # PRIVATE
            }
        }
    }
    view = public_opponent_view(state, my_key="A")
    assert set(view.keys()).issubset(_PUBLIC_FORM_FIELDS)
    assert "squad_clubs" not in view
    assert "avg_goals_scored" not in view
    assert "cohesion" not in view
    assert view["name"] == "France"


# ---------------------------------------------------------------------------
# Physiologist (deterministic, from scheduling facts)
# ---------------------------------------------------------------------------
def test_physiologist_factor_differs_by_travel_and_rest():
    final = _run(chaos_prob=0.0)
    a_fit = final["teams"]["A"]["fitness"]["fitness_degradation_factor"]
    b_fit = final["teams"]["B"]["fitness"]["fitness_degradation_factor"]
    assert a_fit < b_fit   # Portugal travelled (Atlanta->Vancouver); France stayed put


# ---------------------------------------------------------------------------
# Strategist intensity stays within the guardrail range
# ---------------------------------------------------------------------------
def test_strategist_intensity_within_valid_range():
    final = _run(chaos_prob=0.0)
    for key in ("A", "B"):
        v = final["teams"][key]["strategy"]["strategic_intensity_multiplier"]
        assert 0.78 <= v <= 1.14, f"Team {key} intensity {v} outside valid range"


# ---------------------------------------------------------------------------
# ELO derivation
# ---------------------------------------------------------------------------
def test_elo_base_lambda_ordering():
    from fifa_predictor.math_engine import elo_base_lambda
    assert elo_base_lambda(1876, 1500) > elo_base_lambda(1500, 1876)


def test_elo_base_lambda_equal_teams():
    from fifa_predictor.math_engine import elo_base_lambda
    assert abs(elo_base_lambda(1700, 1700, tournament_avg=1.35) - 1.35) < 0.01


# ---------------------------------------------------------------------------
# Match runner data pipeline
# ---------------------------------------------------------------------------
def test_match_runner_builds_scenario_from_data():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from match_runner import build_scenario, load_fixtures, load_ratings, load_results

    fixtures = load_fixtures()
    ratings  = load_ratings()
    results  = load_results()
    assert len(fixtures) >= 72

    k5 = next(f for f in fixtures if f["id"] == "K5")
    scenario = build_scenario(k5, ratings, fixtures, results)
    assert scenario.team_a.name == "Portugal"
    assert scenario.team_b.name == "Colombia"
    assert scenario.team_a.elo  == 1766.18
    assert scenario.team_b.elo  == 1693.09
    assert scenario.team_a.group == "K"


def test_city_alias_resolution():
    from fifa_predictor.data_sources import resolve_city
    assert resolve_city("Inglewood")       == "Los Angeles"
    assert resolve_city("East Rutherford") == "New York"
    assert resolve_city("Foxboro")         == "Boston"
    assert resolve_city("Arlington")       == "Dallas"
    assert resolve_city("Santa Clara")     == "San Francisco"
    assert resolve_city("Vancouver")       == "Vancouver"
