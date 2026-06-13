"""
tests/test_math.py
==================

Unit tests for the deterministic numeric core (no API key, no network).
Covers the independent-Poisson grid, the chaos scenario set, and the
mixture-of-grids combination.
"""

import math

from fifa_predictor.math_engine import (
    DEFAULT_MAX_GOALS,
    build_chaos_scenarios,
    elo_base_lambda,
    mixture_grid,
    poisson_grid,
    _poisson_pmf,
)


# ---------------------------------------------------------------------------
# Poisson primitives + single-scenario grid
# ---------------------------------------------------------------------------
def test_poisson_pmf_matches_formula():
    for lam in (0.5, 1.3, 2.7):
        for k in range(0, 6):
            expected = (lam ** k) * math.exp(-lam) / math.factorial(k)
            assert abs(_poisson_pmf(k, lam) - expected) < 1e-12


def test_poisson_grid_probabilities_sum_to_one():
    grid = poisson_grid(1.7, 1.1)
    total = grid["home_win"] + grid["draw"] + grid["away_win"]
    assert abs(total - 1.0) < 5e-4


def test_poisson_grid_default_spans_zero_to_seven():
    grid = poisson_grid(1.4, 1.2)
    assert grid["max_goals"] == DEFAULT_MAX_GOALS == 7
    assert len(grid["grid"]) == 8 and len(grid["grid"][0]) == 8
    # the full matrix itself sums to ~1.0
    assert abs(sum(p for row in grid["grid"] for p in row) - 1.0) < 1e-4


def test_poisson_grid_favours_higher_lambda():
    grid = poisson_grid(2.4, 0.8)
    assert grid["home_win"] > grid["away_win"]
    assert grid["home_win"] > grid["draw"]


def test_poisson_grid_symmetry_gives_equal_win_probs():
    grid = poisson_grid(1.5, 1.5)
    assert abs(grid["home_win"] - grid["away_win"]) < 1e-9


def test_expected_goals_match_lambda():
    grid = poisson_grid(2.3, 0.9)
    # grid-consistent means are close to the input lambdas (tiny truncation loss)
    assert abs(grid["expected_goals_home"] - 2.3) < 0.02
    assert abs(grid["expected_goals_away"] - 0.9) < 0.01


def test_top_scores_sorted_descending():
    grid = poisson_grid(1.4, 1.2)
    probs = [s["prob"] for s in grid["most_likely_scores"]]
    assert probs == sorted(probs, reverse=True)
    assert len(grid["most_likely_scores"]) == 6


# ---------------------------------------------------------------------------
# Chaos scenario set
# ---------------------------------------------------------------------------
def test_chaos_scenarios_probabilities_sum_to_one():
    for p in (0.0, 0.1, 0.22, 0.5):
        scenarios = build_chaos_scenarios(p)
        assert abs(sum(s["probability"] for s in scenarios) - 1.0) < 1e-6


def test_chaos_baseline_weight_equals_one_minus_p():
    scenarios = build_chaos_scenarios(0.22)
    baseline = next(s for s in scenarios if s["event_type"] == "none")
    assert abs(baseline["probability"] - 0.78) < 1e-6


def test_chaos_zero_probability_is_baseline_only():
    scenarios = build_chaos_scenarios(0.0)
    assert len(scenarios) == 1
    assert scenarios[0]["event_type"] == "none"
    assert scenarios[0]["probability"] == 1.0


def test_chaos_red_card_suppresses_disrupted_boosts_opponent():
    scenarios = build_chaos_scenarios(0.22)
    red_a = next(s for s in scenarios if s["label"] == "straight_red_card_vs_A")
    assert red_a["mult_a"] < 1.0      # team A (disrupted) suppressed
    assert red_a["mult_b"] > 1.0      # team B (opponent) boosted


def test_chaos_var_penalty_is_additive_to_opponent():
    scenarios = build_chaos_scenarios(0.22)
    var_a = next(s for s in scenarios if s["label"] == "var_penalty_vs_A")
    # penalty conceded by A -> awarded to B -> additive xG on B, none on A
    assert var_a["add_b"] > 0.0
    assert var_a["add_a"] == 0.0


# ---------------------------------------------------------------------------
# Mixture of grids
# ---------------------------------------------------------------------------
def test_mixture_grid_sums_to_one():
    scenarios = build_chaos_scenarios(0.22)
    grid = mixture_grid(2.3, 0.9, scenarios)
    total = grid["home_win"] + grid["draw"] + grid["away_win"]
    assert abs(total - 1.0) < 1e-4


def test_mixture_with_no_chaos_equals_plain_poisson():
    scenarios = build_chaos_scenarios(0.0)            # baseline only
    mix = mixture_grid(2.3, 0.9, scenarios)
    plain = poisson_grid(2.3, 0.9)
    assert abs(mix["home_win"] - plain["home_win"]) < 1e-9
    assert abs(mix["draw"] - plain["draw"]) < 1e-9


def test_mixture_fattens_tails_versus_blended_lambdas():
    """
    The mixture of grids must preserve more tail mass than blending the lambdas
    and running a single Poisson (which collapses the bimodality).
    """
    scenarios = build_chaos_scenarios(0.22)
    mix = mixture_grid(2.3, 0.9, scenarios)

    # Blend lambdas first, then a single Poisson grid.
    wa = sum(s["probability"] * (2.3 * s["mult_a"] + s["add_a"]) for s in scenarios)
    wb = sum(s["probability"] * (0.9 * s["mult_b"] + s["add_b"]) for s in scenarios)
    blended = poisson_grid(wa, wb)

    def tail(grid):  # P(home scores >= 4)
        return sum(p for i, row in enumerate(grid["grid"]) for p in row if i >= 4)

    # same expected goals, but the mixture keeps a heavier tail
    assert abs(mix["expected_goals_home"] - blended["expected_goals_home"]) < 0.02
    assert tail(mix) >= tail(blended)


# ---------------------------------------------------------------------------
# ELO anchor
# ---------------------------------------------------------------------------
def test_elo_base_lambda_ordering():
    assert elo_base_lambda(1876, 1500) > elo_base_lambda(1500, 1876)


def test_elo_base_lambda_equal_teams():
    assert abs(elo_base_lambda(1700, 1700, tournament_avg=1.35) - 1.35) < 0.01
