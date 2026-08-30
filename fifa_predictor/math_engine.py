"""
math_engine.py
==============

The deterministic numeric core. This module is pure (no I/O, no LLM) and is
fully unit-testable.

Responsibilities:

1. `elo_base_lambda` - ELO win-probability -> expected-goals anchor. Passed to
   Gemini in the Pitch Simulator as a statistical anchor.
2. `poisson_grid` - independent-Poisson scoreline matrix (0-0 .. N-N) for a
   single (lambda_home, lambda_away) pair, aggregated into Win/Draw/Loss.
3. The chaos mixture: `build_chaos_scenarios` enumerates the probability-weighted
   black-swan scenarios (including the no-chaos baseline) and `mixture_grid`
   combines a separate Poisson grid per scenario into one distribution.

Why a *mixture* of grids rather than blending lambdas? Blending the expected
goals first and running a single Poisson collapses the result into one unimodal
shape and erases the fat tail that chaos modelling exists to capture. A mixture
of grids keeps each scenario's distinct distribution and only then averages them
- same mean, correct spread. See `mixture_grid`.
"""

from __future__ import annotations

import math
from typing import Dict, List

DEFAULT_MAX_GOALS = 7  # grid spans 0-0 .. 7-7 (wide enough not to truncate strong teams)


# ---------------------------------------------------------------------------
# 1. ELO anchor (the statistical strength signal handed to Gemini)
# ---------------------------------------------------------------------------
# --- ELO anchor tuning constants (the single tuning point for the anchor) -----
ELO_SCALE_DIVISOR = 400.0   # definitional ELO constant; do NOT treat as a knob
# Curvature of expected-goals vs win-probability. THIS is the opinionatedness dial:
#   k < 1  -> muted spread, total goals FALL as the ELO gap widens
#   k = 1  -> total goals flat across all gaps
#   k > 1  -> opinionated spread, total goals RISE as the gap widens   <-- chosen
ELO_LAMBDA_EXPONENT = 0.75
ELO_LAMBDA_FLOOR = 0.3      # binds for the underdog in big mismatches (>~300 ELO)
ELO_LAMBDA_CEIL  = 3.5      # binds for the favourite only at high tournament_avg


def elo_base_lambda(
    team_elo: float,
    opp_elo: float,
    tournament_avg: float = 1.35,
) -> float:
    """
    ELO-based expected-goals anchor.

    Derived from the standard ELO win-probability formula: a higher-rated team
    is expected to score more. Equal teams -> tournament average. In live mode
    Gemini receives this as an anchor and adjusts it from the full tactical and
    contextual picture.

    The curvature exponent (ELO_LAMBDA_EXPONENT) is the key tuning dial. With it
    set above 1.0 the anchor is deliberately opinionated AND total expected goals
    rise as the ELO gap widens (the favourite gains more than the underdog loses).
    """
    exp_diff = 10 ** ((team_elo - opp_elo) / ELO_SCALE_DIVISOR)
    win_prob = exp_diff / (1.0 + exp_diff)
    scale = (win_prob / 0.5) ** ELO_LAMBDA_EXPONENT   # equal teams -> 1.0
    lam = tournament_avg * scale
    return float(max(ELO_LAMBDA_FLOOR, min(ELO_LAMBDA_CEIL, round(lam, 4))))


# ---------------------------------------------------------------------------
# 1b. Home advantage (separate scenario: a host playing in its OWN country)
# ---------------------------------------------------------------------------
# Applied ONLY when the venue city is inside the team's own country — a true
# home World Cup match (e.g. Mexico in Mexico City, USA in Dallas, Canada in
# Toronto). This is distinct from `is_host_nation`, which merely flags the three
# host countries regardless of where a given fixture is played.
#
# Modelled the standard football way: the home side's expected goals are boosted
# and the visitor's are slightly suppressed (crowd, familiarity, no travel).
# Historically World Cup hosts overperform markedly, so the boost is sizeable.
HOME_ADV_BOOST    = 1.18   # multiplier on the home team's lambda
HOME_ADV_SUPPRESS = 0.94   # multiplier on the visiting team's lambda

# Which 2026 host city belongs to which nation. Used to detect a true home game.
CITY_COUNTRY_2026: Dict[str, str] = {
    # Mexico
    "Mexico City": "Mexico", "Guadalajara": "Mexico", "Monterrey": "Mexico",
    # Canada
    "Toronto": "Canada", "Vancouver": "Canada",
    # United States (everything else)
    "Atlanta": "United States", "Boston": "United States", "Dallas": "United States",
    "Houston": "United States", "Kansas City": "United States",
    "Los Angeles": "United States", "Miami": "United States", "New York": "United States",
    "Philadelphia": "United States", "San Francisco": "United States",
    "Seattle": "United States",
}
# team-name aliases that map onto a host country
_TEAM_COUNTRY_ALIASES: Dict[str, str] = {
    "USA": "United States", "United States": "United States",
    "Canada": "Canada", "Mexico": "Mexico",
}


def is_true_home_match(team_name: str, venue_city: str) -> bool:
    """True iff `team_name` is a host nation AND `venue_city` is in that country."""
    country = _TEAM_COUNTRY_ALIASES.get((team_name or "").strip())
    if not country:
        return False
    return CITY_COUNTRY_2026.get((venue_city or "").strip()) == country


def home_advantage_factors(
    team_a_name: str,
    team_b_name: str,
    venue_city: str,
) -> tuple:
    """
    Return (mult_a, mult_b) home-advantage multipliers for this fixture.

    If neither side is playing a true home match (or — pathologically — both
    are), returns (1.0, 1.0) so the anchor is untouched.
    """
    a_home = is_true_home_match(team_a_name, venue_city)
    b_home = is_true_home_match(team_b_name, venue_city)
    if a_home and not b_home:
        return HOME_ADV_BOOST, HOME_ADV_SUPPRESS
    if b_home and not a_home:
        return HOME_ADV_SUPPRESS, HOME_ADV_BOOST
    return 1.0, 1.0


# ---------------------------------------------------------------------------
# 2. Poisson primitives + single-scenario grid
# ---------------------------------------------------------------------------
def _poisson_pmf(k: int, lam: float) -> float:
    """P(X = k) = (lambda^k * e^-lambda) / k!"""
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def _score_matrix(lambda_home: float, lambda_away: float, max_goals: int) -> List[List[float]]:
    """
    Independent-Poisson joint distribution over 0..max_goals for each team,
    renormalised over the truncated grid so it sums to 1.
    """
    home_pmf = [_poisson_pmf(i, lambda_home) for i in range(max_goals + 1)]
    away_pmf = [_poisson_pmf(j, lambda_away) for j in range(max_goals + 1)]
    grid = [[home_pmf[i] * away_pmf[j] for j in range(max_goals + 1)]
            for i in range(max_goals + 1)]
    total = sum(sum(row) for row in grid)
    if total > 0:
        grid = [[p / total for p in row] for row in grid]
    return grid


def _aggregate(grid: List[List[float]], max_goals: int) -> dict:
    """
    Turn a normalised scoreline matrix into a PoissonReport-shaped dict:
    W/D/L, the full grid, the top scorelines, and the (grid-consistent)
    expected goals for each side.
    """
    n = max_goals
    home_win = sum(grid[i][j] for i in range(n + 1) for j in range(n + 1) if i > j)
    draw     = sum(grid[i][i] for i in range(n + 1))
    away_win = sum(grid[i][j] for i in range(n + 1) for j in range(n + 1) if i < j)

    eg_home = sum(i * sum(grid[i][j] for j in range(n + 1)) for i in range(n + 1))
    eg_away = sum(j * sum(grid[i][j] for i in range(n + 1)) for j in range(n + 1))

    scores = [
        {"home": i, "away": j, "prob": round(grid[i][j], 4)}
        for i in range(n + 1) for j in range(n + 1)
    ]
    scores.sort(key=lambda s: s["prob"], reverse=True)

    return {
        "home_win": round(home_win, 4),
        "draw": round(draw, 4),
        "away_win": round(away_win, 4),
        "most_likely_scores": scores[:6],
        "expected_goals_home": round(eg_home, 3),
        "expected_goals_away": round(eg_away, 3),
        "grid": [[round(p, 6) for p in row] for row in grid],
        "max_goals": n,
    }


def poisson_grid(lambda_home: float, lambda_away: float,
                 max_goals: int = DEFAULT_MAX_GOALS) -> dict:
    """
    Full scoreline probability matrix (0-0 .. max-max) under independent Poisson,
    aggregated into Win/Draw/Loss. Returns a dict shaped like schemas.PoissonReport
    (now including the complete `grid`).
    """
    return _aggregate(_score_matrix(lambda_home, lambda_away, max_goals), max_goals)


# ---------------------------------------------------------------------------
# 3. Chaos as an in-run, probability-weighted mixture
# ---------------------------------------------------------------------------
# Each black-swan scenario is modelled by its *expected* effect on the two base
# lambdas. The per-event factors below are applied only over the expected
# fraction of the match that remains after the event strikes (a card in minute
# 10 hurts more than one in minute 85, so we integrate over timing into a single
# expected multiplier). A VAR penalty is modelled additively (a converted
# spot-kick adds ~0.76 expected goals to the awarded side) rather than
# multiplicatively, because a multiplier would scale wrongly for a low-lambda
# team.
#
# These constants are the single tuning point for chaos behaviour.
# These constants are the single tuning point for chaos SEVERITY. The chaos
# *frequency* (which event, which team, how likely) now comes from each team's
# historical chaos_profile — see build_chaos_scenarios. The split of frequency
# is data-driven; the magnitude of each event's effect stays fixed here.
CHAOS_EVENT_WEIGHTS: Dict[str, float] = {
    "straight_red_card": 0.45,
    "var_penalty":       0.35,
    "freak_injury":      0.20,
}
CHAOS_TEAM_SPLIT = 0.5            # fallback only: P(event befalls A) with no profiles
CHAOS_EXPECTED_REMAINING = 0.5   # expected fraction of match left when it strikes
CHAOS_PENALTY_XG = 0.76          # additive xG for a converted VAR penalty

# per-event effect on the disrupted team (sup) and its opponent (bst), expressed
# over the *whole* match before the expected-remaining scaling is applied.
_CHAOS_SUP = {"straight_red_card": 0.72, "freak_injury": 0.90, "var_penalty": 1.00}
_CHAOS_BST = {"straight_red_card": 1.12, "freak_injury": 1.03, "var_penalty": 1.00}

# Default per-match propensities if a team carries no chaos_profile.
_DEFAULT_CHAOS_PROFILE = {
    "red_card": 0.10, "penalty_for": 0.17, "penalty_against": 0.16, "injury": 0.08,
}


def _effective_mult(per_event_factor: float) -> float:
    """Blend the per-event factor over the expected remaining match fraction."""
    f = CHAOS_EXPECTED_REMAINING
    return round(1.0 - f + f * per_event_factor, 4)


def _event_effects(event_type: str, team: str) -> dict:
    """The lambda adjustments for one (event, disrupted-team) pair."""
    mult_a = mult_b = 1.0
    add_a = add_b = 0.0
    if event_type == "var_penalty":
        # penalty awarded to the OPPONENT of the disrupted (conceding) team
        if team == "A":
            add_b = CHAOS_PENALTY_XG
        else:
            add_a = CHAOS_PENALTY_XG
    else:
        sup = _effective_mult(_CHAOS_SUP[event_type])
        bst = _effective_mult(_CHAOS_BST[event_type])
        if team == "A":
            mult_a, mult_b = sup, bst
        else:
            mult_b, mult_a = sup, bst
    return {"mult_a": mult_a, "mult_b": mult_b, "add_a": add_a, "add_b": add_b}


def _propensities(rates_a: dict, rates_b: dict) -> List[tuple]:
    """
    Per-(event, disrupted-team) raw propensities from the two teams' histories.

    Returns [(event_type, affected_team, raw_weight), ...]. `affected_team` is
    the DISRUPTED side: for a penalty that is the side that CONCEDES it, so a
    penalty awarded to A is modelled as event befalling B (and vice-versa).
    A penalty's likelihood blends the awarded side's penalty_for with the
    conceding side's penalty_against.
    """
    ra = {**_DEFAULT_CHAOS_PROFILE, **(rates_a or {})}
    rb = {**_DEFAULT_CHAOS_PROFILE, **(rates_b or {})}
    pen_to_a = 0.5 * (ra["penalty_for"] + rb["penalty_against"])   # B concedes
    pen_to_b = 0.5 * (rb["penalty_for"] + ra["penalty_against"])   # A concedes
    return [
        ("straight_red_card", "A", ra["red_card"]),
        ("straight_red_card", "B", rb["red_card"]),
        ("freak_injury",      "A", ra["injury"]),
        ("freak_injury",      "B", rb["injury"]),
        ("var_penalty",       "B", pen_to_a),   # disrupted = conceding side
        ("var_penalty",       "A", pen_to_b),
    ]


def build_chaos_scenarios(
    base_probability: float,
    rates_a: dict | None = None,
    rates_b: dict | None = None,
) -> List[dict]:
    """
    Enumerate the probability-weighted chaos scenarios for a single run.

    Returns a list of scenario dicts whose `probability` values sum to 1.0,
    starting with the no-chaos baseline (weight 1 - base_probability) followed
    by one entry per (event_type x affected_team). Each scenario carries the
    multiplicative and additive adjustments to apply to the base lambdas:

        {label, event_type, affected_team, probability,
         mult_a, mult_b, add_a, add_b}

    Frequency vs severity:
      * The TOTAL chaos mass is `base_probability` (the single global intensity
        knob, unchanged).
      * That mass is split across the (event x team) scenarios in proportion to
        each team's historical chaos_profile — so an ill-disciplined side eats
        more of the red-card mass than its opponent, a side that wins lots of
        penalties tilts the penalty mass, etc.
      * The *magnitude* of each event's effect stays fixed in the constants
        above (a red card costs the same once it happens, whoever it happens to).

    If both `rates_a` and `rates_b` are omitted, falls back to the original
    symmetric model (global CHAOS_EVENT_WEIGHTS, 50/50 team split) so dry-runs
    and legacy callers behave exactly as before.

    `affected_team` is the DISRUPTED side (the one a card/injury hits, or the one
    that concedes the VAR penalty).
    """
    p = max(0.0, min(1.0, base_probability))
    scenarios: List[dict] = [{
        "label": "baseline", "event_type": "none", "affected_team": "",
        "probability": round(1.0 - p, 6),
        "mult_a": 1.0, "mult_b": 1.0, "add_a": 0.0, "add_b": 0.0,
    }]

    if rates_a is None and rates_b is None:
        # ---- legacy symmetric fallback ----
        w_total = sum(CHAOS_EVENT_WEIGHTS.values()) or 1.0
        weighted = []
        for event_type, w in CHAOS_EVENT_WEIGHTS.items():
            for team, split in (("A", CHAOS_TEAM_SPLIT), ("B", 1.0 - CHAOS_TEAM_SPLIT)):
                weighted.append((event_type, team, (w / w_total) * split))
    else:
        # ---- data-driven: distribute p by historical propensity ----
        raw = _propensities(rates_a or {}, rates_b or {})
        raw_total = sum(r for _, _, r in raw) or 1.0
        weighted = [(et, tm, r / raw_total) for et, tm, r in raw]

    for event_type, team, frac in weighted:
        prob = round(p * frac, 6)
        if prob <= 0:
            continue
        eff = _event_effects(event_type, team)
        scenarios.append({
            "label": f"{event_type}_vs_{team}",
            "event_type": event_type,
            "affected_team": team,
            "probability": prob,
            **eff,
        })
    return scenarios


def _scenario_lambdas(base_a: float, base_b: float, sc: dict) -> tuple:
    la = max(0.05, base_a * sc.get("mult_a", 1.0) + sc.get("add_a", 0.0))
    lb = max(0.05, base_b * sc.get("mult_b", 1.0) + sc.get("add_b", 0.0))
    return la, lb


def mixture_grid(base_a: float, base_b: float, scenarios: List[dict],
                 max_goals: int = DEFAULT_MAX_GOALS) -> dict:
    """
    Combine a separate independent-Poisson grid per scenario into ONE scoreline
    distribution, weighted by each scenario's probability.

    This is the statistically correct object for "with probability p_i the match
    is scenario i": P = sum_i p_i * Grid(lambda_a^i, lambda_b^i). The reported
    expected goals are the mixture mean (computed from the combined grid, so the
    displayed xG and the displayed distribution are always consistent).
    """
    n = max_goals
    acc = [[0.0] * (n + 1) for _ in range(n + 1)]
    w_sum = 0.0
    for sc in scenarios:
        w = sc.get("probability", 0.0)
        if w <= 0:
            continue
        la, lb = _scenario_lambdas(base_a, base_b, sc)
        sub = _score_matrix(la, lb, n)
        for i in range(n + 1):
            row = sub[i]
            arow = acc[i]
            for j in range(n + 1):
                arow[j] += w * row[j]
        w_sum += w
    if w_sum > 0:
        acc = [[p / w_sum for p in row] for row in acc]
    return _aggregate(acc, n)
