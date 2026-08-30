"""
schemas.py
==========

Strict typed contracts for everything that flows through the graph.

* The *payloads* (what each agent produces) are Pydantic v2 models. They double
  as the JSON Schemas we hand to Claude for forced structured output, so the
  LLM is constrained to emit exactly these shapes.
* The *graph state* (`AgentState`) is a `TypedDict` with channel reducers, which
  is what LangGraph expects.

Every LLM-facing model is given full defaults, so a partial or soft-failed LLM
response never crashes the graph - missing fields simply fall back to defaults.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


# ===========================================================================
# 1. COMMON RESEARCHER OUTPUTS  (impartial, computed for BOTH teams)
# ===========================================================================
class FitnessProfile(BaseModel):
    """
    The Physiologist multiplier, baked into the Researcher's payload.

    `fitness_degradation_factor` is a multiplier on a team's baseline lambda:
      < 1.0  -> degraded (fatigue/travel/climate)
      = 1.0  -> neutral
      > 1.0  -> boosted (host nation / hyper-familiar conditions)
    """

    rest_days: int = 4
    travel_km: float = 0.0
    is_host_nation: bool = False
    altitude_m: float = 0.0
    climate_note: str = ""
    rest_penalty: float = 0.0
    travel_penalty: float = 0.0
    climate_penalty: float = 0.0
    host_bonus: float = 0.0
    fitness_degradation_factor: float = 1.0


class SquadEntry(BaseModel):
    """One player→club pairing. A list of these (instead of a Dict[str,str] map)
    so the whole TeamForm is representable as a native Gemini response_schema —
    arbitrary-key maps are not."""

    player: str = ""
    club: str = ""


class TeamForm(BaseModel):
    """Raw, strategy-free statistics for one team (the 'global pulse')."""

    name: str = "Unknown"
    fifa_rank: int = 20
    recent_results: List[str] = Field(default_factory=list)        # e.g. ["W","W","D"]
    avg_goals_scored: Optional[float] = None    # recent attacking rate (computed from results)
    avg_goals_conceded: Optional[float] = None  # recent defensive rate (computed from results)
    recent_xg_for: Optional[float] = None
    recent_xg_against: Optional[float] = None
    injuries: List[str] = Field(default_factory=list)
    suspensions: List[str] = Field(default_factory=list)
    likely_lineup: str = ""   # predicted XI + formation from press conf / reporters
    squad_clubs: List[SquadEntry] = Field(default_factory=list)    # player→club pairings
    public_weaknesses: List[str] = Field(default_factory=list)     # publicly known frailties
    notes: str = ""


class GlobalResearch(BaseModel):
    """The structured object the Researcher streams out for the whole system."""

    team_a_form: TeamForm = Field(default_factory=TeamForm)
    team_b_form: TeamForm = Field(default_factory=TeamForm)
    team_a_fitness: FitnessProfile = Field(default_factory=FitnessProfile)
    team_b_fitness: FitnessProfile = Field(default_factory=FitnessProfile)
    fixture_summary: str = ""
    sources: List[str] = Field(default_factory=list)


# ===========================================================================
# 2. INDEPENDENT (per-team) AGENT OUTPUTS
# ===========================================================================
class ChemistryCluster(BaseModel):
    club: str = ""
    players: List[str] = Field(default_factory=list)


class CohesionReport(BaseModel):
    """Alchemist output - club-level synergy within a team's own 26-man roster."""

    cohesion_multiplier: float = 1.0          # ~[0.85, 1.20]
    clusters: List[ChemistryCluster] = Field(default_factory=list)
    reasoning: str = ""


class StrategyReport(BaseModel):
    """Strategist output - tournament game theory for THIS team only."""

    matrix_mode: Literal["Full Intensity", "Passive/Rotation", "Targeted Bracket"] = "Full Intensity"
    strategic_intensity_multiplier: float = 1.0   # ~[0.80, 1.10]
    target_bracket: str = ""                       # optional knockout-path intent
    reasoning: str = ""


class ScoutReport(BaseModel):
    """Scout output - analysis of the OPPONENT's public, non-hidden weaknesses."""

    opponent_name: str = ""
    opponent_weaknesses: List[str] = Field(default_factory=list)
    exploit_vectors: List[str] = Field(default_factory=list)   # how WE attack them
    reasoning: str = ""


class TacticalPlan(BaseModel):
    """
    Tactician output - the BLIND tactical setup profile.

    'Blind' because it is formed without sight of the opponent's plan; only the
    Pitch Simulator later collides the two blind plans.
    """

    plan_type: Literal["Plan A (Identity)", "Plan B (Counter)"] = "Plan A (Identity)"
    formation: str = "4-3-3"
    style: Literal[
        "high_press", "possession", "low_block_counter", "balanced", "direct"
    ] = "balanced"
    intended_tactical_fit: float = 1.0          # self-assessed fit vs scouted weaknesses
    key_instructions: List[str] = Field(default_factory=list)
    reasoning: str = ""


# ===========================================================================
# 3. COMMON SIMULATION / MARKET / CHAOS / JUDGE OUTPUTS
# ===========================================================================
class TeamLambda(BaseModel):
    """The fully-fused expected-goals (lambda) breakdown for one team."""

    name: str = ""
    lambda_base: float = 1.3
    cohesion: float = 1.0
    strategic_intensity: float = 1.0
    tactical_fit: float = 1.0
    fitness_degradation: float = 1.0
    tournament_momentum: float = 1.0   # deterministic, from actual MD1+ results
    dark_horse: float = 1.0            # manual per-team under/over-rating factor
    lambda_final: float = 1.3


class PitchResult(BaseModel):
    team_a: TeamLambda = Field(default_factory=TeamLambda)
    team_b: TeamLambda = Field(default_factory=TeamLambda)
    collision_note: str = ""
    phase: Literal["pre_chaos", "post_chaos"] = "pre_chaos"


class BookmakerReport(BaseModel):
    """Market anchor - implied probabilities from betting exchanges."""

    implied_home_win: float = 0.40
    implied_draw: float = 0.27
    implied_away_win: float = 0.33
    overround: float = 0.0
    source: str = "web_search market scan"
    reasoning: str = ""


class ChaosScenario(BaseModel):
    """
    One probability-weighted black-swan scenario in the single-run mixture.

    `affected_team` is the DISRUPTED side. The adjustments are applied to the
    base lambdas as `lambda * mult + add` per team before that scenario's
    Poisson grid is built.
    """

    label: str = "baseline"
    event_type: Literal[
        "none", "straight_red_card", "var_penalty", "freak_injury"
    ] = "none"
    affected_team: Literal["A", "B", ""] = ""
    probability: float = 1.0
    mult_a: float = 1.0
    mult_b: float = 1.0
    add_a: float = 0.0
    add_b: float = 0.0


class ChaosModel(BaseModel):
    """
    The Chaos Agent's output: the full weighted scenario set for ONE run.

    There is no random roll - every scenario (including the no-chaos baseline)
    contributes its probability mass to the Judge's mixture-of-grids on every
    run. `scenarios` probabilities sum to 1.0.
    """

    base_probability: float = 0.22                 # P(some black-swan occurs)
    scenarios: List[ChaosScenario] = Field(default_factory=list)
    summary: str = ""


class ScoreProb(BaseModel):
    home: int = 0
    away: int = 0
    prob: float = 0.0


class PoissonReport(BaseModel):
    home_win: float = 0.0
    draw: float = 0.0
    away_win: float = 0.0
    most_likely_scores: List[ScoreProb] = Field(default_factory=list)
    expected_goals_home: float = 0.0
    expected_goals_away: float = 0.0
    grid: List[List[float]] = Field(default_factory=list)   # full 0-0..N-N matrix
    max_goals: int = 4                                       # grid spans 0..max_goals


class FinalReport(BaseModel):
    """Judge output - the executive summary."""

    headline: str = ""
    predicted_scoreline: str = ""
    win_draw_loss: Dict[str, float] = Field(default_factory=dict)
    model_vs_market: str = ""
    chaos_impact: str = ""
    narrative: str = ""
    confidence: Literal["low", "medium", "high"] = "medium"


class JudgeProse(BaseModel):
    """
    The PROSE-only slice the Judge asks the LLM for.

    The numbers (scoreline, W/D/L, confidence) are computed deterministically, so
    the model is asked for exactly these four narrative fields — nothing it's told
    to 'leave blank'. That removes the cue that made small models return an
    all-empty FinalReport, and (having no Dict field) it qualifies for native
    response_schema, which constrains the model to actually fill the prose.
    """

    headline: str = ""
    model_vs_market: str = ""
    narrative: str = ""
    chaos_impact: str = ""


# ===========================================================================
# 4. GRAPH STATE  (LangGraph TypedDict + channel reducers)
# ===========================================================================
def merge_teams(left: Optional[dict], right: Optional[dict]) -> dict:
    """
    Reducer for the per-team channel.

    The two independent branches (Team A, Team B) and the sequential agents
    within each branch all write into `state["teams"]`. This deep-merges at the
    team level so concurrent partial updates from parallel supersteps never
    clobber one another. It is the mechanism that keeps the branches isolated
    yet mergeable.
    """
    out: dict = dict(left or {})
    for team_key, patch in (right or {}).items():
        base = dict(out.get(team_key, {}))
        base.update(patch or {})
        out[team_key] = base
    return out


class AgentState(TypedDict, total=False):
    """
    The single global payload the Manager orchestrates.

    Channels with reducers can be written by parallel nodes in the same
    superstep; channels without are written by exactly one node.
    """

    config: dict                                          # frozen scenario + settings
    teams: Annotated[dict, merge_teams]                   # {"A": {...}, "B": {...}}
    bookmaker: dict                                       # BookmakerReport (dump)
    pitch: dict                                           # PitchResult (dump)
    chaos: dict                                           # ChaosModel (dump)
    poisson: dict                                         # PoissonReport (dump)
    report: dict                                          # FinalReport (dump)
    step_log: Annotated[List[str], operator.add]          # human-readable trace
