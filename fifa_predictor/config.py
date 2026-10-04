"""
config.py
=========

Runtime configuration and the lean match scenario specification.

Design principle: TeamScenario contains ONLY what is factual, fixed, and known
before the match. It answers: who, where, when, how strong (ELO), and what is
their group standing. Everything qualitative — form, injuries, squad, tactics,
strengths, weaknesses — is discovered by the agents at runtime.

This is the equivalent of the coaching staff's briefing pack: just the facts.
The analysts (agents) do the rest.

MODEL ROUTING (free-tier quota strategy)
----------------------------------------
Gemini free-tier rate limits are PER MODEL (per project/key), so splitting work
across two models gives two independent daily quota buckets per key. We exploit that:

  * research_model  → the GROUNDED web-research calls (Researcher, Bookmaker).
    Only ~2 such calls per fixture, and they benefit most from real Google
    Search grounding. Routed to gemini-2.5-flash-lite.
  * reasoning_model / judge_model → all the UNGROUNDED structured calls
    (Alchemist, Strategist, Scout, Tactician, PitchSim, Judge). These are pure
    reasoning over already-gathered context; routed to gemini-3.1-flash-lite,
    which carries the bulk of the per-run request volume on its own quota bucket.

NB (probed Jun 2026): the free daily allowance is ~20 requests/day/key/model
(resets 00:00 Pacific). Routing every tier to ONE model collapses the two buckets
into one and exhausts it twice as fast — keep the two models distinct on free tier.

CAPACITY OVERLOAD (503 / "overloaded" / UNAVAILABLE)
----------------------------------------------------
A 503 is transient server load on a model. The LLM wrapper rotates to the next KEY
immediately (a different key's backend often serves the same model fine during a
spike), and only falls back to exponential backoff once the key pool is exhausted.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------
# Grounded research → its own quota bucket, few calls per run.
DEFAULT_RESEARCH_MODEL  = "gemini-2.5-flash-lite"
# Ungrounded reasoning + final report → high-RPM model (bulk of calls).
DEFAULT_REASONING_MODEL = "gemini-3.1-flash-lite"
DEFAULT_JUDGE_MODEL     = "gemini-3.1-flash-lite"

# Optional capacity-fallback chain for 503/"overloaded" errors (comma/space/newline
# separated). Empty by default; on a 503 the wrapper rotates KEYS instead. Override
# via FIFA_FALLBACK_MODELS if you want a second model as a separate capacity pool.
DEFAULT_FALLBACK_MODELS = ""


@dataclass
class Settings:
    """Infrastructure / environment configuration."""

    gemini_api_key: Optional[str] = None
    # Ordered failover pool. The LLM wrapper rotates to the next key on an
    # auth/quota error (401/403/429). Populated from GEMINI_API_KEY +
    # GEMINI_API_KEYS by from_env(); gemini_api_key stays the primary (first).
    gemini_api_keys: list = field(default_factory=list)

    research_model:  str = DEFAULT_RESEARCH_MODEL
    reasoning_model: str = DEFAULT_REASONING_MODEL
    judge_model:     str = DEFAULT_JUDGE_MODEL

    # Ordered capacity-fallback chain shared by all tiers (each tier skips its own
    # primary). Tried, in order, only on a 503/overload error. See module docstring.
    fallback_models: list = field(default_factory=list)

    max_output_tokens: int  = 10000

    odds_api_key: Optional[str] = None

    # Chaos engine (modelled as a single-run probability-weighted mixture).
    # `chaos_base_probability` is P(some black-swan occurs); set to 0.0 for a
    # clean baseline. The per-event weights and multipliers live in math_engine.
    chaos_base_probability: float = 0.3

    # Scoreline grid spans 0-0 .. max_goals-max_goals. 7 keeps the grid wide
    # enough that high-lambda favourites (xG ~3) aren't truncated.
    max_goals: int = 7

    @classmethod
    def from_env(cls) -> "Settings":
        # Build the ordered key failover pool from both env vars:
        #   GEMINI_API_KEY   — the single primary key
        #   GEMINI_API_KEYS  — comma/whitespace/newline-separated extra keys
        # The primary goes first; duplicates are dropped, order preserved.
        pool: list[str] = []
        for raw in [os.getenv("GEMINI_API_KEY", "")] + \
                   re.split(r"[,\s]+", os.getenv("GEMINI_API_KEYS", "") or ""):
            k = (raw or "").strip()
            if k and k not in pool:
                pool.append(k)

        # Capacity-fallback chain: split on comma/space/newline, drop blanks,
        # de-duplicate while preserving order.
        fallbacks: list[str] = []
        for raw in re.split(r"[,\s]+", os.getenv("FIFA_FALLBACK_MODELS", DEFAULT_FALLBACK_MODELS) or ""):
            m = (raw or "").strip()
            if m and m not in fallbacks:
                fallbacks.append(m)

        return cls(
            gemini_api_key   = pool[0] if pool else None,
            gemini_api_keys  = pool,
            research_model   = os.getenv("FIFA_RESEARCH_MODEL",  DEFAULT_RESEARCH_MODEL),
            reasoning_model  = os.getenv("FIFA_REASONING_MODEL", DEFAULT_REASONING_MODEL),
            judge_model      = os.getenv("FIFA_JUDGE_MODEL",     DEFAULT_JUDGE_MODEL),
            fallback_models  = fallbacks,
            max_output_tokens= int(os.getenv("FIFA_MAX_TOKENS",   "10000")),
            odds_api_key     = os.getenv("ODDS_API_KEY"),
            chaos_base_probability = float(os.getenv("FIFA_CHAOS_PROB", "0.3")),
            max_goals        = int(os.getenv("FIFA_MAX_GOALS", "7")),
        )

    def validate(self) -> None:
        """Ensure we can actually run live (a Gemini key is present)."""
        if not self.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Export it or add it to a .env file."
            )


# ---------------------------------------------------------------------------
# Match scenario — minimal factual context only
# ---------------------------------------------------------------------------
@dataclass
class TeamScenario:
    """
    The immutable facts about one team entering this fixture.

    Contains what any observer can verify before kickoff — who they are, where
    and when they play, their rating and ranking, their group standing and the
    World Cup results they have already played — plus the deterministic levers
    read from the data files (chaos profile, momentum, dark-horse factor).
    Nothing qualitative: form, injuries and tactics are the agents' job.
    """
    name:                  str
    matchday:              int            # the team's Nth match: group 1-3, knockouts 4-8
    group:                 str            # the team's group letter, e.g. "K"
    date:                  str            # ISO-8601 e.g. "2026-06-27"
    group_points:          int            # points BEFORE this match (group stage; 0 in knockouts)
    host_city_this_match:  str            # canonical host city (post alias-mapping)
    host_city_last_match:  str            # previous fixture city (for Haversine travel)
    rest_days:             int            # days since previous fixture
    elo:                   float = 1500.0 # ELO rating from team_ratings.json
    fifa_rank:             int   = 30     # FIFA/Coca-Cola ranking
    confederation:         str   = ""     # UEFA / CONMEBOL / CAF / AFC / CONCACAF / OFC
    is_host_nation:        bool  = False  # USA, Canada, Mexico
    notes:                 str   = ""     # optional manual context (coach can override)
    chaos_profile:         dict  = field(default_factory=dict)  # per-match black-swan rates
    # Intra-tournament momentum, computed deterministically from ACTUAL results.json
    # (MD1 onward). 1.0 at MD1; >1.0 after strong results, <1.0 after poor ones.
    tournament_momentum:   float = 1.0
    tournament_form_summary: str = ""     # e.g. "MD1: W 2-0 vs South Africa"
    tournament_results:    list  = field(default_factory=list)  # structured WC matches played
    # Manual, tunable under/over-rating factor for dark horses (1.0 = neutral).
    dark_horse_factor:     float = 1.0


@dataclass
class MatchScenario:
    team_a:             TeamScenario
    team_b:             TeamScenario
    competition:        str   = "FIFA World Cup 2026"
    stage:              str   = "Group Stage"  # e.g. "Group K — Matchday 3" or "Round of 16"
    knockout_round:     str   = ""             # "R32" … "F" for a knockout tie, "" in the group stage
    tournament_avg_goals: float = 1.35


# ---------------------------------------------------------------------------
# Demo scenario: fixture K5, Portugal vs Colombia (Group K, Matchday 3).
# Used by the CLI demo (python -m fifa_predictor.main) and the offline tests.
# ---------------------------------------------------------------------------
def demo_scenario() -> MatchScenario:
    """
    A static snapshot of fixture K5 exactly as match_runner.build_scenario()
    derives it from the data files (fixtures, ratings, results before kickoff).

    Portugal: 4 pts (D 1-1 DR Congo, W 5-0 Uzbekistan), Philadelphia -> New York.
    Colombia: 6 pts (W 3-1 Uzbekistan, W 1-0 DR Congo), Boston -> New York.

    Everything qualitative (injuries, form, squad news) is still discovered by
    the Researcher at runtime. For any other fixture, use
    `python match_runner.py --fixture <ID>`.
    """
    portugal = TeamScenario(
        name="Portugal",
        matchday=3,
        group="K",
        date="2026-06-27",
        group_points=4,
        host_city_last_match="Philadelphia",
        host_city_this_match="New York",
        rest_days=4,
        elo=1767.85,
        fifa_rank=5,
        confederation="UEFA",
        is_host_nation=False,
        chaos_profile={"red_card": 0.07, "penalty_for": 0.179,
                       "penalty_against": 0.14, "injury": 0.07},
        tournament_momentum=1.045,
        tournament_form_summary="MD1: D 1-1 vs DR Congo; MD2: W 5-0 vs Uzbekistan",
        tournament_results=[
            {"md": 1, "stage": "MD1", "opponent": "DR Congo", "gf": 1, "ga": 1, "result": "D"},
            {"md": 2, "stage": "MD2", "opponent": "Uzbekistan", "gf": 5, "ga": 0, "result": "W"},
        ],
    )
    colombia = TeamScenario(
        name="Colombia",
        matchday=3,
        group="K",
        date="2026-06-27",
        group_points=6,
        host_city_last_match="Boston",
        host_city_this_match="New York",
        rest_days=4,
        elo=1698.35,
        fifa_rank=13,
        confederation="CONMEBOL",
        is_host_nation=False,
        chaos_profile={"red_card": 0.13, "penalty_for": 0.2,
                       "penalty_against": 0.18, "injury": 0.08},
        tournament_momentum=1.045,
        tournament_form_summary="MD1: W 3-1 vs Uzbekistan; MD2: W 1-0 vs DR Congo",
        tournament_results=[
            {"md": 1, "stage": "MD1", "opponent": "Uzbekistan", "gf": 3, "ga": 1, "result": "W"},
            {"md": 2, "stage": "MD2", "opponent": "DR Congo", "gf": 1, "ga": 0, "result": "W"},
        ],
        dark_horse_factor=1.03,
    )
    return MatchScenario(
        team_a=portugal,
        team_b=colombia,
        stage="Group K — Matchday 3",
    )
