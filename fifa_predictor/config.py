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
Gemini free-tier rate limits are PER MODEL, so splitting work across two models
gives two independent quota buckets. We exploit that:

  * research_model  → the GROUNDED web-research calls (Researcher, Bookmaker).
    Only ~2 such calls per fixture, and they benefit most from a stronger model
    + real Google Search grounding. Routed to Gemini 3 Flash.
  * reasoning_model / judge_model → all the UNGROUNDED structured calls
    (Alchemist, Strategist, Scout, Tactician, PitchSim, Judge). These are pure
    reasoning over already-gathered context; Flash-Lite is plenty and carries
    the bulk of the per-run request volume on its own quota bucket.
"""

from __future__ import annotations

import os
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
# Grounded research → stronger model, separate quota bucket, few calls/run.
DEFAULT_RESEARCH_MODEL  = "gemini-2.5-flash-lite"
# Ungrounded reasoning + final report → cheap, high-RPM model (bulk of calls).
DEFAULT_REASONING_MODEL = "gemini-3.1-flash-lite"
DEFAULT_JUDGE_MODEL     = "gemini-3.1-flash-lite"


@dataclass
class Settings:
    """Infrastructure / environment configuration."""

    gemini_api_key: Optional[str] = None

    research_model:  str = DEFAULT_RESEARCH_MODEL
    reasoning_model: str = DEFAULT_REASONING_MODEL
    judge_model:     str = DEFAULT_JUDGE_MODEL

    max_web_searches:  int  = 6
    max_output_tokens: int  = 10000

    odds_api_key: Optional[str] = None

    # Chaos engine (modelled as a single-run probability-weighted mixture).
    # `chaos_base_probability` is P(some black-swan occurs); set to 0.0 for a
    # clean baseline. The per-event weights and multipliers live in math_engine.
    chaos_base_probability: float = 0.22

    # Scoreline grid spans 0-0 .. max_goals-max_goals.
    max_goals: int = 7

    @classmethod
    def from_env(cls) -> "Settings":
        def _bool(name: str, default: bool) -> bool:
            v = os.getenv(name)
            return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}

        return cls(
            gemini_api_key   = os.getenv("GEMINI_API_KEY"),
            research_model   = os.getenv("FIFA_RESEARCH_MODEL",  DEFAULT_RESEARCH_MODEL),
            reasoning_model  = os.getenv("FIFA_REASONING_MODEL", DEFAULT_REASONING_MODEL),
            judge_model      = os.getenv("FIFA_JUDGE_MODEL",     DEFAULT_JUDGE_MODEL),
            max_web_searches = int(os.getenv("FIFA_MAX_SEARCHES", "6")),
            max_output_tokens= int(os.getenv("FIFA_MAX_TOKENS",   "10000")),
            odds_api_key     = os.getenv("ODDS_API_KEY"),
            chaos_base_probability = float(os.getenv("FIFA_CHAOS_PROB", "0.22")),
            max_goals        = int(os.getenv("FIFA_MAX_GOALS", "7")),
        )

    def validate(self) -> None:
        """Ensure we can actually run live (a Gemini key is present)."""
        if not self.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Export it or add it to a .env file."
            )


def load_settings() -> Settings:
    return Settings.from_env()


# ---------------------------------------------------------------------------
# Match scenario — minimal factual context only
# ---------------------------------------------------------------------------
@dataclass
class TeamScenario:
    """
    The immutable facts about one team entering this fixture.

    Contains only what any observer can verify without watching the match:
    who they are, where they play, when, how strong they are by ranking,
    and what their group standing is. Nothing qualitative — that is the
    Researcher's job.
    """
    name:                  str
    matchday:              int
    group:                 str            # e.g. "K"
    date:                  str            # ISO-8601 e.g. "2026-06-27"
    group_points:          int            # points BEFORE this match (from results.json)
    host_city_this_match:  str            # canonical host city (post alias-mapping)
    host_city_last_match:  str            # previous fixture city (for Haversine travel)
    rest_days:             int            # days since previous fixture
    elo:                   float = 1500.0 # Live ELO from team_ratings.json
    fifa_rank:             int   = 30     # FIFA/Coca-Cola ranking
    confederation:         str   = ""     # UEFA / CONMEBOL / CAF / AFC / CONCACAF / OFC
    is_host_nation:        bool  = False  # USA, Canada, Mexico
    notes:                 str   = ""     # optional manual context (coach can override)
    chaos_profile:         dict  = field(default_factory=dict)  # per-match black-swan rates


@dataclass
class MatchScenario:
    team_a:             TeamScenario
    team_b:             TeamScenario
    competition:        str   = "FIFA World Cup 2026"
    stage:              str   = "Group Stage"
    tournament_avg_goals: float = 1.35


# ---------------------------------------------------------------------------
# Default test scenario (Portugal vs France — hypothetical MD3 group match)
# Used by the CLI and the test suite.
# ---------------------------------------------------------------------------
def default_portugal_france_scenario() -> MatchScenario:
    """
    Hypothetical reference fixture used for CLI demos and offline tests.

    Portugal: 6 pts, qualified, rotating squad, Atlanta → Vancouver travel.
    France:   4 pts, must get a result, rested in Vancouver.

    The Researcher will find real current data (injuries, form, squad) in live
    mode. The ELO and group_points are the only inputs the agents start with.
    """
    portugal = TeamScenario(
        name="Portugal",
        matchday=3,
        group="K",
        date="2026-06-27",
        group_points=6,
        host_city_last_match="Atlanta",
        host_city_this_match="Vancouver",
        rest_days=3,
        elo=1766.18,
        fifa_rank=5,
        confederation="UEFA",
        is_host_nation=False,
    )
    france = TeamScenario(
        name="France",
        matchday=3,
        group="I",
        date="2026-06-27",
        group_points=4,
        host_city_last_match="Vancouver",
        host_city_this_match="Vancouver",
        rest_days=6,
        elo=1869.43,
        fifa_rank=1,
        confederation="UEFA",
        is_host_nation=False,
    )
    return MatchScenario(
        team_a=portugal,
        team_b=france,
        stage="Group Stage - Matchday 3 (hypothetical)",
    )