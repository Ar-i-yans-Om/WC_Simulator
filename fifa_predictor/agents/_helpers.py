"""
agents/_helpers.py
==================

Small utilities shared by all agent nodes.

The most important one is `public_opponent_view`, which enforces the isolation
contract: an independent agent for one team may only ever see the *publicly
observable* facts about the other team (name, rank, recent results, publicly
reported injuries/weaknesses) - never the opponent's internal cohesion,
strategy, tactics or computed lambda. This is how "Team A has zero visibility
into Team B's internal state" is implemented mechanically.
"""

from __future__ import annotations

from typing import Dict


# Whitelist of opponent fields that are considered public knowledge.
_PUBLIC_FORM_FIELDS = {
    "name",
    "fifa_rank",
    "recent_results",
    "injuries",
    "suspensions",
    "public_weaknesses",
}


def get_team(state: dict, team_key: str) -> dict:
    """Return this team's mutable packet from state, creating it if absent."""
    return dict(state.get("teams", {}).get(team_key, {}))


def public_opponent_view(state: dict, my_key: str) -> Dict:
    """
    Return ONLY the public-knowledge slice of the opponent's form.

    Deliberately strips squad_clubs, xG internals and everything the opponent's
    private agents produced. Scouts use this; nothing else may read the
    opponent's packet.
    """
    opp_key = "B" if my_key == "A" else "A"
    opp_form = state.get("teams", {}).get(opp_key, {}).get("form", {})
    return {k: opp_form.get(k) for k in _PUBLIC_FORM_FIELDS if k in opp_form}


def team_label(state: dict, team_key: str) -> str:
    return state.get("teams", {}).get(team_key, {}).get("form", {}).get("name", team_key)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
