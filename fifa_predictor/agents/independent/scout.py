"""
agents/independent/scout.py
============================

Scout (Opponent Analysis) — INDEPENDENT, one per team.
"""

from __future__ import annotations

import logging

from ...schemas import ScoutReport
from .._helpers import public_opponent_view, team_label

logger = logging.getLogger(__name__)

_SCOUT_SYSTEM = (
    "You are an opposition scout for a single national team. You only have "
    "access to PUBLIC information about the opponent (recent results, reported "
    "injuries, widely-discussed weaknesses). Identify exploitable weaknesses and "
    "concrete vectors to attack them. Do not speculate about secret tactics."
)


def make_scout_node(llms: dict, settings, team_key: str):
    reasoning = llms["reasoning"]

    def scout_node(state: dict) -> dict:
        opp_public = public_opponent_view(state, team_key)
        opp_name   = opp_public.get("name", "Opponent")
        my_name    = team_label(state, team_key)

        logger.info("")
        logger.info("[Scout-%s] ── %s scouting %s ──────────────────────",
                    team_key, my_name, opp_name)
        logger.info("[Scout-%s]  Opponent public data available:", team_key)
        logger.info("[Scout-%s]    FIFA rank: %s", team_key, opp_public.get("fifa_rank", "?"))
        logger.info("[Scout-%s]    Recent results: %s",
                    team_key, "  ".join(opp_public.get("recent_results", [])[:6]) or "none")
        logger.info("[Scout-%s]    Known injuries: %s",
                    team_key, ", ".join(opp_public.get("injuries", [])[:5]) or "none")
        logger.info("[Scout-%s]    Public weaknesses: %s",
                    team_key, " · ".join(opp_public.get("public_weaknesses", [])) or "none listed")

        logger.info("[Scout-%s]  Asking Gemini to identify exploit vectors...", team_key)
        report = reasoning.structured(
            ScoutReport,
            system=_SCOUT_SYSTEM,
            user=(
                f"I scout for {my_name}. Public info about our opponent "
                f"{opp_name}:\n{opp_public}\n"
                "List their exploitable public weaknesses and the specific "
                "vectors we should use to attack them."
            ),
        )
        report.opponent_name = opp_name

        logger.info("[Scout-%s]  ── SCOUT REPORT ────────────────────────────", team_key)
        for i, w in enumerate(report.opponent_weaknesses or [], 1):
            logger.info("[Scout-%s]    Weakness %d: %s", team_key, i, w)
        for i, v in enumerate(report.exploit_vectors or [], 1):
            logger.info("[Scout-%s]    Vector  %d: %s", team_key, i, v)
        if report.reasoning:
            logger.info("[Scout-%s]  Reasoning: %s", team_key, report.reasoning[:200])

        return {
            "teams": {team_key: {"scout": report.model_dump()}},
            "step_log": [
                f"[Scout-{team_key}] {my_name} on {opp_name}: "
                f"{len(report.opponent_weaknesses)} weakness(es): "
                + (", ".join(report.opponent_weaknesses[:3]) if report.opponent_weaknesses else "none")
                + " | vectors: "
                + (", ".join(report.exploit_vectors[:2]) if report.exploit_vectors else "none")
            ],
        }

    return scout_node
