"""
agents/independent/tactician.py
================================

Tactician (Operational Command) — INDEPENDENT, one per team.
"""

from __future__ import annotations

import logging

from ...schemas import TacticalPlan
from .._helpers import get_team, team_label

logger = logging.getLogger(__name__)

_STYLES_GUIDE = """\
Available styles — pick exactly ONE based on your situation:
  high_press        Win ball high up the pitch. Intense. Best with fit squad and
                    fast forwards. Effective vs possession teams. Tiring.
  possession        Control tempo, pass-and-move. Best technically gifted squads.
                    Neutralises direct/long-ball. Can be passive vs deep block.
  low_block_counter Sit deep, absorb, hit on the break. Best when outgunned or
                    protecting a lead. Effective vs high press / high line.
  balanced          Flexible mid-block, transitions both ways. No strong edge.
                    Use when no clear structural matchup advantage.
  direct            Long balls, aerial duels, set-piece threat. Effective vs
                    high lines and compact defences.\
"""

_TACTICIAN_SYSTEM = (
    "You are the head coach's operational planner for ONE national team. "
    "You have three inputs: your squad's availability, your strategic intensity "
    "mode, and a scout report on the opponent's public weaknesses. "
    "Your job is to pick the style and formation that best exploits the opponent "
    "given your constraints. Think like a real coach: if the scout says the "
    "opponent plays a high defensive line, consider low_block_counter or direct "
    "even if your default identity is possession. Rotation mode doesn't force "
    "you to be passive — it means you protect certain players, not that you "
    "abandon your tactical plan. Pick Plan A (your identity) only if it "
    "genuinely suits the matchup; pick Plan B (counter/exploit) if the scout "
    "data gives you a clear edge with a different approach."
)


def make_tactician_node(llms: dict, settings, team_key: str):
    reasoning = llms["reasoning"]

    def tactician_node(state: dict) -> dict:
        team     = get_team(state, team_key)
        name     = team_label(state, team_key)
        strategy = team.get("strategy", {})
        scout    = team.get("scout", {})
        form     = team.get("form", {})

        intensity_mode = strategy.get("matrix_mode", "Full Intensity")
        intensity_val  = strategy.get("strategic_intensity_multiplier", 1.0)
        opp_weaknesses = scout.get("opponent_weaknesses", [])
        exploit_vectors= scout.get("exploit_vectors", [])
        injuries       = form.get("injuries", [])
        recent_results = form.get("recent_results", [])

        tag = "Tactician"
        logger.info("")
        logger.info("[%s-%s] ── %s tactical plan ──────────────────────────",
                    tag, team_key, name)

        logger.info("[%s-%s]  Inputs:", tag, team_key)
        logger.info("[%s-%s]    Strategy mode : %s  (intensity ×%.4f)",
                    tag, team_key, intensity_mode, intensity_val)
        logger.info("[%s-%s]    Injuries      : %s",
                    tag, team_key, ", ".join(injuries[:4]) if injuries else "none")
        logger.info("[%s-%s]    Opp weaknesses: %s",
                    tag, team_key,
                    " · ".join(opp_weaknesses[:3]) if opp_weaknesses else "none")
        logger.info("[%s-%s]    Exploit vectors: %s",
                    tag, team_key,
                    " · ".join(exploit_vectors[:2]) if exploit_vectors else "none")
        logger.info("[%s-%s]  Asking Gemini to build gameplan...", tag, team_key)

        plan = reasoning.structured(
            TacticalPlan,
            system=_TACTICIAN_SYSTEM,
            user=(
                f"My team: {name}.\n"
                f"Strategic mode: {intensity_mode} (intensity ×{intensity_val}).\n"
                f"Confirmed injuries/doubts: {injuries if injuries else 'none'}.\n"
                f"Recent form: {recent_results if recent_results else 'unknown'}.\n"
                f"Scout report — opponent public weaknesses: "
                f"{opp_weaknesses if opp_weaknesses else 'none identified'}.\n"
                f"Scout exploit vectors: {exploit_vectors if exploit_vectors else 'none'}.\n\n"
                f"{_STYLES_GUIDE}\n\n"
                "Instructions:\n"
                "1. Choose Plan A (play to our identity) OR Plan B (counter/exploit).\n"
                "2. Choose ONE style from the list above — reason specifically about "
                "which style best exploits the scouted weaknesses. Do NOT default to "
                "'balanced' unless no other style fits.\n"
                "3. Choose a formation (e.g. 4-3-3, 4-4-2, 4-2-3-1, 5-3-2, 3-5-2).\n"
                "4. Set intended_tactical_fit in [0.88, 1.15]: how well does this plan "
                "exploit the opponent? High (1.10+) = clear structural advantage; "
                "low (0.88-0.95) = forced defensive choice.\n"
                "5. Give 3 specific key_instructions for this match.\n"
                "6. Write a concise reasoning (2-3 sentences) explaining WHY this "
                "style/plan gives you the best edge in this specific matchup."
            ),
        )
        plan.intended_tactical_fit = round(
            max(0.88, min(1.15, plan.intended_tactical_fit)), 4
        )

        # Log the full gameplan
        logger.info("[%s-%s]  ── GAMEPLAN ─────────────────────────────────────", tag, team_key)
        logger.info("[%s-%s]    Plan      : %s", tag, team_key, plan.plan_type)
        logger.info("[%s-%s]    Formation : %s", tag, team_key, plan.formation)
        logger.info("[%s-%s]    Style     : %s", tag, team_key, plan.style)
        logger.info("[%s-%s]    Fit score : %.4f", tag, team_key, plan.intended_tactical_fit)
        if plan.key_instructions:
            for i, instr in enumerate(plan.key_instructions, 1):
                logger.info("[%s-%s]    Instr %d   : %s", tag, team_key, i, instr)
        if plan.reasoning:
            logger.info("[%s-%s]    Reasoning : %s", tag, team_key,
                        plan.reasoning[:300].replace("\n", " "))

        key = "tactics"
        return {
            "teams": {team_key: {key: plan.model_dump()}},
            "step_log": [
                f"[{tag}-{team_key}] {name}: {plan.plan_type} | {plan.formation} | "
                f"{plan.style} | fit {plan.intended_tactical_fit} | "
                f"instructions: {', '.join(plan.key_instructions[:2])}"
            ],
        }

    return tactician_node
