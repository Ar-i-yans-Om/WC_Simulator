"""
agents/independent/strategist.py
=================================

Strategist (Tournament Game Theorist) — INDEPENDENT, one per team.
"""

from __future__ import annotations

import logging

from ...schemas import StrategyReport
from .._helpers import get_team

logger = logging.getLogger(__name__)

_STRATEGIST_SYSTEM = """\
You are the tournament game-theory strategist embedded in a national football
coaching staff at FIFA World Cup 2026.

TOURNAMENT RULES (2026 format):
- 12 groups of 4 teams
- Top 2 from each group qualify directly
- 8 best third-placed teams also advance (making 32 total)
- MD3 group games kick off simultaneously to prevent collusion
- Two semifinal bracket pathways — finishing 1st vs 2nd can completely change
  who you face in the round of 16, quarter-final and semi-final

YOUR JOB:
Based on the group situation and the analyst's briefing, decide the optimal
strategic intensity for this match. Think like a real manager:
- Consider qualification scenarios (what results combinations mean for you)
- Consider bracket implications (is the 2nd-place path easier or harder?)
- Consider squad management (injuries, fitness, upcoming knockout matches)
- Consider opponent strength (is it worth spending energy here?)

Output a specific strategic_intensity_multiplier within the range for your
chosen mode. The exact number matters — reason about it.\
"""

_MODE_RANGES   = {
    "Passive/Rotation": (0.78, 0.95),
    "Full Intensity":   (1.02, 1.14),
    "Targeted Bracket": (0.90, 1.02),
}
_MODE_DEFAULTS = {
    "Passive/Rotation": 0.86,
    "Full Intensity":   1.07,
    "Targeted Bracket": 0.96,
}


def make_strategist_node(llms: dict, settings, team_key: str):
    reasoning = llms["reasoning"]

    def strategist_node(state: dict) -> dict:
        cfg      = state["config"]
        scn_team = cfg["scenario"]["team_a"] if team_key == "A" else cfg["scenario"]["team_b"]
        opp_scn  = cfg["scenario"]["team_b"] if team_key == "A" else cfg["scenario"]["team_a"]
        name     = scn_team["name"]

        points   = scn_team.get("group_points", 0)
        matchday = scn_team.get("matchday", 1)
        elo      = scn_team.get("elo", 1500.0)
        opp_elo  = opp_scn.get("elo", 1500.0)
        group    = scn_team.get("group", "?")
        conf     = scn_team.get("confederation", "")

        logger.info("")
        logger.info("[Strategist-%s] ── %s game-theory brief ──────────────", team_key, name)
        logger.info("[Strategist-%s]  Group %s  |  MD%d  |  %d pts  |  ELO %.0f  |  %s",
                    team_key, group, matchday, points, elo, conf)
        logger.info("[Strategist-%s]  Opponent: %s  (ELO %.0f)",
                    team_key, opp_scn.get("name", "?"), opp_elo)

        team   = get_team(state, team_key)
        form   = team.get("form", {})
        recent = form.get("recent_results", [])
        injuries = form.get("injuries", [])

        mode = "Full Intensity"
        report = StrategyReport(
            matrix_mode=mode,
            strategic_intensity_multiplier=_MODE_DEFAULTS[mode],
        )
        logger.info("[Strategist-%s]  Asking Gemini to reason about optimal strategy...", team_key)

        enriched = reasoning.structured(
            StrategyReport,
            system=_STRATEGIST_SYSTEM,
            user=(
                f"My team: {name} ({conf}), Group {group}, Matchday {matchday}.\n"
                f"Our ELO: {elo}. Opponent: {opp_scn.get('name','?')} (ELO {opp_elo}).\n"
                f"Our group points going into this match: {points}.\n"
                f"Tournament form so far (actual WC results): "
                f"{scn_team.get('tournament_form_summary') or 'no matches played yet (this is MD1)'}.\n"
                f"Recent form: {recent if recent else 'see briefing'}.\n"
                f"Squad concerns: {injuries if injuries else 'none confirmed'}.\n"
                f"Notes: {scn_team.get('notes', 'none')}.\n\n"
                "STEP 1 — Choose your matrix_mode:\n"
                "  'Passive/Rotation'  if you can afford to protect legs for knockouts\n"
                "  'Full Intensity'    if you need points from this match\n"
                "  'Targeted Bracket'  if you are through but want a specific group position\n\n"
                "STEP 2 — Set strategic_intensity_multiplier:\n"
                f"  Passive/Rotation:  {_MODE_RANGES['Passive/Rotation']}\n"
                f"  Full Intensity:    {_MODE_RANGES['Full Intensity']}\n"
                f"  Targeted Bracket:  {_MODE_RANGES['Targeted Bracket']}\n\n"
                "STEP 3 — Reason through the group maths: what combinations "
                "of results guarantee/risk qualification? Bracket worth fighting for?"
            ),
        )

        chosen_mode = enriched.matrix_mode
        if chosen_mode not in _MODE_RANGES:
            chosen_mode = mode
        lo, hi = _MODE_RANGES[chosen_mode]
        raw_intensity = enriched.strategic_intensity_multiplier or _MODE_DEFAULTS[chosen_mode]
        clamped = round(max(lo, min(hi, raw_intensity)), 4)

        report.matrix_mode = chosen_mode
        report.strategic_intensity_multiplier = clamped
        report.reasoning = enriched.reasoning or f"{chosen_mode} — intensity {clamped}."
        report.target_bracket = enriched.target_bracket or ""

        logger.info("[Strategist-%s]  Decision: %s  |  intensity ×%.4f",
                    team_key, report.matrix_mode, report.strategic_intensity_multiplier)
        lo, hi = _MODE_RANGES.get(report.matrix_mode, (0.78, 1.14))
        logger.info("[Strategist-%s]  Range for this mode: [%.2f, %.2f]", team_key, lo, hi)
        if report.reasoning:
            reasoning_preview = report.reasoning[:300].replace("\n", " ")
            logger.info("[Strategist-%s]  Reasoning: %s", team_key, reasoning_preview)
        if report.target_bracket:
            logger.info("[Strategist-%s]  Target bracket: %s", team_key, report.target_bracket)

        return {
            "teams": {team_key: {"strategy": report.model_dump()}},
            "step_log": [
                f"[Strategist-{team_key}] {name}: {report.matrix_mode} "
                f"| intensity ×{report.strategic_intensity_multiplier} "
                f"| \"{report.reasoning[:120]}\"" if report.reasoning else ""
            ],
        }

    return strategist_node
