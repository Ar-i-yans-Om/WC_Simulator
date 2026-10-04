"""
agents/common/pitch_simulator.py
=================================

Pitch Simulator — Common node. The only node that sees both teams simultaneously.

It produces each team's BASELINE expected goals (xG) for the normal course of
play. Black-swan events (red card / VAR penalty / injury) are NOT modelled here:
they are folded in downstream by the Chaos Agent + Judge as a probability-weighted
mixture, so this node must estimate clean, no-disruption lambdas to avoid
double-counting chaos.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from ...math_engine import elo_base_lambda, home_advantage_factors
from ...schemas import PitchResult, TeamLambda

logger = logging.getLogger(__name__)


class _PitchEstimate(BaseModel):
    lambda_a: float = 1.3
    lambda_b: float = 1.3
    collision_note: str = ""
    reasoning: str = ""


_PITCH_SYSTEM = """\
You are an impartial match analyst assessing a football fixture for a World Cup
prediction model. You receive both teams' tactical plans, recent form and an
ELO-based statistical anchor.

Your task: estimate the BASELINE expected goals (xG) for each team over 90
minutes of NORMAL play, based ONLY on:
  - the ELO / quality gap (the anchor is provided), and
  - how the two tactical styles and formations interact (does one shape exploit
    the other? does a low block smother a possession side? does a high line
    invite direct balls in behind?).

Do NOT factor in squad chemistry, strategic intensity, or fitness/travel fatigue.
Those levers are quantified elsewhere and multiplied into your estimate
automatically afterwards — including them here would DOUBLE-COUNT them. They are
shown to you as context for the collision_note only.

Also do NOT bake in red cards, VAR penalties, freak injuries or other black-swan
disruptions — those are folded in later as a probability-weighted mixture. Assume
a clean, full-strength game.

Typical ranges: 0.5 (very defensive/dominated) to 2.5 (dominant/clinical).
Most World Cup matches fall between 0.8 and 1.9 per team.
Set lambda_a and lambda_b independently — they do not need to be symmetric.\
"""


def _describe_team(name: str, packet: dict, elo: float, group_points: int,
                   knockout: bool = False) -> str:
    form     = packet.get("form", {})
    fitness  = packet.get("fitness", {})
    cohesion = packet.get("cohesion", {})
    strategy = packet.get("strategy", {})
    tactics  = packet.get("tactics", {})

    gs, gc = form.get("avg_goals_scored"), form.get("avg_goals_conceded")
    form_rate = (f"{gs:.2f} scored / {gc:.2f} conceded (recent)"
                 if gs is not None and gc is not None
                 else "no recent-scoring rate available")

    strength = f"ELO {elo}" + ("" if knockout else f", {group_points} group pts")
    return (
        f"=== {name} ===\n"
        f"Strength: {strength}.\n"
        f"Form: {form_rate}. "
        f"Results: {form.get('recent_results', [])}.\n"
        f"Injuries: {form.get('injuries', []) or form.get('suspensions', []) or 'none'}.\n"
        f"Fitness: ×{fitness.get('fitness_degradation_factor', 1.0)} "
        f"(rest {fitness.get('rest_days','?')}d, "
        f"travel {fitness.get('travel_km', 0):.0f}km).\n"
        f"Chemistry: cohesion ×{cohesion.get('cohesion_multiplier', 1.0)} "
        f"({len(cohesion.get('clusters', []))} club cluster(s)).\n"
        f"Strategy: {strategy.get('matrix_mode','?')} — "
        f"intensity ×{strategy.get('strategic_intensity_multiplier', 1.0)}. "
        f"{strategy.get('reasoning','')}.\n"
        f"Tactics: {tactics.get('plan_type','?')} / "
        f"{tactics.get('formation','?')} / {tactics.get('style','?')}. "
        f"Instructions: {tactics.get('key_instructions',[])}. "
        f"Reasoning: {tactics.get('reasoning','')}."
    )


def make_pitch_simulator_node(llms: dict, settings):
    sim_llm = llms["reasoning"]

    def pitch_simulator_node(state: dict) -> dict:
        cfg  = state["config"]
        scn  = cfg["scenario"]
        tournament_avg = scn.get("tournament_avg_goals", 1.35)

        teams  = state.get("teams", {})
        a, b   = teams.get("A", {}), teams.get("B", {})
        a_name = a.get("form", {}).get("name", "Team A")
        b_name = b.get("form", {}).get("name", "Team B")
        a_elo  = scn["team_a"].get("elo", 1500.0)
        b_elo  = scn["team_b"].get("elo", 1500.0)
        a_pts  = scn["team_a"].get("group_points", 0)
        b_pts  = scn["team_b"].get("group_points", 0)
        knockout = bool(scn.get("knockout_round"))

        a_plan = a.get("tactics", {})
        b_plan = b.get("tactics", {})

        anchor_a = round(elo_base_lambda(a_elo, b_elo, tournament_avg), 2)
        anchor_b = round(elo_base_lambda(b_elo, a_elo, tournament_avg), 2)

        # Home advantage: only when a host nation is playing in its OWN country.
        venue_city = scn["team_a"].get("host_city_this_match", "")
        ha_a, ha_b = home_advantage_factors(a_name, b_name, venue_city)
        if (ha_a, ha_b) != (1.0, 1.0):
            anchor_a = round(anchor_a * ha_a, 2)
            anchor_b = round(anchor_b * ha_b, 2)
            _home = a_name if ha_a > ha_b else b_name
            logger.info("[PitchSim]  Home advantage: %s playing in %s "
                        "(×%.2f home / ×%.2f away)",
                        _home, venue_city, max(ha_a, ha_b), min(ha_a, ha_b))

        logger.info("")
        logger.info("[PitchSim] ══ TACTICAL COLLISION (baseline) ═══════════════════")
        logger.info("[PitchSim]  %s  ─  ELO %.0f | λ_anchor %.2f | %s / %s | "
                    "intensity ×%.2f | fitness ×%.3f",
                    a_name, a_elo, anchor_a,
                    a_plan.get("style", "?"), a_plan.get("formation", "?"),
                    a.get("strategy", {}).get("strategic_intensity_multiplier", 1.0),
                    a.get("fitness", {}).get("fitness_degradation_factor", 1.0))
        logger.info("[PitchSim]  %s  ─  ELO %.0f | λ_anchor %.2f | %s / %s | "
                    "intensity ×%.2f | fitness ×%.3f",
                    b_name, b_elo, anchor_b,
                    b_plan.get("style", "?"), b_plan.get("formation", "?"),
                    b.get("strategy", {}).get("strategic_intensity_multiplier", 1.0),
                    b.get("fitness", {}).get("fitness_degradation_factor", 1.0))

        logger.info("[PitchSim]  Asking Gemini to assess baseline expected goals...")
        estimate = sim_llm.structured(
            _PitchEstimate,
            system=_PITCH_SYSTEM,
            user=(
                f"Fixture: {a_name} vs {b_name} "
                f"({scn.get('competition','FIFA World Cup 2026')}, "
                f"{scn.get('stage','Group Stage')}).\n\n"
                f"{_describe_team(a_name, a, a_elo, a_pts, knockout)}\n\n"
                f"{_describe_team(b_name, b, b_elo, b_pts, knockout)}\n\n"
                f"ELO-based statistical anchor (rating gap and home advantage only; "
                f"ignores tactics): "
                f"{a_name} {anchor_a} / {b_name} {anchor_b}.\n\n"
                f"Estimate lambda_a ({a_name} xG) and lambda_b ({b_name} xG) "
                f"for the full 90 minutes of normal play (no black-swan events). "
                f"In collision_note: one sharp sentence on how these two tactical "
                f"approaches interact."
            ),
        )

        # The LLM returns a CLEAN tactical/ELO baseline (it is told NOT to bake in
        # chemistry, intensity or fitness). The quantified levers are then fused in
        # deterministically, so every agent's output actually moves the number:
        # λ = base × coh × int × fit × fit-deg × momentum × dark-horse, clamped
        # to [0.3, 3.5].
        base_a = round(max(0.3, min(3.5, estimate.lambda_a)), 4)
        base_b = round(max(0.3, min(3.5, estimate.lambda_b)), 4)

        coh_a  = a.get("cohesion", {}).get("cohesion_multiplier", 1.0)
        int_a  = a.get("strategy", {}).get("strategic_intensity_multiplier", 1.0)
        fit_a  = a.get("tactics", {}).get("intended_tactical_fit", 1.0)
        fdeg_a = a.get("fitness", {}).get("fitness_degradation_factor", 1.0)
        # Deterministic scenario levers (from actual results.json / manual tuning).
        mom_a  = scn["team_a"].get("tournament_momentum", 1.0)
        dh_a   = scn["team_a"].get("dark_horse_factor", 1.0)

        coh_b  = b.get("cohesion", {}).get("cohesion_multiplier", 1.0)
        int_b  = b.get("strategy", {}).get("strategic_intensity_multiplier", 1.0)
        fit_b  = b.get("tactics", {}).get("intended_tactical_fit", 1.0)
        fdeg_b = b.get("fitness", {}).get("fitness_degradation_factor", 1.0)
        mom_b  = scn["team_b"].get("tournament_momentum", 1.0)
        dh_b   = scn["team_b"].get("dark_horse_factor", 1.0)

        lam_final_a = round(max(0.3, min(3.5, base_a * coh_a * int_a * fit_a * fdeg_a * mom_a * dh_a)), 4)
        lam_final_b = round(max(0.3, min(3.5, base_b * coh_b * int_b * fit_b * fdeg_b * mom_b * dh_b)), 4)
        note        = estimate.collision_note or f"{a_name} vs {b_name}: tactical assessment."

        lam_a = TeamLambda(name=a_name, lambda_base=base_a,
                           cohesion=coh_a, strategic_intensity=int_a,
                           tactical_fit=fit_a, fitness_degradation=fdeg_a,
                           tournament_momentum=mom_a, dark_horse=dh_a,
                           lambda_final=lam_final_a)
        lam_b = TeamLambda(name=b_name, lambda_base=base_b,
                           cohesion=coh_b, strategic_intensity=int_b,
                           tactical_fit=fit_b, fitness_degradation=fdeg_b,
                           tournament_momentum=mom_b, dark_horse=dh_b,
                           lambda_final=lam_final_b)

        if estimate.reasoning:
            logger.info("[PitchSim]  Gemini reasoning: %s",
                        estimate.reasoning[:300].replace("\n", " "))

        logger.info("[PitchSim]  ── RESULT ──────────────────────────────────")
        logger.info("[PitchSim]  λ %s = %.4f = base %.2f × coh %.2f × int %.2f × fit %.2f × fitdeg %.3f × mom %.3f × dh %.2f  (ELO anchor %.2f)",
                    lam_a.name, lam_a.lambda_final, base_a, coh_a, int_a, fit_a, fdeg_a, mom_a, dh_a, anchor_a)
        logger.info("[PitchSim]  λ %s = %.4f = base %.2f × coh %.2f × int %.2f × fit %.2f × fitdeg %.3f × mom %.3f × dh %.2f  (ELO anchor %.2f)",
                    lam_b.name, lam_b.lambda_final, base_b, coh_b, int_b, fit_b, fdeg_b, mom_b, dh_b, anchor_b)
        logger.info("[PitchSim]  Assessment: %s", note)

        result = PitchResult(team_a=lam_a, team_b=lam_b, collision_note=note)
        return {
            "pitch": result.model_dump(),
            "step_log": [
                f"[PitchSim] λ {lam_a.name}={lam_a.lambda_final} | "
                f"λ {lam_b.name}={lam_b.lambda_final} | {note}"
            ],
        }

    return pitch_simulator_node