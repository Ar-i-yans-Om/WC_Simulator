"""
agents/common/chaos_agent.py
============================

Chaos Agent (Black-Swan Modeller) - Common node.

This node does NOT roll a die. Instead of firing a single random event once per
run (which would force you to Monte-Carlo many runs to recover a distribution),
it emits the FULL probability-weighted scenario set for this match in one pass:

    * the no-chaos baseline                                 (weight 1 - p)
    * each black-swan event x each team it could befall      (summing to p)

Each scenario carries the expected multiplicative / additive adjustment to the
base lambdas. The Judge then builds a separate Poisson grid per scenario and
combines them into one mixture distribution - so every possible scenario is
weighted into a single run.

`p` is `settings.chaos_base_probability` (set it to 0.0 for a clean baseline).
The scenario weights, per-event multipliers and the VAR-penalty additive xG live
in `math_engine` as the single chaos tuning point.
"""

from __future__ import annotations
import logging
logger = logging.getLogger(__name__)

from ...math_engine import build_chaos_scenarios
from ...schemas import ChaosModel, ChaosScenario


def make_chaos_node(llms: dict, settings):

    def chaos_node(state: dict) -> dict:
        p = float(getattr(settings, "chaos_base_probability", 0.22))
        scn = state.get("config", {}).get("scenario", {})
        rates_a = scn.get("team_a", {}).get("chaos_profile") or None
        rates_b = scn.get("team_b", {}).get("chaos_profile") or None
        logger.info("")
        logger.info("[Chaos] ── Weighted black-swan model (p=%.2f) ─────────────", p)
        if rates_a or rates_b:
            logger.info("[Chaos]  using per-team historical profiles "
                        "(frequency data-driven, severity fixed)")

        scenarios = [ChaosScenario(**sc) for sc in build_chaos_scenarios(p, rates_a, rates_b)]
        chaos_mass = round(sum(s.probability for s in scenarios if s.event_type != "none"), 4)
        summary = (
            f"No single event is sampled; {chaos_mass:.0%} of the scoreline "
            f"distribution is contributed by black-swan scenarios "
            f"(red card / VAR penalty / injury, either team), "
            f"{1 - chaos_mass:.0%} by the baseline."
        )
        model = ChaosModel(base_probability=p, scenarios=scenarios, summary=summary)

        for s in scenarios:
            if s.event_type == "none":
                logger.info("[Chaos]  baseline                p=%.3f", s.probability)
            else:
                logger.info("[Chaos]  %-24s p=%.3f  (×A %.2f ×B %.2f +A %.2f +B %.2f)",
                            s.label, s.probability, s.mult_a, s.mult_b, s.add_a, s.add_b)

        return {
            "chaos": model.model_dump(),
            "step_log": [f"[Chaos] Weighted model: {summary}"],
        }

    return chaos_node