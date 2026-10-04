"""
agents/common/judge.py
=======================

Judge Agent (The Final Critic) - a Common node and the graph's terminus.

Responsibilities:
  1. Build the scoreline distribution as a MIXTURE of Poisson grids - one grid
     per chaos scenario (including the no-chaos baseline), weighted by each
     scenario's probability (math_engine.mixture_grid). The chaos contribution
     is therefore folded into this single run rather than sampled. Result: exact
     0-0..N-N probabilities aggregated into Win/Draw/Loss. Always real maths.
  2. Reconcile the model against the Bookmaker's market anchor.
  3. Compile the executive FinalReport.

The numeric verdict (probabilities, predicted scoreline, confidence) is computed
deterministically here. The judge LLM only writes the *prose* - headline,
narrative, model-vs-market read, chaos commentary.
"""

from __future__ import annotations
import logging
logger = logging.getLogger(__name__)

from ...math_engine import mixture_grid, poisson_grid
from ...schemas import FinalReport, JudgeProse, PoissonReport


_JUDGE_SYSTEM = (
    "You are the head analyst delivering a final, executive match prediction. "
    "You are given a deterministic statistical model output (Poisson scoreline "
    "probabilities), a betting-market anchor, and a chaos model. The scoreline "
    "distribution is a probability-weighted MIXTURE that already folds in "
    "black-swan scenarios (red card / VAR penalty / injury, weighted by their "
    "likelihood) - so chaos is baked into the numbers, not a separate 'what if'. "
    "Write a crisp, honest verdict. Respect the model's numbers - do not invent "
    "different probabilities - but interpret them, compare them to the market, "
    "and explain how much the weighted chaos contribution moved the picture. "
    "Be specific and avoid hype."
)


def _confidence(home_win: float, draw: float, away_win: float) -> str:
    """Confidence from how decisively one outcome leads the field."""
    top = max(home_win, draw, away_win)
    spread = top - sorted([home_win, draw, away_win])[1]  # lead over 2nd place
    if top >= 0.55 or spread >= 0.20:
        return "high"
    if top >= 0.42 or spread >= 0.10:
        return "medium"
    return "low"


def _verdict_word(home_win: float, draw: float, away_win: float,
                  team_a: str, team_b: str) -> str:
    if home_win >= draw and home_win >= away_win:
        return f"{team_a} win"
    if away_win >= home_win and away_win >= draw:
        return f"{team_b} win"
    return "Draw"


def make_judge_node(llms: dict, settings):
    judge_llm = llms["judge"]

    def judge_node(state: dict) -> dict:
        cfg = state["config"]
        scn = cfg["scenario"]
        team_a = scn["team_a"]["name"]
        team_b = scn["team_b"]["name"]

        pitch = state.get("pitch", {})
        lam_a = pitch.get("team_a", {}).get("lambda_final", 1.3)
        lam_b = pitch.get("team_b", {}).get("lambda_final", 1.3)

        chaos = state.get("chaos", {})
        scenarios = chaos.get("scenarios", [])
        max_goals = state["config"].get("settings", {}).get("max_goals", 7)

        logger.info("")
        logger.info("[Judge] ══ MIXTURE POISSON VERDICT ═══════════════════════════")
        logger.info("[Judge]  Baseline xG → %s λ=%.4f  ·  %s λ=%.4f", team_a, lam_a, team_b, lam_b)

        # --- 1. Mixture of Poisson grids (home = team_a) ------------------
        # One grid per chaos scenario (incl. the no-chaos baseline), weighted by
        # probability, combined into a single distribution. Falls back to a plain
        # baseline grid if the chaos model is somehow absent.
        if scenarios:
            grid = mixture_grid(lam_a, lam_b, scenarios, max_goals=max_goals)
        else:
            grid = poisson_grid(lam_a, lam_b, max_goals=max_goals)
        poisson = PoissonReport(**grid)

        logger.info("[Judge]  Poisson W/D/L: %s %.1f%%  ·  Draw %.1f%%  ·  %s %.1f%%",
                    team_a, poisson.home_win*100, poisson.draw*100, team_b, poisson.away_win*100)
        top = poisson.most_likely_scores[0] if poisson.most_likely_scores else None
        predicted_scoreline = (
            f"{team_a} {top.home}-{top.away} {team_b}" if top else "n/a"
        )
        wdl = {
            f"{team_a}_win": poisson.home_win,
            "draw": poisson.draw,
            f"{team_b}_win": poisson.away_win,
        }
        confidence = _confidence(poisson.home_win, poisson.draw, poisson.away_win)
        verdict = _verdict_word(poisson.home_win, poisson.draw, poisson.away_win,
                                team_a, team_b)

        # --- 2. Market reconciliation (deterministic fallback text) --------
        bm = state.get("bookmaker", {})
        m_home = bm.get("implied_home_win", 0.0)
        m_draw = bm.get("implied_draw", 0.0)
        m_away = bm.get("implied_away_win", 0.0)
        if poisson.most_likely_scores:
            scores_str = "  ".join(f"{s.home}-{s.away}({s.prob:.0%})" for s in poisson.most_likely_scores[:4])
            logger.info("[Judge]  Top scorelines: %s", scores_str)
        edge = poisson.home_win - m_home
        market_text = (
            f"Model gives {team_a} {poisson.home_win:.0%} vs market {m_home:.0%} "
            f"(edge {edge:+.0%}); draw {poisson.draw:.0%} vs {m_draw:.0%}; "
            f"{team_b} {poisson.away_win:.0%} vs {m_away:.0%}."
        )

        # --- 3. Chaos impact (deterministic fallback text) -----------------
        chaos_mass = round(
            sum(s.get("probability", 0.0) for s in scenarios if s.get("event_type") != "none"),
            4,
        )
        if chaos_mass > 0:
            chaos_text = (
                f"Black-swan scenarios (red card / VAR penalty / injury) carry "
                f"{chaos_mass:.0%} of the probability mass and are already folded "
                f"into the W/D/L and scoreline grid via a weighted mixture of "
                f"{len(scenarios)} Poisson grids; the remaining {1 - chaos_mass:.0%} "
                f"is the baseline. The baseline xG excludes these events; the "
                f"grid's mean includes them."
            )
        else:
            chaos_text = "Chaos disabled (p=0); the distribution is the pure baseline."

        # --- 4. Build the deterministic report (authoritative numbers) -----
        report = FinalReport(
            headline=f"{verdict} most likely ({predicted_scoreline}).",
            predicted_scoreline=predicted_scoreline,
            win_draw_loss=wdl,
            model_vs_market=market_text,
            chaos_impact=chaos_text,
            narrative=(
                f"Baseline xG (pre-chaos): {team_a} {lam_a} vs {team_b} {lam_b}; chaos folded into the grid below. "
                f"{poisson.home_win:.0%}/{poisson.draw:.0%}/{poisson.away_win:.0%} "
                f"W/D/L. {pitch.get('collision_note','')}"
            ),
            confidence=confidence,
        )

        # --- 5. LLM prose enrichment -------------------------------------
        context = (
            f"Fixture: {team_a} (home slot) vs {team_b}.\n"
            f"Baseline xG (pre-chaos): {team_a}={lam_a}, {team_b}={lam_b}.\n"
            f"Poisson W/D/L: {poisson.home_win:.4f}/{poisson.draw:.4f}/"
            f"{poisson.away_win:.4f}.\n"
            f"Top scorelines: "
            + ", ".join(
                f"{s.home}-{s.away} ({s.prob:.1%})"
                for s in poisson.most_likely_scores[:4]
            )
            + f".\nMarket: {market_text}\nChaos: {chaos_text}\n"
            f"Collision: {pitch.get('collision_note','')}"
        )
        # Ask the model to PRODUCE prose, not preserve numbers: "keep X / use
        # exact Y / enrich the rest" phrasing cues small reasoning models to
        # treat the object as pre-filled and return all-empty strings.
        enriched = judge_llm.structured(
            JudgeProse,
            system=_JUDGE_SYSTEM,
            user=(
                "Write the executive match report as JSON. The statistical model "
                "is final and appears in REFERENCE DATA — do not recompute or "
                "alter any probabilities or the scoreline.\n\n"
                "Populate ALL FOUR fields with original prose (every one non-empty):\n"
                "  • headline — one punchy sentence naming the likely result.\n"
                "  • model_vs_market — 1-2 sentences on where the model and "
                "market agree or diverge and what that implies.\n"
                "  • narrative — 2-4 sentences on WHY the match projects this way "
                "(tactics, chemistry, fitness, the weighted chaos mass), not a "
                "restatement of the percentages.\n"
                "  • chaos_impact — one sentence on how the weighted black-swan "
                "contribution shifts the picture."
            ),
            context=context,
        )
        # Keep deterministic numbers authoritative; take the LLM's prose.
        report.headline = enriched.headline or report.headline
        report.narrative = enriched.narrative or report.narrative
        if enriched.model_vs_market:
            report.model_vs_market = enriched.model_vs_market
        if enriched.chaos_impact:
            report.chaos_impact = enriched.chaos_impact
        report.predicted_scoreline = predicted_scoreline
        report.win_draw_loss = wdl
        report.confidence = confidence

        logger.info("[Judge]  Market anchor: %s %.1f%%  draw %.1f%%  %s %.1f%%",
                    team_a, m_home*100, m_draw*100, team_b, m_away*100)
        logger.info("[Judge]  ─── VERDICT: %s | %s | confidence %s ───",
                    verdict, predicted_scoreline, confidence)
        if report.narrative:
            logger.info("[Judge]  Narrative: %s", report.narrative[:300].replace("\n"," "))
        logger.info("━" * 60)
        return {
            "poisson": poisson.model_dump(),
            "report": report.model_dump(),
            "step_log": [
                f"[Judge] {verdict}: {predicted_scoreline} | "
                f"W/D/L {poisson.home_win:.0%}/{poisson.draw:.0%}/"
                f"{poisson.away_win:.0%} | confidence {confidence}."
            ],
        }

    return judge_node