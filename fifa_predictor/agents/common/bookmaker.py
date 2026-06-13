"""
agents/common/bookmaker.py
===========================

Bookmaker Agent (Market Anchor) - a Common node.

Provides the "wisdom of the crowd" reality check: implied match probabilities
distilled from real betting markets. It runs in parallel with the team branches
(it only needs the fixture identity, not any tactical state) and feeds the Judge
alongside the model's own Poisson output.

Data acquisition (no mock data):
  1. If an ODDS_API_KEY is configured -> pull live decimal H2H odds from
     the-odds-api.com and convert them to overround-normalised probabilities.
  2. Otherwise -> ask Gemini (Google Search grounding) to scan current public
     market odds and emit a BookmakerReport via the structured path.

Because the betting market is global and impartial, the Bookmaker reads ONLY the
fixture identity from config - never a team's private branch state. That keeps
it a genuine external anchor rather than a leak of the model's own reasoning.
"""

from __future__ import annotations
import logging
logger = logging.getLogger(__name__)

from ...data_sources import (
    implied_probs_from_decimal, odds_api_fixture_odds, team_names_match,
)
from ...schemas import BookmakerReport


_BOOKMAKER_SYSTEM = (
    "You are a betting-market analyst. Report only the current market consensus "
    "for the given fixture: search for published bookmaker or exchange odds and "
    "translate them into implied win/draw/loss probabilities. Do not give your "
    "own prediction - mirror the market."
)


def _market_prompt(team_a: str, team_b: str, competition: str) -> str:
    return (
        f"Find the most recent published betting odds for the {competition} "
        f"match {team_a} vs {team_b}. Report the implied probabilities of a "
        f"{team_a} win, a draw, and a {team_b} win. If exact odds for this "
        f"fixture are unavailable, use the closest current market pricing for "
        f"these two teams and say so. Express each as a probability in [0,1]."
    )


def _match_team_names(events, team_a: str, team_b: str):
    """DEPRECATED — superseded by server-side targeting in odds_api_fixture_odds."""
    for ev in events or []:
        pair = (ev.get("home_team", ""), ev.get("away_team", ""))
        if (any(team_names_match(team_a, n) for n in pair)
                and any(team_names_match(team_b, n) for n in pair)):
            return ev
    return None


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return None
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _odds_from_event(ev: dict, team_a: str, team_b: str):
    """
    Extract MEDIAN decimal H2H odds (home=team_a, draw, away=team_b) from a
    the-odds-api event. Median (not mean) is used so a single mis-priced or
    swapped book can't skew the consensus. Outcomes are mapped to our teams by
    alias-aware name matching, and `h2h_lay` exchange markets are ignored.
    """
    home_prices, draw_prices, away_prices = [], [], []
    for bk in ev.get("bookmakers", []):
        for market in bk.get("markets", []):
            if market.get("key") != "h2h":          # skip h2h_lay etc.
                continue
            for outcome in market.get("outcomes", []):
                name = outcome.get("name") or ""
                price = outcome.get("price")
                if not price:
                    continue
                if name.strip().lower() == "draw":
                    draw_prices.append(price)
                elif team_names_match(team_a, name):
                    home_prices.append(price)
                elif team_names_match(team_b, name):
                    away_prices.append(price)

    if not (home_prices and draw_prices and away_prices):
        return None

    return _median(home_prices), _median(draw_prices), _median(away_prices)


def make_bookmaker_node(llms: dict, settings):
    research_llm = llms["research"]
    structure_llm = llms["reasoning"]

    def bookmaker_node(state: dict) -> dict:
        logger.info("")
        logger.info("[Bookmaker] ── Scanning betting markets ───────────────────")
        cfg = state["config"]
        scn = cfg["scenario"]
        team_a = scn["team_a"]["name"]
        team_b = scn["team_b"]["name"]
        competition = scn.get("competition", "FIFA World Cup 2026")

        # --- Path 1: live odds API (targeted single fixture) --------------
        if settings.odds_api_key:
            match_date = scn["team_a"].get("date")
            ev = odds_api_fixture_odds(
                settings.odds_api_key, team_a, team_b, date=match_date,
            )
            if ev:
                odds = _odds_from_event(ev, team_a, team_b)
                if odds:
                    probs = implied_probs_from_decimal(*odds)
                    report = BookmakerReport(
                        source="the-odds-api.com (live H2H, bookmaker-median)",
                        reasoning=(
                            f"Median decimal odds across {len(ev.get('bookmakers', []))} "
                            f"books: home {odds[0]:.2f} / draw {odds[1]:.2f} / "
                            f"away {odds[2]:.2f}."
                        ),
                        **probs,
                    )
                    logger.info("[Bookmaker]  Source: %s", report.source)
                    logger.info("[Bookmaker]  %s win=%.1f%%  draw=%.1f%%  %s win=%.1f%%",
                                team_a, report.implied_home_win*100,
                                report.implied_draw*100,
                                team_b, report.implied_away_win*100)
                    return {
                        "bookmaker": report.model_dump(),
                        "step_log": [
                            f"[Bookmaker] Live market: {team_a} "
                            f"{report.implied_home_win:.0%} / draw "
                            f"{report.implied_draw:.0%} / {team_b} "
                            f"{report.implied_away_win:.0%}."
                        ],
                    }

        # --- Path 2: web_search market scan -> structured ------------------
        raw = research_llm.research(
            _BOOKMAKER_SYSTEM,
            _market_prompt(team_a, team_b, competition),
            max_searches=max(2, settings.max_web_searches // 2),
        )
        report = structure_llm.structured(
            BookmakerReport,
            system=(
                "Convert the market analyst's notes into a BookmakerReport. "
                "implied_home_win is the probability the FIRST team wins, "
                "implied_away_win the SECOND team. The three probabilities "
                "should sum to roughly 1.0."
            ),
            user=(
                f"First team (home slot): {team_a}. Second team (away slot): "
                f"{team_b}. Emit the implied probabilities."
            ),
            context=raw,
        )
        # Normalise in case the model's three probs don't sum to 1.
        tot = report.implied_home_win + report.implied_draw + report.implied_away_win
        if tot > 0:
            report.implied_home_win = round(report.implied_home_win / tot, 4)
            report.implied_draw = round(report.implied_draw / tot, 4)
            report.implied_away_win = round(report.implied_away_win / tot, 4)
        logger.info("[Bookmaker]  Source: %s", report.source)
        logger.info("[Bookmaker]  %s win=%.1f%%  draw=%.1f%%  %s win=%.1f%%",
                    team_a, report.implied_home_win*100,
                    report.implied_draw*100,
                    team_b, report.implied_away_win*100)
        return {
            "bookmaker": report.model_dump(),
            "step_log": [
                f"[Bookmaker] Market scan: {team_a} "
                f"{report.implied_home_win:.0%} / draw "
                f"{report.implied_draw:.0%} / {team_b} "
                f"{report.implied_away_win:.0%}."
            ],
        }

    return bookmaker_node