"""
agents/independent/alchemist.py
================================

Alchemist (Squad Chemistry) — INDEPENDENT, one instance per team.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

from ...schemas import ChemistryCluster, CohesionReport
from .._helpers import clamp, get_team

logger = logging.getLogger(__name__)

_ALCHEMIST_SYSTEM = (
    "You are a squad-chemistry analyst for a single national team. You only see "
    "your own team. Identify clusters of players who play together regularly at "
    "the same club, as these have pre-built understanding. Output the clusters; "
    "do not invent players who were not provided. Each real player appears at "
    "most once; never list the same person twice (e.g. with and without a "
    "'(captain)' tag)."
)

# Tags that can be appended to a player's name in the source data and must be
# stripped before identity comparison, e.g. "Ladislav Krejčí (captain)".
_NAME_TAG_RE = re.compile(r"\s*\((?:captain|c|vc|gk)\)\s*$", re.IGNORECASE)


def _canon(name: str) -> str:
    """Normalize a player name for identity comparison (strip role tags/space)."""
    return _NAME_TAG_RE.sub("", (name or "").strip()).strip()


def _dedup_players(players) -> list:
    """Drop duplicate players (by canonical name), preserving first-seen order
    and preferring the longer/tagged display string."""
    best: dict[str, str] = {}
    order: list[str] = []
    for p in players or []:
        key = _canon(p)
        if not key:
            continue
        if key not in best:
            best[key] = p
            order.append(key)
        elif len(p) > len(best[key]):
            best[key] = p  # keep the more descriptive label (e.g. with tag)
    return [best[k] for k in order]


def _dedup_clusters(clusters):
    """Clean each cluster's player list and drop clusters that fall below 2
    *distinct* players after dedup."""
    cleaned = []
    for c in clusters or []:
        players = _dedup_players(getattr(c, "players", []) or [])
        if len(players) >= 2:
            cleaned.append(ChemistryCluster(club=getattr(c, "club", ""), players=players))
    return cleaned


def _cohesion_from_clusters(clusters, squad_size: int = 26) -> float:
    """Soft cohesion factor. Per-extra-teammate bonus is intentionally small and
    has diminishing returns, so one large club block can't dominate the lambda.
    Total uplift is capped at +8%, so the multiplier spans [1.00, 1.08]."""
    if not clusters:
        return 1.0
    bonus = 0.0
    for c in clusters:
        n = len(getattr(c, "players", []) or [])
        if n >= 2:
            # diminishing: 2->0.01, 3->0.017, 4->0.022, 5->0.025 ...
            extra = n - 1
            bonus += 0.01 * sum(0.7 ** i for i in range(extra))
    return round(clamp(1.0 + min(bonus, 0.08), 0.85, 1.20), 4)


def make_alchemist_node(llms: dict, settings, team_key: str):
    reasoning = llms["reasoning"]

    def alchemist_node(state: dict) -> dict:
        team        = get_team(state, team_key)
        form        = team.get("form", {})
        # squad_clubs is a list of {player, club} entries — fold to a {name: club}
        # map for clustering/logging (a plain dict is accepted too).
        raw_clubs   = form.get("squad_clubs", []) or []
        if isinstance(raw_clubs, dict):
            squad_clubs = dict(raw_clubs)
        else:
            squad_clubs = {e.get("player"): e.get("club", "")
                           for e in raw_clubs if isinstance(e, dict) and e.get("player")}
        name        = form.get("name", team_key)

        logger.info("")
        logger.info("[Alchemist-%s] ── %s squad chemistry ──────────────", team_key, name)

        # ── Input snapshot ────────────────────────────────────────────────
        logger.debug("[Alchemist-%s]  Input: squad_clubs has %d player(s)", team_key, len(squad_clubs))
        if squad_clubs:
            club_freq = Counter(squad_clubs.values())
            top_clubs = sorted(club_freq.items(), key=lambda x: -x[1])[:5]
            logger.debug(
                "[Alchemist-%s]  Top clubs by player count: %s",
                team_key,
                ", ".join(f"{c}={n}" for c, n in top_clubs),
            )
        else:
            logger.debug("[Alchemist-%s]  No squad_clubs data available from Researcher.", team_key)

        # ── Cluster derivation ────────────────────────────────────────────
        # No squad map → nothing to cluster (neutral). With a map, let Gemini
        # group it, then normalize names and dedup so the same player can't form
        # a phantom cluster (e.g. "X (captain)" + "X").
        if not squad_clubs:
            report = CohesionReport(
                cohesion_multiplier=1.0,
                clusters=[],
                reasoning="No squad-club map available; cohesion left neutral.",
            )
        else:
            report = reasoning.structured(
                CohesionReport,
                system=_ALCHEMIST_SYSTEM,
                user=(
                    f"This is the {name} squad. Player → current club map:\n"
                    f"{squad_clubs}\n"
                    "Group players who play for the same club into clusters "
                    "(two or more players each). cohesion_multiplier is computed "
                    "from your clusters, so leave it at its default."
                ),
            )
            report.clusters = _dedup_clusters(report.clusters)
            report.cohesion_multiplier = _cohesion_from_clusters(report.clusters)

        # ── Cluster breakdown ─────────────────────────────────────────────
        if report.clusters:
            logger.info("[Alchemist-%s]  %d club cluster(s) found:", team_key, len(report.clusters))
            for cl in sorted(report.clusters, key=lambda c: -len(c.players or [])):
                players_str = ", ".join(cl.players[:5]) if cl.players else "?"
                suffix = f" … (+{len(cl.players) - 5} more)" if len(cl.players or []) > 5 else ""
                logger.info(
                    "[Alchemist-%s]    %-22s  %d player(s)  →  %s%s",
                    team_key, cl.club, len(cl.players or []), players_str, suffix,
                )
        else:
            logger.info(
                "[Alchemist-%s]  No multi-player club clusters found "
                "(squad data not available or all singletons).",
                team_key,
            )

        logger.info("[Alchemist-%s]  Cohesion multiplier: ×%.4f", team_key, report.cohesion_multiplier)
        if report.reasoning:
            logger.info("[Alchemist-%s]  Reasoning: %s", team_key, report.reasoning[:200])

        # ── State-write summary ───────────────────────────────────────────
        cohesion_dump = report.model_dump()
        logger.info("[Alchemist-%s] ── Writing to AgentState ────────────────────────", team_key)
        logger.info(
            "[Alchemist-%s]  teams[%s]['cohesion']['cohesion_multiplier'] = %.4f",
            team_key, team_key, cohesion_dump["cohesion_multiplier"],
        )
        logger.info(
            "[Alchemist-%s]  teams[%s]['cohesion']['clusters'] = %d cluster(s): %s",
            team_key,
            team_key,
            len(cohesion_dump.get("clusters") or []),
            ", ".join(
                f"{c['club']}×{len(c.get('players') or [])}"
                for c in sorted(
                    cohesion_dump.get("clusters") or [],
                    key=lambda x: -len(x.get("players") or []),
                )[:5]
            ) or "none",
        )
        logger.info(
            "[Alchemist-%s]  teams[%s]['cohesion']['reasoning'] = %.120s%s",
            team_key,
            team_key,
            cohesion_dump.get("reasoning") or "",
            "…" if len(cohesion_dump.get("reasoning") or "") > 120 else "",
        )
        logger.debug(
            "[Alchemist-%s]  Full cohesion dump: %s",
            team_key,
            cohesion_dump,
        )

        return {
            "teams": {team_key: {"cohesion": cohesion_dump}},
            "step_log": [
                f"[Alchemist-{team_key}] {name}: cohesion ×{report.cohesion_multiplier} "
                f"| {len(report.clusters)} cluster(s): "
                + (", ".join(
                    f"{c.club}×{len(c.players or [])}"
                    for c in sorted(report.clusters, key=lambda x: -len(x.players or []))[:3]
                ) or "none")
            ],
        }

    return alchemist_node