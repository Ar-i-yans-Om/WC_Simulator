"""
agents/common/researcher.py
============================

Researcher (The Global Pulse) — Common node.

Acts as the objective data lifeline for the entire system. Fetches raw
statistics, recent team form, injury lists, suspension updates, possible
starting lineups, and scheduling context for BOTH teams simultaneously,
then consolidates everything — including deterministic fitness facts — into
a single pre-match intelligence report passed to all downstream agents.

Design principles:
  - Web search: recent results, injuries, suspensions, likely lineups,
    group standings, manager quotes, head-to-head. Raw facts, no strategy.
  - players.json: factual squad roster anchor. Cross-referenced against live
    injury/suspension news. squad_clubs map seeded here for the Alchemist.
  - Fitness (rest days, travel, altitude, climate): computed deterministically
    from data_sources.py — these are objective scheduling facts, not opinions.
    They are surfaced alongside research findings so every downstream agent
    has the full physical picture.
  - ELO and FIFA rank: passed through as raw strength signals. Never used to
    derive goals or probabilities here — that is the Pitch Simulator's job.
  - No dry-run branch. The node always runs fully.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from ...data_sources import (
    compute_fitness_profile, travel_distance_km, venue_altitude_m,
    venue_climate, team_home_climate,
)
from ...schemas import GlobalResearch, TeamForm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# players.json loader
# ---------------------------------------------------------------------------

def _load_players_roster(data_dir: str | Path | None = None) -> dict[str, list[dict]]:
    """
    Load players.json → dict keyed by country name:
        { "Mexico": [ {No, Pos, Player, DOB_Age, Caps, Goals, Club}, ... ], ... }

    Walks up from this file to find the project root's data/ directory if
    data_dir is not supplied. Returns {} silently if the file is absent.
    """
    if data_dir is None:
        # researcher.py → common/ → agents/ → fifa_predictor/ → project_root/
        # 4 parents up from this file reaches the outer project root where data/ lives.
        # Layout is fixed and known: do not use a walk-up loop (fragile if any
        # intermediate directory ever gains its own data/ folder).
        project_root = Path(__file__).resolve().parents[3]
        data_dir = project_root / "data"

    if data_dir is None:
        logger.warning("[Researcher] players.json not found — roster will be empty")
        return {}

    players_path = Path(data_dir) / "players.json"
    if not players_path.exists():
        logger.warning("[Researcher] players.json not found at %s", players_path)
        return {}

    try:
        with open(players_path, encoding="utf-8") as fh:
            raw: list[dict] = json.load(fh)
        roster: dict[str, list[dict]] = {}
        for player in raw:
            country = player.get("Country", "")
            if country:
                roster.setdefault(country, []).append(player)
        logger.info(
            "[Researcher] Loaded players.json — %d players across %d teams",
            len(raw), len(roster),
        )
        return roster
    except Exception as exc:
        logger.warning("[Researcher] Failed to parse players.json: %s", exc)
        return {}


def _squad_clubs_from_roster(players: list[dict]) -> dict[str, str]:
    """Return {player_name: club} for every player in the roster list."""
    return {
        p["Player"]: p["Club"]
        for p in players
        if p.get("Player") and p.get("Club")
    }


def _roster_summary(players: list[dict]) -> str:
    """
    Compact positional text block injected into the research prompt so Gemini
    can cross-reference known names against live injury/suspension news.

    Example output:
        GK: Raúl Rangel (Guadalajara, 14 caps)
        DF: Jorge Sánchez (PAOK, 59 caps, 3 goals), ...
    """
    by_pos: dict[str, list[str]] = {"GK": [], "DF": [], "MF": [], "FW": []}
    for p in players:
        pos = p.get("Pos", "MF")
        goals = p.get("Goals", 0)
        line = (
            f"{p.get('Player', '?')} "
            f"({p.get('Club', '?')}, {p.get('Caps', 0)} caps"
            + (f", {goals} goals" if goals else "")
            + ")"
        )
        by_pos.setdefault(pos, []).append(line)

    parts: list[str] = []
    for pos in ("GK", "DF", "MF", "FW"):
        group = by_pos.get(pos, [])
        if group:
            parts.append(f"  {pos}: {',  '.join(group)}")
    return "\n".join(parts) if parts else "  (no roster data available)"


# ---------------------------------------------------------------------------
# Fitness helpers
# ---------------------------------------------------------------------------

def _compute_fitness(team_scn: dict) -> dict:
    """
    Deterministically compute the physical fitness profile from scheduling
    facts. Returns the plain dict produced by compute_fitness_profile()
    (do NOT call .model_dump() on it).
    """
    travel = travel_distance_km(
        team_scn.get("host_city_last_match", ""),
        team_scn.get("host_city_this_match", ""),
    )
    venue_city = team_scn.get("host_city_this_match", "")
    altitude = venue_altitude_m(venue_city)
    v_temp, v_humid = venue_climate(venue_city)
    h_temp, h_humid, alt_acclim = team_home_climate(team_scn.get("name", ""))
    return compute_fitness_profile(
        rest_days=team_scn.get("rest_days", 4),
        travel_km=travel,
        altitude_m=altitude,
        is_host_nation=team_scn.get("is_host_nation", False),
        venue_temp_c=v_temp,
        venue_humidity=v_humid,
        home_temp_c=h_temp,
        home_humidity=h_humid,
        acclimatised_to_altitude=alt_acclim,
    )


def _fitness_context_block(team_name: str, fit: dict, team_scn: dict) -> str:
    """
    Human-readable fitness summary injected into the consolidated report
    prompt so Gemini can reason about it alongside web-search findings.
    """
    travel_km = fit.get("travel_km", 0)
    alt_m = fit.get("altitude_m", 0)
    rest = fit.get("rest_days", 4)
    factor = fit.get("fitness_degradation_factor", 1.0)
    r_pen = fit.get("rest_penalty", 0.0)
    t_pen = fit.get("travel_penalty", 0.0)
    c_pen = fit.get("climate_penalty", 0.0)

    lines = [
        f"{team_name} PHYSICAL / SCHEDULING CONTEXT:",
        f"  Rest days since last match : {rest}d",
        f"  Travel to this venue        : {travel_km:.0f} km"
        + (f" (from {team_scn.get('host_city_last_match','?')} to "
           f"{team_scn.get('host_city_this_match','?')})" if travel_km > 0 else ""),
        f"  Venue altitude              : {alt_m:.0f} m"
        + (" (high-altitude adjustment applies)" if alt_m >= 1500 else ""),
        f"  Fitness degradation factor  : ×{factor:.3f}",
        f"  Component penalties         : rest={r_pen:.3f}  travel={t_pen:.3f}  climate={c_pen:.3f}",
    ]
    if team_scn.get("is_host_nation"):
        lines.append("  Host-nation advantage       : yes (crowd / familiarity)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_RESEARCH_SYSTEM = """\
You are the head analyst for a national football coaching staff preparing for a
FIFA World Cup 2026 group-stage match.

YOUR JOB IS TO GATHER AND CONSOLIDATE RAW FACTS into a pre-match intelligence
report. Do not evaluate tactics, recommend strategy, or predict outcomes —
those tasks belong to other staff members who will read your report.

Search the web thoroughly using multiple queries. Prioritise:
  • Official team news, squad announcements, and verified press conferences
  • Reputable sports news outlets (BBC Sport, ESPN, Sky Sports, The Athletic,
    Reuters, AFP, AS, Marca, L'Équipe, Goal.com, Transfermarkt, Sofascore)
  • FIFA and confederation official channels
  • Live pre-match reports published in the last 72 hours

For each finding: cite the publication name. Flag anything unconfirmed with
"(unconfirmed)". State explicitly when information is unavailable rather than
guessing. Report each team in full before moving to the next.\
"""


def _research_prompt(
    scn: dict,
    a_roster: str,
    b_roster: str,
    a_fitness_block: str,
    b_fitness_block: str,
) -> str:
    a = scn["team_a"]
    b = scn["team_b"]
    competition = scn.get("competition", "FIFA World Cup 2026")

    return f"""\
══════════════════════════════════════════════════════════════
PRE-MATCH INTELLIGENCE BRIEF — {a['name'].upper()} vs {b['name'].upper()}
Competition : {competition}
Group {a.get('group', '?')}  |  Matchday {a.get('matchday', '?')}  |  {a.get('date', '2026')}
Venue city  : {a.get('host_city_this_match', 'TBC')}
══════════════════════════════════════════════════════════════

STRENGTH SIGNALS (reference only — do not compute anything from these):
  {a['name']}: ELO {a.get('elo', '?')} · FIFA rank #{a.get('fifa_rank', '?')} · {a.get('confederation', '')}
  {b['name']}: ELO {b.get('elo', '?')} · FIFA rank #{b.get('fifa_rank', '?')} · {b.get('confederation', '')}
  Group points entering this match: {a['name']}={a.get('group_points', 0)} pts  {b['name']}={b.get('group_points', 0)} pts

──────────────────────────────────────────────────────────────
COMPUTED PHYSICAL & SCHEDULING CONTEXT
(These are objective facts already calculated — incorporate them into
your report alongside what you find via web search)

{a_fitness_block}

{b_fitness_block}
──────────────────────────────────────────────────────────────

KNOWN SQUAD ROSTERS (from official pre-tournament registration)
Use these as your reference list when searching for injury/suspension news.
Cross-reference player names carefully — flag who is out, doubtful, or carrying knocks.

{a['name']} SQUAD:
{a_roster}

{b['name']} SQUAD:
{b_roster}
──────────────────────────────────────────────────────────────

For EACH team, search and report the following. Keep teams clearly separated.

─── SECTION 1 · RECENT RESULTS (last 5–6 competitive matches) ───
  For each match: date · opponent · competition · venue (H/A/N) · scoreline.
  List chronologically, most recent last.
  Raw scorelines only — do NOT compute averages.

─── SECTION 2 · INJURY & SUSPENSION STATUS ──────────────────────
  Search for the latest (last 72 h) official injury and suspension news.
  For each player: name · issue · status (confirmed out / doubtful / knock)
  · expected return if known.
  Cross-reference against the squad list above.
  State "none confirmed as of [search date]" if nothing found.

─── SECTION 3 · POSSIBLE STARTING LINEUP ────────────────────────
  Based on the latest manager press conference, rotation patterns, and
  beat-reporter predictions, report the most likely starting XI.
  Format: Formation (e.g. 4-3-3), then GK · defenders · midfielders · forwards
  with club in parentheses. Flag confirmed starters and confirmed rests.
  Note any positional battles or uncertainty.

─── SECTION 4 · SQUAD FITNESS & PHYSICAL NOTES ──────────────────
  Cross-reference the computed physical context above with any published
  reporting on fatigue, overloaded players, or acclimatisation concerns.
  Are there publicly reported worries about high-altitude adaptation for
  this venue? Any players singled out as carrying knocks or heavy minutes?
  Flag relevant quotes from the manager.

─── SECTION 5 · GROUP STANDINGS & QUALIFICATION CONTEXT ─────────
  Current official Group {a.get('group', '?')} standings: P W D L GF GA GD Pts for all teams.
  What does each result scenario (win/draw/loss) mean for both sides?
  Cite published qualification arithmetic or journalist analysis.

─── SECTION 6 · MANAGER PRESS CONFERENCE & MORALE ──────────────
  Direct quotes or attributed summaries from both managers in the last
  48–72 h. Any reported morale issues, motivation narratives, milestone
  caps, revenge-fixture context, or internal team news.

─── SECTION 7 · HEAD-TO-HEAD RECORD ────────────────────────────
  Last 5 meetings: date · competition · venue · scoreline.
  Any significant historical patterns mentioned in published reporting.

Search {a['name']} across all sections first, then {b['name']}.
Cite publication names throughout. Be specific and factual.
"""


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log_team(label: str, form: TeamForm, fit: dict, players_raw: list[dict]) -> None:
    """Emit the full per-team intelligence picture to the terminal."""
    logger.info("")
    logger.info("  ┌─ %s ─────────────────────────────────────────", label.upper())

    # Form stats — only print if populated from live data (not schema defaults)
    if form.avg_goals_scored and form.avg_goals_scored != 1.3:
        logger.info("  │  Form      : scored %.2f / conceded %.2f (last 5–6 matches)",
                    form.avg_goals_scored, form.avg_goals_conceded)
    if form.recent_results:
        logger.info("  │  Results   : %s", "  ".join(form.recent_results[:6]))

    # Fitness
    logger.info(
        "  │  Fitness   : ×%.3f  (rest %dd · travel %.0f km · altitude %.0f m)",
        fit["fitness_degradation_factor"],
        fit["rest_days"],
        fit["travel_km"],
        fit["altitude_m"],
    )
    if fit["travel_km"] > 0 or fit["altitude_m"] > 0:
        logger.info(
            "  │             rest_pen=%.3f · travel_pen=%.3f · climate_pen=%.3f",
            fit["rest_penalty"], fit["travel_penalty"], fit["climate_penalty"],
        )

    # Injury / suspension
    injuries = form.injuries or []
    suspensions = form.suspensions or []
    logger.info("  │  Injuries  : %s", ", ".join(injuries) if injuries else "none confirmed")
    if suspensions:
        logger.info("  │  Suspended : %s", ", ".join(suspensions))

    # Likely lineup
    if form.likely_lineup:
        logger.info("  │  Lineup    : %s", form.likely_lineup)

    # Publicly known weaknesses
    weaknesses = form.public_weaknesses or []
    if weaknesses:
        logger.info("  │  Weaknesses: %s", " · ".join(weaknesses))

    # Club distribution from verified squad
    squad = form.squad_clubs or {}
    if squad:
        clubs = Counter(squad.values())
        top = sorted(clubs.items(), key=lambda x: -x[1])[:6]
        logger.info("  │  Club spread (top): %s",
                    "  ".join(f"{c}×{n}" for c, n in top))
    elif players_raw:
        logger.info("  │  Roster    : %d players registered", len(players_raw))

    logger.info("  └────────────────────────────────────────────────────")


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------

def make_researcher_node(llms: dict, settings, data_dir: str | Path | None = None):
    research_llm = llms["research"]
    structure_llm = llms["reasoning"]

    # Load roster once at graph-build time — shared across all fixtures
    all_rosters = _load_players_roster(data_dir)

    def researcher_node(state: dict) -> dict:
        cfg   = state["config"]
        scn   = cfg["scenario"]
        a_scn = scn["team_a"]
        b_scn = scn["team_b"]

        logger.info("")
        logger.info("━" * 60)
        logger.info(
            "[Researcher] 🔍  %s vs %s  |  %s  |  Group %s  MD%s",
            a_scn["name"], b_scn["name"],
            scn.get("competition", "FIFA World Cup 2026"),
            a_scn.get("group", "?"),
            a_scn.get("matchday", "?"),
        )
        logger.info(
            "[Researcher] ELO: %s %.0f  ·  %s %.0f  |  Group pts: %s=%d  %s=%d",
            a_scn["name"], a_scn.get("elo", 0),
            b_scn["name"], b_scn.get("elo", 0),
            a_scn["name"], a_scn.get("group_points", 0),
            b_scn["name"], b_scn.get("group_points", 0),
        )

        # ── 1. Roster seed (no API, always runs) ─────────────────────────
        a_players = all_rosters.get(a_scn["name"], [])
        b_players = all_rosters.get(b_scn["name"], [])
        a_squad_clubs = _squad_clubs_from_roster(a_players)
        b_squad_clubs = _squad_clubs_from_roster(b_players)

        logger.info(
            "[Researcher] Roster: %s=%d players  ·  %s=%d players",
            a_scn["name"], len(a_players),
            b_scn["name"], len(b_players),
        )

        # ── 2. Fitness (deterministic, always runs) ───────────────────────
        logger.info("[Researcher] Computing fitness profiles...")
        a_fit = _compute_fitness(a_scn)
        b_fit = _compute_fitness(b_scn)
        logger.info(
            "[Researcher] Fitness: %s ×%.3f  ·  %s ×%.3f",
            a_scn["name"], a_fit["fitness_degradation_factor"],
            b_scn["name"], b_fit["fitness_degradation_factor"],
        )

        # ── 3. Initialise forms from roster baseline ──────────────────────
        a_form = TeamForm(
            name=a_scn["name"],
            fifa_rank=a_scn.get("fifa_rank", 30),
            squad_clubs=a_squad_clubs,
        )
        b_form = TeamForm(
            name=b_scn["name"],
            fifa_rank=b_scn.get("fifa_rank", 30),
            squad_clubs=b_squad_clubs,
        )

        fixture_summary = ""
        sources: list[str] = []

        # ── 4. Live web research ──────────────────────────────────────────
        logger.info("[Researcher] Dispatching Google Search grounding...")

        a_fitness_block = _fitness_context_block(a_scn["name"], a_fit, a_scn)
        b_fitness_block = _fitness_context_block(b_scn["name"], b_fit, b_scn)

        raw_text = research_llm.research(
            _RESEARCH_SYSTEM,
            _research_prompt(
                scn,
                _roster_summary(a_players),
                _roster_summary(b_players),
                a_fitness_block,
                b_fitness_block,
            ),
            max_searches=settings.max_web_searches,
        )
        logger.info("[Researcher] Search complete — structuring findings...")

        # ── 5. Structure the raw briefing into schema ─────────────────────
        structured: GlobalResearch = structure_llm.structured(
            GlobalResearch,
            system=(
                "You are converting a raw pre-match intelligence briefing into a "
                "structured data schema. Extract facts as reported — do not infer, "
                "compute, or add anything absent from the briefing.\n\n"
                f"team_a_form MUST have name='{a_scn['name']}'. "
                f"team_b_form MUST have name='{b_scn['name']}'.\n\n"
                "FIELD GUIDANCE:\n"
                "  recent_results   : each entry a short string, e.g. 'W 2-1 vs France (WCQ)'.\n"
                "  injuries         : confirmed-out players only. Format: 'Name (reason)'.\n"
                "  suspensions      : confirmed suspensions only.\n"
                "  likely_lineup    : predicted XI and formation if reported, e.g. "
                "'4-3-3: Lloris; Pavard Upamecano Konate Hernandez; ...'.\n"
                "  squad_clubs      : player_name → club for ALL squad members. "
                "The pre-loaded roster data is the ground truth; update only where "
                "the briefing reports a transfer or loan change.\n"
                "  avg_goals_scored / avg_goals_conceded: leave at schema defaults — "
                "do NOT compute these from raw scorelines.\n"
                "  public_weaknesses: only weaknesses explicitly cited in published "
                "reporting. No editorial additions.\n"
                "  fitness fields   : leave at defaults — already computed separately."
            ),
            user=(
                f"Team A is {a_scn['name']}, Team B is {b_scn['name']}.\n"
                "Populate both teams fully from the briefing. "
                "Never leave name as 'Unknown'. "
                "Merge squad_clubs: use the roster seed as the base, overwrite only "
                "where the briefing reports a confirmed club change."
            ),
            context=raw_text,
        )

        # ── 6. Merge live findings into roster-seeded forms ───────────────
        _EMPTY_DEFAULTS = {
            attr: getattr(TeamForm(), attr, None)
            for attr in ("injuries", "suspensions", "public_weaknesses",
                         "recent_results", "likely_lineup",
                         "avg_goals_scored", "avg_goals_conceded")
        }

        def _merge_live(base: TeamForm, live: TeamForm, seed_clubs: dict) -> TeamForm:
            """
            Enrich the roster-seeded base form with live research findings.
            squad_clubs: seed (players.json) is the ground truth; live
            overwrites only on conflict (transfer / loan corrections).
            """
            for attr, empty_val in _EMPTY_DEFAULTS.items():
                val = getattr(live, attr, None)
                if val is not None and val != empty_val:
                    setattr(base, attr, val)

            merged_clubs = dict(seed_clubs)
            if live.squad_clubs:
                merged_clubs.update(live.squad_clubs)
            base.squad_clubs = merged_clubs
            return base

        a_form = _merge_live(a_form, structured.team_a_form, a_squad_clubs)
        b_form = _merge_live(b_form, structured.team_b_form, b_squad_clubs)

        # Enforce correct names regardless of what the LLM wrote
        a_form.name = a_scn["name"]
        b_form.name = b_scn["name"]
        fixture_summary = structured.fixture_summary
        sources = structured.sources

        # ── 7. Log the consolidated intelligence picture ──────────────────
        logger.info("[Researcher] ── CONSOLIDATED INTELLIGENCE BRIEF ─────────")
        _log_team(a_form.name, a_form, a_fit, a_players)
        _log_team(b_form.name, b_form, b_fit, b_players)
        if fixture_summary:
            logger.info("[Researcher] Fixture context: %s", fixture_summary)
        if sources:
            logger.info("[Researcher] Sources cited  : %s", ", ".join(sources[:6]))
        logger.info("━" * 60)

        # ── 8. Step log entries for the audit trail ───────────────────────
        def _step_entry(form: TeamForm, fit: dict, scn_team: dict) -> str:
            injuries_str  = ", ".join(form.injuries)    if form.injuries    else "none"
            sus_str       = ", ".join(form.suspensions) if form.suspensions else "none"
            results_str   = ", ".join(form.recent_results[:3]) if form.recent_results else "n/a"
            lineup_str    = form.likely_lineup or "not confirmed"
            return (
                f"[Researcher] {form.name}: "
                f"recent={results_str} · "
                f"injuries={injuries_str} · suspended={sus_str} · "
                f"lineup={lineup_str} · "
                f"fitness=×{fit['fitness_degradation_factor']:.3f} "
                f"(rest {fit['rest_days']}d · travel {fit['travel_km']:.0f}km · "
                f"alt {fit['altitude_m']:.0f}m) · "
                f"squad={len(form.squad_clubs or {})} players"
            )

        return {
            "teams": {
                "A": {"form": a_form.model_dump(), "fitness": a_fit},
                "B": {"form": b_form.model_dump(), "fitness": b_fit},
            },
            "step_log": [
                _step_entry(a_form, a_fit, a_scn),
                _step_entry(b_form, b_fit, b_scn),
            ],
        }

    return researcher_node