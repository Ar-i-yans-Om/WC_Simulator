"""
match_runner.py
===============

Central driver for the FIFA World Cup 2026 Prediction Engine.

Place at the PROJECT ROOT (same level as requirements.txt):

    fifa_predictor/
    ├── match_runner.py        ← run this
    ├── data/
    │   ├── fixtures.json      ← official match schedule (from FIFA/ESPN)
    │   ├── team_ratings.json  ← ELO ratings + FIFA rankings per team
    │   ├── results.json       ← fill in scores as matches are played
    │   └── predictions.json   ← WRITTEN by this runner; read by the dashboard
    ├── fifa_predictor/
    └── requirements.txt

HOW THE DATA PIPELINE WORKS
----------------------------
1. fixtures.json provides the schedule: who plays who, where, and when.
2. team_ratings.json provides ELO and FIFA rank per team (the strength signal).
3. results.json tracks match scores as the tournament progresses — the runner
   computes group standings from these to determine each team's points.
4. From the schedule the runner derives:
   - rest_days: date difference between a team's matches
   - host_city_last_match: the city of their previous fixture
   - is_host_nation: USA, Canada, Mexico
5. City name aliases are resolved (e.g. Inglewood → Los Angeles).
6. After each successful prediction the runner writes the result into
   predictions.json (keyed by fixture id). The Streamlit dashboard reads ONLY
   this file — it never calls the pipeline. Re-running a fixture overwrites just
   that entry; all other entries are left untouched.

The agents receive ONLY this lean context. Everything qualitative — form,
injuries, squad, tactics, weaknesses — is discovered by the Researcher via
Gemini's Google Search grounding at runtime.

USAGE
------
  python match_runner.py                              # all unplayed fixtures
  python match_runner.py --fixture K5                # single fixture by ID
  python match_runner.py --fixture K5 K6 L1          # several fixtures by ID
  python match_runner.py --group K                   # all fixtures in a group
  python match_runner.py --md 3                      # all matchday 3 fixtures
  python match_runner.py --no-chaos                  # disable weighted chaos
  python match_runner.py --all                       # including already-played
  python match_runner.py --hardcoded                 # use HARDCODED_FIXTURES below
  python match_runner.py --no-write                  # don't update predictions.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from fifa_predictor.config import MatchScenario, Settings, TeamScenario
from fifa_predictor.logger import setup as setup_logging
from fifa_predictor.data_sources import resolve_city
from fifa_predictor.graph import run_prediction

DATA_DIR         = _ROOT / "data"
FIXTURES_FILE    = DATA_DIR / "fixtures.json"
RATINGS_FILE     = DATA_DIR / "team_ratings.json"
RESULTS_FILE     = DATA_DIR / "results.json"
PREDICTIONS_FILE = DATA_DIR / "predictions.json"

HOST_NATIONS = {"United States", "USA", "Canada", "Mexico"}

# ---------------------------------------------------------------------------
# Knockout stage
# ---------------------------------------------------------------------------
# A fixture is a knockout tie iff it carries a "round" key (group matches never
# do). Knockout home/away are SLOT REFERENCES, resolved from results.json:
#   "1A"/"2B"/"3E" — 1st/2nd/3rd-placed team of that group (from group standings)
#   "W77"          — winner of match 77
#   "L101"         — loser of match 101
# resolve_bracket() turns these into concrete team names once the feeding
# results are known; unresolved slots are left as-is (that tie can't run yet).
ROUND_LABELS = {
    "R32": "Round of 32", "R16": "Round of 16", "QF": "Quarter-final",
    "SF": "Semi-final",   "3P": "Third-place play-off", "F": "Final",
}
# Pseudo-matchday per round (keeps a monotonic "how deep in the tournament"
# signal for the agents; group stage is 1-3, knockouts continue from 4).
ROUND_MATCHDAY = {"R32": 4, "R16": 5, "QF": 6, "SF": 7, "3P": 8, "F": 8}

_GROUP_SLOT_RE = re.compile(r"^([123])([A-L])$")
_MATCH_SLOT_RE = re.compile(r"^([WL])(\d+)$")


def is_knockout(fx: dict) -> bool:
    return bool(fx.get("round"))


def looks_like_slot(s: str) -> bool:
    """True if `s` is still an unresolved slot reference (not a real team name)."""
    return bool(_GROUP_SLOT_RE.match(s or "") or _MATCH_SLOT_RE.match(s or ""))


# ===========================================================================
# HARDCODED_FIXTURES
# Edit this list to run quick one-off tests without touching the JSON files.
# ===========================================================================
HARDCODED_FIXTURES = [
    {
        "id": "TEST_POR_FRA",
        "group": "TEST",
        "md": 3,
        "home": "Portugal",
        "away": "France",
        "date": "2026-06-27",
        "venue": "BC Place",
        "city": "Vancouver",
    },
]


# ===========================================================================
# Data loading
# ===========================================================================
def load_json(path: Path, label: str) -> dict | list:
    if not path.exists():
        print(f"[WARN] {path.name} not found — {label} will use defaults.")
        return {}
    with open(path) as f:
        return json.load(f)


def load_fixtures() -> list[dict]:
    data = load_json(FIXTURES_FILE, "schedule")
    if isinstance(data, dict):
        return data.get("matches", [])
    return [m for m in data if m.get("id")]


def load_ratings() -> dict:
    data = load_json(RATINGS_FILE, "ELO ratings")
    if isinstance(data, dict) and "teams" in data:
        return data["teams"]   # user's format has a "teams" key
    return data


def load_results() -> dict:
    data = load_json(RESULTS_FILE, "match results")
    if isinstance(data, dict):
        results_list = data.get("results", [])
    else:
        results_list = data
    return {r["id"]: r for r in results_list if r.get("id")}


def load_dark_horses() -> dict:
    """Optional {team_name: factor} map from team_ratings.json's top-level
    'dark_horses' key. A factor > 1.0 boosts an under-rated side, < 1.0 fades an
    over-rated one. Absent/blank → every team is neutral (1.0)."""
    data = load_json(RATINGS_FILE, "dark horses")
    raw = data.get("dark_horses", {}) if isinstance(data, dict) else {}
    # Keep only numeric factors (ignore helper keys like "_comment").
    out = {}
    for team, factor in (raw or {}).items():
        if isinstance(factor, (int, float)) and not isinstance(factor, bool):
            out[team] = float(factor)
    return out


# ===========================================================================
# Knockout bracket resolution
# ===========================================================================
def _group_place_map(fixtures: list[dict], results: dict) -> dict:
    """
    Compute the finishing position of every team within its group and return a
    slot map { "1A": <winner of A>, "2A": <runner-up>, "3A": <third>, ... }.

    Uses ONLY group-stage fixtures (those without a "round" key). Tie-breaking:
    points, then goal difference, then goals for — the same order the dashboard
    and FIFA regulations use for the first three criteria.
    """
    tables: dict[str, dict] = defaultdict(lambda: defaultdict(
        lambda: {"pts": 0, "gf": 0, "ga": 0}))
    for fx in fixtures:
        if is_knockout(fx):
            continue
        r = results.get(fx["id"])
        if not r or not r.get("played"):
            continue
        hs, as_ = r.get("home_score"), r.get("away_score")
        if hs is None or as_ is None:
            continue
        hs, as_ = int(hs), int(as_)
        g = fx.get("group", "")
        for name, gf, ga in [(fx["home"], hs, as_), (fx["away"], as_, hs)]:
            s = tables[g][name]
            s["gf"] += gf
            s["ga"] += ga
            s["pts"] += 3 if gf > ga else (1 if gf == ga else 0)

    slot_map: dict[str, str] = {}
    for g, teams in tables.items():
        ordered = sorted(
            teams.items(),
            key=lambda kv: (-kv[1]["pts"],
                            -(kv[1]["gf"] - kv[1]["ga"]),
                            -kv[1]["gf"]),
        )
        for i, (name, _) in enumerate(ordered):
            slot_map[f"{i + 1}{g}"] = name
    return slot_map


def _match_winner_loser(fx: dict, results: dict) -> tuple[Optional[str], Optional[str]]:
    """
    (winner, loser) team names for a resolved knockout fixture, or (None, None)
    if not decided yet. Draws (knockouts can't end level) are resolved via an
    explicit "winner" field on the result (the shootout victor); without it a
    level score stays unresolved.
    """
    home, away = fx.get("home"), fx.get("away")
    if looks_like_slot(home) or looks_like_slot(away):
        return None, None
    r = results.get(fx["id"])
    if not r or not r.get("played"):
        return None, None
    hs, as_ = r.get("home_score"), r.get("away_score")
    if hs is None or as_ is None:
        return None, None
    hs, as_ = int(hs), int(as_)
    if hs > as_:
        return home, away
    if as_ > hs:
        return away, home
    # Level after normal/extra time → shootout winner must be named explicitly.
    w = r.get("winner")
    if w == home:
        return home, away
    if w == away:
        return away, home
    return None, None


def resolve_bracket(fixtures: list[dict], results: dict) -> list[dict]:
    """
    Return a copy of `fixtures` where every knockout fixture's slot references
    (home/away) are replaced by concrete team names wherever they can be derived
    from results.json. Group slots (1A/2B/3E) resolve from final group standings;
    match slots (W77/L101) resolve once the feeding tie has a decided result.

    Fixtures are processed in match-number order so a round's winners are known
    before the next round that consumes them. Slots that cannot be resolved yet
    are left untouched (that tie simply isn't runnable until its feeders finish).
    """
    slot_map = _group_place_map(fixtures, results)
    resolved = [dict(fx) for fx in fixtures]
    by_id = {fx["id"]: fx for fx in resolved}

    ko = sorted((fx for fx in resolved if is_knockout(fx)),
                key=lambda f: f.get("match_no", 0))
    for fx in ko:
        for side in ("home", "away"):
            ref = fx[side]
            if ref in slot_map:
                fx[side] = slot_map[ref]
        # Once both sides are concrete and the tie has a result, publish its
        # winner/loser so later rounds (W##/L##) can resolve off it.
        w, l = _match_winner_loser(fx, results)
        n = fx.get("match_no")
        if w:
            slot_map[f"W{n}"] = w
        if l:
            slot_map[f"L{n}"] = l
    return resolved


# ===========================================================================
# Group standings computation
# ===========================================================================
def compute_standings(fixtures: list[dict], results: dict) -> dict:
    """
    Compute group standings from played results.

    Returns: { team_name: {"points": int, "gf": int, "ga": int, "gd": int,
                           "played": int, "won": int, "drawn": int, "lost": int} }
    """
    standings = defaultdict(lambda: {"points": 0, "gf": 0, "ga": 0,
                                     "gd": 0, "played": 0, "won": 0,
                                     "drawn": 0, "lost": 0, "group": ""})

    for fx in fixtures:
        r = results.get(fx["id"])
        if not r or not r.get("played"):
            continue
        home, away = fx["home"], fx["away"]
        hs, as_ = r.get("home_score"), r.get("away_score")
        if hs is None or as_ is None:
            continue
        hs, as_ = int(hs), int(as_)
        group = fx.get("group", "")

        for name, scored, conceded in [(home, hs, as_), (away, as_, hs)]:
            s = standings[name]
            s["group"]  = group
            s["played"] += 1
            s["gf"]     += scored
            s["ga"]     += conceded
            s["gd"]      = s["gf"] - s["ga"]
            if scored > conceded:
                s["points"] += 3
                s["won"]    += 1
            elif scored == conceded:
                s["points"] += 1
                s["drawn"]  += 1
            else:
                s["lost"]   += 1

    return dict(standings)


def points_before(team: str, fixture_id: str, fixtures: list[dict],
                  results: dict) -> int:
    """Points accumulated by `team` in matches played BEFORE `fixture_id`."""
    target_fx = next((f for f in fixtures if f["id"] == fixture_id), None)
    if not target_fx:
        return 0
    target_date = target_fx.get("date", "9999-12-31")
    total = 0
    for fx in fixtures:
        if is_knockout(fx):          # group points ignore knockout results
            continue
        if fx.get("date", "9999-12-31") >= target_date:
            continue
        if team not in (fx["home"], fx["away"]):
            continue
        r = results.get(fx["id"])
        if not r or not r.get("played"):
            continue
        hs, as_ = r.get("home_score"), r.get("away_score")
        if hs is None or as_ is None:
            continue
        hs, as_ = int(hs), int(as_)
        is_home = (fx["home"] == team)
        scored    = hs if is_home else as_
        conceded  = as_ if is_home else hs
        if scored > conceded:
            total += 3
        elif scored == conceded:
            total += 1
    return total


# ===========================================================================
# Tournament momentum (intra-tournament form from ACTUAL results.json)
# ===========================================================================
# Per-match momentum contribution from the goal margin, capped at ±0.09 so a
# single rout can't dominate. Averaged across the team's played WC matches and
# clamped to ±10% — a deliberately gentle nudge on top of the (static) ELO,
# since ELO already encodes baseline strength and is not re-derived mid-event.
_MOMENTUM_PER_GD       = 0.03
_MOMENTUM_PER_MATCH_CAP = 0.09
_MOMENTUM_FLOOR, _MOMENTUM_CEIL = 0.90, 1.10


def tournament_form(team: str, fixture_id: str, fixtures: list[dict],
                    results: dict) -> dict:
    """
    The team's ACTUAL World Cup matches played BEFORE `fixture_id`, read straight
    from results.json (never predictions). Returns structured per-match detail, a
    deterministic momentum multiplier, and a human-readable summary. Empty/neutral
    for MD1 (no prior matches) or when nothing is marked played.
    """
    empty = {"played": 0, "results": [], "gf": 0, "ga": 0, "gd": 0,
             "momentum_multiplier": 1.0, "summary": ""}
    target = next((f for f in fixtures if f["id"] == fixture_id), None)
    if not target:
        return empty
    target_date = target.get("date", "9999-12-31")

    played: list[dict] = []
    for fx in sorted(fixtures, key=lambda f: f.get("date", "")):
        if fx["id"] == fixture_id or team not in (fx["home"], fx["away"]):
            continue
        if fx.get("date", "9999-12-31") >= target_date:
            continue
        r = results.get(fx["id"])
        if not r or not r.get("played"):
            continue
        hs, as_ = r.get("home_score"), r.get("away_score")
        if hs is None or as_ is None:
            continue
        hs, as_ = int(hs), int(as_)
        is_home = (fx["home"] == team)
        gf = hs if is_home else as_
        ga = as_ if is_home else hs
        opp = fx["away"] if is_home else fx["home"]
        res = "W" if gf > ga else ("D" if gf == ga else "L")
        # Label group matches "MD#"; knockout ties by their round (R32/R16/...).
        stage_lbl = fx["round"] if is_knockout(fx) else f"MD{fx.get('md')}"
        played.append({"md": fx.get("md"), "stage": stage_lbl, "opponent": opp,
                       "gf": gf, "ga": ga, "result": res})

    if not played:
        return empty

    deltas = [max(-_MOMENTUM_PER_MATCH_CAP,
                  min(_MOMENTUM_PER_MATCH_CAP, _MOMENTUM_PER_GD * (m["gf"] - m["ga"])))
              for m in played]
    momentum = round(max(_MOMENTUM_FLOOR,
                         min(_MOMENTUM_CEIL, 1.0 + sum(deltas) / len(deltas))), 4)
    gf = sum(m["gf"] for m in played)
    ga = sum(m["ga"] for m in played)
    summary = "; ".join(
        f"{m.get('stage', 'MD' + str(m['md']))}: {m['result']} {m['gf']}-{m['ga']} vs {m['opponent']}"
        for m in played
    )
    return {"played": len(played), "results": played, "gf": gf, "ga": ga,
            "gd": gf - ga, "momentum_multiplier": momentum, "summary": summary}


# ===========================================================================
# Travel / rest derivation
# ===========================================================================
def previous_match(team: str, fixture_id: str, fixtures: list[dict]) -> Optional[dict]:
    """Return the fixture this team played immediately before `fixture_id`."""
    target = next((f for f in fixtures if f["id"] == fixture_id), None)
    if not target:
        return None
    target_date = target.get("date", "")

    previous = None
    for fx in fixtures:
        if fx["id"] == fixture_id:
            continue
        if team not in (fx["home"], fx["away"]):
            continue
        if fx.get("date", "") < target_date:
            if previous is None or fx["date"] > previous["date"]:
                previous = fx
    return previous


def compute_rest_days(team: str, fixture: dict, fixtures: list[dict]) -> int:
    prev = previous_match(team, fixture["id"], fixtures)
    if not prev or not fixture.get("date") or not prev.get("date"):
        return 4   # default: assume normal rest for MD1
    try:
        d1 = datetime.strptime(prev["date"],    "%Y-%m-%d").date()
        d2 = datetime.strptime(fixture["date"], "%Y-%m-%d").date()
        return max(1, (d2 - d1).days)
    except Exception:
        return 4


def city_of_previous(team: str, fixture: dict, fixtures: list[dict]) -> str:
    prev = previous_match(team, fixture["id"], fixtures)
    if not prev:
        return resolve_city(fixture.get("city", ""))   # MD1: no travel
    return resolve_city(prev.get("city", ""))


# ===========================================================================
# Scenario builder
# ===========================================================================
def build_scenario(fixture: dict, ratings: dict, fixtures: list[dict],
                   results: dict, dark_horses: dict | None = None) -> MatchScenario:
    home_name = fixture["home"]
    away_name = fixture["away"]
    date_str  = fixture.get("date",  "2026-06-11")
    city_now  = resolve_city(fixture.get("city", ""))
    dark_horses = dark_horses or {}

    knockout  = is_knockout(fixture)
    rnd       = fixture.get("round", "")
    if knockout:
        group    = rnd                          # e.g. "R16" (shown in briefings)
        matchday = ROUND_MATCHDAY.get(rnd, 4)    # pseudo-MD continuing from group stage
        stage    = ROUND_LABELS.get(rnd, "Knockout")
        # Single-elimination directive for the Strategist (no rotation maths here).
        ko_notes = (f"{stage}: single-elimination knockout — win or the tournament "
                    "is over. There is no group table and no rotation calculus; "
                    "every match is must-win.")
    else:
        group    = fixture.get("group", "?")
        matchday = fixture.get("md", 1)
        stage    = f"Group {group} — Matchday {matchday}"
        ko_notes = ""

    def _build_team(name: str) -> TeamScenario:
        r       = ratings.get(name, {})
        # Group points are meaningless once the knockouts start.
        pts     = 0 if knockout else points_before(name, fixture["id"], fixtures, results)
        rest    = compute_rest_days(name, fixture, fixtures)
        prev_c  = city_of_previous(name, fixture, fixtures)
        tform   = tournament_form(name, fixture["id"], fixtures, results)

        return TeamScenario(
            name                 = name,
            matchday             = matchday,
            group                = group,
            date                 = date_str,
            group_points         = pts,
            host_city_this_match = city_now,
            host_city_last_match = prev_c,
            rest_days            = rest,
            elo                  = float(r.get("elo", 1500.0)),
            fifa_rank            = int(r.get("fifa_rank", 30)),
            confederation        = r.get("confederation", ""),
            is_host_nation       = name in HOST_NATIONS,
            chaos_profile        = r.get("chaos_profile", {}),
            tournament_momentum     = tform["momentum_multiplier"],
            tournament_form_summary = tform["summary"],
            tournament_results      = tform["results"],
            dark_horse_factor       = float(dark_horses.get(name, 1.0)),
            notes                   = ko_notes,
        )

    return MatchScenario(
        team_a             = _build_team(home_name),
        team_b             = _build_team(away_name),
        competition        = "FIFA World Cup 2026",
        stage              = stage,
        tournament_avg_goals = 1.35,
    )


# ===========================================================================
# Output formatting
# ===========================================================================
def _rule(title: str = "", w: int = 68) -> str:
    if not title:
        return "=" * w
    return f"== {title} " + "=" * max(0, w - len(title) - 4)


def print_result(fixture: dict, scenario: MatchScenario, final: dict) -> None:
    report  = final.get("report", {})
    poisson = final.get("poisson", {})
    pitch   = final.get("pitch", {})
    chaos   = final.get("chaos", {})
    a, b    = scenario.team_a.name, scenario.team_b.name

    fid = fixture["id"]
    md = scenario.team_a.matchday
    print(f"\n{_rule(f'{a} vs {b}  [MD{md} · {fid}]')}")
    print(f"  {report.get('headline', '(no headline)')}")
    print(f"  Scoreline : {report.get('predicted_scoreline', 'n/a')}")
    print(f"  Confidence: {report.get('confidence', 'n/a')}")
    print()
    print(f"  {a:<18} win : {poisson.get('home_win', 0):.1%}")
    print(f"  Draw              :  {poisson.get('draw', 0):.1%}")
    print(f"  {b:<18} win : {poisson.get('away_win', 0):.1%}")
    print()
    for slot, team_name in (("team_a", a), ("team_b", b)):
        tl = pitch.get(slot, {})
        if tl:
            print(f"  xG {team_name:<16}: anchor {tl.get('lambda_base',0):.2f} → "
                  f"baseline {tl.get('lambda_final',0):.2f}")
    cm = chaos.get("summary", "")
    if cm:
        print(f"\n  ⚡ Chaos (weighted in): {cm}")
    grid = poisson.get("grid", [])
    if grid:
        n = len(grid) - 1
        print(f"\n  Scoreline grid (rows {a} 0-{n}, cols {b} 0-{n}, %):")
        print("       " + "".join(f"{j:>6}" for j in range(n + 1)))
        for i, row in enumerate(grid):
            print(f"   {i:>3} " + "".join(f"{p*100:6.2f}" for p in row))
    print(f"\n  Market: {report.get('model_vs_market','')}")
    print(f"  Narrative: {report.get('narrative','')}")

    q = final.get("quality") or {}
    if q:
        if q.get("ok"):
            print("  Quality: ✓ clean — all calls optimal, research grounded")
        else:
            issues = ", ".join(f"{e['call']}:{e['issue']}" for e in q.get("issues", [])[:6])
            tag = "⚠ DEGRADED" + ("" if q.get("grounded") else " · NOT grounded")
            print(f"  Quality: {tag} ({q.get('n_issues',0)} issue(s)) — {issues}")


# ===========================================================================
# predictions.json writer  (consumed by the Streamlit dashboard)
# ===========================================================================
def _pct(x) -> Optional[int]:
    """Convert a 0–1 probability (or already-0–100 value) to an int percentage."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if x <= 1.0:           # fraction → percent
        x *= 100.0
    return int(round(x))


def _top_scorelines_from_grid(grid: list[list[float]], k: int = 3) -> list[dict]:
    """
    Pull the k most-likely scorelines out of the Poisson grid.
    grid[i][j] = P(home i, away j). Returns [{"score": "i-j", "prob": pct}, ...].
    """
    if not grid:
        return []
    cells = []
    for i, row in enumerate(grid):
        for j, p in enumerate(row):
            cells.append((p, i, j))
    cells.sort(reverse=True)
    out = []
    for p, i, j in cells[:k]:
        pct = _pct(p)
        if pct is not None:
            out.append({"score": f"{i}-{j}", "prob": pct})
    return out


def _best_outcome_from_grid(grid, home, away, threshold=0.05):
    """
    Highest-scoring plausible scoreline with prob > threshold (5%), regardless of
    which team it favours. Among cells above the floor, picks the most emphatic
    result: maximise the winner's goals, then the loser's goals, then probability.
    Returns {"team": <winner name or "Draw">, "score": "winner-loser", "prob": pct}
    or None.
    """
    if not grid:
        return None
    best = None
    for i, row in enumerate(grid):
        for j, p in enumerate(row):
            if p is None or p <= threshold:
                continue
            hi, lo = (i, j) if i >= j else (j, i)
            key = (hi, lo, p)
            if best is None or key > best[0]:
                best = (key, i, j, p)
    if best is None:
        return None
    _, i, j, p = best
    if i > j:
        team, score = home, f"{i}-{j}"
    elif j > i:
        team, score = away, f"{j}-{i}"
    else:
        team, score = "Draw", f"{i}-{j}"
    return {"team": team, "score": score, "prob": _pct(p)}


def build_prediction_entry(fixture: dict, scenario: MatchScenario,
                           final: dict) -> dict:
    """
    Map a pipeline result (the `final` dict from run_prediction) onto the
    predictions.json contract the dashboard reads. Tolerates missing fields —
    anything absent is written as null/empty so the UI shows it as pending.
    """
    report  = final.get("report", {})    or {}
    poisson = final.get("poisson", {})   or {}
    pitch   = final.get("pitch", {})     or {}
    book    = final.get("bookmaker", {}) or {}
    grid    = poisson.get("grid", [])    or []

    eg_home = (pitch.get("team_a") or {}).get("lambda_final")
    eg_away = (pitch.get("team_b") or {}).get("lambda_final")

    return {
        "fixture_id": fixture["id"],
        "group":      fixture.get("round") or fixture.get("group"),
        "md":         fixture.get("match_no") or fixture.get("md"),
        "round":      fixture.get("round"),
        "match_no":   fixture.get("match_no"),
        "home":       scenario.team_a.name,
        "away":       scenario.team_b.name,
        # ---- pipeline output ----
        "prob_home_win":       _pct(poisson.get("home_win")),
        "prob_draw":           _pct(poisson.get("draw")),
        "prob_away_win":       _pct(poisson.get("away_win")),
        "expected_goals":      {
            "home": round(float(eg_home), 2) if eg_home is not None else None,
            "away": round(float(eg_away), 2) if eg_away is not None else None,
        },
        "predicted_scoreline": report.get("predicted_scoreline"),
        "top_scorelines":      _top_scorelines_from_grid(grid),
        "top_outcome":         _best_outcome_from_grid(
            grid, scenario.team_a.name, scenario.team_b.name),
        "scoreline_grid":      [[round(p, 5) for p in row] for row in grid],
        # ---- extra context (the dashboard ignores unknown keys safely) ----
        "confidence":          report.get("confidence"),
        "headline":            report.get("headline"),
        "narrative":           report.get("narrative"),
        "model_vs_market":     report.get("model_vs_market"),
        # structured market anchor (lets the dashboard draw an exact model-vs-market bar)
        "market_probs":        {
            "implied_home_win": book.get("implied_home_win"),
            "implied_draw":     book.get("implied_draw"),
            "implied_away_win": book.get("implied_away_win"),
        } if book else None,
        # Run quality: ok=True means every LLM call took the optimal route;
        # grounded=True means live web data backed the research. Used by
        # --rerun-degraded to find fixtures worth re-running.
        "quality":             final.get("quality"),
        "updated_at":          datetime.now().isoformat(timespec="seconds"),
    }


def _load_predictions_file() -> dict:
    """Read existing predictions.json (keyed by fixture id). Empty dict if absent/bad."""
    if not PREDICTIONS_FILE.exists():
        return {}
    try:
        with open(PREDICTIONS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data.get("predictions", data)
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def write_prediction(fixture: dict, scenario: MatchScenario, final: dict) -> None:
    """
    Merge one fixture's prediction into predictions.json. Only the matching
    fixture id is overwritten; every other entry is preserved. This is what
    makes matchday-by-matchday runs accumulate correctly.
    """
    entry = build_prediction_entry(fixture, scenario, final)
    store = _load_predictions_file()
    store[fixture["id"]] = entry

    DATA_DIR.mkdir(exist_ok=True)
    tmp = PREDICTIONS_FILE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)
    tmp.replace(PREDICTIONS_FILE)   # atomic on the same filesystem
    print(f"  ✎ predictions.json updated for {fixture['id']}")


# ===========================================================================
# Main runner
# ===========================================================================
def run_fixture(fixture: dict, ratings: dict, fixtures: list[dict],
                results: dict, settings: Settings, verbose: bool,
                write: bool = True, dark_horses: dict | None = None) -> Optional[dict]:
    a, b = fixture["home"], fixture["away"]
    fid  = fixture["id"]
    stage_lbl = fixture["round"] if is_knockout(fixture) else f"MD{fixture.get('md', 1)}"
    print(f"\n  Running: {a} vs {b}  [{fid}]  ({stage_lbl}, {fixture.get('date','')})")
    try:
        scenario = build_scenario(fixture, ratings, fixtures, results, dark_horses)
        final    = run_prediction(settings=settings, scenario=scenario)
        print_result(fixture, scenario, final)
        print("\n  ── Agent trace ──")
        for ln in final.get("step_log", []):
            if ln:
                print(f"    {ln}")
        if write:
            try:
                write_prediction(fixture, scenario, final)
            except Exception as wexc:
                print(f"  [WARN] could not write prediction for {fid}: {wexc}")
        return final
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="FIFA WC 2026 Prediction Engine — reads data/ files"
    )
    p.add_argument("--fixture",    metavar="ID", nargs="+",
                   help="One or more fixture IDs (e.g. K5  or  K5 K6 L1)")
    p.add_argument("--group",      metavar="GRP", help="All fixtures in a group (e.g. K)")
    p.add_argument("--md",         type=int,       help="All fixtures on a matchday (1/2/3)")
    p.add_argument("--round",      metavar="RND",
                   help="All fixtures in a knockout round (R32/R16/QF/SF/3P/F)")
    p.add_argument("--all",        action="store_true", help="Include already-played fixtures")
    p.add_argument("--rerun-degraded", action="store_true",
                   help="Re-run only fixtures whose existing prediction was DEGRADED "
                        "(quality.ok == false in predictions.json)")
    p.add_argument("--hardcoded",  action="store_true", help="Use HARDCODED_FIXTURES in this file")
    p.add_argument("--no-chaos",   action="store_true", help="Disable the weighted chaos contribution")
    p.add_argument("--no-write",   action="store_true", help="Don't update predictions.json")
    p.add_argument("--verbose",    action="store_true", help="Print full agent trace")
    p.add_argument("--delay",      type=float, default=5.0,
                   help="Seconds between fixtures in a batch (default 5)")
    args = p.parse_args(argv)

    setup_logging(verbose=args.verbose)
    settings = Settings.from_env()
    if args.no_chaos:
        settings.chaos_base_probability = 0.0

    write_predictions = not args.no_write
    dark_horses = load_dark_horses()
    if dark_horses:
        print(f"Dark-horse factors active for {len(dark_horses)} team(s): "
              + ", ".join(f"{k}×{v}" for k, v in dark_horses.items()))

    if args.hardcoded:
        fixtures = HARDCODED_FIXTURES
        ratings  = load_ratings()
        results  = {}
        write_predictions = False   # test fixtures never touch predictions.json
        print(f"Using {len(fixtures)} hardcoded fixture(s).")
    else:
        fixtures_all = load_fixtures()
        ratings      = load_ratings()
        results      = load_results()
        # Turn knockout slot references (1A / W77 / L101) into concrete teams
        # wherever the feeding results are already known.
        fixtures_all = resolve_bracket(fixtures_all, results)
        print(f"Loaded {len(fixtures_all)} fixtures · "
              f"{sum(1 for r in results.values() if r.get('played'))} played.")

        # Filter
        if args.rerun_degraded:
            store = _load_predictions_file()
            degraded = {fid for fid, e in store.items()
                        if isinstance(e, dict) and (e.get("quality") or {}).get("ok") is False}
            fixtures = [f for f in fixtures_all if f["id"] in degraded]
            print(f"Re-running {len(fixtures)} degraded fixture(s): "
                  + (", ".join(sorted(degraded)) if degraded else "none"))
        elif args.fixture:
            wanted = set(args.fixture)
            fixtures = [f for f in fixtures_all if f["id"] in wanted]
        elif args.group:
            fixtures = [f for f in fixtures_all
                        if f.get("group","").upper() == args.group.upper()]
        elif args.md:
            fixtures = [f for f in fixtures_all if f.get("md") == args.md]
        elif args.round:
            fixtures = [f for f in fixtures_all
                        if f.get("round", "").upper() == args.round.upper()]
        else:
            fixtures = fixtures_all

        if not args.all and not args.rerun_degraded:
            # Skip matches already played
            fixtures = [f for f in fixtures
                        if not results.get(f["id"], {}).get("played")]

        # A knockout tie whose feeders haven't finished still has slot refs for
        # home/away — it can't run yet. Drop it with a note (unless the user
        # asked for a specific id, in which case surface why it's stuck).
        runnable = []
        for f in fixtures:
            unresolved = looks_like_slot(f.get("home", "")) or looks_like_slot(f.get("away", ""))
            if unresolved:
                print(f"  ⏳ {f['id']} not runnable yet — awaiting "
                      f"{f.get('home')} vs {f.get('away')} (feeders undecided).")
            else:
                runnable.append(f)
        fixtures = runnable

    if not fixtures:
        print("No matching fixtures found.")
        return 1

    settings.validate()

    succeeded = 0
    for i, fx in enumerate(fixtures):
        if i > 0:
            time.sleep(args.delay)
        r = run_fixture(fx, ratings, fixtures_all if not args.hardcoded else fixtures,
                        results, settings, args.verbose, write=write_predictions,
                        dark_horses=dark_horses)
        if r:
            succeeded += 1

    print(f"\n{_rule()}")
    print(f"Completed {succeeded}/{len(fixtures)} fixtures.")
    if write_predictions:
        print(f"Predictions written to {PREDICTIONS_FILE.relative_to(_ROOT)}")
    return 0 if succeeded == len(fixtures) else 1


if __name__ == "__main__":
    raise SystemExit(main())