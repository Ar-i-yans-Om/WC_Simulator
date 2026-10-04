"""
main.py
=======

Command-line demo. Runs fixture K5 (Portugal vs Colombia, Group K, Matchday 3)
end-to-end and prints the executive report plus the full agent trace. For any
other fixture, use match_runner.py.

Usage
-----
    # Live run (needs GEMINI_API_KEY in the environment or a .env file)
    python -m fifa_predictor.main

    # Clean baseline (disable the weighted chaos contribution)
    python -m fifa_predictor.main --no-chaos

    # Print the compiled LangGraph Mermaid diagram and exit
    python -m fifa_predictor.main --print-graph
"""

from __future__ import annotations

import argparse
import sys

from .config import Settings, demo_scenario
from .graph import export_mermaid, run_prediction
from .logger import setup as setup_logging


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def _rule(title: str = "", width: int = 70) -> str:
    if not title:
        return "=" * width
    pad = width - len(title) - 2
    return f"== {title} " + "=" * max(0, pad)


def _print_report(final: dict) -> None:
    report = final.get("report", {})
    poisson = final.get("poisson", {})
    bookmaker = final.get("bookmaker", {})
    chaos = final.get("chaos", {})
    pitch = final.get("pitch", {})

    print("\n" + _rule("EXECUTIVE PREDICTION"))
    print(report.get("headline", "(no headline)"))
    print(f"Predicted scoreline : {report.get('predicted_scoreline', 'n/a')}")
    print(f"Confidence          : {report.get('confidence', 'n/a')}")

    print("\n" + _rule("WIN / DRAW / LOSS"))
    for k, v in report.get("win_draw_loss", {}).items():
        label = k.replace("_", " ")
        print(f"  {label:<22} {v:6.1%}")

    print("\n" + _rule("BASELINE EXPECTED GOALS (lambda breakdown)"))
    for slot in ("team_a", "team_b"):
        tl = pitch.get(slot, {})
        if tl:
            print(
                f"  {tl.get('name','?'):<12} base {tl.get('lambda_base',0):.2f} "
                f"x coh {tl.get('cohesion',1):.2f} x int "
                f"{tl.get('strategic_intensity',1):.2f} x fit "
                f"{tl.get('tactical_fit',1):.2f} x fit-deg "
                f"{tl.get('fitness_degradation',1):.2f} x mom "
                f"{tl.get('tournament_momentum',1):.2f} x dh "
                f"{tl.get('dark_horse',1):.2f} = "
                f"lambda {tl.get('lambda_final',0):.2f}"
            )

    print("\n" + _rule("MOST LIKELY SCORELINES"))
    for s in poisson.get("most_likely_scores", [])[:6]:
        print(f"  {s.get('home')}-{s.get('away')}   {s.get('prob',0):6.1%}")

    grid = poisson.get("grid", [])
    if grid:
        a_name = pitch.get("team_a", {}).get("name", "A")
        b_name = pitch.get("team_b", {}).get("name", "B")
        n = len(grid) - 1
        print("\n" + _rule(f"FULL SCORELINE GRID  (rows={a_name} 0-{n}, cols={b_name} 0-{n}, %)"))
        header = "      " + "".join(f"{j:>6}" for j in range(n + 1))
        print(header)
        for i, row in enumerate(grid):
            print(f"  {i:>3} " + "".join(f"{p*100:6.2f}" for p in row))

    print("\n" + _rule("MODEL vs MARKET"))
    print(f"  {report.get('model_vs_market','(n/a)')}")
    print(f"  market source: {bookmaker.get('source','n/a')}")

    print("\n" + _rule("CHAOS (weighted into the single run)"))
    print(f"  {chaos.get('summary','')}")
    print(f"  {report.get('chaos_impact','')}")

    print("\n" + _rule("NARRATIVE"))
    print("  " + report.get("narrative", "(none)"))


def _print_trace(final: dict) -> None:
    print("\n" + _rule("AGENT TRACE (step_log)"))
    for line in final.get("step_log", []):
        print(f"  {line}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_settings(args) -> Settings:
    settings = Settings.from_env()
    if args.no_chaos:
        settings.chaos_base_probability = 0.0
    return settings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="FIFA World Cup 2026 multi-agent match predictor")
    parser.add_argument("--no-chaos", action="store_true",
                        help="Disable the weighted chaos contribution (clean baseline).")
    parser.add_argument("--print-graph", action="store_true",
                        help="Print the compiled LangGraph Mermaid diagram and exit.")
    args = parser.parse_args(argv)

    if args.print_graph:
        print(export_mermaid(Settings()))
        return 0

    setup_logging()
    settings = build_settings(args)

    scenario = demo_scenario()
    a, b = scenario.team_a.name, scenario.team_b.name

    print(_rule(f"{a} vs {b}  |  {scenario.stage}"))
    print(f"{a}: ELO {scenario.team_a.elo}, {scenario.team_a.group_points} pts, "
          f"rest {scenario.team_a.rest_days}d, "
          f"{scenario.team_a.host_city_last_match} -> {scenario.team_a.host_city_this_match}")
    print(f"{b}: ELO {scenario.team_b.elo}, {scenario.team_b.group_points} pts, "
          f"rest {scenario.team_b.rest_days}d, "
          f"{scenario.team_b.host_city_last_match} -> {scenario.team_b.host_city_this_match}")
    print(f"chaos p={settings.chaos_base_probability:.2f}  |  grid 0-0..{settings.max_goals}-{settings.max_goals}")

    try:
        final = run_prediction(settings=settings, scenario=scenario)
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print("Hint: set GEMINI_API_KEY in your environment or a .env file.",
              file=sys.stderr)
        return 1

    _print_report(final)
    _print_trace(final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
