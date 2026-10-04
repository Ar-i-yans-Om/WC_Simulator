"""
fifa_predictor
==============

A multi-agent FIFA World Cup 2026 match-prediction system built on LangGraph
and Google Gemini (the google-genai SDK, with Google Search grounding).

Design philosophy
-----------------
A football match is treated as the *collision* of two independent teams, each
with its own objectives, squad chemistry, game-theory context and tactical plan,
filtered through impartial global factors (data, environment, market, chaos).

The architecture therefore enforces a hard separation between:

* COMMON / SYMMETRIC nodes - impartial, global, see everything
    Manager, Researcher (+Physiologist), Pitch Simulator, Chaos Agent,
    Bookmaker, Judge.

* INDEPENDENT nodes - instantiated *per team*, fully isolated
    Alchemist, Strategist, Scout, Tactician.
  Team A instances have ZERO visibility into Team B's internal state.

The numeric core is never mocked: the ELO anchor, the Physiologist fitness
factor and the Poisson scoreline mixture are always computed deterministically
from real inputs. Gemini (with Google Search grounding for research) provides
the qualitative layer on top of that numeric core.
"""

from .graph import build_graph, run_prediction
from .config import Settings, demo_scenario

__all__ = [
    "build_graph",
    "run_prediction",
    "Settings",
    "demo_scenario",
]
