"""
fifa_predictor
==============

A high-fidelity, multi-agent FIFA World Cup match prediction system built on
LangGraph and the Anthropic API.

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

The numeric core is never mocked: base lambda, the Physiologist fitness factor
and the Poisson scoreline grid are always computed deterministically from real
inputs. The Anthropic API (with the server-side web_search tool) provides the
qualitative/research layer on top of that numeric core.
"""

from .graph import build_graph, run_prediction
from .config import Settings, load_settings, default_portugal_france_scenario

__all__ = [
    "build_graph",
    "run_prediction",
    "Settings",
    "load_settings",
    "default_portugal_france_scenario",
]
