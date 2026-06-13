"""
graph.py
========

The LangGraph assembly: nodes, the independent (per-team) branches, and the
parallel fan-out / fan-in.

Topology
--------

    manager
       |
    researcher  (Global Pulse + Physiologist multiplier, BOTH teams)
       |
       +--------------------+--------------------+
       |                    |                    |
   alchemist_A          alchemist_B          bookmaker  (market anchor, parallel)
       |                    |                    |
   strategist_A         strategist_B            END
       |                    |
    scout_A              scout_B
       |                    |
  tactician_A          tactician_B
       \\                   /
        \\                 /
        pitch_simulator   (the ONLY node that sees both blind plans; BASELINE xG)
              |
         chaos_agent      (emits the weighted black-swan scenario set; no roll)
              |
            judge --> END (mixture of Poisson grids over all chaos scenarios)

Team A's branch has zero edges to Team B's branch: isolation is structural, not
just convention. The two 4-node chains are balanced, so `tactician_A` and
`tactician_B` always complete in the same LangGraph superstep - which makes the
fan-in at `pitch_simulator` fire exactly once. Chaos is no longer a branch: the
Chaos Agent always runs and the Judge folds every scenario into one run.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

from langgraph.graph import END, START, StateGraph

from .agents.common.bookmaker import make_bookmaker_node
from .agents.common.chaos_agent import make_chaos_node
from .agents.common.judge import make_judge_node
from .agents.common.manager import manager_node
from .agents.common.pitch_simulator import make_pitch_simulator_node
from .agents.common.researcher import make_researcher_node
from .agents.independent.alchemist import make_alchemist_node
from .agents.independent.scout import make_scout_node
from .agents.independent.strategist import make_strategist_node
from .agents.independent.tactician import make_tactician_node
from .config import MatchScenario, Settings, default_portugal_france_scenario
from .llm import build_llms
from .schemas import AgentState


def build_graph(settings: Settings, llms: Optional[dict] = None):
    """
    Construct and COMPILE the LangGraph workflow.

    Compilation needs no API key (it only validates the topology), so this is
    safe to call in offline/CI contexts. `llms` is injected for testing; in
    normal use it is built from `settings`.
    """
    if llms is None:
        llms = build_llms(settings)

    g = StateGraph(AgentState)

    # ----- Common nodes ----------------------------------------------------
    g.add_node("manager", manager_node)
    g.add_node("researcher", make_researcher_node(llms, settings))
    g.add_node("bookmaker", make_bookmaker_node(llms, settings))
    g.add_node("pitch_simulator", make_pitch_simulator_node(llms, settings))
    g.add_node("chaos_agent", make_chaos_node(llms, settings))
    g.add_node("judge", make_judge_node(llms, settings))

    # ----- Independent nodes: separate isolated instances per team ---------
    for team in ("A", "B"):
        suffix = team.lower()
        g.add_node(f"alchemist_{suffix}", make_alchemist_node(llms, settings, team))
        g.add_node(f"strategist_{suffix}", make_strategist_node(llms, settings, team))
        g.add_node(f"scout_{suffix}", make_scout_node(llms, settings, team))
        g.add_node(f"tactician_{suffix}", make_tactician_node(llms, settings, team))

    # ----- Spine -----------------------------------------------------------
    g.add_edge(START, "manager")
    g.add_edge("manager", "researcher")

    # ----- Fan-out: two isolated team branches + the market anchor ---------
    g.add_edge("researcher", "alchemist_a")
    g.add_edge("researcher", "alchemist_b")
    g.add_edge("researcher", "bookmaker")

    # Team A chain (blind to B)
    g.add_edge("alchemist_a", "strategist_a")
    g.add_edge("strategist_a", "scout_a")
    g.add_edge("scout_a", "tactician_a")

    # Team B chain (blind to A)
    g.add_edge("alchemist_b", "strategist_b")
    g.add_edge("strategist_b", "scout_b")
    g.add_edge("scout_b", "tactician_b")

    # The market anchor runs in parallel and terminates; the Judge reads its
    # result from shared state. Terminating at END avoids an early, duplicate
    # trigger of the Judge while keeping the node from being a dead-end.
    g.add_edge("bookmaker", END)

    # ----- Fan-in: both blind plans collide on the pitch -------------------
    g.add_edge("tactician_a", "pitch_simulator")
    g.add_edge("tactician_b", "pitch_simulator")

    # ----- Chaos -> Judge --------------------------------------------------
    # The Chaos Agent emits the weighted scenario set (no roll, no branch);
    # the Judge folds every scenario into a single mixture distribution.
    g.add_edge("pitch_simulator", "chaos_agent")
    g.add_edge("chaos_agent", "judge")

    # ----- Terminus --------------------------------------------------------
    g.add_edge("judge", END)

    return g.compile()


# ---------------------------------------------------------------------------
# Convenience: build the initial state and run an end-to-end prediction
# ---------------------------------------------------------------------------
def _scenario_to_config(scenario: MatchScenario, settings: Settings) -> dict:
    """Freeze the scenario + a settings snapshot into the immutable config."""
    return {
        "scenario": dataclasses.asdict(scenario),
        "settings": {
            "chaos_base_probability": settings.chaos_base_probability,
            "max_goals": settings.max_goals,
        },
    }


def initial_state(scenario: MatchScenario, settings: Settings) -> dict:
    """The seed payload handed to the compiled graph."""
    return {
        "config": _scenario_to_config(scenario, settings),
        "teams": {},          # reducer channel: seed empty so merges are clean
        "step_log": [],
    }


def run_prediction(
    settings: Optional[Settings] = None,
    scenario: Optional[MatchScenario] = None,
    llms: Optional[dict] = None,
) -> dict:
    """
    Build, compile and invoke the graph for one fixture.

    Returns the final AgentState dict (containing `report`, `poisson`,
    `bookmaker`, `pitch`, `chaos`, `step_log`).
    """
    settings = settings or Settings.from_env()
    scenario = scenario or default_portugal_france_scenario()
    # Only require a live API key when we are building the real LLM clients.
    # Injected `llms` (e.g. a test stub) bypass the key requirement.
    if llms is None:
        settings.validate()

    app = build_graph(settings, llms=llms)
    return app.invoke(initial_state(scenario, settings))


def export_mermaid(settings: Optional[Settings] = None, llms: Optional[dict] = None) -> str:
    """Return the compiled graph's Mermaid diagram (for docs / inspection)."""
    settings = settings or Settings()
    app = build_graph(settings, llms=llms)
    try:
        return app.get_graph().draw_mermaid()
    except Exception as exc:  # pragma: no cover
        return f"(mermaid export unavailable: {exc})"
