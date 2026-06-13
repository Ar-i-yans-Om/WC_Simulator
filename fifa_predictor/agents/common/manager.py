"""
agents/common/manager.py
=========================

Manager (Orchestrator). In this design the heavy routing is expressed
declaratively in the graph topology (graph.py) and in the Chaos conditional
edge; the Manager node itself is the single entry point that validates the
frozen config payload and seeds the trace. Keeping routing in the graph (rather
than a giant supervisor switch) makes the independent team branches and the
chaos fallback edge visible in the compiled diagram.
"""

from __future__ import annotations
from ...schemas import AgentState # type: ignore


def manager_node(state: AgentState) -> dict:
    cfg = state.get("config", {})
    a = cfg.get("scenario", {}).get("team_a", {}).get("name", "Team A")
    b = cfg.get("scenario", {}).get("team_b", {}).get("name", "Team B")
    stage = cfg.get("scenario", {}).get("stage", "")
    return {
        "step_log": [f"[Manager] Orchestrating {a} vs {b} | {stage}. Dispatching Researcher."],
    }
