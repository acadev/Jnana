"""Bounded, durable ProtoGnosis tournament iterations.

This module is intentionally a library API: callers own scheduling and model
configuration.  It loads one workflow-scoped ``ContextMemory`` JSON file, runs a
small number of pairwise ranking tasks serially, persists the result, and
returns a compact snapshot suitable for another system to render.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import random

from .core.coscientist import CoScientist
from .core.multi_llm_config import LLMConfig
from .agents.laya_ranking_agent import LayaRankingAgent


def _compact_hypothesis(hypothesis: dict[str, Any]) -> dict[str, Any]:
    """Return ranking data without exposing full hypothesis bodies."""

    matches = hypothesis.get("tournament_matches") or 0
    return {
        "hypothesis_id": str(hypothesis.get("hypothesis_id") or ""),
        "summary": " ".join(str(hypothesis.get("summary") or "").split())[:240],
        "elo_rating": round(float(hypothesis.get("elo_rating") or 0.0), 1),
        "tournament_matches": len(matches) if isinstance(matches, list) else int(matches),
    }


def run_incremental_tournament(
    *,
    state_file: str | Path,
    llm_config: LLMConfig,
    match_count: int = 1,
    top_k: int = 3,
) -> dict[str, Any]:
    """Run and persist one bounded tournament iteration.

    The caller must serialize access to ``state_file``.  One worker is used so
    ranking tasks and the terminal rankings update execute in a deterministic
    order against the JSON-backed context memory.
    """

    if match_count < 0:
        raise ValueError("match_count must be non-negative")
    if top_k < 1:
        raise ValueError("top_k must be at least one")

    state_path = Path(state_file).expanduser()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    coscientist = CoScientist(
        llm_config=llm_config,
        storage_path=str(state_path),
        max_workers=1,
    )
    coscientist.start()
    try:
        hypothesis_count = len(coscientist.get_all_hypotheses())
        if hypothesis_count < 2:
            return {
                "status": "skipped",
                "reason": "fewer_than_two_hypotheses",
                "hypothesis_count": hypothesis_count,
                "matches_requested": match_count,
                "matches_completed": 0,
                "top_hypotheses": [],
            }

        before_matches = len(coscientist.memory.tournament_state.get("matches", []))
        if match_count:
            coscientist.run_tournament(match_count=match_count)
            coscientist.wait_for_completion()
        statistics = coscientist.get_statistics()
        coscientist.memory.save()
        after_matches = len(coscientist.memory.tournament_state.get("matches", []))
        matches_completed = after_matches - before_matches
        status = "completed" if matches_completed == match_count else "incomplete"
        result = {
            "status": status,
            "hypothesis_count": hypothesis_count,
            "matches_requested": match_count,
            "matches_completed": matches_completed,
            "top_hypotheses": [
                _compact_hypothesis(hypothesis)
                for hypothesis in coscientist.get_top_hypotheses(top_k)
            ],
            "statistics": statistics,
        }
        if status == "incomplete":
            result["reason"] = "scheduled tournament matches did not complete"
        return result
    finally:
        coscientist.stop()


def run_batched_laya_tournament(
    *, state_file: str | Path, llm_config: LLMConfig, match_count: int = 1,
    top_k: int = 3, batch_size: int | None = None,
) -> dict[str, Any]:
    """Batch Laya inference, then apply Elo/state changes deterministically."""
    if match_count < 0:
        raise ValueError("match_count must be non-negative")
    if top_k < 1:
        raise ValueError("top_k must be at least one")
    state_path = Path(state_file).expanduser()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    coscientist = CoScientist(llm_config=llm_config, storage_path=str(state_path), max_workers=1)
    hypotheses = coscientist.get_all_hypotheses()
    if len(hypotheses) < 2:
        return {"status": "skipped", "reason": "fewer_than_two_hypotheses",
                "hypothesis_count": len(hypotheses), "matches_requested": match_count,
                "matches_completed": 0, "top_hypotheses": []}
    ranking_ids = coscientist.supervisor.agent_types.get("ranking", [])
    agent = coscientist.supervisor.agents[ranking_ids[0]] if ranking_ids else None
    if not isinstance(agent, LayaRankingAgent):
        raise RuntimeError("The registered ranking agent is not LayaRankingAgent")
    pairs = [tuple(random.sample(hypotheses, 2)) for _ in range(match_count)]
    before = len(coscientist.memory.tournament_state.get("matches", []))
    if pairs:
        agent.judge_batch(pairs, batch_size=batch_size)
    after = len(coscientist.memory.tournament_state.get("matches", []))
    rankings = sorted(hypotheses, key=lambda h: h.elo_rating, reverse=True)
    coscientist.memory.tournament_state["rankings"] = [
        {"rank": index + 1, "hypothesis_id": h.hypothesis_id,
         "elo_rating": h.elo_rating, "summary": h.summary}
        for index, h in enumerate(rankings)
    ]
    coscientist.memory.save()
    completed = after - before
    return {
        "status": "completed" if completed == match_count else "incomplete",
        "hypothesis_count": len(hypotheses), "matches_requested": match_count,
        "matches_completed": completed, "inference_mode": "laya.predict_batch",
        "batch_size": batch_size,
        "top_hypotheses": [_compact_hypothesis(h.to_dict()) for h in rankings[:top_k]],
    }
