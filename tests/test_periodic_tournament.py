"""Tests for durable bounded ProtoGnosis tournament iterations."""

from __future__ import annotations

import asyncio
from pathlib import Path

from jnana.protognosis.agents.specialized_agents import RankingAgent
from jnana.protognosis.core.agent_core import ContextMemory, ResearchHypothesis, Task
from jnana.protognosis.core.multi_llm_config import LLMConfig
from jnana.protognosis.periodic_tournament import run_incremental_tournament


class FixedWinnerLLM:
    def generate_with_json_output(self, *_args, **_kwargs):
        return (
            {
                "criteria_comparison": [],
                "overall_winner": "A",
                "reasoning": "A is stronger.",
                "winner_key_advantages": ["testability"],
                "loser_key_weaknesses": ["specificity"],
            },
            0,
            0,
        )


def test_ranking_agent_persists_wins_losses_and_timestamp() -> None:
    memory = ContextMemory()
    first = ResearchHypothesis("A", "First", "agent", hypothesis_id="a")
    second = ResearchHypothesis("B", "Second", "agent", hypothesis_id="b")
    memory.add_hypothesis(first)
    memory.add_hypothesis(second)
    agent = RankingAgent("ranking-0", FixedWinnerLLM(), memory)

    result = asyncio.run(
        agent.execute_task(
            Task(
                task_type="tournament_match",
                agent_type="ranking",
                params={"hypothesis1_id": "a", "hypothesis2_id": "b"},
            )
        )
    )

    assert result["winner"] == "A"
    assert first.tournament_wins == 1
    assert second.tournament_losses == 1
    assert first.last_tournament_time is not None
    assert second.last_tournament_time is not None


def test_incremental_tournament_skips_when_fewer_than_two_hypotheses(tmp_path: Path) -> None:
    result = run_incremental_tournament(
        state_file=tmp_path / "memory.json",
        llm_config=LLMConfig(provider="openai", model="test", api_key="test"),
        match_count=1,
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "fewer_than_two_hypotheses"
