"""Tests for durable bounded ProtoGnosis tournament iterations."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import jnana.protognosis.periodic_tournament as periodic_tournament
from jnana.protognosis.utils.jnana_adapter import JnanaProtoGnosisAdapter

from jnana.protognosis.agents.specialized_agents import RankingAgent
from jnana.protognosis.core.agent_core import ContextMemory, ResearchHypothesis, SupervisorAgent, Task
from jnana.protognosis.core.llm_interface import OpenAILLM
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


class TuplePlanLLM:
    def generate_with_json_output(self, *_args, **_kwargs):
        return ({"main_objective": "Test", "domain": "math", "evaluation_criteria": ["correctness"]}, 5, 3)


def test_supervisor_unwraps_structured_response_tuple() -> None:
    plan = SupervisorAgent(TuplePlanLLM(), ContextMemory()).parse_research_goal("Test arithmetic.")

    assert plan["original_research_goal"] == "Test arithmetic."
    assert plan["domain"] == "math"


def test_openai_omits_temperature_when_adapter_requires_it() -> None:
    llm = OpenAILLM(api_key="test", model="test", model_adapter={"omit_temperature": True})
    create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))], usage=None))
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    assert llm.generate("Test.") == "ok"
    assert "temperature" not in create.call_args.kwargs


def test_incremental_tournament_reports_unfinished_matches(monkeypatch, tmp_path: Path) -> None:
    class IncompleteTournament:
        def __init__(self, **_kwargs):
            self.memory = SimpleNamespace(tournament_state={"matches": []}, save=lambda: None)

        def start(self):
            pass

        def stop(self):
            pass

        def get_all_hypotheses(self):
            return [{"hypothesis_id": "a"}, {"hypothesis_id": "b"}]

        def run_tournament(self, **_kwargs):
            pass

        def wait_for_completion(self):
            pass

        def get_statistics(self):
            return {}

        def get_top_hypotheses(self, _top_k):
            return []

    monkeypatch.setattr(periodic_tournament, "CoScientist", IncompleteTournament)

    result = periodic_tournament.run_incremental_tournament(
        state_file=tmp_path / "memory.json", llm_config=LLMConfig(provider="openai"), match_count=1
    )

    assert result["status"] == "incomplete"
    assert result["matches_completed"] == 0


def test_jnana_adapter_preserves_model_adapter() -> None:
    class ModelManager:
        def get_default_config(self):
            return {
                "provider": "openai",
                "model": "local-model",
                "model_adapter": {"omit_temperature": True},
            }

        def get_model_for_agent(self, _agent_type):
            return None

    config = JnanaProtoGnosisAdapter(ModelManager())._convert_model_config()

    assert config.default.model_adapter == {"omit_temperature": True}
