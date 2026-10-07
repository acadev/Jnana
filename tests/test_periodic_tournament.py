"""Tests for durable bounded ProtoGnosis tournament iterations."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import jnana.protognosis.periodic_tournament as periodic_tournament
from jnana.protognosis.utils.jnana_adapter import JnanaProtoGnosisAdapter

from jnana.protognosis.agents.specialized_agents import RankingAgent
from jnana.protognosis.agents.laya_ranking_agent import LayaRankingAgent
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


class FixedLayaRouter:
    def __init__(self):
        self.calls = []

    def predict(self, state, questions, **kwargs):
        self.calls.append((state, questions, kwargs))
        return self._result()

    def predict_batch(self, requests, **kwargs):
        self.batch_call = (requests, kwargs)
        return [self._result() for _ in requests]

    @staticmethod
    def _result():
        return {
            "answers": {
                "criterion_0": {"choice": "B", "answer_confidence": 0.81},
                "criterion_1": {"choice": "A", "answer_confidence": 0.62},
                "overall_winner": {"choice": "B", "answer_confidence": 0.77},
            }
        }


def test_laya_agent_determines_and_records_tournament_winner() -> None:
    memory = ContextMemory()
    memory.metadata["research_goal"] = "Find the most testable explanation."
    memory.metadata["research_plan_config"] = {
        "evaluation_criteria": ["novelty", "testability"]
    }
    first = ResearchHypothesis("A", "First", "agent", hypothesis_id="a")
    second = ResearchHypothesis("B", "Second", "agent", hypothesis_id="b")
    memory.add_hypothesis(first)
    memory.add_hypothesis(second)
    router = FixedLayaRouter()
    agent = LayaRankingAgent("laya-0", FixedWinnerLLM(), memory, router=router)

    result = asyncio.run(
        agent.execute_task(
            Task(
                task_type="tournament_match",
                agent_type="ranking",
                params={"hypothesis1_id": "a", "hypothesis2_id": "b"},
            )
        )
    )

    assert result["winner"] == "B"
    assert first.tournament_losses == 1
    assert second.tournament_wins == 1
    match = memory.tournament_state["matches"][0]
    assert match["decision_engine"] == "laya"
    assert match["confidence"] == 0.77
    assert router.calls[0][2] == {"model": "multilingual", "max_len": 8192}
    assert "Hypothesis A:\nA" in router.calls[0][0]


def test_laya_batch_applies_ordered_updates_and_persists_once(tmp_path: Path) -> None:
    state = tmp_path / "memory.json"
    memory = ContextMemory(str(state))
    memory.metadata["research_plan_config"] = {
        "evaluation_criteria": ["novelty", "testability"]
    }
    first = ResearchHypothesis("A", "First", "agent", hypothesis_id="a")
    second = ResearchHypothesis("B", "Second", "agent", hypothesis_id="b")
    memory.add_hypothesis(first)
    memory.add_hypothesis(second)
    router = FixedLayaRouter()
    agent = LayaRankingAgent("laya-0", FixedWinnerLLM(), memory, router=router)

    matches = agent.judge_batch([(first, second), (first, second)], batch_size=2)

    assert len(matches) == 2
    assert all(match["inference_mode"] == "predict_batch" for match in matches)
    assert first.tournament_losses == 2
    assert second.tournament_wins == 2
    assert router.batch_call[1] == {"batch_size": 2, "sort_by_length": True}
    persisted = json.loads(state.read_text())
    assert len(persisted["tournament_state"]["matches"]) == 2


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


class ChoiceLayaRouter:
    def __init__(self, choice, confidence):
        self.choice = choice
        self.confidence = confidence
        self.batch_calls = 0

    def predict_batch(self, requests, **_kwargs):
        self.batch_calls += 1
        return [{
            "answers": {
                "criterion_0": {"choice": self.choice, "answer_confidence": self.confidence},
                "overall_winner": {"choice": self.choice, "answer_confidence": self.confidence},
            }
        } for _ in requests]

    def predict(self, _state, _questions, **_kwargs):
        return self.predict_batch([None])[0]


def ensemble_fixture(tmp_path, routers, policy="majority_confidence_tiebreak"):
    state = tmp_path / "ensemble.json"
    memory = ContextMemory(str(state))
    memory.metadata["research_plan_config"] = {"evaluation_criteria": ["testability"]}
    first = ResearchHypothesis("A", "First", "agent", hypothesis_id="a")
    second = ResearchHypothesis("B", "Second", "agent", hypothesis_id="b")
    memory.add_hypothesis(first)
    memory.add_hypothesis(second)
    agent = LayaRankingAgent(
        "laya-ensemble", FixedWinnerLLM(), memory, routers=routers,
        aggregation_policy=policy,
    )
    return state, memory, first, second, agent


def test_laya_ensemble_majority_and_provenance_are_persisted(tmp_path: Path) -> None:
    models = [
        {"name": "calibrated", "router": ChoiceLayaRouter("A", .60),
         "provenance": {"checkpoint": "step-1500"}},
        {"name": "tournament", "router": ChoiceLayaRouter("B", .90),
         "provenance": {"checkpoint": "step-2200"}},
        {"name": "balanced", "router": ChoiceLayaRouter("B", .70),
         "provenance": {"checkpoint": "step-2400"}},
    ]
    state, _memory, first, second, agent = ensemble_fixture(tmp_path, models)
    match = agent.judge_batch([(first, second)], batch_size=4)[0]

    assert match["overall_winner"] == "B"
    assert match["decision_engine"] == "laya_ensemble"
    assert match["aggregation"]["resolution"] == "majority_vote"
    assert match["aggregation"]["vote_counts"] == {"A": 1, "B": 2}
    assert [item["name"] for item in match["model_judgments"]] == [
        "calibrated", "tournament", "balanced"
    ]
    assert match["model_judgments"][1]["provenance"]["checkpoint"] == "step-2200"
    assert all(item["router"].batch_calls == 1 for item in models)
    assert first.tournament_losses == 1 and second.tournament_wins == 1
    persisted = json.loads(state.read_text())
    assert persisted["tournament_state"]["matches"][0]["model_judgments"] == match["model_judgments"]


def test_laya_ensemble_confidence_breaks_equal_vote_count(tmp_path: Path) -> None:
    models = [
        {"name": "a", "router": ChoiceLayaRouter("A", .55), "weight": 1.0},
        {"name": "b", "router": ChoiceLayaRouter("B", .80), "weight": 1.0},
    ]
    _state, _memory, first, second, agent = ensemble_fixture(tmp_path, models)
    match = agent.judge_batch([(first, second)])[0]
    assert match["overall_winner"] == "B"
    assert match["aggregation"]["resolution"] == "confidence_weighted_tiebreak"


def test_laya_ensemble_exact_tie_does_not_change_elo(tmp_path: Path) -> None:
    models = [
        {"name": "a", "router": ChoiceLayaRouter("A", .75)},
        {"name": "b", "router": ChoiceLayaRouter("B", .75)},
    ]
    _state, _memory, first, second, agent = ensemble_fixture(tmp_path, models)
    match = agent.judge_batch([(first, second)])[0]
    assert match["overall_winner"] == "tie"
    assert match["aggregation"]["resolution"] == "unresolved_tie"
    assert first.elo_rating == second.elo_rating == 1200
    assert first.tournament_wins == second.tournament_wins == 0


def test_laya_ensemble_unanimous_policy_abstains_on_disagreement(tmp_path: Path) -> None:
    models = [ChoiceLayaRouter("A", .99), ChoiceLayaRouter("B", .51)]
    _state, _memory, first, second, agent = ensemble_fixture(
        tmp_path, models, policy="unanimous"
    )
    match = agent.judge_batch([(first, second)])[0]
    assert match["overall_winner"] == "tie"
    assert match["aggregation"]["policy"] == "unanimous"


def test_coscientist_configures_and_persists_laya_ensemble(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("jnana.protognosis.core.coscientist.create_llm", lambda *_a, **_k: FixedWinnerLLM())
    from jnana.protognosis.core.coscientist import CoScientist

    state = tmp_path / "coscientist.json"
    coscientist = CoScientist(
        llm_config=LLMConfig(provider="openai", model="unused", api_key="unused"),
        storage_path=str(state), max_workers=1,
    )
    agent = coscientist.configure_laya_judges([
        {"name": "one", "router": ChoiceLayaRouter("A", .7),
         "provenance": {"checkpoint": "step-1500"}},
        {"name": "two", "router": ChoiceLayaRouter("B", .8),
         "provenance": {"checkpoint": "step-2200"}},
    ], aggregation_policy="unanimous")

    assert isinstance(agent, LayaRankingAgent)
    persisted = json.loads(state.read_text())
    config = persisted["metadata"]["laya_judging"]
    assert config["aggregation_policy"] == "unanimous"
    assert [item["name"] for item in config["models"]] == ["one", "two"]
    assert "router" not in config["models"][0]
