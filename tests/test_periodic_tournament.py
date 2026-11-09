"""Tests for durable bounded ProtoGnosis tournament iterations."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import jnana.protognosis.periodic_tournament as periodic_tournament
from jnana.protognosis.utils.jnana_adapter import JnanaProtoGnosisAdapter

from jnana.protognosis.agents.specialized_agents import RankingAgent
from jnana.protognosis.core.agent_core import ContextMemory, ResearchHypothesis, SupervisorAgent, Task
from jnana.protognosis.core.coscientist import CoScientist
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


# --- Swiss / deep-Swiss scheme -------------------------------------------------
#
# _swiss_pairings and _build_pairings do not touch `self`, so they are exercised
# as unbound methods (self=None) without constructing a full CoScientist.


def _stub(hid: str, elo: float) -> SimpleNamespace:
    return SimpleNamespace(hypothesis_id=hid, elo_rating=elo)


def test_swiss_pairings_sort_by_elo_and_pair_neighbours() -> None:
    hyps = [_stub("a", 1200), _stub("b", 1400), _stub("c", 1000), _stub("d", 1300)]

    pairs = CoScientist._swiss_pairings(None, hyps)

    ids = [(h1.hypothesis_id, h2.hypothesis_id) for h1, h2 in pairs]
    assert ids == [("b", "d"), ("a", "c")]  # 1400-1300 play, 1200-1000 play


def test_swiss_pairings_give_odd_field_a_bye() -> None:
    hyps = [_stub("a", 1200), _stub("b", 1400), _stub("c", 1000)]

    pairs = CoScientist._swiss_pairings(None, hyps)

    paired = {h.hypothesis_id for pair in pairs for h in pair}
    assert len(pairs) == 1
    assert "c" not in paired  # lowest-rated sits this round out


def test_build_pairings_counts_and_rejects_unknown_scheme() -> None:
    # _build_pairings delegates to self._swiss_pairings, so borrow both methods
    # onto a bare object rather than passing self=None.
    class _Pairer:
        _swiss_pairings = CoScientist._swiss_pairings
        _build_pairings = CoScientist._build_pairings

    pairer = _Pairer()
    hyps = [_stub("a", 1200), _stub("b", 1400), _stub("c", 1000)]

    assert len(pairer._build_pairings(hyps, "random", 5)) == 5
    # match_count is ignored by swiss: one round == floor(n / 2) pairs.
    assert len(pairer._build_pairings(hyps, "swiss", 99)) == 1
    with pytest.raises(ValueError):
        pairer._build_pairings(hyps, "bogus", 1)


class _FakeSwissCoScientist:
    """Records one Swiss round of matches per run_tournament call."""

    created: list["_FakeSwissCoScientist"] = []

    def __init__(self, **kwargs):
        self.memory = SimpleNamespace(
            tournament_state={"matches": []}, save=lambda: None)
        self.hyps = [{"hypothesis_id": c} for c in "abcd"]
        self.scheme_calls: list[str] = []
        self.max_workers = kwargs.get("max_workers")
        _FakeSwissCoScientist.created.append(self)

    def start(self): pass
    def stop(self): pass
    def get_statistics(self): return {}
    def get_all_hypotheses(self): return self.hyps
    def get_top_hypotheses(self, top_k): return self.hyps[:top_k]

    def run_tournament(self, match_count=10, hypothesis_ids=None, scheme="random"):
        self.scheme_calls.append(scheme)
        pairs = len(self.hyps) // 2  # one Swiss round
        self.memory.tournament_state["matches"].extend([{}] * pairs)
        return {"scheme": scheme, "match_count": pairs,
                "hypotheses_count": len(self.hyps)}


def test_incremental_tournament_swiss_runs_requested_rounds(monkeypatch, tmp_path: Path) -> None:
    _FakeSwissCoScientist.created.clear()
    monkeypatch.setattr(periodic_tournament, "CoScientist", _FakeSwissCoScientist)

    result = run_incremental_tournament(
        state_file=tmp_path / "memory.json",
        llm_config=LLMConfig(provider="openai"),
        scheme="swiss",
        rounds=3,
        top_k=2,
    )

    instance = _FakeSwissCoScientist.created[0]
    assert instance.scheme_calls == ["swiss", "swiss", "swiss"]  # one call per round
    assert result["status"] == "completed"
    assert result["scheme"] == "swiss"
    assert result["rounds_requested"] == 3
    assert result["matches_completed"] == 6  # 3 rounds x 2 pairs


def test_incremental_tournament_swiss_passes_workers_to_pool(monkeypatch, tmp_path: Path) -> None:
    _FakeSwissCoScientist.created.clear()
    monkeypatch.setattr(periodic_tournament, "CoScientist", _FakeSwissCoScientist)

    run_incremental_tournament(
        state_file=tmp_path / "memory.json",
        llm_config=LLMConfig(provider="openai"),
        scheme="swiss",
        rounds=2,
        workers=16,
    )

    assert _FakeSwissCoScientist.created[0].max_workers == 16  # parallel round


def test_incremental_tournament_rejects_parallel_random(tmp_path: Path) -> None:
    # Random pairs can overlap, so parallel matches would race on Elo.
    with pytest.raises(ValueError):
        run_incremental_tournament(
            state_file=tmp_path / "memory.json",
            llm_config=LLMConfig(provider="openai"),
            scheme="random",
            workers=8,
        )


def test_incremental_tournament_rejects_non_positive_workers(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_incremental_tournament(
            state_file=tmp_path / "memory.json",
            llm_config=LLMConfig(provider="openai"),
            scheme="swiss",
            workers=0,
        )


def test_incremental_tournament_rejects_unknown_scheme(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_incremental_tournament(
            state_file=tmp_path / "memory.json",
            llm_config=LLMConfig(provider="openai"),
            scheme="bogus",
        )


def test_incremental_tournament_swiss_requires_positive_rounds(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_incremental_tournament(
            state_file=tmp_path / "memory.json",
            llm_config=LLMConfig(provider="openai"),
            scheme="swiss",
            rounds=0,
        )


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
