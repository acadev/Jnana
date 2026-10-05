"""Laya-backed tournament judge for ProtoGnosis."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Iterable

from .specialized_agents import RankingAgent


class LayaRankingAgent(RankingAgent):
    """Determine tournament winners with Laya's calibrated decision engine.

    Laya is loaded lazily on the first match, so constructing a ``CoScientist``
    does not download a checkpoint. A router may be injected for testing or for
    applications that already manage a shared Laya router.
    """

    def __init__(self, agent_id, llm, memory, router=None):
        super().__init__(agent_id, llm, memory)
        self.router = router

    def _get_router(self):
        if self.router is None:
            try:
                from laya import Router
            except ImportError as exc:
                raise RuntimeError(
                    "Laya is required to judge tournament matches; install it with "
                    "`python -m pip install laya`."
                ) from exc
            self.router = Router(max_loaded=1)
        return self.router

    @staticmethod
    def _state(hypothesis1, hypothesis2, research_goal: str) -> str:
        return (
            f"Research goal:\n{research_goal}\n\n"
            f"Hypothesis A:\n{hypothesis1.content}\n\n"
            f"Hypothesis B:\n{hypothesis2.content}"
        )

    @staticmethod
    def _questions(criteria: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        options = {
            "A": "Hypothesis A is stronger",
            "B": "Hypothesis B is stronger",
            "tie": "Neither hypothesis is meaningfully stronger",
        }
        questions: Dict[str, Dict[str, Any]] = {}
        for index, criterion in enumerate(criteria):
            questions[f"criterion_{index}"] = {
                "type": "choice",
                "instructions": f"Which hypothesis is stronger on {criterion}?",
                "criteria": options,
            }
        questions["overall_winner"] = {
            "type": "choice",
            "instructions": (
                "Which hypothesis better addresses the research goal overall, "
                "considering scientific validity, novelty, testability, impact, "
                "clarity, and the listed evaluation criteria?"
            ),
            "criteria": options,
        }
        return questions

    @staticmethod
    def _winner_label(value: Any) -> str:
        label = str(value).strip().upper()
        if label in {"A", "HYPOTHESIS A"}:
            return "A"
        if label in {"B", "HYPOTHESIS B"}:
            return "B"
        return "tie"

    async def _judge_match(self, hypothesis1, hypothesis2, research_goal, criteria,
                           prompt, schema, system_prompt):
        del prompt, schema, system_prompt
        result = await asyncio.to_thread(
            self._get_router().predict,
            self._state(hypothesis1, hypothesis2, research_goal),
            self._questions(criteria),
            model="multilingual",
            max_len=8192,
        )
        return self._normalize_result(result, criteria), 0, 0

    def _normalize_result(self, result, criteria):
        answers = result.get("answers", {})
        comparisons = []
        for index, criterion in enumerate(criteria):
            answer = answers.get(f"criterion_{index}", {})
            comparisons.append({
                "criterion": criterion,
                "hypothesis_a_strengths": "",
                "hypothesis_b_strengths": "",
                "winner": self._winner_label(answer.get("choice")),
                "confidence": answer.get("answer_confidence"),
            })
        overall = answers.get("overall_winner", {})
        winner = self._winner_label(overall.get("choice"))
        confidence = overall.get("answer_confidence")
        response = {
            "criteria_comparison": comparisons,
            "overall_winner": winner,
            "reasoning": (
                "Laya selected the winner using a calibrated, non-autoregressive "
                f"decision over {len(comparisons)} evaluation criteria."
            ),
            "winner_key_advantages": [
                item["criterion"] for item in comparisons if item["winner"] == winner
            ] if winner in {"A", "B"} else [],
            "loser_key_weaknesses": [],
            "decision_engine": "laya",
            "confidence": confidence,
        }
        return response

    def judge_batch(self, pairs, *, batch_size=None):
        """Batch inference, then apply Elo updates serially in pair order."""
        criteria = self.memory.metadata.get("research_plan_config", {}).get(
            "evaluation_criteria", ["novelty", "plausibility", "testability"]
        )
        goal = self.memory.metadata.get("research_goal", "")
        questions = self._questions(criteria)
        requests = [{"state": self._state(a, b, goal), "questions": questions,
                     "model": "multilingual", "max_len": 8192} for a, b in pairs]
        results = self._get_router().predict_batch(
            requests, batch_size=batch_size, sort_by_length=True
        )
        if len(results) != len(pairs):
            raise RuntimeError(f"Laya returned {len(results)}/{len(pairs)} judgments")
        storage_path = self.memory.storage_path
        self.memory.storage_path = None
        try:
            records = [self._apply_result(a, b, self._normalize_result(result, criteria))
                       for (a, b), result in zip(pairs, results)]
        finally:
            self.memory.storage_path = storage_path
        self.memory.save()
        return records

    def _apply_result(self, first, second, response):
        winner_label = self._winner_label(response.get("overall_winner"))
        winner = loser = None
        if winner_label == "A":
            winner, loser = first, second
        elif winner_label == "B":
            winner, loser = second, first
        if winner is not None and loser is not None:
            winner_expected = self._calculate_expected_score(winner.elo_rating, loser.elo_rating)
            loser_expected = self._calculate_expected_score(loser.elo_rating, winner.elo_rating)
            winner.elo_rating += self.k_factor * (1 - winner_expected)
            loser.elo_rating += self.k_factor * (0 - loser_expected)
        match = {
            "match_id": str(time.time()), "hypothesis1_id": first.hypothesis_id,
            "hypothesis2_id": second.hypothesis_id,
            "criteria_comparison": response["criteria_comparison"],
            "overall_winner": winner_label, "reasoning": response["reasoning"],
            "winner_key_advantages": response["winner_key_advantages"],
            "loser_key_weaknesses": response["loser_key_weaknesses"],
            "decision_engine": "laya", "confidence": response.get("confidence"),
            "inference_mode": "predict_batch",
        }
        timestamp = time.time()
        first.add_tournament_match(match); second.add_tournament_match(match)
        first.last_tournament_time = timestamp; second.last_tournament_time = timestamp
        if winner is first:
            first.tournament_wins += 1; second.tournament_losses += 1
        elif winner is second:
            second.tournament_wins += 1; first.tournament_losses += 1
        self.memory.update_hypothesis(first); self.memory.update_hypothesis(second)
        self.memory.tournament_state["matches"].append(match)
        return match
