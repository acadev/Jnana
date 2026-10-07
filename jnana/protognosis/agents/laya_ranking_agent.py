"""Laya-backed single-model and ensemble tournament judges for ProtoGnosis."""

from __future__ import annotations

import asyncio
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Mapping

from .specialized_agents import RankingAgent


class LayaRankingAgent(RankingAgent):
    """Judge tournament matches with one or more auditable Laya routers.

    ``router=...`` preserves the original single-model API. ``routers=[...]`` accepts
    router objects or mappings with ``name``, ``router``, optional positive
    ``weight``, and optional JSON-serializable ``provenance``.
    """

    def __init__(self, agent_id, llm, memory, router=None, routers=None,
                 aggregation_policy="majority_confidence_tiebreak"):
        super().__init__(agent_id, llm, memory)
        if router is not None and routers is not None:
            raise ValueError("Pass either router or routers, not both")
        self.router = router
        self.routers = self._normalize_router_specs(routers) if routers is not None else None
        self.aggregation_policy = aggregation_policy

    @staticmethod
    def _normalize_router_specs(routers):
        specs = []
        for index, value in enumerate(routers):
            spec = dict(value) if isinstance(value, Mapping) else {"router": value}
            if "router" not in spec:
                raise ValueError("Each router mapping requires a 'router' value")
            spec.setdefault("name", f"laya-{index}")
            spec.setdefault("weight", 1.0)
            spec.setdefault("provenance", {})
            if not str(spec["name"]).strip() or float(spec["weight"]) <= 0:
                raise ValueError("Router names must be nonempty and weights positive")
            if any(item["name"] == spec["name"] for item in specs):
                raise ValueError(f"Duplicate router name: {spec['name']}")
            specs.append(spec)
        if not specs:
            raise ValueError("routers must contain at least one model")
        return specs

    def configure_routers(self, routers, aggregation_policy=None):
        """Replace the active model set without replacing the native Jnana agent."""
        self.routers = self._normalize_router_specs(routers)
        self.router = None
        if aggregation_policy is not None:
            self.aggregation_policy = aggregation_policy
        return self

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

    def _router_specs(self):
        if self.routers is not None:
            return self.routers
        return [{"name": "laya-0", "router": self._get_router(), "weight": 1.0,
                 "provenance": {}}]

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
        questions = {}
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
        state, questions = self._state(hypothesis1, hypothesis2, research_goal), self._questions(criteria)
        specs = self._router_specs()
        raw = []
        for spec in specs:
            raw.append(await asyncio.to_thread(
                spec["router"].predict, state, questions, model="multilingual", max_len=8192
            ))
        normalized = [self._normalize_result(result, criteria) for result in raw]
        return self._aggregate_results(normalized, specs), 0, 0

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
        return {
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
            "confidence": overall.get("answer_confidence"),
        }

    def _aggregate_results(self, normalized, specs):
        if len(normalized) == 1:
            response = dict(normalized[0])
            response["aggregation"] = {
                "policy": "single_model", "resolution": "single_model",
                "vote_counts": {response["overall_winner"]: 1}, "model_count": 1,
            }
        elif self.aggregation_policy == "unanimous":
            labels = [item["overall_winner"] for item in normalized]
            winner = labels[0] if len(set(labels)) == 1 else "tie"
            response = self._ensemble_response(normalized, specs, winner, "unanimous")
        elif self.aggregation_policy == "majority_confidence_tiebreak":
            labels = [item["overall_winner"] for item in normalized]
            counts = Counter(labels)
            highest = max(counts.values())
            leaders = [label for label, count in counts.items() if count == highest]
            if len(leaders) == 1:
                winner, resolution = leaders[0], "majority_vote"
            else:
                support = defaultdict(float)
                for item, spec in zip(normalized, specs):
                    support[item["overall_winner"]] += float(spec["weight"]) * float(
                        item.get("confidence") or 0.0
                    )
                best = max(support[label] for label in leaders)
                tied = [label for label in leaders if support[label] == best]
                winner = tied[0] if len(tied) == 1 else "tie"
                resolution = "confidence_weighted_tiebreak" if len(tied) == 1 else "unresolved_tie"
            response = self._ensemble_response(normalized, specs, winner, resolution)
        else:
            raise ValueError(f"Unknown Laya aggregation policy: {self.aggregation_policy}")
        response["model_judgments"] = [
            {
                "name": spec["name"], "weight": float(spec["weight"]),
                "provenance": spec["provenance"],
                "overall_winner": item["overall_winner"],
                "confidence": item.get("confidence"),
                "criteria_comparison": item["criteria_comparison"],
            }
            for item, spec in zip(normalized, specs)
        ]
        return response

    def _ensemble_response(self, normalized, specs, winner, resolution):
        vote_counts = Counter(item["overall_winner"] for item in normalized)
        confidence_support = defaultdict(float)
        for item, spec in zip(normalized, specs):
            confidence_support[item["overall_winner"]] += float(spec["weight"]) * float(
                item.get("confidence") or 0.0
            )
        winning_confidences = [float(item["confidence"]) for item in normalized
                               if item["overall_winner"] == winner and item.get("confidence") is not None]
        comparisons = []
        for index in range(len(normalized[0]["criteria_comparison"])):
            votes = Counter(item["criteria_comparison"][index]["winner"] for item in normalized)
            highest = max(votes.values())
            leaders = [label for label, count in votes.items() if count == highest]
            comparisons.append({
                "criterion": normalized[0]["criteria_comparison"][index]["criterion"],
                "hypothesis_a_strengths": "", "hypothesis_b_strengths": "",
                "winner": leaders[0] if len(leaders) == 1 else "tie", "confidence": None,
            })
        return {
            "criteria_comparison": comparisons, "overall_winner": winner,
            "reasoning": (
                f"{len(normalized)} Laya models were aggregated with "
                f"{self.aggregation_policy}; resolution={resolution}."
            ),
            "winner_key_advantages": [x["criterion"] for x in comparisons
                                      if x["winner"] == winner] if winner in {"A", "B"} else [],
            "loser_key_weaknesses": [], "decision_engine": "laya_ensemble",
            "confidence": sum(winning_confidences) / len(winning_confidences)
                          if winning_confidences else None,
            "aggregation": {
                "policy": self.aggregation_policy, "resolution": resolution,
                "vote_counts": dict(vote_counts),
                "confidence_support": dict(confidence_support),
                "model_count": len(normalized),
            },
        }

    def judge_batch(self, pairs, *, batch_size=None):
        """Batch every model, aggregate, then apply native Elo updates serially."""
        criteria = self.memory.metadata.get("research_plan_config", {}).get(
            "evaluation_criteria", ["novelty", "plausibility", "testability"]
        )
        goal = self.memory.metadata.get("research_goal", "")
        questions = self._questions(criteria)
        requests = [{"state": self._state(a, b, goal), "questions": questions,
                     "model": "multilingual", "max_len": 8192} for a, b in pairs]
        specs, by_model = self._router_specs(), []
        for spec in specs:
            results = spec["router"].predict_batch(
                requests, batch_size=batch_size, sort_by_length=True
            )
            if len(results) != len(pairs):
                raise RuntimeError(
                    f"Laya model {spec['name']} returned {len(results)}/{len(pairs)} judgments"
                )
            by_model.append([self._normalize_result(result, criteria) for result in results])
        storage_path = self.memory.storage_path
        self.memory.storage_path = None
        try:
            records = [
                self._apply_result(a, b, self._aggregate_results(
                    [model_results[index] for model_results in by_model], specs
                ))
                for index, (a, b) in enumerate(pairs)
            ]
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
            "decision_engine": response.get("decision_engine", "laya"),
            "confidence": response.get("confidence"), "inference_mode": "predict_batch",
        }
        if "aggregation" in response:
            match["aggregation"] = response["aggregation"]
        if "model_judgments" in response:
            match["model_judgments"] = response["model_judgments"]
        timestamp = time.time()
        first.add_tournament_match(match)
        second.add_tournament_match(match)
        first.last_tournament_time = timestamp
        second.last_tournament_time = timestamp
        if winner is first:
            first.tournament_wins += 1
            second.tournament_losses += 1
        elif winner is second:
            second.tournament_wins += 1
            first.tournament_losses += 1
        self.memory.update_hypothesis(first)
        self.memory.update_hypothesis(second)
        self.memory.tournament_state["matches"].append(match)
        return match
