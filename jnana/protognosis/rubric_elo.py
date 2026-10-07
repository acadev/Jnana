"""Domain-general rubric-derived Elo for bidirectional hypothesis fixtures."""
from __future__ import annotations

from math import isclose
from typing import Any, Iterable, Mapping


def normalize_rubric(rubric: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate and normalize criterion weights to sum to one."""
    rows = []
    for item in rubric:
        row = dict(item)
        criterion_id = str(row.get("id", "")).strip()
        label = str(row.get("label", criterion_id)).strip()
        weight = float(row.get("weight", 0.0))
        if not criterion_id or not label or weight <= 0:
            raise ValueError("Each rubric criterion requires id, label, and positive weight")
        if any(existing["id"] == criterion_id for existing in rows):
            raise ValueError(f"Duplicate rubric criterion id: {criterion_id}")
        rows.append({**row, "id": criterion_id, "label": label, "weight": weight})
    if not rows:
        raise ValueError("Rubric requires at least one criterion")
    total = sum(row["weight"] for row in rows)
    for row in rows:
        row["weight"] /= total
    return rows


def semantic_choice(match: Mapping[str, Any], label: Any) -> str | None:
    """Map rendered A/B choice to the stable hypothesis identity."""
    value = str(label).strip().upper()
    if value == "A":
        return str(match["hypothesis1_id"])
    if value == "B":
        return str(match["hypothesis2_id"])
    return None


def criterion_choices(match: Mapping[str, Any]) -> dict[str, str]:
    """Return criterion label -> rendered winner for a persisted native match."""
    return {
        str(item["criterion"]): str(item.get("winner", "tie"))
        for item in match.get("criteria_comparison", [])
    }


def score_bidirectional_fixture(
    ab: Mapping[str, Any], ba: Mapping[str, Any], rubric: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Convert two orientations into a weighted fractional score for AB hypothesis 1."""
    rubric = normalize_rubric(rubric)
    if (ab["hypothesis1_id"], ab["hypothesis2_id"]) != (
        ba["hypothesis2_id"], ba["hypothesis1_id"]
    ):
        raise ValueError("Matches are not opposite orientations of one fixture")
    first_id, second_id = str(ab["hypothesis1_id"]), str(ab["hypothesis2_id"])
    ab_choices, ba_choices = criterion_choices(ab), criterion_choices(ba)
    results, score = [], 0.0
    for row in rubric:
        label = row["label"]
        winner_ab = semantic_choice(ab, ab_choices.get(label, "tie"))
        winner_ba = semantic_choice(ba, ba_choices.get(label, "tie"))
        resolved = winner_ab is not None and winner_ab == winner_ba
        value = 1.0 if resolved and winner_ab == first_id else 0.0 if resolved else 0.5
        score += row["weight"] * value
        results.append({
            "criterion_id": row["id"], "criterion": label, "weight": row["weight"],
            "winner_ab": winner_ab, "winner_ba": winner_ba,
            "consensus_winner": winner_ab if resolved else None,
            "score_first": value, "resolution": "resolved" if resolved else "neutralized",
        })
    return {
        "hypothesis_ids": [first_id, second_id], "score_first": score,
        "score_second": 1.0 - score, "criteria": results,
        "resolved_weight": sum(x["weight"] for x in results if x["resolution"] == "resolved"),
    }


def expected_score(first_rating: float, second_rating: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((second_rating - first_rating) / 400.0))


def update_fractional_elo(
    ratings: dict[str, float], first_id: str, second_id: str,
    score_first: float, k_factor: float = 32.0,
) -> dict[str, float]:
    """Apply a zero-sum Elo update using a fractional rubric score."""
    if not 0.0 <= score_first <= 1.0:
        raise ValueError("score_first must lie in [0, 1]")
    before = ratings[first_id] + ratings[second_id]
    expected = expected_score(ratings[first_id], ratings[second_id])
    delta = k_factor * (score_first - expected)
    ratings[first_id] += delta
    ratings[second_id] -= delta
    if not isclose(before, ratings[first_id] + ratings[second_id], abs_tol=1e-9):
        raise RuntimeError("Elo update was not zero-sum")
    return {"expected_first": expected, "delta_first": delta, "delta_second": -delta}
