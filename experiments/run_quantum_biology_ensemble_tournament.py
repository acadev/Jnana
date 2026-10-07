#!/usr/bin/env python3
"""Run a native Jnana three-checkpoint tournament and derive consensus Elo."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jnana.protognosis.agents.laya_checkpoint_router import checkpoint_router_specs
from jnana.protognosis.core.agent_core import ResearchHypothesis
from jnana.protognosis.core.coscientist import CoScientist
from jnana.protognosis.core.multi_llm_config import LLMConfig

CRITERIA = [
    "physical plausibility at biological temperature and solvent conditions",
    "evidence that a genuinely quantum state persists rather than only quantum chemistry occurring",
    "causal connection between the quantum state and a physiological outcome",
    "specificity and feasibility of discriminating experiments",
    "robustness against classical kinetic, structural, and statistical alternatives",
    "generality across realistic biological boundary conditions",
]


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def expected_score(a: float, b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((b - a) / 400.0))


def update_elo(ratings, winner, loser, k=32.0):
    ew = expected_score(ratings[winner], ratings[loser])
    el = expected_score(ratings[loser], ratings[winner])
    ratings[winner] += k * (1.0 - ew)
    ratings[loser] += k * (0.0 - el)


def semantic_winner(match):
    label = match["overall_winner"]
    if label == "A":
        return match["hypothesis1_id"]
    if label == "B":
        return match["hypothesis2_id"]
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--archive", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--policy", choices=["majority_confidence_tiebreak", "unanimous"],
                   default="majority_confidence_tiebreak")
    a = p.parse_args()

    source = json.loads(Path(a.source).read_text())
    hypotheses = source["hypotheses"]
    if len(hypotheses) != 50 or len({x["hypothesis_id"] for x in hypotheses}) != 50:
        raise RuntimeError("Expected exactly 50 uniquely identified hypotheses")
    archive = Path(a.archive).resolve()
    state_path, report_path = Path(a.state), Path(a.report)
    state_path.unlink(missing_ok=True)

    config = LLMConfig(provider="openai", model="unused", api_key="local-no-auth",
                       base_url="http://127.0.0.1:9/v1")
    coscientist = CoScientist(llm_config=config, storage_path=str(state_path), max_workers=1)
    coscientist.memory.metadata.update({
        "research_goal": source["question"],
        "research_plan_config": {"evaluation_criteria": CRITERIA},
        "campaign": {
            "name": "quantum-biology-50-three-checkpoint-ensemble",
            "protocol": "complete bidirectional round robin",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    })
    for row in hypotheses:
        content = (
            f"Mechanistic hypothesis: {row['mechanism']}\n"
            f"Physiological process: {row['physiological_process']}\n"
            f"Discriminating predictions: {'; '.join(row['discriminating_predictions'])}\n"
            f"Decisive falsifier: {row['decisive_falsifier']}\n"
            f"Boundary conditions: {row['boundary_conditions']}\n"
            f"Current evidence status: {row['evidence_status']}"
        )
        coscientist.memory.add_hypothesis(ResearchHypothesis(
            content=content, summary=row["title"], agent_id="quantum-biology-curated",
            hypothesis_id=row["hypothesis_id"], metadata={
                "title": row["title"], "domain": row["domain"],
                "evidence_status": row["evidence_status"],
            },
        ))

    routers = checkpoint_router_specs(
        archive / "model",
        [
            {"name": "calibrated-1500", "path": archive / "checkpoints/step-0001500",
             "role": "lowest NLL and probability drift"},
            {"name": "transfer-2200", "path": archive / "checkpoints/step-0002200",
             "role": "maximum protected hard consistency"},
            {"name": "balanced-2400", "path": archive / "checkpoints/step-0002400",
             "role": "held-out/transfer Pareto compromise"},
        ],
        device="cuda", batch_size=a.batch_size,
    )
    agent = coscientist.configure_laya_judges(routers, aggregation_policy=a.policy)
    hs = coscientist.get_all_hypotheses()
    pairs, fixture_ids = [], []
    for first, second in itertools.combinations(hs, 2):
        fixture_ids.append((first.hypothesis_id, second.hypothesis_id))
        pairs.extend([(first, second), (second, first)])

    started = time.perf_counter()
    native_matches = agent.judge_batch(pairs, batch_size=a.batch_size)
    inference_seconds = time.perf_counter() - started
    coscientist.memory.tournament_state["rankings"] = [
        {"rank": i + 1, "hypothesis_id": h.hypothesis_id,
         "elo_rating": h.elo_rating, "summary": h.summary}
        for i, h in enumerate(sorted(hs, key=lambda x: x.elo_rating, reverse=True))
    ]
    coscientist.memory.save()

    ratings = {h.hypothesis_id: 1200.0 for h in hs}
    wins, losses, unresolved = ({h.hypothesis_id: 0 for h in hs} for _ in range(3))
    fixtures = []
    for index, (first_id, second_id) in enumerate(fixture_ids):
        ab, ba = native_matches[2 * index], native_matches[2 * index + 1]
        winner_ab, winner_ba = semantic_winner(ab), semantic_winner(ba)
        winner = winner_ab if winner_ab is not None and winner_ab == winner_ba else None
        if winner is not None:
            loser = second_id if winner == first_id else first_id
            update_elo(ratings, winner, loser)
            wins[winner] += 1
            losses[loser] += 1
        else:
            unresolved[first_id] += 1
            unresolved[second_id] += 1
        fixtures.append({
            "fixture": index + 1, "hypothesis_ids": [first_id, second_id],
            "winner_ab": winner_ab, "winner_ba": winner_ba,
            "consensus_winner": winner,
            "resolution": "decisive" if winner else "orientation_disagreement_or_tie",
            "native_match_ids": [ab["match_id"], ba["match_id"]],
        })
    by_id = {h.hypothesis_id: h for h in hs}
    consensus = sorted([
        {"rank": 0, "hypothesis_id": hid, "title": by_id[hid].summary,
         "domain": by_id[hid].metadata["domain"], "evidence_status": by_id[hid].metadata["evidence_status"],
         "elo_rating": rating, "wins": wins[hid], "losses": losses[hid],
         "unresolved": unresolved[hid]}
        for hid, rating in ratings.items()
    ], key=lambda x: x["elo_rating"], reverse=True)
    for rank, row in enumerate(consensus, 1):
        row["rank"] = rank

    decisive = sum(item["consensus_winner"] is not None for item in fixtures)
    model_votes = {
        name: dict(__import__("collections").Counter(
            judgment["overall_winner"] for match in native_matches
            for judgment in match["model_judgments"] if judgment["name"] == name
        )) for name in ["calibrated-1500", "transfer-2200", "balanced-2400"]
    }
    model_unanimity = sum(
        len({x["overall_winner"] for x in match["model_judgments"]}) == 1
        for match in native_matches
    )
    persisted = json.loads(state_path.read_text())
    ids = {x["hypothesis_id"] for x in persisted["hypotheses"]}
    stored = persisted["tournament_state"]["matches"]
    checks = {
        "hypothesis_count": len(ids) == 50,
        "native_match_count": len(stored) == 2450,
        "one_record_per_oriented_pair": len({
            (m["hypothesis1_id"], m["hypothesis2_id"]) for m in stored
        }) == 2450,
        "referential_integrity": all(
            m["hypothesis1_id"] in ids and m["hypothesis2_id"] in ids for m in stored
        ),
        "ensemble_attribution": all(
            m.get("decision_engine") == "laya_ensemble" and
            len(m.get("model_judgments", [])) == 3 for m in stored
        ),
        "fixture_count": len(fixtures) == 1225,
    }
    report = {
        "schema_version": "jnana_quantum_biology_ensemble_v1",
        "question": source["question"], "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(Path(a.source).resolve()), "archive": str(archive),
        "aggregation_policy": a.policy, "criteria": CRITERIA,
        "counts": {"hypotheses": 50, "fixtures": 1225,
                   "oriented_native_matches": 2450, "model_inferences": 7350,
                   "questions_per_inference": len(CRITERIA) + 1,
                   "decisive_fixtures": decisive, "unresolved_fixtures": 1225 - decisive,
                   "model_unanimous_oriented_matches": model_unanimity},
        "timing": {"ensemble_inference_seconds": inference_seconds,
                   "oriented_matches_per_second": 2450 / inference_seconds},
        "checks": checks, "per_model_vote_counts": model_votes,
        "native_ranking": persisted["tournament_state"]["rankings"],
        "consensus_ranking": consensus, "fixtures": fixtures,
    }
    dump(report_path, report)
    print(json.dumps({
        "report": str(report_path), "state": str(state_path), "counts": report["counts"],
        "timing": report["timing"], "checks": checks,
        "consensus_top5": consensus[:5],
    }, indent=2))
    if not all(checks.values()):
        raise RuntimeError(f"Integrity checks failed: {checks}")


if __name__ == "__main__":
    main()
