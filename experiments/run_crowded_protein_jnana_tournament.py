#!/usr/bin/env python3
"""Run a checkpointed 25-hypothesis, 300-match Jnana/Laya tournament."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

QUESTION = """Formulate competing, high-yield computational and structural strategies to model the three-dimensional structures, conformational ensembles, stability, association, and function of proteins inside a highly crowded cellular macromolecular environment spanning dilute conditions through 40% occupied volume fraction.

The scope is intracellular proteins in explicit or effective cytoplasmic/nucleoplasmic environments, not dilute-solution structure prediction alone. The field must cover atomistic and coarse-grained molecular dynamics; explicit polydisperse crowders; implicit crowding/free-volume and scaled-particle models; Brownian and dissipative-particle dynamics; lattice/polymer and field-theoretic models; multiscale QM/MM where relevant; integrative structural modeling constrained by in-cell NMR, FRET, EPR, cross-linking, cryo-ET, scattering, and proteomics; deep-learning structure/ensemble prediction corrected for cellular context; enhanced sampling and reweighting; hydrodynamics; electrostatics, metabolites, ions, water activity, viscosity and confinement; transient quinary interactions; chaperones and active ATP-driven remodeling; cotranslational folding; phase separation, aggregation and kinetic arrest; and a strong null in which dilute predictions plus excluded-volume corrections suffice.

Generate exactly 25 mutually distinguishable hypotheses. Each must state: scope and crowding regime; minimal causal mechanism; representation and simulation/inference method; required observables and training/validation data; quantitative predictions as occupancy approaches 40%; a decisive falsifier; the best discriminating benchmark or experiment; major computational cost and identifiability risks; and boundary conditions. Separate equilibrium thermodynamic effects from kinetic, hydrodynamic, and active nonequilibrium effects. Do not treat compaction, slowed diffusion, puncta, or agreement with one structural observable as proof of mechanism."""

MATCH_CRITERIA = [
    "physical fidelity at 0-40% occupied volume",
    "ability to predict structures and conformational ensembles quantitatively",
    "identifiability from feasible in-cell measurements",
    "multiscale computational tractability",
    "falsifiability against strong mechanistic alternatives",
    "generalization across protein and crowder classes",
]
ROUNDS = (50, 50, 50, 50, 50, 50)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def git_info(root: Path) -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=True).stdout.strip()
    diff = run("diff", "--binary")
    return {
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--short")),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
    }


def rows_from(response) -> list[dict]:
    if isinstance(response, tuple):
        response = response[0]
    if not isinstance(response, dict) or not isinstance(response.get("hypotheses"), list):
        raise ValueError("generation response lacks a hypotheses list")
    return response["hypotheses"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=".loam/state/crowded-protein-structure-tournament")
    parser.add_argument("--endpoint", default="http://localhost:51900/v1")
    parser.add_argument("--generation-model", default="argo:gpt-5.6-terra")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out = Path(args.out_dir).resolve()
    state = out / "jnana-memory.json"
    manifest_path = out / "progress.json"
    report_path = out / "tournament.json"
    rounds_dir = out / "rounds"
    out.mkdir(parents=True, exist_ok=True)
    rounds_dir.mkdir(exist_ok=True)
    if state.exists() and not args.resume:
        raise SystemExit(f"Refusing to overwrite {state}; use --resume")

    jnana_root = (Path.cwd().parent / "jnana").resolve()
    sys.path.insert(0, str(jnana_root))
    from jnana.protognosis.core.agent_core import ResearchHypothesis
    from jnana.protognosis.core.coscientist import CoScientist
    from jnana.protognosis.core.multi_llm_config import LLMConfig
    from jnana.protognosis.periodic_tournament import run_batched_laya_tournament

    config = LLMConfig(
        provider="openai", model=args.generation_model,
        api_key=os.environ.get("OPENAI_API_KEY", "local-no-auth"), base_url=args.endpoint,
        model_adapter={"omit_temperature": True, "json_max_tokens": 32768},
    )
    manifest = {
        "status": "starting", "question": QUESTION, "requested_hypotheses": 25,
        "requested_matches": 300, "round_schedule": list(ROUNDS),
        "hypotheses_persisted": 0, "matches_completed": 0, "current_round": 0,
        "generation_attempts": [], "updated_at": now(),
        "engine": {"name": "Jnana/ProtoGnosis + Laya", "pairing": "native random pair periodic tournament",
                   "judge": "Laya 0.3.x calibrated decision engine", "jnana": git_info(jnana_root),
                   "loam": git_info(Path.cwd()), "python": sys.executable,
                   "endpoint": args.endpoint, "generation_model": args.generation_model},
    }
    if manifest_path.exists() and args.resume:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["generation_attempts"] = old.get("generation_attempts", [])
    dump(manifest_path, manifest)

    memory = json.loads(state.read_text(encoding="utf-8")) if state.exists() else {}
    if len(memory.get("hypotheses", [])) < 25:
        c = CoScientist(llm_config=config, storage_path=str(state), max_workers=1)
        c.start()
        started = time.time()
        try:
            c.memory.metadata["research_goal"] = QUESTION
            c.memory.metadata["research_plan_config"] = {"evaluation_criteria": MATCH_CRITERIA}
            seen = {hashlib.sha256((h.content + "\n" + h.summary).strip().lower().encode()).hexdigest()
                    for h in c.get_all_hypotheses()}
            for attempt in range(1, 4):
                missing = 25 - len(c.get_all_hypotheses())
                if missing <= 0:
                    break
                manifest.update(status="generation_batch_in_flight", hypotheses_persisted=len(c.get_all_hypotheses()), updated_at=now())
                dump(manifest_path, manifest)
                llm = c._get_llm_for_agent("generation", "crowded-protein-generation")
                schema = {"hypotheses": [{"title": "string", "content": "string", "summary": "string",
                          "key_novelty_aspects": ["string"], "testable_predictions": ["string"],
                          "validation_method": "string", "boundary_conditions": ["string"]}]}
                t0 = time.time()
                returned = rows_from(llm.generate_with_json_output(QUESTION, schema))
                accepted = []
                rejected = duplicates = 0
                for row in returned:
                    if not all(str(row.get(k, "")).strip() for k in ("title", "content", "summary")):
                        rejected += 1
                        continue
                    digest = hashlib.sha256((row["content"] + "\n" + row["summary"]).strip().lower().encode()).hexdigest()
                    if digest in seen:
                        duplicates += 1
                        continue
                    seen.add(digest)
                    accepted.append(ResearchHypothesis(
                        content=row["content"], summary=row["summary"], agent_id="crowded-protein-generation",
                        metadata={"title": row["title"], "key_novelty_aspects": row.get("key_novelty_aspects", []),
                                  "testable_predictions": row.get("testable_predictions", []),
                                  "validation_method": row.get("validation_method", ""),
                                  "boundary_conditions": row.get("boundary_conditions", []),
                                  "generation_strategy": "batched_diversity"}))
                    if len(accepted) >= missing:
                        break
                for hypothesis in accepted:
                    c.memory.add_hypothesis(hypothesis)
                c.memory.save()
                manifest["generation_attempts"].append({"attempt": attempt, "requested_missing": missing,
                    "returned": len(returned), "accepted": len(accepted), "rejected": rejected,
                    "duplicates": duplicates, "seconds": round(time.time() - t0, 2)})
                manifest.update(hypotheses_persisted=len(c.get_all_hypotheses()), updated_at=now())
                dump(manifest_path, manifest)
        finally:
            c.stop()
        memory = json.loads(state.read_text(encoding="utf-8"))
        manifest["generation_seconds"] = round(time.time() - started, 2)

    hypotheses = memory.get("hypotheses", [])
    if len(hypotheses) != 25 or len({h["hypothesis_id"] for h in hypotheses}) != 25:
        raise RuntimeError(f"Hypothesis integrity failure: {len(hypotheses)}/25")
    manifest["hypotheses_persisted"] = len(hypotheses)
    # Ensure the judging rubric survives older/resumed state.
    memory.setdefault("metadata", {})["research_goal"] = QUESTION
    memory["metadata"]["research_plan_config"] = {"evaluation_criteria": MATCH_CRITERIA}
    dump(state, memory)

    completed = len(memory.get("tournament_state", {}).get("matches", []))
    report = {"schema_version": "loam_jnana_laya_crowded_protein_v1", "question": QUESTION,
              "created_at": now(), "engine": manifest["engine"], "round_schedule": list(ROUNDS), "rounds": []}
    cumulative = 0
    for index, batch in enumerate(ROUNDS, 1):
        cumulative += batch
        artifact = rounds_dir / f"round-{index:02d}.json"
        if cumulative <= completed and artifact.exists():
            report["rounds"].append(json.loads(artifact.read_text(encoding="utf-8")))
            continue
        before = len(json.loads(state.read_text(encoding="utf-8")).get("tournament_state", {}).get("matches", []))
        needed = cumulative - before
        manifest.update(status="matches_in_flight", current_round=index, matches_completed=before, updated_at=now())
        dump(manifest_path, manifest)
        t0 = time.time()
        result = run_batched_laya_tournament(
            state_file=state, llm_config=config, match_count=needed, top_k=5, batch_size=16
        )
        current = json.loads(state.read_text(encoding="utf-8"))
        after = len(current.get("tournament_state", {}).get("matches", []))
        if result.get("matches_completed") != needed or after != cumulative:
            raise RuntimeError(f"Round {index} persisted {after-before}/{needed} matches")
        top = sorted(current["hypotheses"], key=lambda h: h.get("elo_rating", 1200), reverse=True)[:5]
        validation = {"class": "structural and tournament-integrity gate", "status": "passed",
                      "checks": {"exact_hypothesis_count": len(current["hypotheses"]) == 25,
                                 "unique_hypothesis_ids": len({h["hypothesis_id"] for h in current["hypotheses"]}) == 25,
                                 "match_referential_integrity": all(m.get("hypothesis1_id") in {h["hypothesis_id"] for h in current["hypotheses"]} and m.get("hypothesis2_id") in {h["hypothesis_id"] for h in current["hypotheses"]} for m in current["tournament_state"]["matches"]),
                                 "laya_attribution": all(m.get("decision_engine") == "laya" for m in current["tournament_state"]["matches"][-needed:])},
                      "note": "This gate checks run integrity, not empirical support; formal Loam evidence credit remains zero."}
        if not all(validation["checks"].values()):
            validation["status"] = "failed"
            raise RuntimeError(f"Round {index} integrity gate failed")
        item = {"round": index, "matches_in_round": needed, "cumulative_matches": after,
                "elapsed_seconds": round(time.time() - t0, 2), "tournament_result": result,
                "current_top_5": [{"hypothesis_id": h["hypothesis_id"], "title": h.get("metadata", {}).get("title", ""),
                                   "summary": h.get("summary", ""), "elo_rating": h.get("elo_rating", 1200)} for h in top],
                "validation": validation}
        dump(artifact, item)
        report["rounds"].append(item)
        manifest.update(status="round_complete", matches_completed=after, updated_at=now())
        dump(manifest_path, manifest)

    memory = json.loads(state.read_text(encoding="utf-8"))
    report["hypotheses"] = memory["hypotheses"]
    report["matches"] = memory["tournament_state"]["matches"]
    report["final_ranking"] = sorted([
        {"hypothesis_id": h["hypothesis_id"], "title": h.get("metadata", {}).get("title", ""),
         "summary": h.get("summary", ""), "content": h.get("content", ""), "elo_rating": h.get("elo_rating", 1200),
         "wins": h.get("tournament_wins", 0), "losses": h.get("tournament_losses", 0),
         "matches": len(h.get("tournament_matches", [])), "formal_loam_evidence_credit": 0}
        for h in memory["hypotheses"]], key=lambda h: h["elo_rating"], reverse=True)
    report["completed_at"] = now()
    dump(report_path, report)
    manifest.update(status="completed", matches_completed=300, completed_at=now(), report=str(report_path), updated_at=now())
    dump(manifest_path, manifest)
    print(json.dumps({"status": "completed", "hypotheses": 25, "matches": 300, "report": str(report_path)}))


if __name__ == "__main__":
    main()
