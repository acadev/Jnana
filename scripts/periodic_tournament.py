#!/usr/bin/env python3
"""Run a durable Jnana/ProtoGnosis hypothesis-tournament iteration.

This runner targets OpenAI-compatible ``/chat/completions`` endpoints and keeps
all research state in a ContextMemory JSON document, so independent scheduled
invocations continue the same tournament.
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Support direct execution from a repository checkout without requiring an
# editable installation. The repository root is one level above ``scripts``.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import requests

from jnana.protognosis.core.coscientist import CoScientist
from jnana.protognosis.core.multi_llm_config import LLMConfig


DEFAULT_STRATEGIES = [
    "literature_exploration",
    "scientific_debate",
    "assumptions_identification",
    "research_expansion",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal", required=True, help="Research objective for this persistent tournament.")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible API base URL, normally ending in /v1.")
    parser.add_argument("--model", required=True, help="Model ID accepted by the endpoint.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable containing the API key.")
    parser.add_argument("--state-file", required=True, type=Path, help="Durable ContextMemory JSON state path.")
    parser.add_argument("--report-dir", required=True, type=Path, help="Directory for immutable run reports.")
    parser.add_argument("--new-hypotheses", type=int, default=4)
    parser.add_argument("--matches", type=int, default=10)
    parser.add_argument("--evolutions", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--json-max-tokens", type=int, default=2048,
                        help="Maximum completion tokens for generation, review, and tournament JSON responses.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--skip-endpoint-check", action="store_true")
    parser.add_argument("--allow-goal-change", action="store_true", help="Allow replacement of an existing state file's research goal.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.base_url.startswith(("https://", "http://")):
        raise ValueError("--base-url must start with http:// or https://")
    for name in ("new_hypotheses", "matches", "evolutions", "top_k", "max_tokens", "json_max_tokens"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.matches and not args.state_file.exists() and args.new_hypotheses < 2:
        raise ValueError("A first run needs at least two --new-hypotheses before it can schedule matches")


def endpoint_check(base_url: str, api_key: str, model: str, timeout: float) -> None:
    """Verify credentials, routing, and the minimal Chat Completions contract."""
    url = base_url.rstrip("/") + "/chat/completions"
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": "Reply exactly OK."}], "max_tokens": 32},
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Endpoint check failed ({response.status_code}): {response.text[:500]}")
    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Endpoint returned success but not an OpenAI Chat Completions response") from exc
    if not content:
        raise RuntimeError("Endpoint returned an empty completion during endpoint check")


def report_payload(coscientist: CoScientist, args: argparse.Namespace, started_at: str) -> Dict[str, Any]:
    top = coscientist.get_top_hypotheses(args.top_k)
    stats = coscientist.get_statistics()
    return {
        "schema_version": "periodic-tournament-run-v1",
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "research_goal": args.goal,
        "endpoint": {"base_url": args.base_url.rstrip("/"), "model": args.model, "api_key_env": args.api_key_env},
        "budget": {"new_hypotheses": args.new_hypotheses, "matches": args.matches, "evolutions": args.evolutions},
        "statistics": stats,
        "top_hypotheses": top,
        "model_usage": coscientist.get_agent_usages(verbose=False, output_path=None),
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Required API key environment variable is unset: {args.api_key_env}")

    if not args.skip_endpoint_check:
        endpoint_check(args.base_url, api_key, args.model, args.timeout)

    args.state_file.parent.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    config = LLMConfig(
        provider="openai",
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        model_adapter={"omit_temperature": True, "json_max_tokens": args.json_max_tokens},
    )
    # One worker avoids simultaneous read-modify-write races in the JSON state file.
    coscientist = CoScientist(llm_config=config, storage_path=str(args.state_file), max_workers=1)
    prior_goal = coscientist.memory.metadata.get("research_goal")
    if prior_goal and prior_goal != args.goal and not args.allow_goal_change:
        raise RuntimeError("State file belongs to a different goal. Use a new --state-file or --allow-goal-change.")

    coscientist.start()
    try:
        if not prior_goal or args.allow_goal_change:
            coscientist.set_research_goal(args.goal)

        if args.new_hypotheses:
            coscientist.generate_hypotheses(args.new_hypotheses, DEFAULT_STRATEGIES)
            coscientist.wait_for_completion()

        hypotheses = coscientist.get_all_hypotheses()
        if hypotheses:
            coscientist.review_hypotheses([h.hypothesis_id for h in hypotheses], ["initial_review"])
            coscientist.wait_for_completion()

        if args.matches:
            if len(coscientist.get_all_hypotheses()) < 2:
                raise RuntimeError("Tournament requires at least two hypotheses; increase --new-hypotheses.")
            coscientist.run_tournament(args.matches)
            coscientist.wait_for_completion()

        if args.evolutions and coscientist.get_all_hypotheses():
            coscientist.evolve_hypotheses(count=args.evolutions, evolution_types=["improve_hypothesis"], top_k=max(2, args.top_k))
            coscientist.wait_for_completion()

        # Explicitly persist any final rankings/statistics before the process exits.
        payload = report_payload(coscientist, args, started_at)
        coscientist.memory.save()
        report_path = args.report_dir / f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"report_path": str(report_path), "statistics": payload["statistics"], "top_hypotheses": payload["top_hypotheses"]}, indent=2))
        return 0
    finally:
        coscientist.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        sys.exit(main())
    except Exception as exc:
        logging.exception("Periodic tournament failed: %s", exc)
        sys.exit(1)
