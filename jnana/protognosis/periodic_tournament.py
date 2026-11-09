"""Bounded, durable ProtoGnosis tournament iterations.

This module is intentionally a library API: callers own scheduling and model
configuration.  It loads one workflow-scoped ``ContextMemory`` JSON file, runs a
small number of pairwise ranking tasks serially, persists the result, and
returns a compact snapshot suitable for another system to render.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .core.coscientist import CoScientist
from .core.multi_llm_config import LLMConfig


def _await_matches(
    coscientist: CoScientist,
    before: int,
    wanted: int,
    max_seconds: float | None,
    poll_seconds: float = 2.0,
) -> str:
    """Block until ``wanted`` new matches are recorded, or the deadline passes.

    ``CoScientist.wait_for_completion()`` must NOT be used for this.
    ``CoScientist.start()`` runs the whole agent ensemble -- generation,
    reflection, ranking, evolution, proximity, meta-review -- and those agents
    keep enqueueing their own work, so the queue it waits on never empties and
    the call does not return. ``match_count`` only seeds a few extra ranking
    tasks on top of that, which is why it never behaved like a bound.

    Measured 2026-10-31: ``match_count=1`` against a 6-hypothesis pool recorded
    24 matches and was still going -- more than the 15 pairings that pool even
    admits -- while on a 351-hypothesis pool the evolution agents grew it to 376
    mid-run. Polling the recorded match count makes ``match_count`` an actual
    bound; ``max_seconds`` is the backstop for a pool that stops producing.
    """

    deadline = None if max_seconds is None else time.monotonic() + max_seconds
    while True:
        recorded = len(coscientist.memory.tournament_state.get("matches", []))
        if recorded - before >= wanted:
            return "match_count_reached"
        if deadline is not None and time.monotonic() >= deadline:
            return "deadline_reached"
        time.sleep(poll_seconds)


def _run_swiss_rounds(
    coscientist: CoScientist,
    rounds: int,
    max_seconds: float | None,
) -> str:
    """Run ``rounds`` sequential Swiss rounds, blocking on each before the next.

    Each round is paired from the CURRENT Elo standings, which the ranking agent
    updates per match (see ``specialized_agents._tournament_match``). So a round
    must finish recording before the next is paired -- otherwise later rounds
    pair on stale ratings and Swiss degenerates to random. ``_await_matches``
    is that barrier.

    The ``max_seconds`` budget is split evenly across rounds so one slow round
    cannot starve the rest. The last stop reason reached is returned; a deadline
    in any round stops the loop and yields whatever ranking exists.

    Note: like the random path, the barrier counts ALL new matches, including
    any the ensemble's own ranking agents record. A round can therefore clear
    on fewer of its own pairings than it scheduled; ``match_count`` has always
    been a bound here, not an exact quota.
    """
    if rounds < 1:
        raise ValueError("rounds must be at least one")
    round_deadline = None if max_seconds is None else max_seconds / rounds
    stop_reason = "not_requested"
    for _ in range(rounds):
        before = len(coscientist.memory.tournament_state.get("matches", []))
        info = coscientist.run_tournament(scheme="swiss")
        wanted = int(info.get("match_count") or 0)
        if wanted == 0:
            return "no_pairings"
        stop_reason = _await_matches(coscientist, before, wanted, round_deadline)
        if stop_reason == "deadline_reached":
            break
    return stop_reason


def _compact_hypothesis(hypothesis: dict[str, Any]) -> dict[str, Any]:
    """Return ranking data without exposing full hypothesis bodies."""

    return {
        "hypothesis_id": str(hypothesis.get("hypothesis_id") or ""),
        "summary": " ".join(str(hypothesis.get("summary") or "").split())[:240],
        "elo_rating": round(float(hypothesis.get("elo_rating") or 0.0), 1),
        "tournament_matches": int(hypothesis.get("tournament_matches") or 0),
    }


def run_incremental_tournament(
    *,
    state_file: str | Path,
    llm_config: LLMConfig,
    match_count: int = 1,
    top_k: int = 3,
    max_seconds: float | None = None,
    scheme: str = "random",
    rounds: int = 1,
    workers: int = 1,
) -> dict[str, Any]:
    """Run and persist one bounded tournament iteration.

    The caller must serialize access to ``state_file``.  One worker is used so
    ranking tasks and the terminal rankings update execute in a deterministic
    order against the JSON-backed context memory.

    ``match_count`` is a real upper bound on the matches this call waits for,
    and ``max_seconds`` wall-clocks the wait so a caller on a finite compute
    allocation always gets a ranking and a selection back. Leave ``max_seconds``
    as ``None`` only when the caller has no deadline at all; see
    :func:`_await_matches` for why an unbounded wait does not terminate.

    ``scheme`` selects the pairing: ``"random"`` (default, unchanged -- a single
    batch of ``match_count`` random matches) or ``"swiss"``, which plays
    ``rounds`` sequential Swiss rounds, re-pairing on the current Elo between
    rounds (``match_count`` does not apply). "Deep Swiss" is just a larger
    ``rounds``.

    ``workers`` runs a round's matches in parallel (the round-synchronous
    multi-dueling pattern). This is only offered for ``"swiss"``: a Swiss round's
    pairings are DISJOINT (each hypothesis plays once), so concurrent matches
    never touch the same Elo rating, and ``ContextMemory``'s lock serializes the
    shared match list and the JSON save. The between-round barrier
    (:func:`_await_matches`) still holds -- round r+1 is not paired until round
    r's matches are all recorded, so later rounds pair on complete standings.
    ``"random"`` pairs can overlap, so it stays serial.
    """

    if match_count < 0:
        raise ValueError("match_count must be non-negative")
    if top_k < 1:
        raise ValueError("top_k must be at least one")
    if scheme not in ("random", "swiss"):
        raise ValueError(f"unknown tournament scheme: {scheme!r}")
    if scheme == "swiss" and rounds < 1:
        raise ValueError("rounds must be at least one for the swiss scheme")
    if workers < 1:
        raise ValueError("workers must be at least one")
    if workers > 1 and scheme != "swiss":
        # Random pairs can share a hypothesis, so parallel matches would race on
        # its Elo (read-modify-write). Only Swiss guarantees disjoint pairings.
        raise ValueError("workers > 1 is only supported for the swiss scheme")

    state_path = Path(state_file).expanduser()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    coscientist = CoScientist(
        llm_config=llm_config,
        storage_path=str(state_path),
        max_workers=workers,
    )
    coscientist.start()
    try:
        hypothesis_count = len(coscientist.get_all_hypotheses())
        if hypothesis_count < 2:
            return {
                "status": "skipped",
                "reason": "fewer_than_two_hypotheses",
                "hypothesis_count": hypothesis_count,
                "matches_requested": match_count,
                "matches_completed": 0,
                "top_hypotheses": [],
            }

        before_matches = len(coscientist.memory.tournament_state.get("matches", []))
        stop_reason = "not_requested"
        if scheme == "swiss":
            # Swiss owns its own round loop; each round pairs on the Elo the
            # previous round produced. match_count does not apply.
            stop_reason = _run_swiss_rounds(coscientist, rounds, max_seconds)
        elif match_count:
            coscientist.run_tournament(match_count=match_count)
            stop_reason = _await_matches(
                coscientist, before_matches, match_count, max_seconds)
        statistics = coscientist.get_statistics()
        coscientist.memory.save()
        after_matches = len(coscientist.memory.tournament_state.get("matches", []))
        matches_completed = after_matches - before_matches
        ranked = [
            _compact_hypothesis(hypothesis)
            for hypothesis in coscientist.get_top_hypotheses(top_k)
        ]
        # `status` reports whether this call produced a USABLE RANKING, not
        # whether an arbitrary match quota was met. Consumers gate on it:
        # loam's Tournament.latest_report() returns a report only when
        # status == "completed", so anything else makes the ranking invisible
        # and the round dispatches nothing.
        #
        # Two earlier definitions were both wrong for that consumer. `==
        # match_count` reported "incomplete" when the ensemble's own ranking
        # agents did MORE matches than asked, which is the normal case. `>=
        # match_count` then reported "incomplete" whenever max_seconds fired
        # first -- which is exactly what the deadline exists for, so the
        # backstop produced a ranking and simultaneously marked it unusable.
        # Observed 2026-10-31: 6 hypotheses ranked 1410/1390/1281/1106 Elo,
        # discarded because 104 < 200 requested.
        #
        # A ranking is usable when there is one. Partialness is still fully
        # reported through stop_reason, matches_requested and matches_completed.
        status = "completed" if ranked else "incomplete"
        result = {
            "status": status,
            "scheme": scheme,
            "stop_reason": stop_reason,
            "hypothesis_count": hypothesis_count,
            "matches_requested": match_count,
            "matches_completed": matches_completed,
            "top_hypotheses": ranked,
            "statistics": statistics,
        }
        if scheme == "swiss":
            result["rounds_requested"] = rounds
        if status == "incomplete":
            result["reason"] = "tournament produced no ranked hypotheses"
        elif scheme == "swiss":
            # match_count is not a Swiss quota; partial means a deadline cut the
            # round loop short before all `rounds` were played.
            if stop_reason == "deadline_reached":
                result["partial"] = (
                    f"ranking from Swiss rounds cut short by deadline "
                    f"({matches_completed} matches played)"
                )
        elif matches_completed < match_count:
            # Usable, but say plainly that it was cut short.
            result["partial"] = (
                f"ranking from {matches_completed} of {match_count} requested matches "
                f"({stop_reason})"
            )
        return result
    finally:
        coscientist.stop()
