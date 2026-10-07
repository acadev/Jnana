#!/usr/bin/env python3
"""Configure Jnana with three archived Laya checkpoints.

This example assumes execution on Lambda where the durable archive is mounted.
It shows registration only; populate ContextMemory with hypotheses before calling
``run_batched_laya_tournament``.
"""
from pathlib import Path

from jnana.protognosis.agents.laya_checkpoint_router import checkpoint_router_specs
from jnana.protognosis.core.coscientist import CoScientist
from jnana.protognosis.core.multi_llm_config import LLMConfig

ARCHIVE = Path("/lambda_stor/data/avasan/laya-finetune/multisource-nli4ct-2026-10-07")
STATE = ARCHIVE / "examples" / "ensemble-jnnana-memory.json"

# The checkpoints are correlated trajectory states, not independent models.
# Their roles intentionally emphasize different measured behavior.
routers = checkpoint_router_specs(
    ARCHIVE / "model",
    [
        {
            "name": "calibrated-step-1500",
            "path": ARCHIVE / "checkpoints/step-0001500",
            "role": "lowest NLL and original-50 probability drift",
            "weight": 1.0,
        },
        {
            "name": "transfer-step-2200",
            "path": ARCHIVE / "checkpoints/step-0002200",
            "role": "maximum original-50 hard consistency",
            "weight": 1.0,
        },
        {
            "name": "balanced-step-2400",
            "path": ARCHIVE / "checkpoints/step-0002400",
            "role": "held-out/transfer Pareto compromise",
            "weight": 1.0,
        },
    ],
    device="cuda",
    batch_size=32,
)

# This LLM is unused by Laya ranking but remains required by other Jnana agents.
config = LLMConfig(
    provider="openai", model="unused", api_key="local-no-auth",
    base_url="http://127.0.0.1:9/v1",
)
coscientist = CoScientist(llm_config=config, storage_path=str(STATE), max_workers=1)
coscientist.configure_laya_judges(
    routers,
    # Use "unanimous" for conservative abstention, or the default below for
    # majority voting with confidence used only when vote counts tie.
    aggregation_policy="majority_confidence_tiebreak",
)
print(coscientist.memory.metadata["laya_judging"])
