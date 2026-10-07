#!/usr/bin/env python3
"""Download private GitHub weights and register them with native Jnana."""

from pathlib import Path

from jnana.protognosis.agents.laya_github_weights import (
    github_checkpoint_router_specs,
)
from jnana.protognosis.core.coscientist import CoScientist
from jnana.protognosis.core.multi_llm_config import LLMConfig

# Authentication is owned by the GitHub CLI. Run `gh auth login` once; do not
# place a GitHub token in this script or in Jnana configuration.
routers = github_checkpoint_router_specs(
    repository="architvasan/laya-scientific-weights",
    release="scientific-laya-v1",
    steps=(1500, 2200, 2400),
    cache_dir=Path.home() / ".cache" / "jnana" / "laya",
    device="cuda",
    batch_size=128,
)

# Other Jnana agents still require an LLM configuration. Laya ranking itself
# does not call this placeholder endpoint.
llm_config = LLMConfig(
    provider="openai",
    model="unused",
    api_key="local-no-auth",
    base_url="http://127.0.0.1:9/v1",
)
state = Path.home() / ".local" / "share" / "jnana" / "github-laya-memory.json"
coscientist = CoScientist(
    llm_config=llm_config,
    storage_path=str(state),
    max_workers=1,
)
coscientist.configure_laya_judges(
    routers,
    aggregation_policy="majority_confidence_tiebreak",
)

for spec in routers:
    print(f"{spec['name']}: {spec['provenance']['checkpoint']}")
print(coscientist.memory.metadata["laya_judging"])
