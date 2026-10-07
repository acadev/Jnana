import base64
import hashlib
import json
from pathlib import Path

import pytest

from jnana.protognosis.agents import laya_github_weights as weights


def manifest_for(payload: bytes) -> dict:
    return {
        "schema_version": "laya.scientific.weights.v1",
        "release": "test-release",
        "base_model": "example/base",
        "training": {
            "adapter": {
                "rank": 16,
                "alpha": 32,
                "dropout": 0.05,
                "targets": ["Wqkv", "Wo"],
            }
        },
        "assets": [
            {
                "name": "step-1500.safetensors",
                "step": 1500,
                "role": "balanced",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }


def test_fetch_manifest_uses_authenticated_gh_api(monkeypatch):
    manifest = manifest_for(b"weights")
    encoded = base64.b64encode(json.dumps(manifest).encode()).decode()
    calls = []

    def fake_run(command):
        calls.append(command)
        return json.dumps({"encoding": "base64", "content": encoded})

    monkeypatch.setattr(weights, "_run", fake_run)
    assert weights.fetch_registry_manifest("owner/private") == manifest
    assert calls == [
        [
            "gh",
            "api",
            "repos/owner/private/contents/model-manifest.json?ref=main",
        ]
    ]


def test_download_verifies_and_reuses_cached_checkpoint(monkeypatch, tmp_path):
    payload = b"tiny verified checkpoint"
    manifest = manifest_for(payload)
    downloads = []

    monkeypatch.setattr(
        weights,
        "fetch_registry_manifest",
        lambda _repository: manifest,
    )

    def fake_run(command):
        downloads.append(command)
        destination = Path(command[command.index("--dir") + 1])
        (destination / "step-1500.safetensors").write_bytes(payload)
        return ""

    monkeypatch.setattr(weights, "_run", fake_run)
    returned_manifest, checkpoints = weights.download_release_checkpoints(
        repository="owner/private",
        release="test-release",
        steps=[1500],
        cache_dir=tmp_path,
    )

    checkpoint = checkpoints[0]["path"]
    assert returned_manifest == manifest
    assert (checkpoint / "model.safetensors").read_bytes() == payload
    assert (checkpoint / "COMPLETE").read_text() == "verified\n"
    assert json.loads((checkpoint / "config.json").read_text())["lora_rank"] == 16
    assert len(downloads) == 1

    weights.download_release_checkpoints(
        repository="owner/private",
        release="test-release",
        steps=[1500],
        cache_dir=tmp_path,
    )
    assert len(downloads) == 1


def test_download_rejects_hash_mismatch(monkeypatch, tmp_path):
    manifest = manifest_for(b"expected")
    manifest["assets"][0]["bytes"] = len(b"tampered")
    monkeypatch.setattr(
        weights,
        "fetch_registry_manifest",
        lambda _repository: manifest,
    )

    def fake_run(command):
        destination = Path(command[command.index("--dir") + 1])
        (destination / "step-1500.safetensors").write_bytes(b"tampered")
        return ""

    monkeypatch.setattr(weights, "_run", fake_run)
    with pytest.raises(RuntimeError, match="SHA-256 verification failed"):
        weights.download_release_checkpoints(
            repository="owner/private",
            release="test-release",
            steps=[1500],
            cache_dir=tmp_path,
        )


def test_unknown_step_lists_available_steps():
    with pytest.raises(ValueError, match="Available steps: 1500"):
        weights._selected_assets(manifest_for(b"weights"), [2200])


def test_github_router_specs_connects_prepared_weights_to_native_router(
    monkeypatch,
    tmp_path,
):
    prepared = weights.PreparedLayaWeights(
        base_model=tmp_path / "base",
        checkpoints=(
            {
                "name": "step-1500",
                "path": tmp_path / "step-1500",
                "role": "balanced",
                "weight": 1.0,
            },
        ),
        manifest=manifest_for(b"weights"),
    )
    monkeypatch.setattr(
        weights, "prepare_github_laya_weights", lambda **_kwargs: prepared
    )

    captured = {}

    def fake_router_specs(base_model, checkpoints, *, device, batch_size):
        captured.update(
            base_model=base_model,
            checkpoints=checkpoints,
            device=device,
            batch_size=batch_size,
        )
        return ["router-spec"]

    monkeypatch.setattr(
        "jnana.protognosis.agents.laya_checkpoint_router.checkpoint_router_specs",
        fake_router_specs,
    )

    result = weights.github_checkpoint_router_specs(
        steps=[1500],
        device="cpu",
        batch_size=7,
    )
    assert result == ["router-spec"]
    assert captured == {
        "base_model": prepared.base_model,
        "checkpoints": prepared.checkpoints,
        "device": "cpu",
        "batch_size": 7,
    }
