"""Download verified scientific Laya checkpoints from a GitHub release.

The downloader delegates authentication to the GitHub CLI. It never reads or
stores the user's GitHub token. Release assets are verified against the model
registry manifest before they become visible in the local cache.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

DEFAULT_REPOSITORY = "architvasan/laya-scientific-weights"
DEFAULT_RELEASE = "scientific-laya-v1"
DEFAULT_STEPS = (1500, 2200, 2400)
DEFAULT_CACHE = Path.home() / ".cache" / "jnana" / "laya"


@dataclass(frozen=True)
class PreparedLayaWeights:
    """Local files required by :func:`checkpoint_router_specs`."""

    base_model: Path
    checkpoints: tuple[dict, ...]
    manifest: Mapping


def _run(command: Sequence[str]) -> str:
    """Run a command and return stdout with a useful failure message."""

    try:
        result = subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "The GitHub CLI (`gh`) is required. Install it and run `gh auth login`."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "unknown error").strip()
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{detail}") from exc
    return result.stdout


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_registry_manifest(
    repository: str = DEFAULT_REPOSITORY,
    *,
    revision: str = "main",
) -> dict:
    """Read ``model-manifest.json`` from a GitHub repository via ``gh``."""

    response = _run(
        [
            "gh",
            "api",
            f"repos/{repository}/contents/model-manifest.json?ref={revision}",
        ]
    )
    payload = json.loads(response)
    if payload.get("encoding") != "base64" or not payload.get("content"):
        raise RuntimeError("GitHub returned an unsupported manifest response")

    manifest = json.loads(base64.b64decode(payload["content"]).decode("utf-8"))
    if manifest.get("schema_version") != "laya.scientific.weights.v1":
        raise RuntimeError("Unsupported Laya weight manifest schema")
    return manifest


def _selected_assets(manifest: Mapping, steps: Iterable[int]) -> list[Mapping]:
    requested = tuple(dict.fromkeys(int(step) for step in steps))
    available = {int(asset["step"]): asset for asset in manifest.get("assets", [])}
    missing = [step for step in requested if step not in available]
    if missing:
        choices = ", ".join(map(str, sorted(available)))
        raise ValueError(
            f"Checkpoint step(s) unavailable: {missing}. Available steps: {choices}"
        )
    return [available[step] for step in requested]


def _checkpoint_config(manifest: Mapping, asset: Mapping) -> dict:
    adapter = manifest.get("training", {}).get("adapter", {})
    return {
        "source_repository": manifest.get("repository", DEFAULT_REPOSITORY),
        "source_release": manifest["release"],
        "source_asset": asset["name"],
        "step": int(asset["step"]),
        "role": asset.get("role"),
        "sha256": asset["sha256"],
        "lora_rank": int(adapter["rank"]),
        "lora_alpha": int(adapter["alpha"]),
        "lora_dropout": float(adapter.get("dropout", 0.05)),
        "lora_targets": list(adapter.get("targets", ["Wqkv", "Wo"])),
    }


def _download_checkpoint(
    repository: str,
    release: str,
    asset: Mapping,
    destination: Path,
    manifest: Mapping,
) -> None:
    """Download and atomically install one verified checkpoint."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-",
        dir=destination.parent,
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        _run(
            [
                "gh",
                "release",
                "download",
                release,
                "--repo",
                repository,
                "--pattern",
                str(asset["name"]),
                "--dir",
                str(temporary),
            ]
        )
        downloaded = temporary / str(asset["name"])
        if not downloaded.is_file():
            raise RuntimeError(f"GitHub did not download {asset['name']}")
        if downloaded.stat().st_size != int(asset["bytes"]):
            raise RuntimeError(f"Size verification failed for {asset['name']}")
        if _sha256(downloaded) != asset["sha256"]:
            raise RuntimeError(f"SHA-256 verification failed for {asset['name']}")

        staging = destination.parent / f".{destination.name}.installing"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir()
        os.replace(downloaded, staging / "model.safetensors")
        (staging / "config.json").write_text(
            json.dumps(_checkpoint_config(manifest, asset), indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "COMPLETE").write_text("verified\n", encoding="utf-8")
        shutil.rmtree(destination, ignore_errors=True)
        os.replace(staging, destination)


def _checkpoint_is_valid(path: Path, asset: Mapping) -> bool:
    weights = path / "model.safetensors"
    return (
        (path / "COMPLETE").is_file()
        and weights.is_file()
        and weights.stat().st_size == int(asset["bytes"])
        and _sha256(weights) == asset["sha256"]
    )


def download_release_checkpoints(
    *,
    repository: str = DEFAULT_REPOSITORY,
    release: str = DEFAULT_RELEASE,
    steps: Iterable[int] = DEFAULT_STEPS,
    cache_dir: str | Path = DEFAULT_CACHE,
    force: bool = False,
) -> tuple[dict, tuple[dict, ...]]:
    """Download selected checkpoints and return router-ready metadata.

    Existing cache entries are reused only after their size and SHA-256 digest
    have been verified.
    """

    manifest = fetch_registry_manifest(repository)
    if manifest.get("release") != release:
        raise RuntimeError(
            f"Manifest release {manifest.get('release')!r} does not match {release!r}"
        )

    release_root = Path(cache_dir).expanduser().resolve() / release
    checkpoints = []
    for asset in _selected_assets(manifest, steps):
        step = int(asset["step"])
        path = release_root / "checkpoints" / f"step-{step:07d}"
        if force or not _checkpoint_is_valid(path, asset):
            _download_checkpoint(repository, release, asset, path, manifest)
        checkpoints.append(
            {
                "name": f"scientific-laya-step-{step:04d}",
                "path": path,
                "role": asset.get("role"),
                "weight": 1.0,
            }
        )

    registry_path = release_root / "model-manifest.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest, tuple(checkpoints)


def download_base_model(
    model_id: str,
    *,
    cache_dir: str | Path = DEFAULT_CACHE,
) -> Path:
    """Download the pinned public Laya base model from Hugging Face."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Install `huggingface-hub` to download the Laya base model."
        ) from exc

    target = Path(cache_dir).expanduser().resolve() / "base-models"
    target.mkdir(parents=True, exist_ok=True)
    return Path(snapshot_download(model_id, cache_dir=target, max_workers=4))


def prepare_github_laya_weights(
    *,
    repository: str = DEFAULT_REPOSITORY,
    release: str = DEFAULT_RELEASE,
    steps: Iterable[int] = DEFAULT_STEPS,
    cache_dir: str | Path = DEFAULT_CACHE,
    base_model: str | Path | None = None,
    download_base: bool = True,
    force: bool = False,
) -> PreparedLayaWeights:
    """Prepare a verified local release for Jnana's checkpoint routers."""

    manifest, checkpoints = download_release_checkpoints(
        repository=repository,
        release=release,
        steps=steps,
        cache_dir=cache_dir,
        force=force,
    )

    if base_model is not None:
        base_path = Path(base_model).expanduser().resolve()
    elif download_base:
        base_path = download_base_model(
            str(manifest["base_model"]),
            cache_dir=cache_dir,
        )
    else:
        raise ValueError("Pass base_model or enable download_base")

    return PreparedLayaWeights(
        base_model=base_path,
        checkpoints=checkpoints,
        manifest=manifest,
    )


def github_checkpoint_router_specs(
    *,
    repository: str = DEFAULT_REPOSITORY,
    release: str = DEFAULT_RELEASE,
    steps: Iterable[int] = DEFAULT_STEPS,
    cache_dir: str | Path = DEFAULT_CACHE,
    base_model: str | Path | None = None,
    download_base: bool = True,
    force: bool = False,
    device: str = "cuda",
    batch_size: int = 32,
):
    """Download verified weights and return native Jnana router specs."""

    from .laya_checkpoint_router import checkpoint_router_specs

    prepared = prepare_github_laya_weights(
        repository=repository,
        release=release,
        steps=steps,
        cache_dir=cache_dir,
        base_model=base_model,
        download_base=download_base,
        force=force,
    )
    return checkpoint_router_specs(
        prepared.base_model,
        prepared.checkpoints,
        device=device,
        batch_size=batch_size,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download verified scientific Laya weights from GitHub.",
    )
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--steps", nargs="+", type=int, default=list(DEFAULT_STEPS))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument(
        "--base-model",
        help="Existing local Laya base-model directory; skips its HF download.",
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Download checkpoint assets without downloading the base model.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.weights_only:
        manifest, checkpoints = download_release_checkpoints(
            repository=args.repository,
            release=args.release,
            steps=args.steps,
            cache_dir=args.cache_dir,
            force=args.force,
        )
        prepared = {
            "base_model": args.base_model,
            "checkpoints": list(checkpoints),
            "manifest_release": manifest["release"],
        }
    else:
        result = prepare_github_laya_weights(
            repository=args.repository,
            release=args.release,
            steps=args.steps,
            cache_dir=args.cache_dir,
            base_model=args.base_model,
            force=args.force,
        )
        prepared = {
            "base_model": str(result.base_model),
            "checkpoints": list(result.checkpoints),
            "manifest_release": result.manifest["release"],
        }

    for checkpoint in prepared["checkpoints"]:
        checkpoint["path"] = str(checkpoint["path"])
    print(json.dumps(prepared, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
